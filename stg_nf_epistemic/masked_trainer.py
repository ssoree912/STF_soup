"""
Baseline masked-ensemble trainer reused by the epistemic variant.
"""

import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.scoring_utils import score_dataset
from .masked_model import MaskedEnsembleSTGNF, create_masked_ensemble_stg_nf


class MaskedEnsembleTrainer:
    """Training loop for masked STG-NF ensembles with flexible member scheduling."""

    def __init__(
        self,
        model: MaskedEnsembleSTGNF,
        args,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        test_metadata=None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_metadata = test_metadata
        self.logger = logger or logging.getLogger(__name__)

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.ensemble_strategy = getattr(args, "ensemble_strategy", "random")
        self.ensemble_switch_freq = getattr(args, "ensemble_switch_freq", 100)
        self.best_auc = None

    def _get_optimizer(self):
        if self.args.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=getattr(self.args, "weight_decay", 0),
            )
        if self.args.optimizer == "adamax":
            return optim.Adamax(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=getattr(self.args, "weight_decay", 0),
            )
        return optim.SGD(self.model.parameters(), lr=self.args.lr)

    def _get_scheduler(self):
        if getattr(self.args, "lr_decay", None):
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1,
                gamma=self.args.lr_decay,
            )
        return None

    def _select_ensemble_member(self, iteration: int) -> Optional[int]:
        if self.ensemble_strategy == "random":
            return np.random.randint(0, self.model.ensemble_size)
        if self.ensemble_strategy == "sequential":
            return (iteration // self.ensemble_switch_freq) % self.model.ensemble_size
        if self.ensemble_strategy == "round_robin":
            return iteration % self.model.ensemble_size
        if self.ensemble_strategy == "all":
            return None
        return 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        usage = {i: 0 for i in range(self.model.ensemble_size)}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        for iteration, data_arr in enumerate(pbar):
            try:
                data = [d.to(self.model.device, non_blocking=True) for d in data_arr]
                score = data[-2].amin(dim=-1)
                label = data[-1]
                samp = data[0] if getattr(self.args, "model_confidence", False) else data[0][:, :2]

                ensemble_idx = self._select_ensemble_member(iteration)
                if ensemble_idx is not None:
                    usage[ensemble_idx] += 1

                self.optimizer.zero_grad()
                if self.ensemble_strategy == "ensemble_loss":
                    member_nlls = []
                    for idx in range(self.model.ensemble_size):
                        _, nll = self.model(samp.float(), label=label, score=score, mask_index=idx)
                        if getattr(self.args, "model_confidence", False):
                            nll = nll * score
                        member_nlls.append(nll)
                    loss = torch.stack(member_nlls).mean(dim=0).mean()
                else:
                    _, nll = self.model(samp.float(), label=label, score=score, mask_index=ensemble_idx)
                    if getattr(self.args, "model_confidence", False):
                        nll = nll * score
                    loss = nll.mean()

                if not torch.isnan(loss):
                    loss.backward()
                    if getattr(self.args, "clip_grad", 0) > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.optimizer.step()
                    batch_size = samp.shape[0]
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                else:
                    self.logger.warning("Encountered NaN loss, skipping update.")

                current_loss = total_loss / max(total_samples, 1)
                display_idx = "All" if ensemble_idx is None else ensemble_idx
                pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Ensemble": display_idx})
            except KeyboardInterrupt:
                self.logger.info("Training interrupted by user.")
                break
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning(f"Error in iteration {iteration}: {exc}")
                continue

        avg_loss = total_loss / max(total_samples, 1)
        if self.scheduler:
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
        else:
            lr = self.args.lr

        epoch_stats = {
            "avg_loss": avg_loss,
            "learning_rate": lr,
            "total_samples": total_samples,
            **{f"ensemble_{idx}_usage": count for idx, count in usage.items()},
        }
        self.logger.info(f"Epoch {epoch + 1} - Loss: {avg_loss:.6f}, LR: {lr:.2e}")
        return epoch_stats

    def evaluate(self) -> Dict[str, float]:
        if self.test_loader is None or self.test_metadata is None:
            return {}

        self.model.eval()
        probs = torch.empty(0, device=self.model.device)
        with torch.no_grad():
            for data_arr in tqdm(self.test_loader, desc="Evaluating", leave=False):
                data = [d.to(self.model.device, non_blocking=True) for d in data_arr]
                score = data[-2].amin(dim=-1)
                samp = data[0] if getattr(self.args, "model_confidence", False) else data[0][:, :2]
                _, nll = self.model(
                    samp.float(),
                    label=torch.ones(data[0].shape[0], device=self.model.device),
                    score=score,
                    mask_index=None,
                )
                if getattr(self.args, "model_confidence", False):
                    nll = nll * score
                probs = torch.cat((probs, -nll), dim=0)

        scores = probs.cpu().detach().numpy().squeeze().copy(order="C")
        auc, scores_np, labels_np, roc_parts = score_dataset(scores, self.test_metadata, args=self.args)
        return {
            "auc": auc,
            "roc_auc": auc,
            "scores": scores_np,
            "labels": labels_np,
            "roc_parts": roc_parts,
        }

    def evaluate_ensemble_members(self) -> Dict[str, float]:
        if self.test_loader is None or self.test_metadata is None:
            return {}

        self.model.eval()
        results = {}
        with torch.no_grad():
            for idx in range(self.model.ensemble_size):
                probs = torch.empty(0, device=self.model.device)
                for data_arr in tqdm(self.test_loader, desc=f"Eval Member {idx}", leave=False):
                    data = [d.to(self.model.device, non_blocking=True) for d in data_arr]
                    score = data[-2].amin(dim=-1)
                    samp = data[0] if getattr(self.args, "model_confidence", False) else data[0][:, :2]
                    _, nll = self.model(
                        samp.float(),
                        label=torch.ones(data[0].shape[0], device=self.model.device),
                        score=score,
                        mask_index=idx,
                    )
                    if getattr(self.args, "model_confidence", False):
                        nll = nll * score
                    probs = torch.cat((probs, -nll), dim=0)
                scores = probs.cpu().detach().numpy().squeeze().copy(order="C")
                auc, _, _, _ = score_dataset(scores, self.test_metadata, args=self.args)
                results[f"member_{idx}_auc"] = auc

        member_aucs = [results[f"member_{idx}_auc"] for idx in range(self.model.ensemble_size)]
        results["members_avg_auc"] = float(np.mean(member_aucs))
        results["members_std_auc"] = float(np.std(member_aucs))
        return results

    def train(self, epochs: int, save_dir: Optional[str] = None):
        history = []
        self.logger.info("Starting masked-ensemble training.")
        for epoch in range(epochs):
            start_time = time.time()
            train_stats = self.train_epoch(epoch)
            eval_stats = self.evaluate()
            member_stats = self.evaluate_ensemble_members()
            epoch_stats = {
                "epoch": epoch + 1,
                "time": time.time() - start_time,
                **train_stats,
                **eval_stats,
                **member_stats,
            }
            history.append(epoch_stats)

            current_auc = eval_stats.get("auc")
            if current_auc is not None and (self.best_auc is None or current_auc > self.best_auc):
                self.best_auc = current_auc
                if save_dir:
                    self.save_checkpoint(save_dir, epoch, is_best=True)
                    self.logger.info(f"New best model saved (AUC {current_auc:.4f}).")

            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir, epoch, is_best=False)

        self.logger.info(f"Training complete. Best AUC: {self.best_auc}.")
        return history

    def save_checkpoint(self, save_dir: str, epoch: int, is_best: bool = False):
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_auc": self.best_auc,
            "ensemble_size": self.model.ensemble_size,
            "mask_type": self.model.mask_type,
            "args": self.args,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint_path = os.path.join(save_dir, f"ensemble_checkpoint_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = os.path.join(save_dir, "ensemble_checkpoint_best.pth")
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_auc = checkpoint.get("best_auc")
        start_epoch = checkpoint.get("epoch", 0)
        self.logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {start_epoch}).")
        return start_epoch


def create_ensemble_trainer(
    model_args: dict,
    ensemble_size: int = 3,
    mask_type: str = "random",
    training_args=None,
    train_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    test_metadata=None,
    device: str = "cuda",
) -> MaskedEnsembleTrainer:
    model = create_masked_ensemble_stg_nf(
        model_args,
        ensemble_size=ensemble_size,
        mask_type=mask_type,
        device=device,
    )
    return MaskedEnsembleTrainer(
        model=model,
        args=training_args,
        train_loader=train_loader,
        test_loader=test_loader,
        test_metadata=test_metadata,
    )

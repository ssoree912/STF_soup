import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.scoring_utils import score_dataset

from .masked_model import MaskedEnsembleSTGNF, create_masked_ensemble_stg_nf
from .masked_trainer import MaskedEnsembleTrainer
from .mixture_utils import compute_mixture_nll


class EpistemicMaskedEnsembleTrainer(MaskedEnsembleTrainer):
    """
    Trainer that mirrors the mixture-based likelihood aggregation used in
    ``nflows_epistemic``. Instead of sampling individual ensemble members per
    iteration, every forward pass evaluates all masks and aggregates their log
    probabilities with a log-mean-exp reduction.
    """

    def __init__(
        self,
        model: MaskedEnsembleSTGNF,
        args,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        test_metadata=None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            train_loader=train_loader,
            test_loader=test_loader,
            test_metadata=test_metadata,
            logger=logger,
        )
        self.ensemble_strategy = "mixture"

    def _forward_all_members(
        self,
        samp: torch.Tensor,
        label: torch.Tensor,
        score: torch.Tensor,
    ) -> List[torch.Tensor]:
        member_nlls: List[torch.Tensor] = []
        for member_idx in range(self.model.ensemble_size):
            _, nll = self.model(
                samp.float(),
                label=label,
                score=score,
                mask_index=member_idx,
            )
            if getattr(self.args, "model_confidence", False):
                nll = nll * score
            member_nlls.append(nll)
        return member_nlls

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        ensemble_usage = {i: 0 for i in range(self.model.ensemble_size)}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        for iteration, data_arr in enumerate(pbar):
            try:
                data = [d.to(self.model.device, non_blocking=True) for d in data_arr]
                score = data[-2].amin(dim=-1)
                label = data[-1]

                if getattr(self.args, "model_confidence", False):
                    samp = data[0]
                else:
                    samp = data[0][:, :2]

                batch_size = samp.shape[0]

                self.optimizer.zero_grad()

                member_nlls = self._forward_all_members(samp, label, score)
                mixture_nll = compute_mixture_nll(member_nlls, samp.shape)
                loss = mixture_nll.mean()

                if not torch.isnan(loss):
                    loss.backward()
                    if hasattr(self.args, "clip_grad") and self.args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.optimizer.step()

                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    for idx in range(self.model.ensemble_size):
                        ensemble_usage[idx] += batch_size
                else:
                    self.logger.warning(f"NaN loss at iteration {iteration}")

                current_loss = total_loss / max(total_samples, 1)
                pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Ensemble": "mixture"})

            except KeyboardInterrupt:
                self.logger.info("Training interrupted by user")
                break
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning(f"Error in iteration {iteration}: {exc}")
                continue

        avg_loss = total_loss / max(total_samples, 1)

        if self.scheduler:
            self.scheduler.step()
            new_lr = self.scheduler.get_lr()[0]
        else:
            new_lr = self.args.lr

        usage_stats = {f"ensemble_{idx}_usage": count for idx, count in ensemble_usage.items()}

        epoch_stats = {
            "avg_loss": avg_loss,
            "learning_rate": new_lr,
            "total_samples": total_samples,
            **usage_stats,
        }

        self.logger.info(f"Epoch {epoch + 1} - Mixture loss: {avg_loss:.6f}, LR: {new_lr:.2e}")

        return epoch_stats

    def evaluate(self) -> Dict[str, float]:
        if self.test_loader is None or self.test_metadata is None:
            return {}

        self.model.eval()
        probs = torch.empty(0, device=self.model.device)

        with torch.no_grad():
            for data_arr in tqdm(self.test_loader, desc="Evaluating (mixture)", leave=False):
                data = [d.to(self.model.device, non_blocking=True) for d in data_arr]
                score = data[-2].amin(dim=-1)

                if getattr(self.args, "model_confidence", False):
                    samp = data[0]
                else:
                    samp = data[0][:, :2]

                labels = torch.ones(samp.shape[0], device=self.model.device)
                member_nlls = self._forward_all_members(samp, labels, score)
                mixture_nll = compute_mixture_nll(member_nlls, samp.shape)
                probs = torch.cat((probs, -mixture_nll), dim=0)

        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order="C")
        auc, scores_np, labels_np, roc_parts = score_dataset(prob_mat_np, self.test_metadata, args=self.args)

        return {
            "auc": auc,
            "roc_auc": auc,
            "scores": scores_np,
            "labels": labels_np,
            "roc_parts": roc_parts,
        }


def create_epistemic_trainer(
    model_args: dict,
    ensemble_size: int = 3,
    mask_type: str = "random",
    training_args=None,
    train_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    test_metadata=None,
    device: str = "cuda",
) -> EpistemicMaskedEnsembleTrainer:
    """
    Convenience factory mirroring ``create_ensemble_trainer`` but wired to the
    mixture-based trainer defined here.
    """
    model = create_masked_ensemble_stg_nf(
        model_args,
        ensemble_size=ensemble_size,
        mask_type=mask_type,
        device=device,
    )

    return EpistemicMaskedEnsembleTrainer(
        model=model,
        args=training_args,
        train_loader=train_loader,
        test_loader=test_loader,
        test_metadata=test_metadata,
    )

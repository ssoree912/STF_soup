"""
Train\Test helper, based on awesome previous work by https://github.com/amirmk89/gepc
"""

import os
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils import prune
from utils.scoring_utils import score_dataset


def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def compute_loss(nll, reduction="mean", mean=0):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "logsumexp":
        losses = {"nll": torch.logsumexp(nll, dim=0)}
    elif reduction == "exp":
        losses = {"nll": torch.exp(torch.mean(nll) - mean)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


class Trainer:
    def __init__(self, args, model, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None, test_metadata=None):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_metadata = test_metadata
        # Loss, Optimizer and Scheduler
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)
        self.prune_epoch = getattr(self.args, 'prune_epoch', None)
        self.unprune_epoch = getattr(self.args, 'unprune_epoch', None)
        self.prune_method = getattr(self.args, 'prune_method', 'magnitude')
        self.magnitude_ratio = float(getattr(self.args, 'prune_ratio_magnitude', 0.0))
        self.random_ratio = float(getattr(self.args, 'prune_ratio_random', 0.0))
        self.prune_ratio = getattr(self.args, 'prune_ratio', 0.0)
        self.random_seed = getattr(self.args, 'prune_random_seed', None)
        if self.prune_ratio and not (self.magnitude_ratio or self.random_ratio):
            if self.prune_method == 'random':
                self.random_ratio = float(self.prune_ratio)
            else:
                self.magnitude_ratio = float(self.prune_ratio)
        total_ratio = self.magnitude_ratio + self.random_ratio
        if total_ratio <= 0:
            self.prune_epoch = None
            self.unprune_epoch = None
        self._pruned_params = []
        self.pruning_applied = False
        self.best_auc = None

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        elif self.args.optimizer == 'adamx':
            if self.args.lr:
                return optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adamax(self.model.parameters())
        return optim.SGD(self.model.parameters(), lr=self.args.lr)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.model_lr, self.args.model_lr_decay, self.scheduler)

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = self.args

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = filename
        try:
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.set_actnorm_init()
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(filename, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))

    def train(self, log_writer=None, clip=100):
        time_str = time.strftime("%b%d_%H%M_")
        checkpoint_filename = time_str + '_checkpoint.pth.tar'
        start_epoch = 0
        num_epochs = self.args.epochs
        self.model.train()
        self.model = self.model.to(self.args.device)
        key_break = False
        for epoch in range(start_epoch, num_epochs):
            self._maybe_apply_pruning(epoch)
            self._maybe_remove_pruning(epoch)
            if key_break:
                break
            print("Starting Epoch {} / {}".format(epoch + 1, num_epochs))
            pbar = tqdm(self.train_loader)
            for itern, data_arr in enumerate(pbar):
                try:
                    data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
                    score = data[-2].amin(dim=-1)
                    label = data[-1]
                    if self.args.model_confidence:
                        samp = data[0]
                    else:
                        samp = data[0][:, :2]
                    z, nll = self.model(samp.float(), label=label, score=score)
                    if nll is None:
                        continue
                    if self.args.model_confidence:
                        nll = nll * score
                    losses = compute_loss(nll, reduction="mean")["total_loss"]
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pbar.set_description("Loss: {}".format(losses.item()))
                    if log_writer is not None:
                        log_writer.add_scalar('NLL Loss', losses.item(), epoch * len(self.train_loader) + itern)

                except KeyboardInterrupt:
                    print('Keyboard Interrupted. Save results? [yes/no]')
                    choice = input().lower()
                    if choice == "yes":
                        key_break = True
                        break
                    else:
                        exit(1)

            self.save_checkpoint(epoch, filename=checkpoint_filename)
            new_lr = self.adjust_lr(epoch)
            print('Checkpoint Saved. New LR: {0:.3e}'.format(new_lr))
            if self.test_metadata is not None:
                auc = self._evaluate_auc()
                if (self.best_auc is None) or (auc > self.best_auc):
                    self.best_auc = auc
                    best_path = os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar')
                    shutil.copy(os.path.join(self.args.ckpt_dir, checkpoint_filename), best_path)
                    print(f'New best checkpoint saved with AUC {auc * 100:.2f}%')

    def test(self):
        self.model.eval()
        self.model.to(self.args.device)
        pbar = tqdm(self.test_loader)
        probs = torch.empty(0).to(self.args.device)
        print("Starting Test Eval")
        for itern, data_arr in enumerate(pbar):
            data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
            score = data[-2].amin(dim=-1)
            if self.args.model_confidence:
                samp = data[0]
            else:
                samp = data[0][:, :2]
            with torch.no_grad():
                z, nll = self.model(samp.float(), label=torch.ones(data[0].shape[0]), score=score)
            if self.args.model_confidence:
                nll = nll * score
            probs = torch.cat((probs, -1 * nll), dim=0)
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        return prob_mat_np

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state

    def _collect_prunable_parameters(self):
        params = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                params.append((module, 'weight'))
        return params

    def _maybe_apply_pruning(self, epoch):
        if self.prune_epoch is None:
            return
        if self.pruning_applied:
            return
        if (epoch + 1) < self.prune_epoch:
            return
        params = self._collect_prunable_parameters()
        if not params:
            print("Warning: no prunable parameters found; skipping pruning")
            self.prune_epoch = None
            return
        if self.magnitude_ratio > 0:
            prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=self.magnitude_ratio)
            print(f"Applied global magnitude pruning at epoch {epoch + 1} with ratio {self.magnitude_ratio:.2f}")
        if self.random_ratio > 0:
            cpu_state = None
            cuda_state = None
            if self.random_seed is not None:
                cpu_state = torch.random.get_rng_state()
                if torch.cuda.is_available():
                    cuda_state = torch.cuda.get_rng_state_all()
                    torch.cuda.manual_seed_all(self.random_seed)
                torch.manual_seed(self.random_seed)
            prune.global_unstructured(params, pruning_method=prune.RandomUnstructured, amount=self.random_ratio)
            if self.random_seed is not None:
                torch.random.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)
            print(f"Applied global random pruning at epoch {epoch + 1} with ratio {self.random_ratio:.2f} (seed {self.random_seed})")
        self._pruned_params = params
        self.pruning_applied = True

    def _maybe_remove_pruning(self, epoch):
        if self.unprune_epoch is None:
            return
        if not self.pruning_applied:
            return
        if (epoch + 1) < self.unprune_epoch:
            return
        for module, _ in self._pruned_params:
            try:
                prune.remove(module, 'weight')
            except ValueError:
                continue
        self.pruning_applied = False
        self._pruned_params = []
        print(f"Pruning mask removed at epoch {epoch + 1}")

    def _evaluate_auc(self):
        normality_scores = self.test()
        auc, _, _, _ = score_dataset(normality_scores, self.test_metadata, args=self.args)
        self.model.train()
        return auc

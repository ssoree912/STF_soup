import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

from models.STG_NF.model_pose import STG_NF


class FisherSTGNF:
    """Fisher Information computation for STG-NF models with uncertainty weighting support."""
    
    def __init__(self, model: STG_NF, device: torch.device, logger: Optional[logging.Logger] = None, args=None):
        self.model = model
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.args = args
        
    def _mergeable_name_param(self) -> List[Tuple[str, torch.nn.Parameter]]:
        pairs: List[Tuple[str, torch.nn.Parameter]] = []
        for name, param in self.model.named_parameters():
            if (not param.requires_grad) or self._should_skip_parameter(name, param):
                continue
            pairs.append((name, param))
        return pairs
    
    def _should_skip_parameter(self, name: str, param: torch.Tensor) -> bool:
        """Check if parameter should be skipped for Fisher computation."""
        if name.endswith("actnorm.inited"):
            return True
        banned_tokens = (".actnorm.", "running_mean", "running_var", "num_batches_tracked")
        if any(token in name for token in banned_tokens):
            return True
        return False
    
    def _compute_fisher_for_batch(self, batch_data: Tuple,
                                  name_param: List[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information for a single batch."""
        data, labels, scores = batch_data[:3]

        # Move to device
        data = data.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        scores = scores.to(self.device, non_blocking=True)
        batch_size = data.shape[0]

        fisher_dict = {name: torch.zeros_like(param, device=self.device)
                       for name, param in name_param}

        # Optional filtering: keep only "clean" (low-score) samples
        clean_quantile = getattr(self.args, 'fisher_clean_quantile', None) if self.args else None
        keep_indices = torch.arange(batch_size, device=self.device)
        if clean_quantile is not None and 0.0 < clean_quantile < 1.0:
            flat_scores = scores.view(batch_size, -1).mean(dim=1)
            thresh = torch.quantile(flat_scores.detach(), float(clean_quantile))
            keep_mask = flat_scores <= thresh
            if keep_mask.any():
                keep_indices = keep_indices[keep_mask]
            else:
                keep_indices = keep_indices[:0]

        if keep_indices.numel() == 0:
            return fisher_dict

        for b in keep_indices.tolist():
            sample_data = data[b:b+1]
            sample_label = labels[b:b+1]
            sample_score = scores[b:b+1]

            sample_fishers = self._compute_fisher_single_sample(
                sample_data, sample_label, sample_score, name_param, self.args
            )

            for name in fisher_dict:
                fisher_dict[name] += sample_fishers[name]

        for name in fisher_dict:
            fisher_dict[name] /= max(keep_indices.numel(), 1)

        return fisher_dict
    
    def _compute_fisher_single_sample(self, data: torch.Tensor,
                                      labels: torch.Tensor,
                                      scores: torch.Tensor,
                                      name_param: List[Tuple[str, torch.nn.Parameter]],
                                      args) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information for a single sample."""
        sample_fishers = {name: torch.zeros_like(param, device=self.device)
                          for name, param in name_param}

        variables = [param for _, param in name_param]
        for param in variables:
            param.requires_grad_(True)

        try:
            # Process data same as evaluation - use only first 2 channels unless model_confidence is True
            if getattr(args, 'model_confidence', False):
                samp = data
            else:
                samp = data[:, :2]  # Only use first 2 channels
            
            # Forward pass through STG-NF model
            _, nll = self.model(samp.float(), label=labels, score=scores)
            
            # Apply model_confidence weighting if enabled
            if getattr(args, 'model_confidence', False):
                nll = nll * scores
            
            # Use negative log-likelihood as the loss
            loss = nll.mean()
            
            # Compute gradients
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=variables,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )

            # Accumulate squared gradients (Fisher Information)
            for (name, _), grad in zip(name_param, grads):
                if grad is not None:
                    sample_fishers[name] = grad ** 2
                else:
                    sample_fishers[name] = torch.zeros_like(sample_fishers[name], device=self.device)

        except RuntimeError as e:
            self.logger.warning(f"Error computing Fisher for sample: {e}")
            # Return zero Fisher if computation fails
            for name, param in name_param:
                sample_fishers[name] = torch.zeros_like(param, device=self.device)

        return sample_fishers
    
    def _compute_uncertainty_weighted_fisher_for_batch(self, batch_data: Tuple,
                                                      name_param: List[Tuple[str, torch.nn.Parameter]],
                                                      uncertainty_weights: torch.Tensor,
                                                      use_batch_approx: bool = True) -> Dict[str, torch.Tensor]:
        """Compute uncertainty-weighted Fisher Information for a single batch."""
        data, labels, scores = batch_data[:3]

        # Move to device
        data = data.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        scores = scores.to(self.device, non_blocking=True)
        uncertainty_weights = uncertainty_weights.to(self.device, non_blocking=True)
        batch_size = data.shape[0]

        fisher_dict = {name: torch.zeros_like(param, device=self.device)
                       for name, param in name_param}

        if use_batch_approx:
            # FAST APPROXIMATION: compute batch-level gradients once, weighted by uncertainty
            variables = [param for _, param in name_param]

            # Clear existing gradients
            for param in variables:
                if param.grad is not None:
                    param.grad = None

            if getattr(self.args, "model_confidence", False):
                samp = data
            else:
                samp = data[:, :2]

            _, nll = self.model(samp.float(), label=labels, score=scores)
            if getattr(self.args, "model_confidence", False):
                nll = nll * scores

            weight_view = uncertainty_weights.view([-1] + [1] * (nll.ndim - 1))
            weighted_loss = (nll * weight_view).mean()

            grads = torch.autograd.grad(
                outputs=weighted_loss,
                inputs=variables,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            for (name, _), grad in zip(name_param, grads):
                if grad is not None:
                    fisher_dict[name] = grad.pow(2)
        else:
            # SLOW EXACT: Process each sample individually (original method)
            for b in range(batch_size):
                sample_data = data[b:b+1]
                sample_label = labels[b:b+1]
                sample_score = scores[b:b+1]
                sample_weight = uncertainty_weights[b]

                sample_fishers = self._compute_fisher_single_sample(
                    sample_data, sample_label, sample_score, name_param, self.args
                )

                # Weight Fisher information by uncertainty weight
                for name in fisher_dict:
                    fisher_dict[name] += sample_fishers[name] * sample_weight

            # Average over batch size
            for name in fisher_dict:
                fisher_dict[name] /= batch_size

        return fisher_dict

    def compute_fisher_for_model(self, dataloader, max_batches: int = 100) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix for the entire model."""
        self.logger.info("Computing Fisher Information Matrix for STG-NF model...")

        name_param = self._mergeable_name_param()
        self.logger.info(f"Found {len(name_param)} mergeable parameters")

        fishers = {name: torch.zeros_like(param, device=self.device)
                   for name, param in name_param}

        self.model.eval()
        n_batches = 0

        # Process data in batches
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Computing Fisher", leave=False)):
            if batch_idx >= max_batches:
                self.logger.info(f"Reached maximum batches limit ({max_batches})")
                break
                
            try:
                batch_fishers = self._compute_fisher_for_batch(batch_data, name_param)

                for name in fishers:
                    fishers[name] += batch_fishers[name].detach()

                n_batches += 1

                # Clear cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                self.logger.warning(f"CUDA OOM at batch {batch_idx}, clearing cache and continuing...")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                self.logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Average over all batches
        if n_batches > 0:
            for name in fishers:
                fishers[name] /= n_batches

        self.logger.info(f"Fisher computation completed. Processed {n_batches} batches.")
        return fishers

    def compute_uncertainty_weighted_fisher(self, dataloader, uncertainty_weights: torch.Tensor, 
                                          max_batches: int = 100, use_batch_approx: bool = True, 
                                          subsample_ratio: float = 1.0) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix weighted by epistemic uncertainty."""
        mode_str = "FAST (batch approx)" if use_batch_approx else "EXACT (sample-wise)"
        self.logger.info(f"Computing Uncertainty-Weighted Fisher [{mode_str}] for STG-NF model...")

        name_param = self._mergeable_name_param()
        self.logger.info(f"Found {len(name_param)} mergeable parameters")

        fishers = {name: torch.zeros_like(param, device=self.device)
                   for name, param in name_param}

        self.model.eval()
        n_batches = 0
        sample_idx = 0

        # Apply subsampling
        effective_max_batches = max(1, int(max_batches * subsample_ratio))
        if subsample_ratio < 1.0:
            self.logger.info(f"Using subsampling: {effective_max_batches}/{max_batches} batches")

        # Process data in batches
        desc = f"UW-Fisher[{'FAST' if use_batch_approx else 'EXACT'}]"
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            if batch_idx >= effective_max_batches:
                self.logger.info(f"Reached maximum batches limit ({effective_max_batches})")
                break
                
            try:
                # Get batch size
                batch_size = batch_data[0].shape[0]
                
                # Get uncertainty weights for this batch
                batch_weights = uncertainty_weights[sample_idx:sample_idx + batch_size]
                if batch_weights.shape[0] < batch_size:
                    self.logger.warning(f"Not enough uncertainty weights for batch {batch_idx}")
                    break
                
                batch_fishers = self._compute_uncertainty_weighted_fisher_for_batch(
                    batch_data, name_param, batch_weights, use_batch_approx)

                for name in fishers:
                    fishers[name] += batch_fishers[name].detach()

                n_batches += 1
                sample_idx += batch_size

                # Clear cache periodically
                if batch_idx % 5 == 0:  # More frequent cache clearing
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                self.logger.warning(f"CUDA OOM at batch {batch_idx}, clearing cache and continuing...")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                self.logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Average over all batches
        if n_batches > 0:
            for name in fishers:
                fishers[name] /= n_batches

        self.logger.info(f"Uncertainty-weighted Fisher computation completed. Processed {n_batches} batches.")
        return fishers


def save_fisher_info(fishers: Dict[str, torch.Tensor], save_path: str,
                     logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = logging.getLogger(__name__)
    payload = {name: tensor.cpu() for name, tensor in fishers.items()}
    torch.save(payload, save_path)
    logger.info(f"Saved Fisher Information to {save_path}")


def load_fisher_info(load_path: str, device: torch.device,
                     logger: Optional[logging.Logger] = None) -> Dict[str, torch.Tensor]:
    if logger is None:
        logger = logging.getLogger(__name__)
    payload = torch.load(load_path, map_location=device)
    fisher_dict = {name: tensor.to(device=device) for name, tensor in payload.items()}
    logger.info(f"Loaded Fisher Information from {load_path}")
    return fisher_dict

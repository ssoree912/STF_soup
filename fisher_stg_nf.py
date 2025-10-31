import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

from models.STG_NF.model_pose import STG_NF


class FisherSTGNF:
    """Fisher Information computation for STG-NF models."""
    
    def __init__(self, model: STG_NF, device: torch.device, logger: Optional[logging.Logger] = None):
        self.model = model
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
    def get_mergeable_parameters(self) -> List[torch.nn.Parameter]:
        """Get model parameters that can be merged (excluding bias and 1D parameters)."""
        mergeable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() > 1:
                mergeable_params.append(param)
        return mergeable_params
    
    def _should_skip_parameter(self, name: str, param: torch.Tensor) -> bool:
        """Check if parameter should be skipped for Fisher computation."""
        # Skip actnorm related parameters
        if ".actnorm." in name:
            return True
        if name.endswith("actnorm.inited"):
            return True
        # Skip running statistics
        if any(skip in name for skip in ["running_mean", "running_var", "num_batches_tracked"]):
            return True
        # Skip 1D parameters (bias, etc.)
        if param.dim() <= 1:
            return True
        # Skip prior parameters
        if "prior_h" in name:
            return True
        return False
    
    def _compute_fisher_for_batch(self, batch_data: Tuple, variables: List[torch.nn.Parameter]) -> List[torch.Tensor]:
        """Compute Fisher Information for a single batch."""
        data, labels, scores = batch_data[:3]
        
        # Move to device
        data = data.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)  
        scores = scores.to(self.device, non_blocking=True)
        
        batch_size = data.shape[0]
        batch_fishers = []
        for param in variables:
            batch_fishers.append(torch.zeros_like(param, device=self.device))
        
        # Process each sample in the batch
        for b in range(batch_size):
            sample_data = data[b:b+1]  # Keep batch dimension
            sample_label = labels[b:b+1]
            sample_score = scores[b:b+1]
            
            # Compute Fisher for this sample
            sample_fishers = self._compute_fisher_single_sample(
                sample_data, sample_label, sample_score, variables
            )
            
            # Accumulate Fisher information
            for i, fisher in enumerate(sample_fishers):
                if fisher is not None:
                    batch_fishers[i] += fisher
        
        # Average over batch size
        for fisher in batch_fishers:
            fisher /= batch_size
            
        return batch_fishers
    
    def _compute_fisher_single_sample(self, data: torch.Tensor, 
                                    labels: torch.Tensor,
                                    scores: torch.Tensor,
                                    variables: List[torch.nn.Parameter]) -> List[torch.Tensor]:
        """Compute Fisher Information for a single sample."""
        sample_fishers = []
        for param in variables:
            sample_fishers.append(torch.zeros_like(param, device=self.device))
        
        # Enable gradients for Fisher computation
        for param in variables:
            param.requires_grad_(True)
        
        try:
            # Forward pass through STG-NF model
            _, nll = self.model(data.float(), label=labels, score=scores)
            
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
            for i, grad in enumerate(grads):
                if grad is not None:
                    sample_fishers[i] = grad ** 2
                else:
                    sample_fishers[i] = torch.zeros_like(variables[i], device=self.device)
                    
        except RuntimeError as e:
            self.logger.warning(f"Error computing Fisher for sample: {e}")
            # Return zero Fisher if computation fails
            for i, param in enumerate(variables):
                sample_fishers[i] = torch.zeros_like(param, device=self.device)
        
        return sample_fishers
    
    def compute_fisher_for_model(self, dataloader, max_batches: int = 100) -> List[torch.Tensor]:
        """Compute Fisher Information Matrix for the entire model."""
        self.logger.info("Computing Fisher Information Matrix for STG-NF model...")
        
        variables = self.get_mergeable_parameters()
        self.logger.info(f"Found {len(variables)} mergeable parameters")
        
        # Initialize Fisher accumulators
        fishers = []
        for param in variables:
            fishers.append(torch.zeros_like(param, device=self.device))
        
        self.model.eval()
        n_batches = 0
        
        # Process data in batches
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Computing Fisher", leave=False)):
            if batch_idx >= max_batches:
                self.logger.info(f"Reached maximum batches limit ({max_batches})")
                break
                
            try:
                batch_fishers = self._compute_fisher_for_batch(batch_data, variables)
                
                # Accumulate Fisher information
                for i, batch_fisher in enumerate(batch_fishers):
                    fishers[i] += batch_fisher.detach()
                
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
            for fisher in fishers:
                fisher /= n_batches
        
        self.logger.info(f"Fisher computation completed. Processed {n_batches} batches.")
        return fishers


def save_fisher_info(fishers: List[torch.Tensor], save_path: str, 
                    logger: Optional[logging.Logger] = None):
    """Save Fisher Information to disk."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Convert to CPU and save as list
    fisher_list = [fisher.cpu() for fisher in fishers]
    torch.save(fisher_list, save_path)
    logger.info(f"Saved Fisher Information to {save_path}")


def load_fisher_info(load_path: str, device: torch.device, 
                    logger: Optional[logging.Logger] = None) -> List[torch.Tensor]:
    """Load Fisher Information from disk."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    fisher_list = torch.load(load_path, map_location=device)
    logger.info(f"Loaded Fisher Information from {load_path}")
    return fisher_list
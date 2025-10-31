import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Sequence
import logging
from collections import namedtuple
from tqdm import tqdm

from models.STG_NF.model_pose import STG_NF
from fisher_stg_nf import FisherSTGNF
from utils.scoring_utils import score_dataset

MergeResult = namedtuple("MergeResult", ["coefficients", "score"])


class FisherSoupSTGNF:
    """Fisher Soup implementation for STG-NF models with memory optimization."""
    
    def __init__(self, device: torch.device, logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
    
    def print_merge_result(self, result: MergeResult):
        """Print merge result in a readable format."""
        self.logger.info(f"Merging coefficients: {result.coefficients}")
        self.logger.info("Scores:")
        for name, value in result.score.items():
            self.logger.info(f"  {name}: {value:.4f}")
    
    def create_pairwise_grid_coeffs(self, n_weightings: int) -> List[Tuple[float, float]]:
        """Create pairwise grid coefficients for two models."""
        n_weightings -= 2
        denom = n_weightings + 1
        weightings = [((i + 1) / denom, 1 - (i + 1) / denom) for i in range(n_weightings)]
        weightings = [(0.0, 1.0)] + weightings + [(1.0, 0.0)]
        weightings.reverse()
        return weightings
    
    def create_random_coeffs(self, n_models: int, n_weightings: int, seed: Optional[int] = None) -> List[List[float]]:
        """Create random coefficients using Dirichlet distribution."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Sample from Dirichlet distribution
        alpha = np.ones(n_models)
        coefficients = []
        for _ in range(n_weightings):
            coeff = np.random.dirichlet(alpha)
            coefficients.append(coeff.tolist())
        
        return coefficients
    
    def _should_skip_parameter(self, name: str, tensor: torch.Tensor) -> bool:
        """Check if parameter should be skipped during merging."""
        if name.endswith("actnorm.inited"):
            return True
        if tensor.ndim <= 1:
            return True
        if any(token in name for token in [".actnorm.", "running_mean", "running_var", 
                                          "num_batches_tracked", "prior_h"]):
            return True
        return False
    
    def _merge_with_coeffs(self, 
                          models: List[STG_NF],
                          coefficients: Sequence[float],
                          fishers: Optional[List[List[torch.Tensor]]] = None,
                          fisher_floor: float = 1e-6,
                          favor_target_model: bool = True,
                          normalize_fishers: bool = True,
                          combined_mask: Optional[Dict[str, torch.Tensor]] = None) -> STG_NF:
        """Merge models using Fisher-weighted averaging."""
        n_models = len(models)
        assert len(coefficients) == n_models
        
        # Create output model as a copy of the first model
        output_model = self.clone_model(models[0])
        output_state = output_model.state_dict()
        
        # Get state dicts from all models
        model_states = [model.state_dict() for model in models]
        
        # Normalize Fisher information if requested
        fisher_norms = None
        if normalize_fishers and fishers is not None:
            fisher_norms = []
            for fisher_list in fishers:
                norm_const = torch.sqrt(sum(torch.sum(f ** 2) for f in fisher_list))
                fisher_norms.append(norm_const)
        
        # Map parameter names to Fisher indices
        param_to_fisher_idx = {}
        if fishers is not None:
            fisher_idx = 0
            for name, tensor in output_state.items():
                if not self._should_skip_parameter(name, tensor):
                    param_to_fisher_idx[name] = fisher_idx
                    fisher_idx += 1
        
        # Merge each parameter
        for name, output_tensor in output_state.items():
            if self._should_skip_parameter(name, output_tensor):
                continue
                
            # Collect tensors from all models
            tensors = [state[name] for state in model_states]
            
            # Initialize accumulators
            numerator = None
            denominator = None
            
            for model_idx, (tensor, coeff) in enumerate(zip(tensors, coefficients)):
                # Get Fisher information for this parameter
                fisher_diag = 1.0
                if fishers is not None and name in param_to_fisher_idx:
                    fisher_idx = param_to_fisher_idx[name]
                    if fisher_idx < len(fishers[model_idx]):
                        fisher_diag = fishers[model_idx][fisher_idx]
                        
                        # Apply normalization
                        if fisher_norms is not None:
                            fisher_diag = fisher_diag / fisher_norms[model_idx]
                
                # Apply fisher floor (except for target model if favor_target_model is True)
                if not favor_target_model or model_idx != 0:
                    if isinstance(fisher_diag, torch.Tensor):
                        fisher_diag = torch.clamp(fisher_diag, min=fisher_floor)
                    else:
                        fisher_diag = max(fisher_diag, fisher_floor)
                
                # Compute weighted terms
                weight = coeff * fisher_diag
                contrib = tensor * weight
                
                if numerator is None:
                    numerator = contrib
                    denominator = weight
                else:
                    numerator = numerator + contrib
                    denominator = denominator + weight
            
            # Update output parameter
            if isinstance(denominator, torch.Tensor):
                denominator = torch.clamp(denominator, min=1e-12)
            else:
                denominator = max(denominator, 1e-12)
                
            output_state[name] = numerator / denominator
        
        # Apply combined mask if provided
        if combined_mask is not None:
            self.apply_mask_to_state_dict(output_state, combined_mask)
        
        # Load the merged state dict
        output_model.load_state_dict(output_state)
        return output_model
    
    def clone_model(self, model: STG_NF) -> STG_NF:
        """Create a deep copy of the model."""
        # Get model arguments from the original model
        model_args = {
            'n_blocks': getattr(model, 'n_blocks', 6),
            'input_size': getattr(model, 'input_size', 2),
            'hidden_size': getattr(model, 'hidden_size', 256),
            'n_hidden': getattr(model, 'n_hidden', 1),
            'cond_label_size': getattr(model, 'cond_label_size', 0),
            'score_transform': getattr(model, 'score_transform', 'none'),
        }
        
        # Create new model with same architecture
        new_model = STG_NF(**model_args)
        new_model.load_state_dict(model.state_dict())
        new_model.to(self.device)
        return new_model
    
    def combine_masks(self, masks: Sequence[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
        """Combine multiple pruning masks using OR logic."""
        combined: Dict[str, torch.Tensor] = {}
        has_mask = False
        
        for mask in masks:
            if mask is None:
                continue
            has_mask = True
            for key, tensor in mask.items():
                bool_tensor = (tensor != 0).to(torch.bool)
                if key not in combined:
                    combined[key] = bool_tensor.clone()
                else:
                    combined[key] = torch.logical_or(combined[key], bool_tensor)
        
        if not has_mask:
            return None
        return combined
    
    def apply_mask_to_state_dict(self, state_dict: Dict[str, torch.Tensor], 
                                mask: Optional[Dict[str, torch.Tensor]]):
        """Apply a pruning mask to state dict."""
        if mask is None:
            return
            
        for name, mask_tensor in mask.items():
            if name in state_dict:
                tensor = state_dict[name]
                mask_on_device = mask_tensor.to(tensor.device, dtype=tensor.dtype)
                state_dict[name] = tensor * mask_on_device
    
    def apply_mask_to_model(self, model: STG_NF, mask: Optional[Dict[str, torch.Tensor]]):
        """Apply a pruning mask in-place to the model parameters."""
        if mask is None:
            return
            
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for name, mask_tensor in mask.items():
                if name in param_dict:
                    param = param_dict[name]
                    mask_on_device = mask_tensor.to(param.device, dtype=param.dtype)
                    param.data.mul_(mask_on_device)
    
    def generate_merged_for_coeffs_set(self,
                                     models: List[STG_NF],
                                     coefficients_set: Sequence[Sequence[float]],
                                     fishers: Optional[List[List[torch.Tensor]]] = None,
                                     fisher_floor: float = 1e-6,
                                     favor_target_model: bool = True,
                                     normalize_fishers: bool = True,
                                     combined_mask: Optional[Dict[str, torch.Tensor]] = None):
        """Generate merged models for a set of coefficients."""
        for coefficients in coefficients_set:
            merged_model = self._merge_with_coeffs(
                models=models,
                coefficients=coefficients,
                fishers=fishers,
                fisher_floor=fisher_floor,
                favor_target_model=favor_target_model,
                normalize_fishers=normalize_fishers,
                combined_mask=combined_mask
            )
            yield coefficients, merged_model
    
    def evaluate_model(self, model: STG_NF, test_loader, dataset_test, args) -> Dict[str, float]:
        """Evaluate a single model and return metrics."""
        model.eval()
        
        probs = torch.empty(0, device=self.device)
        
        with torch.no_grad():
            for data_arr in tqdm(test_loader, desc="Evaluating", leave=False):
                data = [d.to(self.device, non_blocking=True) for d in data_arr]
                score = data[-2].amin(dim=-1)
                
                if getattr(args, 'model_confidence', False):
                    samp = data[0]
                else:
                    samp = data[0][:, :2]
                
                _, nll = model(samp.float(), label=torch.ones(data[0].shape[0], device=self.device), score=score)
                
                if getattr(args, 'model_confidence', False):
                    nll = nll * score
                    
                probs = torch.cat((probs, -1 * nll), dim=0)
        
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        auc, scores_np, labels_np, roc_parts = score_dataset(prob_mat_np, dataset_test.metadata, args=args)
        
        return {
            'auc': auc,
            'scores': scores_np,
            'labels': labels_np,
            'roc_parts': roc_parts
        }
    
    def search_merging_coefficients(self,
                                  models: List[STG_NF],
                                  coefficients_set: Sequence[Sequence[float]],
                                  test_loader,
                                  dataset_test,
                                  args,
                                  fishers: Optional[List[List[torch.Tensor]]] = None,
                                  fisher_floor: float = 1e-6,
                                  favor_target_model: bool = True,
                                  normalize_fishers: bool = True,
                                  combined_mask: Optional[Dict[str, torch.Tensor]] = None,
                                  print_results: bool = True) -> List[MergeResult]:
        """Search for optimal merging coefficients."""
        self.logger.info(f"Searching merging coefficients with {len(coefficients_set)} combinations...")
        
        merged_models = self.generate_merged_for_coeffs_set(
            models=models,
            coefficients_set=coefficients_set,
            fishers=fishers,
            fisher_floor=fisher_floor,
            favor_target_model=favor_target_model,
            normalize_fishers=normalize_fishers,
            combined_mask=combined_mask
        )
        
        results = []
        for coeffs, merged_model in merged_models:
            eval_result = self.evaluate_model(merged_model, test_loader, dataset_test, args)
            
            result = MergeResult(
                coefficients=coeffs, 
                score={'auc': eval_result['auc']}
            )
            results.append(result)
            
            if print_results:
                self.print_merge_result(result)
            
            # Clean up merged model to save memory
            del merged_model
            torch.cuda.empty_cache()
        
        # Find best result
        best_result = max(results, key=lambda x: x.score['auc'])
        self.logger.info(f"Best result - Coefficients: {best_result.coefficients}, AUC: {best_result.score['auc']:.4f}")
        
        return results


def load_models_and_fishers(checkpoint_paths: List[str], 
                          fisher_paths: Optional[List[str]],
                          model_args: Dict,
                          device: torch.device,
                          logger: Optional[logging.Logger] = None) -> Tuple[List[STG_NF], Optional[List[List[torch.Tensor]]], List[Optional[Dict[str, torch.Tensor]]]]:
    """Load STG-NF models and their Fisher information."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    models = []
    fishers = None
    masks: List[Optional[Dict[str, torch.Tensor]]] = []
    
    # Load models
    for i, ckpt_path in enumerate(checkpoint_paths):
        logger.info(f"Loading model {i+1}/{len(checkpoint_paths)}: {ckpt_path}")
        
        model = STG_NF(**model_args)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")
        
        model.load_state_dict(state_dict)
        model.to(device)
        models.append(model)
        
        # Load mask if exists
        mask_path = ckpt_path + ".mask"
        if torch.load(mask_path, map_location='cpu') if hasattr(torch, 'load') else None:
            try:
                raw_mask_dict = torch.load(mask_path, map_location='cpu')
                bool_mask = {key: (tensor != 0).to(torch.bool) for key, tensor in raw_mask_dict.items()}
                masks.append(bool_mask)
            except:
                masks.append(None)
        else:
            masks.append(None)
    
    # Load Fisher information if provided
    if fisher_paths is not None:
        assert len(fisher_paths) == len(checkpoint_paths), "Number of Fisher files must match number of checkpoints"
        fishers = []
        
        for i, fisher_path in enumerate(fisher_paths):
            logger.info(f"Loading Fisher info {i+1}/{len(fisher_paths)}: {fisher_path}")
            fisher_list = torch.load(fisher_path, map_location=device)
            fishers.append(fisher_list)
    
    return models, fishers, masks
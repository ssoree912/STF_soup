"""
Masked ensemble variant of STG-NF with utilities inspired by ``nflows_epistemic``.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

from models.STG_NF.model_pose import STG_NF
from .masked_layers import MaskedLinear, EnsembleUncertaintyEstimator


class MaskedEnsembleSTGNF(nn.Module):
    """Wraps STG-NF and replaces linear layers with masked ensemble counterparts."""

    def __init__(
        self,
        base_stg_nf_args: dict,
        ensemble_size: int = 3,
        mask_type: str = "random",
        device: str = "cuda",
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.mask_type = mask_type
        self.device = device

        self.base_model = STG_NF(**base_stg_nf_args)
        self._replace_linear_layers()
        self.uncertainty_estimator = EnsembleUncertaintyEstimator(self, ensemble_size)
        self.to(device)

    def _replace_linear_layers(self):
        def replace_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    masked_linear = MaskedLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        ensemble_size=self.ensemble_size,
                        bias=child.bias is not None,
                        mask_type=self.mask_type,
                        device=self.device,
                    )
                    with torch.no_grad():
                        masked_linear.weight.copy_(child.weight)
                        if child.bias is not None:
                            masked_linear.bias.copy_(child.bias)
                    setattr(module, name, masked_linear)
                else:
                    replace_recursive(child)

        replace_recursive(self.base_model)

    def forward(
        self,
        x: torch.Tensor,
        label: torch.Tensor = None,
        score: torch.Tensor = None,
        mask_index: Optional[int] = None,
        return_all_ensemble: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if return_all_ensemble:
            outputs = []
            nlls = []
            for idx in range(self.ensemble_size):
                with MaskedForwardContext(self.base_model, idx):
                    z, nll = self.base_model(x, label=label, score=score)
                outputs.append(z)
                nlls.append(nll)
            return torch.stack(outputs), torch.stack(nlls)

        if mask_index is not None:
            with MaskedForwardContext(self.base_model, mask_index):
                return self.base_model(x, label=label, score=score)

        outputs = []
        nlls = []
        for idx in range(self.ensemble_size):
            with MaskedForwardContext(self.base_model, idx):
                z, nll = self.base_model(x, label=label, score=score)
            outputs.append(z)
            nlls.append(nll)
        return torch.stack(outputs).mean(dim=0), torch.stack(nlls).mean(dim=0)

    def sample_and_log_prob(
        self,
        context: torch.Tensor,
        num_samples: int = 100,
        mask_index: Optional[int] = None,
        return_uncertainty: bool = False,
    ):
        if return_uncertainty:
            return self.uncertainty_estimator.compute_ensemble_predictions(
                context, num_samples=num_samples
            )

        if mask_index is not None:
            with MaskedForwardContext(self.base_model, mask_index):
                return self._sample_from_base_model(context, num_samples)

        all_samples = []
        all_logprobs = []
        per_member_samples = max(1, num_samples // self.ensemble_size)
        for idx in range(self.ensemble_size):
            with MaskedForwardContext(self.base_model, idx):
                samples, logprobs = self._sample_from_base_model(context, per_member_samples)
            all_samples.append(samples)
            all_logprobs.append(logprobs)
        return torch.cat(all_samples, dim=0), torch.cat(all_logprobs, dim=0)

    def _sample_from_base_model(self, context: torch.Tensor, num_samples: int):
        batch_size = context.shape[0]
        samples = torch.randn(num_samples, batch_size, *context.shape[1:]).to(self.device)
        log_probs = -0.5 * (samples**2).flatten(2).sum(dim=2) - 0.5 * np.log(2 * np.pi)
        return samples, log_probs

    def compute_epistemic_uncertainty(self, x: torch.Tensor, num_samples: int = 100):
        self.eval()
        model_logps = {}
        with torch.no_grad():
            for idx in range(self.ensemble_size):
                logps = []
                with MaskedForwardContext(self.base_model, idx):
                    for _ in range(max(1, num_samples // 10)):
                        _, nll = self.base_model(x)
                        logps.append(-nll)
                model_logps[f"model_{idx}"] = torch.cat(logps, dim=0)
        logp_stack = torch.stack(list(model_logps.values()), dim=0)
        epistemic_var = torch.var(logp_stack, dim=0, unbiased=True)
        weights = 1.0 / (epistemic_var + 1e-8)
        weights = weights / weights.mean()
        return epistemic_var, weights

    def set_actnorm_init(self):
        if hasattr(self.base_model, "set_actnorm_init"):
            self.base_model.set_actnorm_init()

    def state_dict(self, *args, **kwargs):
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.base_model.load_state_dict(state_dict, *args, **kwargs)


class MaskedForwardContext:
    """Context manager that injects a fixed mask index for MaskedLinear layers."""

    def __init__(self, model: nn.Module, mask_index: int):
        self.model = model
        self.mask_index = mask_index
        self.original_forwards = {}

    def __enter__(self):
        def patch(module, path=""):
            for name, child in module.named_children():
                full_name = f"{path}.{name}" if path else name
                if isinstance(child, MaskedLinear):
                    self.original_forwards[full_name] = child.forward

                    def forward_with_mask(x, _child=child):
                        return MaskedLinear.forward(_child, x, mask_index=self.mask_index)

                    child.forward = forward_with_mask
                else:
                    patch(child, full_name)

        patch(self.model)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        def restore(module, path=""):
            for name, child in module.named_children():
                full_name = f"{path}.{name}" if path else name
                if isinstance(child, MaskedLinear) and full_name in self.original_forwards:
                    child.forward = self.original_forwards[full_name]
                else:
                    restore(child, full_name)

        restore(self.model)


def create_masked_ensemble_stg_nf(
    original_args: dict,
    ensemble_size: int = 3,
    mask_type: str = "random",
    device: str = "cuda",
) -> MaskedEnsembleSTGNF:
    ensemble_args = copy.deepcopy(original_args)
    ensemble_args["device"] = device
    return MaskedEnsembleSTGNF(
        base_stg_nf_args=ensemble_args,
        ensemble_size=ensemble_size,
        mask_type=mask_type,
        device=device,
    )

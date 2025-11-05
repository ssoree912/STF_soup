"""
Masked ensemble building blocks for STG-NF.

These layers mirror the fixed-mask strategy used in ``nflows_epistemic`` while
remaining compatible with the existing STG-NF architecture.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    """Linear layer with pre-sampled binary masks per ensemble member."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int = 3,
        bias: bool = True,
        mask_type: str = "random",
        device: str = "cuda",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.mask_type = mask_type
        self.device = device

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.register_buffer("masks", self._generate_masks())
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _generate_masks(self) -> torch.Tensor:
        masks = torch.zeros(self.ensemble_size, self.out_features, self.in_features)
        if self.mask_type == "random":
            for i in range(self.ensemble_size):
                masks[i] = torch.bernoulli(torch.full((self.out_features, self.in_features), 0.7))
        elif self.mask_type == "dropout_pattern":
            dropout_rates = np.linspace(0.1, 0.5, self.ensemble_size)
            for i, rate in enumerate(dropout_rates):
                masks[i] = torch.bernoulli(torch.full((self.out_features, self.in_features), 1 - rate))
        elif self.mask_type == "block_pattern":
            block_size = max(1, self.in_features // self.ensemble_size)
            for i in range(self.ensemble_size):
                mask = torch.zeros(self.out_features, self.in_features)
                start_idx = i * block_size
                end_idx = min((i + 1) * block_size, self.in_features)
                mask[:, start_idx:end_idx] = 1.0
                extra = torch.bernoulli(torch.full((self.out_features, self.in_features), 0.3))
                extra[:, start_idx:end_idx] = 0
                masks[i] = torch.clamp(mask + extra, 0, 1)
        elif self.mask_type == "alternating":
            for i in range(self.ensemble_size):
                mask = torch.zeros(self.out_features, self.in_features)
                if i % 2 == 0:
                    mask[:, ::2] = 1.0
                else:
                    mask[:, 1::2] = 1.0
                extra = torch.bernoulli(torch.full((self.out_features, self.in_features), 0.2))
                masks[i] = torch.clamp(mask + extra, 0, 1)
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")
        return masks

    def forward(self, x: torch.Tensor, mask_index: Optional[int] = None) -> torch.Tensor:
        if mask_index is not None:
            if not 0 <= mask_index < self.ensemble_size:
                raise ValueError(f"mask_index must be within [0, {self.ensemble_size - 1}]")
            masked_weight = self.weight * self.masks[mask_index]
            return F.linear(x, masked_weight, self.bias)

        outputs = []
        for i in range(self.ensemble_size):
            masked_weight = self.weight * self.masks[i]
            outputs.append(F.linear(x, masked_weight, self.bias))
        return torch.stack(outputs).mean(dim=0)


class MaskedMLPBlock(nn.Module):
    """Two-layer MLP block that reuses MaskedLinear at each stage."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        ensemble_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        mask_type: str = "random",
    ):
        super().__init__()
        self.linear1 = MaskedLinear(
            in_features,
            hidden_features,
            ensemble_size=ensemble_size,
            mask_type=mask_type,
        )
        self.linear2 = MaskedLinear(
            hidden_features,
            out_features,
            ensemble_size=ensemble_size,
            mask_type=mask_type,
        )

        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        self.activation = activations.get(activation, nn.ReLU())
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor, mask_index: Optional[int] = None) -> torch.Tensor:
        x = self.linear1(x, mask_index=mask_index)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear2(x, mask_index=mask_index)


class EnsembleUncertaintyEstimator:
    """Helper for sampling ensemble predictions and computing simple metrics."""

    def __init__(self, model, ensemble_size: int = 3):
        self.model = model
        self.ensemble_size = ensemble_size

    def _stack_outputs(self, outputs):
        return torch.stack(outputs) if len(outputs) > 1 else outputs[0].unsqueeze(0)

    def compute_ensemble_predictions(self, x: torch.Tensor, num_samples: int = 100):
        self.model.eval()
        predictions = []
        log_probs = []
        with torch.no_grad():
            per_member_samples = max(1, num_samples // self.ensemble_size)
            for mask_idx in range(self.ensemble_size):
                member_samples = []
                member_logps = []
                for _ in range(per_member_samples):
                    sample, logp = self.model.sample_and_log_prob(x, mask_index=mask_idx)
                    member_samples.append(sample)
                    member_logps.append(logp)
                predictions.append(torch.stack(member_samples))
                log_probs.append(torch.stack(member_logps))

        all_predictions = torch.stack(predictions)
        all_log_probs = torch.stack(log_probs)
        return {
            "predictions": all_predictions,
            "log_probs": all_log_probs,
            "uncertainty": self._compute_uncertainty_metrics(all_predictions, all_log_probs),
        }

    def _compute_uncertainty_metrics(self, predictions: torch.Tensor, log_probs: torch.Tensor):
        individual_entropies = -log_probs.mean(dim=1)
        aleatoric = individual_entropies.mean(dim=0)
        flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
        total_entropy = -flat_log_probs.mean(dim=0)
        epistemic = total_entropy - aleatoric
        flat_preds = predictions.reshape(-1, *predictions.shape[2:])
        pred_mean = flat_preds.mean(dim=0)
        pred_var = ((flat_preds - pred_mean) ** 2).mean(dim=0)
        return {
            "total_entropy": total_entropy,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "predictive_variance": pred_var,
            "individual_entropies": individual_entropies,
        }

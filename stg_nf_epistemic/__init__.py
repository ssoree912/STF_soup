"""
Utilities for epistemic ensemble training with STG-NF.

This package mirrors the mixture-based log probability aggregation strategy
from ``nflows_epistemic`` while reusing the masked-ensemble STG-NF modules
defined in the main codebase.
"""

from .mixture_utils import (
    compute_mixture_nll,
    nlls_to_log_probs,
    log_probs_to_nll,
)
from .masked_model import MaskedEnsembleSTGNF, create_masked_ensemble_stg_nf
from .masked_trainer import MaskedEnsembleTrainer, create_ensemble_trainer
from .trainer import EpistemicMaskedEnsembleTrainer, create_epistemic_trainer

__all__ = [
    "compute_mixture_nll",
    "nlls_to_log_probs",
    "log_probs_to_nll",
    "MaskedEnsembleSTGNF",
    "create_masked_ensemble_stg_nf",
    "MaskedEnsembleTrainer",
    "create_ensemble_trainer",
    "EpistemicMaskedEnsembleTrainer",
    "create_epistemic_trainer",
]

import math
from functools import reduce
from operator import mul
from typing import List, Sequence

import torch


def _feature_volume_from_shape(sample_shape: Sequence[int]) -> int:
    """
    Compute the number of feature elements (C * T * V) for a pose tensor shape.

    Args:
        sample_shape: Tensor shape sequence that includes batch dimension as the
            first element followed by feature dimensions.

    Returns:
        Product of the feature dimensions (i.e. excluding the batch axis).
    """
    if len(sample_shape) < 2:
        raise ValueError("sample_shape must include batch and feature dimensions.")
    return reduce(mul, sample_shape[1:], 1)


def nlls_to_log_probs(
    member_nlls: Sequence[torch.Tensor],
    sample_shape: Sequence[int],
) -> torch.Tensor:
    """
    Convert per-member negative log-likelihoods (in bits per dimension) into log
    probabilities.

    Args:
        member_nlls: Iterable of tensors shaped [batch_size] representing the NLL
            (bits/dim) for each ensemble member.
        sample_shape: Shape tuple describing the input pose tensor, typically
            [batch_size, channels, time, joints].

    Returns:
        Tensor of shape [ensemble_size, batch_size] containing log probabilities
        (natural log) for each member and sample.
    """
    if not member_nlls:
        raise ValueError("member_nlls must contain at least one tensor.")

    volume = _feature_volume_from_shape(sample_shape)
    log_factor = math.log(2.0) * volume

    log_probs: List[torch.Tensor] = []
    for nll in member_nlls:
        if nll.dim() != 1:
            raise ValueError("Each NLL tensor must be 1-D with shape [batch_size].")
        log_probs.append(-nll * log_factor)

    return torch.stack(log_probs, dim=0)


def log_probs_to_nll(
    log_probs: torch.Tensor,
    sample_shape: Sequence[int],
) -> torch.Tensor:
    """
    Convert log probabilities back into bits-per-dimension negative log-likelihoods.

    Args:
        log_probs: Tensor of log probabilities shaped [..., batch_size].
        sample_shape: Input tensor shape (including batch dimension).

    Returns:
        Tensor of NLL values (bits/dim) matching the leading dimensions of
        ``log_probs`` aside from the batch axis.
    """
    volume = _feature_volume_from_shape(sample_shape)
    log_factor = math.log(2.0) * volume
    return -log_probs / log_factor


def compute_mixture_nll(
    member_nlls: Sequence[torch.Tensor],
    sample_shape: Sequence[int],
) -> torch.Tensor:
    """
    Aggregate ensemble member NLLs into a single mixture NLL via log-mean-exp.

    The implementation mirrors the mixture handling in ``nflows_epistemic`` where
    member likelihoods are averaged in probability space.

    Args:
        member_nlls: Sequence of tensors [batch_size], one per ensemble member.
        sample_shape: Shape tuple including batch and feature dims for scaling.

    Returns:
        Tensor [batch_size] containing the mixture NLL (bits/dim).
    """
    if not member_nlls:
        raise ValueError("member_nlls must contain at least one tensor.")

    log_probs = nlls_to_log_probs(member_nlls, sample_shape)
    mixture_log_prob = torch.logsumexp(log_probs, dim=0) - math.log(len(member_nlls))
    return log_probs_to_nll(mixture_log_prob, sample_shape)

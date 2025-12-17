#!/usr/bin/env python3
"""
Uncertainty-weighted Fisher extractor for STG-NF checkpoints (UGM-ready).
  - Computes epistemic uncertainty from multiple checkpoints on the train loader
  - Uses that to weight Fisher information and saves files consumable by ugm_soup.py

Example:
python compute_uw_fisher.py \
  --reference_args experiments/.../args.json \
  --checkpoints ckpt_seed0.pth.tar ckpt_seed1.pth.tar \
  --output_dir results/fisher_uw \
  --device cuda:0 --max_batches 80 --subsample_ratio 0.5 --uw_gamma 1.0
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from args import init_sub_args
from dataset import get_dataset_and_loader
from models.STG_NF.model_pose import STG_NF
from tools.soup.epistemic_uncertainty import EpistemicUncertainty
from tools.soup.fisher_stg_nf import FisherSTGNF
from tools.soup.fisher_soup_stg_nf import FisherSoupSTGNF, _normalize_state_dict
from ugm_soup import _normalize_fisher_to_state_dict
from utils.data_utils import trans_list
from utils.train_utils import init_model_params


def setup_logger(log_level: str) -> logging.Logger:
    logger = logging.getLogger("compute_uw_fisher")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_sequential_loader(orig_loader: DataLoader) -> DataLoader:
    """Rebuild a DataLoader with deterministic ordering."""
    dataset = getattr(orig_loader, "dataset", None)
    if dataset is None:
        raise RuntimeError("Original loader has no dataset; cannot build sequential loader.")

    sampler = SequentialSampler(dataset)
    dl_kwargs = dict(
        batch_size=orig_loader.batch_size,
        sampler=sampler,
        num_workers=orig_loader.num_workers,
        pin_memory=getattr(orig_loader, "pin_memory", False),
        drop_last=getattr(orig_loader, "drop_last", False),
    )
    collate_fn = getattr(orig_loader, "collate_fn", None)
    if collate_fn is not None:
        dl_kwargs["collate_fn"] = collate_fn
    prefetch_factor = getattr(orig_loader, "prefetch_factor", None)
    if prefetch_factor is not None and dl_kwargs["num_workers"] > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor
    if getattr(orig_loader, "persistent_workers", False) and dl_kwargs["num_workers"] > 0:
        dl_kwargs["persistent_workers"] = True

    return DataLoader(dataset, **dl_kwargs)


def load_reference_args(args_path: Path, device_override: str = None) -> argparse.Namespace:
    with open(args_path, "r") as fp:
        ref_dict = json.load(fp)
    ref_args = argparse.Namespace(**ref_dict)
    if device_override:
        ref_args.device = device_override
    ref_args.only_test = False
    return ref_args


def load_model(ckpt_path: Path,
               model_args: Dict,
               device: torch.device,
               logger: logging.Logger) -> Tuple[STG_NF, Dict[str, torch.Tensor]]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")

    state = _normalize_state_dict(state)

    model = STG_NF(**model_args)
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    if missing_keys:
        logger.warning("Missing keys when loading %s: %s", ckpt_path, missing_keys)
    if unexpected_keys:
        logger.warning("Unexpected keys when loading %s: %s", ckpt_path, unexpected_keys)
    if hasattr(model, "set_actnorm_init"):
        model.set_actnorm_init()
    model.to(device)
    model.eval()
    return model, state


def compute_uncertainty_weights(models: List[STG_NF],
                                loader: DataLoader,
                                ref_args: argparse.Namespace,
                                fisher_helper: FisherSoupSTGNF,
                                args: argparse.Namespace,
                                device: torch.device,
                                logger: logging.Logger) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
    """Compute stabilized uncertainty weights using existing helpers."""
    unc_calc = EpistemicUncertainty(models, device, logger)
    model_logps = unc_calc.compute_log_likelihoods(loader, ref_args, max_batches=args.max_batches)
    epistemic_var = unc_calc.compute_epistemic_uncertainty(
        model_logps,
        unbiased=bool(args.uw_var_unbiased),
    )
    var_stats = {
        "mean": float(epistemic_var.mean().item()),
        "std": float(epistemic_var.std(unbiased=False).item()),
        "min": float(epistemic_var.min().item()),
        "max": float(epistemic_var.max().item()),
        "q10": float(torch.quantile(epistemic_var, 0.10).item()),
        "q50": float(torch.quantile(epistemic_var, 0.50).item()),
        "q90": float(torch.quantile(epistemic_var, 0.90).item()),
    }

    weights, weight_stats = fisher_helper._stabilize_uncertainty_weights(
        epistemic_var,
        eps=1e-8,
        shrink_alpha=float(args.uw_shrink_alpha),
        gamma=float(args.uw_gamma),
        qclip=float(args.uw_qclip),
        wmin=float(args.uw_wmin),
        wmax=float(args.uw_wmax) if args.uw_wmax > 0.0 else None,
    )
    fisher_helper.last_uncertainty_var_stats = var_stats
    fisher_helper.last_uncertainty_weight_stats = weight_stats
    fisher_helper.last_uncertainty_weights = weights

    logger.info(
        "Uncertainty weights ready | mean=%.4f std=%.4f min=%.4f max=%.4f corr=%.4f",
        weight_stats["mean"],
        weight_stats["std"],
        weight_stats["min"],
        weight_stats["max"],
        weight_stats["corr"],
    )
    return weights, var_stats, weight_stats


def clamp_and_mix_fisher(fisher_dict: Dict[str, torch.Tensor],
                         fisher_helper: FisherSoupSTGNF,
                         mix_eta: float,
                         floor: float) -> Dict[str, torch.Tensor]:
    mixed = fisher_helper._mix_uniform_with_fisher(fisher_dict, mix_eta)
    if floor > 0.0:
        return {k: torch.clamp(v, min=floor) for k, v in mixed.items()}
    return mixed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute uncertainty-weighted Fisher matrices for UGM experiments."
    )
    ap.add_argument("--reference_args", type=Path, required=True, help="Path to training args.json")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to process")
    ap.add_argument("--output_dir", type=Path, required=True, help="Directory to save UW-Fisher files")
    ap.add_argument("--device", default=None, help="Device override (cpu / cuda[:id] / mps)")
    ap.add_argument("--max_batches", type=int, default=100, help="Max batches for uncertainty/Fisher computation")
    ap.add_argument("--subsample_ratio", type=float, default=0.5, help="Subsample ratio for Fisher pass (0,1]")
    ap.add_argument("--fisher_eps", type=float, default=1e-8, help="Epsilon when aligning Fisher to state_dict")
    ap.add_argument("--fisher_floor", type=float, default=1e-8, help="Clamp Fisher entries to at least this value")
    ap.add_argument("--fisher_mix_eta", type=float, default=0.0, help="Mix ratio with uniform Fisher (0 disables)")
    ap.add_argument("--exact", action="store_true", help="Use sample-wise Fisher (slower but exact)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--uw_var_unbiased", action="store_true", help="Use unbiased variance for epistemic uncertainty")
    ap.add_argument("--uw_shrink_alpha", type=float, default=0.0, help="Shrink variance toward mean [0,1]")
    ap.add_argument("--uw_gamma", type=float, default=1.0, help="Exponent on 1/var (w = var^-gamma)")
    ap.add_argument("--uw_qclip", type=float, default=100.0, help="Quantile clip for weights (0-100, 100=off)")
    ap.add_argument("--uw_wmin", type=float, default=0.0, help="Minimum weight after normalization")
    ap.add_argument("--uw_wmax", type=float, default=0.0, help="Maximum weight (<=0 disables)")
    ap.add_argument("--log_level", default="INFO", help="Logging level")
    return ap.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    set_seeds(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.subsample_ratio <= 0.0 or args.subsample_ratio > 1.0:
        logger.warning("subsample_ratio %.3f out of range (0,1]; using 1.0", args.subsample_ratio)
        args.subsample_ratio = 1.0

    ref_args = load_reference_args(args.reference_args, args.device)
    ref_args, model_args = init_sub_args(ref_args)
    dataset, loader = get_dataset_and_loader(ref_args, trans_list=trans_list, only_test=False)
    model_args = init_model_params(ref_args, dataset)

    train_loader = loader.get("train")
    if train_loader is None:
        raise RuntimeError("Training loader is required for uncertainty-weighted Fisher computation.")

    device = torch.device(ref_args.device)
    seq_loader = build_sequential_loader(train_loader)

    logger.info("Loading %d checkpoints...", len(args.checkpoints))
    models: List[STG_NF] = []
    model_states: List[Dict[str, torch.Tensor]] = []
    for ckpt in args.checkpoints:
        model, state = load_model(Path(ckpt), model_args, device, logger)
        models.append(model)
        model_states.append(state)

    fisher_helper = FisherSoupSTGNF(device, logger, model_args=model_args)
    weights, var_stats, weight_stats = compute_uncertainty_weights(
        models=models,
        loader=seq_loader,
        ref_args=ref_args,
        fisher_helper=fisher_helper,
        args=args,
        device=device,
        logger=logger,
    )

    logger.info("Computing UW-Fisher for each checkpoint...")
    for idx, (ckpt_path, model, state) in enumerate(zip(args.checkpoints, models, model_states)):
        fisher_calc = FisherSTGNF(model, device, logger, ref_args)
        uw_fisher = fisher_calc.compute_uncertainty_weighted_fisher(
            seq_loader,
            weights,
            max_batches=args.max_batches,
            use_batch_approx=not args.exact,
            subsample_ratio=args.subsample_ratio,
        )
        uw_fisher = clamp_and_mix_fisher(
            uw_fisher,
            fisher_helper=fisher_helper,
            mix_eta=float(args.fisher_mix_eta),
            floor=float(args.fisher_floor),
        )
        uw_fisher = _normalize_fisher_to_state_dict(
            fisher_obj=uw_fisher,
            model_state=state,
            eps=float(args.fisher_eps),
        )
        uw_fisher_cpu = {k: v.detach().cpu() for k, v in uw_fisher.items() if torch.is_tensor(v)}

        ckpt_path = Path(ckpt_path)
        folder_tag = ckpt_path.parent.name or "ckpt"
        out_name = f"fisher_uw_{folder_tag}_{ckpt_path.stem}.pt"
        out_path = args.output_dir / out_name

        payload = {
            "fisher": uw_fisher_cpu,
            "metadata": {
                "type": "uncertainty_weighted",
                "checkpoint": str(ckpt_path),
                "max_batches": int(args.max_batches),
                "subsample_ratio": float(args.subsample_ratio),
                "use_batch_approx": bool(not args.exact),
                "fisher_eps": float(args.fisher_eps),
                "fisher_floor": float(args.fisher_floor),
                "fisher_mix_eta": float(args.fisher_mix_eta),
                "uw_settings": {
                    "uw_var_unbiased": bool(args.uw_var_unbiased),
                    "uw_shrink_alpha": float(args.uw_shrink_alpha),
                    "uw_gamma": float(args.uw_gamma),
                    "uw_qclip": float(args.uw_qclip),
                    "uw_wmin": float(args.uw_wmin),
                    "uw_wmax": float(args.uw_wmax),
                },
                "var_stats": var_stats,
                "weight_stats": weight_stats,
                "num_models": len(models),
            },
        }
        torch.save(payload, out_path)
        logger.info("Saved UW-Fisher %d/%d to %s", idx + 1, len(models), out_path)

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()

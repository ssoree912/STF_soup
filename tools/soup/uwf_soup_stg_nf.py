#!/usr/bin/env python3
"""
Run Uncertainty-Weighted Fisher Soup (UWF-Soup) for STG-NF checkpoints using the
project's standard data/loading utilities and the enhanced FisherSoupSTGNF class.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from args import init_sub_args
from dataset import get_dataset_and_loader
from tools.soup.fisher_soup_stg_nf import FisherSoupSTGNF, load_models_and_fishers
from utils.data_utils import trans_list
from utils.train_utils import init_model_params


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("uwf_soup")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def load_reference_args(path: Path) -> argparse.Namespace:
    with open(path, "r") as fp:
        payload = json.load(fp)
    return argparse.Namespace(**payload)


def dump_results(path: Path, results: List) -> None:
    serializable: List[Dict] = []
    for res in results:
        coeffs = [float(c) for c in res.coefficients]
        score = {k: float(v) for k, v in res.score.items()}
        serializable.append({"coefficients": coeffs, "score": score})
    with open(path, "w") as handle:
        json.dump(serializable, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uncertainty-Weighted Fisher Soup runner for STG-NF checkpoints.")
    parser.add_argument("--reference_args", type=Path, required=True, help="args.json from a reference training run")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint paths to merge")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the merged model")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu / cuda:0")
    parser.add_argument("--mask_name", default="pruning_mask.pt", help="Pruning mask filename")
    parser.add_argument("--n_weightings", type=int, default=10, help="Number of coefficient candidates")
    parser.add_argument("--max_batches", type=int, default=100, help="Max batches for uncertainty/Fisher computation")
    parser.add_argument("--fisher_floor", type=float, default=1e-6, help="Minimum Fisher weight")
    parser.add_argument("--subsample_ratio", type=float, default=0.5, help="Subsample ratio for UW-Fisher (0,1]")
    parser.add_argument("--uw_var_unbiased", action="store_true", help="Use unbiased variance estimate")
    parser.add_argument("--uw_shrink_alpha", type=float, default=0.0, help="Shrinkage coefficient towards mean variance")
    parser.add_argument("--uw_gamma", type=float, default=1.0, help="Power applied to 1/var (w = var^-gamma)")
    parser.add_argument("--uw_qclip", type=float, default=99.5, help="Quantile clip for uncertainty weights (0-100)")
    parser.add_argument("--uw_wmin", type=float, default=0.1, help="Minimum uncertainty weight after normalization")
    parser.add_argument("--uw_wmax", type=float, default=5.0, help="Maximum uncertainty weight (<=0 disables)")
    parser.add_argument("--fisher_mix_eta", type=float, default=0.0, help="Mix ratio with uniform Fisher (0-1)")
    parser.add_argument("--sens_regex", type=str, default="", help="Regex for sensitive params needing special Fisher scaling")
    parser.add_argument("--sens_fisher_gamma", type=float, default=1.0, help="Gamma for sensitive parameter Fisher scaling")
    parser.add_argument("--sens_fisher_qclip", type=float, default=100.0, help="Quantile clip for sensitive Fisher")
    parser.add_argument("--actnorm_after_steps", type=int, default=0, help="ActNorm calibration steps after merging")
    parser.add_argument("--no_normalize_fishers", action="store_true", help="Disable Fisher normalization")
    parser.add_argument("--no_favor_target", action="store_true", help="Disable favoring the target model")
    parser.add_argument("--log_level", default="INFO", help="Logging level (INFO/DEBUG/...)")
    parser.add_argument("--save_results_json", action="store_true", help="Save all coefficient evaluations to JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ref_args = load_reference_args(args.reference_args)
    if args.device:
        ref_args.device = args.device
    ref_args.only_test = False
    for attr in [
        "uw_var_unbiased",
        "uw_shrink_alpha",
        "uw_gamma",
        "uw_qclip",
        "uw_wmin",
        "uw_wmax",
        "fisher_mix_eta",
        "sens_regex",
        "sens_fisher_gamma",
        "sens_fisher_qclip",
        "actnorm_after_steps",
    ]:
        setattr(ref_args, attr, getattr(args, attr))

    ref_args, model_args = init_sub_args(ref_args)
    dataset, loader = get_dataset_and_loader(ref_args, trans_list=trans_list, only_test=False)
    model_args = init_model_params(ref_args, dataset)

    device = torch.device(ref_args.device)
    train_loader = loader["train"]
    test_loader = loader["test"]
    dataset_test = dataset["test"]

    if train_loader is None:
        raise RuntimeError("Training loader is required for uncertainty-weighted Fisher computation.")

    models, fishers, masks = load_models_and_fishers(
        checkpoint_paths=[str(p) for p in args.checkpoints],
        fisher_paths=None,
        model_args=model_args,
        device=device,
        logger=logger,
        mask_name=args.mask_name,
    )

    fisher_soup = FisherSoupSTGNF(device, logger, model_args=model_args)
    combined_mask = fisher_soup.combine_masks(masks)
    for model, mask in zip(models, masks):
        fisher_soup.apply_mask_to_model(model, mask)

    results, best_result, best_model = fisher_soup.uncertainty_weighted_fisher_soup(
        models=models,
        dataloader=train_loader,
        test_loader=test_loader,
        dataset_test=dataset_test,
        args=ref_args,
        n_weightings=args.n_weightings,
        fisher_floor=args.fisher_floor,
        favor_target_model=not args.no_favor_target,
        normalize_fishers=not args.no_normalize_fishers,
        combined_mask=combined_mask,
        max_batches=args.max_batches,
        print_results=True,
        subsample_ratio=args.subsample_ratio,
    )

    metadata = {
        "method": "uwf_soup",
        "reference_args": str(args.reference_args),
        "checkpoints": [str(p) for p in args.checkpoints],
        "n_weightings": args.n_weightings,
        "max_batches": args.max_batches,
        "subsample_ratio": args.subsample_ratio,
        "fisher_floor": args.fisher_floor,
        "favor_target_model": not args.no_favor_target,
        "normalize_fishers": not args.no_normalize_fishers,
        "uw_var_unbiased": args.uw_var_unbiased,
        "uw_shrink_alpha": args.uw_shrink_alpha,
        "uw_gamma": args.uw_gamma,
        "uw_qclip": args.uw_qclip,
        "uw_wmin": args.uw_wmin,
        "uw_wmax": args.uw_wmax,
        "fisher_mix_eta": args.fisher_mix_eta,
        "sens_regex": args.sens_regex,
        "sens_fisher_gamma": args.sens_fisher_gamma,
        "sens_fisher_qclip": args.sens_fisher_qclip,
        "actnorm_after_steps": args.actnorm_after_steps,
        "best_coefficients": [float(c) for c in best_result.coefficients],
        "best_score": best_result.score,
    }
    if getattr(fisher_soup, "last_uncertainty_var_stats", None):
        metadata["uncertainty_var_stats"] = fisher_soup.last_uncertainty_var_stats
    if getattr(fisher_soup, "last_uncertainty_weight_stats", None):
        metadata["uncertainty_weight_stats"] = fisher_soup.last_uncertainty_weight_stats

    payload = {
        "state_dict": best_model.state_dict(),
        "soup_metadata": metadata,
    }
    torch.save(payload, args.output)
    logger.info("Saved merged model to %s", args.output)

    if args.save_results_json:
        results_path = args.output.with_suffix(args.output.suffix + ".results.json")
        dump_results(results_path, results)
        logger.info("Saved evaluation results to %s", results_path)

    metadata_path = args.output.with_suffix(args.output.suffix + "_metadata.json")
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    logger.info("Saved metadata to %s", metadata_path)


if __name__ == "__main__":
    main()

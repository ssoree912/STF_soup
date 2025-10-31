#!/usr/bin/env python3
"""
Fisher Soup helper for STG-NF that takes model directories (containing best checkpoints)
and automatically computes Fisher information before performing Fisher-weighted soup.
"""

import argparse
import logging
import os
import sys
import random
import gc
import json
from pathlib import Path
from typing import List, Optional, Sequence
from torch.utils.data import DataLoader, Subset

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from args import init_sub_args
from dataset import get_dataset_and_loader
from models.STG_NF.model_pose import STG_NF
from utils.data_utils import trans_list
from utils.train_utils import init_model_params
from fisher_stg_nf import FisherSTGNF, save_fisher_info
from fisher_soup_stg_nf import FisherSoupSTGNF, load_models_and_fishers


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("folder_fisher_soup_stg_nf")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def load_reference_args(args_path: Path) -> argparse.Namespace:
    """Load reference arguments from args.json file."""
    with open(args_path, "r") as fp:
        args_dict = json.load(fp)
    return argparse.Namespace(**args_dict)


def resolve_device(requested: Optional[str], gpu_id: int, logger: logging.Logger) -> torch.device:
    """Resolve the device to use."""
    if requested:
        req = requested.lower()
        if req.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(requested if ":" in requested else f"cuda:{gpu_id}")
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        if req == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("MPS requested but not available; falling back to CPU.")
            return torch.device("cpu")
        if req == "cpu":
            return torch.device("cpu")
        logger.warning("Unknown device %s; falling back to CPU.", requested)
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_coefficients(strategy: str,
                          n_models: int,
                          n_combinations: int,
                          random_seed: int,
                          fisher_soup: FisherSoupSTGNF) -> List[Sequence[float]]:
    """Generate coefficient combinations based on strategy."""
    if strategy == "uniform":
        return [[1.0 / n_models] * n_models]
    if strategy == "grid":
        if n_models == 2:
            return fisher_soup.create_pairwise_grid_coeffs(n_combinations)
        return fisher_soup.create_random_coeffs(n_models, n_combinations, random_seed)
    if strategy == "random":
        return fisher_soup.create_random_coeffs(n_models, n_combinations, random_seed)
    raise ValueError(f"Unknown strategy: {strategy}")


def _build_memory_safe_loader(orig_loader: DataLoader,
                              batch_size: int,
                              num_workers: int,
                              pin_memory: bool,
                              prefetch_factor: int,
                              persistent_workers: bool,
                              subset_size: int,
                              seed: int,
                              logger: logging.Logger) -> DataLoader:
    """Build a memory-safe DataLoader."""
    dataset = getattr(orig_loader, "dataset", None)
    if dataset is None:
        raise RuntimeError("Original loader has no 'dataset' attribute; cannot rebuild memory-safe loader.")
    
    total = len(dataset) if hasattr(dataset, "__len__") else None

    if subset_size and total and subset_size < total:
        rng = random.Random(seed)
        indices = rng.sample(range(total), subset_size)
        dataset = Subset(dataset, indices)
        logger.info("Fisher subset enabled: using %d / %d samples", subset_size, total)
    else:
        logger.info("Fisher subset disabled or not needed: using full dataset%s",
                    f" ({total} samples)" if total is not None else "")

    # Build a fresh DataLoader with conservative memory settings
    dl_kwargs = dict(batch_size=batch_size,
                     shuffle=False,
                     num_workers=num_workers,
                     pin_memory=pin_memory,
                     drop_last=False)
    
    # prefetch_factor / persistent_workers are only valid when num_workers > 0
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor
        dl_kwargs["persistent_workers"] = persistent_workers

    return DataLoader(dataset, **dl_kwargs)


def compute_fisher_for_checkpoint(checkpoint_path: Path,
                                  reference_args: argparse.Namespace,
                                  fisher_output: Path,
                                  device: torch.device,
                                  logger: logging.Logger,
                                  args: argparse.Namespace) -> Path:
    """Compute Fisher information for a single checkpoint."""
    if fisher_output.exists():
        logger.info("Using cached Fisher info at %s", fisher_output)
        return fisher_output

    logger.info("Computing Fisher information for %s", checkpoint_path)
    
    # Initialize model arguments
    ref_args, model_args = init_sub_args(reference_args)
    dataset, loader = get_dataset_and_loader(ref_args, trans_list=trans_list, only_test=False)
    model_args = init_model_params(ref_args, dataset)
    
    # Create model
    model = STG_NF(**model_args)
    
    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    # Handle pruned weights stored as weight_orig/weight_mask
    merged_state = {}
    for key, value in list(state_dict.items()):
        if key.endswith("weight_orig"):
            base_key = key[:-len("weight_orig")] + "weight"
            mask_key = key[:-len("weight_orig")] + "weight_mask"
            mask_tensor = state_dict.get(mask_key)
            if mask_tensor is None:
                mask_tensor = torch.ones_like(value)
            merged_state[base_key] = value * mask_tensor
        elif key.endswith("weight_mask"):
            continue
        elif key.endswith("actnorm.inited"):
            merged_state[key] = torch.ones_like(value) if torch.is_tensor(value) else 1
        else:
            merged_state[key] = value

    model.load_state_dict(merged_state, strict=False)
    if hasattr(model, "set_actnorm_init"):
        model.set_actnorm_init()
    model.to(device)

    # Initialize Fisher computation
    fisher_computer = FisherSTGNF(model=model, device=device, logger=logger, args=ref_args)

    # Prepare training loader for Fisher computation (memory-safe)
    train_loader = loader['train']
    safe_loader = _build_memory_safe_loader(
        orig_loader=train_loader,
        batch_size=args.fisher_batch_size,
        num_workers=args.fisher_num_workers,
        pin_memory=args.fisher_pin_memory,
        prefetch_factor=args.fisher_prefetch_factor,
        persistent_workers=args.fisher_persistent_workers,
        subset_size=max(0, int(args.fisher_subset)),
        seed=args.random_seed,
        logger=logger
    )

    # Perform Fisher computation
    fisher_info = fisher_computer.compute_fisher_for_model(safe_loader, max_batches=args.fisher_max_batches)
    
    # Save Fisher information
    fisher_output.parent.mkdir(parents=True, exist_ok=True)
    save_fisher_info(fisher_info, str(fisher_output), logger=logger)

    # Cleanup
    del fisher_info, fisher_computer, model, train_loader, safe_loader
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return fisher_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Fisher info and perform Fisher soup for STG-NF checkpoints stored in folders."
    )
    parser.add_argument("--folders", nargs="+", required=True,
                        help="One or more folders containing best checkpoints.")
    parser.add_argument("--output", required=True, help="Output path for merged model.")
    parser.add_argument("--device", default=None, help="Device override (cpu, cuda, cuda:0, mps).")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU index when using CUDA.")
    parser.add_argument("--fisher_dir", default=None,
                        help="Directory to cache Fisher information (defaults to output directory).")
    parser.add_argument("--strategy", choices=["grid", "random", "uniform"], default="grid")
    parser.add_argument("--n_combinations", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--fisher_floor", type=float, default=1e-6)
    parser.add_argument("--no_favor_target", action="store_true")
    parser.add_argument("--no_normalize_fishers", action="store_true")
    parser.add_argument("--evaluate", action="store_true")

    # Fisher computation memory-safe options
    parser.add_argument("--fisher_batch_size", type=int, default=1, 
                        help="Batch size for Fisher computation (memory-safe).")
    parser.add_argument("--fisher_num_workers", type=int, default=0, 
                        help="DataLoader workers for Fisher computation.")
    parser.add_argument("--fisher_pin_memory", action="store_true", 
                        help="Use pinned memory for Fisher DataLoader (default off).")
    parser.add_argument("--fisher_prefetch_factor", type=int, default=2, 
                        help="Prefetch factor when num_workers>0.")
    parser.add_argument("--fisher_persistent_workers", action="store_true", 
                        help="Use persistent workers when num_workers>0 (default off).")
    parser.add_argument("--fisher_subset", type=int, default=800, 
                        help="Use at most this many samples for Fisher (0 = use all).")
    parser.add_argument("--fisher_max_batches", type=int, default=100,
                        help="Maximum number of batches for Fisher computation.")

    # Evaluation memory-safe options
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--eval_pin_memory", action="store_true")
    parser.add_argument("--eval_prefetch_factor", type=int, default=2)
    parser.add_argument("--eval_persistent_workers", action="store_true")
    parser.add_argument("--eval_subset", type=int, default=0, 
                        help="Use at most this many samples for evaluation (0 = use all).")

    # Checkpoint patterns
    parser.add_argument("--checkpoint_pattern", default="checkpoint_best.pth.tar",
                        help="Checkpoint filename pattern to look for.")

    return parser.parse_args()


def main():
    # Set multiprocessing method to spawn
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    args = parse_args()
    logger = setup_logger()

    # Memory-safety environment defaults
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    # Set random seeds
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    device = resolve_device(args.device, args.gpu_id, logger)
    logger.info("Using device: %s", device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fisher_cache_dir = Path(args.fisher_dir) if args.fisher_dir else output_path.parent / "fisher_cache"
    fisher_cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths: List[Path] = []
    fisher_paths: List[Path] = []
    reference_args_path = None

    # Find checkpoints and prepare Fisher computation
    for folder in args.folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        ckpt_path = folder_path / args.checkpoint_pattern
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {args.checkpoint_pattern} not found in {folder}")

        # Use the first folder's args.json as reference
        if reference_args_path is None:
            reference_args_path = folder_path / "args.json"
            if not reference_args_path.exists():
                raise FileNotFoundError(f"No args.json found in {folder}")

        checkpoint_paths.append(ckpt_path)
        fisher_path = fisher_cache_dir / f"fisher_{folder_path.name}.pt"
        fisher_paths.append(fisher_path)

    # Load reference arguments
    reference_args = load_reference_args(reference_args_path)
    if args.device:
        reference_args.device = device.type
    
    # Compute Fisher information for all checkpoints
    logger.info("Computing Fisher information for all checkpoints...")
    computed_fisher_paths = []
    for ckpt_path, fisher_path in zip(checkpoint_paths, fisher_paths):
        computed_path = compute_fisher_for_checkpoint(
            ckpt_path, reference_args, fisher_path, device, logger, args
        )
        computed_fisher_paths.append(computed_path)

    # Prepare model arguments
    ref_args, model_args = init_sub_args(reference_args)
    dataset, loader = get_dataset_and_loader(ref_args, trans_list=trans_list, only_test=args.evaluate)
    model_args = init_model_params(ref_args, dataset)

    # Load models and fishers
    models, fishers, masks = load_models_and_fishers(
        checkpoint_paths=[str(p) for p in checkpoint_paths],
        fisher_paths=[str(p) for p in computed_fisher_paths],
        model_args=model_args,
        device=device,
        logger=logger
    )

    fisher_soup = FisherSoupSTGNF(device, logger, model_args=model_args)
    combined_mask = fisher_soup.combine_masks(masks)
    for model, mask in zip(models, masks):
        fisher_soup.apply_mask_to_model(model, mask)

    # Generate coefficient combinations
    coefficients_set = generate_coefficients(
        strategy=args.strategy,
        n_models=len(models),
        n_combinations=args.n_combinations,
        random_seed=args.random_seed,
        fisher_soup=fisher_soup
    )
    logger.info("Generated %d coefficient combinations using %s strategy",
                len(coefficients_set), args.strategy)

    # Prepare evaluation if requested
    test_loader = None
    dataset_test = None
    if args.evaluate:
        logger.info("Preparing dataset for evaluation...")
        test_loader = loader['test']
        dataset_test = dataset['test']
        
        test_loader = _build_memory_safe_loader(
            orig_loader=test_loader,
            batch_size=args.eval_batch_size,
            num_workers=args.eval_num_workers,
            pin_memory=args.eval_pin_memory,
            prefetch_factor=args.eval_prefetch_factor,
            persistent_workers=args.eval_persistent_workers,
            subset_size=max(0, int(args.eval_subset)),
            seed=args.random_seed,
            logger=logger
        )

    # Search for optimal coefficients or use default
    if test_loader is not None:
        results = fisher_soup.search_merging_coefficients(
            models=models,
            coefficients_set=coefficients_set,
            test_loader=test_loader,
            dataset_test=dataset_test,
            args=ref_args,
            fishers=fishers,
            fisher_floor=args.fisher_floor,
            favor_target_model=not args.no_favor_target,
            normalize_fishers=not args.no_normalize_fishers,
            combined_mask=combined_mask,
            print_results=True
        )
        best_result = max(results, key=lambda x: x.score["roc_auc"])
        best_coefficients = best_result.coefficients
        logger.info("Using best coefficients: %s (ROC AUC: %.4f)", best_coefficients, best_result.score["roc_auc"])
    else:
        best_coefficients = coefficients_set[0]
        best_result = None
        logger.info("Using coefficients without evaluation: %s", best_coefficients)

    # Create final merged model
    merged_models = fisher_soup.generate_merged_for_coeffs_set(
        models=models,
        coefficients_set=[best_coefficients],
        fishers=fishers,
        fisher_floor=args.fisher_floor,
        favor_target_model=not args.no_favor_target,
        normalize_fishers=not args.no_normalize_fishers,
        combined_mask=combined_mask
    )
    _, final_model = next(merged_models)
    fisher_soup.apply_mask_to_model(final_model, combined_mask)

    # Save merged model
    final_state_dict = final_model.state_dict()
    
    # Create soup payload similar to original fisher_soup.py
    soup_metadata = {
        "method": "fisher",
        "folders": [str(p) for p in args.folders],
        "checkpoints": [str(p) for p in checkpoint_paths],
        "fisher_paths": [str(p) for p in computed_fisher_paths],
        "coefficients": [float(c) for c in best_coefficients],
        "strategy": args.strategy,
        "n_combinations": args.n_combinations,
        "fisher_floor": args.fisher_floor,
        "favor_target_model": not args.no_favor_target,
        "normalize_fishers": not args.no_normalize_fishers,
        "random_seed": args.random_seed,
        "device": str(device),
        "mask_applied": combined_mask is not None,
    }
    
    if best_result is not None:
        soup_metadata["evaluation"] = {
            "auc": float(best_result.score["auc"]),
            "roc_auc": float(best_result.score["roc_auc"])
        }

    payload = {
        "state_dict": final_state_dict,
        "soup_metadata": soup_metadata,
    }
    
    torch.save(payload, str(output_path))
    logger.info("Saved merged checkpoint to %s", output_path)

    # Save combined mask if exists
    if combined_mask is not None:
        mask_path = output_path.with_suffix(output_path.suffix + ".mask")
        mask_cpu = {key: tensor.to("cpu") for key, tensor in combined_mask.items()}
        torch.save(mask_cpu, mask_path)
        logger.info("Saved combined mask to %s", mask_path)

    # Save metadata
    metadata_path = output_path.with_suffix(output_path.suffix + "_metadata.json")
    with open(metadata_path, "w") as handle:
        json.dump(soup_metadata, handle, indent=2, sort_keys=True)
    logger.info("Saved metadata to %s", metadata_path)

    # Final evaluation if requested
    if args.evaluate and test_loader is not None:
        final_eval = fisher_soup.evaluate_model(final_model, test_loader, dataset_test, ref_args)
        logger.info("Final merged model performance:")
        logger.info("  ROC AUC: %.4f", final_eval["roc_auc"])

    logger.info("Folder Fisher Soup for STG-NF completed successfully!")


if __name__ == "__main__":
    main()

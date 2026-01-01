#!/usr/bin/env python3
"""
Score-space ensemble for STG-NF (normality score = -NLL).

- Extract per-sample normality_scores from multiple checkpoints using the SAME test_loader order.
- Ensemble in score space:
  1) weighted mean (alphas)
  2) optional z-score normalization per model
  3) optional rank-average (percentile-style)

- Grid search over alphas to maximize roc_auc / pr_auc / f1.

Example:
python tools/soup/score_ensemble_stg_nf.py \
  --reference_args experiments/.../args.json \
  --checkpoints ckptA.pth.tar ckptB.pth.tar ckptC.pth.tar \
  --device cuda:0 --gpu_id 0 \
  --grid_search --grid_values 0.0 0.25 0.5 0.75 1.0 --grid_normalize \
  --select_by roc_auc \
  --normalize_scores zscore \
  --output_json results/ensemble/ens.metrics.json \
  --save_scores_dir results/ensemble/scores_cache
"""

import argparse
import itertools
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from models.STG_NF.model_pose import STG_NF
from utils.data_utils import trans_list
from utils.train_utils import init_model_params
from utils.scoring_utils import get_dataset_scores, smooth_scores
from utils.unlearning_utils import prepare_batch, reduce_conf_score


def _setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("score_ensemble")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def _load_reference_args(path: Path) -> argparse.Namespace:
    with open(path, "r") as f:
        payload = json.load(f)
    return argparse.Namespace(**payload)


def _load_args_from_checkpoint(path: Path) -> Optional[argparse.Namespace]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "args" in ckpt:
        ckpt_args = ckpt["args"]
        if isinstance(ckpt_args, argparse.Namespace):
            return ckpt_args
        if isinstance(ckpt_args, dict):
            return argparse.Namespace(**ckpt_args)
    return None


def _merge_with_defaults(loaded_args: Optional[argparse.Namespace]) -> argparse.Namespace:
    base_args = init_parser().parse_args([])
    if loaded_args is None:
        return base_args
    for k, v in vars(loaded_args).items():
        setattr(base_args, k, v)
    return base_args


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state: Dict[str, torch.Tensor] = {}
    for k, v in list(state_dict.items()):
        if k.endswith("weight_orig"):
            base_key = k[:-len("weight_orig")] + "weight"
            mask_key = k[:-len("weight_orig")] + "weight_mask"
            mask = state_dict.get(mask_key)
            if mask is None:
                mask = torch.ones_like(v)
            new_state[base_key] = v * mask
        elif k.endswith("weight_mask"):
            continue
        elif k.endswith("actnorm.inited"):
            new_state[k] = torch.ones_like(v) if torch.is_tensor(v) else 1
        else:
            new_state[k] = v
    return new_state


def _load_ckpt_state(path: str) -> Dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    elif isinstance(raw, dict):
        sd = raw
    else:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return _normalize_state_dict(sd)


def _sanitize_name(path: str) -> str:
    base = os.path.basename(path)
    for suf in [".pth.tar", ".pth", ".pt"]:
        if base.endswith(suf):
            base = base[: -len(suf)]
    return base.replace("/", "_")


@torch.no_grad()
def extract_normality_scores(
    model: STG_NF,
    test_loader,
    ref_args,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    scores: List[torch.Tensor] = []
    model_conf = bool(getattr(ref_args, "model_confidence", False))

    for batch in test_loader:
        x, conf_score, _ = prepare_batch(batch, device, model_conf)
        label = torch.ones(x.size(0), device=device)
        _, nll = model(x, label=label)
        if model_conf:
            nll = nll * reduce_conf_score(conf_score)
        scores.append((-1.0 * nll).detach().cpu())

    if not scores:
        return np.zeros((0,), dtype=np.float32)
    out = torch.cat(scores, dim=0).view(-1).numpy().astype(np.float32, copy=False)
    return out


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(x.mean()) if x.size else 0.0
    s = float(x.std()) if x.size else 1.0
    s = max(s, eps)
    return ((x - m) / s).astype(np.float32, copy=False)


def _rank01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32)
    return ranks


def _compute_metrics_from_normality_scores(
    normality_scores: np.ndarray,
    metadata,
    ref_args,
    f1_threshold: float,
) -> Dict[str, float]:
    gt_arr, scores_arr = get_dataset_scores(normality_scores, metadata, args=ref_args)
    scores_arr = smooth_scores(scores_arr)

    gt_np = np.concatenate(gt_arr) if len(gt_arr) else np.zeros((0,), dtype=np.int64)
    sc_np = np.concatenate(scores_arr) if len(scores_arr) else np.zeros((0,), dtype=np.float32)

    if sc_np.size:
        if np.isposinf(sc_np).any():
            sc_np[np.isposinf(sc_np)] = np.max(sc_np[~np.isposinf(sc_np)])
        if np.isneginf(sc_np).any():
            sc_np[np.isneginf(sc_np)] = np.min(sc_np[~np.isneginf(sc_np)])

    roc = float(roc_auc_score(gt_np, sc_np)) if gt_np.size else 0.0
    pr = float(average_precision_score(gt_np, sc_np)) if gt_np.size else 0.0
    f1 = float(f1_score(gt_np, sc_np >= float(f1_threshold))) if gt_np.size else 0.0
    return {"roc_auc": roc, "pr_auc": pr, "f1": f1}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_args", type=Path, default=None)
    ap.add_argument("--reference_ckpt", type=Path, default=None,
                    help="checkpoint path that contains args (state['args'])")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--checkpoints", nargs="+", required=True)

    ap.add_argument("--device", default=None, help="Override device (e.g., cuda:0).")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--log_level", default="INFO")

    ap.add_argument(
        "--normalize_scores",
        choices=["none", "zscore", "rank"],
        default="zscore",
        help="Per-model score normalization before ensembling.",
    )

    ap.add_argument("--f1_threshold", type=float, default=0.5)

    ap.add_argument("--grid_search", action="store_true")
    ap.add_argument("--grid_values", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--grid_normalize", action="store_true")
    ap.add_argument("--select_by", choices=["roc_auc", "pr_auc", "f1"], default="roc_auc")

    ap.add_argument("--save_scores_dir", type=Path, default=None,
                    help="If set, cache per-ckpt normality_scores as .npy.")
    ap.add_argument("--force_recompute", action="store_true",
                    help="Ignore cached .npy and recompute scores.")

    ap.add_argument("--output_json", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    logger = _setup_logger(args.log_level)

    if args.reference_args is None and args.reference_ckpt is None:
        raise ValueError("Provide --reference_args or --reference_ckpt")

    loaded_args = None
    reference_source = None
    if args.reference_args is not None:
        loaded_args = _load_reference_args(args.reference_args)
        reference_source = str(args.reference_args)
    elif args.reference_ckpt is not None:
        loaded_args = _load_args_from_checkpoint(args.reference_ckpt)
        reference_source = str(args.reference_ckpt)
        if loaded_args is None:
            logger.warning("checkpoint has no args: %s. Using defaults.", args.reference_ckpt)

    ref_args = _merge_with_defaults(loaded_args)
    if args.dataset is not None:
        ref_args.dataset = args.dataset
    if args.data_dir is not None:
        ref_args.data_dir = args.data_dir

    if args.device:
        ref_args.device = args.device
    dev_str = str(getattr(ref_args, "device", "cpu"))
    if dev_str.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(dev_str if ":" in dev_str else f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    logger.info("Using device=%s", device)

    ref_args, model_args = init_sub_args(ref_args)

    dataset, loader = get_dataset_and_loader(ref_args, trans_list=trans_list, only_test=True)
    model_args = init_model_params(ref_args, dataset)

    test_loader = loader["test"]
    test_metadata = dataset["test"].metadata

    scores_list: List[np.ndarray] = []
    ckpt_names: List[str] = []

    if args.save_scores_dir is not None:
        args.save_scores_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_path in args.checkpoints:
        ckpt_name = _sanitize_name(ckpt_path)
        ckpt_names.append(ckpt_name)

        cache_path = None
        if args.save_scores_dir is not None:
            cache_path = args.save_scores_dir / f"{ckpt_name}.normality.npy"

        if cache_path is not None and cache_path.exists() and not args.force_recompute:
            logger.info("Loading cached scores: %s", cache_path)
            sc = np.load(cache_path)
            scores_list.append(sc.astype(np.float32, copy=False))
            continue

        logger.info("Loading checkpoint: %s", ckpt_path)
        state = _load_ckpt_state(ckpt_path)

        model = STG_NF(**model_args)
        model.load_state_dict(state, strict=False)
        if hasattr(model, "set_actnorm_init"):
            model.set_actnorm_init()
        model.to(device)

        sc = extract_normality_scores(model, test_loader, ref_args, device)
        scores_list.append(sc)

        if cache_path is not None:
            np.save(cache_path, sc)
            logger.info("Saved cached scores: %s", cache_path)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not scores_list:
        raise RuntimeError("No scores extracted.")
    n0 = scores_list[0].shape[0]
    for i, sc in enumerate(scores_list):
        if sc.shape[0] != n0:
            raise RuntimeError(f"Score length mismatch: idx={i} got {sc.shape[0]} expected {n0}")

    normed: List[np.ndarray] = []
    for sc in scores_list:
        if args.normalize_scores == "none":
            normed.append(sc.astype(np.float32, copy=False))
        elif args.normalize_scores == "zscore":
            normed.append(_zscore(sc))
        elif args.normalize_scores == "rank":
            normed.append(_rank01(sc))
        else:
            raise ValueError("Unknown normalize_scores")

    k_models = len(normed)
    logger.info("Ensembling K=%d models, N=%d samples, norm=%s", k_models, n0, args.normalize_scores)

    def eval_alphas(alphas: List[float]) -> Dict[str, float]:
        a = np.asarray(alphas, dtype=np.float32)
        if a.size != k_models:
            raise ValueError("alpha size mismatch")
        ens = np.zeros_like(normed[0], dtype=np.float32)
        for w, sc in zip(a, normed):
            ens += float(w) * sc
        return _compute_metrics_from_normality_scores(ens, test_metadata, ref_args, args.f1_threshold)

    best = None
    best_alphas = None
    best_metrics = None
    logs = []

    if not args.grid_search:
        alphas = [1.0 / k_models] * k_models
        metrics = eval_alphas(alphas)
        best_alphas, best_metrics = alphas, metrics
        best = float(metrics[args.select_by])
        logs.append({"alphas": alphas, "metrics": metrics})
        logger.info("Equal-weight metrics: %s", metrics)
    else:
        grid_values = list(args.grid_values)
        logger.info("Grid search values=%s normalize=%s select_by=%s", grid_values, args.grid_normalize, args.select_by)

        for combo in itertools.product(grid_values, repeat=k_models):
            combo = list(combo)
            if args.grid_normalize:
                s = sum(combo)
                if s == 0:
                    continue
                alphas = [c / s for c in combo]
            else:
                alphas = combo
                if sum(alphas) == 0:
                    continue

            metrics = eval_alphas(alphas)
            logs.append({"alphas": alphas, "metrics": metrics})

            score = metrics.get(args.select_by)
            if score is None or (isinstance(score, float) and not np.isfinite(score)):
                continue
            score = float(score)
            if best is None or score > best:
                best = score
                best_alphas = alphas
                best_metrics = metrics
                logger.info("New best %s=%.4f with alphas=%s metrics=%s", args.select_by, best, best_alphas, best_metrics)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "reference_source": reference_source,
        "checkpoints": args.checkpoints,
        "ckpt_names": ckpt_names,
        "normalize_scores": args.normalize_scores,
        "f1_threshold": float(args.f1_threshold),
        "grid_search": bool(args.grid_search),
        "grid_values": list(args.grid_values),
        "grid_normalize": bool(args.grid_normalize),
        "select_by": args.select_by,
        "best_alphas": best_alphas,
        "best_metrics": best_metrics,
        "logs": logs,
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved ensemble result JSON: %s", args.output_json)


if __name__ == "__main__":
    main()

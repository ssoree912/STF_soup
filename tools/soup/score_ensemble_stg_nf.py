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
from torch.utils.data import DataLoader, Subset
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


def _extract_normality_scores_indexed(
    model: STG_NF,
    dataset,
    indices: List[int],
    ref_args,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    if not indices:
        return np.zeros((0,), dtype=np.float32)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model.eval()
    scores: List[torch.Tensor] = []
    model_conf = bool(getattr(ref_args, "model_confidence", False))
    for batch in loader:
        x, conf_score, _ = prepare_batch(batch, device, model_conf)
        label = torch.ones(x.size(0), device=device)
        _, nll = model(x, label=label)
        if model_conf:
            nll = nll * reduce_conf_score(conf_score)
        scores.append((-1.0 * nll).detach().cpu())
    if not scores:
        return np.zeros((0,), dtype=np.float32)
    return torch.cat(scores, dim=0).view(-1).numpy().astype(np.float32, copy=False)


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(x.mean()) if x.size else 0.0
    s = float(x.std()) if x.size else 1.0
    s = max(s, eps)
    return ((x - m) / s).astype(np.float32, copy=False)

def _zscore_with_stats(x: np.ndarray, mean: float, std: float, eps: float = 1e-8) -> np.ndarray:
    s = float(std)
    s = max(s, eps)
    return ((x - float(mean)) / s).astype(np.float32, copy=False)


def _rank01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32)
    return ranks


def _get_normal_indices(dataset, normal_label: int = 1) -> List[int]:
    if hasattr(dataset, "labels"):
        return [i for i, y in enumerate(dataset.labels) if int(y) == int(normal_label)]
    return list(range(len(dataset)))


def _ecdf_p_left(cal_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = cal_sorted.size
    if n == 0:
        return np.full_like(x, 0.5, dtype=np.float32)
    idx = np.searchsorted(cal_sorted, x, side="right")
    p = (idx + 1.0) / (n + 2.0)
    return p.astype(np.float32, copy=False)


def _ecdf_p_right(cal_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = cal_sorted.size
    if n == 0:
        return np.full_like(x, 0.5, dtype=np.float32)
    idx = np.searchsorted(cal_sorted, x, side="left")
    cnt_ge = n - idx
    p = (cnt_ge + 1.0) / (n + 2.0)
    return p.astype(np.float32, copy=False)


def _fisher_combine(p_stack: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p_stack, eps, 1.0)
    return (-2.0 * np.sum(np.log(p), axis=0)).astype(np.float32)

def _fisher_combine_weighted(p_stack: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p_stack, eps, 1.0)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    w = w / (float(w.sum()) + 1e-12)
    return (-2.0 * np.sum(w * np.log(p), axis=0)).astype(np.float32)


def _stouffer_combine(p_stack: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    from scipy.stats import norm
    p = np.clip(p_stack, eps, 1.0 - eps)
    z = norm.ppf(1.0 - p)
    return (np.sum(z, axis=0) / np.sqrt(p.shape[0])).astype(np.float32)

def _stouffer_combine_weighted(p_stack: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    from scipy.stats import norm
    p = np.clip(p_stack, eps, 1.0 - eps)
    z = norm.ppf(1.0 - p)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    w = w / (float(w.sum()) + 1e-12)
    denom = float(np.sqrt(np.sum(w.squeeze() ** 2)) + 1e-12)
    return (np.sum(w * z, axis=0) / denom).astype(np.float32)


def _min_combine(p_stack: np.ndarray) -> np.ndarray:
    return (1.0 - np.min(p_stack, axis=0)).astype(np.float32)

def _best_f1_quantile_threshold(sc: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    if sc.size == 0 or gt.size == 0 or len(np.unique(gt)) < 2:
        return {"best_f1": float("nan"), "best_thr": float("nan")}
    qs = np.linspace(0.01, 0.99, 99, dtype=np.float32)
    thrs = np.quantile(sc, qs)
    best_f1 = -1.0
    best_thr = float(thrs[0])
    for thr in thrs:
        pred = (sc >= float(thr))
        f1v = float(f1_score(gt, pred, zero_division=0))
        if f1v > best_f1:
            best_f1 = f1v
            best_thr = float(thr)
    return {"best_f1": float(best_f1), "best_thr": float(best_thr)}


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
    best = _best_f1_quantile_threshold(sc_np, gt_np) if gt_np.size else {"best_f1": float("nan"), "best_thr": float("nan")}
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "f1": f1,
        "best_f1": float(best["best_f1"]),
        "best_thr": float(best["best_thr"]),
    }

def _load_or_compute_scores_for_indices(
    ckpt_path: str,
    ckpt_name: str,
    split_tag: str,
    dataset_split,
    indices: List[int],
    model_args,
    ref_args,
    device: torch.device,
    save_scores_dir: Optional[Path],
    force_recompute: bool,
) -> np.ndarray:
    cache_path = None
    if save_scores_dir is not None:
        cache_path = save_scores_dir / f"{ckpt_name}.{split_tag}.npy"
        if cache_path.exists() and not force_recompute:
            return np.load(cache_path).astype(np.float32, copy=False)

    state = _load_ckpt_state(ckpt_path)
    model = STG_NF(**model_args)
    model.load_state_dict(state, strict=False)
    if hasattr(model, "set_actnorm_init"):
        model.set_actnorm_init()
    model.to(device)

    scores = _extract_normality_scores_indexed(
        model=model,
        dataset=dataset_split,
        indices=indices,
        ref_args=ref_args,
        device=device,
        batch_size=getattr(ref_args, "batch_size", 256),
        num_workers=getattr(ref_args, "num_workers", 4),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if cache_path is not None:
        np.save(cache_path, scores)
    return scores.astype(np.float32, copy=False)


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

    ap.add_argument("--pvalue_ensemble", action="store_true",
                    help="Use normal-only ECDF to compute p-values and combine them.")
    ap.add_argument("--pvalue_tail", choices=["left", "right"], default="left",
                    help="left: p = P(cal <= s). right: p = P(cal >= s)")
    ap.add_argument("--pvalue_cal_split", choices=["train", "test"], default="train",
                    help="Calibration split for ECDF (normal-only).")
    ap.add_argument("--pvalue_normal_label", type=int, default=1)
    ap.add_argument("--pvalue_weights", nargs="+", type=float, default=None,
                    help="Optional weights for weighted p-value combine.")

    ap.add_argument("--zscore_cal_split", choices=["none", "train", "test"], default="train",
                    help="Where to estimate (mean,std) for zscore calibration.")
    ap.add_argument("--zscore_normal_label", type=int, default=1,
                    help="Normal label used for zscore calibration.")

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

    need_train_for_p = bool(args.pvalue_ensemble and args.pvalue_cal_split == "train")
    need_train_for_z = bool(args.normalize_scores == "zscore" and args.zscore_cal_split == "train")
    only_test = not (need_train_for_p or need_train_for_z)
    dataset, loader = get_dataset_and_loader(ref_args, trans_list=trans_list, only_test=only_test)
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
    z_stats = None
    if args.normalize_scores == "zscore" and args.zscore_cal_split != "none":
        split = args.zscore_cal_split
        if split not in dataset:
            raise ValueError(f"zscore_cal_split={split} not found in dataset.")
        ds_cal = dataset[split]
        cal_idx = _get_normal_indices(ds_cal, normal_label=args.zscore_normal_label)
        if not cal_idx:
            logger.warning("No normal samples for zscore calibration; using full split.")
            cal_idx = list(range(len(ds_cal)))

        z_stats = []
        for ckpt_path, ckpt_name in zip(args.checkpoints, ckpt_names):
            cal_scores = _load_or_compute_scores_for_indices(
                ckpt_path=ckpt_path,
                ckpt_name=ckpt_name,
                split_tag=f"zcal_{split}",
                dataset_split=ds_cal,
                indices=cal_idx,
                model_args=model_args,
                ref_args=ref_args,
                device=device,
                save_scores_dir=args.save_scores_dir,
                force_recompute=args.force_recompute,
            )
            mu = float(np.mean(cal_scores)) if cal_scores.size else 0.0
            sd = float(np.std(cal_scores)) if cal_scores.size else 1.0
            z_stats.append({"mean": mu, "std": sd, "n": int(cal_scores.size)})

        for sc, st in zip(scores_list, z_stats):
            normed.append(_zscore_with_stats(sc, st["mean"], st["std"]))
    else:
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

    pvalue_ensembles = None
    if args.pvalue_ensemble:
        cal_split = args.pvalue_cal_split
        if cal_split == "test":
            logger.warning("pvalue_cal_split=test uses evaluation split for calibration.")
        if cal_split not in dataset:
            raise ValueError(f"pvalue_cal_split={cal_split} not found in dataset.")
        ds_cal = dataset[cal_split]
        cal_idx = _get_normal_indices(ds_cal, normal_label=args.pvalue_normal_label)
        if not cal_idx:
            logger.warning("No normal samples found for calibration; using full split.")
            cal_idx = list(range(len(ds_cal)))

        cal_scores_list = []
        for ckpt_path, ckpt_name in zip(args.checkpoints, ckpt_names):
            cal_scores = _load_or_compute_scores_for_indices(
                ckpt_path=ckpt_path,
                ckpt_name=ckpt_name,
                split_tag=f"pcal_{cal_split}",
                dataset_split=ds_cal,
                indices=cal_idx,
                model_args=model_args,
                ref_args=ref_args,
                device=device,
                save_scores_dir=args.save_scores_dir,
                force_recompute=args.force_recompute,
            )
            cal_scores_list.append(cal_scores.astype(np.float32, copy=False))

        p_list = []
        for cal_scores, test_scores in zip(cal_scores_list, scores_list):
            cal_sorted = np.sort(cal_scores.astype(np.float32, copy=False))
            if args.pvalue_tail == "left":
                p = _ecdf_p_left(cal_sorted, test_scores)
            else:
                p = _ecdf_p_right(cal_sorted, test_scores)
            p_list.append(p)
        p_stack = np.stack(p_list, axis=0)

        anom_fisher = _fisher_combine(p_stack)
        anom_stouffer = _stouffer_combine(p_stack)
        anom_min = _min_combine(p_stack)

        if args.pvalue_weights is not None:
            w = np.asarray(args.pvalue_weights, dtype=np.float32)
            if w.size != k_models:
                raise ValueError(f"--pvalue_weights must have length K={k_models}")
        elif best_alphas is not None:
            w = np.asarray(best_alphas, dtype=np.float32)
        else:
            w = np.ones((k_models,), dtype=np.float32) / float(k_models)

        anom_fisher_w = _fisher_combine_weighted(p_stack, w)
        anom_stouffer_w = _stouffer_combine_weighted(p_stack, w)

        pvalue_ensembles = {
            "pvalue_fisher": _compute_metrics_from_normality_scores(-anom_fisher, test_metadata, ref_args, args.f1_threshold),
            "pvalue_stouffer": _compute_metrics_from_normality_scores(-anom_stouffer, test_metadata, ref_args, args.f1_threshold),
            "pvalue_min": _compute_metrics_from_normality_scores(-anom_min, test_metadata, ref_args, args.f1_threshold),
            "pvalue_fisher_weighted": _compute_metrics_from_normality_scores(-anom_fisher_w, test_metadata, ref_args, args.f1_threshold),
            "pvalue_stouffer_weighted": _compute_metrics_from_normality_scores(-anom_stouffer_w, test_metadata, ref_args, args.f1_threshold),
            "pvalue_weights_used": w.tolist(),
        }

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
        "pvalue_ensemble": bool(args.pvalue_ensemble),
        "pvalue_tail": args.pvalue_tail,
        "pvalue_cal_split": args.pvalue_cal_split,
        "pvalue_normal_label": int(args.pvalue_normal_label),
        "pvalue_weights": args.pvalue_weights,
        "zscore_cal_split": args.zscore_cal_split,
        "zscore_normal_label": int(args.zscore_normal_label),
        "zscore_calibration_stats": z_stats,
        "best_alphas": best_alphas,
        "best_metrics": best_metrics,
        "logs": logs,
        "pvalue_ensembles": pvalue_ensembles,
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved ensemble result JSON: %s", args.output_json)


if __name__ == "__main__":
    main()

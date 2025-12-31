#!/usr/bin/env python3
"""
UGM soup with df1 forgetting constraint (delta_df >= threshold).
"""
import argparse
import copy
import itertools
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from args import init_sub_args
from dataset import get_dataset_and_loader
from models.STG_NF.model_pose import STG_NF
from utils.data_utils import trans_list
from utils.train_utils import init_model_params
from utils.scoring_utils import get_dataset_scores, smooth_scores
from utils.unlearning_utils import (
    build_indexed_loader,
    prepare_batch,
    reduce_conf_score,
    select_df1_tail,
)


def _setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("ugm_soup")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def _load_reference_args(path: Path) -> argparse.Namespace:
    with open(path, "r") as f:
        payload = json.load(f)
    return argparse.Namespace(**payload)


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = {}
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
            if torch.is_tensor(v):
                new_state[k] = torch.ones_like(v)
            else:
                new_state[k] = 1
        else:
            new_state[k] = v
    return new_state


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return _normalize_state_dict(sd)


def _split_fisher_payload(raw):
    if isinstance(raw, dict) and "fisher" in raw:
        return raw["fisher"], raw.get("metadata")
    return raw, None


def _normalize_fisher_to_state_dict(fisher_obj, model_state: Dict[str, torch.Tensor], eps: float):
    if isinstance(fisher_obj, dict) and set(fisher_obj.keys()) == set(model_state.keys()):
        return fisher_obj

    if isinstance(fisher_obj, dict) and "fisher_list" not in fisher_obj and not all(
        k.startswith("param_") for k in fisher_obj.keys()
    ):
        out = {}
        for k, t in model_state.items():
            if k in fisher_obj:
                out[k] = fisher_obj[k]
            else:
                out[k] = torch.full_like(t, eps)
        return out

    raise ValueError("Unsupported Fisher format for this script.")


def _validate_keys(models: List[Dict[str, torch.Tensor]], fishers: List[Dict[str, torch.Tensor]]):
    mk = set(models[0].keys())
    fk = set(fishers[0].keys())
    for i in range(1, len(models)):
        if set(models[i].keys()) != mk:
            raise ValueError(f"Model keys differ at {i}")
    for i in range(1, len(fishers)):
        if set(fishers[i].keys()) != fk:
            raise ValueError(f"Fisher keys differ at {i}")
    if mk != fk:
        raise ValueError("Model keys and Fisher keys must match.")


def ugm_merge_state_dicts(
    models: List[Dict[str, torch.Tensor]],
    fishers: List[Dict[str, torch.Tensor]],
    alphas: torch.Tensor,
    ref_idx: int = 0,
    eps: float = 1e-8,
    use_ref_fisher: bool = True,
) -> Dict[str, torch.Tensor]:
    if len(models) != len(fishers):
        raise ValueError("models/fishers mismatch")
    if not models:
        raise ValueError("no models")
    if ref_idx < 0 or ref_idx >= len(models):
        raise ValueError("bad ref_idx")

    _validate_keys(models, fishers)

    k_models = len(models)
    alphas = torch.as_tensor(alphas, dtype=torch.float32)
    if alphas.numel() != k_models:
        raise ValueError("alphas length mismatch")

    merged = copy.deepcopy(models[ref_idx])
    ref_sd = models[ref_idx]

    if use_ref_fisher:
        h0_sd = fishers[ref_idx]
    else:
        h0_sd = {k: torch.full_like(v, eps) for k, v in ref_sd.items() if torch.is_tensor(v)}

    bar_h = {}
    for k, ref_t in ref_sd.items():
        if not torch.is_tensor(ref_t) or not ref_t.is_floating_point():
            continue
        fish = torch.stack([f[k] for f in fishers], dim=0)
        a = alphas.view(k_models, *([1] * (fish.dim() - 1)))
        bar = h0_sd[k].clone() + torch.sum(a * fish, dim=0)
        bar_h[k] = bar.clamp_min(eps)

    for k, ref_t in ref_sd.items():
        if not torch.is_tensor(ref_t) or not ref_t.is_floating_point():
            merged[k] = ref_t
            continue
        fish = torch.stack([f[k] for f in fishers], dim=0)
        params = torch.stack([m[k] for m in models], dim=0)
        a = alphas.view(k_models, *([1] * (params.dim() - 1)))

        h0 = h0_sd[k]
        h0_plus_ht = h0.unsqueeze(0) + fish
        denom = bar_h[k]
        w = h0_plus_ht / denom.unsqueeze(0)

        inc = params - ref_t.unsqueeze(0)
        delta = torch.sum(a * w * inc, dim=0)
        merged[k] = ref_t + delta

    return merged


@torch.no_grad()
def _eval_normality_scores(model, test_loader, ref_args, device):
    model.eval()
    scores = []
    model_conf = bool(getattr(ref_args, "model_confidence", False))
    for batch in test_loader:
        x, score, _ = prepare_batch(batch, device, model_conf)
        label = torch.ones(x.size(0), device=device)
        _, nll = model(x, label=label)
        if model_conf:
            nll = nll * reduce_conf_score(score)
        scores.append((-1 * nll).detach().cpu())
    if not scores:
        return None
    return torch.cat(scores, dim=0).numpy().squeeze().copy(order="C")


def _eval_test_metrics(model, test_loader, metadata, ref_args, device, f1_threshold: float):
    normality_scores = _eval_normality_scores(model, test_loader, ref_args, device)
    if normality_scores is None:
        return {"roc_auc": 0.0, "pr_auc": 0.0, "f1": 0.0}

    gt_arr, scores_arr = get_dataset_scores(normality_scores, metadata, args=ref_args)
    scores_arr = smooth_scores(scores_arr)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    if scores_np.size:
        scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
        scores_np[scores_np == -np.inf] = scores_np[scores_np != -np.inf].min()

    roc = float(roc_auc_score(gt_np, scores_np)) if gt_np.size else 0.0
    pr = float(average_precision_score(gt_np, scores_np)) if gt_np.size else 0.0
    f1 = float(f1_score(gt_np, scores_np >= f1_threshold)) if gt_np.size else 0.0
    return {"roc_auc": roc, "pr_auc": pr, "f1": f1}


@torch.no_grad()
def eval_df1_delta_df(
    model: STG_NF,
    train_dataset,
    df1_sids: List[int],
    cache_train: dict,
    ref_args,
    batch_size: int,
    num_workers: int,
    sample_size: int,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    sids = list(df1_sids)
    if sample_size > 0 and len(sids) > sample_size:
        sids = rng.sample(sids, sample_size)

    base_mean = float(np.mean([cache_train["sB"][sid] for sid in sids])) if sids else 0.0

    loader = build_indexed_loader(
        train_dataset, sids, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    nlls = []
    model.eval()
    device = next(model.parameters()).device
    model_conf = bool(getattr(ref_args, "model_confidence", False))
    for batch in loader:
        x, score, label = prepare_batch(batch, device, model_conf)
        _, nll = model(x, label=label)
        if model_conf:
            nll = nll * reduce_conf_score(score)
        nlls.append(nll.detach().cpu())

    cur_mean = float(torch.cat(nlls).mean().item()) if nlls else 0.0
    delta_df = cur_mean - base_mean
    delta_rel = delta_df / (abs(base_mean) + 1e-12)
    return {
        "df1_base_mean": base_mean,
        "df1_cur_mean": cur_mean,
        "df1_delta_df": float(delta_df),
        "df1_delta_rel": float(delta_rel),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_args", type=Path, required=True)
    ap.add_argument("--cache_train_path", type=Path, required=True)
    ap.add_argument("--alpha_df1", type=float, default=0.01)

    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--fishers", nargs="+", required=True)

    ap.add_argument("--ref_idx", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--no_ref_fisher", action="store_true")

    ap.add_argument("--grid_search", action="store_true")
    ap.add_argument("--grid_values", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--grid_normalize", action="store_true")
    ap.add_argument("--select_by", choices=["roc_auc", "pr_auc", "f1"], default="roc_auc")

    ap.add_argument("--df1_min_delta", type=float, default=0.3)
    ap.add_argument("--df1_sample_size", type=int, default=512)
    ap.add_argument("--df1_eval_batch_size", type=int, default=256)
    ap.add_argument("--df1_seed", type=int, default=0)

    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--f1_threshold", type=float, default=0.5)

    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--metrics_json", type=Path, default=None)
    ap.add_argument("--log_level", default="INFO")
    return ap.parse_args()


def main():
    args = parse_args()
    logger = _setup_logger(args.log_level)

    if len(args.checkpoints) != len(args.fishers):
        raise ValueError("checkpoints and fishers must align 1:1")

    ref_args = _load_reference_args(args.reference_args)
    if args.device:
        ref_args.device = args.device
    ref_args, model_args = init_sub_args(ref_args)

    dataset, loader = get_dataset_and_loader(ref_args, trans_list=trans_list, only_test=False)
    model_args = init_model_params(ref_args, dataset)
    train_dataset = dataset["train"]
    test_loader = loader["test"]
    test_metadata = dataset["test"].metadata

    dev_str = str(ref_args.device)
    if dev_str.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(dev_str if ":" in dev_str else f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    logger.info("Using device=%s", device)

    with open(args.cache_train_path, "rb") as f:
        cache_train = pickle.load(f)
    df1_sids = sorted(select_df1_tail(cache_train["sB"], alpha=args.alpha_df1))
    logger.info("DF1 size=%d (alpha_df1=%.4g)", len(df1_sids), args.alpha_df1)
    if df1_sids:
        all_scores = np.array(list(cache_train["sB"].values()), dtype=np.float32)
        df1_scores = np.array([cache_train["sB"][sid] for sid in df1_sids], dtype=np.float32)
        if all_scores.size and df1_scores.size:
            df1_mean = float(df1_scores.mean())
            df1_min = float(df1_scores.min())
            df1_max = float(df1_scores.max())
            all_mean = float(all_scores.mean())
            all_min = float(all_scores.min())
            all_max = float(all_scores.max())
            df1_mean_pct = float(np.mean(all_scores <= df1_mean))
            logger.info(
                "DF1 sB stats: mean=%.6g min=%.6g max=%.6g | overall mean=%.6g min=%.6g max=%.6g | "
                "df1_mean_percentile=%.3f",
                df1_mean, df1_min, df1_max,
                all_mean, all_min, all_max,
                df1_mean_pct,
            )

    models = [_load_state_dict(p) for p in args.checkpoints]
    raw_fishers = [torch.load(p, map_location="cpu") for p in args.fishers]
    fishers = []
    fisher_metas = []
    for sd, raw in zip(models, raw_fishers):
        fobj, meta = _split_fisher_payload(raw)
        fishers.append(_normalize_fisher_to_state_dict(fobj, sd, eps=args.eps))
        fisher_metas.append(meta)

    use_ref_fisher = not args.no_ref_fisher

    def evaluate_merged(state_dict: Dict[str, torch.Tensor]):
        model = STG_NF(**model_args)
        model.load_state_dict(state_dict, strict=False)
        if hasattr(model, "set_actnorm_init"):
            model.set_actnorm_init()
        model.to(device)

        df1_stats = eval_df1_delta_df(
            model=model,
            train_dataset=train_dataset,
            df1_sids=df1_sids,
            cache_train=cache_train,
            ref_args=ref_args,
            batch_size=args.df1_eval_batch_size,
            num_workers=getattr(ref_args, "num_workers", 4),
            sample_size=args.df1_sample_size,
            seed=args.df1_seed,
        )

        return df1_stats, model

    if not args.grid_search:
        raise ValueError("Use --grid_search for now (df1 constraint is for combo selection).")

    k_models = len(args.checkpoints)
    grid_values = list(args.grid_values)

    best = None
    best_state = None
    best_alphas = None
    logs = []

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

        merged = ugm_merge_state_dicts(
            models=models,
            fishers=fishers,
            alphas=torch.tensor(alphas, dtype=torch.float32),
            ref_idx=args.ref_idx,
            eps=args.eps,
            use_ref_fisher=use_ref_fisher,
        )

        df1_stats, model = evaluate_merged(merged)
        ok = df1_stats["df1_delta_df"] >= float(args.df1_min_delta)
        metrics = dict(df1_stats)

        if ok:
            test_metrics = _eval_test_metrics(
                model,
                test_loader,
                test_metadata,
                ref_args,
                device,
                f1_threshold=float(args.f1_threshold),
            )
            metrics.update(test_metrics)
        else:
            metrics.update({"roc_auc": None, "pr_auc": None, "f1": None})

        logs.append({"alphas": alphas, "ok_df1": ok, "metrics": metrics})

        if ok:
            logger.info(
                "alphas=%s | ok_df1=%s | df1_delta=%.4f | roc=%.4f pr=%.4f f1=%.4f",
                [round(a, 3) for a in alphas],
                ok,
                metrics["df1_delta_df"],
                metrics["roc_auc"],
                metrics["pr_auc"],
                metrics["f1"],
            )
        else:
            logger.info(
                "alphas=%s | ok_df1=%s | df1_delta=%.4f | test=skipped",
                [round(a, 3) for a in alphas],
                ok,
                metrics["df1_delta_df"],
            )

        if not ok:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        key = args.select_by
        score = metrics.get(key)
        if score is None or (isinstance(score, float) and not np.isfinite(score)):
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        score = float(score)
        if best is None or score > best:
            best = score
            best_state = merged
            best_alphas = alphas
            logger.info("New best %s=%.4f with alphas=%s", key, score, best_alphas)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_json or args.output.with_suffix(".metrics.json")

    if best_state is None:
        logger.warning("No combination satisfied df1 constraint (min_delta=%.4f).", args.df1_min_delta)
        with open(metrics_path, "w") as f:
            json.dump({"best": None, "logs": logs, "df1_min_delta": args.df1_min_delta}, f, indent=2)
        return

    torch.save(best_state, args.output)

    best_metrics = None
    for row in logs:
        if row["alphas"] == best_alphas:
            best_metrics = row["metrics"]
            break

    payload = {
        "soup_path": str(args.output),
        "best_alphas": best_alphas,
        "select_by": args.select_by,
        "df1_min_delta": args.df1_min_delta,
        "f1_threshold": args.f1_threshold,
        "best_metrics": best_metrics,
        "logs": logs,
        "fisher_meta": fisher_metas,
        "checkpoints": args.checkpoints,
        "fishers": args.fishers,
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved best merged ckpt to %s", args.output)
    logger.info("Saved metrics/logs to %s", metrics_path)


if __name__ == "__main__":
    main()

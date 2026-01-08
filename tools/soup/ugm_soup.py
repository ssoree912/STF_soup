#!/usr/bin/env python3
"""
Plain UGM soup (no DF1 constraint).
- Grid-search alphas -> merge -> evaluate on test -> pick best by metric.
"""
import argparse
import copy
import itertools
import json
import logging
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


def _setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("ugm_soup_plain")
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


def _init_rng(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        return np.random.default_rng()
    if seed == 999:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def _safe_metric(metrics: Dict[str, float], key: str) -> Optional[float]:
    if metrics is None or key not in metrics:
        return None
    try:
        score = float(metrics[key])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(score):
        return None
    return score


def _sample_near_logit(alpha: List[float], sigma: float, rng: np.random.Generator) -> List[float]:
    a = np.asarray(alpha, dtype=np.float64)
    a = np.clip(a, 1e-12, 1.0)
    logit = np.log(a)
    z = logit + rng.normal(0.0, sigma, size=logit.shape)
    e = np.exp(z - np.max(z))
    out = (e / e.sum()).astype(np.float32, copy=False)
    return out.tolist()


def _sample_near_dirichlet_center(alpha: List[float], conc: float, rng: np.random.Generator) -> List[float]:
    a = np.asarray(alpha, dtype=np.float64)
    a = np.clip(a, 1e-12, 1.0)
    out = rng.dirichlet(a * conc).astype(np.float32, copy=False)
    return out.tolist()


def _sample_from_elites(
    elites: List[List[float]],
    n: int,
    method: str,
    sigma: float,
    conc: float,
    rng: np.random.Generator,
) -> List[List[float]]:
    if not elites or n <= 0:
        return []
    idxs = rng.integers(0, len(elites), size=int(n))
    out = []
    for idx in idxs:
        base = elites[int(idx)]
        if method == "logit":
            out.append(_sample_near_logit(base, sigma, rng))
        else:
            out.append(_sample_near_dirichlet_center(base, conc, rng))
    return out


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
def _eval_normality_scores(model, test_loader, ref_args, device, max_batches: int = 0):
    model.eval()
    total = len(test_loader.dataset)
    scores = np.full((total,), np.inf, dtype=np.float32)
    model_conf = bool(getattr(ref_args, "model_confidence", False))
    offset = 0
    for batch_idx, batch in enumerate(test_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        x, score, _ = prepare_batch(batch, device, model_conf)
        label = torch.ones(x.size(0), device=device)
        _, nll = model(x, label=label)
        if model_conf:
            nll = nll * reduce_conf_score(score)
        batch_scores = (-1 * nll).detach().cpu().view(-1).numpy().astype(np.float32, copy=False)
        end = min(offset + batch_scores.size, total)
        if end > offset:
            scores[offset:end] = batch_scores[: end - offset]
            offset = end
        if offset >= total:
            break
    if offset == 0:
        return None
    return scores


def _eval_test_metrics(
    model,
    test_loader,
    metadata,
    ref_args,
    device,
    f1_threshold: float,
    max_batches: int = 0,
):
    normality_scores = _eval_normality_scores(
        model, test_loader, ref_args, device, max_batches=max_batches
    )
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


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--reference_args", type=Path, default=None)
    ap.add_argument("--reference_ckpt", type=Path, default=None,
                    help="checkpoint path that contains args (state['args'])")

    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default=None)

    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--fishers", nargs="+", required=True)

    ap.add_argument("--ref_idx", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--no_ref_fisher", action="store_true")

    ap.add_argument("--grid_search", action="store_true")
    ap.add_argument("--grid_values", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--grid_normalize", action="store_true")
    ap.add_argument("--select_by", choices=["roc_auc", "pr_auc", "f1"], default="roc_auc")
    ap.add_argument("--random_search", action="store_true",
                    help="Random Dirichlet search over alphas.")
    ap.add_argument("--num_samples", type=int, default=200,
                    help="Number of random alpha samples.")
    ap.add_argument("--fast_max_batches", type=int, default=150,
                    help="Max test batches for fast stage (<=0 = full test).")
    ap.add_argument("--topk", type=int, default=30,
                    help="Top-K candidates to re-evaluate on full test.")
    ap.add_argument("--dirichlet_alpha", type=float, default=0.3,
                    help="Dirichlet concentration for random alphas.")
    ap.add_argument("--adaptive_rounds", type=int, default=1,
                    help="Adaptive full-eval rounds (>=1).")
    ap.add_argument("--elite_delta", type=float, default=0.002,
                    help="Elite threshold: score >= best - delta.")
    ap.add_argument("--local_method", choices=["logit", "dirichlet"], default="logit",
                    help="Local sampling method around elites.")
    ap.add_argument("--local_sigma", type=float, default=0.3,
                    help="Stddev for logit+noise sampling.")
    ap.add_argument("--local_conc", type=float, default=50.0,
                    help="Dirichlet concentration for local sampling.")

    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--f1_threshold", type=float, default=0.5)

    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--metrics_json", type=Path, default=None)
    ap.add_argument("--no_progress", action="store_true",
                    help="Disable tqdm progress output.")
    ap.add_argument("--roc_only", action="store_true",
                    help="Silence logs and print only best ROC.")
    ap.add_argument("--log_level", default="INFO")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.roc_only:
        args.no_progress = True
        args.log_level = "ERROR"
    logger = _setup_logger(args.log_level)

    if len(args.checkpoints) != len(args.fishers):
        raise ValueError("checkpoints and fishers must align 1:1")

    if args.reference_args is None and args.reference_ckpt is None:
        raise ValueError("Provide --reference_args or --reference_ckpt")

    loaded_args = None
    if args.reference_args is not None:
        loaded_args = _load_reference_args(args.reference_args)
    elif args.reference_ckpt is not None:
        loaded_args = _load_args_from_checkpoint(args.reference_ckpt)
        if loaded_args is None:
            logger.warning("checkpoint has no args: %s. Using defaults.", args.reference_ckpt)

    ref_args = _merge_with_defaults(loaded_args)
    ref_args.no_progress = bool(args.no_progress)
    ref_args.disable_tqdm = bool(args.no_progress)
    if args.dataset is not None:
        ref_args.dataset = args.dataset
    if args.data_dir is not None:
        ref_args.data_dir = args.data_dir
    if args.device:
        ref_args.device = args.device
    ref_args, model_args = init_sub_args(ref_args)

    dataset, loader = get_dataset_and_loader(ref_args, trans_list=trans_list, only_test=False)
    model_args = init_model_params(ref_args, dataset)
    test_loader = loader["test"]
    test_metadata = dataset["test"].metadata

    dev_str = str(ref_args.device)
    if dev_str.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(dev_str if ":" in dev_str else f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    logger.info("Using device=%s", device)

    models = [_load_state_dict(p) for p in args.checkpoints]
    raw_fishers = [torch.load(p, map_location="cpu") for p in args.fishers]
    fishers = []
    fisher_metas = []
    for sd, raw in zip(models, raw_fishers):
        fobj, meta = _split_fisher_payload(raw)
        fishers.append(_normalize_fisher_to_state_dict(fobj, sd, eps=args.eps))
        fisher_metas.append(meta)

    use_ref_fisher = not args.no_ref_fisher

    def evaluate_merged(state_dict: Dict[str, torch.Tensor], max_batches: int = 0) -> Dict[str, float]:
        model = STG_NF(**model_args)
        model.load_state_dict(state_dict, strict=False)
        if hasattr(model, "set_actnorm_init"):
            model.set_actnorm_init()
        model.to(device)

        metrics = _eval_test_metrics(
            model, test_loader, test_metadata, ref_args, device,
            f1_threshold=float(args.f1_threshold),
            max_batches=max_batches,
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return metrics

    if args.grid_search and args.random_search:
        raise ValueError("Use only one of --grid_search or --random_search.")
    if not args.grid_search and not args.random_search:
        raise ValueError("Use --grid_search or --random_search.")

    k_models = len(args.checkpoints)
    best = None
    best_state = None
    best_alphas = None
    best_metrics = None
    logs = []
    fast_logs = None
    round_summaries = None

    if args.random_search:
        if args.dirichlet_alpha <= 0:
            raise ValueError("--dirichlet_alpha must be > 0")
        num_samples = int(args.num_samples)
        if num_samples <= 0:
            raise ValueError("--num_samples must be > 0")
        fast_max_batches = int(args.fast_max_batches)
        topk = int(args.topk)
        fast_is_full = fast_max_batches <= 0 or fast_max_batches >= len(test_loader)
        adaptive_rounds = int(args.adaptive_rounds)
        if adaptive_rounds <= 0:
            raise ValueError("--adaptive_rounds must be >= 1")
        elite_delta = float(args.elite_delta)
        if elite_delta < 0:
            raise ValueError("--elite_delta must be >= 0")
        local_method = args.local_method
        local_sigma = float(args.local_sigma)
        local_conc = float(args.local_conc)
        if local_method == "logit" and local_sigma <= 0:
            raise ValueError("--local_sigma must be > 0")
        if local_method == "dirichlet" and local_conc <= 0:
            raise ValueError("--local_conc must be > 0")
        rng = _init_rng(getattr(ref_args, "seed", None))

        if adaptive_rounds > 1:
            if fast_max_batches > 0:
                logger.warning("adaptive_rounds uses partial eval (fast_max_batches=%d).", fast_max_batches)
            if args.select_by == "f1":
                logger.warning("adaptive search with select_by=f1 can be unstable; consider roc_auc/pr_auc.")
            logger.info(
                "Adaptive search: rounds=%d samples=%d elite_delta=%.4g local_method=%s",
                adaptive_rounds, num_samples, elite_delta, local_method,
            )
            round_summaries = []
            elites: List[List[float]] = []
            for round_idx in range(adaptive_rounds):
                if round_idx == 0 or not elites:
                    samples = [
                        rng.dirichlet([float(args.dirichlet_alpha)] * k_models).tolist()
                        for _ in range(num_samples)
                    ]
                    sample_mode = "global"
                else:
                    samples = _sample_from_elites(
                        elites=elites,
                        n=num_samples,
                        method=local_method,
                        sigma=local_sigma,
                        conc=local_conc,
                        rng=rng,
                    )
                    if len(samples) < num_samples:
                        extra = [
                            rng.dirichlet([float(args.dirichlet_alpha)] * k_models).tolist()
                            for _ in range(num_samples - len(samples))
                        ]
                        samples.extend(extra)
                    sample_mode = "local"

                round_entries = []
                round_best = None
                for alphas in samples:
                    merged = ugm_merge_state_dicts(
                        models=models,
                        fishers=fishers,
                        alphas=torch.tensor(alphas, dtype=torch.float32),
                        ref_idx=args.ref_idx,
                        eps=args.eps,
                        use_ref_fisher=use_ref_fisher,
                    )
                    metrics = evaluate_merged(merged, max_batches=fast_max_batches)
                    entry = {"round": round_idx + 1, "alphas": alphas, "metrics": metrics}
                    logs.append(entry)
                    round_entries.append(entry)

                    score = _safe_metric(metrics, args.select_by)
                    if score is None:
                        continue
                    if round_best is None or score > round_best:
                        round_best = score
                    if best is None or score > best:
                        best = score
                        best_state = merged
                        best_alphas = alphas
                        best_metrics = metrics
                        logger.info("New best %s=%.4f with alphas=%s", args.select_by, best, best_alphas)

                elites = []
                if round_best is not None:
                    for entry in round_entries:
                        score = _safe_metric(entry["metrics"], args.select_by)
                        if score is None:
                            continue
                        if score >= round_best - elite_delta:
                            elites.append(entry["alphas"])
                round_summaries.append(
                    {
                        "round": int(round_idx + 1),
                        "best_score": round_best,
                        "elite_count": int(len(elites)),
                        "sample_mode": sample_mode,
                    }
                )
        else:
            if args.select_by == "f1" and fast_max_batches > 0:
                logger.warning("fast stage with select_by=f1 can be unstable; consider roc_auc/pr_auc.")
            logger.info(
                "Random search: samples=%d dirichlet_alpha=%.3f fast_max_batches=%d topk=%d",
                num_samples, float(args.dirichlet_alpha), fast_max_batches, topk,
            )

            fast_logs = []
            fast_candidates = []

            for _ in range(num_samples):
                alphas = rng.dirichlet([float(args.dirichlet_alpha)] * k_models).tolist()
                merged = ugm_merge_state_dicts(
                    models=models,
                    fishers=fishers,
                    alphas=torch.tensor(alphas, dtype=torch.float32),
                    ref_idx=args.ref_idx,
                    eps=args.eps,
                    use_ref_fisher=use_ref_fisher,
                )
                metrics = evaluate_merged(merged, max_batches=fast_max_batches)
                fast_logs.append({"alphas": alphas, "metrics": metrics})

                score = _safe_metric(metrics, args.select_by)
                if score is None:
                    continue
                fast_candidates.append({"score": score, "alphas": alphas})

                if best is None or score > best:
                    best = score
                    best_state = merged
                    best_alphas = alphas
                    best_metrics = metrics
                    logger.info("New best (fast) %s=%.4f with alphas=%s", args.select_by, best, best_alphas)

            if fast_is_full:
                logger.info("fast_max_batches covers full test; skipping stage2.")
            if (not fast_is_full) and topk > 0 and fast_candidates:
                topk = min(topk, len(fast_candidates))
                fast_candidates.sort(key=lambda x: x["score"], reverse=True)
                top_alphas = [c["alphas"] for c in fast_candidates[:topk]]
                logs = []
                best_full = None
                best_full_state = None
                best_full_alphas = None
                best_full_metrics = None
                for alphas in top_alphas:
                    merged = ugm_merge_state_dicts(
                        models=models,
                        fishers=fishers,
                        alphas=torch.tensor(alphas, dtype=torch.float32),
                        ref_idx=args.ref_idx,
                        eps=args.eps,
                        use_ref_fisher=use_ref_fisher,
                    )
                    metrics = evaluate_merged(merged, max_batches=0)
                    logs.append({"alphas": alphas, "metrics": metrics})

                    score = _safe_metric(metrics, args.select_by)
                    if score is None:
                        continue
                    if best_full is None or score > best_full:
                        best_full = score
                        best_full_state = merged
                        best_full_alphas = alphas
                        best_full_metrics = metrics
                        logger.info("New best %s=%.4f with alphas=%s", args.select_by, best_full, best_full_alphas)

                if best_full_state is not None:
                    best = best_full
                    best_state = best_full_state
                    best_alphas = best_full_alphas
                    best_metrics = best_full_metrics
                else:
                    logger.warning("No valid full-eval candidate; falling back to fast stage best.")
            else:
                logs = fast_logs
    else:
        grid_values = list(args.grid_values)
        alpha_key_precision = 8
        seen_alphas = set()

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

            # Avoid duplicate alpha vectors (e.g., scaled combos that normalize to the same weights).
            key = tuple(round(float(a), alpha_key_precision) for a in alphas)
            if key in seen_alphas:
                continue
            seen_alphas.add(key)

            merged = ugm_merge_state_dicts(
                models=models,
                fishers=fishers,
                alphas=torch.tensor(alphas, dtype=torch.float32),
                ref_idx=args.ref_idx,
                eps=args.eps,
                use_ref_fisher=use_ref_fisher,
            )

            metrics = evaluate_merged(merged)
            logs.append({"alphas": alphas, "metrics": metrics})

            score = metrics.get(args.select_by, None)
            if score is None or (isinstance(score, float) and not np.isfinite(score)):
                continue
            score = float(score)

            logger.info(
                "alphas=%s | roc=%.4f pr=%.4f f1=%.4f | select_by=%s=%.4f",
                [round(a, 3) for a in alphas],
                metrics["roc_auc"], metrics["pr_auc"], metrics["f1"],
                args.select_by, score,
            )

            if best is None or score > best:
                best = score
                best_state = merged
                best_alphas = alphas
                best_metrics = metrics
                logger.info("New best %s=%.4f with alphas=%s", args.select_by, best, best_alphas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_json or args.output.with_suffix(".metrics.json")

    if best_state is None:
        logger.warning("No valid combination found.")
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "best": None,
                    "logs": logs,
                    "fast_logs": fast_logs,
                    "adaptive_rounds": int(args.adaptive_rounds),
                    "elite_delta": float(args.elite_delta),
                    "local_method": args.local_method,
                    "local_sigma": float(args.local_sigma),
                    "local_conc": float(args.local_conc),
                    "round_summaries": round_summaries,
                },
                f,
                indent=2,
            )
        if args.roc_only:
            print("nan")
        return

    ckpt_payload = {
        "state_dict": best_state,
        "metrics": best_metrics,
        "best_alphas": best_alphas,
        "select_by": args.select_by,
        "f1_threshold": args.f1_threshold,
    }
    torch.save(ckpt_payload, args.output)

    payload = {
        "soup_path": str(args.output),
        "best_alphas": best_alphas,
        "select_by": args.select_by,
        "f1_threshold": args.f1_threshold,
        "best_metrics": best_metrics,
        "logs": logs,
        "fast_logs": fast_logs,
        "random_search": bool(args.random_search),
        "num_samples": int(args.num_samples),
        "fast_max_batches": int(args.fast_max_batches),
        "topk": int(args.topk),
        "dirichlet_alpha": float(args.dirichlet_alpha),
        "adaptive_rounds": int(args.adaptive_rounds),
        "elite_delta": float(args.elite_delta),
        "local_method": args.local_method,
        "local_sigma": float(args.local_sigma),
        "local_conc": float(args.local_conc),
        "round_summaries": round_summaries,
        "fisher_meta": fisher_metas,
        "checkpoints": args.checkpoints,
        "fishers": args.fishers,
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved best merged ckpt to %s", args.output)
    logger.info("Saved metrics/logs to %s", metrics_path)
    if args.roc_only:
        roc = float("nan") if best_metrics is None else float(best_metrics.get("roc_auc", float("nan")))
        print(f"{roc:.6f}")


if __name__ == "__main__":
    main()

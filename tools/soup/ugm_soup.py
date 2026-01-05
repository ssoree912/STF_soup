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

    def evaluate_merged(state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        model = STG_NF(**model_args)
        model.load_state_dict(state_dict, strict=False)
        if hasattr(model, "set_actnorm_init"):
            model.set_actnorm_init()
        model.to(device)

        metrics = _eval_test_metrics(
            model, test_loader, test_metadata, ref_args, device,
            f1_threshold=float(args.f1_threshold),
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return metrics

    if not args.grid_search:
        raise ValueError("Use --grid_search for now (provide --grid_search and --grid_values).")

    k_models = len(args.checkpoints)
    grid_values = list(args.grid_values)
    alpha_key_precision = 8

    best = None
    best_state = None
    best_alphas = None
    best_metrics = None
    logs = []
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
            json.dump({"best": None, "logs": logs}, f, indent=2)
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

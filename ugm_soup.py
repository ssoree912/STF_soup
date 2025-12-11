#!/usr/bin/env python3
"""
UGM 기반 STG-NF 소프 실행 스크립트.
  - 대각 Fisher와 알파를 이용해 UGM 병합
  - 그리드 서치/단일 알파, 평가까지 지원
예시:
python ugm_soup.py --reference_args experiments/.../args.json \\
  --checkpoints ckpt1.pth.tar ckpt2.pth.tar \\
  --fishers fisher1.pt fisher2.pt --alphas 0.5 0.5 \\
  --output results/ugm/merged.pth.tar --evaluate
"""

import argparse
import copy
import itertools
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from sklearn.metrics import average_precision_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from args import init_sub_args
from dataset import get_dataset_and_loader
from models.STG_NF.model_pose import STG_NF
from tools.soup.fisher_soup_stg_nf import FisherSoupSTGNF, _normalize_state_dict
from utils.data_utils import trans_list
from utils.train_utils import init_model_params


def _setup_logger(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("ugm_soup")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return _normalize_state_dict(state)


def _load_mask(checkpoint_path: str, mask_name: str, logger: logging.Logger) -> Optional[Dict[str, torch.Tensor]]:
    ckpt_dir = os.path.dirname(checkpoint_path)
    mask_path = os.path.join(ckpt_dir, mask_name)
    if not mask_path or not os.path.exists(mask_path):
        return None
    try:
        raw = torch.load(mask_path, map_location="cpu")
        return {k: (v != 0).to(torch.bool) for k, v in raw.items()}
    except Exception as exc:
        logger.warning("Failed to load mask from %s: %s", mask_path, exc)
        return None


def _combine_masks(masks: Sequence[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    combined: Dict[str, torch.Tensor] = {}
    has_mask = False
    for mask in masks:
        if mask is None:
            continue
        has_mask = True
        for key, tensor in mask.items():
            bool_tensor = (tensor != 0).to(torch.bool)
            if key not in combined:
                combined[key] = bool_tensor.clone()
            else:
                combined[key] = torch.logical_or(combined[key], bool_tensor)
    return combined if has_mask else None


def _apply_mask_to_state_dict(state: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    masked = copy.deepcopy(state)
    for name, mask_tensor in mask.items():
        if name in masked and torch.is_tensor(masked[name]):
            m = mask_tensor.to(masked[name].device, dtype=masked[name].dtype)
            masked[name] = masked[name] * m
    return masked


def _load_fisher_raw(path: str):
    return torch.load(path, map_location="cpu")


def _split_fisher_payload(raw) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
    """
    Fisher 파일이 {"fisher": ..., "metadata": ...} 형태면 분리하고,
    그렇지 않으면 그대로 반환한다.
    """
    if isinstance(raw, dict) and "fisher" in raw:
        return raw["fisher"], raw.get("metadata")
    return raw, None


def _extract_fisher_list(fisher_obj) -> Optional[List[torch.Tensor]]:
    if isinstance(fisher_obj, list):
        return fisher_obj
    if isinstance(fisher_obj, dict):
        if "fisher_list" in fisher_obj and isinstance(fisher_obj["fisher_list"], list):
            return fisher_obj["fisher_list"]

        def _key_sort_key(key: str):
            suffix = key.split("_")[-1]
            return (0, int(suffix)) if suffix.isdigit() else (1, key)

        return [fisher_obj[k] for k in sorted(fisher_obj.keys(), key=_key_sort_key)]
    return None


def _normalize_fisher_to_state_dict(
    fisher_obj,
    model_state: Dict[str, torch.Tensor],
    eps: float,
) -> Dict[str, torch.Tensor]:
    """
    Convert various Fisher formats to a state_dict-shaped dict.

    Accepts:
    - dict matching model_state keys
    - dict subset of model_state keys (fills missing with eps)
    - list of fishers (ordered as mergeable params with dim>1)
    - dict with key 'fisher_list' -> list
    - dict with param_0/param_1... keys -> list order
    """
    if isinstance(fisher_obj, dict):
        fisher_keys = set(fisher_obj.keys())
        model_keys = set(model_state.keys())
        is_param_index_dict = all(k.startswith("param_") for k in fisher_keys)

        if fisher_keys == model_keys:
            return fisher_obj
        if not is_param_index_dict and "fisher_list" not in fisher_obj:
            out: Dict[str, torch.Tensor] = {}
            for key, tensor in model_state.items():
                if key in fisher_obj:
                    out[key] = fisher_obj[key]
                elif torch.is_tensor(tensor):
                    out[key] = torch.full_like(tensor, eps)
                else:
                    out[key] = tensor
            return out

    fisher_list = _extract_fisher_list(fisher_obj)
    if fisher_list is None:
        raise ValueError(
            "Unsupported Fisher format; expected dict keyed like state_dict or list/fisher_list."
        )

    out: Dict[str, torch.Tensor] = {}
    list_idx = 0
    mergeable_needed = sum(
        1
        for t in model_state.values()
        if torch.is_tensor(t) and t.is_floating_point() and t.dim() > 1
    )
    if len(fisher_list) < mergeable_needed:
        raise ValueError(
            f"Fisher list length {len(fisher_list)} is smaller than expected mergeable params {mergeable_needed}."
        )

    for key, tensor in model_state.items():
        if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.dim() > 1:
            fisher_tensor = fisher_list[list_idx]
            list_idx += 1
            out[key] = fisher_tensor
        elif torch.is_tensor(tensor):
            out[key] = torch.full_like(tensor, eps)
        else:
            out[key] = tensor

    return out


def _validate_keys(models: List[Dict[str, torch.Tensor]], fishers: List[Dict[str, torch.Tensor]]) -> None:
    model_keys = set(models[0].keys())
    fisher_keys = set(fishers[0].keys())

    for idx, state in enumerate(models[1:], start=1):
        if set(state.keys()) != model_keys:
            raise ValueError(f"Model state_dict keys differ at index {idx}.")

    for idx, fisher in enumerate(fishers[1:], start=1):
        if set(fisher.keys()) != fisher_keys:
            raise ValueError(f"Fisher keys differ at index {idx}.")

    if model_keys != fisher_keys:
        raise ValueError("Model keys and Fisher keys must match one-to-one for UGM merging.")

'''
ref_tensor : θ_0 
delta : Σ_k α_k * (H_0 + H_k) / (H_0 + Σ_j α_j * H_j) * (θ_k - θ_0)
'''
def ugm_merge_state_dicts(
    models: List[Dict[str, torch.Tensor]], #각 모델 파라미터
    fishers: List[Dict[str, torch.Tensor]], #각 모델의 diagonal fisher 정보
    alphas: torch.Tensor, #각 모델의 가중치
    ref_idx: int = 0, #참조 모델 인덱스() 참조 모델은 UGM 병합 시 기준이 되는 모델)
    eps: float = 1e-8, #수치 안정성 상수
    use_ref_fisher: bool = True, #참조 모델의 Fisher 사용 여부
) -> Dict[str, torch.Tensor]:
    """Merge multiple model state_dicts using uncertainty-based gradient matching."""
    if len(models) != len(fishers): 
        raise ValueError("Number of models and fishers must match.")
    if not models:
        raise ValueError("At least one model is required for merging.")
    if ref_idx < 0 or ref_idx >= len(models):
        raise IndexError(f"ref_idx {ref_idx} is out of range for {len(models)} models.")

    _validate_keys(models, fishers)

    K = len(models) #모델 개수
    alphas = torch.as_tensor(alphas, dtype=torch.float32) #알파텐서 : 각 모델의 가중치
    if alphas.numel() != K:
        raise ValueError(f"alphas must have length {K}, got {alphas.numel()}.")

    merged = copy.deepcopy(models[ref_idx]) #병합된 모델 파라미터
    ref_sd = models[ref_idx] #참조 모델 파라미터

    if use_ref_fisher: #참조 모델의 fisher 사용
        H0_sd = fishers[ref_idx]  
    else:
        H0_sd = {k: torch.full_like(v, eps) for k, v in ref_sd.items()} #참조 모델의 파라미터와 동일한 형태로 eps로 채운 딕셔너리 eps : 수치 안정성 상수

    barH: Dict[str, torch.Tensor] = {} #병합된 모델의 Fisher 정보
    for key in ref_sd.keys(): 
        ref_tensor = ref_sd[key] 
        if not torch.is_tensor(ref_tensor) or not ref_tensor.is_floating_point():
            continue

        fish = torch.stack([f[key] for f in fishers], dim=0) #각 모델의 해당 파라미터에 대한 Fisher 정보를 스택
        a = alphas.view(K, *([1] * (fish.dim() - 1))) #알파 텐서를 해당 파라미터의 차원에 맞게 리쉐이프
        bar = H0_sd[key].clone() 
        bar = bar + torch.sum(a * fish, dim=0) #각 모델의 Fisher 정보에 알파를 곱한 후 참조 모델의 Fisher 정보와 더함
        barH[key] = bar.clamp_min(eps)#H_0 + Σ_k α_k * H_k

    for key in merged.keys():
        ref_tensor = ref_sd[key]
        if not torch.is_tensor(ref_tensor) or not ref_tensor.is_floating_point():
            merged[key] = ref_tensor
            continue

        fish = torch.stack([f[key] for f in fishers], dim=0) #각 모델의 해당 파라미터에 대한 Fisher 정보를 스택
        params = torch.stack([m[key] for m in models], dim=0) 

        a = alphas.view(K, *([1] * (params.dim() - 1))) 

        H0 = H0_sd[key] 
        H0_plus_Ht = H0.unsqueeze(0) + fish #H_0 + H_k

        denom = barH[key] #H_0 + Σ_j α_j * H_j
        W = H0_plus_Ht / denom.unsqueeze(0) #(H_0 + H_k) / (H_0 + Σ_j α_j * H_j)
        weights = a * W

        inc = params - ref_tensor.unsqueeze(0) #(θ_k - θ_0) 
        delta = torch.sum(weights * inc, dim=0) 
        merged[key] = ref_tensor + delta 

    return merged


def merge_from_paths(
    checkpoints: Sequence[str],
    fisher_paths: Sequence[str],
    alphas: Sequence[float],
    output_path: Path,
    ref_idx: int = 0,
    eps: float = 1e-8,
    use_ref_fisher: bool = True,
    mask_name: str = "pruning_mask.pt",
    apply_masks: bool = True,
    logger: Optional[logging.Logger] = None,
    save_output: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], List[Optional[Dict]]]:
    logger = logger or _setup_logger()

    if len(checkpoints) != len(fisher_paths):
        raise ValueError("Checkpoint and Fisher path counts must match.")

    logger.info("Loading %d checkpoints and Fisher files...", len(checkpoints))
    models = [_load_state_dict(p) for p in checkpoints]
    raw_fishers = [_load_fisher_raw(p) for p in fisher_paths]
    fisher_metas: List[Optional[Dict]] = []
    fishers = [
        _normalize_fisher_to_state_dict(
            _split_fisher_payload(raw)[0],
            model_state=sd,
            eps=eps,
        )
        for raw, sd in zip(raw_fishers, models)
    ]
    for raw in raw_fishers:
        _, meta = _split_fisher_payload(raw)
        fisher_metas.append(meta)

    masks = [_load_mask(p, mask_name, logger) for p in checkpoints]
    combined_mask = _combine_masks(masks)

    merged = ugm_merge_state_dicts(
        models=models,
        fishers=fishers,
        alphas=torch.tensor(alphas, dtype=torch.float32),
        ref_idx=ref_idx,
        eps=eps,
        use_ref_fisher=use_ref_fisher,
    )

    if apply_masks and combined_mask is not None:
        merged = _apply_mask_to_state_dict(merged, combined_mask)

    if save_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged, output_path)
        logger.info("Saved UGM-merged checkpoint to %s", output_path)

    return merged, combined_mask, fisher_metas


def _load_reference_args(path: Path) -> argparse.Namespace:
    with open(path, "r") as fp:
        payload = json.load(fp)
    return argparse.Namespace(**payload)


def _prepare_eval(
    reference_args: Path,
    device_override: Optional[str],
    gpu_id: int,
    logger: logging.Logger,
):
    ref_args = _load_reference_args(reference_args)
    if device_override:
        ref_args.device = device_override
    ref_args.only_test = True

    ref_args, model_args = init_sub_args(ref_args)
    dataset, loader = get_dataset_and_loader(ref_args, trans_list=trans_list, only_test=True)
    model_args = init_model_params(ref_args, dataset)

    requested_device = str(ref_args.device)
    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(requested_device if ":" in requested_device else f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA requested but not available. Falling back to CPU.")
    elif requested_device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            logger.warning("MPS requested but not available. Falling back to CPU.")
    else:
        device = torch.device("cpu")

    test_loader = loader["test"]
    dataset_test = dataset["test"]
    return ref_args, model_args, dataset_test, test_loader, device


def _evaluate_state_dict(
    state: Dict[str, torch.Tensor],
    model_args: Dict,
    ref_args: argparse.Namespace,
    test_loader,
    dataset_test,
    device: torch.device,
    logger: logging.Logger,
    f1_threshold: float,
) -> Dict[str, float]:
    model = STG_NF(**model_args)
    model.load_state_dict(state, strict=False)
    if hasattr(model, "set_actnorm_init"):
        model.set_actnorm_init()
    model.to(device)

    soup_helper = FisherSoupSTGNF(device, logger, model_args=model_args)
    eval_result = soup_helper.evaluate_model(model, test_loader, dataset_test, ref_args)

    scores_np = eval_result["scores"]
    labels_np = eval_result["labels"]
    roc_auc = float(eval_result["roc_auc"])
    pr_auc = float(average_precision_score(labels_np, scores_np))
    f1 = float(f1_score(labels_np, scores_np >= f1_threshold))

    return {"roc_auc": roc_auc, "auc": roc_auc, "pr_auc": pr_auc, "f1": f1}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UGM soup (Fisher-weighted) for STG-NF checkpoints."
    )
    parser.add_argument("--reference_args", type=Path, help="Path to args.json from training (required for evaluation).")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to merge.")
    parser.add_argument("--fishers", nargs="+", required=True, help="Fisher/Hessian paths aligned with checkpoints.")
    parser.add_argument("--alphas", nargs="+", type=float, help="Scalar coefficients (one per checkpoint).")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the merged checkpoint.")
    parser.add_argument("--ref_idx", type=int, default=0, help="Reference checkpoint index.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Numerical stability constant.")
    parser.add_argument("--no_ref_fisher", action="store_true", help="Use constant prior instead of reference Fisher for H0.")
    parser.add_argument("--no_masks", action="store_true", help="Do not apply combined pruning masks after merging.")
    parser.add_argument("--mask_name", default="pruning_mask.pt", help="Pruning mask filename (per checkpoint directory).")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on merged checkpoint.")
    parser.add_argument("--device", default=None, help="Override device for evaluation (cpu, cuda, or mps).")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU index to use when device is cuda without explicit id.")
    parser.add_argument("--metrics_json", type=Path, default=None, help="Path to save evaluation metrics JSON.")
    parser.add_argument("--grid_search", action="store_true", help="Grid search over alpha combinations.")
    parser.add_argument(
        "--grid_values",
        nargs="+",
        type=float,
        default=None,
        help="Base values for each alpha when running grid search (default: 0.0 0.5 1.0).",
    )
    parser.add_argument(
        "--grid_normalize",
        action="store_true",
        help="Normalize each alpha combination so that the sum is 1 (skip all-zero combos).",
    )
    parser.add_argument("--select_by", choices=["roc_auc", "pr_auc", "auc", "f1"], default="roc_auc", help="Metric to select best combo.")
    parser.add_argument("--log_level", default="INFO", help="Logging level.")
    parser.add_argument("--f1_threshold", type=float, default=0.5, help="Threshold on score for F1 computation.")
    return parser.parse_args()


def main():
    args = _parse_args()
    logger = _setup_logger(args.log_level)
    use_ref_fisher = not args.no_ref_fisher
    apply_masks = not args.no_masks

    if not args.grid_search and (args.alphas is None or len(args.alphas) == 0):
        raise ValueError("--alphas is required unless --grid_search is set.")

    if args.evaluate or args.grid_search:
        if args.reference_args is None:
            raise ValueError("--reference_args is required when evaluation or grid search is enabled.")
        ref_args, model_args, dataset_test, test_loader, device = _prepare_eval(
            args.reference_args, args.device, args.gpu_id, logger
        )
    else:
        ref_args = None
        model_args = None
        dataset_test = None
        test_loader = None
        device = torch.device("cpu")

    # Single merge mode
    if not args.grid_search:
        merged_state, combined_mask, fisher_metas = merge_from_paths(
            checkpoints=args.checkpoints,
            fisher_paths=args.fishers,
            alphas=args.alphas,
            output_path=args.output,
            ref_idx=args.ref_idx,
            eps=args.eps,
            use_ref_fisher=use_ref_fisher,
            mask_name=args.mask_name,
            apply_masks=apply_masks,
            logger=logger,
            save_output=True,
        )

        if not args.evaluate:
            return

        metrics = _evaluate_state_dict(
            merged_state,
            model_args,
            ref_args,
            test_loader,
            dataset_test,
            device,
            logger,
            args.f1_threshold,
        )
        metrics_path = args.metrics_json or args.output.with_suffix(".metrics.json")
        payload = {
            "soup_path": str(args.output),
            "alphas": [float(a) for a in args.alphas],
            "mask_applied": bool(combined_mask is not None and apply_masks),
            "f1_threshold": float(args.f1_threshold),
            "fisher_meta": fisher_metas,
            **metrics,
        }
        with open(metrics_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Saved evaluation metrics to %s", metrics_path)
        return

    # Grid search mode
    grid_values = args.grid_values or [0.0, 0.5, 1.0]
    logger.info("Running grid search over alphas with base values: %s", grid_values)

    K = len(args.checkpoints)
    best_metrics: Optional[Dict[str, float]] = None
    best_alphas: Optional[List[float]] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_mask: Optional[Dict[str, torch.Tensor]] = None
    best_fisher_meta: Optional[List[Optional[Dict]]] = None
    combo_logs: List[Dict] = []

    for combo in itertools.product(grid_values, repeat=K):
        combo = list(combo)
        if args.grid_normalize:
            s = sum(combo)
            if s == 0:
                continue
            alphas = [c / s for c in combo]
        else:
            alphas = combo

        logger.info("Trying alphas: %s", alphas)

        merged_state, combined_mask, fisher_metas = merge_from_paths(
            checkpoints=args.checkpoints,
            fisher_paths=args.fishers,
            alphas=alphas,
            output_path=args.output,
            ref_idx=args.ref_idx,
            eps=args.eps,
            use_ref_fisher=use_ref_fisher,
            mask_name=args.mask_name,
            apply_masks=apply_masks,
            logger=logger,
            save_output=False,
        )

        metrics = _evaluate_state_dict(
            merged_state, model_args, ref_args, test_loader, dataset_test, device, logger, args.f1_threshold
        )
        logger.info(
            "Metrics for alphas %s | roc_auc=%.4f pr_auc=%.4f f1(th=%.3f)=%.4f",
            alphas,
            metrics.get("roc_auc", float("nan")),
            metrics.get("pr_auc", float("nan")),
            args.f1_threshold,
            metrics.get("f1", float("nan")),
        )
        combo_logs.append({
            "alphas": [float(a) for a in alphas],
            "mask_applied": bool(combined_mask is not None and apply_masks),
            "metrics": metrics,
            "fisher_meta": fisher_metas,
        })
        metric_value = metrics.get(args.select_by)
        if metric_value is None:
            raise ValueError(f"Metric {args.select_by} not found in evaluation metrics.")

        if best_metrics is None or metric_value > best_metrics.get(args.select_by, float("-inf")):
            best_metrics = metrics
            best_alphas = alphas
            best_state = merged_state
            best_mask = combined_mask
            best_fisher_meta = fisher_metas
            logger.info("New best %s = %.4f with alphas %s", args.select_by, metric_value, alphas)

    if best_state is None or best_metrics is None or best_alphas is None:
        logger.warning("No valid alpha combination found during grid search.")
        return

    if apply_masks and best_mask is not None:
        best_state = _apply_mask_to_state_dict(best_state, best_mask)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)

    metrics_path = args.metrics_json or args.output.with_suffix(".metrics.json")
    payload = {
        "soup_path": str(args.output),
        "alphas": [float(a) for a in best_alphas],
        "mask_applied": bool(best_mask is not None and apply_masks),
        "f1_threshold": float(args.f1_threshold),
        "combo_logs": combo_logs,
        "best_fisher_meta": best_fisher_meta,
        **best_metrics,
    }
    with open(metrics_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Grid search completed. Best alphas: %s", best_alphas)
    logger.info("Saved best merged checkpoint to %s and metrics to %s", args.output, metrics_path)


if __name__ == "__main__":
    main()

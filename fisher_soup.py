"""
Fisher Soup 실행 스크립트(STG-NF 단순 병합).
  - 지정된 ckpt/Fisher로 계수 탐색 후 병합/평가
예시:
python fisher_soup.py --checkpoints ckpt1 ckpt2 --fisher_paths f1 f2 \\
  --reference_args experiments/.../args.json --output results/fisher_soup.pth.tar
"""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from args import init_sub_args
from dataset import get_dataset_and_loader
from models.STG_NF.model_pose import STG_NF
from utils.data_utils import trans_list
from utils.scoring_utils import score_dataset
from utils.train_utils import init_model_params

SKIP_SOUP_SUBSTRINGS = (
    ".actnorm.",
    "running_mean",
    "running_var",
    "num_batches_tracked",
    "prior_h",
)


def _should_skip_parameter(name: str, tensor: torch.Tensor) -> bool:
    if name.endswith("actnorm.inited"):
        return True
    if tensor.ndim <= 1:
        return True
    if any(token in name for token in SKIP_SOUP_SUBSTRINGS):
        return True
    return False


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
    for key in list(state.keys()):
        if key.endswith("actnorm.inited"):
            tensor = state[key]
            if torch.is_tensor(tensor):
                state[key] = torch.ones_like(tensor)
            else:
                state[key] = 1
        if key.endswith("weight_orig"):
            weight_key = key.replace("weight_orig", "weight")
            mask_key = key.replace("weight_orig", "weight_mask")
            orig_weight = state.pop(key)
            mask_tensor = state.pop(mask_key, torch.ones_like(orig_weight))
            state[weight_key] = orig_weight * mask_tensor
    return state


def _load_fisher(fisher_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict]:
    fisher_blob = torch.load(fisher_path, map_location="cpu")
    if isinstance(fisher_blob, dict) and "fisher" in fisher_blob:
        return fisher_blob["fisher"], fisher_blob.get("metadata", {})
    if isinstance(fisher_blob, dict):
        return fisher_blob, {}
    raise ValueError(f"Unsupported fisher format at {fisher_path}")


def _find_file(run_dir: Path, explicit_name: Optional[str], pattern: str) -> Path:
    if explicit_name:
        candidate = run_dir / explicit_name
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{candidate} not found")
    matches = sorted(run_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {run_dir}")
    return matches[0]


def _persist_evaluation(eval_dir: Path, auc: float, scores: np.ndarray, labels: np.ndarray,
                        roc_parts: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict:
    eval_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    fpr, tpr, thresholds = [np.asarray(arr) for arr in roc_parts]
    metrics_payload = {
        "auc": float(auc),
        "num_samples": int(scores.shape[0]),
        "scores_file": "scores_labels.npz",
        "roc_files": {
            "csv": "roc_curve.csv",
            "npz": "roc_curve.npz"
        }
    }
    with open(eval_dir / "metrics.json", "w") as fp:
        json.dump(metrics_payload, fp, indent=2, sort_keys=True)
    np.savez(eval_dir / "scores_labels.npz", scores=scores, labels=labels)
    roc_matrix = np.stack([fpr, tpr, thresholds], axis=1)
    np.savetxt(eval_dir / "roc_curve.csv", roc_matrix, delimiter=",", header="fpr,tpr,threshold", comments="")
    np.savez(eval_dir / "roc_curve.npz", fpr=fpr, tpr=tpr, thresholds=thresholds)
    return metrics_payload


def _load_reference_args(args_path: Path) -> Namespace:
    with open(args_path, "r") as fp:
        args_dict = json.load(fp)
    return Namespace(**args_dict)


def _evaluate_soup_model(state_dict: Dict[str, torch.Tensor],
                         reference_args_path: Path,
                         device_override: Optional[str] = None):
    if not reference_args_path.exists():
        raise FileNotFoundError(f"No args.json found at {reference_args_path}")
    ref_args = _load_reference_args(reference_args_path)
    if device_override is not None:
        ref_args.device = device_override
    ref_args.only_test = True

    ref_args, model_args = init_sub_args(ref_args)
    dataset, loader = get_dataset_and_loader(ref_args, trans_list=trans_list, only_test=True)
    model_args = init_model_params(ref_args, dataset)
    model = STG_NF(**model_args)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: missing keys when loading soup state dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: unexpected keys when loading soup state dict: {unexpected_keys}")
    if hasattr(model, "set_actnorm_init"):
        model.set_actnorm_init()
    device = torch.device(ref_args.device)
    model.to(device)
    model.eval()

    probs = torch.empty(0, device=device)
    test_loader = loader['test']
    for data_arr in tqdm(test_loader, desc="Soup Evaluation", leave=False):
        data = [d.to(device, non_blocking=True) for d in data_arr]
        score = data[-2].amin(dim=-1)
        if getattr(ref_args, 'model_confidence', False):
            samp = data[0]
        else:
            samp = data[0][:, :2]
        with torch.no_grad():
            _, nll = model(samp.float(), label=torch.ones(data[0].shape[0], device=device), score=score)
        if getattr(ref_args, 'model_confidence', False):
            nll = nll * score
        probs = torch.cat((probs, -1 * nll), dim=0)

    prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
    auc, scores_np, labels_np, roc_parts = score_dataset(prob_mat_np, dataset["test"].metadata, args=ref_args)
    return auc, scores_np, labels_np, roc_parts, ref_args


def _combine_uniform(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    reference = states[0]
    average_names = [name for name, tensor in reference.items() if not _should_skip_parameter(name, tensor)]
    combined: Dict[str, torch.Tensor] = {name: torch.zeros_like(reference[name]) for name in average_names}

    for state in states:
        for name in average_names:
            combined[name] += state[name]

    num_models = len(states)
    combined = {name: tensor / num_models for name, tensor in combined.items()}

    result = {name: reference[name].clone() for name in reference.keys()}
    for name in combined:
        result[name] = combined[name]
    return result


def _flag_actnorm_inited(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for name in list(state.keys()):
        if name.endswith("actnorm.inited"):
            tensor = state[name]
            if torch.is_tensor(tensor):
                state[name] = torch.ones_like(tensor)
            else:
                state[name] = 1
        if ".actnorm.bias" in name or ".actnorm.logs" in name:
            tensor = state[name]
            if tensor.isnan().any():
                raise ValueError(f"ActNorm parameter {name} contains NaNs")
    return state


def _project_invertible(W: torch.Tensor, sigma_min: float = 1e-3) -> torch.Tensor:
    # Ensure 1x1 invertible conv (or similar square matrices) remain well-conditioned
    # by clamping minimal singular value.
    U, S, Vh = torch.linalg.svd(W)
    S = S.clamp_min(sigma_min)
    return (U @ torch.diag(S) @ Vh).to(W.dtype)

def _project_invertible_layers(state: Dict[str, torch.Tensor], sigma_min: float = 1e-3) -> Dict[str, torch.Tensor]:
    for name, tensor in state.items():
        # Heuristic: square 2D weights that likely belong to invertible 1x1 conv/perm layers
        if tensor.ndim == 2 and tensor.shape[0] == tensor.shape[1]:
            lowered = name.lower()
            if any(tag in lowered for tag in ("invconv", "invertible", "perm", "1x1", "conv1x1")):
                try:
                    state[name] = _project_invertible(tensor, sigma_min=sigma_min)
                except Exception as e:
                    print(f"Warning: SVD projection failed for {name}: {e}")
    return state


def _combine_fisher(states: List[Dict[str, torch.Tensor]], fisher_list: List[Dict[str, torch.Tensor]],
                    eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    reference = states[0]
    average_names = [name for name, tensor in reference.items() if not _should_skip_parameter(name, tensor)]
    combined: Dict[str, torch.Tensor] = {name: torch.zeros_like(reference[name]) for name in average_names}
    denom: Dict[str, torch.Tensor] = {name: torch.zeros_like(reference[name]) for name in average_names}
    warned_missing = set()
    for state, fisher in zip(states, fisher_list):
        for name in average_names:
            tensor = state[name]
            fisher_tensor = fisher.get(name)
            if fisher_tensor is None:
                if name not in warned_missing:
                    print(f"Warning: missing fisher entry for '{name}', falling back to uniform weight")
                    warned_missing.add(name)
                fisher_tensor = torch.ones_like(tensor)

            # Per-tensor Fisher normalization (mean-based), with safety clamps
            f = fisher_tensor.abs()
            scale = f.mean().clamp_min(eps)
            f = (f / scale).clamp_min(1e-8)

            combined[name] += tensor * f
            denom[name]    += f

    for name in average_names:
        denom_tensor = denom[name].clamp_min(eps)
        combined[name] = combined[name] / denom_tensor
    result = {name: reference[name].clone() for name in reference.keys()}
    for name in combined:
        result[name] = combined[name]
    return result


def _aggregate_fisher(fisher_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    total: Dict[str, torch.Tensor] = {}
    for fisher in fisher_list:
        for name, tensor in fisher.items():
            if name not in total:
                total[name] = tensor.clone()
            else:
                total[name] += tensor
    return total


def _compute_fisher_norms(fishers: List[Dict[str, torch.Tensor]]) -> Optional[List[torch.Tensor]]:
    if not fishers or not fishers[0]:
        return None
    norms = []
    for fisher in fishers:
        total = torch.tensor(0.0)
        for tensor in fisher.values():
            total = total + torch.sum(tensor ** 2)
        norms.append(torch.sqrt(total).clamp_min(1e-12))
    return norms


def _combine_with_coeffs(states: List[Dict[str, torch.Tensor]],
                         fishers: List[Dict[str, torch.Tensor]],
                         coeffs: Sequence[float],
                         method: str,
                         fisher_floor: float,
                         favor_target_model: bool,
                         normalize_fisher: bool) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    reference = states[0]
    names_to_mix = [name for name, tensor in reference.items() if not _should_skip_parameter(name, tensor)]
    coeffs = list(coeffs)

    fisher_norms = _compute_fisher_norms(fishers) if (method == "fisher" and normalize_fisher) else None

    combined = {name: reference[name].clone() for name in reference.keys()}
    aggregated_fisher = {} if fishers and fishers[0] else {}

    for name in names_to_mix:
        numerator = None
        denominator = None
        fisher_accum = None
        for idx, (state, coeff) in enumerate(zip(states, coeffs)):
            tensor = state[name]
            if method == "uniform" or not fishers or not fishers[idx]:
                weight = torch.tensor(coeff, dtype=tensor.dtype, device=tensor.device)
                if tensor.ndim > 0:
                    weight = weight
                contrib = tensor * weight
                denom_contrib = weight
            else:
                fisher_tensor = fishers[idx].get(name)
                if fisher_tensor is None:
                    fisher_tensor = torch.ones_like(tensor)
                if fisher_norms is not None:
                    fisher_tensor = fisher_tensor / fisher_norms[idx]
                if favor_target_model and idx == 0:
                    clamped = fisher_tensor
                else:
                    clamped = torch.clamp(fisher_tensor, min=fisher_floor)
                weight_tensor = clamped * coeff
                contrib = tensor * weight_tensor
                denom_contrib = weight_tensor
                if aggregated_fisher is not None:
                    fisher_accum = fisher_accum + (fisher_tensor * coeff) if fisher_accum is not None else (fisher_tensor * coeff)

            numerator = contrib if numerator is None else numerator + contrib
            denominator = denom_contrib if denominator is None else denominator + denom_contrib

        combined[name] = numerator / denominator.clamp_min(1e-12)
        if aggregated_fisher is not None and fisher_accum is not None:
            aggregated_fisher[name] = fisher_accum

    combined = _flag_actnorm_inited(combined)
    combined = _project_invertible_layers(combined, sigma_min=1e-3)

    return combined, aggregated_fisher


def _generate_coefficients(n_models: int, strategy: str, n_combinations: int, seed: int) -> List[List[float]]:
    if strategy == "uniform" or n_models == 1:
        return [[1.0 / n_models] * n_models]

    rng = np.random.default_rng(seed)
    if strategy == "grid" and n_models == 2:
        points = max(2, n_combinations)
        weights = []
        for i in range(points + 1):
            w0 = i / points
            w1 = 1.0 - w0
            weights.append([w0, w1])
        return weights

    # Random Dirichlet sampling
    coeffs = []
    alpha = np.ones(n_models)
    for _ in range(n_combinations):
        coeffs.append(rng.dirichlet(alpha).tolist())
    return coeffs


def load_states_and_fishers(run_dirs: List[Path],
                            checkpoint_name: Optional[str],
                            fisher_name: Optional[str],
                            checkpoint_pattern: str,
                            fisher_pattern: str) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Dict]]:
    states: List[Dict[str, torch.Tensor]] = []
    fishers: List[Dict[str, torch.Tensor]] = []
    meta: List[Dict] = []

    for run_dir in run_dirs:
        ckpt_path = _find_file(run_dir, checkpoint_name, checkpoint_pattern)
        fisher_path = _find_file(run_dir, fisher_name, fisher_pattern) if fisher_name or fisher_pattern else None

        state_dict = _load_checkpoint(ckpt_path)
        states.append(state_dict)

        fisher_dict = None
        fisher_meta = {}
        if fisher_path:
            fisher_dict, fisher_meta = _load_fisher(fisher_path)
            fishers.append(fisher_dict)
        else:
            fishers.append({})
        meta.append({
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt_path),
            "fisher": str(fisher_path) if fisher_path else None,
            "fisher_meta": fisher_meta,
        })

    return states, fishers, meta


def save_soup(output_path: Path,
              state_dict: Dict[str, torch.Tensor],
              metadata: Dict) -> None:
    payload = {
        "state_dict": state_dict,
        "soup_metadata": metadata,
    }
    torch.save(payload, output_path)


def save_metadata(output_dir: Path, metadata: Dict) -> None:
    meta_path = output_dir / "soup_metadata.json"
    with open(meta_path, "w") as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Fisher Soup model combiner")
    parser.add_argument("--run_dirs", nargs="+", required=True, help="Run directories containing checkpoints and fisher_diag.pt")
    parser.add_argument("--output_path", required=True, help="Path to save the combined soup checkpoint (.pth.tar)")
    parser.add_argument("--method", choices=["fisher", "uniform"], default="fisher", help="Combination strategy")
    parser.add_argument("--checkpoint_name", default=None, help="Specific checkpoint file name inside each run dir")
    parser.add_argument("--fisher_name", default=None, help="Specific fisher file name inside each run dir (default: fisher_diag.pt)")
    parser.add_argument("--checkpoint_pattern", default="checkpoint_best.pth.tar", help="Glob pattern to locate checkpoint if name not provided")
    parser.add_argument("--fisher_pattern", default="fisher_diag.pt", help="Glob pattern to locate fisher file if name not provided")
    parser.add_argument("--save_fisher", action="store_true", help="Save aggregated fisher tensor alongside soup checkpoint")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on the combined soup and save ROC metrics")
    parser.add_argument("--eval_device", default=None, help="Device override for soup evaluation (e.g., cpu or cuda:0)")
    parser.add_argument("--strategy", choices=["uniform", "grid", "random"], default="uniform",
                        help="Coefficient generation strategy for soup (ignored when method=uniform and strategy=uniform")
    parser.add_argument("--n_combinations", type=int, default=10, help="Number of coefficient combinations to evaluate")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for coefficient sampling")
    parser.add_argument("--fisher_floor", type=float, default=1e-8, help="Minimum fisher value applied during merging")
    parser.add_argument("--no_favor_target", action="store_true", help="Do not treat the first model specially when applying fisher floor")
    parser.add_argument("--no_normalize_fisher", action="store_true", help="Disable fisher L2 normalization")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dirs = [Path(run_dir).expanduser().resolve() for run_dir in args.run_dirs]
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    states, fishers, member_meta = load_states_and_fishers(
        run_dirs=run_dirs,
        checkpoint_name=args.checkpoint_name,
        fisher_name=args.fisher_name,
        checkpoint_pattern=args.checkpoint_pattern,
        fisher_pattern=args.fisher_pattern,
    )

    n_models = len(states)
    if n_models == 0:
        raise ValueError("No checkpoints found to combine")

    if args.method == "uniform":
        coefficient_sets = [[1.0 / n_models] * n_models]
    else:
        strategy = args.strategy if args.strategy != "uniform" else ("grid" if n_models == 2 else "random")
        coefficient_sets = _generate_coefficients(n_models, strategy, args.n_combinations, args.random_seed)

    favor_target = not args.no_favor_target
    normalize_fisher = not args.no_normalize_fisher

    best_state = None
    best_fisher = {}
    best_coeffs = None
    best_auc = float("-inf")
    best_eval_payload = None
    search_results = []
    reference_args_path = run_dirs[0] / "args.json"

    for coeffs in coefficient_sets:
        combined_state, aggregated_fisher = _combine_with_coeffs(
            states=states,
            fishers=fishers,
            coeffs=coeffs,
            method=args.method,
            fisher_floor=args.fisher_floor,
            favor_target_model=favor_target,
            normalize_fisher=normalize_fisher,
        )

        if args.evaluate:
            auc, scores_np, labels_np, roc_parts, ref_args = _evaluate_soup_model(
                combined_state,
                reference_args_path=reference_args_path,
                device_override=args.eval_device,
            )
            search_results.append({"coefficients": coeffs, "auc": float(auc)})
            if auc > best_auc:
                best_auc = auc
                best_state = combined_state
                best_fisher = aggregated_fisher
                best_coeffs = coeffs
                best_eval_payload = (scores_np, labels_np, roc_parts, ref_args)
        else:
            if best_state is None:
                best_state = combined_state
                best_fisher = aggregated_fisher
                best_coeffs = coeffs

    if best_state is None:
        raise RuntimeError("Failed to create soup model")

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    soup_metadata = {
        "method": args.method,
        "members": member_meta,
        "strategy": args.strategy,
        "n_combinations": args.n_combinations,
        "random_seed": args.random_seed,
        "coefficients": best_coeffs,
        "fisher_floor": args.fisher_floor,
        "favor_target_model": favor_target,
        "normalize_fisher": normalize_fisher,
    }
    evaluation_summary = None
    if args.evaluate:
        if best_eval_payload is None:
            raise RuntimeError("Failed to evaluate any soup combination")
        scores_np, labels_np, roc_parts, ref_args = best_eval_payload
        eval_dir = output_path.parent / f"{output_path.stem}_eval"
        metrics_payload = _persist_evaluation(eval_dir, best_auc, scores_np, labels_np, roc_parts)
        evaluation_summary = {
            "auc": metrics_payload["auc"],
            "num_samples": metrics_payload["num_samples"],
            "metrics_dir": str(eval_dir),
            "roc_csv": str(eval_dir / "roc_curve.csv"),
            "roc_npz": str(eval_dir / "roc_curve.npz"),
            "scores_npz": str(eval_dir / "scores_labels.npz"),
            "reference_args": str(reference_args_path),
            "device": str(ref_args.device),
            "dataset": ref_args.dataset,
        }
        soup_metadata["evaluation"] = evaluation_summary
        soup_metadata["searched_coefficients"] = search_results

    if args.save_fisher and best_fisher:
        fisher_output = output_path.with_suffix(".fisher.pt")
        torch.save(best_fisher, fisher_output)
        soup_metadata["aggregated_fisher"] = str(fisher_output)

    save_soup(output_path, best_state, soup_metadata)
    save_metadata(output_path.parent, soup_metadata)
    print(f"Saved soup checkpoint to {output_path}")
    if evaluation_summary is not None:
        print(f"Soup evaluation AUC: {evaluation_summary['auc'] * 100:.2f}% saved under {evaluation_summary['metrics_dir']}")
    if args.save_fisher and best_fisher:
        print(f"Saved aggregated fisher to {output_path.with_suffix('.fisher.pt')}")


if __name__ == "__main__":
    main()

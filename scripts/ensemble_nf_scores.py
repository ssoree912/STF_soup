import os
import json
import argparse
import random
import sys
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure project root is on sys.path for local imports.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.data_utils import trans_list
from utils.unlearning_utils import (
    load_model_from_checkpoint,
    prepare_batch,
    reduce_conf_score,
)
from utils.scoring_utils import get_dataset_scores

# -------------------------
# Utils
# -------------------------
class IndexedDataset(Dataset):
    """Return (data, sid, score, label) for stable indexing."""
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        sid = self.indices[i]
        data, trans_index, score, label = self.dataset[sid]
        return data, sid, score, label


def build_indexed_loader(dataset: Dataset, indices: List[int], batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        IndexedDataset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def infer_scores(
    model,
    dataset: Dataset,
    indices: List[int],
    device: torch.device,
    model_confidence: bool,
    batch_size: int,
    num_workers: int,
    use_conf_score: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      scores: (N,) anomaly score (higher = more anomalous). Here we use nll.
      labels: (N,) raw labels from dataset
    """
    loader = build_indexed_loader(dataset, indices, batch_size, num_workers, shuffle=False)
    model.eval()

    scores_out = np.zeros((len(indices),), dtype=np.float32)
    labels_out = np.zeros((len(indices),), dtype=np.int64)

    # Map sid -> position in output array
    pos = {sid: i for i, sid in enumerate(indices)}

    for batch in loader:
        x, score, label = prepare_batch(batch, device, model_confidence)
        _, nll = model(x, label=label)  # nll: (B,)

        if use_conf_score:
            nll = nll * reduce_conf_score(score)

        sids = batch[1]
        if torch.is_tensor(sids):
            sids = sids.tolist()

        nll_np = nll.detach().cpu().numpy().astype(np.float32)
        lab_np = label.detach().cpu().numpy().astype(np.int64)

        for j, sid in enumerate(sids):
            k = pos[int(sid)]
            scores_out[k] = nll_np[j]
            labels_out[k] = lab_np[j]

    return scores_out, labels_out


def rank_average(scores_list: List[np.ndarray]) -> np.ndarray:
    """
    Rank fusion (scale-invariant).
    Convert each score array to ranks in [0,1], then average.
    """
    ranks = []
    for s in scores_list:
        order = np.argsort(s)                 # ascending
        r = np.empty_like(order, dtype=np.float32)
        r[order] = np.arange(len(s), dtype=np.float32)
        r = r / max(1.0, float(len(s) - 1))   # [0,1]
        ranks.append(r)
    return np.mean(np.stack(ranks, axis=0), axis=0)


def zscore_mean(scores_list: List[np.ndarray], eps: float = 1e-8) -> np.ndarray:
    zs = []
    for s in scores_list:
        mu = float(np.mean(s))
        sd = float(np.std(s))
        zs.append((s - mu) / (sd + eps))
    return np.mean(np.stack(zs, axis=0), axis=0)


def max_fusion(scores_list: List[np.ndarray]) -> np.ndarray:
    return np.max(np.stack(scores_list, axis=0), axis=0)


def weighted_mean(scores_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    w = np.array(weights, dtype=np.float32)
    w = w / (float(w.sum()) + 1e-8)
    S = np.stack(scores_list, axis=0)  # (M,N)
    return (w[:, None] * S).sum(axis=0)


def compute_auc_metrics(y_true: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    """
    y_true: 1 = anomaly(positive), 0 = normal(negative)
    score: higher => more anomalous
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    out = {}
    # AUROC는 양/음 둘 다 있어야 의미가 있음
    if len(np.unique(y_true)) < 2:
        out["auc"] = float("nan")
        out["auprc"] = float("nan")
        return out

    out["auc"] = float(roc_auc_score(y_true, score))
    out["auprc"] = float(average_precision_score(y_true, score))
    return out


def best_f1(y_true: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    """
    Find best F1 over thresholds on score.
    """
    from sklearn.metrics import f1_score

    if len(np.unique(y_true)) < 2:
        return {"f1": float("nan"), "thr": float("nan")}

    # 후보 threshold: unique scores의 분위수 기반으로 줄여서 탐색 (속도/안정)
    qs = np.linspace(0.0, 1.0, 200)
    thrs = np.quantile(score, qs)

    best = (-1.0, None)
    for t in thrs:
        pred = (score >= t).astype(np.int32)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best[0]:
            best = (float(f1), float(t))
    return {"f1": best[0], "thr": best[1]}


def eval_metrics_with_gt(scores: np.ndarray, dataset, args) -> Dict[str, float]:
    """
    Evaluate with GT masks (normal=1, abnormal=0 in this codebase).
    scores should be "normality" (higher = more normal), consistent with train_eval.
    """
    gt_arr, scores_arr = get_dataset_scores(scores, dataset.metadata, args=args)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    metrics = compute_auc_metrics(gt_np.astype(np.int32), scores_np)
    f1 = best_f1(gt_np.astype(np.int32), scores_np)
    return {**metrics, **f1}


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    k = max(1, int(n * val_ratio))
    val = idx[:k]
    test = idx[k:]
    return val, test


def grid_search_weights(
    y_val: np.ndarray,
    scores_val_list: List[np.ndarray],
    step: float = 0.1,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Simple grid search for 3 models: weights in {0,step,2step,...,1} sum=1.
    목적: val AUROC 최대
    """
    M = len(scores_val_list)
    if M != 3:
        return [1.0 / M] * M, {"auc": float("nan"), "auprc": float("nan")}

    best_w = None
    best_auc = -1.0
    best_metrics = None

    grid = np.arange(0.0, 1.0 + 1e-8, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-8:
                continue
            if w3 < 0:
                w3 = 0.0
            w = [float(w1), float(w2), float(w3)]
            s = weighted_mean(scores_val_list, w)
            m = compute_auc_metrics(y_val, s)
            auc = m.get("auc", float("nan"))
            if np.isnan(auc):
                continue
            if auc > best_auc:
                best_auc = auc
                best_w = w
                best_metrics = m

    if best_w is None:
        best_w = [1/3, 1/3, 1/3]
        best_metrics = compute_auc_metrics(y_val, weighted_mean(scores_val_list, best_w))
    return best_w, best_metrics


# -------------------------
# Main
# -------------------------
def parse_args():
    p = init_parser()
    p.add_argument("--checkpoints", type=str, nargs="+", required=True,
                   help="ckpt paths (e.g., df1_retrain.pth.tar df2_retrain.pth.tar df3_retrain.pth.tar)")
    p.add_argument("--output_path", type=str, default="outputs/unlearning/ensemble_results.json")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--val_ratio_for_weights", type=float, default=0.0,
                   help=">0이면 split subset을 val로 떼서 weights search 후 test에 적용")
    p.add_argument("--val_seed", type=int, default=0)
    p.add_argument("--weight_grid_step", type=float, default=0.1)
    p.add_argument("--eval_with_gt", action="store_true",
                   help="GT 마스크 기반 평가(ShanghaiTech/UBnormal test에 권장)")

    p.add_argument("--use_conf_score", action="store_true",
                   help="nll에 reduce_conf_score(score)를 곱할지 여부 (학습/캐시와 일치시켜야 함)")
    p.add_argument("--normal_label", type=int, default=1,
                   help="dataset label 중 'normal'로 간주할 값. anomaly는 (label != normal_label)로 처리")
    return p.parse_args()


def main():
    args = parse_args()
    args, _ = init_sub_args(args)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    device = torch.device(args.device)

    # dataset
    dataset, _ = get_dataset_and_loader(args, trans_list=trans_list, only_test=False)
    ds = dataset[args.split]
    indices = list(range(len(ds)))

    # optional: split subset for weight tuning
    val_idx = None
    test_idx = indices
    if args.val_ratio_for_weights and args.val_ratio_for_weights > 0:
        val_idx, test_idx = split_indices(len(ds), args.val_ratio_for_weights, args.val_seed)

    # load scores per model
    scores_by_model: Dict[str, np.ndarray] = {}
    labels_ref = None

    for ckpt in args.checkpoints:
        model = load_model_from_checkpoint(args, dataset, ckpt).to(device)
        # use full indices (to keep alignment), then slice
        scores_all, labels_all = infer_scores(
            model=model,
            dataset=ds,
            indices=indices,
            device=device,
            model_confidence=args.model_confidence,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_conf_score=args.use_conf_score,
        )
        name = os.path.basename(ckpt)
        scores_by_model[name] = scores_all

        if labels_ref is None:
            labels_ref = labels_all
        else:
            # sanity check
            if not np.array_equal(labels_ref, labels_all):
                raise RuntimeError("Labels mismatch across runs. Dataset indexing may be inconsistent.")

    labels = labels_ref
    labels_unique = np.unique(labels) if labels is not None else np.array([])
    use_gt_eval = bool(args.eval_with_gt) or (labels_unique.size < 2)

    if use_gt_eval:
        val_idx = None
        test_idx = indices
        scores_by_model = {k: -v for k, v in scores_by_model.items()}
        eval_mode = "gt_normality"
    else:
        eval_mode = "label_anomaly"
    y_true = (labels != int(args.normal_label)).astype(np.int32) if labels is not None else None

    # slice val/test
    def take(arr, idxs):
        return arr if idxs is None else arr[np.array(idxs, dtype=np.int64)]

    y_test = take(y_true, test_idx) if y_true is not None else None
    model_names = list(scores_by_model.keys())
    scores_test_list = [take(scores_by_model[n], test_idx) for n in model_names]

    # base: report each model
    per_model = {}
    for n, s in zip(model_names, scores_test_list):
        if use_gt_eval:
            per_model[n] = eval_metrics_with_gt(s, ds, args)
        else:
            m = compute_auc_metrics(y_test, s)
            f = best_f1(y_test, s)
            per_model[n] = {**m, **f}

    # ensembles
    ensembles = {}
    # rank-average
    s_rank = rank_average(scores_test_list)
    if use_gt_eval:
        ensembles["rank_avg"] = eval_metrics_with_gt(s_rank, ds, args)
    else:
        ensembles["rank_avg"] = {**compute_auc_metrics(y_test, s_rank), **best_f1(y_test, s_rank)}

    # z-mean
    s_z = zscore_mean(scores_test_list)
    if use_gt_eval:
        ensembles["z_mean"] = eval_metrics_with_gt(s_z, ds, args)
    else:
        ensembles["z_mean"] = {**compute_auc_metrics(y_test, s_z), **best_f1(y_test, s_z)}

    # max
    s_max = max_fusion(scores_test_list)
    if use_gt_eval:
        ensembles["max"] = eval_metrics_with_gt(s_max, ds, args)
    else:
        ensembles["max"] = {**compute_auc_metrics(y_test, s_max), **best_f1(y_test, s_max)}

    # optional: weight tuning on val
    tuned = None
    if (not use_gt_eval) and val_idx is not None and len(args.checkpoints) == 3:
        y_val = y_true[np.array(val_idx, dtype=np.int64)]
        scores_val_list = [scores_by_model[n][np.array(val_idx, dtype=np.int64)] for n in model_names]
        best_w, best_val_metrics = grid_search_weights(
            y_val=y_val,
            scores_val_list=scores_val_list,
            step=args.weight_grid_step,
        )
        tuned = {"weights": dict(zip(model_names, best_w)), "val_metrics": best_val_metrics}

        s_w = weighted_mean(scores_test_list, best_w)
        ensembles["weighted_mean"] = {**compute_auc_metrics(y_test, s_w), **best_f1(y_test, s_w)}
    else:
        ensembles["weighted_mean"] = {"auc": float("nan"), "auprc": float("nan"), "f1": float("nan"), "thr": float("nan")}

    out = {
        "meta": {
            "split": args.split,
            "num_samples": int(len(test_idx)),
            "use_conf_score": bool(args.use_conf_score),
            "normal_label": int(args.normal_label),
            "val_ratio_for_weights": float(args.val_ratio_for_weights),
            "weight_grid_step": float(args.weight_grid_step),
            "checkpoints": args.checkpoints,
            "model_names": model_names,
            "eval_mode": eval_mode,
        },
        "per_model": per_model,
        "ensembles": ensembles,
        "tuned": tuned,
    }

    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()

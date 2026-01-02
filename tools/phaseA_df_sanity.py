#!/usr/bin/env python3
import os
import sys
import json
import pickle
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.data_utils import trans_list
from utils.scoring_utils import get_dataset_scores, smooth_scores
from utils.unlearning_utils import (
    load_model_from_checkpoint,
    prepare_batch,
    reduce_conf_score,
    select_df1_tail,
    select_df2_dynamics_v2,
    select_df2_val_fp_train_df,
    select_df3_subdomain_train_df,
    select_df3_grad_alignment_train_df,
)


def sample_val_sids(all_sids, ratio, seed):
    if ratio <= 0:
        return list(all_sids)
    count = max(1, int(len(all_sids) * ratio))
    if count >= len(all_sids):
        return list(all_sids)
    rng = random.Random(seed)
    return rng.sample(list(all_sids), count)


def filter_normals(dataset, sids, normal_label=1):
    out = []
    if dataset is None:
        return out
    if hasattr(dataset, "labels"):
        labs = dataset.labels
        for sid in sids:
            if int(labs[int(sid)]) == int(normal_label):
                out.append(int(sid))
        return out
    for sid in sids:
        label = dataset[int(sid)][-1]
        if torch.is_tensor(label):
            label = int(label.item())
        else:
            label = int(label)
        if label == int(normal_label):
            out.append(int(sid))
    return out


def sb_stats(sb_map: Dict[int, float], sids: List[int], tau_base: float) -> Dict[str, float]:
    if not sids:
        return {"n": 0}
    arr = np.array([float(sb_map[int(s)]) for s in sids if int(s) in sb_map], dtype=np.float32)
    if arr.size == 0:
        return {"n": int(len(sids)), "note": "no sB found"}
    qs = np.quantile(arr, [0.0, 0.5, 0.9, 0.99, 1.0]).tolist()
    frac_ge_tau = float(np.mean(arr >= float(tau_base)))
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(qs[0]),
        "p50": float(qs[1]),
        "p90": float(qs[2]),
        "p99": float(qs[3]),
        "max": float(qs[4]),
        "frac_ge_tau_base": float(frac_ge_tau),
    }

@torch.no_grad()
def infer_normality_scores_dataset(
    model,
    dataset,
    device: torch.device,
    model_confidence: bool,
    batch_size: int,
    num_workers: int,
    use_conf_score: bool,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model.eval()
    scores = []
    for batch in loader:
        x, score, label = prepare_batch(batch, device, model_confidence)
        _, nll = model(x, label=label)
        if use_conf_score:
            nll = nll * reduce_conf_score(score)
        scores.append((-1.0 * nll).detach().cpu())
    if not scores:
        return np.zeros((0,), dtype=np.float32)
    return torch.cat(scores, dim=0).view(-1).numpy().astype(np.float32, copy=False)


def _frame_scores_and_gt(normality_scores: np.ndarray, metadata, ref_args):
    gt_arr, scores_arr = get_dataset_scores(normality_scores, metadata, args=ref_args)
    scores_arr = smooth_scores(scores_arr)
    gt_np = np.concatenate(gt_arr) if len(gt_arr) else np.zeros((0,), dtype=np.int64)
    sc_np = np.concatenate(scores_arr) if len(scores_arr) else np.zeros((0,), dtype=np.float32)
    if sc_np.size:
        if np.isposinf(sc_np).any():
            sc_np[np.isposinf(sc_np)] = np.max(sc_np[~np.isposinf(sc_np)])
        if np.isneginf(sc_np).any():
            sc_np[np.isneginf(sc_np)] = np.min(sc_np[~np.isneginf(sc_np)])
    return gt_np, sc_np


def _pairwise_corr(scores_by_model: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    keys = list(scores_by_model.keys())
    if not keys:
        return {}
    mats = {k: np.asarray(scores_by_model[k]) for k in keys}
    mask = np.ones_like(mats[keys[0]], dtype=bool)
    for k in keys:
        mask &= np.isfinite(mats[k])
    out = {}
    for a in keys:
        out[a] = {}
        xa = mats[a][mask]
        for b in keys:
            xb = mats[b][mask]
            if xa.size == 0 or xb.size == 0:
                out[a][b] = float("nan")
                continue
            out[a][b] = float(np.corrcoef(xa, xb)[0, 1])
    return out


def _pairwise_spearman(scores_by_model: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    from scipy.stats import spearmanr
    keys = list(scores_by_model.keys())
    if not keys:
        return {}
    mats = {k: np.asarray(scores_by_model[k]) for k in keys}
    mask = np.ones_like(mats[keys[0]], dtype=bool)
    for k in keys:
        mask &= np.isfinite(mats[k])
    out = {}
    for a in keys:
        out[a] = {}
        xa = mats[a][mask]
        for b in keys:
            xb = mats[b][mask]
            if xa.size == 0 or xb.size == 0:
                out[a][b] = float("nan")
                continue
            out[a][b] = float(spearmanr(xa, xb).correlation)
    return out


def _anomaly_top_jaccard(scores_by_model: Dict[str, np.ndarray], gt_np: np.ndarray, top_pct: float):
    keys = list(scores_by_model.keys())
    out = {}
    if gt_np.size == 0:
        return out, {}
    ab_idx = np.where(gt_np == 0)[0]
    if ab_idx.size == 0:
        return out, {}
    frac = max(0.0, min(100.0, float(top_pct))) / 100.0
    k = max(1, int(len(ab_idx) * frac))

    sets = {}
    for kname in keys:
        s = scores_by_model[kname][ab_idx]
        order = np.argsort(s)  # low normality => more anomalous
        top_ids = ab_idx[order[:k]]
        sets[kname] = set(map(int, top_ids))

    for a in keys:
        out[a] = {}
        A = sets[a]
        for b in keys:
            B = sets[b]
            if len(A) == 0 or len(B) == 0:
                out[a][b] = 0.0
                continue
            inter = len(A & B)
            union = len(A | B)
            out[a][b] = float(inter / max(1, union))
    return out, {kname: len(sets[kname]) for kname in keys}


def overlap_table(df_sets: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    keys = list(df_sets.keys())
    sets = {k: set(map(int, v)) for k, v in df_sets.items()}
    out = {}
    for a in keys:
        out[a] = {}
        A = sets[a]
        for b in keys:
            B = sets[b]
            if len(A) == 0 or len(B) == 0:
                out[a][b] = 0.0
                continue
            inter = len(A & B)
            union = len(A | B)
            out[a][b] = float(inter / max(1, union))
    return out


def containment_table(df_sets: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    keys = list(df_sets.keys())
    sets = {k: set(map(int, v)) for k, v in df_sets.items()}
    out = {}
    for a in keys:
        out[a] = {}
        A = sets[a]
        for b in keys:
            B = sets[b]
            if len(A) == 0:
                out[a][b] = 0.0
                continue
            inter = len(A & B)
            out[a][b] = float(inter / max(1, len(A)))
    return out


def parse_args():
    p = init_parser()
    p.add_argument("--cache_path", type=str, required=True)
    p.add_argument("--cache_path_train", type=str, default=None)
    p.add_argument("--cache_path_val", type=str, default=None)

    p.add_argument("--output_dir", type=str, default="outputs/phaseA_df")

    p.add_argument(
        "--df_list",
        type=str,
        default="df1,df2p,df3p",
        help="comma-separated: df1,df2,df2p,df3,df3p",
    )
    p.add_argument(
        "--df_paths",
        nargs="+",
        default=None,
        help="paths to pkl/txt with DF sids; if set, load these instead of selecting",
    )
    p.add_argument(
        "--df_names",
        type=str,
        default=None,
        help="comma-separated names for --df_paths (optional, default: filename stem)",
    )

    p.add_argument("--alpha_df1", type=float, default=0.01)

    p.add_argument("--alpha_df2", type=float, default=0.02)
    p.add_argument("--df2_mode", type=str, default="normalized", choices=["raw", "normalized", "relative"])
    p.add_argument("--df2_robust", action="store_true")
    p.add_argument("--df2_trim_ratio", type=float, default=0.1)
    p.add_argument("--df2_eps", type=float, default=1e-6)

    p.add_argument("--k_df3", type=int, default=20)
    p.add_argument("--top_clusters", type=int, default=2)
    p.add_argument("--tau_q", type=float, default=0.999)

    p.add_argument("--df2p_knn_k", type=int, default=20)
    p.add_argument("--df2p_budget_alpha", type=float, default=0.02)
    p.add_argument("--df2p_budget_max", type=int, default=0)
    p.add_argument("--df2p_sb_max_q", type=float, default=0.95)
    p.add_argument("--df2p_sb_penalty", type=float, default=0.2)
    p.add_argument("--normal_label", type=int, default=1)

    p.add_argument("--df3p_alpha", type=float, default=0.02)
    p.add_argument("--df3p_sample_train", type=int, default=4096)
    p.add_argument("--df3p_sample_val", type=int, default=1024)
    p.add_argument("--df3p_include_regex", type=str, default="prior")
    p.add_argument("--df3p_exclude_regex", type=str, default="")
    p.add_argument("--df3p_max_val_batches", type=int, default=200)

    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--val_seed", type=int, default=0)
    p.add_argument("--tau_from_all_val", action="store_true")

    p.add_argument("--dump_sids", action="store_true")
    p.add_argument("--dump_format", type=str, default="pkl", choices=["pkl", "txt"])
    p.add_argument("--diversity_checkpoints", nargs="+", default=None,
                   help="ckpt paths for score diversity check")
    p.add_argument("--diversity_top_pct", type=float, default=1.0,
                   help="top %% among anomaly frames for Jaccard (default 1.0)")
    p.add_argument("--diversity_batch_size", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args, _ = init_sub_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    cache_path_train = args.cache_path_train or args.cache_path
    cache_path_val = args.cache_path_val or args.cache_path

    with open(cache_path_train, "rb") as f:
        cache_train = pickle.load(f)
    with open(cache_path_val, "rb") as f:
        cache_val = pickle.load(f)

    normal_sids = sorted(cache_train["sB"].keys())
    val_candidates = sorted(cache_val["sB"].keys())
    val_split = cache_val.get("meta", {}).get("split", "train")

    random.seed(args.val_seed)
    np.random.seed(args.val_seed)
    torch.manual_seed(args.val_seed)

    dataset, _ = get_dataset_and_loader(args, trans_list=trans_list, only_test=False)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"] if val_split == "test" else dataset["train"]

    val_sids = sample_val_sids(val_candidates, args.val_ratio, args.val_seed)
    val_norm_sids = filter_normals(val_dataset, val_sids, normal_label=args.normal_label)
    if not val_norm_sids:
        val_norm_sids = list(val_sids)

    val_norm_candidates = filter_normals(val_dataset, val_candidates, normal_label=args.normal_label)
    if not val_norm_candidates:
        val_norm_candidates = list(val_candidates)

    if args.tau_from_all_val:
        s_val_all = np.array([cache_val["sB"][sid] for sid in val_norm_candidates], dtype=np.float32)
        tau_base = float(np.quantile(s_val_all, args.tau_q))
    else:
        s_val_sub = np.array([cache_val["sB"][sid] for sid in val_norm_sids], dtype=np.float32)
        tau_base = float(np.quantile(s_val_sub, args.tau_q))

    s_val_monitor = np.array([cache_val["sB"][sid] for sid in val_norm_sids], dtype=np.float32)
    fpr_base = float(np.mean(s_val_monitor >= tau_base))
    fpr_base = max(fpr_base, 1.0 / max(1, len(val_norm_sids)))

    device = torch.device(args.device)

    df_sets: Dict[str, List[int]] = {}
    df_infos: Dict[str, Dict] = {}
    warnings = []

    df_list = [x.strip() for x in args.df_list.split(",") if x.strip()]
    if args.df_paths:
        paths = args.df_paths
        if len(paths) == 1 and "," in paths[0]:
            paths = [p.strip() for p in paths[0].split(",") if p.strip()]
        if args.df_names:
            names = [n.strip() for n in args.df_names.split(",") if n.strip()]
        else:
            names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        if len(names) != len(paths):
            raise ValueError("--df_names length must match --df_paths length.")

        df_list = list(names)
        for name, path in zip(names, paths):
            if path.endswith(".pkl"):
                with open(path, "rb") as f:
                    df = pickle.load(f)
            elif path.endswith(".txt"):
                with open(path, "r") as f:
                    df = [int(line.strip()) for line in f if line.strip()]
            else:
                raise ValueError(f"Unsupported DF file: {path}")
            df = sorted(list(set(map(int, df))))
            df_sets[name] = df
            df_infos[name] = {"source": path}
    else:
        need_model = ("df3p" in df_list)
        model = None
        if need_model:
            if args.checkpoint is None:
                raise ValueError("df3p requires --checkpoint")
            model = load_model_from_checkpoint(args, dataset, args.checkpoint).to(device)

        for name in df_list:
            if name == "df1":
                df = sorted(list(select_df1_tail(cache_train["sB"], alpha=args.alpha_df1)))
                info = {"alpha_df1": float(args.alpha_df1)}
            elif name == "df2":
                df = sorted(list(select_df2_dynamics_v2(
                    cache_train,
                    alpha_g=args.alpha_df2,
                    mode=args.df2_mode,
                    robust=bool(args.df2_robust),
                    trim_ratio=float(args.df2_trim_ratio),
                    eps=float(args.df2_eps),
                )))
                info = {
                    "alpha_df2": float(args.alpha_df2),
                    "df2_mode": args.df2_mode,
                    "df2_robust": bool(args.df2_robust),
                    "df2_trim_ratio": float(args.df2_trim_ratio),
                    "df2_eps": float(args.df2_eps),
                }
            elif name == "df2p":
                budget_max = None if args.df2p_budget_max <= 0 else int(args.df2p_budget_max)
                df, info = select_df2_val_fp_train_df(
                    cache_train=cache_train,
                    train_sids=normal_sids,
                    cache_val=cache_val,
                    val_sids=val_sids,
                    val_dataset=val_dataset,
                    tau_base=tau_base,
                    normal_label=int(args.normal_label),
                    knn_k=int(args.df2p_knn_k),
                    budget_alpha=float(args.df2p_budget_alpha),
                    budget_max=budget_max,
                    df2p_sb_max_q=float(args.df2p_sb_max_q),
                    df2p_sb_penalty=float(args.df2p_sb_penalty),
                    seed=int(args.val_seed),
                )
                df = sorted(list(set(df)))
                fp_n = int(info.get("num_fp_val", 0))
                knn_k = int(info.get("knn_k", 0))
                budget = int(info.get("budget", 0))
                fp_pool = fp_n * knn_k
                if fp_pool <= 0:
                    warnings.append("df2p: fp_pool is empty (num_fp_val * knn_k == 0).")
                elif budget > fp_pool:
                    warnings.append(
                        f"df2p: budget({budget}) > fp_pool({fp_pool}). "
                        "Reduce budget_alpha or increase knn_k."
                    )
                elif fp_pool > 0 and budget >= int(0.5 * fp_pool):
                    warnings.append(
                        f"df2p: budget({budget}) is close to fp_pool({fp_pool}); "
                        "DF may be overly saturated."
                    )
            elif name == "df3":
                df, info = select_df3_subdomain_train_df(
                    cache_train, normal_sids, cache_val, val_sids,
                    k_clusters=int(args.k_df3),
                    top_clusters=int(args.top_clusters),
                    tau_q=float(args.tau_q),
                    return_cluster_ctx=False,
                )
                df = sorted(list(set(df)))
            elif name == "df3p":
                ex = args.df3p_exclude_regex.strip() or None
                df, info = select_df3_grad_alignment_train_df(
                    base_model=model,
                    train_dataset=train_dataset,
                    train_sids=normal_sids,
                    val_dataset=val_dataset,
                    val_sids=val_sids,
                    device=device,
                    model_confidence=bool(args.model_confidence),
                    use_conf_score_val=bool(args.model_confidence),
                    use_conf_score_train=bool(args.model_confidence),
                    normal_label=int(args.normal_label),
                    alpha=float(args.df3p_alpha),
                    sample_train=int(args.df3p_sample_train),
                    sample_val=int(args.df3p_sample_val),
                    seed=int(args.val_seed),
                    batch_size=1,
                    num_workers=int(args.num_workers),
                    include_regex=str(args.df3p_include_regex),
                    exclude_regex=ex,
                    max_val_batches=int(args.df3p_max_val_batches),
                )
                df = sorted(list(set(df)))
            else:
                raise ValueError(f"Unknown df name: {name}")

            df_sets[name] = df
            df_infos[name] = info

            if args.dump_sids:
                if args.dump_format == "pkl":
                    outp = os.path.join(args.output_dir, f"{name}_sids.pkl")
                    with open(outp, "wb") as f:
                        pickle.dump(df, f)
                else:
                    outp = os.path.join(args.output_dir, f"{name}_sids.txt")
                    with open(outp, "w") as f:
                        for sid in df:
                            f.write(f"{int(sid)}\n")

    per_df_stats = {}
    for name, sids in df_sets.items():
        per_df_stats[name] = {
            "size": int(len(sids)),
            "sB_stats_train": sb_stats(cache_train["sB"], sids, tau_base=tau_base),
        }

    if "df1" in per_df_stats and "df2p" in per_df_stats:
        s1 = per_df_stats["df1"].get("sB_stats_train", {})
        s2 = per_df_stats["df2p"].get("sB_stats_train", {})
        m1 = s1.get("mean")
        m2 = s2.get("mean")
        p1 = s1.get("p50")
        p2 = s2.get("p50")
        if m1 is not None and m2 is not None and m2 >= m1:
            warnings.append(f"df2p mean sB >= df1 mean sB ({m2:.6g} >= {m1:.6g})")
        if p1 is not None and p2 is not None and p2 >= p1:
            warnings.append(f"df2p median sB >= df1 median sB ({p2:.6g} >= {p1:.6g})")

    jac = overlap_table(df_sets)
    cont = containment_table(df_sets)

    overlap_warn_j = 0.5
    overlap_warn_c = 0.8
    for i, a in enumerate(df_list):
        for b in df_list[i + 1:]:
            if jac[a][b] >= overlap_warn_j:
                warnings.append(f"High Jaccard overlap: {a} vs {b} = {jac[a][b]:.3f}")
            if cont[a][b] >= overlap_warn_c:
                warnings.append(f"High containment: {a} in {b} = {cont[a][b]:.3f}")
            if cont[b][a] >= overlap_warn_c:
                warnings.append(f"High containment: {b} in {a} = {cont[b][a]:.3f}")

    diversity = None
    if args.diversity_checkpoints:
        if dataset.get("test") is None:
            raise ValueError("diversity_checkpoints requires test split.")
        ds_test = dataset["test"]
        test_metadata = ds_test.metadata
        batch_size = int(args.diversity_batch_size or args.batch_size)
        use_conf_score = bool(args.model_confidence)

        scores_by_model = {}
        gt_np = None
        for ckpt in args.diversity_checkpoints:
            model = load_model_from_checkpoint(args, dataset, ckpt).to(device)
            scores = infer_normality_scores_dataset(
                model=model,
                dataset=ds_test,
                device=device,
                model_confidence=bool(args.model_confidence),
                batch_size=batch_size,
                num_workers=int(args.num_workers),
                use_conf_score=use_conf_score,
            )
            name = os.path.basename(ckpt)
            gt_here, frame_scores = _frame_scores_and_gt(scores, test_metadata, args)
            if gt_np is None:
                gt_np = gt_here
            scores_by_model[name] = frame_scores

        pearson = _pairwise_corr(scores_by_model)
        spearman = _pairwise_spearman(scores_by_model)
        jaccard, counts = _anomaly_top_jaccard(scores_by_model, gt_np, top_pct=args.diversity_top_pct)
        diversity = {
            "pearson": pearson,
            "spearman": spearman,
            "anomaly_top_pct": float(args.diversity_top_pct),
            "anomaly_top_jaccard": jaccard,
            "anomaly_top_counts": counts,
        }

        for a in pearson:
            for b in pearson[a]:
                if a != b and pearson[a][b] >= 0.98:
                    warnings.append(f"High score correlation (pearson): {a} vs {b} = {pearson[a][b]:.3f}")
        for a in jaccard:
            for b in jaccard[a]:
                if a != b and jaccard[a][b] >= 0.8:
                    warnings.append(f"High anomaly Jaccard: {a} vs {b} = {jaccard[a][b]:.3f}")

    report = {
        "meta": {
            "val_split": val_split,
            "val_seed": int(args.val_seed),
            "val_ratio": float(args.val_ratio),
            "normal_label": int(args.normal_label),
            "tau_q": float(args.tau_q),
            "tau_from_all_val": bool(args.tau_from_all_val),
            "tau_base": float(tau_base),
            "fpr_base": float(fpr_base),
            "df_list": df_list,
        },
        "df_info": df_infos,
        "df_stats": per_df_stats,
        "overlap": {
            "jaccard": jac,
            "containment": cont,
        },
        "diversity": diversity,
        "warnings": warnings,
    }

    out_json = os.path.join(args.output_dir, "phaseA_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("[Phase A] DF sanity report")
    print(f"- tau_base={tau_base:.6f}  fpr_base={fpr_base:.6f}  val_split={val_split}")
    print("- df sizes:", {k: len(v) for k, v in df_sets.items()})
    print("- wrote:", out_json)
    if warnings:
        print("\n[Warnings]")
        for w in warnings:
            print("-", w)
    print("\n[Containment |A∩B|/|A|] (rows=A, cols=B)")
    for a in df_list:
        row = {b: round(cont[a][b], 4) for b in df_list}
        print(a, row)
    print("\n[Jaccard |A∩B|/|A∪B|] (rows=A, cols=B)")
    for a in df_list:
        row = {b: round(jac[a][b], 4) for b in df_list}
        print(a, row)
    print("=" * 80)


if __name__ == "__main__":
    main()

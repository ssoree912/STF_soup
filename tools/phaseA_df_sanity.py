#!/usr/bin/env python3
import os
import sys
import json
import pickle
import random
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.data_utils import trans_list
from utils.unlearning_utils import (
    load_model_from_checkpoint,
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

    df_list = [x.strip() for x in args.df_list.split(",") if x.strip()]
    need_model = ("df3p" in df_list)
    model = None
    if need_model:
        if args.checkpoint is None:
            raise ValueError("df3p requires --checkpoint")
        model = load_model_from_checkpoint(args, dataset, args.checkpoint).to(device)

    df_sets: Dict[str, List[int]] = {}
    df_infos: Dict[str, Dict] = {}
    warnings = []

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

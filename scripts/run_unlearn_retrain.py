#!/usr/bin/env python3
import argparse
import math
import json
import os
import pickle
import random
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from models.STG_NF.modules_pose import gaussian_likelihood, gaussian_p
from utils.data_utils import trans_list
from utils.unlearning_utils import (
    build_indexed_loader,
    eval_nll_on_indices,
    load_model_from_checkpoint,
    prepare_batch,
    reduce_conf_score,
    select_df1_tail,
    select_df2_dynamics_v2,
    select_df2_val_fp_train_df,
    select_df3_grad_alignment_train_df,
    select_df3_subdomain_train_df,
)
from utils.scoring_utils import score_dataset

# -----------------------
# helpers
# -----------------------
def sample_val_sids(all_sids, ratio, seed):
    if ratio <= 0:
        return list(all_sids)
    count = max(1, int(len(all_sids) * ratio))
    if count >= len(all_sids):
        return list(all_sids)
    rng = random.Random(seed)
    return rng.sample(list(all_sids), count)

def _jsonable(val):
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if isinstance(val, (list, tuple)):
        return [_jsonable(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _jsonable(v) for k, v in val.items()}
    return str(val)

def _serialize_args(args):
    return {k: _jsonable(v) for k, v in vars(args).items()}

def filter_val_normals(val_dataset, sids, normal_label=1):
    out = []
    if val_dataset is None:
        return out
    if hasattr(val_dataset, "labels"):
        labs = val_dataset.labels
        for sid in sids:
            if int(labs[int(sid)]) == int(normal_label):
                out.append(int(sid))
        return out
    for sid in sids:
        label = val_dataset[int(sid)][-1]
        if torch.is_tensor(label):
            label = int(label.item())
        else:
            label = int(label)
        if label == int(normal_label):
            out.append(int(sid))
    return out

def forward_nll_components(model, x, label):
    z, logdet = model.flow(x, reverse=False)
    mean, logs = model.prior(x, label)
    objective = logdet + gaussian_likelihood(mean, logs, z)
    denom = math.log(2.0) * x.size(1) * x.size(2) * x.size(3)
    nll = (-objective) / denom
    return z, logdet, mean, logs, nll, denom

def compute_nll_steps(z, logdet, mean, logs, denom):
    logp = gaussian_p(mean, logs, z)
    logp_t = logp.sum(dim=(1, 3))
    logdet_t = (logdet / z.size(2)).unsqueeze(1).expand_as(logp_t)
    return -(logp_t + logdet_t) / denom

def dynamics_metric_from_zsteps(z_steps, mode="normalized", eps=1e-6):
    dz = z_steps[:, 1:] - z_steps[:, :-1]
    dz_norm = torch.linalg.norm(dz, dim=-1)
    if mode == "normalized":
        dz_norm = dz_norm / max(1.0, math.sqrt(z_steps.size(-1)))
    elif mode == "relative":
        base = torch.linalg.norm(z_steps[:, :-1], dim=-1) + eps
        dz_norm = dz_norm / base
    return dz_norm.mean(dim=1)

def dynamics_metric_from_zsteps_np(z_steps, mode="normalized", eps=1e-6):
    dz = z_steps[1:] - z_steps[:-1]
    dz_norm = np.linalg.norm(dz, axis=-1)
    if mode == "normalized":
        dz_norm = dz_norm / max(1.0, math.sqrt(z_steps.shape[-1]))
    elif mode == "relative":
        base = np.linalg.norm(z_steps[:-1], axis=-1) + eps
        dz_norm = dz_norm / base
    return float(dz_norm.mean())

def compute_df2_base_metric(cache_df, df_sids, args):
    if args.df2_unlearn_loss == "zsteps_delta":
        z_cache = cache_df.get("z_steps", {})
        vals = []
        for sid in df_sids:
            z_steps = z_cache.get(int(sid))
            if z_steps is None:
                continue
            vals.append(dynamics_metric_from_zsteps_np(z_steps, mode=args.df2_mode, eps=args.df2_eps))
        return float(np.mean(vals)) if vals else float("nan")
    if args.df2_unlearn_loss == "nll_var":
        nll_cache = cache_df.get("nll_steps", {})
        vals = []
        for sid in df_sids:
            nll_steps = nll_cache.get(int(sid))
            if nll_steps is None:
                continue
            vals.append(float(np.var(nll_steps)))
        return float(np.mean(vals)) if vals else float("nan")
    return float("nan")

def compute_df3_base_metric(cache_df, df_sids, args, df3_ctx):
    emb_cache = cache_df.get("emb", {})
    if args.df3_unlearn_loss == "emb_norm":
        vals = []
        for sid in df_sids:
            emb = emb_cache.get(int(sid))
            if emb is None:
                continue
            vals.append(float(np.sum(emb ** 2)))
        return float(np.mean(vals)) if vals else float("nan")
    if args.df3_unlearn_loss == "cluster_dist":
        if df3_ctx is None:
            return float("nan")
        centers = df3_ctx.get("centers")
        sid_to_cluster = df3_ctx.get("sid_to_cluster", {})
        vals = []
        for sid in df_sids:
            emb = emb_cache.get(int(sid))
            cidx = sid_to_cluster.get(int(sid))
            if emb is None or cidx is None:
                continue
            diff = emb - centers[int(cidx)]
            vals.append(float(np.sum(diff ** 2)))
        return float(np.mean(vals)) if vals else float("nan")
    return float("nan")

def compute_unlearn_score(df_name, z, logdet, mean, logs, nll, denom, batch_sids, df3_ctx_torch, args):
    # df1/df2p/df3p: nll maximize
    if df_name in ("df1", "df2p", "df3p"):
        return nll, True

    # df2: dynamic
    if df_name == "df2":
        if args.df2_unlearn_loss == "nll":
            return nll, True
        if args.df2_unlearn_loss == "zsteps_delta":
            z_steps = z.permute(0, 2, 1, 3).reshape(z.size(0), z.size(2), -1)
            dynamics = dynamics_metric_from_zsteps(z_steps, mode=args.df2_mode, eps=args.df2_eps)
            return -dynamics, False
        if args.df2_unlearn_loss == "nll_var":
            nll_steps = compute_nll_steps(z, logdet, mean, logs, denom)
            return -nll_steps.var(dim=1, unbiased=False), False

    # df3: embedding
    if df_name == "df3":
        if args.df3_unlearn_loss == "nll":
            return nll, True
        emb = z.mean(dim=2).reshape(z.size(0), -1)
        if args.df3_unlearn_loss == "emb_norm":
            return (emb ** 2).sum(dim=1), False
        if args.df3_unlearn_loss == "cluster_dist":
            if df3_ctx_torch is None:
                raise ValueError("df3_unlearn_loss=cluster_dist requires cluster ctx.")
            centers = df3_ctx_torch["centers"]
            sid_to_cluster = df3_ctx_torch["sid_to_cluster"]
            cluster_idx = [sid_to_cluster[int(sid)] for sid in batch_sids]
            idx = torch.tensor(cluster_idx, device=emb.device, dtype=torch.long)
            center_batch = centers[idx]
            dist2 = (emb - center_batch).pow(2).sum(dim=1)
            return dist2, False

    raise ValueError(f"Unsupported df_name={df_name}")

@torch.no_grad()
def eval_nll_and_aux_on_indices(model, dataset, indices, device, model_confidence,
                               batch_size, num_workers, use_conf_score,
                               df_name, args, df3_ctx_np):
    loader = build_indexed_loader(dataset, indices, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    model.eval()

    nll_list, aux_list = [], []

    centers = None
    sid_to_cluster = None
    if df_name == "df3" and args.df3_unlearn_loss == "cluster_dist" and df3_ctx_np is not None:
        centers = df3_ctx_np.get("centers")
        sid_to_cluster = df3_ctx_np.get("sid_to_cluster", {})

    for batch in loader:
        x, score, label = prepare_batch(batch, device, model_confidence)
        z, logdet, mean, logs, nll, denom = forward_nll_components(model, x, label)

        if use_conf_score:
            nll = nll * reduce_conf_score(score)
        nll_list.append(nll.detach().cpu())

        aux_val = None
        if df_name == "df2":
            if args.df2_unlearn_loss == "zsteps_delta":
                z_steps = z.permute(0, 2, 1, 3).reshape(z.size(0), z.size(2), -1)
                aux_val = dynamics_metric_from_zsteps(z_steps, mode=args.df2_mode, eps=args.df2_eps)
            elif args.df2_unlearn_loss == "nll_var":
                nll_steps = compute_nll_steps(z, logdet, mean, logs, denom)
                aux_val = nll_steps.var(dim=1, unbiased=False)
        elif df_name == "df3":
            emb = z.mean(dim=2).reshape(z.size(0), -1)
            if args.df3_unlearn_loss == "emb_norm":
                aux_val = (emb ** 2).sum(dim=1)
            elif args.df3_unlearn_loss == "cluster_dist":
                if centers is None:
                    raise ValueError("cluster_dist needs df3_ctx_np")
                sids = batch[1].tolist() if torch.is_tensor(batch[1]) else list(batch[1])
                cluster_idx = [sid_to_cluster[int(sid)] for sid in sids]
                idx = torch.tensor(cluster_idx, device=emb.device, dtype=torch.long)
                center_batch = torch.tensor(centers, device=emb.device)[idx]
                aux_val = (emb - center_batch).pow(2).sum(dim=1)

        if aux_val is not None:
            aux_list.append(aux_val.detach().cpu())

    nll_arr = torch.cat(nll_list, dim=0).numpy() if nll_list else np.array([])
    aux_mean = float(torch.cat(aux_list, dim=0).mean().item()) if aux_list else None
    return nll_arr, aux_mean

def compute_single_metrics(model, train_dataset, val_dataset, df_sids, val_sids,
                           cache_train, cache_val, tau_base, device,
                           args, use_conf_score_df, use_conf_score_val, df_name, df3_ctx_np):
    # df NLL / aux
    if df_name in ("df2", "df3") and (
        (df_name == "df2" and args.df2_unlearn_loss != "nll")
        or (df_name == "df3" and args.df3_unlearn_loss != "nll")
    ):
        df_nll, aux_mean = eval_nll_and_aux_on_indices(
            model, train_dataset, df_sids, device, args.model_confidence,
            args.batch_size_retrain, args.num_workers, use_conf_score_df,
            df_name, args, df3_ctx_np
        )
    else:
        df_nll = eval_nll_on_indices(
            model, train_dataset, df_sids, device=device,
            model_confidence=args.model_confidence,
            batch_size=args.batch_size_retrain, num_workers=args.num_workers,
            use_conf_score=use_conf_score_df,
        )
        aux_mean = None

    val_nll = eval_nll_on_indices(
        model, val_dataset, val_sids, device=device,
        model_confidence=args.model_confidence,
        batch_size=args.batch_size_retrain, num_workers=args.num_workers,
        use_conf_score=use_conf_score_val,
    )

    base_df = float(np.mean([cache_train["sB"][sid] for sid in df_sids])) if df_sids else 0.0
    base_val = float(np.mean([cache_val["sB"][sid] for sid in val_sids])) if val_sids else 0.0

    out = {
        "df_nll_mean": float(df_nll.mean()) if df_nll.size else 0.0,
        "val_nll_mean": float(val_nll.mean()) if val_nll.size else 0.0,
        #df 데이터의 nll 변화량
        "delta_df": float(df_nll.mean() - base_df) if df_nll.size else 0.0,
        #nllcurrr - nllbase : 검증 데이터의 nll 변화량
        "delta_val": float(val_nll.mean() - base_val) if val_nll.size else 0.0,
        "fpr_val": float(np.mean(val_nll >= tau_base)) if val_nll.size else 0.0,
    }

    if df_name == "df2" and args.df2_unlearn_loss != "nll":
        base_aux = compute_df2_base_metric(cache_train, df_sids, args)
        if aux_mean is not None:
            out["df2_metric_name"] = args.df2_unlearn_loss
            out["df2_metric_base"] = float(base_aux)
            out["df2_metric_mean"] = float(aux_mean)
            out["df2_metric_delta"] = float(aux_mean - base_aux)
    if df_name == "df3" and args.df3_unlearn_loss != "nll":
        base_aux = compute_df3_base_metric(cache_train, df_sids, args, df3_ctx_np)
        if aux_mean is not None:
            out["df3_metric_name"] = args.df3_unlearn_loss
            out["df3_metric_base"] = float(base_aux)
            out["df3_metric_mean"] = float(aux_mean)
            out["df3_metric_delta"] = float(aux_mean - base_aux)
    return out


@torch.no_grad()
def eval_roc_auc(model, test_loader, test_metadata, args, device):
    if test_loader is None or test_metadata is None:
        return None
    was_training = model.training
    model.eval()
    scores = []
    for batch in test_loader:
        x, score, _ = prepare_batch(batch, device, args.model_confidence)
        label = torch.ones(x.size(0), device=device)
        _, nll = model(x, label=label)
        if args.model_confidence:
            nll = nll * reduce_conf_score(score)
        scores.append((-1 * nll).detach().cpu())
    if was_training:
        model.train()
    if not scores:
        return None
    normality_scores = torch.cat(scores, dim=0).numpy().squeeze().copy(order="C")
    auc, _ = score_dataset(normality_scores, test_metadata, args=args)
    return float(auc)


def parse_args():
    parser = init_parser()
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--cache_path_train", type=str, default=None)
    parser.add_argument("--cache_path_val", type=str, default=None)

    # mode
    parser.add_argument("--mode", type=str, default="single", choices=["single", "mix"])
    parser.add_argument("--df_name", type=str, default="df1", choices=["df1", "df2", "df3", "df2p", "df3p"],
                        help="used when --mode single")

    parser.add_argument("--output_dir", type=str, default="outputs/unlearning")

    # DF selection hyperparams
    parser.add_argument("--alpha_df1", type=float, default=0.01)
    parser.add_argument("--alpha_df2", type=float, default=0.02)
    parser.add_argument("--df2_mode", type=str, default="normalized", choices=["raw", "normalized", "relative"])
    parser.add_argument("--df2_robust", action="store_true")
    parser.add_argument("--df2_trim_ratio", type=float, default=0.1)
    parser.add_argument("--df2_eps", type=float, default=1e-6)

    parser.add_argument("--k_df3", type=int, default=20)
    parser.add_argument("--top_clusters", type=int, default=2)
    parser.add_argument("--tau_q", type=float, default=0.999)
    parser.add_argument("--df2_unlearn_loss", type=str, default="zsteps_delta",
                        choices=["nll", "zsteps_delta", "nll_var"])
    parser.add_argument("--df3_unlearn_loss", type=str, default="emb_norm",
                        choices=["nll", "emb_norm", "cluster_dist"])

    # df2' (val FP) options
    parser.add_argument("--df2p_knn_k", type=int, default=20)
    parser.add_argument("--df2p_budget_alpha", type=float, default=0.02)
    parser.add_argument("--df2p_budget_max", type=int, default=0)
    parser.add_argument("--df2p_sb_max_q", type=float, default=0.95)
    parser.add_argument("--df2p_sb_penalty", type=float, default=0.2)
    parser.add_argument("--normal_label", type=int, default=1)

    # df3' (grad alignment) options
    parser.add_argument("--df3p_alpha", type=float, default=0.02)
    parser.add_argument("--df3p_sample_train", type=int, default=4096)
    parser.add_argument("--df3p_sample_val", type=int, default=1024)
    parser.add_argument("--df3p_include_regex", type=str, default="prior")
    parser.add_argument("--df3p_exclude_regex", type=str, default="")
    parser.add_argument("--df3p_max_val_batches", type=int, default=200)

    # val selection
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--val_seed", type=int, default=0)
    parser.add_argument("--tau_from_all_val", action="store_true")

    # training (single)
    parser.add_argument("--batch_size_unlearn", type=int, default=64)
    parser.add_argument("--batch_size_retrain", type=int, default=256)
    parser.add_argument("--lr_unlearn", type=float, default=3e-5)
    parser.add_argument("--steps_unlearn", type=int, default=200)
    parser.add_argument("--lr_retrain", type=float, default=3e-5)
    parser.add_argument("--epochs_retrain", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # DF dump
    parser.add_argument("--dump_df_sids", action="store_true")
    parser.add_argument("--dump_df_format", type=str, default="pkl", choices=["pkl", "txt"])

    # safety
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--fpr_tol", type=float, default=2.0)

    # DR sampling
    parser.add_argument("--dr_ratio", type=float, default=1.0)
    parser.add_argument("--dr_seed", type=int, default=0)

    # MIX는 "나중"에 쓸 거라 최소 옵션만 두고,
    # 네가 만든 mix 스크립트를 그대로 별도 파일로 써도 됨.
    # 필요하면 여기에도 옵션 추가 가능.
    return parser.parse_args()


def main():
    args = parse_args()
    args, _ = init_sub_args(args)

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.val_seed)
    np.random.seed(args.val_seed)
    torch.manual_seed(args.val_seed)

    cache_path_train = args.cache_path_train or args.cache_path
    cache_path_val = args.cache_path_val or args.cache_path

    with open(cache_path_train, "rb") as f:
        cache_train = pickle.load(f)
    with open(cache_path_val, "rb") as f:
        cache_val = pickle.load(f)

    normal_sids = sorted(cache_train["sB"].keys())
    val_candidates = sorted(cache_val["sB"].keys())

    val_split = cache_val.get("meta", {}).get("split", "train")

    # 1) dataset / scoring rule
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=False)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"] if val_split == "test" else dataset["train"]
    test_loader = loader.get("test")
    test_metadata = dataset.get("test").metadata if dataset.get("test") is not None else None

    # 2) tau_base / val_sids (normal-only)
    val_sids = sample_val_sids(val_candidates, args.val_ratio, args.val_seed)
    val_norm_sids = filter_val_normals(val_dataset, val_sids, normal_label=args.normal_label)
    if not val_norm_sids:
        val_norm_sids = list(val_sids)

    val_norm_candidates = filter_val_normals(val_dataset, val_candidates, normal_label=args.normal_label)
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

    use_conf_score_train = bool(args.model_confidence)
    use_conf_score_df = bool(cache_train.get("meta", {}).get("use_conf_score", use_conf_score_train))
    use_conf_score_val = bool(cache_val.get("meta", {}).get("use_conf_score", use_conf_score_train))

    # 3) load base model (df3p needs it)
    model = load_model_from_checkpoint(args, dataset, args.checkpoint).to(device)

    # 4) DF select (single이면 해당 df만 뽑음)
    df3_cluster_ctx_np = None
    df_info = {}
    if args.mode == "single":
        if args.df_name == "df1":
            df_sids = sorted(list(select_df1_tail(cache_train["sB"], alpha=args.alpha_df1)))
            df_info = {"alpha_df1": float(args.alpha_df1)}
        elif args.df_name == "df2":
            df_sids = sorted(list(select_df2_dynamics_v2(
                cache_train, alpha_g=args.alpha_df2, mode=args.df2_mode,
                robust=args.df2_robust, trim_ratio=args.df2_trim_ratio, eps=args.df2_eps
            )))
            df_info = {
                "alpha_df2": float(args.alpha_df2),
                "df2_mode": args.df2_mode,
                "df2_robust": bool(args.df2_robust),
                "df2_trim_ratio": float(args.df2_trim_ratio),
                "df2_eps": float(args.df2_eps),
            }
        elif args.df_name == "df2p":
            budget_max = None if args.df2p_budget_max <= 0 else int(args.df2p_budget_max)
            df_sids, df_info = select_df2_val_fp_train_df(
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
            df_sids = sorted(list(set(df_sids)))
        elif args.df_name == "df3":
            if args.df3_unlearn_loss == "cluster_dist":
                df_sids, info, df3_cluster_ctx_np = select_df3_subdomain_train_df(
                    cache_train, normal_sids, cache_val, val_sids,
                    k_clusters=args.k_df3, top_clusters=args.top_clusters,
                    tau_q=args.tau_q, return_cluster_ctx=True
                )
            else:
                df_sids, info = select_df3_subdomain_train_df(
                    cache_train, normal_sids, cache_val, val_sids,
                    k_clusters=args.k_df3, top_clusters=args.top_clusters, tau_q=args.tau_q
                )
            df_sids = sorted(list(df_sids))
            df_info = info
        elif args.df_name == "df3p":
            ex = args.df3p_exclude_regex.strip() or None
            df_sids, df_info = select_df3_grad_alignment_train_df(
                base_model=model,
                train_dataset=train_dataset,
                train_sids=normal_sids,
                val_dataset=val_dataset,
                val_sids=val_sids,
                device=device,
                model_confidence=bool(args.model_confidence),
                use_conf_score_val=use_conf_score_val,
                use_conf_score_train=use_conf_score_train,
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
            df_sids = sorted(list(set(df_sids)))
        else:
            raise ValueError(f"Unsupported df_name={args.df_name}")

        if not df_sids:
            raise ValueError(f"{args.df_name} selected empty set. df_info={df_info}")
    else:
        raise ValueError("For now, run mix with your dedicated mix script later. (mode=single recommended)")

    if args.dump_df_sids:
        dump_path = os.path.join(args.output_dir, f"{args.df_name}_sids.{args.dump_df_format}")
        if args.dump_df_format == "pkl":
            with open(dump_path, "wb") as f:
                pickle.dump(df_sids, f)
        else:
            with open(dump_path, "w") as f:
                for sid in df_sids:
                    f.write(f"{int(sid)}\n")
        print(f"[INFO] DF sids saved: {dump_path}")

    # 5) DR set
    df_set = set(df_sids)
    dr_sids = [sid for sid in normal_sids if sid not in df_set]
    if args.dr_ratio < 1.0:
        rng = random.Random(args.dr_seed)
        dr_count = max(1, int(len(dr_sids) * args.dr_ratio))
        dr_sids = rng.sample(dr_sids, dr_count)

    df_loader = build_indexed_loader(
        train_dataset, df_sids,
        batch_size=args.batch_size_unlearn,
        num_workers=args.num_workers,
        shuffle=True,
    )
    dr_loader = build_indexed_loader(
        train_dataset, dr_sids,
        batch_size=args.batch_size_retrain,
        num_workers=args.num_workers,
        shuffle=True,
    )

    # df3 ctx torch (only if needed)
    df3_ctx_torch = None
    if args.df_name == "df3" and args.df3_unlearn_loss == "cluster_dist" and df3_cluster_ctx_np is not None:
        df3_ctx_torch = {
            "centers": torch.tensor(df3_cluster_ctx_np["centers"], device=device),
            "sid_to_cluster": df3_cluster_ctx_np["sid_to_cluster"],
        }

    # -----------------------
    # UNLEARN (GA on DF)
    # -----------------------
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_unlearn, weight_decay=0.0)

    it = iter(df_loader)
    eval_history = []
    steps_ran = 0

    for step in range(args.steps_unlearn):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(df_loader)
            batch = next(it)

        x, score, label = prepare_batch(batch, device, args.model_confidence)
        z, logdet, mean, logs, nll, denom = forward_nll_components(model, x, label)
        batch_sids = batch[1].tolist() if torch.is_tensor(batch[1]) else list(batch[1])

        score_val, score_is_nll = compute_unlearn_score(
            args.df_name, z, logdet, mean, logs, nll, denom, batch_sids, df3_ctx_torch, args
        )
        if use_conf_score_df and score_is_nll:
            score_val = score_val * reduce_conf_score(score)

        loss = -score_val.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        steps_ran = step + 1

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            val_nll = eval_nll_on_indices(
                model, val_dataset, val_norm_sids,
                device=device, model_confidence=args.model_confidence,
                batch_size=args.batch_size_retrain,
                num_workers=args.num_workers,
                use_conf_score=use_conf_score_val,
            )
            model.train()
            fpr_val = float(np.mean(val_nll >= tau_base)) if val_nll.size else 0.0
            eval_history.append({"step": step + 1, "fpr_val": fpr_val})
            if fpr_val > args.fpr_tol * fpr_base:
                break

    unlearn_metrics = compute_single_metrics(
        model, train_dataset, val_dataset, df_sids, val_sids,
        cache_train, cache_val, tau_base, device,
        args, use_conf_score_df, use_conf_score_val,
        args.df_name, df3_cluster_ctx_np
    )

    unlearn_ckpt = os.path.join(args.output_dir, f"{args.df_name}_unlearn.pth.tar")
    torch.save({"state_dict": model.state_dict(), "stage": "unlearn", "df_name": args.df_name}, unlearn_ckpt)
    unlearn_roc_auc = eval_roc_auc(model, test_loader, test_metadata, args, device)

    # -----------------------
    # RETRAIN (GD on DR)
    # -----------------------
    use_conf_score_retrain = use_conf_score_train
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_retrain, weight_decay=0.0)
    model.train()

    retrain_history = []
    retrain_roc_history = []
    retrain_roc_best = None
    retrain_best_state = None
    retrain_best_epoch = None
    for epoch in range(args.epochs_retrain):
        for step, batch in enumerate(dr_loader):
            x, score, label = prepare_batch(batch, device, args.model_confidence)
            _, nll = model(x, label=label)
            if use_conf_score_retrain:
                nll = nll * reduce_conf_score(score)

            loss = nll.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        epoch_roc = eval_roc_auc(model, test_loader, test_metadata, args, device)
        retrain_roc_history.append({"epoch": epoch + 1, "roc_auc": epoch_roc})
        if epoch_roc is not None:
            if retrain_roc_best is None or epoch_roc > retrain_roc_best:
                retrain_roc_best = float(epoch_roc)
                retrain_best_state = {
                    k: v.detach().cpu() if torch.is_tensor(v) else v
                    for k, v in model.state_dict().items()
                }
                retrain_best_epoch = epoch + 1

    retrain_metrics = compute_single_metrics(
        model, train_dataset, val_dataset, df_sids, val_sids,
        cache_train, cache_val, tau_base, device,
        args, use_conf_score_df, use_conf_score_val,
        args.df_name, df3_cluster_ctx_np
    )

    retrain_ckpt = os.path.join(args.output_dir, f"{args.df_name}_retrain.pth.tar")
    torch.save({"state_dict": model.state_dict(), "stage": "retrain", "df_name": args.df_name}, retrain_ckpt)
    retrain_best_ckpt = None
    retrain_best_metrics = None
    if retrain_best_state is not None:
        retrain_best_ckpt = os.path.join(args.output_dir, f"{args.df_name}_retrain_best.pth.tar")
        torch.save(
            {
                "state_dict": retrain_best_state,
                "stage": "retrain_best",
                "df_name": args.df_name,
                "best_epoch": retrain_best_epoch,
                "best_roc_auc": retrain_roc_best,
            },
            retrain_best_ckpt,
        )
        model.load_state_dict(retrain_best_state, strict=False)
        retrain_best_metrics = compute_single_metrics(
            model, train_dataset, val_dataset, df_sids, val_sids,
            cache_train, cache_val, tau_base, device,
            args, use_conf_score_df, use_conf_score_val,
            args.df_name, df3_cluster_ctx_np
        )
        retrain_best_roc_auc = eval_roc_auc(model, test_loader, test_metadata, args, device)
        if retrain_best_metrics is not None:
            retrain_best_metrics["roc_auc"] = retrain_best_roc_auc
    if retrain_roc_history:
        retrain_roc_auc_last = retrain_roc_history[-1]["roc_auc"]
    else:
        retrain_roc_auc_last = eval_roc_auc(model, test_loader, test_metadata, args, device)
    retrain_roc_auc_best = retrain_roc_best if retrain_roc_best is not None else retrain_roc_auc_last

    # -----------------------
    # SAVE RESULT
    # -----------------------
    results = {
        "mode": args.mode,
        "df_name": args.df_name,
        "df_info": df_info,
        "tau_base": tau_base,
        "fpr_base": fpr_base,
        "val_split": val_split,
        "tau_from_all_val": bool(args.tau_from_all_val),
        "df_size": len(df_sids),
        "dr_size": len(dr_sids),
        "steps_unlearn_ran": steps_ran,
        "use_conf_score": {
            "train": use_conf_score_train,
            "df": use_conf_score_df,
            "val": use_conf_score_val,
            "retrain": use_conf_score_retrain,
        },
        "args": _serialize_args(args),
        "argv": list(sys.argv),
        "command": " ".join(sys.argv),
        "unlearn_metrics": unlearn_metrics,
        "unlearn_eval_history": eval_history,
        "retrain_metrics": retrain_metrics,
        "unlearn_roc_auc": unlearn_roc_auc,
        "retrain_roc_auc": retrain_roc_auc_best,
        "retrain_roc_auc_last": retrain_roc_auc_last,
        "retrain_roc_auc_history": retrain_roc_history,
        "retrain_best_ckpt": retrain_best_ckpt,
        "retrain_best_metrics": retrain_best_metrics,
        "unlearn_ckpt": unlearn_ckpt,
        "retrain_ckpt": retrain_ckpt,
    }
    out_json = os.path.join(args.output_dir, f"results_{args.df_name}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] saved: {out_json}")


if __name__ == "__main__":
    main()

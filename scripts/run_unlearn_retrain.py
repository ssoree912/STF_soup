import argparse
import math
import json
import os
import pickle
import random
import sys

# Ensure project root is on sys.path for local imports.
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
    select_df3_subdomain_train_df,
    overlap_report,
    check_nll_consistency,   # ✅ 추가
)


def parse_args():
    parser = init_parser()
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--cache_path_train", type=str, default=None)
    parser.add_argument("--cache_path_val", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/unlearning")

    # DF selection
    parser.add_argument("--alpha_df1", type=float, default=0.01)
    parser.add_argument("--alpha_df2", type=float, default=0.02)
    parser.add_argument(
        "--df2_mode",
        type=str,
        default="normalized",
        choices=["raw", "normalized", "relative"],
    )
    parser.add_argument("--df2_robust", action="store_true")
    parser.add_argument("--df2_trim_ratio", type=float, default=0.1)
    parser.add_argument("--df2_eps", type=float, default=1e-6)

    parser.add_argument("--k_df3", type=int, default=20)
    parser.add_argument("--top_clusters", type=int, default=2)
    parser.add_argument("--tau_q", type=float, default=0.999)
    parser.add_argument(
        "--df2_unlearn_loss",
        type=str,
        default="zsteps_delta",
        choices=["nll", "zsteps_delta", "nll_var"],
    )
    parser.add_argument(
        "--df3_unlearn_loss",
        type=str,
        default="emb_norm",
        choices=["nll", "emb_norm", "cluster_dist"],
    )

    # val selection for monitoring
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--val_seed", type=int, default=0)
    parser.add_argument("--tau_from_all_val", action="store_true",
                        help="tau_base를 val_sids가 아니라 cache_val 전체 후보에서 계산 (더 안정적)")

    # training
    parser.add_argument("--batch_size_unlearn", type=int, default=64)
    parser.add_argument("--batch_size_retrain", type=int, default=256)
    parser.add_argument("--lr_unlearn", type=float, default=3e-5)
    parser.add_argument("--steps_unlearn", type=int, default=800)
    parser.add_argument("--lr_retrain", type=float, default=5e-5)
    parser.add_argument("--epochs_retrain", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # safety / early stop
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--fpr_tol", type=float, default=2.0)

    # DR sampling
    parser.add_argument("--dr_ratio", type=float, default=1.0)
    parser.add_argument("--dr_seed", type=int, default=0)

    # checks
    parser.add_argument("--sanity_check_indices", action="store_true")
    parser.add_argument("--nll_check_batches", type=int, default=0,
                        help=">0이면 시작 시 nll consistency 체크 수행 (배치 수)")

    # skip
    parser.add_argument("--skip_df1", action="store_true")
    parser.add_argument("--skip_df2", action="store_true")
    parser.add_argument("--skip_df3", action="store_true")

    return parser.parse_args()


def sample_val_sids(all_sids, ratio, seed):
    if ratio <= 0:
        return list(all_sids)
    count = max(1, int(len(all_sids) * ratio))
    if count >= len(all_sids):
        return list(all_sids)
    rng = random.Random(seed)
    return rng.sample(list(all_sids), count)


def compute_basic_metrics(
    model,
    dataset_df,
    df_sids,
    dataset_val,
    val_sids,
    cache_df,
    cache_val,
    tau_base,
    device,
    model_confidence,
    batch_size,
    num_workers,
    use_conf_score_df,
    use_conf_score_val,
):
    df_nll = eval_nll_on_indices(
        model,
        dataset_df,
        df_sids,
        device=device,
        model_confidence=model_confidence,
        batch_size=batch_size,
        num_workers=num_workers,
        use_conf_score=use_conf_score_df,
    )
    val_nll = eval_nll_on_indices(
        model,
        dataset_val,
        val_sids,
        device=device,
        model_confidence=model_confidence,
        batch_size=batch_size,
        num_workers=num_workers,
        use_conf_score=use_conf_score_val,
    )

    # cache_df["sB"] / cache_val["sB"]는 cache 생성 당시의 스코어링 규칙을 반영한 base 값
    base_df = np.mean([cache_df["sB"][sid] for sid in df_sids]) if df_sids else 0.0
    base_val = np.mean([cache_val["sB"][sid] for sid in val_sids]) if val_sids else 0.0

    metrics = {
        "df_nll_mean": float(df_nll.mean()) if df_nll.size else 0.0,
        "val_nll_mean": float(val_nll.mean()) if val_nll.size else 0.0,
        "delta_df": float(df_nll.mean() - base_df) if df_nll.size else 0.0,
        "delta_val": float(val_nll.mean() - base_val) if val_nll.size else 0.0,
        "fpr_val": float(np.mean(val_nll >= tau_base)) if val_nll.size else 0.0,
    }
    return metrics


def sanity_check_dataset_indices(dataset_obj, sids, name="dataset"):
    if not sids:
        return {"ok": True, "msg": f"{name}: empty sids"}
    try:
        _ = dataset_obj[int(sids[0])]
        _ = dataset_obj[int(sids[len(sids)//2])]
        _ = dataset_obj[int(sids[-1])]
        return {"ok": True, "msg": f"{name}: index access OK (sampled 3 points)"}
    except Exception as e:
        return {"ok": False, "msg": f"{name}: index access FAILED: {repr(e)}"}


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


def compute_unlearn_score(
    df_name,
    z,
    logdet,
    mean,
    logs,
    nll,
    denom,
    batch_sids,
    df3_ctx,
    args,
):
    if df_name == "df1":
        return nll, True

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

    if df_name == "df3":
        if args.df3_unlearn_loss == "nll":
            return nll, True
        emb = z.mean(dim=2).reshape(z.size(0), -1)
        if args.df3_unlearn_loss == "emb_norm":
            return (emb ** 2).sum(dim=1), False
        if args.df3_unlearn_loss == "cluster_dist":
            if df3_ctx is None:
                raise ValueError("df3_unlearn_loss=cluster_dist requires cluster centers.")
            centers = df3_ctx["centers"]
            sid_to_cluster = df3_ctx["sid_to_cluster"]
            try:
                cluster_idx = [sid_to_cluster[int(sid)] for sid in batch_sids]
            except KeyError as exc:
                raise KeyError(f"Missing df3 cluster label for sid={exc}.") from exc
            idx = torch.tensor(cluster_idx, device=emb.device, dtype=torch.long)
            center_batch = centers[idx]
            dist2 = (emb - center_batch).pow(2).sum(dim=1)
            return dist2, False

    raise ValueError(f"Unsupported unlearn loss for df={df_name}")


def main():
    args = parse_args()
    args, _ = init_sub_args(args)

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for unlearning")

    os.makedirs(args.output_dir, exist_ok=True)

    # reproducibility
    random.seed(args.val_seed)
    np.random.seed(args.val_seed)
    torch.manual_seed(args.val_seed)

    cache_path_train = args.cache_path_train or args.cache_path
    cache_path_val = args.cache_path_val or args.cache_path

    with open(cache_path_train, "rb") as f:
        cache_train = pickle.load(f)
    with open(cache_path_val, "rb") as f:
        cache_val = pickle.load(f)

    # ------------------------------------------------------------
    # 1) Prepare sids + tau_base
    # ------------------------------------------------------------
    normal_sids = sorted(cache_train["sB"].keys())   # train normal candidates
    val_candidates = sorted(cache_val["sB"].keys())  # val normal candidates
    val_sids = sample_val_sids(val_candidates, args.val_ratio, args.val_seed)

    # tau_base 계산: 더 안정적으로 하고 싶으면 cache_val 전체 후보에서 뽑기
    if args.tau_from_all_val:
        s_val_all = np.array([cache_val["sB"][sid] for sid in val_candidates], dtype=np.float32)
        tau_base = float(np.quantile(s_val_all, args.tau_q))
    else:
        s_val_sub = np.array([cache_val["sB"][sid] for sid in val_sids], dtype=np.float32)
        tau_base = float(np.quantile(s_val_sub, args.tau_q))

    # fpr_base는 monitoring subset(val_sids)에서 정의(early-stop 기준으로 쓰기 위함)
    s_val_monitor = np.array([cache_val["sB"][sid] for sid in val_sids], dtype=np.float32)
    fpr_base = float(np.mean(s_val_monitor >= tau_base))
    fpr_base = max(fpr_base, 1.0 / max(1, len(val_sids)))

    # ------------------------------------------------------------
    # 2) DF selection
    # ------------------------------------------------------------
    df_sets = {}
    df_info = {}
    df3_cluster_ctx = None
    df_unlearn_loss = {
        "df1": "nll",
        "df2": args.df2_unlearn_loss,
        "df3": args.df3_unlearn_loss,
    }

    if not args.skip_df1:
        df_sets["df1"] = select_df1_tail(cache_train["sB"], alpha=args.alpha_df1)

    if not args.skip_df2:
        df_sets["df2"] = select_df2_dynamics_v2(
            cache_train,
            alpha_g=args.alpha_df2,
            mode=args.df2_mode,
            robust=args.df2_robust,
            trim_ratio=args.df2_trim_ratio,
            eps=args.df2_eps,
        )

    if not args.skip_df3:
        if args.df3_unlearn_loss == "cluster_dist":
            df3, info, df3_cluster_ctx = select_df3_subdomain_train_df(
                cache_train,
                normal_sids,
                cache_val,
                val_sids,
                k_clusters=args.k_df3,
                top_clusters=args.top_clusters,
                tau_q=args.tau_q,
                return_cluster_ctx=True,
            )
        else:
            df3, info = select_df3_subdomain_train_df(
                cache_train,
                normal_sids,
                cache_val,
                val_sids,
                k_clusters=args.k_df3,
                top_clusters=args.top_clusters,
                tau_q=args.tau_q,
            )
        df_sets["df3"] = df3
        df_info["df3"] = info

    overlap = overlap_report(df_sets)
    val_split = cache_val.get("meta", {}).get("split", "train")

    # DF splits dump
    df_splits_path = os.path.join(args.output_dir, "df_splits.json")
    with open(df_splits_path, "w") as f:
        json.dump(
            {
                "df_sets": {k: list(v) for k, v in df_sets.items()},
                "val_sids": list(val_sids),
                "tau_base": tau_base,
                "fpr_base": fpr_base,
                "df_info": df_info,
                "df_unlearn_loss": df_unlearn_loss,
                "overlap": overlap,
                "cache_train_path": cache_path_train,
                "cache_val_path": cache_path_val,
                "val_split": val_split,
                "tau_from_all_val": bool(args.tau_from_all_val),
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # 3) Load datasets
    # ------------------------------------------------------------
    dataset, _ = get_dataset_and_loader(args, trans_list=trans_list, only_test=False)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"] if val_split == "test" else dataset["train"]

    device = torch.device(args.device)

    # ✅ 핵심: 스코어링 규칙을 캐시 meta에 맞춰 일관되게 사용
    # - cache_train/meta/use_conf_score: cache 만들 때 nll에 score를 곱했는지
    # - args.model_confidence: 모델 입력에서 confidence 채널 사용 여부(prepare_batch에서 slicing 여부)
    use_conf_score_train = bool(args.model_confidence)  # retrain에서 쓸 기본 규칙
    use_conf_score_df = bool(cache_train.get("meta", {}).get("use_conf_score", use_conf_score_train))
    use_conf_score_val = bool(cache_val.get("meta", {}).get("use_conf_score", use_conf_score_train))

    results = {
        "tau_base": tau_base,
        "fpr_base": fpr_base,
        "df_info": df_info,
        "df_unlearn_loss": df_unlearn_loss,
        "overlap": overlap,
        "cache_train_path": cache_path_train,
        "cache_val_path": cache_path_val,
        "val_split": val_split,
        "tau_from_all_val": bool(args.tau_from_all_val),
        "checks": {},
        "runs": {},
    }

    # ------------------------------------------------------------
    # 4) Optional sanity checks
    # ------------------------------------------------------------
    if args.sanity_check_indices:
        results["checks"]["train_index_check"] = sanity_check_dataset_indices(train_dataset, normal_sids, "train_dataset")
        results["checks"]["val_index_check"] = sanity_check_dataset_indices(val_dataset, val_sids, "val_dataset")

    # NLL consistency check (extract vs forward) — 모델 1번 로드해서 체크만 하고 버림
    if args.nll_check_batches and args.nll_check_batches > 0:
        model_tmp = load_model_from_checkpoint(args, dataset, args.checkpoint).to(device)
        # 작은 loader (train normal 일부)
        probe_sids = normal_sids[: min(len(normal_sids), args.batch_size_retrain * max(1, args.nll_check_batches))]
        probe_loader = build_indexed_loader(
            train_dataset,
            probe_sids,
            batch_size=args.batch_size_retrain,
            num_workers=args.num_workers,
            shuffle=False,
        )
        results["checks"]["nll_consistency"] = check_nll_consistency(
            model_tmp,
            probe_loader,
            device=device,
            model_confidence=args.model_confidence,
            use_conf_score=use_conf_score_df,    # cache 기반과 맞춰 보기
            max_batches=int(args.nll_check_batches),
        )
        del model_tmp
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ------------------------------------------------------------
    # 5) For each DF: unlearn (GA) -> retrain (GD)
    # ------------------------------------------------------------
    for df_name, df_sids in df_sets.items():
        if not df_sids:
            continue

        df_set = set(df_sids)
        dr_sids = [sid for sid in normal_sids if sid not in df_set]

        if args.dr_ratio < 1.0:
            dr_count = max(1, int(len(dr_sids) * args.dr_ratio))
            rng = random.Random(args.dr_seed)
            dr_sids = rng.sample(dr_sids, dr_count)

        df_loader = build_indexed_loader(
            train_dataset,
            df_sids,
            batch_size=args.batch_size_unlearn,
            num_workers=args.num_workers,
            shuffle=True,
        )
        dr_loader = build_indexed_loader(
            train_dataset,
            dr_sids,
            batch_size=args.batch_size_retrain,
            num_workers=args.num_workers,
            shuffle=True,
        )

        model = load_model_from_checkpoint(args, dataset, args.checkpoint).to(device)

        # ✅ GA 단계는 DF 기준 스코어링을 사용 (NLL 기반일 때만 conf-score 적용)
        use_conf_score_unlearn = use_conf_score_df
        df3_ctx = None
        if df_name == "df3" and df3_cluster_ctx is not None:
            df3_ctx = {
                "centers": torch.tensor(df3_cluster_ctx["centers"], device=device),
                "sid_to_cluster": df3_cluster_ctx["sid_to_cluster"],
            }

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_unlearn, weight_decay=0.0)
        model.train()

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
            unlearn_score, score_is_nll = compute_unlearn_score(
                df_name,
                z,
                logdet,
                mean,
                logs,
                nll,
                denom,
                batch_sids,
                df3_ctx,
                args,
            )
            if use_conf_score_unlearn and score_is_nll:
                unlearn_score = unlearn_score * reduce_conf_score(score)

            loss = -unlearn_score.mean()  # GA: DF 기준 스코어 최대화
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            steps_ran = step + 1

            # early-stop monitor: val에서 FPR 폭주 방지
            if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
                val_nll = eval_nll_on_indices(
                    model,
                    val_dataset,
                    val_sids,
                    device=device,
                    model_confidence=args.model_confidence,
                    batch_size=args.batch_size_retrain,
                    num_workers=args.num_workers,
                    use_conf_score=use_conf_score_val,
                )
                model.train()
                fpr_val = float(np.mean(val_nll >= tau_base)) if val_nll.size else 0.0
                eval_history.append({"step": step + 1, "fpr_val": fpr_val})
                if fpr_val > args.fpr_tol * fpr_base:
                    break

        unlearn_metrics = compute_basic_metrics(
            model,
            train_dataset,
            df_sids,
            val_dataset,
            val_sids,
            cache_train,
            cache_val,
            tau_base,
            device,
            args.model_confidence,
            args.batch_size_retrain,
            args.num_workers,
            use_conf_score_df,
            use_conf_score_val,
        )
        model.train()

        unlearn_ckpt = os.path.join(args.output_dir, f"{df_name}_unlearn.pth.tar")
        torch.save({"state_dict": model.state_dict(), "stage": "unlearn"}, unlearn_ckpt)

        # ✅ retrain은 "train 기본 스코어링"을 사용 (보통 args.model_confidence 기반)
        use_conf_score_retrain = use_conf_score_train

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_retrain, weight_decay=0.0)
        model.train()
        retrain_history = []

        for epoch in range(args.epochs_retrain):
            for batch in dr_loader:
                x, score, label = prepare_batch(batch, device, args.model_confidence)
                _, nll = model(x, label=label)

                if use_conf_score_retrain:
                    nll = nll * reduce_conf_score(score)

                loss = nll.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

            epoch_metrics = compute_basic_metrics(
                model,
                train_dataset,
                df_sids,
                val_dataset,
                val_sids,
                cache_train,
                cache_val,
                tau_base,
                device,
                args.model_confidence,
                args.batch_size_retrain,
                args.num_workers,
                use_conf_score_df,
                use_conf_score_val,
            )
            model.train()
            epoch_metrics["epoch"] = epoch + 1
            retrain_history.append(epoch_metrics)

        retrain_metrics = compute_basic_metrics(
            model,
            train_dataset,
            df_sids,
            val_dataset,
            val_sids,
            cache_train,
            cache_val,
            tau_base,
            device,
            args.model_confidence,
            args.batch_size_retrain,
            args.num_workers,
            use_conf_score_df,
            use_conf_score_val,
        )

        retrain_ckpt = os.path.join(args.output_dir, f"{df_name}_retrain.pth.tar")
        torch.save({"state_dict": model.state_dict(), "stage": "retrain"}, retrain_ckpt)

        results["runs"][df_name] = {
            "df_size": len(df_sids),
            "dr_size": len(dr_sids),
            "steps_unlearn": steps_ran,
            "unlearn_loss": df_unlearn_loss.get(df_name),
            "use_conf_score": {
                "train": use_conf_score_train,
                "df": use_conf_score_df,
                "val": use_conf_score_val,
                "unlearn": use_conf_score_unlearn,
                "retrain": use_conf_score_retrain,
            },
            "unlearn_metrics": unlearn_metrics,
            "unlearn_eval_history": eval_history,
            "retrain_metrics": retrain_metrics,
            "retrain_history": retrain_history,
            "unlearn_ckpt": unlearn_ckpt,
            "retrain_ckpt": retrain_ckpt,
        }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

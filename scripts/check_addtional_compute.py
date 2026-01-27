#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import random
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import torch

from args import init_parser, init_sub_args, create_exp_dirs
from dataset import get_dataset_and_loader
from utils.data_utils import trans_list
from utils.train_utils import init_model_params, dump_args
from utils.scoring_utils import score_dataset

from models.STG_NF.model_pose import STG_NF

# ====== 너가 올린 유틸들 (이미 프로젝트에 있다면 import로 바꿔도 됨) ======
# 여기서는 "같은 파일에 있다" 가정 대신, 네가 올린 코드가 utils/unlearning_utils.py 등에 있다고 보고 import로 씀.
from utils.unlearning_utils import (
    set_seed,
    get_base_indices,
    build_indexed_loader,
    load_model_from_checkpoint,
    prepare_batch,
    reduce_conf_score,
    extract_steps_and_emb,
    build_base_cache,
    select_df1_tail,
    select_df2_dynamics_v2,
    select_df3_subdomain_train_df,
    select_df2_val_fp_train_df,
    df_gradient_ascent,
    eval_nll_on_indices,
)

# ------------------------------------------------------------
# Step-based minimize training (compute-matched 핵심)
# ------------------------------------------------------------
def train_minimize_steps(
    model: STG_NF,
    dataset,
    indices: List[int],
    device: torch.device,
    model_confidence: bool,
    batch_size: int,
    steps: int,
    lr: float,
    num_workers: int = 0,
    use_conf_score: bool = False,
    grad_clip: float = 1.0,
    shuffle: bool = True,
) -> STG_NF:
    loader = build_indexed_loader(
        dataset, indices,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    model.train()
    it = iter(loader)
    for _ in range(int(steps)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x, score, label = prepare_batch(batch, device, model_confidence)
        _, nll = model(x, label=label)
        if use_conf_score:
            nll = nll * reduce_conf_score(score)

        loss = nll.mean()  # minimize NLL
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

    return model


def save_ckpt(path: str, model: STG_NF, meta: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, path)


def _load_reference_args(path: str) -> Optional[argparse.Namespace]:
    if path is None:
        return None
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return argparse.Namespace(**payload)
    raise ValueError(f"Unsupported args.json format: {path}")


def _merge_with_defaults(loaded_args: Optional[argparse.Namespace]) -> argparse.Namespace:
    base_args = init_parser().parse_args([])
    if loaded_args is None:
        return base_args
    for k, v in vars(loaded_args).items():
        setattr(base_args, k, v)
    return base_args


def _apply_overrides(base_args: argparse.Namespace, override_args: argparse.Namespace) -> argparse.Namespace:
    for k, v in vars(override_args).items():
        if v is None:
            continue
        setattr(base_args, k, v)
    return base_args


def parse_args():
    p = argparse.ArgumentParser()

    # reference args.json
    p.add_argument("--reference_args", type=str, required=True,
                   help="Path to args.json from the baseline training run")

    # 기본 (프로젝트 args + override)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--exp_dir", type=str, default="results/unlearning_controls")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)

    # baseline ckpt
    p.add_argument("--reference_ckpt", type=str, required=True)

    # DF 선택 종류
    p.add_argument("--df_name", type=str, required=True,
                   choices=["df1", "df2", "df2p", "df3", "df3p"])

    # 공통 하이퍼
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--model_confidence", dest="model_confidence", action="store_true", default=None)
    p.add_argument("--no_model_confidence", dest="model_confidence", action="store_false")
    p.add_argument("--use_conf_score", dest="use_conf_score", action="store_true", default=None)
    p.add_argument("--no_use_conf_score", dest="use_conf_score", action="store_false")

    # df1
    p.add_argument("--alpha_df1", type=float, default=0.01)

    # df2
    p.add_argument("--alpha_g", type=float, default=0.02)
    p.add_argument("--df2_mode", type=str, default="normalized", choices=["normalized", "relative"])
    p.add_argument("--df2_robust", action="store_true")
    p.add_argument("--df2_trim_ratio", type=float, default=0.1)

    # df3/df3p
    p.add_argument("--k_clusters", type=int, default=20)
    p.add_argument("--top_clusters", type=int, default=2)
    p.add_argument("--tau_q", type=float, default=0.999)

    # df2p
    p.add_argument("--knn_k", type=int, default=20)
    p.add_argument("--budget_alpha", type=float, default=0.02)
    p.add_argument("--df2p_sb_max_q", type=float, default=0.95)
    p.add_argument("--df2p_sb_penalty", type=float, default=0.2)

    # compute-matched steps
    p.add_argument("--steps_ascent", type=int, default=800)
    p.add_argument("--steps_retrain", type=int, default=800)   # epoch 대신 steps로 고정!
    p.add_argument("--lr_ascent", type=float, default=3e-5)
    p.add_argument("--lr_retrain", type=float, default=5e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # 모드
    p.add_argument("--mode", type=str, required=True,
                   choices=["unlearn", "ascent_only", "retrain_only", "continue_full", "all"])

    return p.parse_args()


def main():
    cli_args = parse_args()
    ref_args = _load_reference_args(cli_args.reference_args)
    args = _merge_with_defaults(ref_args)
    args = _apply_overrides(args, cli_args)
    if getattr(args, "dataset", None) is None:
        raise ValueError("--dataset is missing (not found in reference_args)")
    args, _ = init_sub_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)

    # ===== dataset / loader =====
    # 네 프로젝트 get_dataset_and_loader 시그니처에 맞춤
    # only_test=False로 train/test 둘 다 로드
    class Dummy:
        pass

    # 기존 코드가 init_parser 기반 args를 기대하면, 여기를 네 args 체계로 합쳐야 함.
    # 가장 쉬운 방법: 기존 init_parser() args.json을 그대로 불러오는 방식.
    # 여기선 간단히 get_dataset_and_loader가 args.dataset, args.data_dir 등만 본다고 가정.
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=False)

    # ===== baseline model load =====
    base_model = load_model_from_checkpoint(args, dataset, args.reference_ckpt)
    base_model = base_model.to(device)

    # ===== indices =====
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    train_sids = get_base_indices(train_ds, normal_only=True)
    val_sids = get_base_indices(test_ds, normal_only=False)  # 여기서는 test를 val처럼 사용(너 코드와 동일 습관)
    # NOTE: 원래는 val split을 따로 두는게 더 좋음.

    # ===== base cache (train / val) =====
    # 캐시 계산 비용이 크면 batch_size를 키우고 num_workers 늘려도 됨
    train_loader_for_cache = build_indexed_loader(
        train_ds, train_sids,
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
    )
    val_loader_for_cache = build_indexed_loader(
        test_ds, val_sids,
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
    )

    cache_train = build_base_cache(
        base_model, train_loader_for_cache, device,
        model_confidence=args.model_confidence,
        use_conf_score=args.use_conf_score,
    )
    cache_val = build_base_cache(
        base_model, val_loader_for_cache, device,
        model_confidence=args.model_confidence,
        use_conf_score=args.use_conf_score,
    )

    # ===== DF 선택 =====
    if args.df_name == "df1":
        df_indices = select_df1_tail(cache_train["sB"], alpha=args.alpha_df1)

    elif args.df_name == "df2":
        df_indices = select_df2_dynamics_v2(
            cache_train,
            alpha_g=args.alpha_g,
            mode=args.df2_mode,
            robust=args.df2_robust,
            trim_ratio=args.df2_trim_ratio,
        )

    elif args.df_name == "df3":
        df_indices, info = select_df3_subdomain_train_df(
            cache_train, train_sids,
            cache_val, val_sids,
            k_clusters=args.k_clusters,
            top_clusters=args.top_clusters,
            tau_q=args.tau_q,
            return_cluster_ctx=False,
        )

    elif args.df_name == "df3p":
        df_indices, info, _ctx = select_df3_subdomain_train_df(
            cache_train, train_sids,
            cache_val, val_sids,
            k_clusters=args.k_clusters,
            top_clusters=args.top_clusters,
            tau_q=args.tau_q,
            return_cluster_ctx=True,
        )

    elif args.df_name == "df2p":
        # tau_base는 val(여기선 test)에서 추정
        s_val = np.array([cache_val["sB"][sid] for sid in val_sids], dtype=np.float32)
        tau_base = float(np.quantile(s_val, args.tau_q))
        df_indices, info = select_df2_val_fp_train_df(
            cache_train, train_sids,
            cache_val, val_sids,
            val_dataset=test_ds,
            tau_base=tau_base,
            knn_k=args.knn_k,
            budget_alpha=args.budget_alpha,
            df2p_sb_max_q=args.df2p_sb_max_q,
            df2p_sb_penalty=args.df2p_sb_penalty,
            seed=args.seed,
        )

    else:
        raise ValueError(f"Unknown df_name: {args.df_name}")

    df_set = set(map(int, df_indices))
    dr_indices = [sid for sid in train_sids if int(sid) not in df_set]

    # ===== run modes =====
    run_dir = create_exp_dirs(
        args.exp_dir,
        dirmap=os.path.join(args.dataset, f"seed_{args.seed}", args.df_name),
    )

    def _run_mode(mode: str):
        model = load_model_from_checkpoint(args, dataset, args.reference_ckpt).to(device)

        meta = {
            "dataset": args.dataset,
            "seed": args.seed,
            "df_name": args.df_name,
            "mode": mode,
            "df_size": len(df_indices),
            "dr_size": len(dr_indices),
            "steps_ascent": args.steps_ascent,
            "steps_retrain": args.steps_retrain,
            "lr_ascent": args.lr_ascent,
            "lr_retrain": args.lr_retrain,
        }

        if mode == "unlearn":
            # 1) DF ascent
            model = df_gradient_ascent(
                model, train_ds, df_indices,
                device=device,
                model_confidence=args.model_confidence,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_conf_score=args.use_conf_score,
                lr=args.lr_ascent,
                steps=args.steps_ascent,
                grad_clip=args.grad_clip,
            )
            # 2) DR retrain (step-based)
            model = train_minimize_steps(
                model, train_ds, dr_indices,
                device=device,
                model_confidence=args.model_confidence,
                batch_size=args.batch_size,
                steps=args.steps_retrain,
                lr=args.lr_retrain,
                num_workers=args.num_workers,
                use_conf_score=args.use_conf_score,
                grad_clip=args.grad_clip,
            )

        elif mode == "ascent_only":
            model = df_gradient_ascent(
                model, train_ds, df_indices,
                device=device,
                model_confidence=args.model_confidence,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_conf_score=args.use_conf_score,
                lr=args.lr_ascent,
                steps=args.steps_ascent,
                grad_clip=args.grad_clip,
            )

        elif mode == "retrain_only":
            model = train_minimize_steps(
                model, train_ds, dr_indices,
                device=device,
                model_confidence=args.model_confidence,
                batch_size=args.batch_size,
                steps=args.steps_retrain,
                lr=args.lr_retrain,
                num_workers=args.num_workers,
                use_conf_score=args.use_conf_score,
                grad_clip=args.grad_clip,
            )

        elif mode == "continue_full":
            total_steps = int(args.steps_ascent + args.steps_retrain)
            model = train_minimize_steps(
                model, train_ds, train_sids,   # 전체 train
                device=device,
                model_confidence=args.model_confidence,
                batch_size=args.batch_size,
                steps=total_steps,
                lr=args.lr_retrain,            # 보통 retrain lr을 사용(원하면 별도 lr 추가)
                num_workers=args.num_workers,
                use_conf_score=args.use_conf_score,
                grad_clip=args.grad_clip,
            )
        else:
            raise ValueError(mode)

        # ===== evaluate on test =====
        from models.training import Trainer
        from utils.optim_init import init_optimizer, init_scheduler

        # loader는 get_dataset_and_loader에서 받은 loader 사용
        trainer = Trainer(
            args, model, loader["train"], loader["test"],
            optimizer_f=init_optimizer(getattr(args, "model_optimizer", "adamx"), lr=getattr(args, "model_lr", args.lr_retrain)),
            scheduler_f=init_scheduler(getattr(args, "model_sched", "none"), lr=getattr(args, "model_lr", args.lr_retrain), epochs=getattr(args, "epochs", 1)),
        )
        normality_scores = trainer.test()
        auc, _scores = score_dataset(normality_scores, dataset["test"].metadata, args=args)

        meta["roc_auc"] = float(auc)

        # ===== save =====
        ckpt_path = os.path.join(run_dir, f"{mode}_ckpt.pth.tar")
        save_ckpt(ckpt_path, model, meta=meta)
        json_path = os.path.join(run_dir, f"{mode}_metrics.json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(json.dumps(meta, indent=2))
        print(f"[Saved] {ckpt_path}")
        print(f"[Saved] {json_path}")

    if args.mode == "all":
        for m in ["continue_full", "retrain_only", "ascent_only", "unlearn"]:
            _run_mode(m)
    else:
        _run_mode(args.mode)


if __name__ == "__main__":
    main()

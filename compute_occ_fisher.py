#!/usr/bin/env python3
"""
학습 스크립트(train_eval.py)가 사용하는 compute_occ_fisher 방식으로
체크포인트별 대각 Fisher를 계산해 저장하는 실행 스크립트.

예시:
python compute_occ_fisher.py \
  --reference_args experiments/.../args.json \
  --checkpoints ckpt1.pth.tar ckpt2.pth.tar \
  --output_dir results/fisher_occ \
  --device cuda:0 \
  --fisher_max_batches 50 \
  --fisher_floor 1e-8 \
  --normalize
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from args import init_sub_args
from dataset import get_dataset_and_loader
from utils.data_utils import trans_list
from utils.fisher_utils import compute_occ_fisher
from utils.train_utils import init_model_params


def _setup_logger():
    import logging

    logger = logging.getLogger("compute_occ_fisher")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """weight_orig*mask → weight 로 병합, actnorm.inited는 1로 설정."""
    new_state: Dict[str, torch.Tensor] = {}
    for key, value in list(state_dict.items()):
        if key.endswith("weight_orig"):
            base_key = key[:-len("weight_orig")] + "weight"
            mask_key = key[:-len("weight_orig")] + "weight_mask"
            mask_tensor = state_dict.get(mask_key)
            if mask_tensor is None:
                mask_tensor = torch.ones_like(value)
            new_state[base_key] = value * mask_tensor
        elif key.endswith("weight_mask"):
            continue
        elif key.endswith("actnorm.inited"):
            new_state[key] = torch.ones_like(value) if torch.is_tensor(value) else 1
        else:
            new_state[key] = value
    return new_state


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="compute_occ_fisher 방식으로 Fisher 추출")
    ap.add_argument("--reference_args", type=Path, required=True, help="학습 때 저장된 args.json 경로")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="Fisher를 계산할 ckpt 경로들")
    ap.add_argument("--output_dir", type=Path, required=True, help="Fisher 저장 디렉토리")
    ap.add_argument("--device", default=None, help="cpu / cuda[:id] / mps")
    ap.add_argument("--fisher_max_batches", type=int, default=50, help="Fisher 계산에 사용할 최대 배치 수")
    ap.add_argument("--fisher_floor", type=float, default=1e-8, help="Fisher 최소값 클램프")
    ap.add_argument("--normalize", action="store_true", help="Fisher 텐서 정규화 사용")
    return ap.parse_args()


def main():
    args = _parse_args()
    logger = _setup_logger()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # reference args 로더
    with open(args.reference_args, "r") as fp:
        ref_payload = json.load(fp)
    ref_args = argparse.Namespace(**ref_payload)
    if args.device:
        ref_args.device = args.device
    ref_args.only_test = False
    ref_args, model_args = init_sub_args(ref_args)

    dataset, loader = get_dataset_and_loader(ref_args, trans_list=trans_list, only_test=False)
    model_args = init_model_params(ref_args, dataset)
    train_loader = loader["train"]
    if train_loader is None:
        raise RuntimeError("학습 로더가 필요합니다.")

    # 각 체크포인트별 Fisher 계산
    for ckpt_path in args.checkpoints:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")

        logger.info("Loading checkpoint: %s", ckpt_path)
        raw = torch.load(ckpt_path, map_location="cpu")
        if isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
        elif isinstance(raw, dict):
            state = raw
        else:
            raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
        state = _normalize_state_dict(state)

        model = torch.load  # silence linter
        model = None
        model = __import__("models.STG_NF.model_pose", fromlist=["STG_NF"]).STG_NF(**model_args)
        model.load_state_dict(state, strict=False)
        if hasattr(model, "set_actnorm_init"):
            model.set_actnorm_init()

        device = torch.device(ref_args.device)
        model.to(device)

        fisher_payload = compute_occ_fisher(
            model,
            train_loader,
            device=device,
            max_batches=args.fisher_max_batches,
            fisher_floor=args.fisher_floor,
            normalize=args.normalize,
        )

        # ckpt 이름이 동일해도 폴더별로 구분되도록 parent 폴더명을 붙여준다.
        folder_tag = ckpt_path.parent.name or "ckpt"
        out_path = args.output_dir / f"fisher_{folder_tag}_{ckpt_path.stem}.pt"
        torch.save(fisher_payload, out_path)
        logger.info(
            "Saved Fisher (occ) to %s | batches=%s normalize=%s",
            out_path,
            fisher_payload["metadata"]["num_batches"],
            fisher_payload["metadata"]["normalized"],
        )


if __name__ == "__main__":
    main()

"""정상 데이터에 대한 모델의 추론 결과를 미리 저장"""
import argparse
import os
import pickle

import torch

from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.data_utils import trans_list
from utils.unlearning_utils import (
    build_base_cache,
    build_indexed_loader,
    get_base_indices,
    load_model_from_checkpoint,
)


def parse_args():
    parser = init_parser()
    parser.add_argument("--cache_path", type=str, required=True, help="Output path for cache pickle")
    parser.add_argument("--cache_split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--cache_batch_size", type=int, default=None)
    parser.add_argument("--cache_limit", type=int, default=None)
    parser.add_argument("--use_conf_score", action="store_true")
    parser.add_argument("--check_nll_consistency", action="store_true") # NLL 일관성 검사 활성화
    parser.add_argument("--check_nll_batches", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    args, _ = init_sub_args(args)

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for cache building")

    dataset, _ = get_dataset_and_loader(args, trans_list=trans_list, only_test=False)
    base_dataset = dataset[args.cache_split]

    indices = get_base_indices(base_dataset, normal_only=True)
    if args.cache_limit is not None:
        indices = indices[: args.cache_limit]

    batch_size = args.cache_batch_size or args.batch_size
    loader = build_indexed_loader(
        base_dataset,
        indices,
        batch_size=batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = load_model_from_checkpoint(args, dataset, args.checkpoint)
    model = model.to(args.device)

    use_conf_score = args.use_conf_score or args.model_confidence
    cache = build_base_cache(
        model,
        loader,
        device=torch.device(args.device),
        model_confidence=args.model_confidence,
        use_conf_score=use_conf_score,
    )
    cache["meta"] = {
        "checkpoint": args.checkpoint,
        "split": args.cache_split,
        "num_samples": len(indices),
        "model_confidence": args.model_confidence,
        "use_conf_score": use_conf_score,
    }
    if args.check_nll_consistency:
        from utils.unlearning_utils import check_nll_consistency
        check_loader = build_indexed_loader(
            base_dataset,
            indices,
            batch_size=batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        stats = check_nll_consistency(
            model,
            check_loader,
            device=torch.device(args.device),
            model_confidence=args.model_confidence,
            use_conf_score=use_conf_score,
            max_batches=args.check_nll_batches,
        )
        cache["meta"]["nll_consistency"] = stats
        print("nll_consistency:", stats)

    out_dir = os.path.dirname(args.cache_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.cache_path, "wb") as f:
        pickle.dump(cache, f)

    print("Cache saved to {}".format(args.cache_path))


if __name__ == "__main__":
    main()

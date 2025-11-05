# tools/train_nf_ensemble.py
import argparse
from args import init_parser, init_sub_args, create_exp_dirs
from dataset import get_dataset_and_loader
from utils.data_utils import trans_list
from utils.train_utils import init_model_params, dump_args
from stg_nf_epistemic import create_epistemic_trainer, create_ensemble_trainer

def parse_args():
    parser = init_parser()
    parser.add_argument("--ensemble_size", type=int, default=15)
    parser.add_argument("--mask_type", type=str, default="random")
    parser.add_argument("--trainer_type", choices=["mixture", "masked"], default="mixture")
    return parser

def main():
    parser = parse_args()
    args = parser.parse_args()
    args, model_args = init_sub_args(args)

    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list)
    model_args = init_model_params(args, dataset)

    seed_dir = f"{args.dataset}/nf_epistemic/mask_{args.mask_type}/seed_{args.seed}"
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=seed_dir, run_name=args.run_name)

    trainer_factory = create_epistemic_trainer if args.trainer_type == "mixture" else create_ensemble_trainer
    trainer = trainer_factory(
        model_args=model_args,
        ensemble_size=args.ensemble_size,
        mask_type=args.mask_type,
        training_args=args,
        train_loader=loader["train"],
        test_loader=loader["test"],
        test_metadata=dataset["test"].metadata,
        device=args.device,
    )

    dump_args(args, args.ckpt_dir)
    trainer.train(args.epochs, save_dir=args.ckpt_dir)

if __name__ == "__main__":
    main()

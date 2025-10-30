import copy
import json
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params
from utils.scoring_utils import score_dataset
from utils.train_utils import calc_num_of_params
from utils.fisher_utils import compute_occ_fisher


def _configure_seed(args):
    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)


def _ratio_tag(prefix, value):
    formatted = f"{value:.4f}".rstrip('0').rstrip('.')
    formatted = formatted if formatted else "0"
    formatted = formatted.replace('.', 'p')
    return f"{prefix}{formatted}"


def _get_method_dir(args):
    mag_ratio = float(getattr(args, 'prune_ratio_magnitude', 0.0) or 0.0)
    rand_ratio = float(getattr(args, 'prune_ratio_random', 0.0) or 0.0)
    tags = []
    if mag_ratio > 0:
        tags.append(_ratio_tag("mag", mag_ratio))
    if rand_ratio > 0:
        tags.append(_ratio_tag("rnd", rand_ratio))
    prune_epoch = getattr(args, 'prune_epoch', None)
    if prune_epoch:
        tags.append(f"ep{prune_epoch}")
    unprune_epoch = getattr(args, 'unprune_epoch', None)
    if unprune_epoch:
        tags.append(f"un{unprune_epoch}")
    if not tags:
        return "baseline"
    return "prune_" + "_".join(tags)


def _persist_evaluation(ckpt_dir, auc, scores, labels, roc_parts):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    fpr, tpr, thresholds = [np.asarray(arr) for arr in roc_parts]
    metrics_path = os.path.join(ckpt_dir, "metrics.json")
    metrics_payload = {
        "auc": float(auc),
        "num_samples": int(scores.shape[0]),
        "scores_file": "scores_labels.npz",
        "roc_files": {
            "csv": "roc_curve.csv",
            "npz": "roc_curve.npz"
        }
    }
    with open(metrics_path, 'w') as fp:
        json.dump(metrics_payload, fp, indent=2, sort_keys=True)

    np.savez(os.path.join(ckpt_dir, "scores_labels.npz"), scores=scores, labels=labels)
    roc_matrix = np.stack([fpr, tpr, thresholds], axis=1)
    np.savetxt(os.path.join(ckpt_dir, "roc_curve.csv"), roc_matrix, delimiter=',', header='fpr,tpr,threshold', comments='')
    np.savez(os.path.join(ckpt_dir, "roc_curve.npz"), fpr=fpr, tpr=tpr, thresholds=thresholds)


def _run_single_experiment(base_args, run_label=None):
    args = copy.deepcopy(base_args)
    _configure_seed(args)
    args, model_args = init_sub_args(args)
    random_seed = getattr(args, 'prune_random_seed', None)
    rand_dir = f"rand_{random_seed}" if random_seed is not None else "rand_none"
    method_dir = _get_method_dir(args)
    seed_dir = os.path.join(args.dataset, method_dir, f"seed_{args.seed}", rand_dir)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=seed_dir, run_name=run_label)

    pretrained = vars(args).get('checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained is not None))

    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    num_of_params = calc_num_of_params(model)
    trainer = Trainer(args, model, loader['train'], loader['test'],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    if pretrained:
        trainer.load_checkpoint(pretrained)
    else:
        writer = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, 'tensorboard'))
        trainer.train(log_writer=writer)
        writer.flush()
        writer.close()
        dump_args(args, args.ckpt_dir)

    normality_scores = trainer.test()
    auc, scores, labels, roc_parts = score_dataset(normality_scores, dataset["test"].metadata, args=args)
    _persist_evaluation(args.ckpt_dir, auc, scores, labels, roc_parts)
    if getattr(args, 'compute_fisher', False):
        fisher = compute_occ_fisher(model, loader['train'], torch.device(args.device),
                                    max_batches=getattr(args, 'fisher_max_batches', 50),
                                    fisher_floor=getattr(args, 'fisher_floor', 1e-8),
                                    normalize=getattr(args, 'fisher_normalize', False))
        fisher_path = os.path.join(args.ckpt_dir, "fisher_diag.pt")
        torch.save(fisher, fisher_path)

    # Logging and recording results
    print("\n-------------------------------------------------------")
    print("\033[92m Done with {}% AuC for {} samples | Params: {}\033[0m".format(auc * 100, scores.shape[0], num_of_params))
    print("Checkpoint directory:", args.ckpt_dir)
    print("-------------------------------------------------------\n\n")
    return auc, args.ckpt_dir, random_seed


def main():
    parser = init_parser()
    args = parser.parse_args()

    if args.prune_ratio < 0 or args.prune_ratio >= 1.0:
        parser.error("--prune_ratio must be in the range [0, 1).")
    if args.prune_ratio_magnitude < 0 or args.prune_ratio_magnitude >= 1.0:
        parser.error("--prune_ratio_magnitude must be in the range [0, 1).")
    if args.prune_ratio_random < 0 or args.prune_ratio_random >= 1.0:
        parser.error("--prune_ratio_random must be in the range [0, 1).")
    if args.prune_ratio_magnitude == 0 and args.prune_ratio_random == 0 and args.prune_ratio > 0:
        if args.prune_method == 'random':
            args.prune_ratio_random = args.prune_ratio
        else:
            args.prune_ratio_magnitude = args.prune_ratio
    total_ratio = args.prune_ratio_magnitude + args.prune_ratio_random
    if total_ratio >= 1.0:
        parser.error("Combined pruning ratios must sum to less than 1.")
    if total_ratio > 0 and args.prune_epoch < 1:
        parser.error("--prune_epoch must be >= 1 when pruning is enabled.")
    if total_ratio == 0:
        args.unprune_epoch = None
        args.prune_epoch = None
    elif args.unprune_epoch is not None and args.unprune_epoch <= args.prune_epoch:
        parser.error("--unprune_epoch must be greater than --prune_epoch.")

    base_seed_list = args.seed_list if args.seed_list else [args.seed]
    random_seed_list = args.prune_random_seed_list if args.prune_random_seed_list else ([args.prune_random_seed] if args.prune_random_seed is not None else [None])
    num_runs = max(len(base_seed_list), len(random_seed_list))
    if len(base_seed_list) not in (1, num_runs):
        parser.error("Length of --seed_list must match random seed list or be a single value.")
    if len(random_seed_list) not in (1, num_runs):
        parser.error("Length of --prune_random_seed_list must match seed list or be a single value.")
    results = []
    for run_idx in range(num_runs):
        seed = base_seed_list[run_idx] if len(base_seed_list) > 1 else base_seed_list[0]
        random_seed = random_seed_list[run_idx] if len(random_seed_list) > 1 else random_seed_list[0]
        run_args = copy.deepcopy(args)
        run_args.seed = seed
        run_args.seed_list = None
        run_args.prune_random_seed = random_seed
        run_args.prune_random_seed_list = None
        label_random = random_seed if random_seed is not None else 'none'
        if args.run_name:
            run_label = args.run_name if num_runs == 1 else f"{args.run_name}_run{run_idx + 1}"
        else:
            run_label = None
        print(f"\n=== Run {run_idx + 1}/{num_runs} | Train seed {seed} | Random prune seed {label_random} ===")
        auc, ckpt_dir, applied_random_seed = _run_single_experiment(run_args, run_label=run_label)
        results.append((seed, applied_random_seed, auc, ckpt_dir))
    if len(results) > 1:
        print("\nMulti-run summary (AUC %):")
        for seed, random_seed, auc, ckpt_dir in results:
            label_random = random_seed if random_seed is not None else 'none'
            print(f"  Train seed {seed} | Random seed {label_random}: {auc * 100:.2f}%  -> {ckpt_dir}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from args import init_sub_args
from dataset import get_dataset_and_loader
from utils.data_utils import trans_list
from utils.unlearning_utils import load_model_from_checkpoint, prepare_batch, reduce_conf_score


def _load_reference_args(path: Path) -> argparse.Namespace:
    with open(path, "r") as f:
        payload = json.load(f)
    return argparse.Namespace(**payload)


def _load_args_from_checkpoint(path: Path) -> argparse.Namespace:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "args" in ckpt:
        ckpt_args = ckpt["args"]
        if isinstance(ckpt_args, argparse.Namespace):
            return ckpt_args
        if isinstance(ckpt_args, dict):
            return argparse.Namespace(**ckpt_args)
    raise ValueError(
        f"Checkpoint missing args: {path}. "
        "Provide --reference_args or a checkpoint saved with args."
    )


def _build_loader(ref_args: argparse.Namespace, batch_size: int, num_workers: int):
    ref_args.batch_size = batch_size
    ref_args.num_workers = num_workers
    ref_args, _ = init_sub_args(ref_args)
    dataset, loader = get_dataset_and_loader(ref_args, trans_list=trans_list, only_test=False)
    return ref_args, dataset, loader["train"]


def _sanitize_name(path: str) -> str:
    base = os.path.basename(path)
    name = base.replace(".pth.tar", "").replace(".pt", "").replace(".pth", "")
    return name.replace("/", "_")


def _normalize_state_dict(state_dict: dict) -> dict:
    """
    weight_orig + weight_mask -> weight
    drop weight_mask
    actnorm.inited -> 1
    """
    new_state = {}
    for k, v in list(state_dict.items()):
        if k.endswith("weight_orig"):
            base_key = k[:-len("weight_orig")] + "weight"
            mask_key = k[:-len("weight_orig")] + "weight_mask"
            mask = state_dict.get(mask_key)
            if mask is None:
                mask = torch.ones_like(v)
            new_state[base_key] = v * mask
        elif k.endswith("weight_mask"):
            continue
        elif k.endswith("actnorm.inited"):
            new_state[k] = torch.ones_like(v) if torch.is_tensor(v) else 1
        else:
            new_state[k] = v
    return new_state


def _normalize_fisher_sd_like_model_sd(
    fisher_sd: dict,
    model_sd: dict,
    fisher_floor: float,
) -> dict:
    """
    fisher(weight_orig) -> fisher(weight) (mask^2 적용), weight_mask drop.
    """
    out = {}
    for k, v in list(model_sd.items()):
        if k.endswith("weight_orig"):
            base_key = k[:-len("weight_orig")] + "weight"
            mask_key = k[:-len("weight_orig")] + "weight_mask"
            f_worig = fisher_sd.get(k)
            mask = model_sd.get(mask_key)
            if f_worig is None:
                f_worig = torch.full_like(v, float(fisher_floor))
            if mask is None:
                mask = torch.ones_like(v)
            out[base_key] = f_worig * (mask.to(f_worig.device, f_worig.dtype) ** 2)
        elif k.endswith("weight_mask"):
            continue
        elif k.endswith("actnorm.inited"):
            out[k] = torch.ones_like(v) if torch.is_tensor(v) else 1
        else:
            out[k] = fisher_sd.get(k, torch.full_like(v, float(fisher_floor)))
    return out


def _accumulate_fisher(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    model_confidence: bool,
    max_batches: int,
    fisher_floor: float,
):
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad and torch.is_tensor(param):
            fisher[name] = torch.zeros_like(param, device=device)

    model.train()
    num_batches = 0
    num_samples = 0

    for batch in loader:
        if max_batches > 0 and num_batches >= max_batches:
            break
        x, score, label = prepare_batch(batch, device, model_confidence)
        _, nll = model(x, label=label)
        if model_confidence:
            nll = nll * reduce_conf_score(score)
        loss = nll.mean()

        model.zero_grad(set_to_none=True)
        loss.backward()

        for name, param in model.named_parameters():
            if name in fisher and param.grad is not None:
                fisher[name].add_(param.grad.detach() ** 2)

        num_batches += 1
        num_samples += x.size(0)

    if num_batches > 0:
        for name in fisher:
            fisher[name] = fisher[name] / float(num_batches)

    for name in fisher:
        fisher[name].clamp_min_(float(fisher_floor))

    return fisher, num_batches, num_samples


def _normalize_fisher(fisher_sd: dict, eps: float):
    flat = []
    for t in fisher_sd.values():
        if torch.is_tensor(t) and t.is_floating_point():
            flat.append(t.reshape(-1))
    if not flat:
        return fisher_sd
    all_vals = torch.cat(flat, dim=0)
    mean_val = float(all_vals.mean().item()) if all_vals.numel() else 0.0
    if mean_val <= 0:
        return fisher_sd
    out = {}
    for k, v in fisher_sd.items():
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = v / mean_val
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_args", type=Path, default=None)
    ap.add_argument("--reference_ckpt", type=Path, default=None,
                    help="checkpoint path that contains args (state['args'])")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--fisher_max_batches", type=int, default=200)
    ap.add_argument("--fisher_batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--fisher_floor", type=float, default=1e-8)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.reference_args is None and args.reference_ckpt is None:
        raise ValueError("Provide --reference_args or --reference_ckpt")
    if args.reference_args is not None:
        ref_args = _load_reference_args(args.reference_args)
        reference_source = str(args.reference_args)
    else:
        ref_args = _load_args_from_checkpoint(args.reference_ckpt)
        reference_source = str(args.reference_ckpt)
    if args.device:
        ref_args.device = args.device
    device = torch.device(ref_args.device)

    batch_size = int(args.fisher_batch_size or getattr(ref_args, "batch_size", 256))
    num_workers = int(args.num_workers if args.num_workers is not None else getattr(ref_args, "num_workers", 4))

    ref_args, dataset, train_loader = _build_loader(ref_args, batch_size, num_workers)
    model_conf = bool(getattr(ref_args, "model_confidence", False))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_path in args.checkpoints:
        model = load_model_from_checkpoint(ref_args, dataset, ckpt_path).to(device)
        if hasattr(model, "set_actnorm_init"):
            model.set_actnorm_init()
        fisher_params, num_batches, num_samples = _accumulate_fisher(
            model,
            train_loader,
            device=device,
            model_confidence=model_conf,
            max_batches=int(args.fisher_max_batches),
            fisher_floor=float(args.fisher_floor),
        )

        state_dict = model.state_dict()
        fisher_full = {}
        for k, v in state_dict.items():
            if k in fisher_params:
                fisher_full[k] = fisher_params[k].detach().cpu()
            else:
                fisher_full[k] = torch.full_like(v.detach().cpu(), float(args.fisher_floor))

        fisher_full = _normalize_fisher_sd_like_model_sd(
            fisher_full,
            state_dict,
            fisher_floor=float(args.fisher_floor),
        )

        if args.normalize:
            fisher_full = _normalize_fisher(fisher_full, eps=float(args.fisher_floor))
        fisher_full = {
            k: v.detach().cpu() if torch.is_tensor(v) else v
            for k, v in fisher_full.items()
        }

        meta = {
            "checkpoint": ckpt_path,
            "reference_source": reference_source,
            "device": str(device),
            "num_batches": int(num_batches),
            "num_samples": int(num_samples),
            "fisher_max_batches": int(args.fisher_max_batches),
            "fisher_batch_size": int(batch_size),
            "fisher_floor": float(args.fisher_floor),
            "normalize": bool(args.normalize),
        }

        out_name = f"fisher_{_sanitize_name(ckpt_path)}.pt"
        out_path = args.output_dir / out_name
        torch.save({"fisher": fisher_full, "metadata": meta}, out_path)
        print(f"[DONE] saved fisher: {out_path}")


if __name__ == "__main__":
    main()

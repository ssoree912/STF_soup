#!/usr/bin/env python3
"""
Auto soup pipeline for STG-NF checkpoints.

What it does (one shot):
  1) Load checkpoints → run pairwise diagnostics (CKA / FID_like / JS_div / Grad_Cosine)
  2) Classify each pair as SAFE / OK / UNSAFE using robust defaults (tunable via CLI)
  3) Build clusters using SAFE/OK edges → do Fisher merging *within each cluster*
     - Options: BN-adapt before/after, per-layer scale matching, Fisher normalization + clipping
     - Optional sparse-mask preservation via mask intersection (if masks exist)
  4) Optionally do one round of cross-cluster mixing with small α line-search (conservative)
  5) Save all artifacts + pairwise metrics JSON.

Drop this file under your repo's tools/ (or anywhere) and run with the example at the bottom.

Assumptions:
  - Uses your existing utilities: args.init_sub_args, dataset.get_dataset_and_loader,
    fisher_soup_stg_nf.load_models_and_fishers, utils.* helpers from your project.
  - STG_NF forward supports (sample, label, score) returning (latent, nll).

Tested against your diagnose script structure/logs you shared.
"""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# --- project-local imports (same as your diagnose script) ---
from args import init_sub_args
from dataset import get_dataset_and_loader
from fisher_soup_stg_nf import load_models_and_fishers
from models.STG_NF.model_pose import STG_NF
from utils.data_utils import trans_list
from utils.train_utils import init_model_params
# optional: use your project's scoring utility to compute AUROC
try:
    from utils.scoring_utils import score_dataset  # expected to return (roc, auroc) or similar
except Exception:
    score_dataset = None

# =============================================================================
# BN-adapt (update only BN running stats)
# =============================================================================
@torch.no_grad()
def bn_adapt(model: torch.nn.Module, loader, device: torch.device, steps: int = 400,
            use_model_confidence: bool = False) -> None:
    model.train()
    it = iter(loader)
    for _ in range(max(steps, 0)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        data = [d.to(device, non_blocking=True) for d in batch]
        clip_score = data[-2].amin(dim=-1)
        sample = data[0] if use_model_confidence else data[0][:, :2]
        model(sample.float(), label=torch.ones(sample.shape[0], device=device), score=clip_score)
    model.eval()

# =============================================================================
# Scale matching (per-layer weight std alignment)
# =============================================================================

def _layer_std(t: torch.Tensor) -> torch.Tensor:
    return t.std().clamp_min(1e-8)


def scale_match_state(src: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, w in src.items():
        if w.dtype.is_floating_point and w.dim() >= 2 and w.numel() > 16 and k in ref:
            out[k] = w * (_layer_std(ref[k]) / _layer_std(w))
        else:
            out[k] = w
    return out

# =============================================================================
# Fisher merging (layer-normalized + optional percentile clipping + mask preservation)
# =============================================================================

def fisher_merge(states: List[Dict[str, torch.Tensor]],
                 fishers: List[Optional[Dict[str, torch.Tensor]]],
                 mask: Optional[Dict[str, torch.Tensor]] = None,
                 norm: str = "mean",
                 clip_pctl: int = 95) -> Dict[str, torch.Tensor]:
    assert len(states) >= 2
    # Prepare normalized fishers (fallback ones if missing)
    Fn: List[Dict[str, torch.Tensor]] = []
    for i, Fi in enumerate(fishers or []):
        if Fi is None or len(Fi) == 0:
            # uniform fallback: ones on float params
            Fi = {k: torch.ones_like(v) for k, v in states[i].items() if v.dtype.is_floating_point}
        # layer normalization
        Fnorm: Dict[str, torch.Tensor] = {}
        for k, v in Fi.items():
            if norm == "mean":
                scale = v.mean().abs() + 1e-8
            elif norm == "trace":
                scale = v.abs().sum() + 1e-8
            else:
                scale = torch.tensor(1.0, dtype=v.dtype)
            Fnorm[k] = v / scale
        # percentile clip to suppress spikes
        if clip_pctl and 0 < clip_pctl < 100:
            for k, v in Fnorm.items():
                try:
                    th = torch.quantile(v.flatten().abs(), clip_pctl / 100.0)
                    Fnorm[k] = torch.clamp(v, max=th)
                except Exception:
                    pass
        Fn.append(Fnorm)

    keys = states[0].keys()
    merged: Dict[str, torch.Tensor] = {}
    for k in keys:
        num = None
        den = None
        for i in range(len(states)):
            wi = states[i].get(k, None)
            Fi = Fn[i].get(k, None)
            if wi is None or Fi is None:
                continue
            term = Fi * wi
            num = term if num is None else (num + term)
            den = Fi if den is None else (den + Fi)
        if num is None:
            # keep first
            w = states[0][k]
        else:
            w = num / (den + 1e-8)
        if mask is not None and k in mask:
            w = mask[k] * w
        merged[k] = w
    return merged

# =============================================================================
# Diagnostics (CKA / FID_like / JS_div / Spearman / Grad-Cos)
# =============================================================================

@torch.no_grad()
def collect_feats_scores(model: STG_NF, loader, ref_args, device: torch.device,
                         max_batches: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats: List[torch.Tensor] = []
    scores: List[torch.Tensor] = []
    processed = 0
    for batch in loader:
        data = [d.to(device, non_blocking=True) for d in batch]
        clip_score = data[-2].amin(dim=-1)
        sample = data[0] if getattr(ref_args, "model_confidence", False) else data[0][:, :2]
        z, nll = model(sample.float(), label=torch.ones(sample.shape[0], device=device), score=clip_score)
        feats.append(z.reshape(z.shape[0], -1).cpu())
        scores.append((-nll).reshape(-1, 1).cpu())
        processed += 1
        if max_batches is not None and processed >= max_batches:
            break
    return torch.cat(feats, 0), torch.cat(scores, 0)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X.T @ Y).pow(2).sum()
    var1 = (X.T @ X).pow(2).sum().clamp_min(1e-8)
    var2 = (Y.T @ Y).pow(2).sum().clamp_min(1e-8)
    return float(hsic / torch.sqrt(var1 * var2))


def _gaussian_stats(Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = Z.mean(0)
    Zc = Z - mu
    cov = (Zc.T @ Zc) / max(Z.shape[0] - 1, 1)
    return mu, cov


def _trace_sqrt_prod(A: torch.Tensor, B: torch.Tensor) -> float:
    try:
        eigvals = torch.linalg.eigvals(A @ B).real.clamp(min=0)
        return torch.sqrt(eigvals).sum().item()
    except Exception:
        return 0.0


def fid_like(mu1: torch.Tensor, cov1: torch.Tensor, mu2: torch.Tensor, cov2: torch.Tensor) -> float:
    diff = mu1 - mu2
    return float(diff.dot(diff) + torch.trace(cov1 + cov2) - 2.0 * _trace_sqrt_prod(cov1, cov2))


def js_divergence(scores1: torch.Tensor, scores2: torch.Tensor) -> float:
    # scores: [N,1] anomaly score → sigmoid to [0,1], make pseudo-2class
    if scores1.shape[1] == 1:
        p1 = torch.sigmoid(scores1)
        p2 = torch.sigmoid(scores2)
        P = torch.cat([p1, 1 - p1], dim=1)
        Q = torch.cat([p2, 1 - p2], dim=1)
    else:
        P = F.softmax(scores1, dim=1)
        Q = F.softmax(scores2, dim=1)
    eps = 1e-8
    P = P / (P.sum(1, keepdim=True) + eps)
    Q = Q / (Q.sum(1, keepdim=True) + eps)
    M = 0.5 * (P + Q)
    KL = lambda A, B: (A * (A.add(eps).log() - B.add(eps).log())).sum(dim=1)
    return float(0.5 * KL(P, M).mean() + 0.5 * KL(Q, M).mean())


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    rx = torch.argsort(torch.argsort(x, dim=0), dim=0).float()
    ry = torch.argsort(torch.argsort(y, dim=0), dim=0).float()
    rx = (rx - rx.mean()) / (rx.std() + 1e-8)
    ry = (ry - ry.mean()) / (ry.std() + 1e-8)
    return float((rx * ry).mean())


def grad_cos_between(m1: STG_NF, m2: STG_NF, loader, ref_args, device: torch.device, samples: int = 8) -> float:
    m1.eval(); m2.eval()
    it = iter(loader)
    vals: List[float] = []
    for _ in range(samples):
        try:
            batch = next(it)
        except StopIteration:
            break
        def _grad_vec(m: STG_NF) -> torch.Tensor:
            # Ensure grads are enabled in case prior steps disabled them
            for p in m.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
            m.zero_grad(set_to_none=True)
            data = [d.to(device, non_blocking=True) for d in batch]
            clip_score = data[-2].amin(dim=-1)
            sample = data[0] if getattr(ref_args, "model_confidence", False) else data[0][:, :2]
            _, nll = m(sample.float(), label=torch.ones(sample.shape[0], device=device), score=clip_score)
            loss = nll.mean(); loss.backward()
            vec = torch.cat([p.grad.detach().flatten() for p in m.parameters()
                             if p.requires_grad and p.grad is not None], 0)
            m.zero_grad(set_to_none=True)
            return vec
        g1 = _grad_vec(m1); g2 = _grad_vec(m2)
        if g1.numel() != g2.numel():
            continue
        vals.append(float(torch.dot(g1, g2) / (g1.norm() * g2.norm() + 1e-8)))
    return sum(vals) / len(vals) if vals else float("nan")


def diagnose_pair(m1: STG_NF, m2: STG_NF, loader, ref_args, device: torch.device,
                  max_batches: Optional[int], grad_batches: int) -> Dict[str, float]:
    Z1, S1 = collect_feats_scores(m1, loader, ref_args, device, max_batches)
    Z2, S2 = collect_feats_scores(m2, loader, ref_args, device, max_batches)
    mu1, c1 = _gaussian_stats(Z1)
    mu2, c2 = _gaussian_stats(Z2)
    metrics: Dict[str, float] = {
        "CKA": linear_cka(Z1, Z2),
        "FID_like": fid_like(mu1, c1, mu2, c2),
        "JS_div": js_divergence(S1, S2),
        "Score_Spearman": spearman_corr(S1.flatten(), S2.flatten()),
    }
    if grad_batches > 0:
        metrics["Grad_Cosine"] = grad_cos_between(m1, m2, loader, ref_args, device, samples=grad_batches)
    else:
        metrics["Grad_Cosine"] = float("nan")
    return metrics

# =============================================================================
# Pair classification + cluster building
# =============================================================================

def classify(metrics: Dict[str, float], safe_cka=0.90, safe_grad=0.40, safe_fid=120,
             ok_cka=0.80, ok_grad=0.20, ok_fid=300) -> str:
    CKA = metrics.get("CKA", 0.0)
    GC  = metrics.get("Grad_Cosine", 0.0)
    FID = metrics.get("FID_like", float("inf"))
    if CKA >= safe_cka and GC >= safe_grad and FID <= safe_fid:
        return "SAFE"
    if CKA >= ok_cka and GC >= ok_grad and FID <= ok_fid:
        return "OK"
    return "UNSAFE"


def build_clusters(n_nodes: int, edge_labels: Dict[Tuple[int, int], str]) -> List[List[int]]:
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for (i, j), lab in edge_labels.items():
        if lab in ("SAFE", "OK"):
            adj[i].append(j)
            adj[j].append(i)
    seen = set()
    clusters: List[List[int]] = []
    for i in range(n_nodes):
        if i in seen:
            continue
        q = deque([i])
        cur: List[int] = []
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            cur.append(u)
            for v in adj[u]:
                if v not in seen:
                    q.append(v)
        if len(cur) >= 1:
            clusters.append(sorted(cur))
    return clusters

# =============================================================================
# Main
# =============================================================================

def try_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # PyTorch ≥2.4
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser(description="Auto diagnose → cluster → Fisher soup pipeline")
    ap.add_argument("--reference_args", type=Path, required=True, help="args.json from a training run")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="list of checkpoint paths")
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:0 ...")
    ap.add_argument("--max_batches", type=int, default=64)
    ap.add_argument("--grad_batches", type=int, default=8)
    ap.add_argument("--bn_before", type=int, default=500, help="BN-adapt steps per model before diagnostics")
    ap.add_argument("--bn_after", type=int, default=300, help="BN-adapt steps for merged outputs")
    ap.add_argument("--scale_match", action="store_true", help="align per-layer std before merging")
    ap.add_argument("--mask_name", default="pruning_mask.pt")
    ap.add_argument("--fisher_norm", choices=["mean", "trace"], default="mean")
    ap.add_argument("--fisher_clip_pctl", type=int, default=95)
    ap.add_argument("--output_dir", type=Path, default=Path("auto_soup_out"))
    ap.add_argument("--include_unsafe", action="store_true", help="also attempt to merge UNSAFE pairs/clusters")
    ap.add_argument("--eval_merged", action="store_true", help="evaluate AUROC for merged outputs if score_dataset is available")

    # thresholds (override if needed)
    ap.add_argument("--safe_cka", type=float, default=0.90)
    ap.add_argument("--safe_grad", type=float, default=0.40)
    ap.add_argument("--safe_fid", type=float, default=120)
    ap.add_argument("--ok_cka", type=float, default=0.80)
    ap.add_argument("--ok_grad", type=float, default=0.20)
    ap.add_argument("--ok_fid", type=float, default=300)

    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- data & model args
    ref = json.load(open(args.reference_args))
    ref_ns = argparse.Namespace(**ref)
    if args.device:
        ref_ns.device = args.device
    ref_ns.only_test = True
    ref_ns, model_args = init_sub_args(ref_ns)
    dataset, loader = get_dataset_and_loader(ref_ns, trans_list=trans_list, only_test=True)
    model_args = init_model_params(ref_ns, dataset)
    device = torch.device(ref_ns.device)
    test_loader = loader["test"]

    # ---- load models via project util (handles masks)
    models, fishers, masks = load_models_and_fishers(
        checkpoint_paths=args.checkpoints,
        fisher_paths=None,
        model_args=model_args,
        device=device,
        logger=None,
        mask_name=args.mask_name,
    )

    names = [str(p) for p in args.checkpoints]

    # ---- optional BN-adapt before diagnostics
    if args.bn_before > 0:
        for m in models:
            bn_adapt(m, test_loader, device, steps=args.bn_before,
                     use_model_confidence=getattr(ref_ns, "model_confidence", False))

    # ---- diagnostics for all pairs
    pair_metrics: Dict[Tuple[int, int], Dict[str, float]] = {}
    pair_labels: Dict[Tuple[int, int], str] = {}
    for i, j in combinations(range(len(models)), 2):
        met = diagnose_pair(models[i], models[j], test_loader, ref_ns, device, args.max_batches, args.grad_batches)
        lab = classify(met, args.safe_cka, args.safe_grad, args.safe_fid, args.ok_cka, args.ok_grad, args.ok_fid)
        pair_metrics[(i, j)] = met
        pair_labels[(i, j)] = lab
        print(f"[PAIR] {names[i]}  <->  {names[j]}  =>  {lab}  |  {json.dumps(met, indent=None)}")

    with open(args.output_dir / "pair_metrics.json", "w") as f:
        json.dump({f"{names[i]}||{names[j]}": v for (i, j), v in pair_metrics.items()}, f, indent=2)

    # ---- graph clusters from SAFE/OK edges
    clusters = build_clusters(len(models), pair_labels)
    print("[CLUSTERS]", [[names[i] for i in c] for c in clusters])

    # ---- cluster-wise Fisher merging
    result_paths: List[str] = []
    for idx, comp in enumerate(clusters):
        if len(comp) < 2:
            continue  # singletons are skipped
        states: List[Dict[str, torch.Tensor]] = []
        Fs: List[Optional[Dict[str, torch.Tensor]]] = []
        # compute mask intersection (preserve sparsity if masks exist)
        common_mask: Optional[Dict[str, torch.Tensor]] = None
        ref_sd_for_scale: Optional[Dict[str, torch.Tensor]] = None
        for k, i in enumerate(comp):
            # state_dict
            ckpt = try_torch_load(Path(args.checkpoints[i]))
            sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            if args.scale_match:
                if ref_sd_for_scale is None:
                    ref_sd_for_scale = sd
                else:
                    sd = scale_match_state(sd, ref_sd_for_scale)
            states.append(sd)
            # fisher (may be missing)
            Fi = fishers[i] if fishers is not None and len(fishers) > i else None
            Fs.append(Fi)
            # mask intersection
            if masks and len(masks) > i and masks[i]:
                if common_mask is None:
                    common_mask = masks[i]
                else:
                    # elementwise intersection (AND); fall back when key missing
                    common_mask = {kk: (common_mask[kk] * masks[i].get(kk, torch.zeros_like(common_mask[kk])))
                                   for kk in common_mask.keys() if kk in masks[i]}
        merged_sd = fisher_merge(states, Fs, mask=common_mask, norm=args.fisher_norm, clip_pctl=args.fisher_clip_pctl)
        out_path = args.output_dir / f"cluster{idx}_fisher.pth.tar"
        torch.save({"state_dict": merged_sd}, out_path)
        result_paths.append(str(out_path))
        print("[WRITE]", out_path)

    # ---- cross-cluster conservative mixing (top 2 clusters only)
    if len(result_paths) >= 2:
        base_sd = try_torch_load(Path(result_paths[0]))["state_dict"]
        other_sd = try_torch_load(Path(result_paths[1]))["state_dict"]
        for a in [0.2, 0.35, 0.5]:
            mix = {k: (1 - a) * base_sd[k] + a * other_sd.get(k, base_sd[k]) for k in base_sd.keys()}
            out = args.output_dir / f"cross_a{a:.2f}.pth.tar"
            torch.save({"state_dict": mix}, out)
            result_paths.append(str(out))
            print("[WRITE]", out)

    # ---- optional BN-adapt after merging (load merged → BN-adapt → save back)
    if args.bn_after > 0 and len(result_paths) > 0:
        for rp in result_paths:
            # re-load as a model using your project loader to ensure exact arch
            model_list, _, _ = load_models_and_fishers(
                checkpoint_paths=[rp], fisher_paths=None,
                model_args=model_args, device=device, logger=None, mask_name=args.mask_name
            )
            m = model_list[0]
            bn_adapt(m, test_loader, device, steps=args.bn_after,
                     use_model_confidence=getattr(ref_ns, "model_confidence", False))
            # save back
            torch.save({"state_dict": m.state_dict()}, rp)
            print("[BN-ADAPT SAVED]", rp)

    # ---- optional evaluation of merged outputs (AUROC)
    if args.eval_merged and result_paths:
        if score_dataset is None:
            print("[EVAL] utils.scoring_utils.score_dataset not found. Skipping AUROC eval.")
        else:
            eval_summ = {}
            for rp in result_paths:
                # load merged checkpoint as model
                model_list, _, _ = load_models_and_fishers(
                    checkpoint_paths=[rp], fisher_paths=None,
                    model_args=model_args, device=device, logger=None, mask_name=args.mask_name
                )
                m = model_list[0]
                # compute ROC/AUROC using project utility
                try:
                    roc, auroc = score_dataset(m, loader, ref_ns)  # adjust if your util expects (model, loaders, args)
                except TypeError:
                    # fallback: try common signature variants
                    try:
                        roc, auroc = score_dataset(m, loader["test"], ref_ns)
                    except Exception as e:
                        print(f"[EVAL] score_dataset failed for {rp}: {e}")
                        continue
                eval_summ[rp] = {"AUROC": float(auroc)} if isinstance(auroc, (int, float)) else {"metric": str(auroc)}
                print(f"[EVAL] {rp} => AUROC: {eval_summ[rp]}")
            with open(args.output_dir / "eval_results.json", "w") as f:
                json.dump(eval_summ, f, indent=2)
            print("[WRITE]", args.output_dir / "eval_results.json")

    print("[DONE] generated:", result_paths)


if __name__ == "__main__":
    main()

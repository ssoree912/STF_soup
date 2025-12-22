import numpy as np
import torch

from models.training import compute_loss


def _zero_fisher_like(model: torch.nn.Module):
    return {
        name: torch.zeros_like(param, device=param.device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def compute_occ_fisher(model, loader, device, max_batches=50, fisher_floor=1e-8, normalize=False):
    """
    배치 단위로 모델의 피서 정보 계산
    수식 : F_i = E_data[(∂/∂θ_i log p(x|θ))^2]

    """
    model.eval()
    device = torch.device(device)
    model.to(device)
    fisher = _zero_fisher_like(model)
    total_batches = 0
    for batch_idx, data_arr in enumerate(loader):
        if batch_idx >= max_batches:
            break
        data = [d.to(device, non_blocking=True) for d in data_arr]
        score = data[-2].amin(dim=-1)
        label = data[-1]
        if hasattr(model, "args") and getattr(model.args, "model_confidence", False):
            samp = data[0]
        else:
            samp = data[0][:, :2]

        model.zero_grad(set_to_none=True)
        _, nll = model(samp.float(), label=label, score=score)
        if nll is None:
            continue
        loss = compute_loss(nll, reduction="mean")["total_loss"]
        loss.backward()
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            fisher[name] += param.grad.detach() ** 2
        total_batches += 1

    total_batches = max(total_batches, 1)
    for name in fisher:
        fisher[name] = torch.clamp(fisher[name] / total_batches, min=fisher_floor)
        if normalize:
            norm = torch.norm(fisher[name])
            if norm > 0:
                fisher[name] = fisher[name] / norm

    snapshot = {k: v.to('cpu') for k, v in fisher.items()}
    metadata = {
        "num_batches": total_batches,
        "fisher_floor": fisher_floor,
        "normalized": bool(normalize),
    }
    return {"fisher": snapshot, "metadata": metadata}


def compute_occ_metrics(nll_values):
    nll_values = np.asarray(nll_values)
    nll_mean = float(np.mean(nll_values))
    nll_var = float(np.var(nll_values))
    return {
        "nll_mean": nll_mean,
        "nll_var": nll_var,
        "neg_nll_mean": -nll_mean,
        "neg_nll_var": -nll_var,
    }


def compute_zscores(metric_list, keys):
    if not metric_list:
        return []
    stacked = {key: np.array([metrics[key] for metrics in metric_list]) for key in keys}
    zscore_list = []
    for metrics in metric_list:
        z_scores = {}
        for key in keys:
            values = stacked[key]
            mean = values.mean()
            std = values.std()
            if std == 0:
                z = 0.0
            else:
                z = (metrics[key] - mean) / std
            z_scores[key] = float(z)
        zscore_list.append(z_scores)
    return zscore_list

import math
import random
import re
from collections import defaultdict
from typing import Iterable, List, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.STG_NF.model_pose import STG_NF
from models.STG_NF.modules_pose import gaussian_likelihood, gaussian_p
from utils.train_utils import init_model_params

#특정 데이터 삭제를 위한 인덱스 반환
class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Iterable[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        data, trans_index, score, label = self.dataset[self.indices[idx]]
        return data, self.indices[idx], score, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_indexed_loader(
    dataset: Dataset,
    indices: Iterable[int],
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        IndexedDataset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

#캐시/DF 의 후보가 될 전체 인덱스 목록(정상 샘플만 포함)
def get_base_indices(dataset: Dataset, normal_only: bool = True) -> List[int]:
    if hasattr(dataset, "num_samples"):
        base_indices = list(range(dataset.num_samples))
    else:
        base_indices = list(range(len(dataset)))
    if normal_only and hasattr(dataset, "labels"):
        base_indices = [i for i in base_indices if dataset.labels[i] == 1]
    return base_indices

#confidence score 축소 : 어느 한부분이라도 confidence가 낮으면 전체 신뢰도가 낮아지도록
def reduce_conf_score(score: torch.Tensor) -> torch.Tensor:
    if score.ndim > 1:
        return score.amin(dim=-1)
    return score


def prepare_batch(
    batch,
    device: torch.device,
    model_confidence: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, _, score, label = batch
    x = x.to(device, non_blocking=True).float()
    if not model_confidence:
        x = x[:, :2]
    score = score.to(device, non_blocking=True)
    label = label.to(device, non_blocking=True)
    return x, score, label

#DF 선택용 신호 추출 : z_steps, nll_steps, emb, nll_clip    
@torch.no_grad()
def extract_steps_and_emb(
    model: STG_NF,
    x: torch.Tensor,
    label: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z, logdet = model.flow(x, reverse=False)
    mean, logs = model.prior(x, label) 
    objective = logdet + gaussian_likelihood(mean, logs, z) #
    denom = math.log(2.0) * x.size(1) * x.size(2) * x.size(3) 
    nll_clip = (-objective) / denom #전체 클립의 NLL. 높을 수록 이 데이터를 이상하다고 판단(DF1에서 tail)

    z_steps = z.permute(0, 2, 1, 3).reshape(z.size(0), z.size(2), -1) #latent z를 시간축 기준으로 펼쳐서 프레임/타임스텝별 표현(DF2용)
    emb = z.mean(dim=2).reshape(z.size(0), -1) #latent vecotr z의 평균값. 해당 데이터의 특징을 압축한 임베딩 벡터(클러스터링용)

    logp = gaussian_p(mean, logs, z)
    logp_t = logp.sum(dim=(1, 3))
    logdet_t = (logdet / z.size(2)).unsqueeze(1).expand_as(logp_t) #각 타임스텝별 로그 디터미넌트
    nll_steps = -(logp_t + logdet_t) / denom #시간별로 로그 우도를 나눠서 NLL 계산

    return z_steps, nll_steps, emb, nll_clip

# base 스코어 로그를 만드는 함수 
@torch.no_grad()
def build_base_cache(
    model: STG_NF,
    loader: DataLoader,
    device: torch.device,
    model_confidence: bool = False,
    use_conf_score: bool = False,
) -> Dict[str, Dict[int, np.ndarray]]:
    sB, z_steps_cache, nll_steps_cache, emb_cache = {}, {}, {}, {}
    model.eval()

    for batch in loader:
        x, score, label = prepare_batch(batch, device, model_confidence)
        z_steps, nll_steps, emb, nll_clip = extract_steps_and_emb(model, x, label=label)

        if use_conf_score:
            score_w = reduce_conf_score(score)
            nll_clip = nll_clip * score_w

        sid = batch[1]
        if torch.is_tensor(sid):
            sid = sid.tolist()

        nll_clip = nll_clip.detach().cpu().numpy()
        z_steps = z_steps.detach().cpu().numpy()
        nll_steps = nll_steps.detach().cpu().numpy()
        emb = emb.detach().cpu().numpy()

        for i, s in enumerate(sid):
            sB[int(s)] = float(nll_clip[i])
            z_steps_cache[int(s)] = z_steps[i]
            nll_steps_cache[int(s)] = nll_steps[i]
            emb_cache[int(s)] = emb[i]

    return {
        "sB": sB,
        "z_steps": z_steps_cache,
        "nll_steps": nll_steps_cache,
        "emb": emb_cache,
    }

# nll_clip 기준 상위 alpha 비율 선택 : 학습에 방해되는 노이즈 제거용
#sB 값 기준 내림차순
def select_df1_tail(sB: Dict[int, float], alpha: float = 0.01) -> List[int]:
    items = sorted(sB.items(), key=lambda kv: kv[1], reverse=True)
    k = max(1, int(len(items) * alpha))
    return [sid for sid, _ in items[:k]]

#인접 타임스텝 latent 차이 ||z_t - z_{t-1}|| 평균 : 값이 클수록 동적 변화가 큼
def dynamics_metric_from_zsteps(z_steps: np.ndarray) -> float:
    dz = np.linalg.norm(z_steps[1:] - z_steps[:-1], axis=-1) #인접한 프레임 간 차이
    return float(dz.mean())

#인접 타임스텝 NLL 변환량 ||nll_t - nll_{t-1}|| 평균 : 값이 클수록 동적 변화가 큼
def dynamics_metric_from_nll(nll_steps: np.ndarray) -> float:
    dn = np.abs(nll_steps[1:] - nll_steps[:-1])
    return float(dn.mean())

# z_steps 또는 nll_steps 기준 상위 alpha_g 비율 선택 : 동적 변화가 큰 데이터 선택용
def select_df2_dynamics(cache: Dict[str, Dict[int, np.ndarray]], alpha_g: float = 0.02) -> List[int]:
    scores = []
    if cache.get("z_steps"):
        for sid, zst in cache["z_steps"].items():
            scores.append((sid, dynamics_metric_from_zsteps(zst)))
    elif cache.get("nll_steps"):
        for sid, nst in cache["nll_steps"].items():
            scores.append((sid, dynamics_metric_from_nll(nst)))
    else:
        return []

    scores.sort(key=lambda kv: kv[1], reverse=True)
    k = max(1, int(len(scores) * alpha_g))
    return [sid for sid, _ in scores[:k]]
#robust 통계량 : 극단값 제거 후 평균 계산
def trimmed_mean(x: np.ndarray, trim_ratio: float = 0.1) -> float:
    if x.size == 0:
        return 0.0
    x_sorted = np.sort(x)
    k = int(len(x_sorted) * trim_ratio)
    if 2 * k >= len(x_sorted):
        return float(x_sorted.mean())
    return float(x_sorted[k:-k].mean())
#기본 ||∆z|| 계산한 뒤 normalized, relataive : ||Δz|| / (||z||+eps)로 상대 변화량 측정
def dynamics_metric_from_zsteps_v2(
    z_steps: np.ndarray,
    mode: str = "normalized",
    robust: bool = True,
    trim_ratio: float = 0.1,
    eps: float = 1e-6,
) -> float:
    dz_vec = z_steps[1:] - z_steps[:-1]
    dz = np.linalg.norm(dz_vec, axis=-1)

    if mode == "normalized":
        dim = float(z_steps.shape[-1])
        dz = dz / max(1.0, np.sqrt(dim))
    elif mode == "relative":
        base = np.linalg.norm(z_steps[:-1], axis=-1) + eps
        dz = dz / base

    if robust:
        return trimmed_mean(dz, trim_ratio)
    return float(dz.mean())


def select_df2_dynamics_v2(
    cache: Dict[str, Dict[int, np.ndarray]],
    alpha_g: float = 0.02,
    mode: str = "normalized",
    robust: bool = True,
    trim_ratio: float = 0.1,
    eps: float = 1e-6,
) -> List[int]:
    scores = []
    if cache.get("z_steps"):
        for sid, zst in cache["z_steps"].items():
            score = dynamics_metric_from_zsteps_v2(
                zst,
                mode=mode,
                robust=robust,
                trim_ratio=trim_ratio,
                eps=eps,
            )
            scores.append((sid, score))
    elif cache.get("nll_steps"):
        for sid, nst in cache["nll_steps"].items():
            dn = np.abs(nst[1:] - nst[:-1])
            score = trimmed_mean(dn, trim_ratio) if robust else float(dn.mean())
            scores.append((sid, score))
    else:
        return []

    scores.sort(key=lambda kv: kv[1], reverse=True)
    k = max(1, int(len(scores) * alpha_g))
    return [sid for sid, _ in scores[:k]]

#임베딩을 사용하여 K-평균 클러스터링 수행 후, nll_clip이 높은 클러스터 선택
def select_df3_subdomain(
    cache: Dict[str, Dict[int, np.ndarray]],
    val_sids: List[int],
    k_clusters: int = 20,
    top_clusters: int = 2,
    tau_q: float = 0.999,
) -> Tuple[List[int], Dict[str, object]]:
    s_val = np.array([cache["sB"][sid] for sid in val_sids], dtype=np.float32)
    tau_base = float(np.quantile(s_val, tau_q)) #sB(데이터들의 이상치점수)를 확인하여 상위%에 해당하는 컷오프 

    X, kept = [], []
    #emb(데이터 특징) 을 이용해 비슷한 데이터 끼리 k-평균 클러스터링 수행
    for sid in val_sids:
        if sid in cache["emb"]:
            X.append(cache["emb"][sid])
            kept.append(sid)
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] == 0:
        return [], {"tau_base": tau_base, "bad_clusters": [], "top_rates": []}

    k_clusters = max(1, min(k_clusters, X.shape[0]))
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k_clusters, random_state=0, n_init=10).fit(X)
    labels = km.labels_

    clusters = {k: [] for k in range(k_clusters)}
    for sid, lab in zip(kept, labels):
        clusters[int(lab)].append(sid)

    rates = []
    for k in range(k_clusters):
        ss = clusters[k]
        if not ss:
            continue
        r = float(np.mean([cache["sB"][sid] >= tau_base for sid in ss]))
        rates.append((k, r, len(ss)))

    rates.sort(key=lambda x: x[1], reverse=True)
    bad = [k for k, _, _ in rates[:top_clusters]]

    df3 = []
    for k in bad:
        df3.extend(clusters[k])

    info = {"tau_base": tau_base, "bad_clusters": bad, "top_rates": rates[:5]}
    return df3, info

#검증 데이터의 점수 기준으로 불량 기준선 
def select_df3_subdomain_train_df(
    cache_train: Dict[str, Dict[int, np.ndarray]],
    train_sids: List[int],
    cache_val: Dict[str, Dict[int, np.ndarray]],
    val_sids: List[int],
    k_clusters: int = 20,
    top_clusters: int = 2,
    tau_q: float = 0.999,
    return_cluster_ctx: bool = False,
) -> Union[
    Tuple[List[int], Dict[str, object]],
    Tuple[List[int], Dict[str, object], Optional[Dict[str, object]]],
]:
    s_val = np.array([cache_val["sB"][sid] for sid in val_sids], dtype=np.float32)
    tau_base = float(np.quantile(s_val, tau_q))

    Xtr, kept_tr = [], []
    for sid in train_sids:
        if sid in cache_train["emb"]:
            Xtr.append(cache_train["emb"][sid])
            kept_tr.append(sid)
    Xtr = np.asarray(Xtr, dtype=np.float32)
    if Xtr.shape[0] == 0:
        info = {"tau_base": tau_base, "bad_clusters": [], "top_rates": []}
        if return_cluster_ctx:
            return [], info, None
        return [], info

    k = max(1, min(k_clusters, Xtr.shape[0]))
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xtr)

    Xv, kept_v = [], []
    for sid in val_sids:
        if sid in cache_val["emb"]:
            Xv.append(cache_val["emb"][sid])
            kept_v.append(sid)
    Xv = np.asarray(Xv, dtype=np.float32)
    if Xv.shape[0] == 0:
        info = {"tau_base": tau_base, "bad_clusters": [], "top_rates": []}
        if return_cluster_ctx:
            return [], info, None
        return [], info

    val_labels = km.predict(Xv)
    clusters_val = {c: [] for c in range(k)}
    for sid, lab in zip(kept_v, val_labels):
        clusters_val[int(lab)].append(sid)

    rates = []
    for c in range(k):
        ss = clusters_val[c]
        if not ss:
            continue
        r = float(np.mean([cache_val["sB"][sid] >= tau_base for sid in ss]))
        rates.append((c, r, len(ss)))
    rates.sort(key=lambda x: x[1], reverse=True)

    bad_clusters = [c for c, _, _ in rates[:top_clusters]]

    train_labels = km.labels_
    clusters_train = {c: [] for c in range(k)}
    for sid, lab in zip(kept_tr, train_labels):
        clusters_train[int(lab)].append(sid)

    df3_train = []
    for c in bad_clusters:
        df3_train.extend(clusters_train.get(c, []))

    info = {
        "tau_base": tau_base,
        "k_clusters_used": k,
        "bad_clusters": bad_clusters,
        "top_rates": rates[:5],
        "val_cluster_sizes": {c: len(clusters_val[c]) for c in clusters_val},
        "train_cluster_sizes": {c: len(clusters_train[c]) for c in clusters_train},
    }
    if return_cluster_ctx:
        sid_to_cluster = {}
        for c in bad_clusters:
            for sid in clusters_train.get(c, []):
                sid_to_cluster[int(sid)] = int(c)
        cluster_ctx = {
            "centers": km.cluster_centers_.astype(np.float32),
            "sid_to_cluster": sid_to_cluster,
            "bad_clusters": list(bad_clusters),
        }
        return df3_train, info, cluster_ctx
    return df3_train, info


def _get_labels_map(dataset: Dataset, sids: List[int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if hasattr(dataset, "labels"):
        labels = dataset.labels
        for sid in sids:
            out[int(sid)] = int(labels[int(sid)])
        return out

    for sid in sids:
        item = dataset[int(sid)]
        label = item[-1]
        if torch.is_tensor(label):
            label = int(label.item())
        out[int(sid)] = int(label)
    return out


def select_df2_val_fp_train_df(
    cache_train: Dict[str, Dict[int, np.ndarray]],
    train_sids: List[int],
    cache_val: Dict[str, Dict[int, np.ndarray]],
    val_sids: List[int],
    val_dataset: Dataset,
    tau_base: float,
    normal_label: int = 1,
    knn_k: int = 20,
    budget_alpha: float = 0.02,
    budget_max: Optional[int] = None,
    seed: int = 0,
) -> Tuple[List[int], Dict[str, object]]:
    val_label_map = _get_labels_map(val_dataset, val_sids)

    fp_val = []
    for sid in val_sids:
        sid = int(sid)
        if val_label_map.get(sid, None) != int(normal_label):
            continue
        sb = cache_val.get("sB", {}).get(sid)
        if sb is None:
            continue
        if float(sb) >= float(tau_base):
            fp_val.append(sid)

    budget = max(1, int(len(train_sids) * float(budget_alpha)))
    if budget_max is not None:
        budget = min(budget, int(budget_max))

    tr_ids, tr_X = [], []
    emb_train = cache_train.get("emb", {})
    for sid in train_sids:
        sid = int(sid)
        emb = emb_train.get(sid, None)
        if emb is None:
            continue
        tr_ids.append(sid)
        tr_X.append(emb)
    tr_X = np.asarray(tr_X, dtype=np.float32)

    if tr_X.shape[0] == 0 or len(fp_val) == 0:
        info = {
            "tau_base": float(tau_base),
            "num_fp_val": int(len(fp_val)),
            "train_emb_n": int(tr_X.shape[0]),
            "budget": int(budget),
            "note": "empty fp_val or empty embeddings",
        }
        return [], info

    fp_ids, fp_X = [], []
    emb_val = cache_val.get("emb", {})
    for sid in fp_val:
        emb = emb_val.get(int(sid), None)
        if emb is None:
            continue
        fp_ids.append(int(sid))
        fp_X.append(emb)
    fp_X = np.asarray(fp_X, dtype=np.float32)
    if fp_X.shape[0] == 0:
        info = {
            "tau_base": float(tau_base),
            "num_fp_val": int(len(fp_val)),
            "train_emb_n": int(tr_X.shape[0]),
            "budget": int(budget),
            "note": "fp embeddings missing",
        }
        return [], info

    from sklearn.neighbors import NearestNeighbors
    knn_k = max(1, min(int(knn_k), tr_X.shape[0]))
    nn = NearestNeighbors(n_neighbors=knn_k, metric="euclidean")
    nn.fit(tr_X)
    dists, idxs = nn.kneighbors(fp_X, return_distance=True)

    freq = defaultdict(int)
    dist_sum = defaultdict(float)
    for q in range(idxs.shape[0]):
        for j in range(idxs.shape[1]):
            tr_sid = tr_ids[int(idxs[q, j])]
            freq[tr_sid] += 1
            dist_sum[tr_sid] += float(dists[q, j])

    items = []
    for sid, cnt in freq.items():
        items.append((sid, cnt, dist_sum[sid] / max(1, cnt)))
    items.sort(key=lambda x: (-x[1], x[2]))

    df2 = [sid for sid, _, _ in items[:budget]]
    info = {
        "tau_base": float(tau_base),
        "num_fp_val": int(len(fp_ids)),
        "train_emb_n": int(tr_X.shape[0]),
        "knn_k": int(knn_k),
        "budget_alpha": float(budget_alpha),
        "budget": int(budget),
        "top5": items[:5],
    }
    return df2, info


def _select_named_params(
    model: torch.nn.Module,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
) -> List[Tuple[str, torch.nn.Parameter]]:
    inc = re.compile(include_regex) if include_regex else None
    exc = re.compile(exclude_regex) if exclude_regex else None
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if inc and not inc.search(name):
            continue
        if exc and exc.search(name):
            continue
        params.append((name, p))
    return params


def _accumulate_mean_grads(
    model: STG_NF,
    loader: DataLoader,
    device: torch.device,
    model_confidence: bool,
    use_conf_score: bool,
    max_batches: int,
    include_regex: Optional[str],
    exclude_regex: Optional[str],
) -> Dict[str, torch.Tensor]:
    params = _select_named_params(model, include_regex, exclude_regex)
    if not params:
        raise ValueError("No parameters selected for gradient alignment.")

    grad_sum = {name: torch.zeros_like(p, device="cpu") for name, p in params}
    n = 0

    model.train()
    it = iter(loader)
    for _ in range(max_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        x, score, label = prepare_batch(batch, device, model_confidence)
        _, nll = model(x, label=label)
        if use_conf_score:
            nll = nll * reduce_conf_score(score)
        loss = nll.mean()

        model.zero_grad(set_to_none=True)
        loss.backward()

        for name, p in params:
            if p.grad is None:
                continue
            grad_sum[name] += p.grad.detach().cpu()
        n += 1

    if n > 0:
        for k in grad_sum:
            grad_sum[k] /= float(n)
    return grad_sum


def select_df3_grad_alignment_train_df(
    base_model: STG_NF,
    train_dataset: Dataset,
    train_sids: List[int],
    val_dataset: Dataset,
    val_sids: List[int],
    device: torch.device,
    model_confidence: bool,
    use_conf_score_val: bool,
    use_conf_score_train: bool,
    normal_label: int = 1,
    alpha: float = 0.02,
    sample_train: int = 4096,
    sample_val: int = 1024,
    seed: int = 0,
    batch_size: int = 1,
    num_workers: int = 0,
    include_regex: str = "prior",
    exclude_regex: Optional[str] = None,
    max_val_batches: int = 200,
) -> Tuple[List[int], Dict[str, object]]:
    rng = random.Random(seed)

    val_label_map = _get_labels_map(val_dataset, val_sids)
    val_norm = [int(s) for s in val_sids if val_label_map.get(int(s), None) == int(normal_label)]
    if not val_norm:
        return [], {"note": "no normal samples in val_sids"}

    if sample_val > 0 and len(val_norm) > sample_val:
        val_norm = rng.sample(val_norm, sample_val)

    val_loader = build_indexed_loader(
        val_dataset, val_norm, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    max_batches = max_val_batches
    if max_batches <= 0:
        max_batches = math.ceil(len(val_norm) / max(1, batch_size))
    g_val = _accumulate_mean_grads(
        base_model,
        val_loader,
        device=device,
        model_confidence=model_confidence,
        use_conf_score=use_conf_score_val,
        max_batches=int(max_batches),
        include_regex=include_regex,
        exclude_regex=exclude_regex,
    )

    train_pool = list(map(int, train_sids))
    if sample_train > 0 and len(train_pool) > sample_train:
        train_pool = rng.sample(train_pool, sample_train)
    if not train_pool:
        return [], {"note": "empty train pool after sampling"}

    train_loader = build_indexed_loader(
        train_dataset, train_pool, batch_size=1, num_workers=num_workers, shuffle=True
    )

    params = _select_named_params(base_model, include_regex, exclude_regex)
    if not params:
        raise ValueError("No parameters selected for df3' scoring.")

    scores = []
    base_model.train()
    for batch in train_loader:
        sid = batch[1]
        if torch.is_tensor(sid):
            sid = int(sid.item()) if sid.numel() == 1 else int(sid[0].item())
        elif isinstance(sid, (list, tuple)):
            sid = int(sid[0])
        else:
            sid = int(sid)

        x, score_t, label = prepare_batch(batch, device, model_confidence)
        _, nll = base_model(x, label=label)
        if use_conf_score_train:
            nll = nll * reduce_conf_score(score_t)
        loss = nll.mean()

        base_model.zero_grad(set_to_none=True)
        loss.backward()

        align = 0.0
        for name, p in params:
            if p.grad is None:
                continue
            gv = g_val.get(name, None)
            if gv is None:
                continue
            align += float((p.grad.detach().cpu() * gv).sum().item())
        scores.append((sid, align))

    scores.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(len(scores) * float(alpha)))
    df3 = [sid for sid, _ in scores[:k]]

    info = {
        "alpha": float(alpha),
        "k": int(k),
        "sample_train": int(len(scores)),
        "sample_val_norm": int(len(val_norm)),
        "include_regex": include_regex,
        "top5": scores[:5],
    }
    return df3, info


def load_model_from_checkpoint(args, dataset, checkpoint_path: str) -> STG_NF:
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required")
    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.set_actnorm_init()
    return model


@torch.no_grad()
def eval_nll_on_indices(
    model: STG_NF,
    dataset: Dataset,
    indices: List[int],
    device: torch.device,
    model_confidence: bool,
    batch_size: int,
    num_workers: int = 0,
    use_conf_score: bool = False,
) -> np.ndarray:
    loader = build_indexed_loader(
        dataset,
        indices,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    nll_list = []
    model.eval()
    for batch in loader:
        x, score, label = prepare_batch(batch, device, model_confidence)
        _, nll = model(x, label=label)
        if use_conf_score:
            nll = nll * reduce_conf_score(score)
        nll_list.append(nll.detach().cpu().numpy())
    return np.concatenate(nll_list, axis=0) if nll_list else np.array([])


@torch.no_grad()
def check_nll_consistency(
    model: STG_NF,
    loader: DataLoader,
    device: torch.device,
    model_confidence: bool,
    use_conf_score: bool = False,
    max_batches: int = 5,
) -> Dict[str, float]:
    model.eval()
    diffs = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x, score, label = prepare_batch(batch, device, model_confidence)
        _, nll_fwd = model(x, label=label)
        _, _, _, nll_clip = extract_steps_and_emb(model, x, label=label)

        if use_conf_score:
            w = reduce_conf_score(score)
            nll_fwd = nll_fwd * w
            nll_clip = nll_clip * w

        diff = (nll_fwd - nll_clip).abs().detach().cpu().numpy()
        diffs.append(diff)

    diffs = np.concatenate(diffs, axis=0) if diffs else np.array([0.0])
    return {
        "mean_abs_diff": float(diffs.mean()),
        "max_abs_diff": float(diffs.max()),
        "p95_abs_diff": float(np.quantile(diffs, 0.95)),
    }


def overlap_report(dfs: Dict[str, List[int]]) -> Dict[str, float]:
    sets = {k: set(v) for k, v in dfs.items()}
    keys = list(sets.keys())
    out = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            inter = len(sets[a] & sets[b])
            out["overlap_{}_{}".format(a, b)] = inter / max(1, len(sets[a]))
            out["overlap_{}_{}".format(b, a)] = inter / max(1, len(sets[b]))
    return out


def df_gradient_ascent(
    model: STG_NF,
    dataset: Dataset,
    df_indices: List[int],
    device: torch.device,
    model_confidence: bool,
    batch_size: int,
    num_workers: int = 0,
    use_conf_score: bool = False,
    lr: float = 3e-5,
    steps: int = 800,
    grad_clip: float = 1.0,
) -> STG_NF:
    loader = build_indexed_loader(
        dataset,
        df_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    it = iter(loader)
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x, score, label = prepare_batch(batch, device, model_confidence)
        _, nll = model(x, label=label)
        if use_conf_score:
            nll = nll * reduce_conf_score(score)
        loss = -nll.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    return model


def retrain_on_dr(
    model: STG_NF,
    dataset: Dataset,
    dr_indices: List[int],
    device: torch.device,
    model_confidence: bool,
    batch_size: int,
    num_workers: int = 0,
    use_conf_score: bool = False,
    lr: float = 5e-5,
    epochs: int = 2,
    grad_clip: float = 1.0,
) -> STG_NF:
    loader = build_indexed_loader(
        dataset,
        dr_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            x, score, label = prepare_batch(batch, device, model_confidence)
            _, nll = model(x, label=label)
            if use_conf_score:
                nll = nll * reduce_conf_score(score)
            loss = nll.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
    return model

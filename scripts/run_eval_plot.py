import argparse
import json
import os
import pickle
import sys

# Ensure project root is on sys.path for local imports.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.data_utils import trans_list
from utils.scoring_utils import get_dataset_scores, score_auc, score_dataset
from utils.unlearning_utils import eval_nll_on_indices, load_model_from_checkpoint

REPORT_GUIDE = {
    "A_setup": (
        "Base checkpoint, 평가 대상 모델, df_splits.json 경로와 val split 출처를 요약. "
        "tau_base는 val NLL의 tau_q 분위수이며 FPR은 mean(val_nll >= tau_base)로 계산."
    ),
    "B_selection_quality": (
        "DF의 base 난이도 확인: cache_train['sB']로 DF별 base 평균 NLL을 기록하면 좋음. "
        "DF1은 base NLL이 높게 나오는 것이 정상이며 DF2/DF3은 높지 않을 수도 있음."
    ),
    "C_intervention_effect": (
        "delta_df = df_nll_mean - base_df_mean으로 DF 밀어내기 효과 확인. "
        "타겟 DF에서 delta_df가 가장 크면 DF 정의와 업데이트 방향이 일치."
    ),
    "D_retain_quality": (
        "val_nll_mean, val_fpr 변화로 정상 분포 보존성 평가. "
        "val 분포가 함께 오른쪽으로 이동하면 부작용 가능."
    ),
    "E_utility": (
        "test AUROC/AUPRC/F1으로 다운스트림 유틸리티 평가. "
        "score 방향(클수록 이상/정상)은 scoring_utils 구현과 일치해야 함."
    ),
    "F_conclusion_template": (
        "dfX_retrain에서 dfX의 delta_df가 최대이면 타겟팅 적합. "
        "val 변화가 작으면 보존성 양호. "
        "test 성능이 유지/개선되지 않으면 DF에 유용한 정상 다양성이 포함됐을 수 있음."
    ),
    "notes": (
        "base_model을 로드하지만 base 수치는 cache의 sB로만 계산됨. "
        "test 평가의 use_conf_score 정책이 cache와 일치하는지 확인 필요."
    ),
}

def parse_args():
    parser = init_parser()
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--cache_path_val", type=str, default=None)
    parser.add_argument("--df_splits_path", type=str, required=True)
    parser.add_argument("--models", type=str, required=True, help="Comma-separated checkpoint paths")
    parser.add_argument("--model_names", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs/unlearning_eval")
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def best_f1(scores, gt):
    """
    [평가 목적] score threshold를 스윕하며 F1의 최댓값을 추정.
    [주의] 분위수 구간(0.01~0.99)만 탐색하므로 극단값은 놓칠 수 있음.
    """
    from sklearn.metrics import f1_score
    thresholds = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    best = 0.0
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        best = max(best, f1_score(gt, pred))
    return best


def eval_test_metrics(scores, dataset, args):
    """
    [평가 목적] test 전체 AUROC/AUPRC/F1 산출 (train_eval.py와 동일한 스코어링 루틴).
    [주의] score 방향(클수록 anomaly/normal)이 scoring_utils와 일치해야 함.
    """
    auc, scores_np = score_dataset(scores, dataset.metadata, args=args)
    gt_arr, scores_arr = get_dataset_scores(scores, dataset.metadata, args=args)
    gt_np = np.concatenate(gt_arr)
    from sklearn.metrics import average_precision_score
    auprc = average_precision_score(gt_np, scores_np)
    f1 = best_f1(scores_np, gt_np)
    return {"auc": float(auc), "auprc": float(auprc), "f1": float(f1)}


def plot_hist(base_vals, after_vals, out_path, title):
    """[분포 비교] base vs after NLL 히스토그램으로 분포 이동 확인."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.hist(base_vals, bins=50, alpha=0.5, label="base", density=True)
    plt.hist(after_vals, bins=50, alpha=0.5, label="after", density=True)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_ecdf(base_vals, after_vals, out_path, title):
    """[분포 비교] 꼬리(tail) 변화를 보기 위한 ECDF."""
    import matplotlib.pyplot as plt
    def ecdf(x):
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y
    xb, yb = ecdf(base_vals)
    xa, ya = ecdf(after_vals)
    plt.figure(figsize=(6, 4))
    plt.plot(xb, yb, label="base")
    plt.plot(xa, ya, label="after")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scatter(base_vals, after_vals, out_path, title):
    """[샘플별 변화] base NLL(x) -> after NLL(y) 이동을 산점도로 확인."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.scatter(base_vals, after_vals, s=6, alpha=0.4)
    min_v = min(base_vals.min(), after_vals.min())
    max_v = max(base_vals.max(), after_vals.max())
    plt.plot([min_v, max_v], [min_v, max_v], color="black", linewidth=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    """
    DF 품질 평가 요약:
    1) DF별 delta_df로 타겟팅 적합성 평가
    2) val_nll_mean/fpr로 정상 분포 보존성 평가
    3) test AUROC/AUPRC/F1로 유틸리티 평가
    4) 분포 플롯(옵션)으로 정성 확인
    """
    args = parse_args()
    args, _ = init_sub_args(args)

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for base model evaluation")

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cache_path, "rb") as f:
        cache_train = pickle.load(f)
    with open(args.df_splits_path, "r") as f:
        df_splits = json.load(f)
    cache_val_path = args.cache_path_val or df_splits.get("cache_val_path") or args.cache_path
    with open(cache_val_path, "rb") as f:
        cache_val = pickle.load(f)

    df_sets = {k: v for k, v in df_splits["df_sets"].items()}
    val_sids = df_splits.get("val_sids", list(cache_val["sB"].keys()))
    tau_base = float(df_splits.get("tau_base", np.quantile(
        np.array([cache_val["sB"][sid] for sid in val_sids]), 0.999
    )))

    dataset, _ = get_dataset_and_loader(args, trans_list=trans_list, only_test=False)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    val_split = cache_val.get("meta", {}).get("split") or df_splits.get("val_split", "train")
    val_dataset = dataset["test"] if val_split == "test" else dataset["train"]
    device = torch.device(args.device)
    use_conf_score_df = cache_train.get("meta", {}).get("use_conf_score", args.model_confidence)
    use_conf_score_val = cache_val.get("meta", {}).get("use_conf_score", args.model_confidence)
    use_conf_score_test = args.model_confidence

    # base 수치는 cache_*에서만 계산하며, base_model은 호환성 확인용으로 로드.
    base_model = load_model_from_checkpoint(args, dataset, args.checkpoint).to(device)

    model_paths = [p.strip() for p in args.models.split(",") if p.strip()]
    if args.model_names:
        model_names = [n.strip() for n in args.model_names.split(",")]
    else:
        model_names = [os.path.splitext(os.path.basename(p))[0] for p in model_paths]

    results = {
        "meta": {
            "base_checkpoint": args.checkpoint,
            "df_splits_path": args.df_splits_path,
            "cache_train_path": args.cache_path,
            "cache_val_path": cache_val_path,
            "val_split": val_split,
            "tau_base": tau_base,
            "fpr_definition": "mean(val_nll >= tau_base)",
            "use_conf_score_df": bool(use_conf_score_df),
            "use_conf_score_val": bool(use_conf_score_val),
            "use_conf_score_test": bool(use_conf_score_test),
        },
        "report_guide": REPORT_GUIDE,
        "base": {},
        "models": {},
    }

    base_val = np.array([cache_val["sB"][sid] for sid in val_sids], dtype=np.float32)
    base_val_mean = float(base_val.mean()) if base_val.size else 0.0
    results["base"]["val_nll_mean"] = base_val_mean
    results["base"]["fpr_val"] = float(np.mean(base_val >= tau_base))

    for name, path in zip(model_names, model_paths):
        model = load_model_from_checkpoint(args, dataset, path).to(device)

        model_result = {"df": {}, "val": {}}
        val_nll = eval_nll_on_indices(
            model,
            val_dataset,
            val_sids,
            device=device,
            model_confidence=args.model_confidence,
            batch_size=args.batch_size_eval,
            num_workers=args.num_workers,
            use_conf_score=use_conf_score_val,
        )
        model_val_mean = float(val_nll.mean()) if val_nll.size else 0.0
        model_result["val"]["nll_mean"] = model_val_mean
        model_result["val"]["delta_val_mean"] = model_val_mean - base_val_mean
        model_result["val"]["fpr"] = float(np.mean(val_nll >= tau_base)) if val_nll.size else 0.0

        for df_name, df_sids in df_sets.items():
            if not df_sids:
                continue
            base_df = np.array([cache_train["sB"][sid] for sid in df_sids], dtype=np.float32)
            base_df_mean = float(base_df.mean()) if base_df.size else 0.0
            df_nll = eval_nll_on_indices(
                model,
                train_dataset,
                df_sids,
                device=device,
                model_confidence=args.model_confidence,
                batch_size=args.batch_size_eval,
                num_workers=args.num_workers,
                use_conf_score=use_conf_score_df,
            )
            df_nll_mean = float(df_nll.mean()) if df_nll.size else 0.0
            model_result["df"][df_name] = {
                "base_df_mean": base_df_mean,
                "nll_mean": df_nll_mean,
                "delta_df": df_nll_mean - base_df_mean,
            }

            if not args.no_plots:
                try:
                    if df_nll.size and base_df.size:
                        plot_scatter(
                            base_df,
                            df_nll,
                            os.path.join(args.out_dir, "{}_{}_scatter.png".format(df_name, name)),
                            "{} scatter".format(df_name),
                        )
                except ImportError:
                    pass

        if not args.no_plots:
            try:
                if val_nll.size and base_val.size:
                    plot_hist(
                        base_val,
                        val_nll,
                        os.path.join(args.out_dir, "val_hist_{}.png".format(name)),
                        "Val NLL hist",
                    )
                    plot_ecdf(
                        base_val,
                        val_nll,
                        os.path.join(args.out_dir, "val_ecdf_{}.png".format(name)),
                        "Val NLL ECDF",
                    )
            except ImportError:
                pass

        test_indices = list(range(len(test_dataset)))
        test_nll = eval_nll_on_indices(
            model,
            test_dataset,
            test_indices,
            device=device,
            model_confidence=args.model_confidence,
            batch_size=args.batch_size_eval,
            num_workers=args.num_workers,
            use_conf_score=use_conf_score_test,
        )
        test_scores = -1.0 * test_nll if test_nll.size else np.array([])
        if test_scores.size:
            model_result["test_metrics"] = eval_test_metrics(test_scores, test_dataset, args)

        results["models"][name] = model_result

    results_path = os.path.join(args.out_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Eval results saved to {}".format(results_path))


if __name__ == "__main__":
    main()

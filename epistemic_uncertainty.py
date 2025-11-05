import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

from models.STG_NF.model_pose import STG_NF


class EpistemicUncertainty:
    """에피스테믹 불확실성(모델 간 분산) 계산 클래스"""
    
    def __init__(self, models: List[STG_NF], device: torch.device, 
                 logger: Optional[logging.Logger] = None):
        self.models = models
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.n_models = len(models)
        
        # 모든 모델을 평가 모드로 설정
        for model in self.models:
            model.eval()
    
    def compute_log_likelihoods(self, dataloader, args, max_batches: int = None) -> Dict[str, torch.Tensor]:
        """모든 모델에 대해 log-likelihood 계산"""
        self.logger.info(f"Computing log-likelihoods for {self.n_models} models...")
        
        # 각 모델별 log-likelihood 저장
        model_logps = {f"model_{i}": [] for i in range(self.n_models)}
        sample_indices = []
        
        batch_count = 0
        with torch.no_grad():
            for batch_idx, data_arr in enumerate(tqdm(dataloader, desc="Computing log-likelihoods")):
                if max_batches and batch_idx >= max_batches:
                    break
                    
                data = [d.to(self.device, non_blocking=True) for d in data_arr]
                score = data[-2].amin(dim=-1)
                
                # 데이터 전처리 (기존 방식과 동일)
                if getattr(args, 'model_confidence', False):
                    samp = data[0]
                else:
                    samp = data[0][:, :2]
                
                batch_size = samp.shape[0]
                labels = torch.ones(batch_size, device=self.device)
                
                # 각 모델에 대해 log-likelihood 계산
                for model_idx, model in enumerate(self.models):
                    try:
                        _, nll = model(samp.float(), label=labels, score=score)
                        
                        if getattr(args, 'model_confidence', False):
                            nll = nll * score
                        
                        # log-likelihood = -nll
                        log_p = -nll.cpu()
                        model_logps[f"model_{model_idx}"].append(log_p)
                        
                    except Exception as e:
                        self.logger.warning(f"Error computing log-likelihood for model {model_idx}: {e}")
                        # 에러 시 더미 값으로 채움
                        dummy_logp = torch.full((batch_size,), -10.0)
                        model_logps[f"model_{model_idx}"].append(dummy_logp)
                
                # 샘플 인덱스 저장
                start_idx = batch_count * batch_size
                sample_indices.extend(range(start_idx, start_idx + batch_size))
                batch_count += 1
        
        # 리스트를 텐서로 변환
        for key in model_logps:
            model_logps[key] = torch.cat(model_logps[key], dim=0)
        
        self.logger.info(f"Computed log-likelihoods for {len(sample_indices)} samples")
        return model_logps
    
    def compute_epistemic_uncertainty(self, model_logps: Dict[str, torch.Tensor], 
                                    eps: float = 1e-8) -> torch.Tensor:
        """에피스테믹 불확실성(모델 간 분산) 계산"""
        self.logger.info("Computing epistemic uncertainty (model variance)...")
        
        # 모든 모델의 log-likelihood를 스택
        logp_stack = torch.stack([model_logps[f"model_{i}"] for i in range(self.n_models)], dim=0)
        
        # 샘플별 모델 간 분산 계산
        epistemic_var = torch.var(logp_stack, dim=0, unbiased=True)
        
        # 불확실성이 너무 작으면 안정성을 위해 최소값 설정
        epistemic_var = torch.clamp(epistemic_var, min=eps)
        
        self.logger.info(f"Epistemic uncertainty stats - Mean: {epistemic_var.mean():.6f}, "
                        f"Std: {epistemic_var.std():.6f}, "
                        f"Min: {epistemic_var.min():.6f}, Max: {epistemic_var.max():.6f}")
        
        return epistemic_var
    
    def compute_uncertainty_weights(self, epistemic_var: torch.Tensor, 
                                  eps: float = 1e-8) -> torch.Tensor:
        """불확실성을 가중치로 변환 (불확실성 역수)"""
        # w(x) = 1 / (Var[log p_k(x)] + eps)
        weights = 1.0 / (epistemic_var + eps)
        
        # 가중치 정규화 (선택적)
        weights = weights / weights.mean()
        
        self.logger.info(f"Uncertainty weights stats - Mean: {weights.mean():.6f}, "
                        f"Std: {weights.std():.6f}, "
                        f"Min: {weights.min():.6f}, Max: {weights.max():.6f}")
        
        return weights
    
    def analyze_uncertainty_distribution(self, epistemic_var: torch.Tensor, 
                                       weights: torch.Tensor,
                                       save_path: Optional[str] = None):
        """불확실성 분포 분석 및 시각화"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 에피스테믹 불확실성 히스토그램
        axes[0, 0].hist(epistemic_var.numpy(), bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        axes[0, 0].set_title('Epistemic Uncertainty Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Variance of log p(x)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 가중치 히스토그램
        axes[0, 1].hist(weights.numpy(), bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 1].set_title('Uncertainty Weights Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Weight (1/uncertainty)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 불확실성 vs 가중치 산점도
        sample_indices = np.random.choice(len(epistemic_var), size=min(1000, len(epistemic_var)), replace=False)
        scatter = axes[0, 2].scatter(epistemic_var[sample_indices].numpy(), 
                                   weights[sample_indices].numpy(), 
                                   alpha=0.5, c='purple', s=20)
        axes[0, 2].set_xlabel('Epistemic Uncertainty')
        axes[0, 2].set_ylabel('Weight')
        axes[0, 2].set_title('Uncertainty vs Weight Relationship', fontsize=12, fontweight='bold')
        axes[0, 2].set_xscale('log')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 불확실성 분위수 분석
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        uncertainty_quantiles = torch.quantile(epistemic_var, torch.tensor(quantiles))
        weight_quantiles = torch.quantile(weights, torch.tensor(quantiles))
        
        axes[1, 0].plot(quantiles, uncertainty_quantiles.numpy(), 'o-', label='Uncertainty', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Quantile')
        axes[1, 0].set_ylabel('Uncertainty Value')
        axes[1, 0].set_title('Uncertainty Quantiles', fontsize=12, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 가중치 분위수 분석
        axes[1, 1].plot(quantiles, weight_quantiles.numpy(), 'o-', label='Weights', color='orange', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Quantile')
        axes[1, 1].set_ylabel('Weight Value')
        axes[1, 1].set_title('Weight Quantiles', fontsize=12, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 불확실성과 가중치의 누적분포함수 (CDF)
        uncertainty_sorted = torch.sort(epistemic_var)[0].numpy()
        weights_sorted = torch.sort(weights)[0].numpy()
        n_samples = len(uncertainty_sorted)
        cdf_values = np.arange(1, n_samples + 1) / n_samples
        
        axes[1, 2].plot(uncertainty_sorted, cdf_values, label='Uncertainty CDF', linewidth=2)
        axes[1, 2].set_xlabel('Uncertainty Value')
        axes[1, 2].set_ylabel('Cumulative Probability')
        axes[1, 2].set_title('Cumulative Distribution Functions', fontsize=12, fontweight='bold')
        axes[1, 2].set_xscale('log')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Secondary y-axis for weights CDF
        ax2 = axes[1, 2].twinx()
        ax2.plot(weights_sorted, cdf_values, label='Weights CDF', color='orange', linewidth=2)
        ax2.set_ylabel('Cumulative Probability (Weights)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Combine legends
        lines1, labels1 = axes[1, 2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1, 2].legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved uncertainty analysis plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # 통계 정보 출력
        self.logger.info("=== Uncertainty Analysis ===")
        self.logger.info(f"Epistemic Uncertainty - Mean: {epistemic_var.mean():.6f}, "
                        f"Std: {epistemic_var.std():.6f}, "
                        f"Min: {epistemic_var.min():.6f}, "
                        f"Max: {epistemic_var.max():.6f}")
        self.logger.info(f"Uncertainty Weights - Mean: {weights.mean():.6f}, "
                        f"Std: {weights.std():.6f}, "
                        f"Min: {weights.min():.6f}, "
                        f"Max: {weights.max():.6f}")
        
        for i, q in enumerate(quantiles):
            self.logger.info(f"Q{q*100:2.0f}: Uncertainty={uncertainty_quantiles[i]:.6f}, "
                           f"Weight={weight_quantiles[i]:.6f}")
        
        # 상관관계 분석
        correlation = torch.corrcoef(torch.stack([epistemic_var, weights]))[0, 1]
        self.logger.info(f"Correlation between uncertainty and weights: {correlation:.6f}")
        
        return {
            'uncertainty_stats': {
                'mean': epistemic_var.mean().item(),
                'std': epistemic_var.std().item(),
                'min': epistemic_var.min().item(),
                'max': epistemic_var.max().item(),
                'quantiles': uncertainty_quantiles.tolist()
            },
            'weight_stats': {
                'mean': weights.mean().item(),
                'std': weights.std().item(),
                'min': weights.min().item(),
                'max': weights.max().item(),
                'quantiles': weight_quantiles.tolist()
            },
            'correlation': correlation.item()
        }
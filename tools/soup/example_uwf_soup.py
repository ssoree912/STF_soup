#!/usr/bin/env python3
"""
Example usage of Uncertainty-Weighted Fisher Soup (UWF-Soup) for STG-NF models.

This script demonstrates how to use the UWF-Soup method with a simple example.
"""

import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.soup.fisher_soup_stg_nf import FisherSoupSTGNF
from tools.soup.epistemic_uncertainty import EpistemicUncertainty
from models.STG_NF.model_pose import STG_NF


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_dummy_model(device):
    """Create a dummy STG-NF model for testing."""
    model_args = {
        'pose_dims': 51,
        'n_blocks': 4,  # Smaller for testing
        'hidden_size': 128,  # Smaller for testing
        'n_hidden': 1,
        'batch_norm': False,
        'attention': False,
        'custom_trans': False,
    }
    
    model = STG_NF(**model_args)
    model.to(device)
    return model


def create_dummy_data(batch_size=16, seq_len=25, pose_dims=51, device='cpu'):
    """Create dummy data for testing."""
    # Create dummy pose data
    data = torch.randn(batch_size, seq_len, pose_dims, device=device)
    labels = torch.ones(batch_size, device=device)
    scores = torch.rand(batch_size, device=device)
    
    return [(data, labels, scores)]


def main():
    logger = setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dummy models (in practice, these would be loaded from checkpoints)
    logger.info("Creating dummy models for demonstration...")
    n_models = 3
    models = []
    
    for i in range(n_models):
        model = create_dummy_model(device)
        # Add some noise to make models different
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        models.append(model)
    
    logger.info(f"Created {len(models)} models")
    
    # Create dummy data
    logger.info("Creating dummy data...")
    dummy_data = create_dummy_data(device=device)
    
    # Create dummy args
    class DummyArgs:
        def __init__(self):
            self.model_confidence = False
    
    args = DummyArgs()
    
    # Step 1: Compute epistemic uncertainty
    logger.info("Computing epistemic uncertainty...")
    uncertainty_calculator = EpistemicUncertainty(models, device, logger)
    
    # Since we have dummy data, we'll simulate the log-likelihoods
    model_logps = {}
    for i in range(n_models):
        # Simulate log-likelihoods with different variances
        base_logp = torch.randn(16, device=device) * 2.0  # Base log-likelihood
        noise = torch.randn(16, device=device) * (0.5 + i * 0.3)  # Different noise levels
        model_logps[f"model_{i}"] = base_logp + noise
    
    # Compute epistemic uncertainty
    epistemic_var = uncertainty_calculator.compute_epistemic_uncertainty(model_logps)
    uncertainty_weights = uncertainty_calculator.compute_uncertainty_weights(epistemic_var)
    
    logger.info(f"Computed uncertainty for {len(uncertainty_weights)} samples")
    
    # Step 2: Analyze uncertainty distribution
    logger.info("Analyzing uncertainty distribution...")
    analysis_results = uncertainty_calculator.analyze_uncertainty_distribution(
        epistemic_var, uncertainty_weights, save_path="example_uncertainty_analysis.png"
    )
    
    logger.info(f"Uncertainty analysis completed. Correlation: {analysis_results['correlation']:.4f}")
    
    # Step 3: Demonstrate Fisher Soup integration
    logger.info("Demonstrating Fisher Soup integration...")
    fisher_soup = FisherSoupSTGNF(device, logger)
    
    # For demonstration, we'll use simple coefficients
    coefficients_set = [
        [1.0, 0.0, 0.0],  # Only first model
        [0.0, 1.0, 0.0],  # Only second model
        [0.0, 0.0, 1.0],  # Only third model
        [0.5, 0.5, 0.0],  # First two models
        [0.33, 0.33, 0.34],  # All models equally
    ]
    
    logger.info("Creating merged models with different coefficients...")
    for i, coeffs in enumerate(coefficients_set):
        logger.info(f"Configuration {i+1}: {coeffs}")
        
        # Create merged model
        merged_model = fisher_soup._merge_with_coeffs(
            models=models,
            coefficients=coeffs,
            fishers=None,  # No Fisher info for this demo
            combined_mask=None
        )
        
        logger.info(f"Successfully created merged model {i+1}")
        
        # Clean up
        del merged_model
        torch.cuda.empty_cache()
    
    # Step 4: Summary
    logger.info("=== UWF-Soup Example Summary ===")
    logger.info(f"✓ Created {n_models} dummy models")
    logger.info(f"✓ Computed epistemic uncertainty for {len(uncertainty_weights)} samples")
    logger.info(f"✓ Uncertainty stats: mean={epistemic_var.mean():.4f}, std={epistemic_var.std():.4f}")
    logger.info(f"✓ Weight stats: mean={uncertainty_weights.mean():.4f}, std={uncertainty_weights.std():.4f}")
    logger.info(f"✓ Created {len(coefficients_set)} merged model configurations")
    logger.info("✓ Saved uncertainty analysis plot to 'example_uncertainty_analysis.png'")
    
    logger.info("\nFor real usage:")
    logger.info("1. Load multiple trained STG-NF models from checkpoints")
    logger.info("2. Prepare your actual training/validation dataset")
    logger.info("3. Use tools/soup/uwf_soup_stg_nf.py script for complete workflow")
    logger.info("4. Example: python tools/soup/uwf_soup_stg_nf.py --folder_path /path/to/checkpoints --data_path /path/to/data")


if __name__ == "__main__":
    main()

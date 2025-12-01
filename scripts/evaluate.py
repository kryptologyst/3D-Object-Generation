#!/usr/bin/env python3
"""Evaluation script for 3D point cloud generation models."""

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
from omegaconf import DictConfig, OmegaConf

from src.3dgen import (
    PointCloudDataset,
    get_dataloader,
    PointCloudGenerator,
    PointCloudVAE,
    PointCloudGAN,
    PointCloudDiffusion,
    Evaluator,
    set_seed,
    get_device,
)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_model(config: DictConfig, checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    model_type = config.model.type.lower()
    
    if model_type == "vae":
        model = PointCloudVAE(
            num_points=config.data.num_points,
            latent_dim=config.model.z_dim,
            hidden_dim=config.model.hidden_dim,
            beta=config.model.get('beta', 1.0),
        )
    elif model_type == "gan":
        model = PointCloudGAN(
            z_dim=config.model.z_dim,
            num_points=config.data.num_points,
            hidden_dim=config.model.hidden_dim,
        )
    elif model_type == "diffusion":
        model = PointCloudDiffusion(
            num_points=config.data.num_points,
            hidden_dim=config.model.hidden_dim,
            num_timesteps=config.model.get('num_timesteps', 1000),
        )
    elif model_type == "generator":
        model = PointCloudGenerator(
            z_dim=config.model.z_dim,
            num_points=config.data.num_points,
            hidden_dim=config.model.hidden_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model.to(device)


def create_test_dataloader(config: DictConfig):
    """Create test data loader."""
    test_dataset = PointCloudDataset(
        num_points=config.data.num_points,
        synthetic=config.data.synthetic,
        shape_types=config.data.get('shape_types', ["sphere", "cube", "cylinder"]),
        normalize=config.data.normalize,
        augment=False,  # No augmentation for testing
    )
    
    test_loader = get_dataloader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    
    return test_loader


def print_results(results: Dict[str, float]):
    """Print evaluation results in a formatted table."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for metric, value in results.items():
        print(f"{metric.replace('_', ' ').title():<25}: {value:.6f}")
    
    print("="*60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate 3D point cloud generation models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-data-dir",
        type=str,
        help="Path to test data directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file to save results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override with command line arguments
    if args.test_data_dir:
        config.data.data_dir = args.test_data_dir
    if args.seed:
        config.experiment.seed = args.seed
    
    # Setup logging
    setup_logging(config.logging.level)
    logger = logging.getLogger(__name__)
    
    # Set seed
    set_seed(config.experiment.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    model = load_model(config, args.checkpoint, device)
    logger.info("Model loaded successfully")
    
    # Create test data loader
    logger.info("Creating test data loader...")
    test_loader = create_test_dataloader(config)
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create evaluator
    logger.info("Creating evaluator...")
    evaluator = Evaluator(device)
    
    # Evaluate model
    logger.info(f"Evaluating model on {args.num_samples} samples...")
    results = evaluator.evaluate(model, test_loader, num_samples=args.num_samples)
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("Evaluation Results\n")
            f.write("="*60 + "\n")
            for metric, value in results.items():
                f.write(f"{metric}: {value:.6f}\n")
        
        logger.info(f"Results saved to {output_path}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()

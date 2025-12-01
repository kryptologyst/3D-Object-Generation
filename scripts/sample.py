#!/usr/bin/env python3
"""Sampling script for generating 3D point clouds."""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import DictConfig, OmegaConf

from src.3dgen import (
    PointCloudGenerator,
    PointCloudVAE,
    PointCloudGAN,
    PointCloudDiffusion,
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


def generate_samples(
    model,
    num_samples: int,
    device: torch.device,
    steps: Optional[int] = None
):
    """Generate samples from the model."""
    with torch.no_grad():
        if hasattr(model, 'generate'):
            if steps is not None and hasattr(model, 'num_timesteps'):
                # For diffusion models, use custom steps
                samples = model.generate(num_samples, device, steps)
            else:
                samples = model.generate(num_samples, device)
        else:
            # For VAE, sample from prior
            if hasattr(model, 'latent_dim'):
                z = torch.randn(num_samples, model.latent_dim, device=device)
                samples = model.decode(z)
            else:
                z = torch.randn(num_samples, model.z_dim, device=device)
                samples = model(z)
    
    return samples


def visualize_point_cloud(points: np.ndarray, title: str = "Point Cloud", save_path: Optional[str] = None):
    """Visualize a single point cloud."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_samples(samples: torch.Tensor, num_show: int = 4, save_dir: Optional[str] = None):
    """Visualize multiple samples in a grid."""
    num_samples = min(samples.size(0), num_show)
    
    fig = plt.figure(figsize=(15, 15))
    
    for i in range(num_samples):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        points = samples[i].cpu().numpy()
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
        ax.set_title(f'Sample {i + 1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                             points[:, 1].max() - points[:, 1].min(),
                             points[:, 2].max() - points[:, 2].min()]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / "generated_samples.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Samples saved to {save_path}")
    
    plt.show()


def interpolate_latent(model, z1: torch.Tensor, z2: torch.Tensor, steps: int = 10):
    """Interpolate between two latent vectors."""
    interpolations = []
    
    for i in range(steps):
        alpha = i / (steps - 1)
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        with torch.no_grad():
            if hasattr(model, 'decode'):
                sample = model.decode(z_interp)
            else:
                sample = model(z_interp)
        
        interpolations.append(sample)
    
    return torch.stack(interpolations)


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Generate 3D point cloud samples")
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
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Number of denoising steps for diffusion models"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for generated samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Generate interpolation between two random samples"
    )
    parser.add_argument(
        "--interpolation-steps",
        type=int,
        default=10,
        help="Number of interpolation steps"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override with command line arguments
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
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    samples = generate_samples(model, args.num_samples, device, args.steps)
    
    # Save samples
    samples_path = output_dir / "generated_samples.pt"
    torch.save(samples, samples_path)
    logger.info(f"Samples saved to {samples_path}")
    
    # Generate interpolation if requested
    if args.interpolate:
        logger.info("Generating interpolation...")
        
        # Generate two random latent vectors
        if hasattr(model, 'latent_dim'):
            z1 = torch.randn(1, model.latent_dim, device=device)
            z2 = torch.randn(1, model.latent_dim, device=device)
        else:
            z1 = torch.randn(1, model.z_dim, device=device)
            z2 = torch.randn(1, model.z_dim, device=device)
        
        # Interpolate
        interpolations = interpolate_latent(model, z1, z2, args.interpolation_steps)
        
        # Save interpolation
        interp_path = output_dir / "interpolation.pt"
        torch.save(interpolations, interp_path)
        logger.info(f"Interpolation saved to {interp_path}")
        
        # Visualize interpolation
        if not args.no_visualize:
            visualize_samples(interpolations, num_show=args.interpolation_steps)
    
    # Visualize samples
    if not args.no_visualize:
        logger.info("Visualizing samples...")
        visualize_samples(samples, num_show=min(4, args.num_samples), save_dir=str(output_dir))
    
    logger.info("Sampling completed!")


if __name__ == "__main__":
    main()

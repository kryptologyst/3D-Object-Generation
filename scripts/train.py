#!/usr/bin/env python3
"""Main training script for 3D object generation models."""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf

from src.3dgen import (
    PointCloudDataset,
    get_dataloader,
    PointCloudGenerator,
    PointCloudVAE,
    PointCloudGAN,
    PointCloudDiffusion,
    Trainer,
    set_seed,
    get_device,
)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_model(config: DictConfig, device: torch.device):
    """Create model based on configuration."""
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
    
    return model.to(device)


def create_dataloaders(config: DictConfig):
    """Create data loaders."""
    # Training dataset
    train_dataset = PointCloudDataset(
        num_points=config.data.num_points,
        synthetic=config.data.synthetic,
        shape_types=config.data.get('shape_types', ["sphere", "cube", "cylinder"]),
        normalize=config.data.normalize,
        augment=config.data.augment,
    )
    
    # Validation dataset (no augmentation)
    val_dataset = PointCloudDataset(
        num_points=config.data.num_points,
        synthetic=config.data.synthetic,
        shape_types=config.data.get('shape_types', ["sphere", "cube", "cylinder"]),
        normalize=config.data.normalize,
        augment=False,
    )
    
    # Create data loaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train 3D object generation models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vae", "gan", "diffusion", "generator"],
        help="Model type to train"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override with command line arguments
    if args.model:
        config.model.type = args.model
    if args.data_dir:
        config.paths.data_dir = args.data_dir
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.model.learning_rate = args.learning_rate
    if args.seed:
        config.experiment.seed = args.seed
    if args.debug:
        config.experiment.debug = True
    
    # Setup logging
    setup_logging(config.logging.level)
    logger = logging.getLogger(__name__)
    
    # Set seed
    set_seed(config.experiment.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(config.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(config.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    logger.info(f"Creating {config.model.type} model...")
    model = create_model(config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )
    
    # Train model
    logger.info("Starting training...")
    trained_model = trainer.train()
    
    # Test model
    logger.info("Testing model...")
    trainer.test(trained_model)
    
    # Generate samples
    logger.info("Generating samples...")
    samples = trainer.generate_samples(trained_model, num_samples=10)
    
    # Save samples
    samples_path = output_dir / "generated_samples.pt"
    torch.save(samples, samples_path)
    logger.info(f"Generated samples saved to {samples_path}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

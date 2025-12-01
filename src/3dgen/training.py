"""Training framework for 3D point cloud generation models."""

import os
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import wandb
from omegaconf import DictConfig

from .models import PointCloudGenerator, PointCloudVAE, PointCloudGAN, PointCloudDiffusion
from .evaluation import Evaluator, compute_chamfer_distance, compute_emd
from .utils import set_seed


class BaseTrainer(pl.LightningModule):
    """Base trainer class for 3D point cloud generation models."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        **kwargs
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            learning_rate: Learning rate.
            weight_decay: Weight decay for optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize evaluator
        self.evaluator = Evaluator(self.device)
        
        # Metrics
        self.train_chamfer = compute_chamfer_distance
        self.val_chamfer = compute_chamfer_distance
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        raise NotImplementedError
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        raise NotImplementedError


class VAETrainer(BaseTrainer):
    """Trainer for VAE models."""
    
    def __init__(
        self,
        model: PointCloudVAE,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        beta: float = 1.0,
        **kwargs
    ):
        """Initialize VAE trainer.
        
        Args:
            model: VAE model.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            beta: Beta parameter for beta-VAE.
        """
        super().__init__(model, learning_rate, weight_decay, **kwargs)
        self.beta = beta
    
    def training_step(self, batch, batch_idx):
        """Training step for VAE."""
        recon, mu, logvar = self.model(batch)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, batch)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / batch.size(0)  # Normalize by batch size
        
        # Total loss
        loss = recon_loss + self.beta * kl_loss
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True)
        self.log('train_kl_loss', kl_loss, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for VAE."""
        recon, mu, logvar = self.model(batch)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, batch)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / batch.size(0)
        
        # Total loss
        loss = recon_loss + self.beta * kl_loss
        
        # Chamfer distance
        chamfer_dist = torch.mean(compute_chamfer_distance(recon, batch))
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val_chamfer', chamfer_dist, on_step=False, on_epoch=True)
        
        return loss


class GANTrainer(BaseTrainer):
    """Trainer for GAN models."""
    
    def __init__(
        self,
        model: PointCloudGAN,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        beta1: float = 0.5,
        beta2: float = 0.999,
        **kwargs
    ):
        """Initialize GAN trainer.
        
        Args:
            model: GAN model.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            beta1: Beta1 for Adam optimizer.
            beta2: Beta2 for Adam optimizer.
        """
        super().__init__(model, learning_rate, weight_decay, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
    
    def configure_optimizers(self):
        """Configure optimizers for GAN."""
        g_optimizer = optim.Adam(
            self.model.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay
        )
        d_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay
        )
        return [g_optimizer, d_optimizer]
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        """Training step for GAN."""
        if optimizer_idx == 0:  # Generator
            return self._generator_step(batch)
        elif optimizer_idx == 1:  # Discriminator
            return self._discriminator_step(batch)
    
    def _generator_step(self, batch):
        """Generator training step."""
        batch_size = batch.size(0)
        z = torch.randn(batch_size, self.model.z_dim, device=self.device)
        
        # Generate fake samples
        fake_samples = self.model.generator(z)
        
        # Discriminator output for fake samples
        fake_output = self.model.discriminator(fake_samples)
        
        # Generator loss (want discriminator to think fake samples are real)
        g_loss = F.binary_cross_entropy_with_logits(
            fake_output, torch.ones_like(fake_output)
        )
        
        self.log('train_g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
        return g_loss
    
    def _discriminator_step(self, batch):
        """Discriminator training step."""
        batch_size = batch.size(0)
        z = torch.randn(batch_size, self.model.z_dim, device=self.device)
        
        # Generate fake samples
        fake_samples = self.model.generator(z)
        
        # Real samples
        real_output = self.model.discriminator(batch)
        fake_output = self.model.discriminator(fake_samples.detach())
        
        # Discriminator loss
        real_loss = F.binary_cross_entropy_with_logits(
            real_output, torch.ones_like(real_output)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_output, torch.zeros_like(fake_output)
        )
        d_loss = (real_loss + fake_loss) / 2
        
        self.log('train_d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        return d_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for GAN."""
        batch_size = batch.size(0)
        z = torch.randn(batch_size, self.model.z_dim, device=self.device)
        
        # Generate samples
        generated = self.model.generator(z)
        
        # Chamfer distance
        chamfer_dist = torch.mean(compute_chamfer_distance(generated, batch))
        
        self.log('val_chamfer', chamfer_dist, on_step=False, on_epoch=True, prog_bar=True)
        return chamfer_dist


class DiffusionTrainer(BaseTrainer):
    """Trainer for Diffusion models."""
    
    def __init__(
        self,
        model: PointCloudDiffusion,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        **kwargs
    ):
        """Initialize Diffusion trainer.
        
        Args:
            model: Diffusion model.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
        """
        super().__init__(model, learning_rate, weight_decay, **kwargs)
    
    def training_step(self, batch, batch_idx):
        """Training step for Diffusion."""
        batch_size = batch.size(0)
        
        # Random timesteps
        t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise to batch
        noise = torch.randn_like(batch)
        noisy_batch = batch + noise
        
        # Predict noise
        predicted_noise = self.model(noisy_batch, t)
        
        # Loss
        loss = F.mse_loss(predicted_noise, noise)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for Diffusion."""
        batch_size = batch.size(0)
        
        # Random timesteps
        t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise to batch
        noise = torch.randn_like(batch)
        noisy_batch = batch + noise
        
        # Predict noise
        predicted_noise = self.model(noisy_batch, t)
        
        # Loss
        loss = F.mse_loss(predicted_noise, noise)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


class Trainer:
    """Main trainer class for 3D point cloud generation."""
    
    def __init__(
        self,
        config: DictConfig,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
    ):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary.
            model: Model to train.
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            test_dataloader: Test data loader.
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Set seed for reproducibility
        set_seed(config.get('seed', 42))
        
        # Setup logging
        self._setup_logging()
        
        # Create trainer
        self._create_trainer()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_trainer(self):
        """Create PyTorch Lightning trainer."""
        # Callbacks
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.get('checkpoint_dir', 'checkpoints'),
            filename='{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('patience', 10),
            mode='min',
        )
        callbacks.append(early_stop_callback)
        
        # Logger
        logger = TensorBoardLogger(
            save_dir=self.config.get('log_dir', 'logs'),
            name=self.config.get('experiment_name', '3d_generation'),
        )
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config.get('max_epochs', 100),
            callbacks=callbacks,
            logger=logger,
            accelerator=self.config.get('accelerator', 'auto'),
            devices=self.config.get('devices', 1),
            precision=self.config.get('precision', 32),
            gradient_clip_val=self.config.get('gradient_clip_val', 1.0),
            accumulate_grad_batches=self.config.get('accumulate_grad_batches', 1),
        )
    
    def train(self):
        """Train the model."""
        # Create appropriate trainer based on model type
        if isinstance(self.model, PointCloudVAE):
            lightning_model = VAETrainer(self.model, **self.config.model)
        elif isinstance(self.model, PointCloudGAN):
            lightning_model = GANTrainer(self.model, **self.config.model)
        elif isinstance(self.model, PointCloudDiffusion):
            lightning_model = DiffusionTrainer(self.model, **self.config.model)
        else:
            lightning_model = BaseTrainer(self.model, **self.config.model)
        
        # Train
        self.trainer.fit(
            lightning_model,
            self.train_dataloader,
            self.val_dataloader,
        )
        
        return lightning_model
    
    def test(self, model):
        """Test the model."""
        if self.test_dataloader is not None:
            self.trainer.test(model, self.test_dataloader)
    
    def generate_samples(self, model, num_samples: int = 10):
        """Generate samples from the trained model."""
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            if hasattr(model, 'generate'):
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

"""3D point cloud generative models."""

from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PointCloudGenerator(nn.Module):
    """Basic point cloud generator using fully connected layers.
    
    This is a simple baseline model for generating 3D point clouds.
    """
    
    def __init__(
        self,
        z_dim: int = 100,
        num_points: int = 1024,
        hidden_dim: int = 512,
    ):
        """Initialize generator.
        
        Args:
            z_dim: Latent dimension.
            num_points: Number of points in generated point cloud.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.z_dim = z_dim
        self.num_points = num_points
        
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, num_points * 3)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Latent vector of shape (batch_size, z_dim).
            
        Returns:
            Generated point cloud of shape (batch_size, num_points, 3).
        """
        x = self.relu(self.fc1(z))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.tanh(x)
        
        return x.view(-1, self.num_points, 3)


class PointCloudVAE(nn.Module):
    """Variational Autoencoder for point cloud generation.
    
    Implements a VAE with reparameterization trick for 3D point clouds.
    """
    
    def __init__(
        self,
        num_points: int = 1024,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        beta: float = 1.0,
    ):
        """Initialize VAE.
        
        Args:
            num_points: Number of points in point cloud.
            latent_dim: Latent space dimension.
            hidden_dim: Hidden layer dimension.
            beta: Beta parameter for beta-VAE.
        """
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_points * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points * 3),
            nn.Tanh(),
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode point cloud to latent space.
        
        Args:
            x: Point cloud of shape (batch_size, num_points, 3).
            
        Returns:
            Mean and log variance of latent distribution.
        """
        x_flat = x.view(x.size(0), -1)
        h = self.encoder(x_flat)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick.
        
        Args:
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.
            
        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to point cloud.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim).
            
        Returns:
            Reconstructed point cloud of shape (batch_size, num_points, 3).
        """
        x_flat = self.decoder(z)
        return x_flat.view(-1, self.num_points, 3)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input point cloud of shape (batch_size, num_points, 3).
            
        Returns:
            Reconstructed point cloud, mean, and log variance.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate new point clouds.
        
        Args:
            batch_size: Number of point clouds to generate.
            device: Device to generate on.
            
        Returns:
            Generated point clouds.
        """
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decode(z)


class PointCloudDiscriminator(nn.Module):
    """Discriminator for GAN training."""
    
    def __init__(
        self,
        num_points: int = 1024,
        hidden_dim: int = 512,
    ):
        """Initialize discriminator.
        
        Args:
            num_points: Number of points in point cloud.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.num_points = num_points
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(512, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Point cloud of shape (batch_size, num_points, 3).
            
        Returns:
            Discriminator output of shape (batch_size, 1).
        """
        x = x.transpose(1, 2)  # (batch_size, 3, num_points)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        
        x = torch.max(x, dim=2)[0]  # Global max pooling
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PointCloudGAN(nn.Module):
    """GAN for point cloud generation."""
    
    def __init__(
        self,
        z_dim: int = 100,
        num_points: int = 1024,
        hidden_dim: int = 512,
    ):
        """Initialize GAN.
        
        Args:
            z_dim: Latent dimension.
            num_points: Number of points in point cloud.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.z_dim = z_dim
        self.num_points = num_points
        
        self.generator = PointCloudGenerator(z_dim, num_points, hidden_dim)
        self.discriminator = PointCloudDiscriminator(num_points, hidden_dim)
    
    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate point clouds.
        
        Args:
            batch_size: Number of point clouds to generate.
            device: Device to generate on.
            
        Returns:
            Generated point clouds.
        """
        z = torch.randn(batch_size, self.z_dim, device=device)
        return self.generator(z)


class PointCloudDiffusion(nn.Module):
    """Diffusion model for point cloud generation.
    
    Implements a simplified diffusion model for 3D point clouds.
    """
    
    def __init__(
        self,
        num_points: int = 1024,
        hidden_dim: int = 512,
        num_timesteps: int = 1000,
    ):
        """Initialize diffusion model.
        
        Args:
            num_points: Number of points in point cloud.
            hidden_dim: Hidden layer dimension.
            num_timesteps: Number of diffusion timesteps.
        """
        super().__init__()
        self.num_points = num_points
        self.num_timesteps = num_timesteps
        
        # Noise prediction network
        self.noise_net = nn.Sequential(
            nn.Linear(num_points * 3 + 1, hidden_dim),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points * 3),
        )
        
        # Timestep embedding
        self.timestep_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Noisy point cloud of shape (batch_size, num_points, 3).
            t: Timestep tensor of shape (batch_size,).
            
        Returns:
            Predicted noise of shape (batch_size, num_points, 3).
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        # Timestep embedding
        t_emb = self.timestep_embedding(t.unsqueeze(-1).float())
        
        # Concatenate point cloud and timestep embedding
        x_with_t = torch.cat([x_flat, t_emb.view(batch_size, -1)], dim=1)
        
        # Predict noise
        noise_pred = self.noise_net(x_with_t)
        return noise_pred.view(batch_size, self.num_points, 3)
    
    def generate(self, batch_size: int, device: torch.device, steps: int = 100) -> torch.Tensor:
        """Generate point clouds using reverse diffusion.
        
        Args:
            batch_size: Number of point clouds to generate.
            device: Device to generate on.
            steps: Number of denoising steps.
            
        Returns:
            Generated point clouds.
        """
        # Start with pure noise
        x = torch.randn(batch_size, self.num_points, 3, device=device)
        
        # Reverse diffusion process
        for i in range(steps):
            t = torch.full((batch_size,), steps - i - 1, device=device)
            with torch.no_grad():
                noise_pred = self.forward(x, t)
                x = x - noise_pred / steps
        
        return x

"""Unit tests for 3D object generation models."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.3dgen import (
    PointCloudDataset,
    get_dataloader,
    PointCloudGenerator,
    PointCloudVAE,
    PointCloudGAN,
    PointCloudDiffusion,
    compute_chamfer_distance,
    compute_emd,
    compute_coverage,
    set_seed,
    get_device,
)


class TestPointCloudDataset:
    """Test PointCloudDataset class."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = PointCloudDataset(
            num_points=512,
            synthetic=True,
            shape_types=["sphere", "cube"],
            normalize=True,
            augment=False,
        )
        
        assert len(dataset) == 2  # Two shape types
        sample = dataset[0]
        assert sample.shape == (512, 3)
        assert torch.allclose(torch.norm(sample, dim=1), torch.ones(512), atol=1e-6)
    
    def test_dataset_augmentation(self):
        """Test dataset augmentation."""
        dataset = PointCloudDataset(
            num_points=256,
            synthetic=True,
            shape_types=["sphere"],
            normalize=True,
            augment=True,
        )
        
        sample1 = dataset[0]
        sample2 = dataset[0]  # Same index, should be different due to augmentation
        
        # Samples should be different due to random augmentation
        assert not torch.allclose(sample1, sample2, atol=1e-6)
    
    def test_dataloader(self):
        """Test DataLoader creation."""
        dataset = PointCloudDataset(
            num_points=128,
            synthetic=True,
            shape_types=["sphere"],
            normalize=True,
            augment=False,
        )
        
        dataloader = get_dataloader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        
        batch = next(iter(dataloader))
        assert batch.shape == (4, 128, 3)


class TestModels:
    """Test model classes."""
    
    def test_point_cloud_generator(self):
        """Test PointCloudGenerator."""
        model = PointCloudGenerator(z_dim=64, num_points=256, hidden_dim=128)
        
        z = torch.randn(4, 64)
        output = model(z)
        
        assert output.shape == (4, 256, 3)
        assert torch.all(output >= -1) and torch.all(output <= 1)  # Tanh output
    
    def test_point_cloud_vae(self):
        """Test PointCloudVAE."""
        model = PointCloudVAE(
            num_points=128,
            latent_dim=32,
            hidden_dim=64,
            beta=1.0,
        )
        
        x = torch.randn(4, 128, 3)
        recon, mu, logvar = model(x)
        
        assert recon.shape == (4, 128, 3)
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)
        
        # Test generation
        generated = model.generate(4, torch.device('cpu'))
        assert generated.shape == (4, 128, 3)
    
    def test_point_cloud_gan(self):
        """Test PointCloudGAN."""
        model = PointCloudGAN(
            z_dim=64,
            num_points=128,
            hidden_dim=64,
        )
        
        # Test generator
        z = torch.randn(4, 64)
        generated = model.generator(z)
        assert generated.shape == (4, 128, 3)
        
        # Test discriminator
        x = torch.randn(4, 128, 3)
        disc_output = model.discriminator(x)
        assert disc_output.shape == (4, 1)
        
        # Test generation
        generated = model.generate(4, torch.device('cpu'))
        assert generated.shape == (4, 128, 3)
    
    def test_point_cloud_diffusion(self):
        """Test PointCloudDiffusion."""
        model = PointCloudDiffusion(
            num_points=64,
            hidden_dim=32,
            num_timesteps=100,
        )
        
        x = torch.randn(4, 64, 3)
        t = torch.randint(0, 100, (4,))
        
        noise_pred = model(x, t)
        assert noise_pred.shape == (4, 64, 3)
        
        # Test generation
        generated = model.generate(4, torch.device('cpu'), steps=10)
        assert generated.shape == (4, 64, 3)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_chamfer_distance(self):
        """Test Chamfer distance computation."""
        pred = torch.randn(2, 64, 3)
        target = torch.randn(2, 64, 3)
        
        chamfer_dist = compute_chamfer_distance(pred, target)
        assert chamfer_dist.shape == (2,)
        assert torch.all(chamfer_dist >= 0)
    
    def test_emd(self):
        """Test EMD computation."""
        pred = torch.randn(2, 32, 3)  # Smaller for faster computation
        target = torch.randn(2, 32, 3)
        
        emd_dist = compute_emd(pred, target, max_iter=10)
        assert emd_dist.shape == (2,)
        assert torch.all(emd_dist >= 0)
    
    def test_coverage(self):
        """Test coverage computation."""
        pred = torch.randn(2, 64, 3)
        target = torch.randn(2, 64, 3)
        
        coverage = compute_coverage(pred, target, threshold=0.1)
        assert coverage.shape == (2,)
        assert torch.all(coverage >= 0) and torch.all(coverage <= 1)


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate random numbers
        torch_rand1 = torch.randn(10)
        np_rand1 = np.random.randn(10)
        
        # Reset seed and generate again
        set_seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        
        torch_rand2 = torch.randn(10)
        np_rand2 = np.random.randn(10)
        
        # Should be identical
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        
        # Test specific device
        cpu_device = get_device("cpu")
        assert cpu_device == torch.device("cpu")


class TestIntegration:
    """Integration tests."""
    
    def test_training_loop(self):
        """Test basic training loop."""
        # Create model and data
        model = PointCloudGenerator(z_dim=32, num_points=64, hidden_dim=32)
        dataset = PointCloudDataset(
            num_points=64,
            synthetic=True,
            shape_types=["sphere"],
            normalize=True,
            augment=False,
        )
        dataloader = get_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        # Training step
        model.train()
        batch = next(iter(dataloader))
        
        z = torch.randn(batch.size(0), 32)
        generated = model(z)
        
        loss = criterion(generated, batch)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0
    
    def test_evaluation(self):
        """Test model evaluation."""
        model = PointCloudGenerator(z_dim=32, num_points=64, hidden_dim=32)
        dataset = PointCloudDataset(
            num_points=64,
            synthetic=True,
            shape_types=["sphere"],
            normalize=True,
            augment=False,
        )
        dataloader = get_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        # Generate samples
        model.eval()
        with torch.no_grad():
            z = torch.randn(2, 32)
            generated = model(z)
            batch = next(iter(dataloader))
            
            # Compute metrics
            chamfer_dist = compute_chamfer_distance(generated, batch)
            coverage = compute_coverage(generated, batch)
            
            assert chamfer_dist.shape == (2,)
            assert coverage.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__])

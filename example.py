#!/usr/bin/env python3
"""Example script demonstrating 3D object generation framework."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.3dgen import (
    PointCloudDataset,
    get_dataloader,
    PointCloudVAE,
    PointCloudGenerator,
    compute_chamfer_distance,
    set_seed,
    get_device,
)


def visualize_point_cloud(points: np.ndarray, title: str = "Point Cloud"):
    """Visualize a point cloud."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def main():
    """Main example function."""
    print("3D Object Generation Example")
    print("=" * 40)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Create dataset
    print("\n1. Creating synthetic dataset...")
    dataset = PointCloudDataset(
        num_points=256,
        synthetic=True,
        shape_types=["sphere", "cube"],
        normalize=True,
        augment=False,
    )
    
    dataloader = get_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"Dataset size: {len(dataset)}")
    
    # 2. Visualize sample data
    print("\n2. Visualizing sample data...")
    batch = next(iter(dataloader))
    sample = batch[0].numpy()
    visualize_point_cloud(sample, "Sample Point Cloud")
    
    # 3. Create and train a simple generator
    print("\n3. Training simple generator...")
    model = PointCloudGenerator(z_dim=32, num_points=256, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(5):
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            z = torch.randn(batch.size(0), 32, device=device)
            
            generated = model(z)
            loss = torch.mean(compute_chamfer_distance(generated, batch))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/5, Loss: {epoch_loss/len(dataloader):.4f}")
    
    # 4. Generate new samples
    print("\n4. Generating new samples...")
    model.eval()
    with torch.no_grad():
        z = torch.randn(4, 32, device=device)
        generated = model(z)
    
    # Visualize generated samples
    fig, axes = plt.subplots(2, 2, figsize=(15, 15), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    for i in range(4):
        points = generated[i].cpu().numpy()
        axes[i].scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
        axes[i].set_title(f'Generated Sample {i+1}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Create and test VAE
    print("\n5. Testing VAE model...")
    vae = PointCloudVAE(
        num_points=256,
        latent_dim=16,
        hidden_dim=128,
        beta=1.0,
    ).to(device)
    
    # Test VAE forward pass
    test_batch = batch.to(device)
    recon, mu, logvar = vae(test_batch)
    
    print(f"VAE reconstruction shape: {recon.shape}")
    print(f"VAE latent mean shape: {mu.shape}")
    print(f"VAE latent logvar shape: {logvar.shape}")
    
    # Generate from VAE
    with torch.no_grad():
        vae_generated = vae.generate(2, device)
    
    # Visualize VAE samples
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': '3d'})
    
    for i in range(2):
        points = vae_generated[i].cpu().numpy()
        axes[i].scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
        axes[i].set_title(f'VAE Generated Sample {i+1}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()
    
    # 6. Evaluate models
    print("\n6. Evaluating models...")
    with torch.no_grad():
        # Generator evaluation
        z = torch.randn(test_batch.size(0), 32, device=device)
        gen_samples = model(z)
        gen_chamfer = torch.mean(compute_chamfer_distance(gen_samples, test_batch))
        
        # VAE evaluation
        vae_samples = vae.generate(test_batch.size(0), device)
        vae_chamfer = torch.mean(compute_chamfer_distance(vae_samples, test_batch))
        
        print(f"Generator Chamfer Distance: {gen_chamfer:.4f}")
        print(f"VAE Chamfer Distance: {vae_chamfer:.4f}")
    
    print("\n✓ Example completed successfully!")
    print("\nTo explore more:")
    print("- Run: python scripts/train.py --config configs/vae.yaml")
    print("- Run: streamlit run demo/streamlit_app.py")
    print("- Open: notebooks/quick_start.ipynb")


if __name__ == "__main__":
    main()

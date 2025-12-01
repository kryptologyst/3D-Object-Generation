"""Utility functions for 3D object generation."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device: Specific device to use. If None, auto-detect.
        
    Returns:
        PyTorch device object.
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def normalize_point_cloud(points: torch.Tensor) -> torch.Tensor:
    """Normalize point cloud to unit sphere.
    
    Args:
        points: Point cloud tensor of shape (N, 3).
        
    Returns:
        Normalized point cloud tensor.
    """
    centroid = torch.mean(points, dim=0)
    points = points - centroid
    scale = torch.max(torch.norm(points, dim=1))
    points = points / scale
    return points


def random_rotation(points: torch.Tensor) -> torch.Tensor:
    """Apply random rotation to point cloud.
    
    Args:
        points: Point cloud tensor of shape (N, 3).
        
    Returns:
        Rotated point cloud tensor.
    """
    # Generate random rotation matrix
    angle = torch.rand(3) * 2 * np.pi
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    
    # Rotation matrices for each axis
    Rx = torch.tensor([
        [1, 0, 0],
        [0, cos_a[0], -sin_a[0]],
        [0, sin_a[0], cos_a[0]]
    ], dtype=points.dtype, device=points.device)
    
    Ry = torch.tensor([
        [cos_a[1], 0, sin_a[1]],
        [0, 1, 0],
        [-sin_a[1], 0, cos_a[1]]
    ], dtype=points.dtype, device=points.device)
    
    Rz = torch.tensor([
        [cos_a[2], -sin_a[2], 0],
        [sin_a[2], cos_a[2], 0],
        [0, 0, 1]
    ], dtype=points.dtype, device=points.device)
    
    # Combine rotations
    R = Rz @ Ry @ Rx
    return points @ R.T


def jitter_point_cloud(points: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise to point cloud.
    
    Args:
        points: Point cloud tensor of shape (N, 3).
        sigma: Standard deviation of noise.
        
    Returns:
        Jittered point cloud tensor.
    """
    noise = torch.randn_like(points) * sigma
    return points + noise

"""Data loading and preprocessing for 3D point clouds."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from sklearn.datasets import make_sphere, make_cube, make_cylinder


class PointCloudDataset(Dataset):
    """Dataset for 3D point clouds.
    
    Supports both real datasets and synthetic geometric shapes.
    """
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        num_points: int = 1024,
        synthetic: bool = True,
        shape_types: Optional[List[str]] = None,
        normalize: bool = True,
        augment: bool = False,
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing point cloud files.
            num_points: Number of points per point cloud.
            synthetic: Whether to use synthetic geometric shapes.
            shape_types: Types of synthetic shapes to generate.
            normalize: Whether to normalize point clouds.
            augment: Whether to apply data augmentation.
        """
        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment
        
        if synthetic:
            self.shape_types = shape_types or ["sphere", "cube", "cylinder"]
            self.data = self._generate_synthetic_data()
        else:
            if data_dir is None:
                raise ValueError("data_dir must be provided when synthetic=False")
            self.data = self._load_real_data(data_dir)
    
    def _generate_synthetic_data(self) -> List[np.ndarray]:
        """Generate synthetic geometric shapes."""
        data = []
        
        for shape_type in self.shape_types:
            if shape_type == "sphere":
                points, _ = make_sphere(n_samples=self.num_points, noise=0.01)
            elif shape_type == "cube":
                points = self._generate_cube_points()
            elif shape_type == "cylinder":
                points = self._generate_cylinder_points()
            else:
                raise ValueError(f"Unknown shape type: {shape_type}")
            
            data.append(points)
        
        return data
    
    def _generate_cube_points(self) -> np.ndarray:
        """Generate points on a cube surface."""
        # Generate points on each face of a cube
        points_per_face = self.num_points // 6
        points = []
        
        for face in range(6):
            if face < 2:  # Front and back faces
                x = np.random.uniform(-1, 1, points_per_face)
                y = np.random.uniform(-1, 1, points_per_face)
                z = np.full(points_per_face, 1 if face == 0 else -1)
            elif face < 4:  # Left and right faces
                x = np.full(points_per_face, 1 if face == 2 else -1)
                y = np.random.uniform(-1, 1, points_per_face)
                z = np.random.uniform(-1, 1, points_per_face)
            else:  # Top and bottom faces
                x = np.random.uniform(-1, 1, points_per_face)
                y = np.full(points_per_face, 1 if face == 4 else -1)
                z = np.random.uniform(-1, 1, points_per_face)
            
            face_points = np.column_stack([x, y, z])
            points.append(face_points)
        
        return np.vstack(points)[:self.num_points]
    
    def _generate_cylinder_points(self) -> np.ndarray:
        """Generate points on a cylinder surface."""
        # Generate points on cylinder surface
        theta = np.random.uniform(0, 2 * np.pi, self.num_points // 2)
        z = np.random.uniform(-1, 1, self.num_points // 2)
        
        # Cylinder surface points
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Add top and bottom faces
        top_bottom_points = self.num_points - len(theta)
        theta_faces = np.random.uniform(0, 2 * np.pi, top_bottom_points)
        r = np.random.uniform(0, 1, top_bottom_points)
        
        x_faces = r * np.cos(theta_faces)
        y_faces = r * np.sin(theta_faces)
        z_faces = np.random.choice([-1, 1], top_bottom_points)
        
        points = np.column_stack([
            np.concatenate([x, x_faces]),
            np.concatenate([y, y_faces]),
            np.concatenate([z, z_faces])
        ])
        
        return points[:self.num_points]
    
    def _load_real_data(self, data_dir: Union[str, Path]) -> List[np.ndarray]:
        """Load real point cloud data from files."""
        data_dir = Path(data_dir)
        data = []
        
        for file_path in data_dir.glob("*.ply"):
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points)
            
            # Downsample if necessary
            if len(points) > self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=False)
                points = points[indices]
            elif len(points) < self.num_points:
                # Pad with random points
                pad_size = self.num_points - len(points)
                pad_points = np.random.uniform(-1, 1, (pad_size, 3))
                points = np.vstack([points, pad_points])
            
            data.append(points)
        
        return data
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item from dataset."""
        points = self.data[idx].copy()
        
        # Convert to tensor
        points = torch.tensor(points, dtype=torch.float32)
        
        # Normalize if requested
        if self.normalize:
            centroid = torch.mean(points, dim=0)
            points = points - centroid
            scale = torch.max(torch.norm(points, dim=1))
            points = points / scale
        
        # Apply augmentation if requested
        if self.augment:
            if torch.rand(1) < 0.5:
                points = self._random_rotation(points)
            if torch.rand(1) < 0.5:
                points = self._jitter_points(points)
        
        return points
    
    def _random_rotation(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random rotation to points."""
        angle = torch.rand(3) * 2 * np.pi
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_a[0], -sin_a[0]],
            [0, sin_a[0], cos_a[0]]
        ], dtype=points.dtype)
        
        Ry = torch.tensor([
            [cos_a[1], 0, sin_a[1]],
            [0, 1, 0],
            [-sin_a[1], 0, cos_a[1]]
        ], dtype=points.dtype)
        
        Rz = torch.tensor([
            [cos_a[2], -sin_a[2], 0],
            [sin_a[2], cos_a[2], 0],
            [0, 0, 1]
        ], dtype=points.dtype)
        
        R = Rz @ Ry @ Rx
        return points @ R.T
    
    def _jitter_points(self, points: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to points."""
        noise = torch.randn_like(points) * sigma
        return points + noise


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader for point cloud dataset.
    
    Args:
        dataset: Point cloud dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.
        
    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda x: torch.stack(x),
    )

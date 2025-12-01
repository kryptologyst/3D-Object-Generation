"""Evaluation metrics for 3D point cloud generation."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn


def compute_chamfer_distance(
    pred: torch.Tensor, 
    target: torch.Tensor
) -> torch.Tensor:
    """Compute Chamfer distance between two point clouds.
    
    Args:
        pred: Predicted point cloud of shape (batch_size, num_points, 3).
        target: Target point cloud of shape (batch_size, num_points, 3).
        
    Returns:
        Chamfer distance tensor of shape (batch_size,).
    """
    # Compute pairwise distances
    dist_matrix = torch.cdist(pred, target, p=2)
    
    # Chamfer distance: min distance from pred to target + min distance from target to pred
    dist1 = torch.min(dist_matrix, dim=2)[0]  # min distance from pred to target
    dist2 = torch.min(dist_matrix, dim=1)[0]  # min distance from target to pred
    
    chamfer_dist = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    return chamfer_dist


def compute_emd(
    pred: torch.Tensor, 
    target: torch.Tensor,
    max_iter: int = 50
) -> torch.Tensor:
    """Compute Earth Mover's Distance (EMD) between two point clouds.
    
    This is a simplified implementation using iterative assignment.
    
    Args:
        pred: Predicted point cloud of shape (batch_size, num_points, 3).
        target: Target point cloud of shape (batch_size, num_points, 3).
        max_iter: Maximum number of iterations for EMD computation.
        
    Returns:
        EMD tensor of shape (batch_size,).
    """
    batch_size, num_points, _ = pred.shape
    device = pred.device
    
    emd_values = torch.zeros(batch_size, device=device)
    
    for b in range(batch_size):
        pred_b = pred[b]  # (num_points, 3)
        target_b = target[b]  # (num_points, 3)
        
        # Initialize assignment
        assignment = torch.arange(num_points, device=device)
        
        # Iterative assignment
        for _ in range(max_iter):
            # Compute distances for current assignment
            distances = torch.norm(pred_b - target_b[assignment], dim=1)
            
            # Find better assignments
            for i in range(num_points):
                best_j = torch.argmin(torch.norm(pred_b[i:i+1] - target_b, dim=1))
                if torch.norm(pred_b[i] - target_b[best_j]) < distances[i]:
                    assignment[i] = best_j
            
            # Check convergence
            new_distances = torch.norm(pred_b - target_b[assignment], dim=1)
            if torch.allclose(distances, new_distances, atol=1e-6):
                break
        
        emd_values[b] = torch.mean(torch.norm(pred_b - target_b[assignment], dim=1))
    
    return emd_values


def compute_coverage(
    pred: torch.Tensor, 
    target: torch.Tensor,
    threshold: float = 0.1
) -> torch.Tensor:
    """Compute coverage metric for point clouds.
    
    Args:
        pred: Predicted point cloud of shape (batch_size, num_points, 3).
        target: Target point cloud of shape (batch_size, num_points, 3).
        threshold: Distance threshold for coverage.
        
    Returns:
        Coverage tensor of shape (batch_size,).
    """
    dist_matrix = torch.cdist(pred, target, p=2)
    min_distances = torch.min(dist_matrix, dim=2)[0]
    coverage = torch.mean((min_distances < threshold).float(), dim=1)
    return coverage


def compute_minimum_matching_distance(
    pred: torch.Tensor, 
    target: torch.Tensor
) -> torch.Tensor:
    """Compute minimum matching distance (MMD).
    
    Args:
        pred: Predicted point cloud of shape (batch_size, num_points, 3).
        target: Target point cloud of shape (batch_size, num_points, 3).
        
    Returns:
        MMD tensor of shape (batch_size,).
    """
    dist_matrix = torch.cdist(pred, target, p=2)
    mmd = torch.mean(torch.min(dist_matrix, dim=2)[0], dim=1)
    return mmd


class ChamferDistance(Metric):
    """Chamfer distance metric for PyTorch Lightning."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_chamfer", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metric with new predictions and targets."""
        chamfer_dist = compute_chamfer_distance(pred, target)
        self.sum_chamfer += torch.sum(chamfer_dist)
        self.total += pred.size(0)
    
    def compute(self) -> torch.Tensor:
        """Compute final metric value."""
        return self.sum_chamfer / self.total


class EMD(Metric):
    """Earth Mover's Distance metric for PyTorch Lightning."""
    
    def __init__(self, max_iter: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.add_state("sum_emd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metric with new predictions and targets."""
        emd_dist = compute_emd(pred, target, self.max_iter)
        self.sum_emd += torch.sum(emd_dist)
        self.total += pred.size(0)
    
    def compute(self) -> torch.Tensor:
        """Compute final metric value."""
        return self.sum_emd / self.total


class Coverage(Metric):
    """Coverage metric for PyTorch Lightning."""
    
    def __init__(self, threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("sum_coverage", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metric with new predictions and targets."""
        coverage = compute_coverage(pred, target, self.threshold)
        self.sum_coverage += torch.sum(coverage)
        self.total += pred.size(0)
    
    def compute(self) -> torch.Tensor:
        """Compute final metric value."""
        return self.sum_coverage / self.total


class Evaluator:
    """Comprehensive evaluator for 3D point cloud generation models."""
    
    def __init__(self, device: torch.device):
        """Initialize evaluator.
        
        Args:
            device: Device to run evaluation on.
        """
        self.device = device
        self.metrics = {
            'chamfer_distance': ChamferDistance().to(device),
            'emd': EMD().to(device),
            'coverage': Coverage().to(device),
        }
    
    def evaluate(
        self, 
        model: nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model on dataset.
        
        Args:
            model: Model to evaluate.
            dataloader: Data loader for evaluation.
            num_samples: Number of samples to evaluate (None for all).
            
        Returns:
            Dictionary of metric values.
        """
        model.eval()
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_samples is not None and i * dataloader.batch_size >= num_samples:
                    break
                
                batch = batch.to(self.device)
                
                # Generate samples
                if hasattr(model, 'generate'):
                    generated = model.generate(batch.size(0), self.device)
                else:
                    # For VAE, sample from prior
                    if hasattr(model, 'latent_dim'):
                        z = torch.randn(batch.size(0), model.latent_dim, device=self.device)
                        generated = model.decode(z)
                    else:
                        z = torch.randn(batch.size(0), model.z_dim, device=self.device)
                        generated = model(z)
                
                # Update metrics
                for metric in self.metrics.values():
                    metric.update(generated, batch)
        
        # Compute final metrics
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute().item()
        
        return results
    
    def evaluate_generation_quality(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate quality of generated samples against reference.
        
        Args:
            generated: Generated point clouds.
            reference: Reference point clouds.
            
        Returns:
            Dictionary of quality metrics.
        """
        results = {}
        
        # Chamfer distance
        chamfer_dist = compute_chamfer_distance(generated, reference)
        results['chamfer_distance'] = torch.mean(chamfer_dist).item()
        results['chamfer_distance_std'] = torch.std(chamfer_dist).item()
        
        # EMD
        emd_dist = compute_emd(generated, reference)
        results['emd'] = torch.mean(emd_dist).item()
        results['emd_std'] = torch.std(emd_dist).item()
        
        # Coverage
        coverage = compute_coverage(generated, reference)
        results['coverage'] = torch.mean(coverage).item()
        results['coverage_std'] = torch.std(coverage).item()
        
        # MMD
        mmd = compute_minimum_matching_distance(generated, reference)
        results['mmd'] = torch.mean(mmd).item()
        results['mmd_std'] = torch.std(mmd).item()
        
        return results

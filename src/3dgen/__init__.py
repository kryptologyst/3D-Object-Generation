"""3D Object Generation Package.

A modern implementation of 3D object generation using various generative models
including GANs, VAEs, and Diffusion models for point cloud generation.
"""

__version__ = "0.1.0"
__author__ = "AI Projects"

from .utils import set_seed, get_device
from .data import PointCloudDataset, get_dataloader
from .models import PointCloudGenerator, PointCloudVAE, PointCloudGAN, PointCloudDiffusion
from .training import Trainer
from .evaluation import Evaluator, compute_chamfer_distance, compute_emd

__all__ = [
    "set_seed",
    "get_device", 
    "PointCloudDataset",
    "get_dataloader",
    "PointCloudGenerator",
    "PointCloudVAE", 
    "PointCloudGAN",
    "PointCloudDiffusion",
    "Trainer",
    "Evaluator",
    "compute_chamfer_distance",
    "compute_emd",
]

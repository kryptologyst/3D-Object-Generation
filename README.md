# 3D Object Generation

An implementation of 3D object generation using various generative models including GANs, VAEs, and Diffusion models for point cloud generation.

## Features

- **Multiple Model Architectures**: VAE, GAN, Diffusion, and basic Generator models
- **3D-Specific Metrics**: Chamfer Distance, Earth Mover's Distance (EMD), Coverage, and MMD
- **Synthetic Data Generation**: Built-in geometric shape generation (spheres, cubes, cylinders)
- **Modern Training Framework**: PyTorch Lightning with configurable training
- **Interactive Demo**: Streamlit web interface for model interaction
- **Comprehensive Evaluation**: Automated evaluation with multiple metrics
- **Production Ready**: Type hints, documentation, tests, and CI/CD

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/3D-Object-Generation.git
cd 3D-Object-Generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Training a Model

1. **VAE Model**:
```bash
python scripts/train.py --config configs/vae.yaml --epochs 50
```

2. **GAN Model**:
```bash
python scripts/train.py --config configs/gan.yaml --epochs 100
```

3. **Diffusion Model**:
```bash
python scripts/train.py --config configs/diffusion.yaml --epochs 200
```

### Generating Samples

```bash
python scripts/sample.py --config configs/vae.yaml --checkpoint checkpoints/best_model.pt --num-samples 10
```

### Interactive Demo

```bash
streamlit run demo/streamlit_app.py
```

## Model Architectures

### Variational Autoencoder (VAE)
- **Architecture**: Encoder-decoder with reparameterization trick
- **Loss**: Reconstruction loss + KL divergence
- **Use Case**: Smooth latent space interpolation, controlled generation

### Generative Adversarial Network (GAN)
- **Architecture**: Generator-discriminator with spectral normalization
- **Loss**: Adversarial loss with gradient penalty
- **Use Case**: High-quality generation, mode collapse prevention

### Diffusion Model
- **Architecture**: Noise prediction network with timestep embedding
- **Loss**: MSE loss between predicted and actual noise
- **Use Case**: Stable training, high-quality generation

### Basic Generator
- **Architecture**: Simple fully connected network
- **Loss**: Chamfer distance
- **Use Case**: Baseline model, quick prototyping

## Data Pipeline

### Synthetic Data
The framework includes built-in synthetic data generation for geometric shapes:

- **Spheres**: Uniform sampling on sphere surface
- **Cubes**: Surface sampling on cube faces
- **Cylinders**: Surface sampling on cylinder and end caps

### Data Augmentation
- Random rotation
- Gaussian noise jittering
- Point cloud normalization

### Real Data Support
- PLY file format support
- Automatic point cloud normalization
- Configurable point count

## Evaluation Metrics

### Chamfer Distance
Measures the average distance between corresponding points in two point clouds.

### Earth Mover's Distance (EMD)
Computes the minimum cost to transform one point cloud into another.

### Coverage
Measures the fraction of points in the target point cloud that are within a threshold distance of the generated point cloud.

### Minimum Matching Distance (MMD)
Computes the average minimum distance from generated points to target points.

## Configuration

Models are configured using YAML files in the `configs/` directory:

```yaml
# Model configuration
model:
  type: "vae"
  z_dim: 128
  num_points: 1024
  hidden_dim: 512
  learning_rate: 1e-4
  weight_decay: 1e-5

# Data configuration
data:
  batch_size: 32
  num_workers: 4
  synthetic: true
  shape_types: ["sphere", "cube", "cylinder"]
  normalize: true
  augment: true

# Training configuration
training:
  max_epochs: 100
  patience: 10
  gradient_clip_val: 1.0
  precision: 32
  accelerator: "auto"
```

## Project Structure

```
3d-object-generation/
├── src/3dgen/           # Main package
│   ├── __init__.py
│   ├── models.py        # Model architectures
│   ├── data.py          # Data loading and preprocessing
│   ├── training.py      # Training framework
│   ├── evaluation.py    # Evaluation metrics
│   └── utils.py         # Utility functions
├── configs/             # Configuration files
├── scripts/             # Training and sampling scripts
├── demo/                # Interactive demos
├── tests/               # Unit tests
├── data/                # Data directory
├── checkpoints/         # Model checkpoints
├── logs/                # Training logs
├── outputs/             # Generated samples
└── requirements.txt     # Dependencies
```

## Training Commands

### Basic Training
```bash
python scripts/train.py --config configs/default.yaml
```

### Custom Parameters
```bash
python scripts/train.py \
    --config configs/vae.yaml \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 2e-4 \
    --seed 123
```

### Resume Training
```bash
python scripts/train.py \
    --config configs/vae.yaml \
    --checkpoint checkpoints/last.ckpt
```

## Sampling Commands

### Generate Samples
```bash
python scripts/sample.py \
    --config configs/vae.yaml \
    --checkpoint checkpoints/best_model.pt \
    --num-samples 20 \
    --output-dir outputs/
```

### Interpolation
```bash
python scripts/sample.py \
    --config configs/vae.yaml \
    --checkpoint checkpoints/best_model.pt \
    --interpolate \
    --interpolation-steps 20
```

### Diffusion Sampling
```bash
python scripts/sample.py \
    --config configs/diffusion.yaml \
    --checkpoint checkpoints/diffusion_model.pt \
    --steps 100 \
    --num-samples 10
```

## Interactive Demo

The Streamlit demo provides an interactive interface for:

- Model upload and selection
- Parameter adjustment
- Real-time sample generation
- 3D visualization
- Sample download

### Launch Demo
```bash
streamlit run demo/streamlit_app.py
```

### Demo Features
- **Model Selection**: Choose between VAE, GAN, Diffusion, and Generator
- **Parameter Control**: Adjust generation parameters
- **3D Visualization**: Interactive 3D point cloud visualization
- **Download Options**: Save samples as PLY files or ZIP archives
- **Interpolation**: Generate smooth interpolations between samples

## Evaluation

### Automated Evaluation
```bash
python scripts/evaluate.py \
    --config configs/vae.yaml \
    --checkpoint checkpoints/best_model.pt \
    --test-data data/test/
```

### Metric Computation
```python
from src.3dgen import Evaluator, compute_chamfer_distance

evaluator = Evaluator(device)
results = evaluator.evaluate(model, test_dataloader)
print(f"Chamfer Distance: {results['chamfer_distance']:.4f}")
print(f"EMD: {results['emd']:.4f}")
print(f"Coverage: {results['coverage']:.4f}")
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/ scripts/
ruff check src/ tests/ scripts/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Model Cards

### VAE Model
- **Data Sources**: Synthetic geometric shapes
- **Intended Use**: 3D object generation, latent space exploration
- **Limitations**: May produce blurry reconstructions
- **Bias Considerations**: Trained on synthetic data only

### GAN Model
- **Data Sources**: Synthetic geometric shapes
- **Intended Use**: High-quality 3D object generation
- **Limitations**: Potential mode collapse, training instability
- **Bias Considerations**: May favor certain geometric patterns

### Diffusion Model
- **Data Sources**: Synthetic geometric shapes
- **Intended Use**: Stable 3D object generation
- **Limitations**: Slower generation, requires more compute
- **Bias Considerations**: Inherits biases from training data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{3d_object_generation,
  title={3D Object Generation: A Modern Framework for Point Cloud Generation},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/3D-Object-Generation}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- PyTorch Lightning for the training framework
- Open3D for 3D data processing
- Streamlit for the interactive demo
- The open-source community for various dependencies
# 3D-Object-Generation

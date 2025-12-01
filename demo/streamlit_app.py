"""Streamlit demo for 3D point cloud generation."""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import tempfile
import zipfile

from src.3dgen import (
    PointCloudGenerator,
    PointCloudVAE,
    PointCloudGAN,
    PointCloudDiffusion,
    set_seed,
    get_device,
)


def load_model(model_type: str, checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    if model_type == "vae":
        model = PointCloudVAE(
            num_points=1024,
            latent_dim=128,
            hidden_dim=512,
            beta=1.0,
        )
    elif model_type == "gan":
        model = PointCloudGAN(
            z_dim=100,
            num_points=1024,
            hidden_dim=512,
        )
    elif model_type == "diffusion":
        model = PointCloudDiffusion(
            num_points=1024,
            hidden_dim=512,
            num_timesteps=1000,
        )
    elif model_type == "generator":
        model = PointCloudGenerator(
            z_dim=100,
            num_points=1024,
            hidden_dim=512,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model.to(device)


def generate_samples(model, num_samples: int, device: torch.device, steps: int = None):
    """Generate samples from the model."""
    with torch.no_grad():
        if hasattr(model, 'generate'):
            if steps is not None and hasattr(model, 'num_timesteps'):
                samples = model.generate(num_samples, device, steps)
            else:
                samples = model.generate(num_samples, device)
        else:
            if hasattr(model, 'latent_dim'):
                z = torch.randn(num_samples, model.latent_dim, device=device)
                samples = model.decode(z)
            else:
                z = torch.randn(num_samples, model.z_dim, device=device)
                samples = model(z)
    
    return samples


def create_3d_plot(points: np.ndarray, title: str = "Point Cloud"):
    """Create 3D plot using Plotly."""
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points[:, 2],
            colorscale='viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        width=600,
        height=500
    )
    
    return fig


def save_point_cloud(points: np.ndarray, filename: str):
    """Save point cloud as PLY file."""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="3D Object Generation",
        page_icon="🎲",
        layout="wide"
    )
    
    st.title("🎲 3D Object Generation Demo")
    st.markdown("Generate 3D point clouds using various generative models")
    
    # Sidebar
    st.sidebar.header("Model Configuration")
    
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["vae", "gan", "diffusion", "generator"],
        help="Choose the type of generative model"
    )
    
    # Model upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Model Checkpoint",
        type=['pt', 'pth'],
        help="Upload a trained model checkpoint"
    )
    
    if uploaded_file is None:
        st.warning("Please upload a model checkpoint to generate samples.")
        st.stop()
    
    # Generation parameters
    st.sidebar.header("Generation Parameters")
    
    num_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of point clouds to generate"
    )
    
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=1000000,
        value=42,
        help="Random seed for reproducible generation"
    )
    
    steps = None
    if model_type == "diffusion":
        steps = st.sidebar.slider(
            "Denoising Steps",
            min_value=10,
            max_value=1000,
            value=100,
            help="Number of denoising steps for diffusion model"
        )
    
    # Set seed
    set_seed(seed)
    
    # Get device
    device = get_device()
    
    # Load model
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.flush()
            
            model = load_model(model_type, tmp_file.name, device)
            st.sidebar.success("Model loaded successfully!")
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Generate button
    if st.sidebar.button("Generate Samples", type="primary"):
        with st.spinner("Generating samples..."):
            try:
                # Generate samples
                samples = generate_samples(model, num_samples, device, steps)
                
                # Display samples
                st.header("Generated Samples")
                
                cols = st.columns(2)
                
                for i in range(num_samples):
                    col_idx = i % 2
                    with cols[col_idx]:
                        points = samples[i].cpu().numpy()
                        fig = create_3d_plot(points, f"Sample {i + 1}")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                st.header("Download Options")
                
                # Create zip file with all samples
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                    with zipfile.ZipFile(tmp_zip.name, 'w') as zip_file:
                        for i in range(num_samples):
                            points = samples[i].cpu().numpy()
                            
                            # Save as PLY
                            ply_filename = f"sample_{i+1}.ply"
                            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ply') as ply_file:
                                save_point_cloud(points, ply_file.name)
                                zip_file.write(ply_file.name, ply_filename)
                                Path(ply_file.name).unlink()
                    
                    # Download button
                    with open(tmp_zip.name, 'rb') as f:
                        st.download_button(
                            label="Download All Samples (ZIP)",
                            data=f.read(),
                            file_name="generated_samples.zip",
                            mime="application/zip"
                        )
                    
                    Path(tmp_zip.name).unlink()
                
                # Individual sample downloads
                st.subheader("Download Individual Samples")
                for i in range(num_samples):
                    points = samples[i].cpu().numpy()
                    
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ply') as tmp_file:
                        save_point_cloud(points, tmp_file.name)
                        
                        with open(tmp_file.name, 'r') as f:
                            st.download_button(
                                label=f"Download Sample {i+1}",
                                data=f.read(),
                                file_name=f"sample_{i+1}.ply",
                                mime="text/plain"
                            )
                        
                        Path(tmp_file.name).unlink()
                
            except Exception as e:
                st.error(f"Error generating samples: {str(e)}")
    
    # Model information
    st.sidebar.header("Model Information")
    st.sidebar.info(f"Model Type: {model_type.upper()}")
    st.sidebar.info(f"Device: {device}")
    st.sidebar.info(f"Number of Points: 1024")
    
    # Instructions
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Upload a trained model checkpoint
    2. Adjust generation parameters
    3. Click "Generate Samples"
    4. View and download generated point clouds
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and PyTorch")


if __name__ == "__main__":
    main()

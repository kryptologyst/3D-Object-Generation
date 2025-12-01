#!/usr/bin/env python3
"""Setup script for 3D object generation project."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up 3D Object Generation project...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("✗ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        sys.exit(1)
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        sys.exit(1)
    
    # Install pre-commit hooks
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        print("Warning: Pre-commit hooks installation failed, continuing...")
    
    # Create necessary directories
    directories = ["data", "checkpoints", "logs", "outputs", "assets"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("Warning: Some tests failed, but setup completed")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train a model: python scripts/train.py --config configs/vae.yaml")
    print("2. Generate samples: python scripts/sample.py --config configs/vae.yaml --checkpoint checkpoints/best_model.pt")
    print("3. Run demo: streamlit run demo/streamlit_app.py")
    print("4. Open notebook: jupyter notebook notebooks/quick_start.ipynb")


if __name__ == "__main__":
    main()

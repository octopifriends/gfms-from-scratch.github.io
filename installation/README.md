# GeoAI Course Installation Guide

This directory contains installation files and scripts for setting up the GeoAI course environment.

## 📁 Files Overview

| File | Description |
|------|-------------|
| `environment.yml` | General conda environment configuration |
| `GRIT_SETUP.md` | Step-by-step GRIT installation playbook (this is your primary runbook) |
| `environment-mac-gpu.yml` | **Mac-specific environment with GPU support** |
| `requirements.txt` | Pip requirements file (Mac GPU compatible) |
| `setup_geoai_mac.sh` | **Automated setup script for Mac** |
| `verify_geoai_setup.py` | Environment verification script |
| `test_gpu_setup_universal.py` | **Universal GPU testing (Mac MPS + Linux CUDA)** |
| `platform_setup_guide.py` | **Platform-specific installation guidance** |

## 🚀 Quick Setup (Mac Users)

For Mac users (especially Apple Silicon), use the automated setup:

```bash
# Clone the repository and navigate to installation directory
cd installation

# Run the automated setup script
./setup_geoai_mac.sh
```

This script will:
- Create the conda environment with Mac GPU (MPS) support
- Install all required packages
- Set up the Jupyter kernel
- Test GPU functionality

For GRIT or Linux GPU servers, follow the comprehensive runbook:

```bash
open GRIT_SETUP.md
```

## 🔧 Manual Setup Options

### Option 1: Using Conda Environment File (Recommended for Mac)

```bash
# Create environment from Mac-specific file
conda env create -f environment-mac-gpu.yml

# Activate environment
conda activate geoAI

# Install Jupyter kernel
python -m ipykernel install --user --name geoai --display-name "GeoAI"
```

### Option 2: Using Requirements.txt

```bash
# Create new environment
conda create -n geoAI python=3.11 -y
conda activate geoAI

# Install conda packages first (recommended)
conda install pytorch torchvision torchaudio gdal rasterio -c pytorch -c conda-forge -y

# Install remaining packages with pip
pip install -r requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name geoai --display-name "GeoAI"
```

## 🧪 Verification & Testing

After installation, verify your setup:

```bash
# Activate environment
conda activate geoAI

# Platform-specific setup guidance
python platform_setup_guide.py

# Universal GPU testing (works on Mac and Linux)
python test_gpu_setup_universal.py

# Complete environment verification
python verify_geoai_setup.py
```

### Testing Scripts:

**`platform_setup_guide.py`** - Detects your platform and provides specific setup instructions:
- Mac (Intel vs Apple Silicon)
- Linux (with/without NVIDIA GPU)
- Windows compatibility notes

**`test_gpu_setup_universal.py`** - Universal GPU testing:
- ✅ Detects Mac MPS vs Linux CUDA
- ✅ Performance benchmarking
- ✅ Platform-specific recommendations
- ✅ Universal device selection code

**`verify_geoai_setup.py`** - Complete environment verification:
- ✅ All package imports
- ✅ GPU functionality (MPS/CUDA/CPU)
- ✅ Jupyter kernel installation
- ✅ Basic tensor operations

## 🖥️ Platform-Specific Notes

### Mac (Apple Silicon)
- **GPU Support**: Uses Metal Performance Shaders (MPS)
- **Recommended File**: `environment-mac-gpu.yml`
- **PyTorch**: CPU + MPS support (no CUDA needed)
- **Setup Script**: `setup_geoai_mac.sh`

### Mac (Intel)
- **GPU Support**: Limited (CPU primarily)
- **Recommended File**: `environment.yml`
- **PyTorch**: CPU version

### Linux/Windows
- **GPU Support**: CUDA (if NVIDIA GPU available)
- **Recommended File**: `environment.yml` (may need CUDA modifications)
- **PyTorch**: CPU + CUDA support

## 📦 Key Packages Included

### Core AI/ML
- **PyTorch 2.2+** with MPS support
- **TorchVision & TorchAudio**
- **HuggingFace ecosystem** (transformers, datasets)
- **Scikit-learn**
- **PyTorch Lightning**

### Geospatial Libraries
- **GDAL** for geospatial data I/O
- **Rasterio** for raster processing
- **GeoPandas** for vector data
- **XArray** for multi-dimensional data
- **TorchGeo** for geospatial deep learning
- **Earth Engine API**

### Computer Vision
- **OpenCV**
- **Kornia** for differentiable computer vision
- **TIMM** for pre-trained models
- **Einops** for tensor operations

### Jupyter & Visualization
- **JupyterLab**
- **Matplotlib, Seaborn**
- **Folium** for interactive maps

## 🔍 Universal GPU Usage

For cross-platform compatibility (Mac MPS + Linux CUDA):

```python
import torch

# Universal device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps') 
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Use in your code
model = model.to(device)
data = data.to(device)
```

### Platform-Specific Usage:

**Mac (Apple Silicon):**
```python
# Mac MPS-specific
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

**Linux (NVIDIA GPU):**
```python
# Linux CUDA-specific
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## 🐛 Troubleshooting

### Common Issues

1. **MPS not available**
   - Ensure you have macOS 12.3+ and Apple Silicon Mac
   - Check PyTorch version supports MPS (2.0+)

2. **Package conflicts**
   - Use `environment-mac-gpu.yml` for clean installation
   - Avoid mixing conda and pip installations

3. **Jupyter kernel not found**
   - Run: `python -m ipykernel install --user --name geoai --display-name "GeoAI"`
   - Restart Jupyter/Quarto

4. **CUDA errors on Mac**
   - Mac doesn't support CUDA, use MPS instead
   - Use Mac-specific environment files

### Getting Help

- Run `verify_geoai_setup.py` to diagnose issues
- Check environment with: `conda list`
- Test GPU with: `python -c "import torch; print(torch.backends.mps.is_available())"`

## 📚 Course Integration

Once installed:

1. **Start Jupyter**: `jupyter lab`
2. **Select geoai kernel** in notebooks
3. **For Quarto**: `quarto preview` (kernel specified in `_quarto.yml`)
4. **GPU acceleration**: Use `device = torch.device('mps')` in your code

## 🔄 Environment Updates

To update packages:

```bash
conda activate geoAI
conda update --all
```

To recreate environment:

```bash
conda env remove -n geoAI
conda env create -f environment-mac-gpu.yml
```

---

🎯 **Ready to start your GeoAI journey with Mac GPU acceleration!**
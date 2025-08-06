#!/bin/bash

# GeoAI Course Environment Setup Script for Mac
# Optimized for Apple Silicon with GPU (MPS) support

set -e  # Exit on any error

echo "🚀 Setting up GeoAI environment for Mac..."
echo "============================================"

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✅ Using mamba for faster package installation"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "✅ Using conda for package installation"
else
    echo "❌ Error: Neither conda nor mamba found. Please install Miniconda or Mambaforge first."
    echo "   Download from: https://github.com/conda-forge/miniforge"
    exit 1
fi

# Check macOS version for MPS support
MACOS_VERSION=$(sw_vers -productVersion)
echo "📱 macOS version: $MACOS_VERSION"

# Create environment from Mac-specific YAML file
echo "🔨 Creating geoAI conda environment..."
if [ -f "environment-mac-gpu.yml" ]; then
    $CONDA_CMD env create -f environment-mac-gpu.yml
elif [ -f "../environment-mac-gpu.yml" ]; then
    $CONDA_CMD env create -f ../environment-mac-gpu.yml
else
    echo "❌ Error: environment-mac-gpu.yml not found"
    echo "   Please run this script from the project root or installation/ directory"
    exit 1
fi

# Activate the environment
echo "🔄 Activating geoAI environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate geoAI

# Install the Jupyter kernel
echo "📓 Installing Jupyter kernel..."
python -m ipykernel install --user --name geoai --display-name "GeoAI"

# Test GPU support
echo "🧪 Testing GPU (MPS) support..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print('✅ GPU acceleration ready!')
    device = torch.device('mps')
    x = torch.randn(10, 10, device=device)
    print(f'✅ Test tensor created on: {x.device}')
else:
    print('⚠️  MPS not available - using CPU only')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: conda activate geoAI"
echo "2. Start Jupyter: jupyter lab"
echo "3. Or start Quarto preview: quarto preview"
echo ""
echo "💡 Tips:"
echo "- Use device = torch.device('mps') for GPU acceleration in PyTorch"
echo "- The 'geoai' kernel is now available in Jupyter notebooks"
echo "- All course materials can now run with GPU support"
echo ""

# Check if environment was created successfully
if conda env list | grep -q "geoAI"; then
    echo "✅ Environment 'geoAI' created successfully"
else
    echo "❌ Environment creation may have failed"
    exit 1
fi
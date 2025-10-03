#!/usr/bin/env python3
"""
Platform-Specific Setup Guide for GeoAI Environment
Provides installation instructions based on detected system
"""

import platform
import subprocess
import sys
import os

def detect_platform():
    """Detect platform and provide specific setup instructions"""
    system = platform.system()
    machine = platform.machine()
    
    print("🔍 Platform Detection for GeoAI Setup")
    print("=" * 45)
    print(f"System: {system}")
    print(f"Architecture: {machine}")
    print(f"Python: {platform.python_version()}")
    
    return system, machine

def check_conda_installation():
    """Check if conda/mamba is available"""
    conda_available = False
    mamba_available = False
    
    try:
        subprocess.run(['conda', '--version'], capture_output=True, check=True)
        conda_available = True
    except:
        pass
    
    try:
        subprocess.run(['mamba', '--version'], capture_output=True, check=True)
        mamba_available = True
    except:
        pass
    
    return conda_available, mamba_available

def mac_setup_guide(machine):
    """Mac-specific setup instructions"""
    print("\n🍎 Mac Setup Instructions")
    print("-" * 30)
    
    if machine == "arm64":
        print("✅ Apple Silicon Mac detected - GPU acceleration available!")
        print("\n📦 Recommended installation method:")
        print("   ./installation/setup_geoai_mac.sh")
        print("\n📋 Manual steps:")
        print("   1. conda env create -f environment-mac-gpu.yml")
        print("   2. conda activate geoAI")
        print("   3. python -m ipykernel install --user --name geoai --display-name 'GeoAI'")
        
        print("\n🚀 GPU Features:")
        print("   • Metal Performance Shaders (MPS) acceleration")
        print("   • Optimized for Apple Silicon neural engines")
        print("   • Use: device = torch.device('mps')")
        
    else:  # Intel Mac
        print("🔸 Intel Mac detected - Limited GPU acceleration")
        print("\n📦 Recommended installation:")
        print("   conda env create -f environment.yml")
        print("\n⚠️  Note: Intel Macs have limited GPU support")
        print("   • CPU-optimized PyTorch")
        print("   • Consider cloud GPU for intensive training")

def linux_setup_guide():
    """Linux-specific setup instructions"""
    print("\n🐧 Linux Setup Instructions")
    print("-" * 30)
    
    # Check for NVIDIA GPU
    nvidia_available = False
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        nvidia_available = (result.returncode == 0)
    except:
        pass
    
    if nvidia_available:
        print("✅ NVIDIA GPU detected!")
        print("\n📦 CUDA Installation:")
        print("   1. Install NVIDIA drivers (if not already installed)")
        print("   2. Install CUDA toolkit (11.8 or 12.1 recommended)")
        print("   3. Create environment with CUDA support:")
        print("      conda env create -f environment.yml")
        print("   4. Verify CUDA PyTorch:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        print("\n🚀 GPU Features:")
        print("   • Full CUDA acceleration")
        print("   • Support for large models")
        print("   • Use: device = torch.device('cuda')")
        
        print("\n🔧 Environment variables (optional):")
        print("   export CUDA_VISIBLE_DEVICES=0")
        print("   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
        
    else:
        print("🔸 No NVIDIA GPU detected")
        print("\n📦 CPU Installation:")
        print("   conda env create -f environment.yml")
        print("\n💡 For GPU acceleration:")
        print("   • Install NVIDIA drivers: apt install nvidia-driver-XXX")
        print("   • Install CUDA toolkit from NVIDIA website")
        print("   • Restart and run nvidia-smi to verify")

def windows_setup_guide():
    """Windows-specific setup instructions"""
    print("\n🪟 Windows Setup Instructions")
    print("-" * 30)
    print("📦 Recommended approach:")
    print("   1. Install Miniconda or Anaconda")
    print("   2. Use Windows Subsystem for Linux (WSL2) for better compatibility")
    print("   3. conda env create -f environment.yml")
    
    print("\n🚀 GPU Support:")
    print("   • Install NVIDIA drivers for Windows")
    print("   • Install CUDA toolkit (Windows version)")
    print("   • Use: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n💡 Alternative: WSL2 + CUDA")
    print("   • More compatible with Linux-based tools")
    print("   • Follow Linux instructions within WSL2")

def provide_next_steps(system):
    """Provide next steps after installation"""
    print(f"\n✅ Next Steps After Installation")
    print("-" * 35)
    print("1. Activate environment: conda activate geoAI")
    print("2. Test setup: python installation/test_gpu_setup_universal.py")
    print("3. Verify packages: python installation/verify_geoai_setup.py")
    print("4. Start Jupyter: jupyter lab")
    print("5. Test Quarto: quarto preview")
    
    print(f"\n📚 Universal device selection in code:")
    print("""
# Automatic device detection
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Silicon GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Use in your models
model = model.to(device)
data = data.to(device)
""")

def main():
    """Main platform detection and guidance"""
    system, machine = detect_platform()
    conda_available, mamba_available = check_conda_installation()
    
    # Check package manager
    print(f"\n📦 Package Managers:")
    print(f"   Conda: {'✅ Available' if conda_available else '❌ Not found'}")
    print(f"   Mamba: {'✅ Available (faster!)' if mamba_available else '❌ Not found'}")
    
    if not conda_available and not mamba_available:
        print("\n⚠️  No conda/mamba found!")
        print("   Install Miniconda: https://docs.conda.io/en/latest/miniconda.html")
        print("   Or Mambaforge: https://github.com/conda-forge/miniforge")
        return 1
    
    # Platform-specific guidance
    if system == "Darwin":
        mac_setup_guide(machine)
    elif system == "Linux":
        linux_setup_guide()
    elif system == "Windows":
        windows_setup_guide()
    else:
        print(f"\n❓ Unsupported platform: {system}")
        print("   Try the general Linux instructions")
    
    provide_next_steps(system)
    
    print(f"\n🎯 Quick Start (if conda/mamba available):")
    print("-" * 45)
    if system == "Darwin" and machine == "arm64":
        print("   ./installation/setup_geoai_mac.sh")
    else:
        print("   conda env create -f environment.yml")
        print("   conda activate geoAI")
        print("   python -m ipykernel install --user --name geoai")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
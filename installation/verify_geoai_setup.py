#!/usr/bin/env python3
"""
GeoAI Environment Verification Script
Tests all major components of the GeoAI course environment
"""

import sys
import subprocess
import importlib
from pathlib import Path

def test_import(package_name, display_name=None):
    """Test if a package can be imported"""
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {display_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {display_name}: Failed to import - {e}")
        return False

def test_gpu_support():
    """Test GPU support"""
    print("\n🧪 Testing GPU Support:")
    print("-" * 30)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) available")
            
            # Test basic operations
            device = torch.device('mps')
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.mm(x, y)
            
            print(f"✅ GPU tensor operations: SUCCESS")
            print(f"   Device: {z.device}")
            return True
        else:
            print("⚠️  MPS not available - using CPU only")
            print("   This could be due to:")
            print("   - Non-Apple Silicon Mac")
            print("   - macOS < 12.3")
            print("   - PyTorch version doesn't support MPS")
            return False
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_jupyter_kernel():
    """Test if Jupyter kernel is properly installed"""
    print("\n📓 Testing Jupyter Kernel:")
    print("-" * 30)
    
    try:
        result = subprocess.run(['jupyter', 'kernelspec', 'list'], 
                              capture_output=True, text=True)
        
        if 'geoai' in result.stdout:
            print("✅ geoai kernel found")
            return True
        else:
            print("❌ geoai kernel not found")
            print("   Run: python -m ipykernel install --user --name geoai --display-name 'GeoAI'")
            return False
            
    except Exception as e:
        print(f"❌ Jupyter kernel test failed: {e}")
        return False

def main():
    print("🔍 GeoAI Environment Verification")
    print("=" * 50)
    
    # Core packages
    print("\n📦 Core Scientific Packages:")
    print("-" * 30)
    core_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('scipy', 'SciPy')
    ]
    
    core_success = 0
    for pkg, name in core_packages:
        if test_import(pkg, name):
            core_success += 1
    
    # AI/ML packages
    print("\n🤖 AI/ML Packages:")
    print("-" * 30)
    ml_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('sklearn', 'Scikit-learn'),
        ('transformers', 'HuggingFace Transformers'),
        ('datasets', 'HuggingFace Datasets'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('timm', 'TIMM')
    ]
    
    ml_success = 0
    for pkg, name in ml_packages:
        if test_import(pkg, name):
            ml_success += 1
    
    # Geospatial packages
    print("\n🌍 Geospatial Packages:")
    print("-" * 30)
    geo_packages = [
        ('rasterio', 'Rasterio'),
        ('geopandas', 'GeoPandas'),
        ('xarray', 'XArray'),
        ('folium', 'Folium'),
        ('torchgeo', 'TorchGeo'),
        ('ee', 'Earth Engine API')
    ]
    
    geo_success = 0
    for pkg, name in geo_packages:
        if test_import(pkg, name):
            geo_success += 1
    
    # Computer vision packages
    print("\n👁️ Computer Vision Packages:")
    print("-" * 30)
    cv_packages = [
        ('cv2', 'OpenCV'),
        ('kornia', 'Kornia'),
        ('einops', 'Einops'),
        ('PIL', 'Pillow')
    ]
    
    cv_success = 0
    for pkg, name in cv_packages:
        if test_import(pkg, name):
            cv_success += 1
    
    # Jupyter packages
    print("\n📓 Jupyter Packages:")
    print("-" * 30)
    jupyter_packages = [
        ('jupyter', 'Jupyter'),
        ('jupyterlab', 'JupyterLab'),
        ('ipykernel', 'IPython Kernel'),
        ('notebook', 'Notebook')
    ]
    
    jupyter_success = 0
    for pkg, name in jupyter_packages:
        if test_import(pkg, name):
            jupyter_success += 1
    
    # Run additional tests
    gpu_ok = test_gpu_support()
    kernel_ok = test_jupyter_kernel()
    
    # Summary
    print("\n📊 Summary:")
    print("=" * 50)
    print(f"Core packages: {core_success}/{len(core_packages)}")
    print(f"AI/ML packages: {ml_success}/{len(ml_packages)}")
    print(f"Geospatial packages: {geo_success}/{len(geo_packages)}")
    print(f"Computer Vision packages: {cv_success}/{len(cv_packages)}")
    print(f"Jupyter packages: {jupyter_success}/{len(jupyter_packages)}")
    print(f"GPU support: {'✅' if gpu_ok else '⚠️'}")
    print(f"Jupyter kernel: {'✅' if kernel_ok else '❌'}")
    
    total_packages = len(core_packages) + len(ml_packages) + len(geo_packages) + len(cv_packages) + len(jupyter_packages)
    total_success = core_success + ml_success + geo_success + cv_success + jupyter_success
    
    print(f"\nOverall: {total_success}/{total_packages} packages working")
    
    if total_success == total_packages and gpu_ok and kernel_ok:
        print("\n🎉 Environment fully configured and ready for GeoAI work!")
        return 0
    elif total_success >= total_packages * 0.8:
        print("\n✅ Environment mostly configured. Some optional components missing.")
        return 0
    else:
        print("\n⚠️  Environment needs attention. Several packages are missing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Universal GPU Setup Test Script
Tests PyTorch GPU acceleration across Mac (MPS) and Linux (CUDA) environments
"""

import torch
import numpy as np
import sys
import platform
import subprocess
import os
from typing import Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:  # psutil is optional; we'll degrade gracefully
    psutil = None  # type: ignore

def get_system_info():
    """Get system information"""
    system = platform.system()
    machine = platform.machine()
    python_version = platform.python_version()
    
    print(f"🖥️  System: {system} {platform.release()}")
    print(f"🏗️  Architecture: {machine}")
    print(f"🐍 Python: {python_version}")
    
    return system, machine

def check_macos_version():
    """Check macOS version for MPS compatibility"""
    try:
        result = subprocess.run(['sw_vers', '-productVersion'], 
                              capture_output=True, text=True)
        version = result.stdout.strip()
        print(f"🍎 macOS Version: {version}")
        
        # Parse version (e.g., "12.3.1" -> [12, 3, 1])
        version_parts = [int(x) for x in version.split('.')]
        
        # MPS requires macOS 12.3+
        if version_parts[0] > 12 or (version_parts[0] == 12 and version_parts[1] >= 3):
            return True, version
        else:
            return False, version
            
    except Exception as e:
        print(f"⚠️  Could not determine macOS version: {e}")
        return False, "unknown"

def check_nvidia_driver():
    """Check NVIDIA driver on Linux"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract driver version from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version:' in line:
                    driver_version = line.split('Driver Version:')[1].split()[0]
                    print(f"🎮 NVIDIA Driver: {driver_version}")
                    return True, driver_version
            return True, "unknown"
        else:
            return False, None
    except FileNotFoundError:
        return False, None
    except Exception as e:
        print(f"⚠️  Error checking NVIDIA driver: {e}")
        return False, None

def test_device_performance(device, device_name):
    """Test basic performance on a device"""
    print(f"\n⚡ Performance test on {device_name}:")
    print("-" * 40)
    
    try:
        import time
        
        # Test different matrix sizes
        sizes = [100, 500, 1000]
        
        for size in sizes:
            # Create tensors
            start_time = time.time()
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Matrix multiplication
            c = torch.mm(a, b)
            
            # Force computation to complete
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            
            end_time = time.time()
            duration = (end_time - start_time) * 1000  # Convert to ms
            
            print(f"  {size}x{size} matrix multiplication: {duration:.2f} ms")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")


def _format_bytes(num_bytes: int) -> str:
    """Pretty-print bytes with binary units."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def _cuda_memory_info() -> Optional[Tuple[int, int]]:
    """Return (free_bytes, total_bytes) for current CUDA device if available."""
    try:
        if torch.cuda.is_available():
            # Ensure caches are cleared for a better estimate
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return int(free_bytes), int(total_bytes)
    except Exception:
        pass
    return None


def _system_memory_available() -> Optional[int]:
    """Return available system memory in bytes, if psutil is present."""
    try:
        if psutil is not None:
            return int(psutil.virtual_memory().available)
    except Exception:
        return None
    return None


def _estimate_max_model_params(available_bytes: int, bytes_per_param: int, safety_frac: float = 0.85) -> float:
    """Estimate max parameter count given memory and bytes-per-param. Returns billions of params."""
    if available_bytes <= 0 or bytes_per_param <= 0:
        return 0.0
    budget = available_bytes * safety_frac
    params = budget / float(bytes_per_param)
    return params / 1e9  # billions

def test_gpu_setup():
    """Main GPU testing function"""
    print("🧪 Universal GPU Setup Test")
    print("=" * 50)
    
    # Get system information
    system, machine = get_system_info()
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # Test different device types
    devices_tested = []
    recommended_device = "cpu"
    
    print("\n🔍 Checking available devices:")
    print("-" * 35)
    
    # 1. Test CPU (always available)
    print("🔸 CPU: Available")
    devices_tested.append(("cpu", "CPU"))
    
    # 2. Test CUDA (Linux/Windows with NVIDIA)
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        cuda_name = torch.cuda.get_device_name(0)
        print(f"🔸 CUDA: Available ({cuda_count} device(s))")
        print(f"   GPU: {cuda_name}")
        try:
            cc = torch.cuda.get_device_capability(0)
            print(f"   Compute Capability: {cc[0]}.{cc[1]}")
        except Exception:
            pass
        
        # Check NVIDIA driver on Linux
        if system == "Linux":
            driver_available, driver_version = check_nvidia_driver()
            if driver_available:
                print(f"   Driver: {driver_version}")
        
        devices_tested.append(("cuda", "CUDA GPU"))
        recommended_device = "cuda"

        # CUDA memory report
        mem = _cuda_memory_info()
        if mem is not None:
            free_b, total_b = mem
            print("\n🧠 CUDA Memory:")
            print("-" * 40)
            print(f"  Total: {_format_bytes(total_b)}  |  Free (now): {_format_bytes(free_b)}")

            # Rough model size limits
            print("\n📐 Estimated max model size (billions of parameters):")
            print("-" * 55)
            inf16 = _estimate_max_model_params(free_b, bytes_per_param=2)
            inf32 = _estimate_max_model_params(free_b, bytes_per_param=4)
            train_adamw_mp = _estimate_max_model_params(free_b, bytes_per_param=16)  # fp16 params/grad + fp32 master + Adam states
            print(f"  Inference fp16  (≈2 B/param): {inf16:.2f} B params")
            print(f"  Inference fp32  (≈4 B/param): {inf32:.2f} B params")
            print(f"  Train AdamW mp (≈16 B/param): {train_adamw_mp:.2f} B params")
        
    else:
        print("🔸 CUDA: Not available")
        if system == "Linux":
            driver_available, _ = check_nvidia_driver()
            if not driver_available:
                print("   💡 Install NVIDIA drivers and CUDA toolkit for GPU acceleration")
            else:
                print("   💡 NVIDIA driver found but PyTorch CUDA not available")
                print("      Install PyTorch with CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    # 3. Test MPS (Mac with Apple Silicon)
    if torch.backends.mps.is_available():
        print("🔸 MPS: Available (Apple Silicon GPU)")
        devices_tested.append(("mps", "Apple Silicon GPU (MPS)"))
        recommended_device = "mps"
        
        # Additional Mac-specific checks
        if system == "Darwin":
            compatible, macos_version = check_macos_version()
            if not compatible:
                print(f"   ⚠️  macOS {macos_version} detected. MPS requires macOS 12.3+")
        # MPS (unified) memory report
        sys_avail = _system_memory_available()
        if sys_avail is not None:
            print("\n🧠 Unified Memory (approx, available RAM):")
            print("-" * 40)
            print(f"  Available RAM: {_format_bytes(sys_avail)}")
            # Rough limits (shared with OS, activations); use conservative fraction
            print("\n📐 Estimated max model size (billions of parameters, conservative):")
            print("-" * 75)
            inf16 = _estimate_max_model_params(sys_avail, bytes_per_param=2, safety_frac=0.5)
            inf32 = _estimate_max_model_params(sys_avail, bytes_per_param=4, safety_frac=0.5)
            train_adamw_mp = _estimate_max_model_params(sys_avail, bytes_per_param=16, safety_frac=0.4)
            print(f"  Inference fp16  (≈2 B/param): {inf16:.2f} B params")
            print(f"  Inference fp32  (≈4 B/param): {inf32:.2f} B params")
            print(f"  Train AdamW mp (≈16 B/param): {train_adamw_mp:.2f} B params")
                
    else:
        if system == "Darwin":
            print("🔸 MPS: Not available")
            compatible, macos_version = check_macos_version()
            
            if machine == "arm64":  # Apple Silicon
                if not compatible:
                    print(f"   💡 Upgrade to macOS 12.3+ for GPU acceleration (current: {macos_version})")
                else:
                    print("   💡 Install PyTorch with MPS support: pip install torch torchvision torchaudio")
            else:  # Intel Mac
                print("   💡 Intel Macs have limited GPU acceleration support")
        else:
            print("🔸 MPS: Not available (not macOS)")
        # CPU memory report (approx)
        sys_avail = _system_memory_available()
        if sys_avail is not None:
            print("\n🧠 System Memory (approx, available RAM):")
            print("-" * 40)
            print(f"  Available RAM: {_format_bytes(sys_avail)}")
            print("\n📐 Estimated max model size on CPU (billions of parameters, conservative):")
            print("-" * 80)
            inf16 = _estimate_max_model_params(sys_avail, bytes_per_param=2, safety_frac=0.5)
            inf32 = _estimate_max_model_params(sys_avail, bytes_per_param=4, safety_frac=0.5)
            train_adamw_mp = _estimate_max_model_params(sys_avail, bytes_per_param=16, safety_frac=0.4)
            print(f"  Inference fp16  (≈2 B/param): {inf16:.2f} B params")
            print(f"  Inference fp32  (≈4 B/param): {inf32:.2f} B params")
            print(f"  Train AdamW mp (≈16 B/param): {train_adamw_mp:.2f} B params")
    
    # Test actual device functionality
    print(f"\n🧪 Testing device functionality:")
    print("-" * 35)
    
    success_count = 0
    
    for device_str, device_name in devices_tested:
        try:
            device = torch.device(device_str)
            
            # Create test tensors
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            
            # Perform operations
            z = torch.mm(x, y)
            
            # Test data movement
            z_cpu = z.cpu()
            z_back = z_cpu.to(device)
            
            print(f"✅ {device_name}: Functional")
            print(f"   Device: {z.device}")
            print(f"   Shape: {z.shape}")
            
            success_count += 1
            
            # Run performance test for GPU devices
            if device_str in ['cuda', 'mps']:
                test_device_performance(device, device_name)
                
        except Exception as e:
            print(f"❌ {device_name}: Failed - {e}")
    
    # Provide recommendations
    print(f"\n🎯 Recommendations:")
    print("-" * 20)
    print(f"Recommended device: {recommended_device}")
    
    # Code examples
    print(f"\n📝 Usage in your code:")
    print("-" * 25)
    
    if recommended_device == "cuda":
        print("# CUDA GPU")
        print("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        print("model = model.to(device)")
        print("data = data.to(device)")
        print("\n# Check GPU memory:")
        print("print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')")
        
    elif recommended_device == "mps":
        print("# Apple Silicon GPU")
        print("device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')")
        print("model = model.to(device)")
        print("data = data.to(device)")
        print("\n# Note: Some operations may fall back to CPU on MPS")
        
    else:
        print("# CPU (fallback)")
        print("device = torch.device('cpu')")
        print("model = model.to(device)")
        print("# Consider using torch.set_num_threads() to optimize CPU performance")
    
    # Universal device selection
    print(f"\n🔄 Universal device selection:")
    print("-" * 35)
    print("""# Automatic device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps') 
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device('cpu')
    print("Using CPU")

model = model.to(device)
data = data.to(device)""")
    
    # Summary
    print(f"\n📊 Summary:")
    print("=" * 15)
    print(f"System: {system} ({machine})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Devices tested: {len(devices_tested)}")
    print(f"Functional devices: {success_count}")
    print(f"Recommended: {recommended_device.upper()}")
    
    if success_count == len(devices_tested) and recommended_device != "cpu":
        print("\n🎉 GPU acceleration is available and working!")
        return 0
    elif success_count > 0:
        print("\n✅ Basic functionality working. GPU acceleration may be limited.")
        return 0
    else:
        print("\n⚠️  Some issues detected. Check installation.")
        return 1

if __name__ == "__main__":
    sys.exit(test_gpu_setup())
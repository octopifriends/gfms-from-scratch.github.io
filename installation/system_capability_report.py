#!/usr/bin/env python3
"""
System Capability Report

A lightweight helper script to summarize compute capabilities for the GeoAI course:
- OS and Python
- CPU model, logical/physical cores
- Memory
- GPU details (CUDA/MPS/ROCm), drivers, and PyTorch acceleration flags

Usage:
  python installation/system_capability_report.py
  python installation/system_capability_report.py --json           # machine-readable JSON
  python installation/system_capability_report.py --require-gpu    # exit 1 if no GPU acceleration
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional


def safe_run(cmd: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def get_os_info() -> Dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def get_cpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "model": None,
        "max_frequency_mhz": None,
    }

    system = platform.system()

    # Try psutil for extra details
    try:
        import psutil  # type: ignore

        info["physical_cores"] = psutil.cpu_count(logical=False)
        freq = getattr(psutil, "cpu_freq", None)
        if callable(freq):
            f = freq()
            if f and f.max:
                info["max_frequency_mhz"] = int(f.max)
    except Exception:
        pass

    # OS-specific model/cores
    if system == "Darwin":
        model = safe_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or None
        phys = safe_run(["sysctl", "-n", "hw.physicalcpu"]) or None
        if phys and phys.isdigit():
            info["physical_cores"] = int(phys)
        info["model"] = model
    elif system == "Linux":
        if shutil.which("lscpu"):
            out = safe_run(["lscpu"]) or ""
            model_match = re.search(r"Model name:\s*(.+)", out)
            cores_phys_match = re.search(r"Core\(s\) per socket:\s*(\d+)", out)
            sockets_match = re.search(r"Socket\(s\):\s*(\d+)", out)
            if model_match:
                info["model"] = model_match.group(1).strip()
            if cores_phys_match and sockets_match:
                try:
                    info["physical_cores"] = int(cores_phys_match.group(1)) * int(sockets_match.group(1))
                except Exception:
                    pass
        else:
            # Fallback parse /proc/cpuinfo
            cpuinfo = safe_run(["bash", "-lc", "cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2"]) or ""
            info["model"] = cpuinfo.strip() or None
    else:
        # Windows or other
        info["model"] = platform.processor() or None

    return info


def _mem_gb(value_bytes: Optional[int]) -> Optional[float]:
    if value_bytes is None:
        return None
    return round(value_bytes / 1e9, 2)


def get_memory_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"total_gb": None, "available_gb": None}
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["total_gb"] = _mem_gb(getattr(vm, "total", None))
        info["available_gb"] = _mem_gb(getattr(vm, "available", None))
        return info
    except Exception:
        pass

    system = platform.system()
    if system == "Darwin":
        total = safe_run(["sysctl", "-n", "hw.memsize"])
        total_int = int(total) if total and total.isdigit() else None
        info["total_gb"] = _mem_gb(total_int)
        # Available memory is trickier without psutil; omit
    elif system == "Linux":
        meminfo = safe_run(["bash", "-lc", "cat /proc/meminfo"]) or ""
        total_kb = None
        avail_kb = None
        mt = re.search(r"MemTotal:\s*(\d+) kB", meminfo)
        ma = re.search(r"MemAvailable:\s*(\d+) kB", meminfo)
        if mt:
            total_kb = int(mt.group(1))
        if ma:
            avail_kb = int(ma.group(1))
        info["total_gb"] = round((total_kb or 0) * 1024 / 1e9, 2) if total_kb else None
        info["available_gb"] = round((avail_kb or 0) * 1024 / 1e9, 2) if avail_kb else None
    return info


def parse_nvcc_version(text: str) -> Optional[str]:
    # Typical: Cuda compilation tools, release 12.4, V12.4.131
    m = re.search(r"release\s+([0-9.]+)", text)
    return m.group(1) if m else None


def get_gpu_info() -> Dict[str, Any]:
    gpu: Dict[str, Any] = {
        "torch_present": False,
        "acceleration_available": False,
        "cuda": {"available": False, "device_count": 0, "devices": [], "torch_cuda_version": None, "cudnn_enabled": None, "cudnn_version": None},
        "mps": {"available": False},
        "rocm": {"available": False, "version": None},
        "nvidia_smi": {"present": False, "driver_version": None, "gpus": []},
        "nvcc": {"present": False, "version": None},
    }

    # Torch-level detection
    try:
        import torch  # type: ignore

        gpu["torch_present"] = True

        # CUDA (NVIDIA)
        try:
            cuda_avail = torch.cuda.is_available()
        except Exception:
            cuda_avail = False
        if cuda_avail:
            count = torch.cuda.device_count()
            devices: List[Dict[str, Any]] = []
            for idx in range(count):
                props = torch.cuda.get_device_properties(idx)
                capability = (
                    f"{getattr(props, 'major', None)}.{getattr(props, 'minor', None)}"
                    if hasattr(props, "major") and hasattr(props, "minor")
                    else None
                )
                devices.append(
                    {
                        "index": idx,
                        "name": getattr(props, "name", None),
                        "total_memory_gb": round(getattr(props, "total_memory", 0) / 1e9, 2),
                        "capability": capability,
                    }
                )
            gpu["cuda"].update(
                {
                    "available": True,
                    "device_count": count,
                    "devices": devices,
                    "torch_cuda_version": getattr(torch.version, "cuda", None),
                    "cudnn_enabled": torch.backends.cudnn.is_available(),
                    "cudnn_version": torch.backends.cudnn.version(),
                }
            )

        # MPS (Apple Silicon)
        try:
            mps_avail = torch.backends.mps.is_available()
        except Exception:
            mps_avail = False
        if mps_avail:
            gpu["mps"]["available"] = True

        # ROCm (AMD)
        hip_version = getattr(torch.version, "hip", None)
        if hip_version:
            gpu["rocm"]["available"] = True
            gpu["rocm"]["version"] = hip_version

        gpu["acceleration_available"] = bool(
            gpu["cuda"]["available"] or gpu["mps"]["available"] or gpu["rocm"]["available"]
        )

    except Exception:
        pass

    # nvidia-smi (if present)
    if shutil.which("nvidia-smi"):
        out = safe_run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]) or ""
        gpus = []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            entry = {"name": parts[0] if len(parts) > 0 else None}
            if len(parts) > 1:
                entry["driver_version"] = parts[1]
            if len(parts) > 2:
                # e.g., "40960 MiB"
                mem_match = re.search(r"(\d+)\s*MiB", parts[2])
                entry["total_memory_gb"] = round((int(mem_match.group(1)) / 1024), 2) if mem_match else None
            gpus.append(entry)
        gpu["nvidia_smi"]["present"] = True
        if gpus:
            gpu["nvidia_smi"]["driver_version"] = gpus[0].get("driver_version")
        gpu["nvidia_smi"]["gpus"] = gpus

    # nvcc (CUDA toolkit)
    if shutil.which("nvcc"):
        nvcc_out = safe_run(["nvcc", "--version"]) or ""
        gpu["nvcc"]["present"] = True
        gpu["nvcc"]["version"] = parse_nvcc_version(nvcc_out)

    return gpu


def build_report() -> Dict[str, Any]:
    os_info = get_os_info()
    cpu_info = get_cpu_info()
    mem_info = get_memory_info()
    gpu_info = get_gpu_info()

    # Guidance
    guidance: List[str] = []
    system = os_info.get("system")
    machine = os_info.get("machine")

    has_gpu = gpu_info.get("acceleration_available", False)
    torch_present = gpu_info.get("torch_present", False)
    cuda_avail = bool(gpu_info.get("cuda", {}).get("available"))
    mps_avail = bool(gpu_info.get("mps", {}).get("available"))

    if not has_gpu:
        if system == "Darwin" and machine.lower() in ("arm64", "aarch64"):
            guidance.append("Apple Silicon detected but no MPS. Ensure PyTorch with MPS support is installed.")
        elif system == "Linux":
            if gpu_info.get("nvidia_smi", {}).get("present"):
                guidance.append("NVIDIA driver present, but PyTorch CUDA not available. Install torch with CUDA support.")
            else:
                guidance.append("No GPU acceleration detected. For NVIDIA GPUs, install drivers and CUDA-enabled PyTorch.")
        else:
            guidance.append("No GPU acceleration detected; CPU-only.")
    else:
        if cuda_avail:
            nvcc = gpu_info.get("nvcc", {}).get("version")
            torch_cuda = gpu_info.get("cuda", {}).get("torch_cuda_version")
            if torch_cuda and nvcc and (torch_cuda.split(".")[0] != nvcc.split(".")[0]):
                guidance.append("PyTorch CUDA version and nvcc major versions differ; ensure compatibility.")
        if mps_avail:
            guidance.append("MPS available. Some ops may fall back to CPU in PyTorch on macOS.")
    if not torch_present:
        guidance.append("PyTorch not found; install torch to use GPU acceleration checks.")

    return {
        "os": os_info,
        "cpu": cpu_info,
        "memory": mem_info,
        "gpu": gpu_info,
        "guidance": guidance,
    }


def print_human(report: Dict[str, Any]) -> None:
    os_info = report["os"]
    cpu = report["cpu"]
    mem = report["memory"]
    gpu = report["gpu"]

    print("🖥️  System")
    print("-" * 40)
    print(f"OS: {os_info['system']} {os_info['release']} ({os_info['machine']})")
    print(f"Version: {os_info['version']}")
    print(f"Python: {os_info['python']}")

    print("\n🧠 CPU")
    print("-" * 40)
    print(f"Model: {cpu.get('model')}")
    print(f"Physical cores: {cpu.get('physical_cores')}")
    print(f"Logical cores:  {cpu.get('logical_cores')}")
    if cpu.get("max_frequency_mhz"):
        print(f"Max frequency: {cpu['max_frequency_mhz']} MHz")

    print("\n💾 Memory")
    print("-" * 40)
    print(f"Total:     {mem.get('total_gb')} GB")
    if mem.get("available_gb") is not None:
        print(f"Available: {mem.get('available_gb')} GB")

    print("\n🎮 GPU / Acceleration")
    print("-" * 40)
    acc = "Yes" if gpu.get("acceleration_available") else "No"
    print(f"Acceleration available: {acc}")

    # CUDA
    cuda = gpu.get("cuda", {})
    if cuda.get("available"):
        print("CUDA: Available")
        print(f"  PyTorch CUDA: {cuda.get('torch_cuda_version')}")
        if cuda.get("cudnn_enabled"):
            print(f"  cuDNN: Yes (v{cuda.get('cudnn_version')})")
        else:
            print("  cuDNN: Not available")
        for dev in cuda.get("devices", []) or []:
            name = dev.get("name")
            mem_gb = dev.get("total_memory_gb")
            idx = dev.get("index")
            print(f"  GPU {idx}: {name} ({mem_gb} GB)")
    else:
        print("CUDA: Not available")

    # MPS
    mps = gpu.get("mps", {})
    print(f"MPS (Apple Silicon): {'Available' if mps.get('available') else 'Not available'}")

    # ROCm
    rocm = gpu.get("rocm", {})
    if rocm.get("available"):
        print(f"ROCm: Available (HIP {rocm.get('version')})")
    else:
        print("ROCm: Not available")

    # Drivers and toolkits
    if gpu.get("nvidia_smi", {}).get("present"):
        print("\nNVIDIA Driver and GPUs (nvidia-smi):")
        print(f"  Driver: {gpu['nvidia_smi'].get('driver_version')}")
        for g in gpu["nvidia_smi"].get("gpus", []) or []:
            print(f"  - {g.get('name')} ({g.get('total_memory_gb')} GB)")
    if gpu.get("nvcc", {}).get("present"):
        print(f"CUDA Toolkit (nvcc): {gpu['nvcc'].get('version')}")

    # Guidance
    if report.get("guidance"):
        print("\n💡 Notes / Guidance")
        print("-" * 40)
        for g in report["guidance"]:
            print(f"- {g}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report system compute capabilities for GeoAI")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    parser.add_argument("--require-gpu", action="store_true", help="Exit with code 1 if no GPU acceleration available")
    args = parser.parse_args()

    report = build_report()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human(report)

    if args.require_gpu and not report.get("gpu", {}).get("acceleration_available"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())



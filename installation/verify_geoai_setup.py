#!/usr/bin/env python3
"""
GeoAI Environment Verification Script (Unified)
Combines previous verify_geoai_setup.py and scripts/validate_environment.py
"""

import sys
import os
import subprocess
import importlib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")


def print_success(text: str) -> None:
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text: str) -> None:
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_warning(text: str) -> None:
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_info(text: str) -> None:
    print(f"ℹ️  {text}")


class GeoAIEnvironmentVerifier:
    def __init__(self) -> None:
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'environment': os.environ.get('CONDA_DEFAULT_ENV', 'unknown'),
            'tests': {}
        }
        self.models_dir = Path.home() / 'geoAI' / 'models'

    def _test_import(self, package_name: str, display_name: str | None = None) -> bool:
        if display_name is None:
            display_name = package_name
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{display_name}: {version}")
            return True
        except ImportError as e:
            print_error(f"{display_name}: Failed to import - {e}")
            return False

    def check_python_version(self) -> bool:
        print_info("Checking Python version...")
        version = sys.version_info
        if version.major == 3 and version.minor >= 11:
            print_success(f"Python {version.major}.{version.minor}.{version.micro}")
            self.results['tests']['python_version'] = {'status': 'pass', 'version': f"{version.major}.{version.minor}.{version.micro}"}
            return True
        print_error(f"Python {version.major}.{version.minor}.{version.micro} - Need Python 3.11+")
        self.results['tests']['python_version'] = {'status': 'fail', 'version': f"{version.major}.{version.minor}.{version.micro}"}
        return False

    def check_conda_environment(self) -> bool:
        print_info("Checking conda environment...")
        env_name = os.environ.get('CONDA_DEFAULT_ENV')
        if env_name == 'geoAI':
            print_success(f"Running in {env_name} environment")
            self.results['tests']['conda_env'] = {'status': 'pass', 'environment': env_name}
            return True
        print_error(f"Expected 'geoAI' environment, found: {env_name}")
        self.results['tests']['conda_env'] = {'status': 'fail', 'environment': env_name}
        return False

    def check_core_packages(self) -> bool:
        print_info("Checking core scientific packages...")
        core_packages = [
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('matplotlib', 'Matplotlib'),
            ('seaborn', 'Seaborn'),
            ('scipy', 'SciPy')
        ]
        all_ok = True
        ok_count = 0
        for pkg, name in core_packages:
            ok = self._test_import(pkg, name)
            ok_count += 1 if ok else 0
            all_ok = all_ok and ok
        self.results['tests']['core_packages'] = {'status': 'pass' if all_ok else 'partial', 'ok': ok_count, 'total': len(core_packages)}
        return all_ok

    def check_ml_packages(self) -> bool:
        print_info("Checking AI/ML packages...")
        ml_packages = [
            ('torch', 'PyTorch'),
            ('torchvision', 'TorchVision'),
            ('sklearn', 'Scikit-learn'),
            ('transformers', 'HuggingFace Transformers'),
            ('datasets', 'HuggingFace Datasets'),
            ('pytorch_lightning', 'PyTorch Lightning'),
            ('timm', 'TIMM')
        ]
        all_ok = True
        ok_count = 0
        for pkg, name in ml_packages:
            ok = self._test_import(pkg, name)
            ok_count += 1 if ok else 0
            all_ok = all_ok and ok
        self.results['tests']['ml_packages'] = {'status': 'pass' if all_ok else 'partial', 'ok': ok_count, 'total': len(ml_packages)}
        return all_ok

    def check_geospatial_packages(self) -> bool:
        print_info("Checking geospatial packages...")
        geo_packages = [
            ('rasterio', 'Rasterio'),
            ('geopandas', 'GeoPandas'),
            ('xarray', 'XArray'),
            ('folium', 'Folium'),
            ('torchgeo', 'TorchGeo'),
            ('ee', 'Earth Engine API')
        ]
        all_ok = True
        ok_count = 0
        for pkg, name in geo_packages:
            ok = self._test_import(pkg, name)
            ok_count += 1 if ok else 0
            all_ok = all_ok and ok
        self.results['tests']['geospatial_packages'] = {'status': 'pass' if all_ok else 'partial', 'ok': ok_count, 'total': len(geo_packages)}
        return all_ok

    def check_cv_packages(self) -> bool:
        print_info("Checking computer vision packages...")
        cv_packages = [
            ('cv2', 'OpenCV'),
            ('kornia', 'Kornia'),
            ('einops', 'Einops'),
            ('PIL', 'Pillow')
        ]
        all_ok = True
        ok_count = 0
        for pkg, name in cv_packages:
            ok = self._test_import(pkg, name)
            ok_count += 1 if ok else 0
            all_ok = all_ok and ok
        self.results['tests']['cv_packages'] = {'status': 'pass' if all_ok else 'partial', 'ok': ok_count, 'total': len(cv_packages)}
        return all_ok

    def check_jupyter_packages(self) -> bool:
        print_info("Checking Jupyter packages...")
        jupyter_packages = [
            ('jupyter', 'Jupyter'),
            ('jupyterlab', 'JupyterLab'),
            ('ipykernel', 'IPython Kernel'),
            ('notebook', 'Notebook')
        ]
        all_ok = True
        ok_count = 0
        for pkg, name in jupyter_packages:
            ok = self._test_import(pkg, name)
            ok_count += 1 if ok else 0
            all_ok = all_ok and ok
        self.results['tests']['jupyter_packages'] = {'status': 'pass' if all_ok else 'partial', 'ok': ok_count, 'total': len(jupyter_packages)}
        return all_ok

    def test_gpu_support(self) -> bool:
        print_info("Testing GPU support...")
        try:
            import torch
            import platform
            system = platform.system()
            print_success(f"PyTorch: {torch.__version__}")
            gpu_available = False

            if torch.cuda.is_available():
                cuda_count = torch.cuda.device_count()
                cuda_name = torch.cuda.get_device_name(0)
                print_success(f"CUDA available: {cuda_count} device(s)")
                print_success(f"Primary GPU: {cuda_name}")
                # sanity op
                device = torch.device('cuda')
                x = torch.randn(100, 100, device=device)
                y = torch.randn(100, 100, device=device)
                z = torch.mm(x, y)
                print_success(f"CUDA tensor operations OK on {z.device}")
                gpu_available = True
                self.results['tests']['pytorch_gpu'] = {'status': 'pass', 'type': 'cuda', 'gpu_name': cuda_name, 'count': cuda_count}
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print_success("MPS (Apple Metal) available")
                device = torch.device('mps')
                x = torch.randn(100, 100, device=device)
                y = torch.randn(100, 100, device=device)
                z = torch.mm(x, y)
                print_success(f"MPS tensor operations OK on {z.device}")
                gpu_available = True
                self.results['tests']['pytorch_gpu'] = {'status': 'pass', 'type': 'mps'}
            else:
                print_warning("No GPU acceleration available (CUDA or MPS)")
                if system == 'Linux':
                    print_info("Hint: Install NVIDIA drivers and use CUDA-enabled PyTorch builds.")
                self.results['tests']['pytorch_gpu'] = {'status': 'fail', 'error': 'no_gpu_available'}

            # Recommended usage snippet
            print("\nRecommended usage:")
            if gpu_available:
                print("  device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))")
            else:
                print("  device = torch.device('cpu')")
            print("  model = model.to(device)")
            print("  data = data.to(device)")

            return gpu_available
        except Exception as e:
            print_error(f"GPU test failed: {e}")
            self.results['tests']['pytorch_gpu'] = {'status': 'fail', 'error': str(e)}
            return False

    def check_jupyter_kernel(self) -> bool:
        print_info("Checking Jupyter kernel registration...")
        try:
            result = subprocess.run(['jupyter', 'kernelspec', 'list'], capture_output=True, text=True)
            if 'geoai' in result.stdout.lower():
                print_success("geoAI Jupyter kernel installed")
                self.results['tests']['jupyter_kernel'] = {'status': 'pass'}
                return True
            print_error("geoAI Jupyter kernel not found")
            print_info("Run: python -m ipykernel install --user --name geoai --display-name 'GeoAI'")
            self.results['tests']['jupyter_kernel'] = {'status': 'fail'}
            return False
        except Exception as e:
            print_error(f"Jupyter kernel check failed: {e}")
            self.results['tests']['jupyter_kernel'] = {'status': 'fail', 'error': str(e)}
            return False

    def performance_benchmarks(self) -> bool:
        print_info("Running performance benchmarks...")
        try:
            import torch
            import numpy as np
            import time
            benchmarks = {}
            # CPU matmul
            start = time.time()
            a = np.random.randn(1000, 1000)
            b = np.random.randn(1000, 1000)
            _ = np.dot(a, b)
            cpu_time = time.time() - start
            print_success(f"CPU 1000x1000 matmul: {cpu_time:.3f}s")
            benchmarks['cpu_matmul_s'] = cpu_time
            # GPU matmul if available
            if torch.cuda.is_available():
                start = time.time()
                a_gpu = torch.randn(1000, 1000, device='cuda')
                b_gpu = torch.randn(1000, 1000, device='cuda')
                _ = torch.mm(a_gpu, b_gpu)
                torch.cuda.synchronize()
                gpu_time = time.time() - start
                print_success(f"GPU 1000x1000 matmul: {gpu_time:.3f}s (speedup {cpu_time/gpu_time:.1f}x)")
                benchmarks['gpu_matmul_s'] = gpu_time
                benchmarks['speedup'] = cpu_time / gpu_time
            self.results['tests']['performance'] = {'status': 'pass', **benchmarks}
            return True
        except Exception as e:
            print_error(f"Benchmark failed: {e}")
            self.results['tests']['performance'] = {'status': 'fail', 'error': str(e)}
            return False

    def check_earth_engine(self) -> bool:
        print_info("Checking Earth Engine authentication...")
        try:
            import ee
            ee.Initialize()
            _ = ee.Image("LANDSAT/LC08/C02/T1_TOA/LC08_123032_20140515").getInfo()
            print_success("Earth Engine: Authenticated and working")
            self.results['tests']['earth_engine'] = {'status': 'pass'}
            return True
        except Exception as e:
            err = str(e)
            print_error(f"Earth Engine: {err}")
            self.results['tests']['earth_engine'] = {'status': 'fail', 'error': err}
            return False

    def check_huggingface_access(self) -> bool:
        print_info("Checking HuggingFace Hub access...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            user_info = api.whoami()
            print_success(f"HuggingFace: Authenticated as {user_info.get('name')}")
            self.results['tests']['huggingface'] = {'status': 'pass', 'user': user_info.get('name')}
            return True
        except Exception as e:
            print_error(f"HuggingFace: {e}")
            self.results['tests']['huggingface'] = {'status': 'fail', 'error': str(e)}
            return False

    def check_foundation_models(self) -> bool:
        print_info("Checking foundation model registry...")
        ok = False
        registry_path = self.models_dir / 'model_registry.json'
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                have_any = 0
                for model_name in ['prithvi', 'satmae', 'clip_base']:
                    if model_name in registry.get('models', {}):
                        model_path = Path(registry['models'][model_name]['path'])
                        if model_path.exists():
                            print_success(f"{model_name}: Available")
                            have_any += 1
                        else:
                            print_warning(f"{model_name}: Path not found")
                ok = have_any > 0
            except Exception as e:
                print_error(f"Model registry read error: {e}")
        else:
            print_warning("Model registry not found (run installation/scripts/install_foundation_models.sh)")
        self.results['tests']['foundation_models'] = {'status': 'pass' if ok else 'fail'}
        return ok

    def test_model_loading(self) -> bool:
        print_info("Testing HF model loading (distilbert-base-uncased)...")
        try:
            import torch
            from transformers import AutoModel
            model = AutoModel.from_pretrained("distilbert-base-uncased")
            if torch.cuda.is_available():
                model = model.to('cuda')
                print_success("Loaded on GPU")
            else:
                print_success("Loaded on CPU")
            self.results['tests']['model_loading'] = {'status': 'pass', 'model': 'distilbert-base-uncased'}
            return True
        except Exception as e:
            print_error(f"Model loading failed: {e}")
            self.results['tests']['model_loading'] = {'status': 'fail', 'error': str(e)}
            return False

    def save_report(self) -> None:
        report_path = Path.home() / 'geoAI' / 'validation_report.json'
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print_info(f"Report saved to: {report_path}")

    def run_all(self) -> bool:
        print_header("GeoAI Environment Verification")
        tests = [
            ("Python Version", self.check_python_version),
            ("Conda Environment", self.check_conda_environment),
            ("Core Packages", self.check_core_packages),
            ("AI/ML Packages", self.check_ml_packages),
            ("Geospatial Packages", self.check_geospatial_packages),
            ("CV Packages", self.check_cv_packages),
            ("Jupyter Packages", self.check_jupyter_packages),
            ("PyTorch & GPU", self.test_gpu_support),
            ("Jupyter Kernel", self.check_jupyter_kernel),
            ("HuggingFace Access", self.check_huggingface_access),
            ("Earth Engine", self.check_earth_engine),
            ("Foundation Models", self.check_foundation_models),
            ("Model Loading", self.test_model_loading),
            ("Performance", self.performance_benchmarks),
        ]
        for name, fn in tests:
            print_header(name)
            try:
                fn()
            except Exception as e:
                print_error(f"Unexpected error in {name}: {e}")
                self.results['tests'][name.lower().replace(' ', '_')] = {'status': 'error', 'error': str(e)}

        # Summary
        print_header("SUMMARY")
        passed = len([t for t in self.results['tests'].values() if t.get('status') == 'pass'])
        failed = len([t for t in self.results['tests'].values() if t.get('status') == 'fail'])
        partial = len([t for t in self.results['tests'].values() if t.get('status') == 'partial'])
        total = len(self.results['tests'])
        print(f"Total tests: {total}\nPassed: {passed}\nPartial: {partial}\nFailed: {failed}")
        self.save_report()
        # Success if no hard failures in critical areas
        critical_ok = self.results['tests'].get('pytorch_gpu', {}).get('status') in {'pass', 'partial'} and \
                      self.results['tests'].get('jupyter_kernel', {}).get('status') in {'pass'}
        return (passed + partial) >= total * 0.8 and critical_ok


def main() -> int:
    verifier = GeoAIEnvironmentVerifier()
    ok = verifier.run_all()
    if ok:
        print_success("Environment ready for GeoAI!")
        return 0
    print_warning("Environment has issues. See report for details.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
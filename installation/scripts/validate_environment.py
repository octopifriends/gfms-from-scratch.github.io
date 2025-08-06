#!/usr/bin/env python3
"""
Environment Validation Script for GEOG 288KC
Comprehensive testing of the geoAI-gpu environment and foundation models
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

# Colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

class EnvironmentValidator:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'environment': os.environ.get('CONDA_DEFAULT_ENV', 'unknown'),
            'tests': {}
        }
        self.models_dir = Path.home() / 'geoAI' / 'models'
        
    def check_python_version(self):
        """Check Python version"""
        print_info("Checking Python version...")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 11:
            print_success(f"Python {version.major}.{version.minor}.{version.micro}")
            self.results['tests']['python_version'] = {'status': 'pass', 'version': f"{version.major}.{version.minor}.{version.micro}"}
            return True
        else:
            print_error(f"Python {version.major}.{version.minor}.{version.micro} - Need Python 3.11+")
            self.results['tests']['python_version'] = {'status': 'fail', 'version': f"{version.major}.{version.minor}.{version.micro}"}
            return False

    def check_conda_environment(self):
        """Check if running in correct conda environment"""
        print_info("Checking conda environment...")
        
        env_name = os.environ.get('CONDA_DEFAULT_ENV')
        if env_name == 'geoAI-gpu':
            print_success(f"Running in {env_name} environment")
            self.results['tests']['conda_env'] = {'status': 'pass', 'environment': env_name}
            return True
        else:
            print_error(f"Expected 'geoAI-gpu' environment, found: {env_name}")
            self.results['tests']['conda_env'] = {'status': 'fail', 'environment': env_name}
            return False

    def check_core_packages(self):
        """Check installation of core packages"""
        print_info("Checking core scientific packages...")
        
        core_packages = [
            ('numpy', '1.24.0'),
            ('pandas', '2.0.0'), 
            ('matplotlib', '3.7.0'),
            ('scipy', '1.10.0'),
            ('scikit-learn', '1.3.0'),
        ]
        
        all_passed = True
        package_results = {}
        
        for package, min_version in core_packages:
            try:
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                print_success(f"{package}: {version}")
                package_results[package] = {'status': 'pass', 'version': version}
            except ImportError:
                print_error(f"{package}: Not installed")
                package_results[package] = {'status': 'fail', 'error': 'not_installed'}
                all_passed = False
                
        self.results['tests']['core_packages'] = package_results
        return all_passed

    def check_pytorch_gpu(self):
        """Check PyTorch and GPU availability"""
        print_info("Checking PyTorch and GPU setup...")
        
        try:
            import torch
            import torchvision
            
            # Check PyTorch version
            torch_version = torch.__version__
            print_success(f"PyTorch: {torch_version}")
            
            # Check CUDA availability
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                
                print_success(f"CUDA: {cuda_version}")
                print_success(f"GPUs available: {gpu_count}")
                print_success(f"Primary GPU: {gpu_name}")
                
                # Test GPU tensor operations
                try:
                    x = torch.randn(1000, 1000, device='cuda')
                    y = torch.mm(x, x.t())
                    print_success("GPU tensor operations: Working")
                    
                    self.results['tests']['pytorch_gpu'] = {
                        'status': 'pass',
                        'torch_version': torch_version,
                        'cuda_version': cuda_version,
                        'gpu_count': gpu_count,
                        'gpu_name': gpu_name
                    }
                    return True
                    
                except Exception as e:
                    print_error(f"GPU tensor operations failed: {e}")
                    self.results['tests']['pytorch_gpu'] = {
                        'status': 'partial',
                        'error': str(e),
                        'torch_version': torch_version
                    }
                    return False
                    
            else:
                print_error("CUDA not available")
                self.results['tests']['pytorch_gpu'] = {
                    'status': 'fail',
                    'error': 'cuda_not_available',
                    'torch_version': torch_version
                }
                return False
                
        except ImportError as e:
            print_error(f"PyTorch not installed: {e}")
            self.results['tests']['pytorch_gpu'] = {'status': 'fail', 'error': 'not_installed'}
            return False

    def check_geospatial_packages(self):
        """Check geospatial packages"""
        print_info("Checking geospatial packages...")
        
        geo_packages = [
            ('rasterio', '1.3.0'),
            ('geopandas', '0.13.0'),
            ('xarray', '2023.1.0'),
            ('earthengine-api', '0.1.350'),
            ('folium', '0.14.0'),
        ]
        
        all_passed = True
        package_results = {}
        
        for package, min_version in geo_packages:
            try:
                if package == 'earthengine-api':
                    import ee
                    module = ee
                else:
                    module = importlib.import_module(package.replace('-', '_'))
                
                version = getattr(module, '__version__', 'unknown')
                print_success(f"{package}: {version}")
                package_results[package] = {'status': 'pass', 'version': version}
            except ImportError:
                print_error(f"{package}: Not installed")
                package_results[package] = {'status': 'fail', 'error': 'not_installed'}
                all_passed = False
                
        self.results['tests']['geospatial_packages'] = package_results
        return all_passed

    def check_ml_packages(self):
        """Check machine learning packages"""
        print_info("Checking ML packages...")
        
        ml_packages = [
            'transformers',
            'datasets', 
            'huggingface_hub',
            'torchgeo',
            'pytorch_lightning',
            'accelerate',
        ]
        
        all_passed = True
        package_results = {}
        
        for package in ml_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                print_success(f"{package}: {version}")
                package_results[package] = {'status': 'pass', 'version': version}
            except ImportError:
                print_error(f"{package}: Not installed")
                package_results[package] = {'status': 'fail', 'error': 'not_installed'}
                all_passed = False
                
        self.results['tests']['ml_packages'] = package_results
        return all_passed

    def check_earth_engine(self):
        """Check Earth Engine authentication"""
        print_info("Checking Earth Engine authentication...")
        
        try:
            import ee
            
            # Try to initialize Earth Engine
            ee.Initialize()
            
            # Test with a simple operation
            image = ee.Image("LANDSAT/LC08/C02/T1_TOA/LC08_123032_20140515")
            info = image.getInfo()
            
            print_success("Earth Engine: Authenticated and working")
            self.results['tests']['earth_engine'] = {'status': 'pass'}
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "not authenticated" in error_msg.lower():
                print_error("Earth Engine: Not authenticated")
                print_info("Run: earthengine authenticate")
            else:
                print_error(f"Earth Engine error: {error_msg}")
                
            self.results['tests']['earth_engine'] = {'status': 'fail', 'error': error_msg}
            return False

    def check_foundation_models(self):
        """Check foundation model availability"""
        print_info("Checking foundation models...")
        
        model_results = {}
        
        # Check model registry
        registry_path = self.models_dir / 'model_registry.json'
        if registry_path.exists():
            print_success("Model registry found")
            
            with open(registry_path, 'r') as f:
                registry = json.load(f)
                
            models_to_check = ['prithvi', 'satmae', 'clip_base']
            
            for model_name in models_to_check:
                if model_name in registry['models']:
                    model_path = Path(registry['models'][model_name]['path'])
                    if model_path.exists():
                        print_success(f"{model_name}: Available")
                        model_results[model_name] = {'status': 'pass', 'path': str(model_path)}
                    else:
                        print_error(f"{model_name}: Path not found")
                        model_results[model_name] = {'status': 'fail', 'error': 'path_not_found'}
                else:
                    print_error(f"{model_name}: Not in registry")
                    model_results[model_name] = {'status': 'fail', 'error': 'not_in_registry'}
                    
        else:
            print_error("Model registry not found")
            print_info("Run: bash installation/scripts/install_foundation_models.sh")
            model_results['registry'] = {'status': 'fail', 'error': 'registry_not_found'}
            
        self.results['tests']['foundation_models'] = model_results
        return len([r for r in model_results.values() if r['status'] == 'pass']) > 0

    def check_huggingface_access(self):
        """Check HuggingFace Hub access"""
        print_info("Checking HuggingFace Hub access...")
        
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            user_info = api.whoami()
            
            print_success(f"HuggingFace Hub: Authenticated as {user_info['name']}")
            self.results['tests']['huggingface'] = {'status': 'pass', 'user': user_info['name']}
            return True
            
        except Exception as e:
            print_error(f"HuggingFace Hub: {e}")
            print_info("Run: huggingface-cli login")
            self.results['tests']['huggingface'] = {'status': 'fail', 'error': str(e)}
            return False

    def test_model_loading(self):
        """Test loading a foundation model"""
        print_info("Testing model loading...")
        
        try:
            import torch
            from transformers import AutoModel
            
            # Try to load a small model for testing
            model_name = "microsoft/DinoV2-small"  # Small model for testing
            
            print_info(f"Loading {model_name} for testing...")
            model = AutoModel.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.to('cuda')
                print_success("Model loaded successfully on GPU")
            else:
                print_success("Model loaded successfully on CPU")
                
            self.results['tests']['model_loading'] = {'status': 'pass', 'test_model': model_name}
            return True
            
        except Exception as e:
            print_error(f"Model loading failed: {e}")
            self.results['tests']['model_loading'] = {'status': 'fail', 'error': str(e)}
            return False

    def check_jupyter_kernel(self):
        """Check Jupyter kernel installation"""
        print_info("Checking Jupyter kernel...")
        
        try:
            result = subprocess.run(['jupyter', 'kernelspec', 'list'], 
                                  capture_output=True, text=True, check=True)
            
            if 'geoai' in result.stdout.lower():
                print_success("geoAI Jupyter kernel installed")
                self.results['tests']['jupyter_kernel'] = {'status': 'pass'}
                return True
            else:
                print_error("geoAI Jupyter kernel not found")
                print_info("Run: python -m ipykernel install --user --name geoAI-gpu")
                self.results['tests']['jupyter_kernel'] = {'status': 'fail', 'error': 'kernel_not_found'}
                return False
                
        except subprocess.CalledProcessError as e:
            print_error(f"Error checking Jupyter kernels: {e}")
            self.results['tests']['jupyter_kernel'] = {'status': 'fail', 'error': str(e)}
            return False

    def performance_benchmarks(self):
        """Run basic performance benchmarks"""
        print_info("Running performance benchmarks...")
        
        try:
            import torch
            import numpy as np
            import time
            
            benchmarks = {}
            
            # CPU benchmark
            print_info("CPU matrix multiplication (1000x1000)...")
            start_time = time.time()
            a = np.random.randn(1000, 1000)
            b = np.random.randn(1000, 1000) 
            c = np.dot(a, b)
            cpu_time = time.time() - start_time
            print_success(f"CPU time: {cpu_time:.3f} seconds")
            benchmarks['cpu_matmul'] = cpu_time
            
            # GPU benchmark (if available)
            if torch.cuda.is_available():
                print_info("GPU matrix multiplication (1000x1000)...")
                start_time = time.time()
                a_gpu = torch.randn(1000, 1000, device='cuda')
                b_gpu = torch.randn(1000, 1000, device='cuda')
                c_gpu = torch.mm(a_gpu, b_gpu)
                torch.cuda.synchronize()  # Wait for GPU to finish
                gpu_time = time.time() - start_time
                print_success(f"GPU time: {gpu_time:.3f} seconds")
                print_success(f"Speedup: {cpu_time/gpu_time:.1f}x")
                benchmarks['gpu_matmul'] = gpu_time
                benchmarks['speedup'] = cpu_time / gpu_time
                
            self.results['tests']['performance'] = {'status': 'pass', 'benchmarks': benchmarks}
            return True
            
        except Exception as e:
            print_error(f"Benchmark failed: {e}")
            self.results['tests']['performance'] = {'status': 'fail', 'error': str(e)}
            return False

    def generate_report(self):
        """Generate validation report"""
        print_header("VALIDATION SUMMARY")
        
        total_tests = len(self.results['tests'])
        passed_tests = len([t for t in self.results['tests'].values() if t['status'] == 'pass'])
        failed_tests = len([t for t in self.results['tests'].values() if t['status'] == 'fail'])
        partial_tests = len([t for t in self.results['tests'].values() if t['status'] == 'partial'])
        
        print(f"\n{Colors.BOLD}Total Tests: {total_tests}{Colors.END}")
        print(f"{Colors.GREEN}✅ Passed: {passed_tests}{Colors.END}")
        print(f"{Colors.RED}❌ Failed: {failed_tests}{Colors.END}")
        if partial_tests > 0:
            print(f"{Colors.YELLOW}⚠️  Partial: {partial_tests}{Colors.END}")
            
        success_rate = (passed_tests + partial_tests) / total_tests * 100
        print(f"\n{Colors.BOLD}Success Rate: {success_rate:.1f}%{Colors.END}")
        
        # Recommendations
        print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")
        
        failed_categories = [name for name, result in self.results['tests'].items() 
                           if result['status'] == 'fail']
        
        if failed_categories:
            if 'pytorch_gpu' in failed_categories:
                print_warning("GPU acceleration not available - check CUDA installation")
            if 'earth_engine' in failed_categories:
                print_warning("Earth Engine not authenticated - run 'earthengine authenticate'")
            if 'foundation_models' in failed_categories:
                print_warning("Foundation models missing - run installation script")
            if 'huggingface' in failed_categories:
                print_warning("HuggingFace not authenticated - run 'huggingface-cli login'")
        else:
            print_success("All critical systems operational!")
            
        # Save report
        report_path = Path.home() / 'geoAI' / 'validation_report.json'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\n{Colors.CYAN}Report saved to: {report_path}{Colors.END}")
        
        return success_rate >= 80.0

    def run_all_tests(self):
        """Run all validation tests"""
        print_header("GEOG 288KC ENVIRONMENT VALIDATION")
        
        tests = [
            ("Python Version", self.check_python_version),
            ("Conda Environment", self.check_conda_environment),
            ("Core Packages", self.check_core_packages),
            ("PyTorch & GPU", self.check_pytorch_gpu),
            ("Geospatial Packages", self.check_geospatial_packages),
            ("ML Packages", self.check_ml_packages),
            ("Earth Engine", self.check_earth_engine),
            ("Foundation Models", self.check_foundation_models),
            ("HuggingFace Access", self.check_huggingface_access),
            ("Jupyter Kernel", self.check_jupyter_kernel),
            ("Model Loading", self.test_model_loading),
            ("Performance", self.performance_benchmarks),
        ]
        
        for test_name, test_func in tests:
            print_header(test_name)
            try:
                test_func()
            except Exception as e:
                print_error(f"Unexpected error in {test_name}: {e}")
                self.results['tests'][test_name.lower().replace(' ', '_')] = {
                    'status': 'error', 
                    'error': str(e)
                }
        
        return self.generate_report()

if __name__ == "__main__":
    validator = EnvironmentValidator()
    success = validator.run_all_tests()
    
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
GPU Setup Testing Script for GEOG 288KC
Comprehensive GPU acceleration testing for geospatial foundation models
"""

import torch
import numpy as np
import time
import psutil
import os
from pathlib import Path
import json
from datetime import datetime

# Colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

class GPUTester:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
    def get_system_info(self):
        """Get detailed system information"""
        print_info("Gathering system information...")
        
        info = {
            'python_version': torch.__version__,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cpu_count': os.cpu_count(),
            'total_memory': psutil.virtual_memory().total / (1024**3),  # GB
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                info[f'gpu_{i}'] = {
                    'name': gpu_props.name,
                    'memory_total': gpu_props.total_memory / (1024**3),  # GB
                    'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                }
        
        self.results['system_info'] = info
        
        # Print summary
        print_success(f"PyTorch: {info['pytorch_version']}")
        if info['cuda_available']:
            print_success(f"CUDA: {info['cuda_version']}")
            print_success(f"GPUs: {info['gpu_count']}")
            for i in range(info['gpu_count']):
                gpu_info = info[f'gpu_{i}']
                print_success(f"  GPU {i}: {gpu_info['name']} ({gpu_info['memory_total']:.1f}GB)")
        else:
            print_error("CUDA not available")
            
        return info['cuda_available']

    def test_basic_gpu_operations(self):
        """Test basic GPU tensor operations"""
        print_info("Testing basic GPU operations...")
        
        if not torch.cuda.is_available():
            print_error("CUDA not available - skipping GPU tests")
            self.results['tests']['basic_gpu'] = {'status': 'skipped', 'reason': 'no_cuda'}
            return False
            
        try:
            # Test tensor creation and movement
            cpu_tensor = torch.randn(1000, 1000)
            gpu_tensor = cpu_tensor.cuda()
            
            # Test basic operations
            result = torch.mm(gpu_tensor, gpu_tensor.t())
            result_cpu = result.cpu()
            
            # Test memory management
            initial_memory = torch.cuda.memory_allocated()
            temp_tensor = torch.randn(5000, 5000, device='cuda')
            peak_memory = torch.cuda.memory_allocated()
            del temp_tensor
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            
            self.results['tests']['basic_gpu'] = {
                'status': 'pass',
                'initial_memory_mb': initial_memory / (1024**2),
                'peak_memory_mb': peak_memory / (1024**2),
                'final_memory_mb': final_memory / (1024**2)
            }
            
            print_success("Basic GPU operations working")
            print_info(f"Memory usage: {initial_memory/(1024**2):.1f} -> {peak_memory/(1024**2):.1f} -> {final_memory/(1024**2):.1f} MB")
            return True
            
        except Exception as e:
            print_error(f"Basic GPU operations failed: {e}")
            self.results['tests']['basic_gpu'] = {'status': 'fail', 'error': str(e)}
            return False

    def benchmark_matrix_operations(self):
        """Benchmark matrix operations on CPU vs GPU"""
        print_info("Benchmarking matrix operations...")
        
        sizes = [1000, 2000, 4000]
        benchmark_results = {}
        
        for size in sizes:
            print_info(f"Testing {size}x{size} matrices...")
            
            # CPU benchmark
            try:
                a_cpu = torch.randn(size, size)
                b_cpu = torch.randn(size, size)
                
                start_time = time.time()
                c_cpu = torch.mm(a_cpu, b_cpu)
                cpu_time = time.time() - start_time
                
                print_success(f"CPU ({size}x{size}): {cpu_time:.3f}s")
                
            except Exception as e:
                print_error(f"CPU benchmark failed: {e}")
                cpu_time = None
            
            # GPU benchmark
            gpu_time = None
            speedup = None
            
            if torch.cuda.is_available():
                try:
                    a_gpu = torch.randn(size, size, device='cuda')
                    b_gpu = torch.randn(size, size, device='cuda')
                    
                    # Warm up
                    _ = torch.mm(a_gpu, b_gpu)
                    torch.cuda.synchronize()
                    
                    start_time = time.time()
                    c_gpu = torch.mm(a_gpu, b_gpu)
                    torch.cuda.synchronize()
                    gpu_time = time.time() - start_time
                    
                    if cpu_time:
                        speedup = cpu_time / gpu_time
                        
                    print_success(f"GPU ({size}x{size}): {gpu_time:.3f}s (speedup: {speedup:.1f}x)")
                    
                except Exception as e:
                    print_error(f"GPU benchmark failed: {e}")
            
            benchmark_results[f'size_{size}'] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup
            }
        
        self.results['tests']['matrix_benchmark'] = benchmark_results
        return True

    def test_mixed_precision(self):
        """Test mixed precision training capabilities"""
        print_info("Testing mixed precision capabilities...")
        
        if not torch.cuda.is_available():
            print_warning("CUDA not available - skipping mixed precision test")
            return False
            
        try:
            from torch.cuda.amp import autocast, GradScaler
            
            # Create a simple model
            model = torch.nn.Sequential(
                torch.nn.Linear(1000, 500),
                torch.nn.ReLU(),
                torch.nn.Linear(500, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 10)
            ).cuda()
            
            optimizer = torch.optim.Adam(model.parameters())
            scaler = GradScaler()
            
            # Test data
            x = torch.randn(64, 1000, device='cuda')
            y = torch.randint(0, 10, (64,), device='cuda')
            criterion = torch.nn.CrossEntropyLoss()
            
            # Mixed precision training step
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(x)
                loss = criterion(outputs, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print_success("Mixed precision training working")
            self.results['tests']['mixed_precision'] = {'status': 'pass'}
            return True
            
        except Exception as e:
            print_error(f"Mixed precision test failed: {e}")
            self.results['tests']['mixed_precision'] = {'status': 'fail', 'error': str(e)}
            return False

    def test_multi_gpu(self):
        """Test multi-GPU capabilities if available"""
        print_info("Testing multi-GPU setup...")
        
        gpu_count = torch.cuda.device_count()
        
        if gpu_count < 2:
            print_warning(f"Only {gpu_count} GPU(s) available - skipping multi-GPU test")
            self.results['tests']['multi_gpu'] = {'status': 'skipped', 'reason': 'insufficient_gpus'}
            return True
            
        try:
            # Test data parallel
            model = torch.nn.Sequential(
                torch.nn.Linear(100, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 10)
            )
            
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            
            model = model.cuda()
            
            # Test forward pass
            x = torch.randn(64, 100, device='cuda')
            output = model(x)
            
            print_success(f"Multi-GPU support working with {gpu_count} GPUs")
            self.results['tests']['multi_gpu'] = {
                'status': 'pass',
                'gpu_count': gpu_count
            }
            return True
            
        except Exception as e:
            print_error(f"Multi-GPU test failed: {e}")
            self.results['tests']['multi_gpu'] = {'status': 'fail', 'error': str(e)}
            return False

    def test_memory_management(self):
        """Test GPU memory management"""
        print_info("Testing GPU memory management...")
        
        if not torch.cuda.is_available():
            print_warning("CUDA not available - skipping memory test")
            return False
            
        try:
            # Get initial memory stats
            torch.cuda.empty_cache()
            initial_allocated = torch.cuda.memory_allocated()
            initial_reserved = torch.cuda.memory_reserved()
            
            print_info(f"Initial memory - Allocated: {initial_allocated/(1024**2):.1f}MB, Reserved: {initial_reserved/(1024**2):.1f}MB")
            
            # Allocate large tensors
            tensors = []
            for i in range(5):
                tensor = torch.randn(2000, 2000, device='cuda')
                tensors.append(tensor)
                current_allocated = torch.cuda.memory_allocated()
                print_info(f"After tensor {i+1}: {current_allocated/(1024**2):.1f}MB allocated")
            
            peak_allocated = torch.cuda.memory_allocated()
            peak_reserved = torch.cuda.memory_reserved()
            
            # Free memory
            del tensors
            torch.cuda.empty_cache()
            
            final_allocated = torch.cuda.memory_allocated()
            final_reserved = torch.cuda.memory_reserved()
            
            print_success("Memory management test completed")
            print_info(f"Peak memory - Allocated: {peak_allocated/(1024**2):.1f}MB, Reserved: {peak_reserved/(1024**2):.1f}MB")
            print_info(f"Final memory - Allocated: {final_allocated/(1024**2):.1f}MB, Reserved: {final_reserved/(1024**2):.1f}MB")
            
            self.results['tests']['memory_management'] = {
                'status': 'pass',
                'initial_allocated_mb': initial_allocated / (1024**2),
                'peak_allocated_mb': peak_allocated / (1024**2),
                'final_allocated_mb': final_allocated / (1024**2)
            }
            
            return True
            
        except Exception as e:
            print_error(f"Memory management test failed: {e}")
            self.results['tests']['memory_management'] = {'status': 'fail', 'error': str(e)}
            return False

    def test_transformer_model(self):
        """Test loading and running a transformer model on GPU"""
        print_info("Testing transformer model on GPU...")
        
        if not torch.cuda.is_available():
            print_warning("CUDA not available - running on CPU")
            device = 'cpu'
        else:
            device = 'cuda'
            
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Load a small transformer model for testing
            model_name = "distilbert-base-uncased"
            print_info(f"Loading {model_name}...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model = model.to(device)
            model.eval()
            
            # Test inference
            text = "This is a test for geospatial foundation models."
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            print_success(f"Transformer model working on {device}")
            print_info(f"Output shape: {outputs.last_hidden_state.shape}")
            
            self.results['tests']['transformer_model'] = {
                'status': 'pass',
                'device': device,
                'model': model_name
            }
            
            return True
            
        except Exception as e:
            print_error(f"Transformer model test failed: {e}")
            self.results['tests']['transformer_model'] = {'status': 'fail', 'error': str(e)}
            return False

    def generate_gpu_report(self):
        """Generate comprehensive GPU test report"""
        print_header("GPU TEST SUMMARY")
        
        if not torch.cuda.is_available():
            print_error("CUDA not available on this system")
            print_info("This system will run on CPU only")
            print_info("For optimal performance, ensure CUDA-capable GPU is available")
            return False
        
        # Calculate test results
        test_results = self.results['tests']
        total_tests = len(test_results)
        passed_tests = len([t for t in test_results.values() if t.get('status') == 'pass'])
        failed_tests = len([t for t in test_results.values() if t.get('status') == 'fail'])
        
        print(f"\n{Colors.BOLD}GPU Tests Summary:{Colors.END}")
        print(f"{Colors.GREEN}✅ Passed: {passed_tests}{Colors.END}")
        print(f"{Colors.RED}❌ Failed: {failed_tests}{Colors.END}")
        
        # Performance summary
        if 'matrix_benchmark' in test_results:
            print(f"\n{Colors.BOLD}Performance Benchmarks:{Colors.END}")
            benchmarks = test_results['matrix_benchmark']
            for size_key, results in benchmarks.items():
                size = size_key.replace('size_', '')
                if results['speedup']:
                    print(f"  {size}x{size}: {results['speedup']:.1f}x speedup")
        
        # Memory summary
        if 'memory_management' in test_results:
            mem_results = test_results['memory_management']
            if mem_results['status'] == 'pass':
                peak_gb = mem_results['peak_allocated_mb'] / 1024
                print(f"\n{Colors.BOLD}Memory Usage:{Colors.END}")
                print(f"  Peak allocation: {peak_gb:.2f}GB")
        
        # Recommendations
        print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")
        
        if failed_tests == 0:
            print_success("All GPU tests passed! System ready for foundation model training.")
        else:
            if any('mixed_precision' in str(test) for test in test_results if test_results[test].get('status') == 'fail'):
                print_warning("Mixed precision issues detected - may need newer GPU")
            if any('memory' in str(test) for test in test_results if test_results[test].get('status') == 'fail'):
                print_warning("Memory issues detected - consider reducing batch sizes")
        
        # Save detailed report
        report_path = Path.home() / 'geoAI' / 'gpu_test_report.json'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\n{Colors.BLUE}Detailed report saved to: {report_path}{Colors.END}")
        
        return failed_tests == 0

    def run_all_tests(self):
        """Run all GPU tests"""
        print_header("GEOG 288KC GPU TESTING")
        
        # System info
        cuda_available = self.get_system_info()
        
        if not cuda_available:
            print_warning("CUDA not available - some tests will be skipped")
        
        # Run tests
        tests = [
            ("Basic GPU Operations", self.test_basic_gpu_operations),
            ("Matrix Benchmarks", self.benchmark_matrix_operations),
            ("Mixed Precision", self.test_mixed_precision),
            ("Multi-GPU Support", self.test_multi_gpu),
            ("Memory Management", self.test_memory_management),
            ("Transformer Model", self.test_transformer_model),
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
        
        return self.generate_gpu_report()

if __name__ == "__main__":
    tester = GPUTester()
    success = tester.run_all_tests()
    
    exit(0 if success else 1)
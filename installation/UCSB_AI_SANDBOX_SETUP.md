# UCSB AI Sandbox Environment Setup for GEOG 288KC

This guide provides comprehensive instructions for setting up the optimal environment for **GEOG 288KC: Geospatial Foundation Models and Applications** on the UCSB GRIT AI Sandbox.

## Overview

The UCSB AI Sandbox is a high-performance computing environment managed by General Research IT (GRIT) that provides:
- **GPU Resources**: NVIDIA A100, H100, and V100 GPUs for model training and inference
- **Pre-configured Software**: CUDA, cuDNN, and basic ML frameworks
- **Shared Storage**: High-performance file systems for large datasets
- **Container Support**: Docker and Singularity for reproducible environments

## Quick Start

```bash
# Clone the course repository
git clone https://github.com/kcaylor/GEOG-288KC-geospatial-foundation-models.git
cd GEOG-288KC-geospatial-foundation-models

# Run the automated setup script
bash scripts/setup_ai_sandbox.sh

# Activate the environment
conda activate geoAI

# Validate installation
python scripts/validate_environment.py
```

## Detailed Setup Instructions

### Step 1: Initial AI Sandbox Access

1. **Request Access** to UCSB AI Sandbox through GRIT:
   - Submit request at: https://grit.ucsb.edu/services/ai-sandbox
   - Specify: GEOG 288KC course enrollment
   - Request: GPU access for geospatial foundation model training

2. **Login to AI Sandbox**:
   ```bash
   ssh your_username@ai-sandbox.ucsb.edu
   ```

3. **Check Available Resources**:
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Check CUDA version
   nvcc --version
   
   # Check available storage
   df -h
   ```

### Step 2: Create Optimized Environment

The course uses a specialized conda environment optimized for GPU-accelerated geospatial AI:

```bash
# Clone course repository
cd ~/
git clone https://github.com/kcaylor/GEOG-288KC-geospatial-foundation-models.git
cd GEOG-288KC-geospatial-foundation-models

# Create the environment from the GPU-optimized specification
conda env create -f environment-gpu.yml

# Activate the environment
conda activate geoAI

# Install additional packages via pip
pip install -r requirements-gpu.txt

# Install Jupyter kernel
python -m ipykernel install --user --name geoai --display-name "GeoAI"
```

### Step 3: Install Foundation Models

Run the automated foundation model installation script:

```bash
# Make script executable
chmod +x scripts/install_foundation_models.sh

# Run installation (this may take 30-60 minutes)
bash scripts/install_foundation_models.sh
```

This script will download and configure:
- **Prithvi-100M**: NASA's 100M parameter foundation model for Earth observation
- **SatMAE**: Self-supervised masked autoencoder for satellite imagery
- **GeoFM**: Multi-modal foundation model for geospatial applications
- **CLIP**: Vision-language models adapted for remote sensing
- **Additional models**: Based on research needs and availability

### Step 4: Validate Installation

```bash
# Run comprehensive validation
python scripts/validate_environment.py

# Test GPU acceleration
python scripts/test_gpu_setup.py

# Test foundation model loading
python scripts/test_foundation_models.py
```

## Environment Specifications

### Hardware Requirements
- **Minimum**: 1 GPU with 16GB VRAM (V100 or better)
- **Recommended**: 1 GPU with 40GB+ VRAM (A100 or H100)
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ for models and datasets

### Software Stack
- **OS**: Ubuntu 20.04/22.04 LTS
- **CUDA**: 11.8 or 12.1
- **Python**: 3.11
- **PyTorch**: 2.1+ with CUDA support
- **GDAL**: 3.6+
- **Earth Engine**: Latest API

## Storage Organization

Recommended directory structure on AI Sandbox:

```
~/geoAI/                          # Course repository
├── models/                       # Foundation models cache
│   ├── prithvi/                 # Prithvi model files
│   ├── satmae/                  # SatMAE model files
│   └── custom/                  # Fine-tuned models
├── data/                        # Course datasets
│   ├── raw/                     # Original data
│   ├── processed/               # Preprocessed data
│   └── samples/                 # Small test datasets
├── experiments/                 # Student experiments
└── shared/                      # Shared resources
```

## GPU Optimization

### Memory Management
```python
# Configure PyTorch for optimal GPU memory usage
import torch
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# Set memory fraction if needed
torch.cuda.set_per_process_memory_fraction(0.8)
```

### Multi-GPU Support
```python
# Check for multiple GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### Efficient Data Loading
```python
# Optimize data loading for GPU training
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)
```

## Foundation Model Integration

### HuggingFace Models
```python
from transformers import AutoModel, AutoProcessor

# Load Prithvi model
model = AutoModel.from_pretrained(
    "ibm-nasa-geospatial/Prithvi-100M",
    cache_dir="/shared/models/prithvi"
)

# Load processor
processor = AutoProcessor.from_pretrained(
    "ibm-nasa-geospatial/Prithvi-100M"
)
```

### TorchGeo Integration
```python
import torchgeo
from torchgeo.models import get_model

# Load pre-trained weights for land cover classification
model = get_model("resnet18", weights="landcoverai")
```

## Data Access and Management

### Earth Engine Setup
```bash
# Authenticate Earth Engine (one-time setup)
earthengine authenticate

# Test authentication
python -c "import ee; ee.Initialize(); print('Earth Engine authenticated successfully')"
```

### Large Dataset Handling
```python
# Use Dask for large dataset processing
import dask.array as da
import xarray as xr

# Load large NetCDF files efficiently
ds = xr.open_mfdataset("data/large/*.nc", chunks={"time": 10})
```

## Performance Monitoring

### GPU Monitoring
```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Log GPU usage to file
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv --loop=5 > gpu_usage.log
```

### Memory Profiling
```python
# Profile memory usage in notebooks
%load_ext memory_profiler
%memit your_function()

# Profile GPU memory
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Common Workflows

### Model Training Pipeline
```bash
# 1. Prepare data
python scripts/prepare_training_data.py --dataset sentinel2 --region california

# 2. Start training with GPU acceleration
python scripts/train_model.py --model prithvi --gpus 1 --batch-size 32

# 3. Monitor training
tensorboard --logdir lightning_logs/

# 4. Evaluate model
python scripts/evaluate_model.py --checkpoint best_model.ckpt
```

### Distributed Training
```bash
# Multi-GPU training on single node
python -m torch.distributed.run --nproc_per_node=4 scripts/train_distributed.py

# Multi-node training (if available)
python -m torch.distributed.run --nnodes=2 --node_rank=0 --master_addr="master_ip" scripts/train_distributed.py
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```python
   # Reduce batch size
   batch_size = 16  # instead of 32
   
   # Use gradient checkpointing
   model.gradient_checkpointing_enable()
   
   # Clear cache between batches
   torch.cuda.empty_cache()
   ```

2. **Slow Data Loading**:
   ```python
   # Increase num_workers
   dataloader = DataLoader(dataset, num_workers=min(8, os.cpu_count()))
   
   # Use faster data formats
   # Convert to HDF5 or Zarr instead of individual files
   ```

3. **Earth Engine Authentication**:
   ```bash
   # Re-authenticate if needed
   earthengine authenticate --force
   
   # Check service account setup
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
   ```

### Getting Help

- **GRIT Support**: Submit tickets at https://grit.ucsb.edu/support
- **Course Forum**: Use GitHub Discussions for technical questions
- **Office Hours**: Fridays 2-5pm for hands-on troubleshooting
- **Documentation**: Comprehensive guides in `/docs` directory

## Security and Best Practices

### Data Security
- **Never commit credentials** to version control
- **Use environment variables** for API keys and tokens
- **Encrypt sensitive data** before uploading to shared storage

### Resource Sharing
- **Monitor GPU usage** and release resources when not actively training
- **Use efficient batch sizes** to maximize throughput
- **Clean up intermediate files** to conserve storage

### Code Organization
- **Use version control** for all experiments
- **Document hyperparameters** and model configurations
- **Create reproducible environments** with exact package versions

---

## Next Steps

After completing the environment setup:

1. **Complete the validation tests** in `scripts/validate_environment.py`
2. **Review the course datasets** in the `data/` directory
3. **Explore example notebooks** in `course-materials/labs/`
4. **Submit your project application** with your research interests

For detailed usage instructions and advanced configuration options, see the individual script documentation and course materials.
#!/bin/bash

# Foundation Model Installation Script for GEOG 288KC
# This script downloads and configures key geospatial foundation models
# Optimized for UCSB AI Sandbox environment

set -e  # Exit on any error

# Configuration (honor shared class dirs if provided)
# MODELS_DIR: where installed models are stored
# CACHE_DIR: Hugging Face cache root for downloads (maps to huggingface_hub snapshot cache)
MODELS_DIR="${GEOAI_MODELS_DIR:-${HOME}/geoAI/models}"
if [[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
  CACHE_DIR="${HUGGINGFACE_HUB_CACHE}"
elif [[ -n "${HF_HOME:-}" ]]; then
  CACHE_DIR="${HF_HOME%/}/hub"
else
  CACHE_DIR="${HOME}/.cache/huggingface/hub"
fi
LOG_FILE="${HOME}/geoAI/installation.log"
CATALOG_FILE="$(cd "$(dirname "$0")" && pwd)/../models_catalog.yml"

# Flags / args
DRYRUN=0
INFO_ONLY=0
SELECT_ALL=0
SELECT_MODELS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
    log "INFO: $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    log "SUCCESS: $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    log "WARNING: $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log "ERROR: $1"
}

usage() {
    cat <<USAGE
Usage: $(basename "$0") [--dryrun] [--info] [--all] [--models name1,name2]

Options:
  --dryrun                Show disk space and total download size for selected models; do not download
  --info                  Show details for all available models from the catalog (size, params, repo)
  --all                   Install all models listed in the catalog
  --models list           Comma-separated list of model keys from the catalog to install

Examples:
  $(basename "$0") --info
  $(basename "$0") --dryrun --all
  $(basename "$0") --models prithvi-100m,satmae-fmow
USAGE
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dryrun)
        DRYRUN=1; shift ;;
      --info)
        INFO_ONLY=1; shift ;;
      --all)
        SELECT_ALL=1; shift ;;
      --models)
        SELECT_MODELS="$2"; shift 2 ;;
      -h|--help)
        usage; exit 0 ;;
      *)
        print_warning "Unknown option: $1"; usage; exit 2 ;;
    esac
  done
}

require_catalog() {
  if [[ ! -f "$CATALOG_FILE" ]]; then
    print_error "Model catalog not found: $CATALOG_FILE"
    echo "Create it or re-install the repository."
    exit 2
  fi
}

list_models_info() {
  require_catalog
  python - <<'PY'
import sys, yaml
from pathlib import Path
cat = Path("""$CATALOG_FILE""")
with cat.open() as f:
    data = yaml.safe_load(f)
models = data.get('models', {})
print("Available models (from catalog):\n")
for key, m in models.items():
    name = m.get('name', key)
    repo = m.get('repo_id', 'unknown')
    size = m.get('size_gb', 'unknown')
    params = m.get('params_m', 'unknown')
    desc = m.get('description', '')
    print(f"- key: {key}\n  name: {name}\n  repo: {repo}\n  size_gb: {size}\n  params_m: {params}\n  {desc}\n")
PY
}

compute_selected_total_size() {
  require_catalog
  SEL="$1"
  python - <<PY
import sys, yaml
from pathlib import Path
sel = set(x.strip() for x in """$SEL""".split(',') if x.strip())
cat = Path("""$CATALOG_FILE""")
with cat.open() as f:
    data = yaml.safe_load(f)
models = data.get('models', {})
if not sel:
    sel = set(models.keys())
total = 0.0
missing = []
rows = []
for k in sel:
    m = models.get(k)
    if not m:
        missing.append(k); continue
    size = float(m.get('size_gb', 0) or 0)
    total += size
    rows.append((k, m.get('name', k), size))
print("Selected models:")
for k, name, size in rows:
    print(f"  - {k:20s} {name:35s} ~{size:.2f} GB")
print(f"\nEstimated total download size: ~{total:.2f} GB")
if missing:
    print("\nWarning: missing keys in catalog:", ', '.join(missing))
PY
}

install_selected_models() {
  require_catalog
  SEL_KEYS="$1"
  MODELS_DIR="$MODELS_DIR" CACHE_DIR="$CACHE_DIR" python - <<'PY'
import os, sys, yaml
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS_DIR = os.environ.get('MODELS_DIR')
CACHE_DIR = os.environ.get('CACHE_DIR')
CATALOG_FILE = os.path.abspath("""$CATALOG_FILE""")
SEL = set(x.strip() for x in """$SEL_KEYS""".split(',') if x.strip())

with open(CATALOG_FILE, 'r') as f:
    data = yaml.safe_load(f)
models = data.get('models', {})
if not SEL:
    SEL = set(models.keys())

to_install = []
for key in SEL:
    m = models.get(key)
    if not m:
        print(f"[WARN] Unknown model key in catalog: {key}")
        continue
    repo = m.get('repo_id')
    dest = m.get('dest', key)
    local_dir = os.path.join(MODELS_DIR, dest)
    to_install.append((key, repo, local_dir))

for key, repo, local_dir in to_install:
    if not repo:
        print(f"[WARN] Skipping {key}: missing repo_id in catalog")
        continue
    if os.path.isdir(local_dir):
        print(f"[SKIP] {key}: already exists at {local_dir}")
        continue
    os.makedirs(local_dir, exist_ok=True)
    print(f"[DL ] {key} from {repo} → {local_dir}")
    try:
        snapshot_download(repo_id=repo, cache_dir=CACHE_DIR, local_dir=local_dir, local_dir_use_symlinks=False)
        print(f"[OK ] {key}: installed")
    except Exception as e:
        print(f"[ERR] {key}: {e}")
PY
}

# Check if running in correct environment
check_environment() {
    print_status "Checking environment..."
    
    if [[ "$CONDA_DEFAULT_ENV" != "geoAI" ]]; then
        print_error "Please activate the geoAI environment first:"
        print_error "conda activate geoAI"
        exit 1
    fi
    
    # Check for required packages
    python -c "import torch, transformers, huggingface_hub, yaml" 2>/dev/null || {
        print_error "Required packages not found. Please install requirements first:"
        print_error "pip install -r installation/requirements-gpu.txt"
        exit 1
    }
    
    print_success "Environment check passed"
}

# Create necessary directories
setup_directories() {
    print_status "Setting up directory structure..."
    
    mkdir -p "$MODELS_DIR"/{prithvi,satmae,geofm,clip,custom,checkpoints}
    mkdir -p "$CACHE_DIR"
    mkdir -p "${HOME}/geoAI/data"/{raw,processed,samples}
    
    print_success "Directory structure created"
}

# Check available disk space
check_disk_space() {
    print_status "Checking available disk space..."
    
    # Cross-platform disk space check
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - df uses different flags and output format
        available_space=$(df -h "$HOME" | tail -1 | awk '{print $4}')
        
        # Parse size with unit (e.g., "18Gi", "1.2Ti", "500Mi")
        if [[ "$available_space" == *"Ti" ]]; then
            # Convert TB to GB (remove 'Ti' and multiply by 1024)
            available_gb=$(echo "$available_space" | sed 's/Ti//' | awk '{printf "%.0f", $1 * 1024}')
        elif [[ "$available_space" == *"Gi" ]]; then
            # Already in GB (remove 'Gi')
            available_gb=$(echo "$available_space" | sed 's/Gi//' | awk '{printf "%.0f", $1}')
        elif [[ "$available_space" == *"Mi" ]]; then
            # Convert MB to GB (remove 'Mi' and divide by 1024)
            available_gb=$(echo "$available_space" | sed 's/Mi//' | awk '{printf "%.0f", $1 / 1024}')
        elif [[ "$available_space" == *"G" ]]; then
            # Handle 'G' without 'i' (remove 'G')
            available_gb=$(echo "$available_space" | sed 's/G//' | awk '{printf "%.0f", $1}')
        elif [[ "$available_space" == *"T" ]]; then
            # Handle 'T' without 'i' (remove 'T' and multiply by 1024)
            available_gb=$(echo "$available_space" | sed 's/T//' | awk '{printf "%.0f", $1 * 1024}')
        elif [[ "$available_space" == *"M" ]]; then
            # Handle 'M' without 'i' (remove 'M' and divide by 1024)
            available_gb=$(echo "$available_space" | sed 's/M//' | awk '{printf "%.0f", $1 / 1024}')
        else
            # Fallback: try to extract just the number
            available_gb=$(echo "$available_space" | sed 's/[^0-9.]//g' | awk '{printf "%.0f", $1}')
        fi
    else
        # Linux - use -BG for GB output
        available_gb=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | sed 's/G//')
    fi
    
    # Ensure we have a valid number
    if ! [[ "$available_gb" =~ ^[0-9]+$ ]]; then
        print_warning "Could not determine disk space automatically"
        print_warning "Foundation models require ~30-50GB. Do you want to continue? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    elif [[ $available_gb -lt 50 ]]; then
        print_warning "Low disk space: ${available_gb}GB available"
        print_warning "Foundation models require ~30-50GB. Continue? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Sufficient disk space: ${available_gb}GB available"
    fi
}

# Install HuggingFace CLI and login
setup_huggingface() {
    print_status "Setting up HuggingFace Hub..."
    
    # Check if already logged in
    if python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
        print_success "Already logged in to HuggingFace Hub"
        return
    fi
    
    print_status "Please login to HuggingFace Hub to access models:"
    print_status "1. Get your token from: https://huggingface.co/settings/tokens"
    print_status "2. Create a READ token if you don't have one"
    
    python -c "from huggingface_hub import login; login()"
    
    print_success "HuggingFace Hub authentication complete"
}

# Download Prithvi Foundation Model (NASA/IBM)
install_prithvi() {
    print_status "Installing Prithvi-100M foundation model..."
    
    local model_path="$MODELS_DIR/prithvi"
    
    if [[ -d "$model_path/Prithvi-100M" ]]; then
        print_warning "Prithvi model already exists, skipping download"
        return
    fi
    
    python << EOF
import os
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoImageProcessor

print("Downloading Prithvi-100M model...")
model_path = snapshot_download(
    repo_id="ibm-nasa-geospatial/Prithvi-100M",
    cache_dir="$CACHE_DIR",
    local_dir="$model_path/Prithvi-100M",
    local_dir_use_symlinks=False
)

print("Loading model to verify installation...")
try:
    model = AutoModel.from_pretrained("$model_path/Prithvi-100M")
    processor = AutoImageProcessor.from_pretrained("$model_path/Prithvi-100M")
    print("✅ Prithvi model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Prithvi model: {e}")
    raise
EOF
    
    print_success "Prithvi-100M model installed successfully"
}

# Download SatMAE (Satellite Masked Autoencoder)
install_satmae() {
    print_status "Installing SatMAE foundation model..."
    
    local model_path="$MODELS_DIR/satmae"
    
    if [[ -d "$model_path/SatMAE" ]]; then
        print_warning "SatMAE model already exists, skipping download"
        return
    fi
    
    python << EOF
import os
from huggingface_hub import snapshot_download
import torch

print("Downloading SatMAE model...")
try:
    model_path = snapshot_download(
        repo_id="microsoft/SatMAE-fMoW",
        cache_dir="$CACHE_DIR",
        local_dir="$model_path/SatMAE",
        local_dir_use_symlinks=False
    )
    print("✅ SatMAE model downloaded successfully")
except Exception as e:
    print(f"❌ Error downloading SatMAE: {e}")
    # Try alternative repository if primary fails
    print("Trying alternative SatMAE source...")
    try:
        model_path = snapshot_download(
            repo_id="facebook/satmae-base-patch16",
            cache_dir="$CACHE_DIR", 
            local_dir="$model_path/SatMAE-alt",
            local_dir_use_symlinks=False
        )
        print("✅ Alternative SatMAE model downloaded")
    except Exception as e2:
        print(f"❌ Alternative SatMAE also failed: {e2}")
        raise
EOF
    
    print_success "SatMAE model installed successfully"
}

# Download CLIP models for remote sensing
install_clip_rs() {
    print_status "Installing CLIP models for remote sensing..."
    
    local model_path="$MODELS_DIR/clip"
    
    python << EOF
import os
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor

models_to_install = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    # Add remote sensing specific CLIP models when available
    # "microsoft/RemoteCLIP" # If/when available
]

for model_id in models_to_install:
    model_name = model_id.split('/')[-1]
    local_path = "$model_path/" + model_name
    
    if os.path.exists(local_path):
        print(f"⚠️  {model_name} already exists, skipping...")
        continue
        
    print(f"📥 Downloading {model_name}...")
    try:
        snapshot_download(
            repo_id=model_id,
            cache_dir="$CACHE_DIR",
            local_dir=local_path,
            local_dir_use_symlinks=False
        )
        
        # Verify model loads
        model = CLIPModel.from_pretrained(local_path)
        processor = CLIPProcessor.from_pretrained(local_path)
        print(f"✅ {model_name} installed and verified")
        
    except Exception as e:
        print(f"❌ Failed to install {model_name}: {e}")
        continue
EOF
    
    print_success "CLIP models installed successfully"
}

# Download additional geospatial models
install_additional_models() {
    print_status "Installing additional geospatial models..."
    
    python << EOF
import os
from huggingface_hub import snapshot_download, hf_hub_download
import torch

# List of additional models to install
additional_models = [
    {
        "repo_id": "microsoft/DinoV2-base",
        "name": "dinov2-base",
        "description": "DINOv2 foundation model (good for geospatial features)"
    },
    {
        "repo_id": "facebook/sam-vit-base",
        "name": "segment-anything-base", 
        "description": "Segment Anything Model (useful for object detection)"
    }
]

for model_info in additional_models:
    model_name = model_info["name"]
    model_path = f"$MODELS_DIR/additional/{model_name}"
    
    if os.path.exists(model_path):
        print(f"⚠️  {model_name} already exists, skipping...")
        continue
        
    print(f"📥 Downloading {model_name}: {model_info['description']}")
    try:
        os.makedirs(f"$MODELS_DIR/additional", exist_ok=True)
        snapshot_download(
            repo_id=model_info["repo_id"],
            cache_dir="$CACHE_DIR",
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        print(f"✅ {model_name} installed successfully")
    except Exception as e:
        print(f"❌ Failed to install {model_name}: {e}")
        continue
EOF
    
    print_success "Additional models installed"
}

# Download sample datasets for testing
download_sample_data() {
    print_status "Downloading sample datasets..."
    
    local data_path="${HOME}/geoAI/data/samples"
    
    python << EOF
import os
from huggingface_hub import hf_hub_download
import requests

# Create samples directory
os.makedirs("$data_path", exist_ok=True)

# Download small sample datasets for testing
samples = [
    {
        "url": "https://github.com/microsoft/torchgeo/raw/main/tests/data/landcoverai/rgb.tif",
        "filename": "sample_rgb.tif",
        "description": "Sample RGB satellite image"
    }
]

for sample in samples:
    filepath = os.path.join("$data_path", sample["filename"])
    
    if os.path.exists(filepath):
        print(f"⚠️  {sample['filename']} already exists, skipping...")
        continue
        
    print(f"📥 Downloading {sample['description']}...")
    try:
        response = requests.get(sample["url"], timeout=30)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            f.write(response.content)
            
        print(f"✅ {sample['filename']} downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download {sample['filename']}: {e}")
        continue

print("Sample data download complete")
EOF
    
    print_success "Sample datasets downloaded"
}

# Create model registry and configuration (based on catalog and what exists)
create_model_registry() {
    print_status "Creating model registry from catalog..."
    python - <<'PY'
import os, json, yaml
from datetime import datetime
from pathlib import Path

MODELS_DIR = os.environ.get('MODELS_DIR')
CATALOG_FILE = os.path.abspath("""$CATALOG_FILE""")
with open(CATALOG_FILE, 'r') as f:
    data = yaml.safe_load(f)

models = {}
for key, m in data.get('models', {}).items():
    dest = m.get('dest', key)
    path = os.path.join(MODELS_DIR, dest)
    if os.path.isdir(path):
        models[key] = {
            'path': path,
            'type': m.get('type', 'model'),
            'domain': m.get('domain', ''),
            'parameters': f"{m.get('params_m', 'unknown')}M",
            'description': m.get('description', ''),
        }

registry = {
    'created': datetime.now().isoformat(),
    'environment': 'geoAI',
    'models': models,
}

out = os.path.join(MODELS_DIR, 'model_registry.json')
os.makedirs(MODELS_DIR, exist_ok=True)
with open(out, 'w') as f:
    json.dump(registry, f, indent=2)
print(f"Model registry written to: {out}")
PY
    print_success "Model registry created"
}

# Generate usage examples
create_examples() {
    print_status "Creating usage examples..."
    
    mkdir -p "${HOME}/geoAI/examples"
    
    cat > "${HOME}/geoAI/examples/load_prithvi.py" << 'EOF'
"""
Example: Loading and using Prithvi foundation model
"""
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np

def load_prithvi_model(model_path="models/prithvi/Prithvi-100M"):
    """Load Prithvi model and processor"""
    model = AutoModel.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, processor, device

def extract_features(model, processor, image_array, device):
    """Extract features from satellite imagery"""
    # Preprocess image
    inputs = processor(images=image_array, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state
    
    return features

if __name__ == "__main__":
    # Load model
    model, processor, device = load_prithvi_model()
    print(f"Prithvi model loaded on {device}")
    
    # Example with random data (replace with real satellite imagery)
    dummy_image = np.random.rand(224, 224, 6)  # 6-band satellite image
    features = extract_features(model, processor, dummy_image, device)
    print(f"Extracted features shape: {features.shape}")
EOF

    cat > "${HOME}/geoAI/examples/load_clip.py" << 'EOF'
"""
Example: Using CLIP for satellite image-text matching
"""
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

def load_clip_model(model_path="models/clip/clip-vit-base-patch32"):
    """Load CLIP model and processor"""
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, processor, device

def compute_similarity(model, processor, image, texts, device):
    """Compute similarity between image and text descriptions"""
    # Process inputs
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Compute similarity
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    return probs

if __name__ == "__main__":
    # Load model
    model, processor, device = load_clip_model()
    print(f"CLIP model loaded on {device}")
    
    # Example usage
    texts = [
        "agricultural field",
        "urban area", 
        "forest",
        "water body"
    ]
    
    # Dummy image (replace with real satellite imagery)
    dummy_image = Image.fromarray(
        (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    )
    
    probs = compute_similarity(model, processor, dummy_image, texts, device)
    
    for i, text in enumerate(texts):
        print(f"{text}: {probs[0][i]:.3f}")
EOF

    print_success "Usage examples created in ~/geoAI/examples/"
}

# Main installation function
main() {
    parse_args "$@"
    print_status "Starting Foundation Model Installation for GEOG 288KC"
    print_status "This process may take 30-60 minutes depending on network speed"
    
    # Info / Dryrun quick paths
    if [[ $INFO_ONLY -eq 1 ]]; then
        list_models_info
        exit 0
    fi

    # Pre-flight checks
    check_environment
    setup_directories
    check_disk_space

    # Dryrun: estimate size for selected models and exit
    if [[ $DRYRUN -eq 1 ]]; then
        if [[ $SELECT_ALL -eq 1 ]]; then
            compute_selected_total_size ""
        else
            compute_selected_total_size "$SELECT_MODELS"
        fi
        print_status "Dry run complete. No models downloaded."
        exit 0
    fi
    
    # HuggingFace setup
    setup_huggingface
    
    # Install foundation models
    print_status "Beginning model downloads..."

    if [[ $SELECT_ALL -eq 1 || -n "$SELECT_MODELS" ]]; then
        if [[ $SELECT_ALL -eq 1 ]]; then
            install_selected_models ""
        else
            install_selected_models "$SELECT_MODELS"
        fi
    else
        # Back-compat default bundle
        install_prithvi
        install_satmae
        install_clip_rs
        install_additional_models
    fi
    
    # Setup supporting materials
    download_sample_data
    create_model_registry
    create_examples
    
    # Final summary
    print_success "Foundation model installation complete!"
    print_status "Summary of installed models:"
    
    du -sh "$MODELS_DIR"/* 2>/dev/null | while read -r size path; do
        basename_path=$(basename "$path")
        echo "  📦 $basename_path: $size"
    done
    
    print_status "Next steps:"
    echo "  1. Run validation script: python installation/verify_geoai_setup.py"
    echo "  2. Test models: python examples/load_prithvi.py"
    echo "  3. Check course materials: ls course-materials/"
    
    print_success "Installation log saved to: $LOG_FILE"
}

# Handle script interruption
trap 'print_error "Installation interrupted"; exit 1' INT TERM

# Run main installation
main "$@"
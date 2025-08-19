## GRIT Installation Playbook (GeoAI Course)

### Installation footprint (approximate)

| Component | Description | Estimated size | Notes |
|---|---|---:|---|
| Conda env (macOS, MPS) | `environment.yml` (CPU/MPS stack, geospatial libs) | 6–8 GB | Depends on platform and solver; includes GDAL, Rasterio, GeoPandas, Torch (CPU/MPS) |
| Conda env (Linux, CUDA) | `installation/environment-gpu.yml` (CUDA 11.8, PyTorch, geospatial libs) | 10–15 GB | Includes `cudatoolkit`, `pytorch-cuda`, NCCL, cuDNN |
| GEO-Bench datasets | CLI-managed dataset store | 50–150+ GB | Varies by suites selected; allocate ≥100 GB for multiple suites |
| Prithvi-100M | Foundation model | ~2.5 GB | Catalog key: `prithvi-100m` |
| Prithvi EO v2 300M | Foundation model | ~3.8 GB | Catalog key: `prithvi-eo-v2-300m` |
| Prithvi EO v2 600M | Foundation model | ~7.5 GB | Catalog key: `prithvi-eo-v2-600m` |
| SatMAE fMoW | MAE | ~1.5 GB | Catalog key: `satmae-fmow` |
| CLIP ViT-B/32 | Vision-lang | ~0.5 GB | Catalog key: `clip-vit-base-patch32` |
| CLIP ViT-L/14 | Vision-lang | ~1.8 GB | Catalog key: `clip-vit-large-patch14` |
| Clay | Remote sensing LMM | ~8.0 GB | Catalog key: `clay` |
| DINOv2-Base | Vision foundation | ~0.9 GB | Catalog key: `dinov2-base` |
| SAM ViT-B | Segmentation | ~1.1 GB | Catalog key: `sam-vit-base` |

- Typical model subsets: Prithvi-100M + SatMAE + CLIP-B/32 ≈ 4.5 GB; adding Prithvi EO v2 300M ≈ 8.3 GB total.
- Full catalog (all entries above) ≈ 27–28 GB. Use the installer `--dryrun` to plan precisely on your host.


This playbook is for UCSB GRIT to provision machines for the GeoAI course. It reuses our existing environment files, Makefile targets, and validation scripts, and adds steps for GEO-Bench and TerraTorch backbones (Prithvi, SatMAE, TIMM).

### Class-wide paths (shared storage)

Set these environment variables before running dataset/model installs to use shared, class-wide locations:

```bash
# GEO-Bench dataset root used by CLI and our Makefile targets
export GEO_BENCH_DIR="/srv/datasets/geobench"   # adjust to shared storage path

# Hugging Face cache (snapshots) for all students (optional but recommended)
export HUGGINGFACE_HUB_CACHE="/srv/models/hf-cache"  # or set HF_HOME=/srv/models/hf and it will use $HF_HOME/hub

# Course model installation root (where models get materialized by the installer)
export GEOAI_MODELS_DIR="/srv/models/geoAI"

mkdir -p "$GEO_BENCH_DIR" "$HUGGINGFACE_HUB_CACHE" "$GEOAI_MODELS_DIR"
```

Notes:
- The model installer honors `GEOAI_MODELS_DIR` and `HUGGINGFACE_HUB_CACHE` (or `HF_HOME`). If not set, it falls back to user-local paths under `~/geoAI` and `~/.cache/huggingface/hub`.
- The GEO-Bench Makefile target uses `GEO_BENCH_DIR` if present.

### Outcomes
- Conda installed; System-wide (or class-wide) `geoAI` environment created (macOS MPS or Linux CUDA)
- `geogfm` installed editable; Jupyter kernel `geoai` registered
- Class-wide installation of GEO-Bench CLI available and datasets stored in a shared path
- Class-wide TerraTorch and TIMM backbones available; Prithvi and SatMAE downloaded
- Validation scripts pass on target hosts

## 1) Install Conda (Miniforge)

Conda is needed for all users, and the class environment (geoAI) should be available to all users via `conda activate geoAI`.

Linux (CUDA servers):
```bash
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh -o ~/miniforge.sh
bash ~/miniforge.sh -b -p $HOME/miniforge
source $HOME/miniforge/bin/activate
conda init bash
```

Prereq on Linux GPU: `nvidia-smi` must work and show at least one GPU.

## 2) Clone the course repository
```bash
git clone https://github.com/kcaylor/GEOG-288KC-geospatial-foundation-models.git ~/geoAI
cd ~/geoAI
```

## 3) Create the `geoAI` conda environment

macOS (MPS):
```bash
conda env create -f environment.yml -n geoAI
conda activate geoAI
```

Linux (CUDA 11.8):
```bash
conda env create -f installation/environment-gpu.yml -n geoAI
conda activate geoAI
```

Idempotent guard (optional):
```bash
make ensure-env ENV_NAME=geoAI
```

## 4) Install our package and Jupyter kernel
```bash
make install ENV_NAME=geoAI
make kernelspec ENV_NAME=geoAI
```

## 5) Install TerraTorch and GEO-Bench CLI
```bash
conda activate geoAI
pip install terratorch geobench
```

## 6) Download GEO-Bench data to shared storage that is accessible to the entire class.
Pick a shared path with sufficient space (≥100 GB if pulling multiple suites):
```bash
export GEO_BENCH_DIR="/srv/datasets/geobench"
mkdir -p "$GEO_BENCH_DIR"
make geobench ENV_NAME=geoAI GEO_BENCH_DIR="$GEO_BENCH_DIR"
```

Persist for users (recommended):
```bash
sudo bash -lc 'echo export GEO_BENCH_DIR="/srv/datasets/geobench" > /etc/profile.d/geoai.sh'
sudo chmod 644 /etc/profile.d/geoai.sh
```

## 7) Install foundation model backbones (Prithvi, SatMAE, CLIP)
```bash
# Prefer non-interactive HF login on shared servers (create token in your HF account)
huggingface-cli login --token "$HF_TOKEN"

bash installation/scripts/install_foundation_models.sh
```

This downloads and verifies models, creates `~/geoAI/models/model_registry.json`, and adds simple usage examples under `~/geoAI/examples/`.

Common options and catalog:

```bash
# Show all available models (from YAML catalog) with sizes and params
bash installation/scripts/install_foundation_models.sh --info

# Estimate total size without downloading (all or a subset)
bash installation/scripts/install_foundation_models.sh --dryrun --all
bash installation/scripts/install_foundation_models.sh --dryrun --models prithvi-100m,prithvi-eo-v2-300m,clay

# Install all catalog models, or a selected subset by key
bash installation/scripts/install_foundation_models.sh --all
bash installation/scripts/install_foundation_models.sh --models prithvi-100m,satmae-fmow
```

- The installer reads `installation/models_catalog.yml` (Prithvi-100M, Prithvi EO v2 300M/600M, SatMAE, CLIP B/32 and L/14, Clay, DINOv2-Base, SAM ViT-B). You can extend this catalog to add more models.

## 8) GPU and environment validation (must pass)

Universal GPU test (Mac MPS or Linux CUDA):
```bash
python installation/test_gpu_setup_universal.py
```

Full environment validation (imports, GPU, EE auth, HF, models, kernel, perf):
```bash
python installation/verify_geoai_setup.py
```

## 9) TerraTorch and backbone smoke tests

TerraTorch + TIMM:
```bash
python - << 'PY'
import terratorch, timm, torch
print("terratorch:", getattr(terratorch, "__version__", "unknown"))
print("timm:", timm.__version__)
print("torch:", torch.__version__)
m = timm.create_model('vit_base_patch16_224', pretrained=True)
print("timm backbone loaded:", m.__class__.__name__)
PY
```

Prithvi from local registry:
```bash
python - << 'PY'
import json
from pathlib import Path
from transformers import AutoModel
reg = Path.home()/"geoAI"/"models"/"model_registry.json"
assert reg.exists(), "Model registry not found; run install_foundation_models.sh"
cfg = json.loads(reg.read_text())
prithvi_path = cfg["models"]["prithvi"]["path"]
model = AutoModel.from_pretrained(prithvi_path)
print("Loaded Prithvi from:", prithvi_path, "->", model.__class__.__name__)
PY
```

Optional example config for later workflows:
```bash
cat book/extras/examples/terratorch-configs/classification_eurosat.yaml
```

## 10) GEO-Bench sanity check
```bash
test -d "$GEO_BENCH_DIR" && find "$GEO_BENCH_DIR" -maxdepth 2 -type d | sed -n '1,20p'
```

## 11) Optional docs and repo tests
```bash
make docs
make test
```

## 12) Common issues
- Linux GPU: `nvidia-smi` must work; ensure env created from `installation/environment-gpu.yml`.
- Mac MPS: macOS ≥ 12.3 and PyTorch ≥ 2.0 required; Apple Silicon only.
- HF auth in non-interactive shells: run `huggingface-cli login --token "$HF_TOKEN"` before model install.
- GEO-Bench not found: set `GEO_BENCH_DIR` globally or per-session.
- Model installs: use `--info` and `--dryrun` to plan disk usage; select with `--models` or `--all`.

## Final verification checklist
```bash
conda activate geoAI && python -V
python installation/test_gpu_setup_universal.py
python installation/verify_geoai_setup.py
python - << 'PY'
import terratorch, timm, torch; print(terratorch.__version__, timm.__version__, torch.__version__)
PY
bash installation/scripts/install_foundation_models.sh
echo "$GEO_BENCH_DIR" && ls -1 "$GEO_BENCH_DIR" | sed -n '1,10p'
```



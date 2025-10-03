# GFM Training Data Directory

This directory contains **training datasets and pipeline outputs** for geospatial foundation model (GFM) development.

## Purpose

- **Large training datasets** from STAC APIs
- **Model training data** and preprocessed files
- **Pipeline outputs** from `build_from_stac.py`
- **Git-ignored** due to size (typically 100MB+ datasets)

## Directory Organization

```
data/
├── aois/                    # Areas of interest (GeoJSON files)
├── out/                     # STAC pipeline outputs (git-ignored)
│   ├── meta/
│   │   ├── scenes.parquet   # Scene manifests
│   │   └── splits/          # Train/val/test splits
│   └── CHECKSUMS.md
└── README.md               # This file
```

## Course vs Training Data

📁 **`data/`** (this directory) - GFM training data and pipeline outputs  
📁 **`book/data/`** - Small sample data for course notebooks and examples

**When to use each:**
- **`data/`** - Python scripts, model training, STAC pipelines
- **`book/data/`** - Quarto notebooks, course materials, examples

## STAC Data Pipeline

### `build_from_stac.py`

A flexible script for building reproducible satellite image datasets from STAC APIs (default: Microsoft Planetary Computer).

**Core Features:**
- Query satellite collections (Sentinel-2, Landsat, etc.) by AOI and time range
- Apply cloud cover and scene count filters
- Generate stratified train/val/test splits with deterministic seeding
- Optional stratified downsampling to target dataset sizes
- Export scene manifests and asset URLs for downstream processing

**Key Outputs** (under `data/out/`):
- `meta/scenes.parquet` - Complete scene manifest with metadata
- `meta/splits/{train,val,test}_scenes.txt` - Scene ID lists for each split
- `CHECKSUMS.md` - File integrity verification

**Usage Examples:**
```bash
# Preview course dataset (120 scenes total)
make data-course-dryrun COURSE_TARGET=120

# Build course dataset with scene cap
make data-course COURSE_TARGET=120

# Build full dataset (no scene limit)
make data-course
```

**Configuration:**
- AOIs defined in `aois/` directory (GeoJSON format)
- Stratification options: by month, AOI, or both
- Reproducible via fixed random seeds
- Supports asset URL signing for private STAC catalogs

## Adding New Data

When adding new datasets:
1. Keep file sizes reasonable (<50MB per file)
2. Document the source and license
3. Update this README
4. Use descriptive filenames
5. Consider using subdirectories for organization

## Data Sources

All datasets are either:
- Public domain
- Openly licensed (MIT, CC, etc.)
- Educational use permitted
- Created synthetically for course purposes

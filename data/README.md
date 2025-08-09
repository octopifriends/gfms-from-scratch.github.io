# Course Data Directory

This directory contains sample datasets used throughout the GEOG 288KC course materials.

## Sample Images

### `landcover_sample.tif`
- **Source**: [TorchGeo test data](https://github.com/microsoft/torchgeo/blob/main/tests/data/landcoverai/images/M-33-20-D-c-4-2.tif)
- **Dataset**: LandCover.ai 
- **Format**: GeoTIFF (RGB satellite imagery)
- **Size**: ~13KB
- **Usage**: Introduction to geospatial tokenization (`course-materials/examples/geospatial_tokenization.qmd`)
- **License**: Same as TorchGeo (MIT License)

## Usage in Lessons

Course materials reference this data directory using relative paths:
- From `course-materials/examples/`: `../../data/`
- From `course-materials/interactive-sessions/`: `../../data/`
- From root-level notebooks: `./data/`

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

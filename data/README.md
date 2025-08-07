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

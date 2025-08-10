# Book Course Data

This directory contains **sample datasets** used in course materials, examples, and interactive sessions within the book.

## Purpose

- **Small sample files** for educational examples
- **Committed to git** for easy distribution
- **Self-contained** with the book materials
- **Cross-platform compatibility** ensured

## Current Files

### `landcover_sample.tif`
- **Source**: [TorchGeo test data](https://github.com/microsoft/torchgeo/blob/main/tests/data/landcoverai/images/M-33-20-D-c-4-2.tif)
- **Dataset**: LandCover.ai 
- **Format**: GeoTIFF (RGB satellite imagery)
- **Size**: ~13KB
- **Usage**: Introduction to geospatial tokenization and pipeline building
- **License**: Same as TorchGeo (MIT License)

## Usage in Course Materials

All book materials reference this directory using relative paths:
- From `course-materials/`: `../data/`
- From `course-materials/examples/`: `../../data/`
- From `course-materials/extras/`: `../../data/`

## Adding New Sample Data

When adding new educational datasets:
1. **Keep files small** (<1MB per file preferred)
2. **Document the source** and license
3. **Update this README**
4. **Use descriptive filenames**
5. **Ensure cross-platform compatibility**

## Data Organization

📁 **`book/data/`** (this directory) - Course sample data for notebooks  
📁 **`data/`** (repository root) - GFM training data and STAC outputs

The root `data/` directory is used by Python scripts outside the book (model training, data pipelines, etc.) and is typically larger and git-ignored.

## License

All sample datasets are either:
- Public domain
- Openly licensed (MIT, CC, etc.)
- Educational use permitted
- Created synthetically for course purposes

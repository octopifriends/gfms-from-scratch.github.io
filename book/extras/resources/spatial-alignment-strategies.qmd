---
title: "Spatial Alignment Strategies for Multi-Scene Processing"
subtitle: "Handling pixel-perfect alignment in geospatial workflows"
---

## Introduction

When processing multiple satellite scenes with spatial subsets, you may encounter slight dimensional mismatches (e.g., 284×285 vs 285×284 pixels). The Week 2 preprocessing pipeline handles this by trimming arrays to a common minimum shape, which works well for most analyses. However, some applications require absolute pixel-perfect alignment.

This guide explains when you need precise alignment and how to implement it.

## When Trimming Is Acceptable

The trimming approach used in Week 2 is appropriate for:

- **Exploratory analysis**: Understanding patterns and trends in your data
- **Statistical aggregation**: Computing median composites, mean NDVI, or temporal statistics
- **Visualization**: Creating RGB composites or thematic maps
- **Educational demos**: Fast iteration and testing during development
- **Large-area analysis**: Where ±20m (1 pixel at 20m resolution) is negligible

**Impact**: Losing 1 pixel (~20m) from a 5km × 5km area (250×250 pixels) affects <0.5% of your data.

## When You Need Absolute Alignment

Use reference grid approaches when:

- **Change detection**: Pixel-by-pixel comparison across dates
- **Time series analysis**: Tracking individual pixels over time
- **Machine learning**: Training models with precise patch locations
- **Sub-pixel registration**: Co-registering different sensors
- **Precise area calculations**: Legal boundaries, carbon accounting, land use reports
- **Regulatory compliance**: Environmental monitoring, agricultural subsidies

## Strategy 1: Reference Grid (Recommended)

Define a master grid before loading any data. All scenes are reprojected to this exact grid.

### Implementation

```python
import rasterio
from rasterio import Affine
import numpy as np
import xarray as xr

def create_reference_grid(bbox, target_crs='EPSG:32611', resolution=20):
    """
    Create a reference grid with exact pixel boundaries.

    Args:
        bbox: [west, south, east, north] in WGS84
        target_crs: Target coordinate reference system
        resolution: Pixel size in meters

    Returns:
        Dictionary with grid parameters
    """
    from pyproj import Transformer

    # Transform bbox to target CRS
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    west, south = transformer.transform(bbox[0], bbox[1])
    east, north = transformer.transform(bbox[2], bbox[3])

    # Snap to exact pixel boundaries (round to nearest resolution)
    west = np.floor(west / resolution) * resolution
    south = np.floor(south / resolution) * resolution
    east = np.ceil(east / resolution) * resolution
    north = np.ceil(north / resolution) * resolution

    # Calculate exact dimensions
    width = int((east - west) / resolution)
    height = int((north - south) / resolution)

    # Create transform
    transform = Affine.translation(west, north) * Affine.scale(resolution, -resolution)

    return {
        'transform': transform,
        'width': width,
        'height': height,
        'crs': target_crs,
        'bounds': (west, south, east, north)
    }


def reproject_to_reference_grid(band_array, src_transform, src_crs, ref_grid):
    """
    Reproject a band to the reference grid with exact alignment.

    Args:
        band_array: Input array to reproject
        src_transform: Source affine transform
        src_crs: Source CRS
        ref_grid: Reference grid parameters from create_reference_grid()

    Returns:
        Reprojected array matching reference grid exactly
    """
    from rasterio.warp import reproject, Resampling

    # Create destination array
    dst_array = np.empty((ref_grid['height'], ref_grid['width']), dtype=band_array.dtype)

    # Reproject to exact grid
    reproject(
        source=band_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_grid['transform'],
        dst_crs=ref_grid['crs'],
        resampling=Resampling.bilinear
    )

    return dst_array
```

### Usage Example

```python
# Step 1: Define reference grid once
ref_grid = create_reference_grid(
    bbox=[-119.85, 34.40, -119.80, 34.45],
    target_crs='EPSG:32611',
    resolution=20
)

# Step 2: Load and reproject each scene to the reference grid
aligned_scenes = []
for item in scene_items:
    # Load bands (this may have variable dimensions)
    band_data = load_sentinel2_bands(item, bands=['B04', 'B03', 'B02', 'B08'])

    # Reproject each band to reference grid
    aligned_bands = {}
    for band_name, band_array in band_data.items():
        aligned_bands[band_name] = reproject_to_reference_grid(
            band_array.values,
            band_array.rio.transform(),
            band_array.rio.crs,
            ref_grid
        )

    aligned_scenes.append(aligned_bands)

# Step 3: All scenes now have identical dimensions - no trimming needed!
# Stack directly with xr.concat()
```

### Advantages
- **Guaranteed alignment**: All scenes match exactly
- **Reproducible**: Save grid parameters for future use
- **Production-ready**: Standard approach in operational systems
- **Educational**: Teaches coordinate systems and transforms

### Disadvantages
- More complex setup
- Requires understanding of coordinate transformations
- Slightly more code to maintain

## Strategy 2: First Scene as Reference

Use the first loaded scene to define the grid for all subsequent scenes.

### Implementation

```python
def align_to_first_scene(scenes):
    """Align all scenes to match the first scene's grid."""
    if not scenes:
        return []

    # First scene becomes the reference
    ref_scene = scenes[0]
    ref_shape = ref_scene['red'].shape
    ref_transform = ref_scene['red'].rio.transform()
    ref_crs = ref_scene['red'].rio.crs

    aligned = [ref_scene]  # First scene is already aligned to itself

    # Align all other scenes
    for scene in scenes[1:]:
        aligned_scene = {}
        for band_name, band_array in scene.items():
            # Reproject to match reference
            aligned_band = band_array.rio.reproject(
                ref_crs,
                shape=ref_shape,
                transform=ref_transform,
                resampling=Resampling.bilinear
            )
            aligned_scene[band_name] = aligned_band
        aligned.append(aligned_scene)

    return aligned
```

### Advantages
- Simple to implement
- Uses rioxarray's built-in functionality
- No need to pre-define grid parameters

### Disadvantages
- Dependent on first scene being representative
- Grid changes if you change which scene is first

## Strategy 3: Rioxarray's reproject_match()

Use xarray's built-in alignment for cleaner code.

### Implementation

```python
def align_with_reproject_match(scenes):
    """Use rioxarray.reproject_match() for alignment."""
    if not scenes:
        return []

    reference = scenes[0]['red']  # Use first scene's red band as reference

    aligned = []
    for scene in scenes:
        aligned_scene = {}
        for band_name, band_array in scene.items():
            # Align to reference grid
            aligned_scene[band_name] = band_array.rio.reproject_match(reference)
        aligned.append(aligned_scene)

    return aligned
```

### Advantages
- Very clean code
- Handles CRS and grid alignment together
- Preserves coordinate metadata

### Disadvantages
- Requires keeping data in memory
- Less explicit about grid parameters

## Strategy 4: Pre-compute Bbox in Target CRS

Transform the bounding box to target CRS and snap to grid boundaries before loading data.

### Implementation

```python
def snap_bbox_to_grid(bbox_wgs84, target_crs='EPSG:32611', resolution=20):
    """
    Transform bbox to target CRS and snap to grid boundaries.

    Returns bbox in target CRS with exact pixel boundaries.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    west, south = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
    east, north = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])

    # Snap to grid
    west = np.floor(west / resolution) * resolution
    south = np.floor(south / resolution) * resolution
    east = np.ceil(east / resolution) * resolution
    north = np.ceil(north / resolution) * resolution

    return [west, south, east, north]

# Usage: Use snapped bbox for all data requests
snapped_bbox = snap_bbox_to_grid([-119.85, 34.40, -119.80, 34.45])
# This bbox now aligns perfectly with 20m pixels
```

### Advantages
- Prevents fractional pixels at boundaries
- Works with existing Week 2 pipeline
- Minimal code changes

### Disadvantages
- Still may have ±1 pixel differences due to other factors
- Doesn't handle scenes from different tiles

## Choosing the Right Strategy

| Use Case | Recommended Strategy | Why |
|----------|---------------------|-----|
| Class projects requiring change detection | Strategy 1 (Reference Grid) | Guaranteed alignment, reusable |
| Quick exploratory analysis | Current trimming approach | Fast, simple, good enough |
| Production monitoring system | Strategy 1 (Reference Grid) | Reproducible, documented |
| Single-session analysis | Strategy 2 (First Scene) | Simple, no pre-planning needed |
| Interactive notebook work | Strategy 3 (reproject_match) | Clean code, easy to understand |
| Preventing issues upfront | Strategy 4 (Snap bbox) | Combines with other strategies |

## Integration with Week 2 Pipeline

To modify the Week 2 pipeline for absolute alignment:

1. **Before batch processing**: Create reference grid
2. **In load_scene_with_cloudmask()**: Add reprojection to reference grid
3. **Remove trimming logic**: From create_temporal_mosaic() and build_temporal_datacube()
4. **Document grid parameters**: Save reference grid for reproducibility

Example modification to `load_scene_with_cloudmask()`:

```python
def load_scene_with_cloudmask_aligned(item, ref_grid, good_pixel_classes=[4, 5, 6]):
    """Load scene and align to reference grid."""
    # Load bands as usual
    band_data = load_sentinel2_bands(item, bands=['B04', 'B03', 'B02', 'B08', 'SCL'])

    # Reproject each band to reference grid
    aligned_data = {}
    for band_name, band_array in band_data.items():
        aligned_data[band_name] = reproject_to_reference_grid(
            band_array.values,
            band_array.rio.transform(),
            band_array.rio.crs,
            ref_grid
        )

    # Apply cloud masking
    masked_data, valid_fraction = apply_cloud_mask(
        aligned_data, aligned_data['SCL'], good_pixel_classes
    )

    return masked_data, valid_fraction
```

## Performance Considerations

- **Reference grid**: Adds ~2-3 seconds per scene for reprojection
- **Memory**: Reprojection requires holding source and destination in memory
- **Disk space**: No difference - aligned and trimmed arrays are similar size
- **Complexity**: Reference grid adds ~20 lines of setup code

For most student projects, the performance cost is acceptable for the guarantee of alignment.

## Further Reading

- [Rasterio Reprojection Guide](https://rasterio.readthedocs.io/en/latest/topics/reproject.html)
- [Rioxarray Spatial Methods](https://corteva.github.io/rioxarray/stable/rioxarray.html)
- [GDAL Coordinate Systems](https://gdal.org/tutorials/osr_api_tut.html)
- [Affine Transformations in Rasterio](https://rasterio.readthedocs.io/en/latest/topics/transforms.html)

## Questions?

If you need help implementing precise alignment for your project:
1. Determine if you truly need pixel-perfect alignment (see "When You Need Absolute Alignment")
2. Choose appropriate strategy based on your use case
3. Test with small subset first
4. Document your grid parameters for reproducibility

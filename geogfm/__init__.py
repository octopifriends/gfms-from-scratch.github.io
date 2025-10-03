# GeogFM package init (Week 1)
# Provides base package visibility for data + core modules

__version__ = "0.1.0"

# Import submodules to make them available
from geogfm import c01

# Expose key functions from c01 for convenience
from geogfm.c01 import (
    verify_environment,
    setup_planetary_computer_auth,
    search_sentinel2_scenes,
    search_STAC_scenes,
    load_sentinel2_bands,
    get_subset_from_scene,
    get_scene_info,
    normalize_band,
    create_rgb_composite,
    calculate_ndvi,
    calculate_band_statistics,
    plot_band_comparison,
    save_geotiff,
    create_scene_tiles,
    test_subset_functionality,
    export_analysis_results,
    load_week1_data,
)

__all__ = [
    "c01",
    # Week 1 functions
    "verify_environment",
    "setup_planetary_computer_auth",
    "search_sentinel2_scenes",
    "search_STAC_scenes",
    "load_sentinel2_bands",
    "get_subset_from_scene",
    "get_scene_info",
    "normalize_band",
    "create_rgb_composite",
    "calculate_ndvi",
    "calculate_band_statistics",
    "plot_band_comparison",
    "save_geotiff",
    "create_scene_tiles",
    "test_subset_functionality",
    "export_analysis_results",
    "load_week1_data",
]

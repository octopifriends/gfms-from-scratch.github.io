# Tangled on 2025-09-26T08:44:18

"""Week 1: Core Tools and Data Access functions for geospatial AI."""

import sys
import importlib.metadata
import warnings
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import time
import logging

# Core geospatial libraries
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np
import pandas as pd
from pystac_client import Client
import planetary_computer as pc

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_environment(required_packages: list) -> dict:
    """
    Verify that all required packages are installed.

    Parameters
    ----------
    required_packages : list
        List of package names to verify

    Returns
    -------
    dict
        Dictionary with package names as keys and versions as values
    """
    results = {}
    missing_packages = []

    for package in required_packages:
        try:
            version = importlib.metadata.version(package)
            results[package] = version
        except importlib.metadata.PackageNotFoundError:
            missing_packages.append(package)
            results[package] = None

    # Report results
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return results

    print(f"✅ All {len(required_packages)} packages verified")
    return results

def setup_planetary_computer_auth() -> bool:
    """
    Configure authentication for Microsoft Planetary Computer.

    Uses environment variables and .env files for credential discovery,
    with graceful degradation to anonymous access.

    Returns
    -------
    bool
        True if authenticated, False for anonymous access
    """
    # Try environment variables first (production)
    auth_key = os.getenv('PC_SDK_SUBSCRIPTION_KEY') or os.getenv('PLANETARY_COMPUTER_API_KEY')

    # Fallback to .env file (development)
    if not auth_key:
        env_file = Path('.env')
        if env_file.exists():
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(('PC_SDK_SUBSCRIPTION_KEY=', 'PLANETARY_COMPUTER_API_KEY=')):
                            auth_key = line.split('=', 1)[1].strip().strip('"\'')
                            break
            except Exception:
                pass  # Continue with anonymous access

    # Configure authentication
    if auth_key and len(auth_key) > 10:
        try:
            pc.set_subscription_key(auth_key)
            logger.info("Planetary Computer authentication successful")
            return True
        except Exception as e:
            logger.warning(f"Authentication failed: {e}")

    logger.info("Using anonymous access (basic rate limits)")
    return False

def search_sentinel2_scenes(
    bbox: List[float],
    date_range: str,
    cloud_cover_max: float = 20,
    limit: int = 10
) -> List:
    """
    Search Sentinel-2 Level 2A scenes using STAC API.

    Parameters
    ----------
    bbox : List[float]
        Bounding box as [west, south, east, north] in WGS84
    date_range : str
        ISO date range: "YYYY-MM-DD/YYYY-MM-DD"
    cloud_cover_max : float
        Maximum cloud cover percentage
    limit : int
        Maximum scenes to return

    Returns
    -------
    List[pystac.Item]
        List of STAC items sorted by cloud cover (ascending)
    """
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    search_params = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": date_range,
        "query": {"eo:cloud_cover": {"lt": cloud_cover_max}},
        "limit": limit
    }

    search_results = catalog.search(**search_params)
    items = list(search_results.items())

    # Sort by cloud cover (best quality first)
    items.sort(key=lambda x: x.properties.get('eo:cloud_cover', 100))

    logger.info(f"Found {len(items)} Sentinel-2 scenes (cloud cover < {cloud_cover_max}%)")
    return items

def load_sentinel2_bands(
    item,
    bands: List[str] = ['B04', 'B03', 'B02', 'B08'],
    subset_bbox: Optional[List[float]] = None,
    max_retries: int = 3
) -> Dict[str, Union[np.ndarray, str]]:
    """
    Load Sentinel-2 bands with optional spatial subsetting.

    Parameters
    ----------
    item : pystac.Item
        STAC item representing the satellite scene
    bands : List[str]
        Spectral bands to load
    subset_bbox : Optional[List[float]]
        Spatial subset as [west, south, east, north] in WGS84
    max_retries : int
        Number of retry attempts per band

    Returns
    -------
    Dict[str, Union[np.ndarray, str]]
        Band arrays plus georeferencing metadata
    """
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds

    band_data = {}
    successful_bands = []
    failed_bands = []

    for band_name in bands:
        if band_name not in item.assets:
            failed_bands.append(band_name)
            continue

        asset_url = item.assets[band_name].href

        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                # URL signing for authenticated access
                signed_url = pc.sign(asset_url)

                # Memory-efficient loading with rasterio
                with rasterio.open(signed_url) as src:
                    # Validate data source
                    if src.width == 0 or src.height == 0:
                        raise ValueError(f"Invalid raster dimensions: {src.width}x{src.height}")

                    if subset_bbox:
                        # Intelligent subsetting with CRS transformation
                        try:
                            # Transform bbox to source CRS if needed
                            if src.crs != rasterio.crs.CRS.from_epsg(4326):
                                subset_bbox_src_crs = transform_bounds(
                                    rasterio.crs.CRS.from_epsg(4326), src.crs, *subset_bbox
                                )
                            else:
                                subset_bbox_src_crs = subset_bbox

                            # Calculate reading window
                            window = from_bounds(*subset_bbox_src_crs, src.transform)

                            # Ensure window is within raster bounds
                            window = window.intersection(
                                rasterio.windows.Window(0, 0, src.width, src.height)
                            )

                            if window.width > 0 and window.height > 0:
                                data = src.read(1, window=window)
                                transform = src.window_transform(window)
                                bounds = rasterio.windows.bounds(window, src.transform)
                                if src.crs != rasterio.crs.CRS.from_epsg(4326):
                                    bounds = transform_bounds(src.crs, rasterio.crs.CRS.from_epsg(4326), *bounds)
                            else:
                                # Fall back to full scene
                                data = src.read(1)
                                transform = src.transform
                                bounds = src.bounds
                        except Exception:
                            # Fall back to full scene on subset error
                            data = src.read(1)
                            transform = src.transform
                            bounds = src.bounds
                    else:
                        # Load full scene
                        data = src.read(1)
                        transform = src.transform
                        bounds = src.bounds

                    if data.size == 0:
                        raise ValueError("Loaded data has zero size")

                    # Store band data and metadata
                    band_data[band_name] = data
                    if 'transform' not in band_data:
                        band_data.update({
                            'transform': transform,
                            'crs': src.crs,
                            'bounds': bounds,
                            'scene_id': item.id,
                            'date': item.properties['datetime'].split('T')[0]
                        })

                    successful_bands.append(band_name)
                    break

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    failed_bands.append(band_name)
                    logger.warning(f"Failed to load band {band_name}: {str(e)[:50]}")
                    break

    # Validate results
    if len(successful_bands) == 0:
        raise Exception(f"Failed to load any bands from scene {item.id}")

    if failed_bands:
        logger.warning(f"Failed to load {len(failed_bands)} bands: {failed_bands}")

    logger.info(f"Successfully loaded {len(successful_bands)} bands: {successful_bands}")
    return band_data

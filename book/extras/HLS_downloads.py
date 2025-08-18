# Colab/Notebook: HLS (HLSS30) samples for Santa Barbara with EO-2.0-friendly filenames
# Installs (safe to re-run)

import os
import datetime as dt
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import rioxarray as rxr

from pystac_client import Client
import planetary_computer as pc
from stackstac import stack
from tqdm import tqdm
# import Path
from pathlib import Path

#%%
# -----------------------
# User params
# -----------------------
OUT_DIR = Path("./hls_santabarbara")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Center on UCSB campus and buffer ~7 km (adjust as you like)
center_lat, center_lon = 34.4138, -119.8489
buffer_km = 7  # controls crop size

# Time window to search (narrow = faster); pick a year or two that suits your course
DATE_START = "2018-01-01"
DATE_END   = "2020-12-31"

# Choose HLSS30 (Sentinel flavor) or HLSL30 (Landsat flavor)
HLS_COLLECTION = "HLSS30"  # or "HLSL30"

# Max number of outputs to save
N_SAMPLES = 4

# Basic filtering
MAX_CLOUD = 20  # % cloud cover threshold
PREFERRED_MONTHS = list(range(4, 11))  # (optional) bias to April–Oct for less cloud

# Bands to export; HLSS30 is 30 m harmonized—common bands below
# You can trim to just RGB (B04,B03,B02) if you want smaller files
BANDS = ["B02","B03","B04","B05","B06","B07","B08","B11","B12"]


# -----------------------
# Build AOI polygon
# -----------------------
wgs84 = "EPSG:4326"
gdf = gpd.GeoDataFrame(geometry=[Point(center_lon, center_lat)], crs=wgs84)
# Project to an equal-area/metric CRS around Santa Barbara for buffering
gdf_m = gdf.to_crs("EPSG:3310")  # California Albers
aoi = gdf_m.buffer(buffer_km * 1000).to_crs(wgs84).iloc[0]

# -----------------------
# STAC search (Microsoft Planetary Computer)
# -----------------------
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

search = catalog.search(
    collections=["hls"],  # MPC groups HLSL30 + HLSS30 under 'hls'
    datetime=f"{DATE_START}/{DATE_END}",
    intersects=aoi.__geo_interface__,
    query={
        "hls:product": {"eq": HLS_COLLECTION},         # HLSS30 or HLSL30
        "eo:cloud_cover": {"lt": MAX_CLOUD},           # cloud screen
    },
    limit=500
)

items = list(search.get_items())
if not items:
    raise SystemExit("No HLS items found—try loosening dates, clouds, or AOI size.")

# Optional: bias to months you prefer (e.g., drier months), then pick well-spaced dates
def sort_key(it):
    d = dt.datetime.fromisoformat(it.properties["datetime"].replace("Z",""))
    month_bias = 0 if d.month in PREFERRED_MONTHS else 1
    return (month_bias, it.properties.get("eo:cloud_cover", 100), d)

items.sort(key=sort_key)

# Keep one per unique date (ish) and spread them out
selected = []
used_days = set()
for it in items:
    d = dt.datetime.fromisoformat(it.properties["datetime"].replace("Z",""))
    yearday = (d.year, d.timetuple().tm_yday)
    # avoid near-duplicates within +/- 10 days to spread the series
    if any(abs(d.timetuple().tm_yday - ud) <= 10 and d.year == uy for (uy, ud) in used_days):
        continue
    selected.append(it)
    used_days.add((d.year, d.timetuple().tm_yday))
    if len(selected) >= N_SAMPLES:
        break

if not selected:
    raise SystemExit("Could not pick samples—try increasing N_SAMPLES or loosening filters.")

print(f"Selected {len(selected)} HLSS30 scenes")

# -----------------------
# Fetch, clip, and save
# -----------------------
# Sign assets for direct reads
signed_items = [pc.sign(it).to_dict() for it in selected]

# Stack the requested bands as a single DataArray per item, then write GeoTIFFs
for it_signed in tqdm(signed_items):
    props = it_signed["properties"]
    item_id = it_signed["id"]                    # e.g., HLS.S30.T11SLT.2018043T183031.v2.0
    tile_id = props.get("hls:tile_id") or props.get("mgrs:tile", "TXXXXXX")
    dt_iso = props["datetime"]
    dt_obj = dt.datetime.fromisoformat(dt_iso.replace("Z",""))
    yyyyddd = f"{dt_obj.year}{dt_obj.timetuple().tm_yday:03d}"
    hhmmss  = dt_obj.strftime("%H%M%S")

    # Build nice filename per your EO-2.0 convention
    out_name = f"SantaBarbara_HLS.S30.{tile_id}.{yyyyddd}T{hhmmss}.v2.0_cropped.tif"
    out_path = OUT_DIR / out_name

    # Build a small STAC "collection" with just this item for stackstac
    single = {"type": "FeatureCollection", "features": [it_signed]}

    # Create a band stack clipped to AOI
    da = stack(
        single,
        assets=BANDS,
        chunks={"x": 1024, "y": 1024},
        resolution=30,                      # HLS is harmonized to 30 m
        bounds=aoi.bounds,                  # quick clip; we’ll mask to exact polygon below
        dtype="uint16",
        fill_value=0,
    )

    # Reorder to (band, y, x) and attach band names
    da = da.compute().transpose("band", "y", "x")
    da = da.assign_coords(band=("band", BANDS))

    # Mask outside the AOI polygon precisely
    # Convert to a DataArray with georeferencing, then rasterize mask
    da = da.rio.write_crs("EPSG:4326", inplace=True).rio.reproject_match(da)  # ensure CRS tag
    # Make a mask raster from AOI (reproject if needed)
    # (rioxarray supports clipping by geometry directly)
    da_clipped = da.rio.clip([aoi.__geo_interface__], crs="EPSG:4326", drop=False)

    # Write Cloud Optimized GeoTIFF (COG-ish) with lossless compression
    da_clipped.rio.to_raster(
        out_path,
        dtype="uint16",
        compress="deflate",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    print(f"Wrote {out_path}")

print("Done! Files are in:", OUT_DIR.resolve())
# %%

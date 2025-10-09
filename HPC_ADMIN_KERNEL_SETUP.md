# HPC Administrator Guide: Configuring Shared geoai Jupyter Kernel

## Purpose

Configure the shared `geoai` Jupyter kernel to properly set GDAL/PROJ environment variables, resolving "proj.db not found" errors that cause geospatial operations to fail or hang.

## Problem Summary

Students using the shared `geoai` Jupyter kernel encounter GDAL/PROJ errors when accessing geospatial data:
- **Error**: `PROJ: proj_create_from_database: proj.db not found`
- **Symptom**: Code fails or becomes extremely slow when using rasterio, GDAL, or STAC data access
- **Cause**: Missing `PROJ_LIB`, `PROJ_DATA`, and `GDAL_DATA` environment variables

## Solution Overview

Set required environment variables in the Jupyter kernel configuration so they are automatically available to all users of the shared kernel.

---

## Implementation Steps

### Step 1: Locate the Kernel Configuration

Find the `geoai` kernel's `kernel.json` file:

```bash
# Typical locations:
jupyter kernelspec list

# Or search directly:
find /usr/local/share/jupyter/kernels -name "kernel.json" -path "*/geoai/*"
find ~/.local/share/jupyter/kernels -name "kernel.json" -path "*/geoai/*"

# For system-wide kernels:
ls -la /usr/local/share/jupyter/kernels/geoai/
```

Example output:
```
Available kernels:
  geoai    /usr/local/share/jupyter/kernels/geoai
  python3  /usr/share/jupyter/kernels/python3
```

### Step 2: Identify the Conda Environment Path

Determine where the `geoai` conda environment is installed:

```bash
# Check current kernel.json for the Python path
cat /usr/local/share/jupyter/kernels/geoai/kernel.json

# The "argv" field shows the Python interpreter, e.g.:
# "/opt/conda/envs/geoai/bin/python"
# This means conda environment is at: /opt/conda/envs/geoai

# Or locate directly:
conda env list | grep geoai
```

Extract the environment path (we'll call it `$GEOAI_ENV_PATH`).

### Step 3: Verify Required Files Exist

Confirm the GDAL/PROJ data files are present:

```bash
# Replace /opt/conda/envs/geoai with your actual path
GEOAI_ENV_PATH="/opt/conda/envs/geoai"

# Check for proj.db
ls -lh $GEOAI_ENV_PATH/share/proj/proj.db

# Check for GDAL data
ls -lh $GEOAI_ENV_PATH/share/gdal/

# Should see proj.db (7-9 MB) and GDAL data files
```

If files are missing, install them:

```bash
conda activate geoai
conda install -c conda-forge proj-data gdal -y
```

### Step 4: Update kernel.json

**IMPORTANT**: Back up the original configuration first:

```bash
KERNEL_DIR="/usr/local/share/jupyter/kernels/geoai"
sudo cp $KERNEL_DIR/kernel.json $KERNEL_DIR/kernel.json.backup
```

Edit the `kernel.json` to add environment variables:

```bash
sudo nano $KERNEL_DIR/kernel.json
# or
sudo vi $KERNEL_DIR/kernel.json
```

**Before** (typical configuration):
```json
{
  "display_name": "geoai",
  "language": "python",
  "argv": [
    "/opt/conda/envs/geoai/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ]
}
```

**After** (with environment variables):
```json
{
  "display_name": "geoai",
  "language": "python",
  "argv": [
    "/opt/conda/envs/geoai/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "env": {
    "PROJ_LIB": "/opt/conda/envs/geoai/share/proj",
    "PROJ_DATA": "/opt/conda/envs/geoai/share/proj",
    "GDAL_DATA": "/opt/conda/envs/geoai/share/gdal",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.vrt",
    "GDAL_HTTP_TIMEOUT": "300",
    "GDAL_HTTP_MAX_RETRY": "5"
  }
}
```

**Key points:**
- Replace `/opt/conda/envs/geoai` with your actual conda environment path
- The `env` section adds environment variables to the kernel
- Additional GDAL settings optimize network data access (STAC/cloud data)

### Step 5: Alternative - Use Wrapper Script (If Preferred)

If you prefer not to modify `kernel.json`, create a wrapper script:

```bash
# Create wrapper script
sudo nano /usr/local/bin/geoai-jupyter-wrapper.sh
```

Contents:
```bash
#!/bin/bash
# Wrapper script for geoai Jupyter kernel with GDAL/PROJ configuration

# Set GDAL/PROJ environment variables
export PROJ_LIB="/opt/conda/envs/geoai/share/proj"
export PROJ_DATA="/opt/conda/envs/geoai/share/proj"
export GDAL_DATA="/opt/conda/envs/geoai/share/gdal"

# Additional GDAL optimization for cloud data
export GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.vrt"
export GDAL_HTTP_TIMEOUT="300"
export GDAL_HTTP_MAX_RETRY="5"

# Execute Python kernel
exec /opt/conda/envs/geoai/bin/python -m ipykernel_launcher "$@"
```

Make it executable:
```bash
sudo chmod +x /usr/local/bin/geoai-jupyter-wrapper.sh
```

Update `kernel.json` to use the wrapper:
```json
{
  "display_name": "geoai",
  "language": "python",
  "argv": [
    "/usr/local/bin/geoai-jupyter-wrapper.sh",
    "-f",
    "{connection_file}"
  ]
}
```

### Step 6: Restart Jupyter Services

For the changes to take effect, restart relevant services:

```bash
# For JupyterHub:
sudo systemctl restart jupyterhub

# For standalone Jupyter:
sudo systemctl restart jupyter

# Or if using supervisord:
sudo supervisorctl restart jupyterhub
```

---

## Verification

### Admin Verification

Test the kernel configuration:

```bash
# Start a Python session with the kernel
/opt/conda/envs/geoai/bin/python -c "
import os
print('PROJ_LIB:', os.environ.get('PROJ_LIB'))
print('PROJ_DATA:', os.environ.get('PROJ_DATA'))
print('GDAL_DATA:', os.environ.get('GDAL_DATA'))
"
```

### User Verification Instructions

Provide this test notebook cell to students:

```python
# Test 1: Check environment variables
import os
print("PROJ_LIB:", os.environ.get('PROJ_LIB'))
print("PROJ_DATA:", os.environ.get('PROJ_DATA'))
print("GDAL_DATA:", os.environ.get('GDAL_DATA'))

# Test 2: Verify proj.db is accessible
from pathlib import Path
proj_db = Path(os.environ['PROJ_LIB']) / 'proj.db'
print(f"\nproj.db exists: {proj_db.exists()}")
if proj_db.exists():
    print(f"proj.db size: {proj_db.stat().st_size / 1024 / 1024:.2f} MB")

# Test 3: Test PROJ functionality
from osgeo import osr
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
print("\n✅ PROJ coordinate transformation successful!")

# Test 4: Test GDAL
from osgeo import gdal
print(f"✅ GDAL version: {gdal.__version__}")

# Test 5: Test geospatial libraries
import rasterio
import geopandas as gpd
print("✅ Rasterio and GeoPandas imported successfully")
```

**Expected output:**
```
PROJ_LIB: /opt/conda/envs/geoai/share/proj
PROJ_DATA: /opt/conda/envs/geoai/share/proj
GDAL_DATA: /opt/conda/envs/geoai/share/gdal

proj.db exists: True
proj.db size: 7.90 MB

✅ PROJ coordinate transformation successful!
✅ GDAL version: 3.10.x
✅ Rasterio and GeoPandas imported successfully
```

---

## Troubleshooting

### Issue 1: Changes Not Taking Effect

**Solution**: Ensure users restart their kernel or start a new notebook
- JupyterLab: Kernel → Restart Kernel
- Jupyter Notebook: Kernel → Restart

Users may need to refresh their browser or log out/in.

### Issue 2: Permission Denied

**Solution**: Check file permissions:
```bash
sudo chmod 644 /usr/local/share/jupyter/kernels/geoai/kernel.json
sudo chown root:root /usr/local/share/jupyter/kernels/geoai/kernel.json
```

### Issue 3: Multiple Kernels Named "geoai"

**Solution**: Remove duplicate kernels:
```bash
jupyter kernelspec list
sudo jupyter kernelspec uninstall geoai
# Then reinstall the correct one
```

### Issue 4: Environment Variables Still Not Set

**Solution**: Verify kernel is being used:
```python
# In notebook, check Python interpreter
import sys
print(sys.executable)
# Should show: /opt/conda/envs/geoai/bin/python
```

If wrong interpreter, kernel may not be properly installed.

---

## Alternative: System-Wide Environment Variables

If modifying individual kernel configurations is not feasible, set environment variables system-wide:

### Option A: Add to JupyterHub Configuration

Edit `/etc/jupyterhub/jupyterhub_config.py`:

```python
# Add environment variables for all users
c.Spawner.environment = {
    'PROJ_LIB': '/opt/conda/envs/geoai/share/proj',
    'PROJ_DATA': '/opt/conda/envs/geoai/share/proj',
    'GDAL_DATA': '/opt/conda/envs/geoai/share/gdal',
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif,.tiff,.vrt',
    'GDAL_HTTP_TIMEOUT': '300',
    'GDAL_HTTP_MAX_RETRY': '5',
}
```

### Option B: Add to Conda Environment Activation

Edit the environment's activation script:

```bash
# Create activation script
mkdir -p /opt/conda/envs/geoai/etc/conda/activate.d
sudo nano /opt/conda/envs/geoai/etc/conda/activate.d/gdal-config.sh
```

Contents:
```bash
#!/bin/bash
# GDAL/PROJ configuration for geoai environment

export PROJ_LIB="$CONDA_PREFIX/share/proj"
export PROJ_DATA="$CONDA_PREFIX/share/proj"
export GDAL_DATA="$CONDA_PREFIX/share/gdal"
export GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.vrt"
export GDAL_HTTP_TIMEOUT="300"
export GDAL_HTTP_MAX_RETRY="5"
```

Make executable:
```bash
sudo chmod +x /opt/conda/envs/geoai/etc/conda/activate.d/gdal-config.sh
```

---

## Technical Background

### Why This Is Necessary

1. **PROJ database**: Contains coordinate system definitions (EPSG codes, transformations)
2. **GDAL data**: Required for format drivers and metadata
3. **HPC environments**: Often have multiple Python/GDAL installations that conflict
4. **Jupyter kernels**: Don't inherit shell environment by default

### What the Variables Do

- `PROJ_LIB` / `PROJ_DATA`: Tell PROJ where to find `proj.db`
- `GDAL_DATA`: Tell GDAL where to find format definitions
- `GDAL_DISABLE_READDIR_ON_OPEN`: Optimize cloud data access
- `CPL_VSIL_CURL_ALLOWED_EXTENSIONS`: Reduce unnecessary network calls
- `GDAL_HTTP_TIMEOUT` / `MAX_RETRY`: Handle network interruptions gracefully

---

## Contact Information

For questions or issues:

- **Course Instructor**: Kelly Caylor
- **Course Materials**: https://github.com/kcaylor/GEOG-288KC-geospatial-foundation-models
- **Detailed Troubleshooting**: See `GDAL_PROJ_HPC_TROUBLESHOOTING.md` in course repository

---

## Summary Checklist

- [ ] Locate geoai kernel.json file
- [ ] Identify conda environment path
- [ ] Verify proj.db and GDAL data files exist
- [ ] Back up original kernel.json
- [ ] Add environment variables to kernel.json
- [ ] Restart Jupyter services
- [ ] Test with verification notebook
- [ ] Notify students of changes
- [ ] Document configuration for future reference

---

## Quick Reference

**Minimal kernel.json addition:**
```json
"env": {
  "PROJ_LIB": "/path/to/geoai/share/proj",
  "PROJ_DATA": "/path/to/geoai/share/proj",
  "GDAL_DATA": "/path/to/geoai/share/gdal"
}
```

**Student test command:**
```python
from osgeo import osr
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
print("✅ Working!")
```

This configuration ensures all students using the shared `geoai` kernel will have proper GDAL/PROJ functionality without individual setup.



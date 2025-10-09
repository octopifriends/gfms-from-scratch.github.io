# HPC Quick Start Guide - GDAL/PROJ Fix

## The Problem You're Experiencing

```
⚠️  PROJ: proj_create_from_database: proj.db not found
❌ Code hangs or fails when accessing Microsoft Planetary Computer STAC
```

## Quick Fix (Copy-Paste Ready)

### Option 1: Run Diagnostic Tool (Recommended)

```bash
# SSH into your HPC
ssh youruser@hpc.university.edu

# Activate your environment
conda activate geoAI

# Run diagnostic and get setup commands
python scripts/diagnose_gdal_hpc.py
```

The tool will show you exactly what to export. Copy those lines!

### Option 2: Manual Setup (Works on Most Systems)

```bash
# Activate environment
conda activate geoAI

# Set environment variables
export PROJ_LIB=$CONDA_PREFIX/share/proj
export PROJ_DATA=$CONDA_PREFIX/share/proj
export GDAL_DATA=$CONDA_PREFIX/share/gdal

# Verify it worked
ls $PROJ_LIB/proj.db  # Should show the file

# Now run your code
python your_script.py
```

### Option 3: For SLURM Jobs

Add these lines at the top of your `.slurm` script:

```bash
#!/bin/bash
#SBATCH --job-name=geoai
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Activate environment
conda activate geoAI

# CRITICAL: Set these BEFORE running Python
export PROJ_LIB=$CONDA_PREFIX/share/proj
export PROJ_DATA=$CONDA_PREFIX/share/proj
export GDAL_DATA=$CONDA_PREFIX/share/gdal

# Now your Python script will work
python my_analysis.py
```

### Option 4: In Your Python Code (Alternative)

Add this at the **very beginning** of your notebook/script (before any imports):

```python
import os
import sys

# Get conda prefix
conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)

# Set GDAL/PROJ environment
os.environ['PROJ_LIB'] = f"{conda_prefix}/share/proj"
os.environ['PROJ_DATA'] = f"{conda_prefix}/share/proj"
os.environ['GDAL_DATA'] = f"{conda_prefix}/share/gdal"

# NOW import geospatial libraries
import rasterio
from osgeo import gdal

print("✅ GDAL/PROJ configured")
```

Or use the built-in function from the course:

```python
# At the top of your notebook/script
from geogfm.c01 import configure_gdal_environment

# Auto-configure
config = configure_gdal_environment()

if config['proj_configured']:
    print("✅ Ready to go!")
```

## Verify It Works

Run this test:

```python
from osgeo import osr

# Should complete without warnings
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
print("✅ PROJ working!")
```

## Permanent Fix (Optional)

Add to your `~/.bashrc`:

```bash
# Add at the end of ~/.bashrc
if [[ $CONDA_DEFAULT_ENV == "geoAI" ]]; then
    export PROJ_LIB=$CONDA_PREFIX/share/proj
    export PROJ_DATA=$CONDA_PREFIX/share/proj
    export GDAL_DATA=$CONDA_PREFIX/share/gdal
fi
```

Then reload:
```bash
source ~/.bashrc
```

## Still Having Issues?

1. **Check your conda environment is activated:**
   ```bash
   echo $CONDA_DEFAULT_ENV  # Should show "geoAI"
   ```

2. **Check proj.db exists:**
   ```bash
   ls $CONDA_PREFIX/share/proj/proj.db
   ```

3. **Run full diagnostic:**
   ```bash
   python scripts/diagnose_gdal_hpc.py > diagnostic.txt
   cat diagnostic.txt
   ```

4. **Check full guide:**
   See [GDAL_PROJ_HPC_TROUBLESHOOTING.md](GDAL_PROJ_HPC_TROUBLESHOOTING.md)

## What Changed in Your Course Materials

✅ **New function**: `configure_gdal_environment()` automatically detects and sets paths  
✅ **Diagnostic tool**: `scripts/diagnose_gdal_hpc.py` identifies issues  
✅ **Documentation**: Added HPC-specific guidance to Week 1 materials  
✅ **Troubleshooting guide**: Complete reference in `GDAL_PROJ_HPC_TROUBLESHOOTING.md`

## Summary

**The fix is simple**: Set three environment variables before running Python:
- `PROJ_LIB`
- `PROJ_DATA`  
- `GDAL_DATA`

Choose whichever method above works best for your workflow!


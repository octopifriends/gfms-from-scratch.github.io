# GDAL/PROJ Troubleshooting Guide for HPC Systems

## Problem Description

When running geospatial code on HPC systems, you may encounter warnings or errors related to GDAL and PROJ:

```
WARNING: PROJ: proj_create_from_database: proj.db not found
ERROR: CPLE_AppDefined in PROJ: proj_create_from_database: ...
```

These errors can cause:
- Code to fail completely
- Extremely slow performance (minutes instead of seconds)
- Incorrect coordinate transformations
- Silent failures in spatial operations

## Why This Happens

On HPC systems:
1. **Multiple installations**: System-wide and conda-based GDAL/PROJ installations may conflict
2. **Missing environment variables**: `PROJ_LIB` and `GDAL_DATA` aren't automatically set
3. **Version mismatches**: Different PROJ database versions from different installations
4. **Incomplete conda activation**: Module system may not fully activate conda environments

## Quick Fix (Recommended)

### Step 1: Run Diagnostic Tool

```bash
# Activate your environment
conda activate geoAI

# Run diagnostic
python scripts/diagnose_gdal_hpc.py

# Or export setup script
python scripts/diagnose_gdal_hpc.py --export-script ~/gdal_setup.sh
source ~/gdal_setup.sh
```

### Step 2: Set Environment Variables

Based on the diagnostic output, add to your script or session:

```bash
# Find your conda environment path first
echo $CONDA_PREFIX

# Set these variables (replace with your actual paths)
export PROJ_LIB=$CONDA_PREFIX/share/proj
export PROJ_DATA=$CONDA_PREFIX/share/proj
export GDAL_DATA=$CONDA_PREFIX/share/gdal

# Verify proj.db exists
ls -lh $PROJ_LIB/proj.db
```

### Step 3: Verify the Fix

```python
from osgeo import gdal, osr

# This should work without warnings
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
print("✅ PROJ working correctly!")
```

## Solutions by Use Case

### Interactive Session (SSH)

```bash
# Login to HPC
ssh user@hpc.university.edu

# Load modules (if required)
module load python/anaconda3

# Activate environment
conda activate geoAI

# Set environment variables
export PROJ_LIB=$CONDA_PREFIX/share/proj
export PROJ_DATA=$CONDA_PREFIX/share/proj
export GDAL_DATA=$CONDA_PREFIX/share/gdal

# Run your code
python my_analysis.py
```

### Batch Job (SLURM)

Create a job script `run_analysis.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=geoai_analysis
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=geoai_%j.out
#SBATCH --error=geoai_%j.err

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load required modules (adjust for your HPC)
module load python/anaconda3

# Activate conda environment
source activate geoAI

# CRITICAL: Set GDAL/PROJ environment variables
export PROJ_LIB=$CONDA_PREFIX/share/proj
export PROJ_DATA=$CONDA_PREFIX/share/proj
export GDAL_DATA=$CONDA_PREFIX/share/gdal

# Verify configuration
echo "PROJ_LIB: $PROJ_LIB"
echo "CONDA_PREFIX: $CONDA_PREFIX"
ls -lh $PROJ_LIB/proj.db

# Run your analysis
python /path/to/your/script.py

echo "Job finished at: $(date)"
```

Submit with:
```bash
sbatch run_analysis.slurm
```

### Jupyter Notebook on HPC

Add to the first cell of your notebook:

```python
import os
import sys

# Get conda environment path
conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)

# Set GDAL/PROJ paths
os.environ['PROJ_LIB'] = f"{conda_prefix}/share/proj"
os.environ['PROJ_DATA'] = f"{conda_prefix}/share/proj"
os.environ['GDAL_DATA'] = f"{conda_prefix}/share/gdal"

# Now import geospatial libraries
import rasterio
from osgeo import gdal, osr

print(f"✅ Environment configured")
print(f"PROJ_LIB: {os.environ['PROJ_LIB']}")
```

Or use the built-in configuration function from the course:

```python
# Import from course materials
from geogfm.c01 import configure_gdal_environment

# Configure automatically
config = configure_gdal_environment()

# Check status
if config['proj_configured']:
    print("✅ GDAL/PROJ ready!")
else:
    print("⚠️ Issues detected:")
    for warning in config['warnings']:
        print(f"  {warning}")
```

### Permanent Fix (in ~/.bashrc)

Add to your `~/.bashrc` or `~/.bash_profile`:

```bash
# GDAL/PROJ configuration for geoAI environment
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

## Advanced Troubleshooting

### Problem: Multiple PROJ Installations

**Symptom**: "DATABASE.LAYOUT.VERSION.MINOR mismatch" warnings

**Solution**: Ensure GDAL uses the conda PROJ installation:

```bash
# Check which PROJ is being used
python -c "import pyproj; print(pyproj.datadir.get_data_dir())"

# Should point to your conda environment
# If not, explicitly set PROJ_LIB before importing any geospatial libraries
```

### Problem: proj.db Missing or Corrupted

**Symptom**: "proj.db not found" or "cannot open database"

**Solution**: Reinstall PROJ data:

```bash
conda activate geoAI
conda install -c conda-forge proj-data --force-reinstall
```

### Problem: Still Getting Warnings After Setting Variables

**Solution**: Import order matters! Set environment variables BEFORE importing GDAL:

```python
# ❌ WRONG
import rasterio  # imports GDAL
os.environ['PROJ_LIB'] = '/path/to/proj'  # too late!

# ✅ CORRECT
import os
os.environ['PROJ_LIB'] = '/path/to/proj'  # set first
import rasterio  # now imports with correct settings
```

### Problem: Works Locally, Fails on HPC

**Common causes**:
1. Different module systems (Lmod, Environment Modules)
2. Different conda initialization methods
3. HPC uses older conda/mamba versions

**Solution**: Use explicit paths instead of `$CONDA_PREFIX`:

```bash
# Find your environment path
conda env list

# Use full path explicitly
export PROJ_LIB=/home/username/mambaforge/envs/geoAI/share/proj
export GDAL_DATA=/home/username/mambaforge/envs/geoAI/share/gdal
```

## Verification Checklist

Run these checks to verify your setup:

```bash
# 1. Check conda environment
conda info --envs
echo $CONDA_PREFIX

# 2. Check environment variables
echo $PROJ_LIB
echo $GDAL_DATA

# 3. Check proj.db exists
ls -lh $PROJ_LIB/proj.db

# 4. Test Python imports
python -c "from osgeo import gdal, osr; print('GDAL OK')"

# 5. Test coordinate transformation
python -c "from osgeo import osr; s=osr.SpatialReference(); s.ImportFromEPSG(4326); print('PROJ OK')"

# 6. Test with pyproj
python -c "import pyproj; print(f'PyProj: {pyproj.proj_version_str}')"
```

All checks should pass without warnings.

## HPC-Specific Notes

### PBS/Torque Systems

Use similar approach but with PBS directives:

```bash
#!/bin/bash
#PBS -N geoai_job
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=4

cd $PBS_O_WORKDIR
source activate geoAI
export PROJ_LIB=$CONDA_PREFIX/share/proj
export PROJ_DATA=$CONDA_PREFIX/share/proj
export GDAL_DATA=$CONDA_PREFIX/share/gdal

python my_script.py
```

### Module System Conflicts

If your HPC uses modules:

```bash
# Unload any system GDAL modules
module unload gdal proj

# Use conda versions exclusively
conda activate geoAI
export PROJ_LIB=$CONDA_PREFIX/share/proj
export GDAL_DATA=$CONDA_PREFIX/share/gdal
```

## Getting Help

If issues persist:

1. Run diagnostic tool and save output:
   ```bash
   python scripts/diagnose_gdal_hpc.py > gdal_diagnostic.txt
   ```

2. Check GDAL/PROJ versions:
   ```bash
   conda list gdal proj pyproj rasterio
   ```

3. Check for version conflicts:
   ```bash
   python -c "from osgeo import gdal; print(gdal.__version__)"
   python -c "import pyproj; print(pyproj.proj_version_str)"
   ```

4. Contact your HPC support with:
   - Diagnostic output
   - Version information
   - Your job script
   - Error messages

## Prevention

To avoid these issues in new environments:

```bash
# Create environment with compatible versions
conda create -n geoAI python=3.11
conda activate geoAI

# Install from conda-forge with specific versions
conda install -c conda-forge \
    gdal=3.10 \
    pyproj=3.7 \
    rasterio=1.4 \
    proj-data \
    geopandas

# Test immediately
python -c "from osgeo import osr; s=osr.SpatialReference(); s.ImportFromEPSG(4326)"
```

## References

- [GDAL Documentation](https://gdal.org/)
- [PROJ Documentation](https://proj.org/)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [Conda-Forge GDAL](https://github.com/conda-forge/gdal-feedstock)



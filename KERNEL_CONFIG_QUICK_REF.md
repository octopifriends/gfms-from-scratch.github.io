# Jupyter Kernel Configuration - Quick Reference

## Problem
```
ERROR: PROJ: proj_create_from_database: proj.db not found
```

## Solution
Add environment variables to shared `geoai` kernel configuration.

---

## Step-by-Step

### 1. Find kernel configuration
```bash
jupyter kernelspec list
# Shows: /usr/local/share/jupyter/kernels/geoai
```

### 2. Back up current config
```bash
sudo cp /path/to/geoai/kernel.json /path/to/geoai/kernel.json.backup
```

### 3. Edit kernel.json
```bash
sudo nano /path/to/geoai/kernel.json
```

### 4. Add environment variables

**BEFORE:**
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

**AFTER:**
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
    "GDAL_DATA": "/opt/conda/envs/geoai/share/gdal"
  }
}
```

⚠️ **Replace `/opt/conda/envs/geoai` with actual path from `argv` line**

### 5. Restart service
```bash
sudo systemctl restart jupyterhub
```

### 6. Verify
Students restart kernel and run:
```python
import os
print(os.environ.get('PROJ_LIB'))
# Should show: /opt/conda/envs/geoai/share/proj
```

---

## Path Finding Template

If `argv` shows: `/SOME/PATH/envs/geoai/bin/python`

Then set:
- `PROJ_LIB`: `/SOME/PATH/envs/geoai/share/proj`
- `PROJ_DATA`: `/SOME/PATH/envs/geoai/share/proj`  
- `GDAL_DATA`: `/SOME/PATH/envs/geoai/share/gdal`

---

## Test Code for Students

```python
from osgeo import osr
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
print("✅ Working!")
```

No errors = Success!

---

## Common Paths

| HPC System | Typical Path |
|------------|-------------|
| Generic conda | `/opt/conda/envs/geoai` |
| Anaconda | `/usr/local/anaconda3/envs/geoai` |
| Miniconda | `/home/shared/miniconda3/envs/geoai` |
| User conda | `/home/username/mambaforge/envs/geoai` |

---

## Rollback (if needed)

```bash
sudo cp /path/to/geoai/kernel.json.backup /path/to/geoai/kernel.json
sudo systemctl restart jupyterhub
```

---

**Full documentation**: `HPC_ADMIN_KERNEL_SETUP.md`



# Email Template for HPC Administrators

---

**Subject: Configuration Request for Shared geoai Jupyter Kernel - GDAL/PROJ Environment Variables**

---

Dear HPC Support Team,

Our course (GEOG 288KC: Geospatial Foundation Models) uses the shared `geoai` Jupyter kernel, and students are encountering errors when working with geospatial data. We need your help configuring the kernel environment variables.

## Problem

Students see this error when running geospatial code:
```
PROJ: proj_create_from_database: proj.db not found
```

This causes code to fail or hang when accessing satellite imagery via STAC APIs.

## Requested Fix

Please add the following environment variables to the shared `geoai` Jupyter kernel configuration:

### Location
Find the kernel configuration file:
```bash
jupyter kernelspec list | grep geoai
# Then edit: /path/to/kernels/geoai/kernel.json
```

### Required Changes

Add this `"env"` section to the `kernel.json` file:

```json
{
  "display_name": "geoai",
  "language": "python",
  "argv": [
    "/path/to/geoai/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "env": {
    "PROJ_LIB": "/path/to/geoai/share/proj",
    "PROJ_DATA": "/path/to/geoai/share/proj",
    "GDAL_DATA": "/path/to/geoai/share/gdal"
  }
}
```

**Note**: Replace `/path/to/geoai` with the actual conda environment path (visible in the `argv` section).

### Finding the Correct Paths

The environment path can be found from the existing kernel.json. For example, if `argv` shows:
```
"/opt/conda/envs/geoai/bin/python"
```

Then the paths should be:
- `PROJ_LIB`: `/opt/conda/envs/geoai/share/proj`
- `PROJ_DATA`: `/opt/conda/envs/geoai/share/proj`
- `GDAL_DATA`: `/opt/conda/envs/geoai/share/gdal`

### After Changes

After updating the configuration:
1. Restart JupyterHub service: `sudo systemctl restart jupyterhub`
2. Students will need to restart their kernels (Kernel → Restart)

## Verification

Students can test the fix with this code:

```python
import os
from osgeo import osr

# Check environment variables
print(f"PROJ_LIB: {os.environ.get('PROJ_LIB')}")

# Test PROJ functionality
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
print("✅ PROJ working!")
```

Expected output:
```
PROJ_LIB: /opt/conda/envs/geoai/share/proj
✅ PROJ working!
```

## Documentation

I've prepared detailed administrator instructions with troubleshooting steps:
- **Full guide**: Attached `HPC_ADMIN_KERNEL_SETUP.md`
- **Course repo**: https://github.com/kcaylor/GEOG-288KC-geospatial-foundation-models

## Timeline

This is affecting student work in Week 1 of our course. We would appreciate this being configured as soon as possible.

## Questions?

Please let me know if you need any clarification or encounter issues with this configuration.

Thank you for your assistance!

Best regards,
Kelly Caylor
GEOG 288KC: Geospatial Foundation Models

---

## Attachments to Send

1. This email
2. `HPC_ADMIN_KERNEL_SETUP.md` (detailed instructions)
3. `GDAL_PROJ_HPC_TROUBLESHOOTING.md` (comprehensive troubleshooting guide)



# What to Send to Your HPC Administrators

## Quick Summary

I've created comprehensive documentation for your HPC administrators to configure the shared `geoai` Jupyter kernel. Here's what to send and how to use it.

---

## Recommended: Send These 3 Files

### 1. **EMAIL_TO_HPC_ADMIN.md** (START HERE)
- **Purpose**: Email template you can copy/paste and send
- **What it contains**: 
  - Clear problem description
  - Exact configuration needed
  - Verification steps for students
  - Timeline/urgency
- **How to use**: Copy the contents and send as an email

### 2. **KERNEL_CONFIG_QUICK_REF.md** (ATTACH THIS)
- **Purpose**: One-page visual reference
- **What it contains**:
  - Before/after configuration examples
  - Step-by-step instructions
  - Common paths for different HPC setups
- **How to use**: Attach to email as quick reference

### 3. **HPC_ADMIN_KERNEL_SETUP.md** (ATTACH THIS)
- **Purpose**: Complete administrator guide
- **What it contains**:
  - Detailed step-by-step instructions
  - Multiple implementation options
  - Troubleshooting section
  - Verification procedures
- **How to use**: Attach to email for full details

---

## Optional: Additional Resources

If your HPC admins want more context or students need individual workarounds:

### 4. **GDAL_PROJ_HPC_TROUBLESHOOTING.md**
- Complete troubleshooting guide for all scenarios
- Solutions for interactive sessions, batch jobs, Jupyter notebooks
- Advanced debugging steps

### 5. **HPC_QUICKSTART.md**  
- Student-facing quick fix guide
- Temporary workarounds while kernel is being configured
- Multiple solution approaches

### 6. **scripts/diagnose_gdal_hpc.py**
- Diagnostic tool to identify configuration issues
- Can help admins verify the fix worked
- Generates setup commands automatically

---

## Sample Email to Send

```
Subject: Configuration Request for Shared geoai Jupyter Kernel

Dear HPC Support Team,

Our course (GEOG 288KC: Geospatial Foundation Models) needs the shared 
geoai Jupyter kernel configured with GDAL/PROJ environment variables.

PROBLEM: Students see "proj.db not found" errors when working with 
geospatial data, causing code to fail or hang.

SOLUTION: Add three environment variables to the kernel.json file:
- PROJ_LIB
- PROJ_DATA  
- GDAL_DATA

I've attached:
1. KERNEL_CONFIG_QUICK_REF.md (one-page visual guide)
2. HPC_ADMIN_KERNEL_SETUP.md (complete instructions)

The quick reference shows the exact JSON to add to the kernel 
configuration. This is a standard fix for HPC Jupyter kernels using 
geospatial libraries.

TIMELINE: Affecting student work in Week 1. Would appreciate this 
being configured ASAP.

Please let me know if you have any questions!

Best regards,
Kelly Caylor

Attachments:
- KERNEL_CONFIG_QUICK_REF.md
- HPC_ADMIN_KERNEL_SETUP.md
```

---

## What the Fix Does

The configuration adds environment variables to the shared kernel so all students automatically have:

```json
"env": {
  "PROJ_LIB": "/path/to/geoai/share/proj",
  "PROJ_DATA": "/path/to/geoai/share/proj",
  "GDAL_DATA": "/path/to/geoai/share/gdal"
}
```

This tells GDAL/PROJ where to find required data files (especially `proj.db`).

---

## What Students Will See

### Before Fix
```python
from osgeo import osr
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
# ERROR: proj.db not found
# Code hangs or fails
```

### After Fix
```python
from osgeo import osr
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
print("✅ Working!")
# Completes instantly with no errors
```

---

## Timeline

1. **Send email with attachments** → Admin receives clear instructions
2. **Admin applies configuration** → 15-30 minutes (includes verification)
3. **Restart JupyterHub service** → 1-2 minutes
4. **Students restart kernels** → Immediate effect
5. **Verify with test code** → Confirms fix worked

---

## If Admins Have Questions

Point them to:

1. **Course repository**: https://github.com/kcaylor/GEOG-288KC-geospatial-foundation-models
2. **This documentation set**: All files in repository root
3. **Your contact info**: For course-specific questions

Common admin questions answered in the full guide:
- "Why is this necessary?" → Technical background section
- "What if we have multiple environments?" → Path finding section  
- "How do we verify it worked?" → Verification section
- "What if students still have issues?" → Troubleshooting section

---

## Student Workaround (Temporary)

While waiting for kernel configuration, students can add this to their notebooks:

```python
# Add at the very top, before any imports
import os
import sys
conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
os.environ['PROJ_LIB'] = f"{conda_prefix}/share/proj"
os.environ['PROJ_DATA'] = f"{conda_prefix}/share/proj"
os.environ['GDAL_DATA'] = f"{conda_prefix}/share/gdal"

# Now import geospatial libraries
import rasterio
from osgeo import gdal
```

Or use the course function:
```python
from geogfm.c01 import configure_gdal_environment
configure_gdal_environment()
```

---

## Files Location

All documentation is in your repository root:
```
/Users/kellycaylor/dev/geoAI/
├── EMAIL_TO_HPC_ADMIN.md            ← Copy/paste for email
├── KERNEL_CONFIG_QUICK_REF.md       ← Attach (1 page)
├── HPC_ADMIN_KERNEL_SETUP.md        ← Attach (complete guide)
├── GDAL_PROJ_HPC_TROUBLESHOOTING.md ← Optional
├── HPC_QUICKSTART.md                ← Optional
└── scripts/
    └── diagnose_gdal_hpc.py         ← Optional diagnostic tool
```

---

## Success Criteria

✅ Admins understand the problem clearly  
✅ Admins have exact configuration to apply  
✅ Admins can verify the fix worked  
✅ Students can test without admin help  
✅ No ongoing maintenance needed  

---

## Next Steps

1. **Review** `EMAIL_TO_HPC_ADMIN.md`
2. **Customize** if needed (add local HPC-specific context)
3. **Attach** the two key files
4. **Send** to HPC support
5. **Follow up** after 2-3 days if no response
6. **Verify** with students once applied

The documentation is designed to make this as easy as possible for HPC staff - they should have everything they need to implement the fix in one sitting.



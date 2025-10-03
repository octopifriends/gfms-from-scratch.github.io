---
title: "Instructor Notes"
subtitle: "GRIT HPC Instructor Notes for GEOG-288KC"
jupyter: geoai
format:
  html:
    code-fold: false
---

# GRIT HPC Instructor Notes for GEOG-288KC

## Course Setup Overview

This document provides additional information for instructors and administrators setting up the GEOG-288KC course on GRIT HPC.

## Key Configuration Details

### Kernel Location

- **Main kernel path**: `/home/g288kc/.local/share/jupyter/kernels/geoai`
- **Type**: Shared kernel accessible to all course users via symbolic links

### Resource Recommendations

- **Default CPUs**: 4 cores (suitable for most coursework)
- **Default RAM**: 16 GB (adjust for data-intensive projects)
- **Max session duration**: 168 hours (7 days)

### Partition Information

- **Required partition**: `grit_nodes`
- **Purpose**: Dedicated compute nodes for GRIT users

## Student Setup Checklist

Ensure each student completes:

1. ✅ GRIT account creation
2. ✅ SSH access verification
3. ✅ Symbolic link creation for geoAI kernel
4. ✅ Successful Jupyter Notebook launch
5. ✅ geoAI kernel selection and verification

## Common Issues and Solutions

### Account Creation Delays

- New accounts may take 24-48 hours to activate
- Verify students are added to appropriate groups

### Kernel Access Problems

Students may need to:

```bash
# Remove existing link if corrupted
rm -rf ~/.local/share/jupyter/kernels/geoai

# Recreate the symbolic link
ln -s /home/g288kc/.local/share/jupyter/kernels/geoai ~/.local/share/jupyter/kernels/
```

### Resource Limitations

- Monitor overall usage to prevent resource exhaustion
- Consider staggering assignment deadlines to distribute load

## Maintenance Tasks

### Regular Checks

- Verify kernel installation is intact
- Monitor disk usage in shared directories
- Check for orphaned Jupyter sessions

### End of Semester

- Remind students to save their work locally
- Clean up temporary files and old sessions
- Archive any course-specific data as needed

## Support Contacts

- **GRIT HPC Support**: Contact through official channels
- **Course Administrator**: Maintain updated contact information
- **Technical Issues**: Document common problems and solutions

---

*Note: This document is for instructor reference and contains administrative details not included in the student guide.*

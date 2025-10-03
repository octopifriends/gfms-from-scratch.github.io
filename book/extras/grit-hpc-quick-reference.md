---
title: "Quick Reference"
subtitle: "GRIT HPC Quick Reference"
jupyter: geoai
format:
  html:
    code-fold: false
---

# GRIT HPC Quick Reference Card

## Quick Setup Checklist

- [ ] GRIT user account created
- [ ] SSH access tested
- [ ] Symbolic link created for geoAI kernel
- [ ] Web portal login successful
- [ ] Jupyter Notebook launched
- [ ] geoAI kernel selected

## Essential Information

### Access URLs

- **SSH**: `ssh hpc.grit.ucsb.edu`
- **Web Portal**: <https://hpc.grit.ucsb.edu>

### Jupyter Configuration

```text
Partition: grit_nodes
Max Duration: 168 hours
Default CPUs: 4
Default RAM: 16 GB
Kernel: geoAI Course
```

### Key Commands

**Create kernel symbolic link:**

```bash
ln -s /home/g288kc/.local/share/jupyter/kernels/geoai ~/.local/share/jupyter/kernels/
```

**Check kernel installation:**

```bash
ls -la ~/.local/share/jupyter/kernels/
```

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Can't SSH | Check network, verify username |
| No geoAI kernel | Re-create symbolic link, restart Jupyter |
| Session won't start | Check existing sessions, reduce resources |
| Kernel not working | Restart kernel from Jupyter menu |

---

*Keep this card handy during setup!*

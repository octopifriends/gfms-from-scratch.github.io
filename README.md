# GEOG 288KC: Geospatial Foundation Models

*Learn to build geospatial foundation models from scratch through hands-on implementation*

**🌐 [View Course Website](https://gfms-from-scratch.github.io))** | **📚 [Installation Guide](installation/README.md)** | **🤝 [Contributing](CONTRIBUTING.md)**

---

## 🚀 Quick Start

### I'm a Student 
1. **Browse the course**: [kellycaylor.github.io/geoAI](https://kellycaylor.github.io/geoAI)
2. **Set up your environment**: Follow [installation/README.md](installation/README.md)  
3. **Start learning**: Begin with [Week 1: Geospatial Data Foundations](https://kellycaylor.github.io/geoAI/chapters/c01-geospatial-data-foundations.html)

### I'm an Instructor/TA
1. **Get the code**: `git clone https://github.com/kellycaylor/geoAI.git && cd geoAI`
2. **Setup everything**: `make setup` (installs conda env, kernel, etc.)
3. **Edit content**: Modify `.qmd` files in `book/chapters/`, preview with `make preview`
4. **See our guides**: [AUTHORING_GUIDE.md](AUTHORING_GUIDE.md) and [CONTRIBUTING.md](CONTRIBUTING.md)

### I'm a Developer
```bash
git clone https://github.com/kellycaylor/geoAI.git
cd geoAI
make setup              # Full environment setup
make preview            # Build and serve locally
```

**Need help?** Check [installation/TROUBLESHOOTING.md](installation/TROUBLESHOOTING.md) or course Slack.

---

## 🎯 What This Course Teaches

Build **geospatial foundation models** from scratch in 10 weeks - from raw satellite data to deployable ML models.

### 📚 Course Structure (10 Weeks)

**🏗️ Weeks 1-3: Build the Architecture**
- **Week 1**: Handle geospatial data (STAC, normalization, patches)
- **Week 2**: Implement attention mechanisms for satellite imagery  
- **Week 3**: Assemble complete Vision Transformer for geospatial data

**🔥 Weeks 4-7: Train Foundation Models**
- **Week 4**: Masked autoencoder pretraining (like MAE, but for Earth data)
- **Week 5**: Optimize training loops and hyperparameters
- **Week 6**: Evaluate and visualize model performance
- **Week 7**: Load and fine-tune existing models (Prithvi, SatMAE)

**🚀 Weeks 8-10: Deploy and Apply**
- **Week 8**: Fine-tune for specific tasks (classification, segmentation)
- **Week 9**: Build inference pipelines and deployment tools
- **Week 10**: Present final projects

### 🧠 What You'll Build
- **Python package** (`geogfm/`) with complete GFM implementation
- **Working models** trained on real satellite data
- **Deployment tools** for running inference at scale
- **Course website** with all materials and examples

---

## 🏗️ How Our System Works

### The Magic: Literate Programming  
We use **Quarto** + **tangle filter** to create both educational content AND working code from the same source.

```
book/chapters/c01-*.qmd  →  [Build Process]  →  📖 Course Website (docs/)
                        →                   →  🐍 Python Package (geogfm/)
```

**Why this is awesome:**
- **Students get**: Beautiful course website + working Python package
- **Instructors get**: One source of truth (no copy-paste hell)
- **Everyone wins**: Content and code always stay in sync

### Key Directories
```
geoAI/
├── book/                    # 📖 Course content (.qmd files)
│   ├── chapters/           # 💻 Weekly sessions (c01-c10)
│   └── extras/             # 📚 Cheatsheets, examples, projects  
├── geogfm/                 # 🧠 Generated Python package
├── docs/                   # 🌐 Generated website
├── data/                   # 📊 Sample datasets  
└── installation/           # 🔧 Setup scripts
```

**What to edit:** `book/` directory (`.qmd` files)  
**What gets generated:** `geogfm/` and `docs/` (don't edit these!)

---

## ⚡ Essential Commands

```bash
# First time setup
make setup              # Install conda env, register kernel, etc.

# Daily workflow (instructors)
make preview            # Edit content + preview in browser
make docs              # Quick build (changed files only)
make docs-full         # Complete rebuild (when things break)

# Troubleshooting  
make clean             # Clear cache and temp files
make kernelspec        # Fix Jupyter kernel issues
```

**Most useful:** `make preview` rebuilds automatically as you edit!

---

## 📝 Editing Course Content

### Basic Structure
```markdown
---
title: "Session Title"
subtitle: "Week N: Specific Topic"
jupyter: geoai
---

## Overview
What students will learn...

### Data Loader → `geogfm/data/loaders.py`

```{python}
#| tangle: geogfm/data/loaders.py
# This code gets extracted to the Python package!
def create_dataloader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size)
```

Explanation of the code...
```

### The Tangle System
- **`#| tangle: path/to/file.py`** → Code block gets written to that file
- **`#| mode: overwrite`** → Replace file contents  
- **`#| mode: append`** → Add to existing file
- **Section headings** should include file paths for clarity

### Example Code Block
````markdown
### Attention Module → `geogfm/modules/attention.py`

```{python}
#| tangle: geogfm/modules/attention.py
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
```
````

Students see this content in the course website AND get working code in the `geogfm` package!

---

## 🧪 Course Data Pipeline

We include a powerful system for building reproducible satellite datasets:

```bash
# Quick dataset for course (120 scenes)
make data-course COURSE_TARGET=120

# Preview what would be downloaded
make data-course-dryrun COURSE_TARGET=120

# Full dataset (no limits)  
make data-course
```

**Features:**
- Queries STAC APIs (Microsoft Planetary Computer, etc.)
- Filters by area, date, cloud cover
- Creates balanced train/val/test splits
- Reproducible with fixed random seeds

---

## 🐛 Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Build fails | `make clean && make docs-full` |
| "Kernel not found" | `make kernelspec` |
| Import errors | `pip install -e .` |
| Preview not updating | Check syntax errors in `.qmd` files |
| Environment issues | `conda activate geoAI` |

**Pro tip:** Most problems are solved with `make clean && make setup`

---

## 📚 For Different Audiences

### Students
- **Start here**: [Course website](https://kellycaylor.github.io/geoAI)
- **Get help**: [Installation guide](installation/README.md) and [Troubleshooting](installation/TROUBLESHOOTING.md)
- **Ask questions**: Course Slack or office hours

### Instructors/TAs  
- **Content editing**: [AUTHORING_GUIDE.md](AUTHORING_GUIDE.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Technical issues**: [installation/TROUBLESHOOTING.md](installation/TROUBLESHOOTING.md)

### Developers
- **Package code**: Browse `geogfm/` for ML implementations
- **Build system**: Check `book/build_docs.py` and `Makefile`
- **Data pipeline**: See `data/build_from_stac.py`

---

## 🤝 Contributing

We welcome contributions! Whether you're:

- 🐛 **Fixing bugs** in content or code
- ✨ **Adding examples** or explanations
- 📚 **Improving documentation**
- 🎯 **Creating new exercises**

**Process:**
1. Fork the repo and create a branch
2. Make your changes in `book/` directory
3. Test with `make preview` and `make docs-full`
4. Submit PR with clear description

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 🚀 Deployment

### GitHub Pages (Automatic)
1. Push changes to `main` branch
2. GitHub Actions builds the site automatically  
3. Site available at: `https://kellycaylor.github.io/geoAI`

### Manual Build
```bash
cd book
python build_docs.py --full    # Build everything
git add docs/ && git commit -m "Update site"
git push origin main           # Deploy
```

---

## 🧠 Technologies Used

- **[Quarto](https://quarto.org/)**: Reproducible publishing system
- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **[TorchGeo](https://github.com/microsoft/torchgeo)**: Geospatial ML utilities  
- **[STAC](https://stacspec.org/)**: SpatioTemporal Asset Catalog for data discovery
- **Custom tangle filter**: Exports code from course content to Python package

---

## 📄 License & Acknowledgments

**License:** MIT License - see [LICENSE](LICENSE) file

**Inspired by:**
- [Prithvi](https://github.com/NASA-IMPACT/Prithvi-100M) - NASA's geospatial foundation model
- [SatMAE](https://github.com/microsoft/SatMAE) - Microsoft's satellite masked autoencoder  
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch image models

**Course:** GEOG 288KC at UC Santa Barbara, Fall 2025

---

**🌟 Ready to build your own geospatial foundation model? [Get started now!](https://kellycaylor.github.io/geoAI)**

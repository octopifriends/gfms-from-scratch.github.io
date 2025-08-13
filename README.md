# Building Geospatial Foundation Models

A complete educational resource and implementation framework for building geospatial foundation models (GFMs) from scratch.

## 📁 Repository Structure

This repository is organized to support both **educational content** (book materials) and **practical implementation** (GFM code):

```
geoAI/
├── book/                           # 📖 Educational content (course → book)
│   ├── chapters/                   # 📚 Chapters (weeks) and supporting materials
│   │   ├── c0X-*.qmd              # 💻 Weekly interactive sessions (10 weeks)
│   │   └── extras/                # 📖 Appendix materials
│   │       ├── cheatsheets/       # 📋 Quick reference guides
│   │       ├── examples/          # 🎯 Practical examples
│   │       ├── lectures/          # 🎓 Presentation materials  
│   │       ├── projects/          # 📁 Project templates
│   │       └── resources/         # 📚 Additional resources
│   ├── docs/                      # 🌐 Compiled website
│   ├── images/                    # 🖼️ Site images
│   ├── index.qmd                  # 🏠 Homepage
│   ├── _quarto.yml               # ⚙️ Book configuration
│   └── build_docs.py             # 🔨 Book building script
├── geogfm/                        # 🧠 GFM implementation code
├── data/                          # 📊 Sample datasets
├── installation/                  # 🔧 Setup & environment
└── tests/                         # 🧪 Test suite
```

## 🚀 Quick Start

### For Students/Learners

1. **View the book online**: [Building Geospatial Foundation Models](https://kellycaylor.github.io/geoAI)
2. **Set up your environment**: Follow the [installation guide](installation/README.md)
3. **Start with Week 1**: Begin with [geospatial data foundations](book/chapters/c01-geospatial-data-foundations.qmd)

### For Developers

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kellycaylor/geoAI.git
   cd geoAI
   ```

2. **Set up the environment**:
   ```bash
   conda env create -f environment.yml
   conda activate geoAI
   ```

3. **Install GFM package** (development mode):
   ```bash
   # Editable install so tangled code in geogfm/ is importable by Quarto
   pip install -e .

4. (Optional) Register the Jupyter kernel name used by Quarto:
   ```bash
   make kernelspec
   ```

5. Build docs:
   ```bash
   make docs       # incremental
   make docs-full  # full build (clears cache)
   ```
   ```

## 📖 About the Book

This educational resource teaches you to build geospatial foundation models from scratch through a 10-week journey:

### 🏗️ Stage 1: Build Architecture (Weeks 1-3)
- **Week 1**: Geospatial data foundations
- **Week 2**: Spatial-temporal attention mechanisms
- **Week 3**: Complete GFM architecture

### 🚀 Stage 2: Train Models (Weeks 4-7)
- **Week 4**: Pretraining implementation
- **Week 5**: Training loop optimization
- **Week 6**: Model evaluation & analysis
- **Week 7**: Integration with existing models

### 🎯 Stage 3: Apply & Deploy (Weeks 8-10)
- **Week 8**: Task-specific fine-tuning
- **Week 9**: Model deployment
- **Week 10**: Project presentations

## 🛠️ Building the Book

The book is built using [Quarto](https://quarto.org/) and outputs to the repository's `docs/` folder for GitHub Pages hosting. Quarto executes with the Jupyter kernel named `geoai` (configured in `book/_quarto.yml`) which should point to the `geoAI` conda environment.

```bash
# Navigate to the book directory
cd book

make docs-full
make docs
python book/build_docs.py --serve
```

### ✂️ Literate Code Export (Quarto tangle)

You can export code blocks from Quarto pages in `book/` directly into Python modules in the repo during build (similar to nbdev, but with `.qmd`). Tangling is enabled by the custom filter configured in `book/_quarto.yml`.

Basic usage inside a code block (hash‑pipe directives):

```python
#| tangle: ../geogfm/models/attention.py
#| mode: append
def scaled_dot_product(q, k, v):
    ...
```

Key options:
- **tangle**: relative output path (from the `.qmd` file location)
- **mode**: `append | overwrite | concat` (default: concat)

Recommended pattern for multi-block modules:
1) First block: `tangle: <path>` + `mode: overwrite` to reset file
2) Subsequent blocks: `tangle: <same path>` + `mode: append`

Exports occur whenever Quarto renders the page, so remember to commit updated files in `geogfm/` after builds.

### 🧩 Why conda + editable install?

- Heavy geospatial and DL dependencies (GDAL, PROJ, PyTorch) are reliably managed by conda.
- Keeping `pyproject.toml` minimal avoids pip/conda conflicts; `pip install -e .` only installs the `geogfm` Python package into the active env.
- As you tangle code from lessons into `geogfm/`, the editable install makes those updates immediately importable during Quarto execution.

## 📊 Building Course Datasets

The repository includes a powerful Makefile-based pipeline for creating reproducible satellite datasets from STAC APIs.

### Key Features

- **STAC Integration**: Query Microsoft Planetary Computer and other STAC APIs
- **Flexible Filtering**: By AOI, date range, cloud cover, and collection type
- **Stratified Splitting**: Deterministic train/val/test splits with balanced temporal/spatial distribution
- **Scene Capping**: Optional downsampling to target dataset sizes for course use
- **Reproducible**: Fixed random seeds ensure consistent results across builds

### Quick Commands

```bash
# Preview course dataset (120 scenes total)
make data-course-dryrun COURSE_TARGET=120

# Build course dataset with scene cap
make data-course COURSE_TARGET=120

# Build full dataset (no scene limit)
make data-course

# Custom dataset with different parameters
make data-dryrun AOI=my_area.geojson START=2023-01-01 END=2023-12-31 CLOUD=20
```

### Configuration Options

The Makefile supports flexible configuration through environment variables:

- `COURSE_TARGET`: Global scene cap (default: 0 = unlimited)
- `COURSE_AOI`: Area of interest GeoJSON file 
- `COURSE_START`/`COURSE_END`: Date range
- `COURSE_CLOUD`: Maximum cloud cover percentage
- `COURSE_MAX`: Max scenes per AOI
- `COURSE_COLLS`: Satellite collections (e.g., sentinel-2-l2a)
- `COURSE_STRATIFY`: Stratification strategy (month, aoi, or both)

### Output Structure

Datasets are built under `data/out/`:
```
data/out/
├── meta/
│   ├── scenes.parquet          # Complete scene manifest
│   └── splits/
│       ├── train_scenes.txt    # Training scene IDs
│       ├── val_scenes.txt      # Validation scene IDs  
│       └── test_scenes.txt     # Test scene IDs
└── CHECKSUMS.md               # File integrity verification
```

### 🌐 GitHub Pages Setup

1. **Build the site**: Run `cd book && python build_docs.py --full`
2. **Commit and push** the changes (including the `docs/` folder)
3. **Enable GitHub Pages**: 
   - Go to repository Settings → Pages
   - Set source to "Deploy from a branch"
   - Choose "main" branch and "/docs" folder
   - Click Save

Your site will be available at `https://yourusername.github.io/geoAI`

## 🧠 GFM Implementation

The `geogfm/` package contains:

- **Core architectures**: Vision transformers for geospatial data
- **Data pipelines**: Efficient loading and preprocessing
- **Training utilities**: Pretraining, fine-tuning, evaluation
- **Deployment tools**: API endpoints and inference interfaces

## 🤝 Contributing

This project welcomes contributions! Whether you're:

- 🐛 **Fixing bugs** in the code or content
- ✨ **Adding features** to the GFM implementation  
- 📚 **Improving documentation** or tutorials
- 🎯 **Creating examples** or use cases

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Quarto](https://quarto.org/) for reproducible publishing
- Powered by [PyTorch](https://pytorch.org/) and [TorchGeo](https://github.com/microsoft/torchgeo)
- Inspired by foundation models like [Prithvi](https://github.com/NASA-IMPACT/Prithvi-100M) and [SatMAE](https://github.com/microsoft/SatMAE)

---

*This project supports GEOG 288KC: Building Geospatial Foundation Models at UC Santa Barbara.*
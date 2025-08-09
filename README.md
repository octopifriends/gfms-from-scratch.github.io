# Building Geospatial Foundation Models

A complete educational resource and implementation framework for building geospatial foundation models (GFMs) from scratch.

## 📁 Repository Structure

This repository is organized to support both **educational content** (book materials) and **practical implementation** (GFM code):

```
geoAI/
├── book/                           # 📖 Educational content (course → book)
│   ├── course-materials/           
│   │   ├── week*.qmd              # 📚 Chapter overviews (10 weeks)
│   │   ├── weekly-sessions/       # 💻 Detailed hands-on tutorials
│   │   └── extras/                # 📖 Appendix materials
│   │       ├── cheatsheets/       # 📋 Quick reference guides
│   │       ├── examples/          # 🎯 Practical examples
│   │       ├── lectures/          # 🎓 Presentation materials  
│   │       ├── projects/          # 📁 Project templates
│   │       └── resources/         # 📚 Additional resources
│   ├── docs/                      # 🌐 Compiled website
│   ├── images/                    # 🖼️ Course images
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
3. **Start with Week 1**: Begin with [geospatial data foundations](book/course-materials/week1.qmd)

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
   pip install -e .
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

The book is built using [Quarto](https://quarto.org/) and outputs to the repository's `docs/` folder for GitHub Pages hosting.

```bash
# Navigate to the book directory
cd book

# Full build (clears cache)
python build_docs.py --full

# Incremental build (faster)
python build_docs.py

# Build and serve locally
python build_docs.py --serve
```

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
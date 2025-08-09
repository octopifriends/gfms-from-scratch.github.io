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
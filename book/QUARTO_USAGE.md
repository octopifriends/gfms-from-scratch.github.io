# Quarto Configuration and Usage Guide

## ✅ Configuration Status

The Quarto configuration has been successfully set up for the **Accelerated Hands-On GeoAI Seminar**. All major issues have been resolved:

- ✅ **Jupyter kernel**: Fixed references from `geoAI` to `geoai`
- ✅ **Filters**: Updated filter paths to correct locations
- ✅ **Extensions**: Installed `fontawesome` extension
- ✅ **Dependencies**: Verified Python/Jupyter integration

## 🚀 Quick Start Commands

### Preview the Website Locally
```bash
cd book
quarto preview --port 4200
```
*Opens a local development server at http://localhost:4200*

### Render the Entire Website
```bash
cd book
quarto render
```
*Builds the complete website to `../docs/` directory*

### Render a Single File
```bash
cd book
quarto render index.qmd
```

### Using the Custom Build Script (Recommended)
```bash
cd book
python build_docs.py --serve     # Build and serve locally
python build_docs.py --full      # Full rebuild
python build_docs.py --clean     # Clean intermediate files
```

## 📁 Directory Structure

```
book/
├── _quarto.yml              # Main configuration
├── _extensions/             # Quarto extensions (fontawesome)
├── index.qmd               # Course homepage
├── Syllabus.md             # Course syllabus
├── chapters/               # Weekly sessions (c01-c06)
├── extras/                 # Supporting materials
│   ├── cheatsheets/       # Quick reference guides
│   ├── examples/          # Practical examples
│   └── projects/          # Project templates
├── images/                # Course imagery
└── tools/                 # Custom filters and scripts
```

## ⚙️ Configuration Details

### Main Configuration (_quarto.yml)

**Key settings:**
- **Project type**: Website
- **Output directory**: `../docs` (for GitHub Pages)
- **Jupyter kernel**: `geoai`
- **Theme**: Custom SCSS (`meds-website-styles.scss`)
- **Filters**: fontawesome, custom tangle.py, path_shortcode.py

### Execution Settings
- **Freeze**: `auto` (caches execution results for faster builds)
- **Code folding**: Disabled for instructional clarity
- **Code tools**: Enabled for interactive code exploration

### Custom Filters
1. **tangle.py**: Extracts code blocks to build the `geogfm` Python package
2. **path_shortcode.py**: Handles `{{< path >}}` shortcodes for relative paths
3. **fontawesome**: Icon support for UI elements

## 🐍 Environment Requirements

### Required Environment
```bash
conda activate geoAI  # Must be active before running Quarto
```

### Required Packages
- **Python 3.11+**
- **Jupyter** with `geoai` kernel
- **panflute** (for Python filters)
- **Standard geo libraries**: rasterio, xarray, matplotlib, etc.

### Verify Setup
```bash
quarto check          # Verify Quarto installation
jupyter kernelspec list | grep geoai  # Verify kernel
python -c "import panflute; print('OK')"  # Verify filter dependencies
```

## 🔧 Common Commands

### Development Workflow
```bash
# 1. Start development server
cd book
quarto preview --port 4200

# 2. Edit content in chapters/ or extras/

# 3. Changes auto-reload in browser

# 4. When ready, build final site
quarto render
```

### Advanced Build Options
```bash
# Custom build script options
python build_docs.py --help          # Show all options
python build_docs.py --only index.qmd  # Render specific file
python build_docs.py --bootstrap     # Render core pages only
python build_docs.py --clean --full  # Clean rebuild
```

### Site Deployment
```bash
# Build for GitHub Pages (outputs to ../docs/)
cd book
quarto render

# Content is ready for deployment from docs/ directory
```

## 🐛 Troubleshooting

### Common Issues

**1. Jupyter kernel 'geoAI' not found**
```bash
# Solution: Ensure geoAI environment is active and kernel name is correct
conda activate geoAI
jupyter kernelspec list
```

**2. Filter errors (fontawesome, etc.)**
```bash
# Solution: Ensure extensions are installed in book directory
cd book
quarto add quarto-ext/fontawesome --no-prompt
```

**3. Python filter errors**
```bash
# Solution: Ensure panflute is installed
pip install panflute
```

**4. Build timeouts**
```bash
# Solution: Use frozen execution for faster builds
cd book
python build_docs.py --full  # Rebuild with caching
```

### Performance Tips

1. **Use incremental builds**: Default behavior only rebuilds changed files
2. **Enable freezing**: Set `freeze: auto` (already configured)
3. **Use build script**: `python build_docs.py` has optimizations
4. **Preview specific files**: `quarto render chapter.qmd` for testing

## 📝 Content Guidelines

### File Structure
- **Chapters**: Use `c01-`, `c02-`, etc. naming pattern
- **Jupyter kernel**: Specify `jupyter: geoai` in YAML frontmatter
- **Code visibility**: Keep `code-fold: false` for instructional content

### Adding New Content
1. Create `.qmd` file in appropriate directory
2. Add to navigation in `_quarto.yml` if needed
3. Test rendering: `quarto render filename.qmd`
4. Preview in context: `quarto preview`

## 📊 Site Analytics

**Current Structure**: 6-week accelerated format
- Week 1: Core Tools and Data Access
- Week 2: Rapid Remote Sensing Preprocessing
- Week 3: Machine Learning on Remote Sensing
- Week 4: Foundation Models in Practice
- Week 5: Fine-Tuning & Transfer Learning
- Week 6: Spatiotemporal Modeling & Projects

**Supporting Materials**:
- 18+ cheatsheets for quick reference
- 6+ practical examples with real code
- Project templates for proposals and results

The configuration is optimized for hands-on learning with immediately executable code examples and practical applications.
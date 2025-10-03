# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **geoAI** repository for developing **GEOG 288KC: Geospatial Foundation Models and Applications** (Fall 2025). The repository contains course materials and build system for creating educational content on state-of-the-art geospatial foundation models (GFMs) for remote sensing and environmental monitoring.

**Repository Structure**:
- **`book/`**: Main course materials built with Quarto containing chapters, cheatsheets, and examples
- **`geogfm/`**: Python package with core modules for geospatial ML
- **`installation/`**: Setup guides and environment configuration files

## Course Context (GEOG 288KC)

**Schedule**: Fridays 9am-12pm + optional lab office hours Fridays 2pm-5pm
**Format**: Project-driven seminar combining lectures, discussions, and hands-on labs
**Focus**: Frontier techniques in geospatial AI including self-supervised learning, masked autoencoders, multimodal sensor fusion, and scalable inference pipelines

### Course Timeline and Deliverables
- **Week 0**: Setup for geospatial AI in UCSB AI Sandbox + Project Applications
- **Week 1**: Introduction to geospatial foundation models 
- **Week 2**: Working with geospatial data and pretrained model outputs
- **Week 3**: Fine-tuning foundation models + **Project Proposals due**
- **Week 4**: Multi-modal and generative models for remote sensing
- **Week 5**: Semi-independent project work and proposal workshops
- **Week 6**: Model adaptation, efficient fine-tuning, and evaluation strategies
- **Week 7**: Scalable analysis pipelines using Earth Engine, TorchGeo, cloud tools + **Initial MVPs due**
- **Week 8**: Building lightweight APIs and applications for model inference
- **Week 9**: Project workshops and synthesis
- **Week 10**: **Final project presentations**

## Key Commands

### Building and Documentation
```bash
# Build the course website (incremental build - only changed files)
cd book
python build_docs.py

# Full rebuild of all files
python build_docs.py --full

# Build and serve locally for preview
python build_docs.py --serve

# Clean intermediate files (HTML, _files/, etc.) from course materials
python build_docs.py --clean

# Clean and full rebuild
python build_docs.py --clean --full
```

### Environment Management
```bash
# Create the conda environment for the course
conda env create -f environment.yml

# Or for faster installation with mamba
mamba env create -f environment.yml

# Activate the environment (build script does this automatically)
conda activate geoAI
```

### Manual Quarto Commands
```bash
cd book
quarto render                    # Build entire site
quarto preview                   # Build and serve locally
quarto render specific_file.qmd  # Build specific file
```

## Course Content Quality Assurance

### /preview Command System

The `/preview` command provides flexible content validation and preview capabilities for course development:

```bash
# Individual file preview and validation
/preview file=chapters/c01-geospatial-data-foundations.qmd

# Directory preview (all .qmd files in directory)
/preview dir=chapters/

# Full site preview
/preview site

# Code snippet testing (embedded .qmd content)
/preview code="
```{python}
import numpy as np
print('Hello World')
```
"

# Preview with specific validation focus
/preview file=chapters/c01-geospatial-data-foundations.qmd focus=code-execution
/preview file=chapters/c01-geospatial-data-foundations.qmd focus=build-only
/preview file=chapters/c01-geospatial-data-foundations.qmd focus=performance
```

#### Command Options:
- `file=<path>` - Preview specific .qmd file
- `dir=<path>` - Preview all .qmd files in directory
- `site` - Preview entire site
- `code="<content>"` - Test code snippet in isolation
- `focus=<type>` - Validation focus area:
  - `code-execution` - Python code testing only
  - `build-only` - Quarto rendering only
  - `performance` - Build time and memory analysis
  - `educational` - Learning objective alignment
  - `full` - Complete validation (default)

#### Integration with course-content-debugger Agent:
The `/preview` command automatically invokes the `course-content-debugger` agent with the appropriate testing scope and provides structured feedback for course development.

## Project Architecture

### High-Level Structure
- **`book/`**: Main course materials built with Quarto
- **`geogfm/`**: Python package for geospatial foundation models
- **Root level**: Container repository for geoAI educational content

### Course Structure (book/)
The course materials are organized for GEOG 288KC's weekly format:

**Current Course Structure**:
- **`chapters/`**: Weekly session content (c01 through c10):
  - Foundation model architectures and deep learning
  - Geospatial data foundations and preprocessing
  - Spatial-temporal attention mechanisms
  - Complete GFM architecture implementation
  - Training, evaluation, and deployment strategies
- **`extras/`**: Supporting materials organized by type:
  - `cheatsheets/`: Quick reference materials for tools and concepts
  - `examples/`: Practical implementation examples
  - `projects/`: Templates for proposals and MVP presentations
  - `resources/`: Course resources and references
- **`docs/`**: Generated static website (GitHub Pages output)
- **`data/`**: Course datasets and sample files
- **`images/`**: Course imagery and visual assets

**Key Features**:
- Project-driven approach with deliverable tracking (Proposals, MVPs, Final Presentations)
- Integration with UCSB AI Sandbox computational environment
- Focus on hands-on labs with geospatial foundation models, Earth Engine, and TorchGeo

### Build System
The course uses a sophisticated Python-based build system (`build_docs.py`) with:
- **Incremental builds**: Only rebuilds changed files using git diff
- **Smart environment handling**: Automatically activates `geoAI` conda environment
- **Progress tracking**: Detailed build progress with timing and ETA
- **Cleaning capabilities**: Removes intermediate files to prevent conflicts

### Content Types
- **`.qmd` files**: Quarto markdown for most course content
- **`.ipynb` files**: Jupyter notebooks for interactive lessons
- **Generated data**: Python scripts create sample datasets for exercises
- **Static assets**: Images, CSS, and other web resources

## Development Workflow

### Standard Content Updates
1. Edit `.qmd` or `.ipynb` files in `book/chapters/` or `book/extras/`
2. Run `cd book && python build_docs.py --serve` for quick preview
3. Review changes in browser
4. Commit and push changes

### Major Changes or Troubleshooting
1. Run `cd book && python build_docs.py --clean --full` for complete rebuild
2. Check `docs/` directory for correct output
3. Verify all course materials render properly

### Environment Requirements
- **Python 3.11+** with pandas, numpy, matplotlib, seaborn, jupyter, jupyterlab
- **Quarto** for static site generation
- **Git** for incremental build detection
- **Conda/Mamba** for environment management

## Prerequisites and Requirements

**For GEOG 288KC students should have**:
- Experience with remote sensing, geospatial data, or ML (Python, Earth Engine, PyTorch)
- A defined project interest area for applying Geospatial Foundation Models
- Access to UCSB AI Sandbox for computational resources

**Technical Requirements**:
- **Python 3.11+** with PyTorch, Earth Engine, TorchGeo, and ML libraries
- **Quarto** for static site generation (if building documentation)
- **Git** for version control and incremental builds
- **Conda/Mamba** for environment management
- **CUDA-capable GPU** recommended for model training (available via UCSB AI Sandbox)

## Development Approach

**Content Development Strategy**:
- Week-by-week content development following the syllabus timeline (Weeks 0-10)
- Project-centric approach with deliverable milestones (Proposals, MVPs, Final Presentations)
- Integration with UCSB AI Sandbox computational environment
- Focus on hands-on labs with geospatial foundation models, Earth Engine, and TorchGeo
- Maintain code visibility in chapter content (no code folding for instructional materials)

## Important Notes

- The `book/` directory contains all course materials organized for GEOG 288KC
- The build system (`build_docs.py`) handles environment activation and incremental builds
- Course content focuses on geospatial foundation models, Earth Engine, TorchGeo, and cloud-based analysis
- Projects are central - students work on applied GFM projects throughout the semester with specific deliverable deadlines
- Course uses UCSB AI Sandbox for computational resources
- Assessment is pass/fail based on attendance, participation, and deliverable submission
- Students may optionally submit projects to Hugging Face or GitHub for broader visibility
- Chapter content should have code blocks visible (no code folding) for better instructional clarity

# CRITICAL CODE QUALITY RULES

**ALWAYS USE THE SIMPLEST SOLUTION** - Not "sometimes" or "usually" - ALWAYS.

**FORBIDDEN PATTERNS IN COURSE MATERIALS:**
- ❌ Code-as-strings: `code = '''def func(): pass'''`
- ❌ Dynamic code execution: `exec()`, `eval()`, `compile()`
- ❌ String-based code generation: Building functions with f.write() calls
- ❌ Template-based code creation: Any programmatic function generation
- ❌ "Clever" solutions that obscure the educational point

**REQUIRED PATTERNS:**
- ✅ Define functions directly in code blocks
- ✅ Write the actual implementation, not generated code
- ✅ Use the most obvious, direct approach
- ✅ Show students clean, simple examples they can understand immediately

**QUALITY CHECK:** Before writing any code, ask:
1. Is this the simplest possible solution?
2. Can a student read and understand this immediately?
3. Am I generating code instead of writing it directly?

If the answer to #3 is yes, STOP and rewrite using direct function definitions.

**THE GOLDEN RULE:** If you need a function, just write the function. Never generate it.
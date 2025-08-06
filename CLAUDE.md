# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **geoAI** repository for developing **GEOG 288KC: Geospatial Foundation Models and Applications** (Fall 2025). The repository contains a template structure and build system for creating educational materials on state-of-the-art geospatial foundation models (GFMs) for remote sensing and environmental monitoring.

**Repository Structure**:
- **`example_course/`**: Complete template structure from EDS 217 showing how to organize course materials using Quarto
- **Target**: Adapt this template to create GEOG 288KC course materials following the syllabus structure

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
python example_course/build_docs.py

# Full rebuild of all files
python example_course/build_docs.py --full

# Build and serve locally for preview
python example_course/build_docs.py --serve

# Clean intermediate files (HTML, _files/, etc.) from course materials
python example_course/build_docs.py --clean

# Clean and full rebuild
python example_course/build_docs.py --clean --full
```

### Environment Management
```bash
# Create the conda environment for the course
cd example_course
conda env create -f environment.yml

# Or for faster installation with mamba
mamba env create -f environment.yml

# Activate the environment (build script does this automatically)
conda activate eds217_2025
```

### Manual Quarto Commands
```bash
cd example_course
quarto render                    # Build entire site
quarto preview                   # Build and serve locally
quarto render specific_file.qmd  # Build specific file
```

## Project Architecture

### High-Level Structure
- **`example_course/`**: Complete Python course materials built with Quarto
- **Root level**: Container repository for geoAI educational content

### Template Structure (example_course/)
The template provides a complete organizational framework that should be adapted for GEOG 288KC:

**Current Template Structure**:
- **`course-materials/`**: All teaching content organized by type:
  - `day1.qmd` through `day9.qmd`: Daily lesson plans and overviews
  - `interactive-sessions/`: Live coding sessions and demonstrations  
  - `coding-colabs/`: Collaborative coding exercises
  - `lectures/`: Formal lecture content (.ipynb and .qmd files)
  - `live-coding/`: Real-time coding demonstrations with student data
  - `eod-practice/`: End-of-day practice exercises
  - `answer-keys/`: Solution files for exercises
  - `cheatsheets/`: Quick reference materials
- **`docs/`**: Generated static website (GitHub Pages output)
- **`nbs/`**: Additional Jupyter notebooks (excluded from builds)
- **`data/`**: Course datasets and data generation scripts
- **`images/`**: Course imagery and visual assets

**Adaptation for GEOG 288KC**:
- Replace daily structure with 10 weekly modules (Week 0-10)
- Focus content on geospatial foundation models instead of Python fundamentals
- Maintain project-driven approach with deliverable tracking
- Include resources for Earth Engine, TorchGeo, and UCSB AI Sandbox

### Build System
The course uses a sophisticated Python-based build system (`build_docs.py`) with:
- **Incremental builds**: Only rebuilds changed files using git diff
- **Smart environment handling**: Automatically activates `eds217_2025` conda environment
- **Progress tracking**: Detailed build progress with timing and ETA
- **Cleaning capabilities**: Removes intermediate files to prevent conflicts

### Content Types
- **`.qmd` files**: Quarto markdown for most course content
- **`.ipynb` files**: Jupyter notebooks for interactive lessons
- **Generated data**: Python scripts create sample datasets for exercises
- **Static assets**: Images, CSS, and other web resources

## Development Workflow

### Standard Content Updates
1. Edit `.qmd` or `.ipynb` files in `example_course/course-materials/`
2. Run `python build_docs.py --serve` for quick preview
3. Review changes in browser
4. Commit and push changes

### Major Changes or Troubleshooting
1. Run `python build_docs.py --clean --full` for complete rebuild
2. Check `docs/` directory for correct output
3. Verify all course materials render properly

### Environment Requirements
- **Python 3.11** with pandas, numpy, matplotlib, seaborn, jupyter, jupyterlab
- **Quarto** for static site generation
- **Git** for incremental build detection
- **Conda/Mamba** for environment management

## Prerequisites and Requirements

**For GEOG 288KC students should have**:
- Experience with remote sensing, geospatial data, or ML (Python, Earth Engine, PyTorch)
- A defined project interest area for applying Geospatial Foundation Models
- Access to UCSB AI Sandbox for computational resources

**Technical Requirements**:
- **Python 3.11** with pandas, numpy, matplotlib, seaborn, jupyter, jupyterlab (for example_course materials)
- **Quarto** for static site generation (if building documentation)
- **Git** for version control and incremental builds
- **Conda/Mamba** for environment management
- **PyTorch, Earth Engine, TorchGeo** likely needed for main course work

## Development Approach

**Template-Based Development**:
- Use `example_course/` as the structural template for building GEOG 288KC materials
- The sophisticated build system and Quarto setup can be directly reused
- Adapt content organization from daily lessons to weekly modules
- Maintain the same development workflow and build commands

**Content Development Strategy**:
- Week-by-week content development following the syllabus timeline
- Project-centric approach with deliverable milestones (Proposals, MVPs, Final Presentations)
- Integration with UCSB AI Sandbox computational environment
- Focus on hands-on labs with geospatial foundation models

## Important Notes

- The `example_course/` serves as a complete template for organizing and building GEOG 288KC course materials
- The build system (`build_docs.py`) can be reused as-is for the new course
- Course content should focus on geospatial foundation models, Earth Engine, TorchGeo, and cloud-based analysis
- Projects are central - students work on applied GFM projects throughout the semester with specific deliverable deadlines
- Course uses UCSB AI Sandbox for computational resources
- Assessment is pass/fail based on attendance, participation, and deliverable submission
- Students may optionally submit projects to Hugging Face or GitHub for broader visibility
# Quick Start Guide for Course Authors

*A practical guide for instructional staff to edit GEOG 288KC course materials*

## 🚀 TLDR - Get Started in 3 Steps

1. **Setup Environment**: `make setup` (installs everything you need)
2. **Edit Content**: Modify `.qmd` files in `book/chapters/` and `book/extras/`
3. **Preview Changes**: `make preview` (builds and opens in browser)

Need help? Check [TROUBLESHOOTING.md](installation/TROUBLESHOOTING.md) or ask on Slack.

---

## 🎯 What You Need to Know

### Our Build System (The Magic)
- **Quarto** converts `.qmd` files → HTML website
- **Tangle filter** extracts code from `.qmd` → actual Python files in `geogfm/`  
- **Build pipeline** processes everything → `docs/` folder for GitHub Pages

**Why this matters**: When you edit course content, you're simultaneously:
1. Creating instructional material (HTML pages)
2. Building the actual Python package (`geogfm/`) students will use

### Key Concepts for Non-Quarto Users
- **`.qmd` files** = Markdown + executable Python code blocks
- **Code blocks with `#| tangle:`** = get extracted to real Python files
- **Rendered site** = lives in `docs/` and gets pushed to GitHub Pages
- **`make` commands** = our shortcuts for common tasks

---

## 📁 File Structure You Care About

```
book/
├── chapters/           # Weekly sessions (c01-*, c02-*, etc.)
├── extras/
│   ├── cheatsheets/   # Quick reference materials  
│   ├── examples/      # Practical examples
│   └── projects/      # Project templates
└── _quarto.yml        # Site configuration
```

**Most of your edits happen in `book/chapters/`**

---

## ⚡ Essential Commands

```bash
# First time setup
make setup              # Install conda env + register Jupyter kernel

# Daily workflow  
make preview            # Build + serve locally (auto-refreshes)
make docs              # Quick build (only changed files)
make docs-full         # Complete rebuild (when things break)

# Troubleshooting
make clean             # Clear cache/temp files
make kernelspec        # Fix Jupyter kernel issues
```

**Pro tip**: Use `make preview` while editing - it rebuilds automatically when you save files.

---

## ✏️ Editing Course Content

### Basic .qmd Structure
```markdown
---
title: "Your Session Title"  
subtitle: "Week N: Specific Topic"
jupyter: geoai
---

## Section Heading

Regular markdown text, explanations, etc.

### Code Block → `geogfm/some/module.py`

```{python}
#| tangle: geogfm/some/module.py
# This code gets extracted to the actual Python file
def some_function():
    return "Hello from geogfm package!"
```

More explanation text...
```

### The Tangle System (Code Extraction)

**What it does**: Code blocks with `#| tangle:` directives get written to real Python files.

**Why it's awesome**: Students see the instructional content AND get a working Python package built from the same source.

**How to use it**:
```python
# This creates/overwrites geogfm/data/loaders.py
#| tangle: geogfm/data/loaders.py

# This appends to the same file  
#| tangle: geogfm/data/loaders.py
#| mode: append
```

**Section headings**: Always include the target file path in your headings:
```markdown
### Data Loaders → `geogfm/data/loaders.py`
```
This helps students understand what files they're building.

---

## 🎨 Content Best Practices

### Do ✅
- **Small, focused code blocks** - one concept per block
- **Clear section headings** with file paths (e.g., `### Attention Module → geogfm/modules/attention.py`)
- **Executable examples** that actually work
- **Fixed random seeds** for reproducible outputs
- **Relative paths** (`../../data/sample.tif`)

### Avoid ❌
- **Huge code blocks** - break them up
- **Absolute paths** (`/Users/yourname/...`)
- **Time-dependent content** - no "today" or "this week"
- **Installation commands** in code blocks - use environment setup instead

### Reproducible Code Pattern
```python
# Always start with this for consistent results
import numpy as np
import torch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
```

---

## 🐛 Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| "Kernel not found" | `make kernelspec` |
| Build fails | `make clean && make docs-full` |
| Code imports fail | `pip install -e .` (in repo root) |
| Preview not updating | Check for syntax errors in your `.qmd` |
| Tangle not working | Verify `#\| tangle:` syntax and file paths |

---

## 🔍 Testing Your Changes

### Before Committing
1. **Local preview works**: `make preview` 
2. **Full build passes**: `make docs-full`
3. **Code imports correctly**: Test in Python/Jupyter
4. **No broken links**: Check all relative links work

### Quick Test Pattern
```bash
# Make your edits
make preview          # Check in browser
make docs-full        # Verify clean build  
git add . && git commit -m "Update session X"
```

---

## 📚 Getting Help

- **Setup Issues**: [installation/TROUBLESHOOTING.md](installation/TROUBLESHOOTING.md)
- **Quarto Basics**: [Quarto Guide](https://quarto.org/docs/guide/)
- **Build Problems**: Check `book/build_docs.py --help`
- **Tangle Filter**: See examples in existing `chapters/*.qmd` files
- **Questions**: Ask Kelly or Anna, or post in course Slack

---

## 💡 Quick Reference

### File Types
- **`.qmd`** - Course content (Quarto markdown + Python)
- **`.py`** - Generated package files (don't edit directly!)
- **`.yml`** - Configuration files

### Important Paths  
- `book/chapters/` - Your main editing area
- `geogfm/` - Generated Python package  
- `docs/` - Generated website (don't edit)
- `data/` - Sample datasets

### Make Commands
- `make setup` - Initial setup
- `make preview` - Edit + preview workflow  
- `make docs` - Quick build
- `make clean` - Fix problems

**Remember**: The build system handles the complexity. Focus on creating clear, executable educational content!
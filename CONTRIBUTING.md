# Contributing to GEOG 288KC

*A practical workflow guide for instructional staff contributing course materials*

## 🚀 Quick Start for Contributors

**Never contributed before?**
1. `make setup` - Install environment
2. Create branch: `git checkout -b week-3-updates`  
3. Edit files in `book/chapters/`
4. Test: `make preview`
5. Submit PR with clear description

**Need help?** Check [AUTHORING_GUIDE.md](AUTHORING_GUIDE.md) for content editing or [TROUBLESHOOTING.md](installation/TROUBLESHOOTING.md) for technical issues.

---

## 🔄 Standard Workflow

### 1. Before You Start
```bash
git checkout main
git pull origin main
make setup                    # First time only
conda activate geoAI         # Every session
```

### 2. Create Your Branch
```bash
# Use descriptive names
git checkout -b week-4-mae-implementation
git checkout -b fix-broken-links-session-2
git checkout -b add-visualization-examples
```

### 3. Make Your Changes
- **Edit content**: Modify `.qmd` files in `book/chapters/` or `book/extras/`
- **Test locally**: `make preview` (rebuilds automatically on save)
- **Verify build**: `make docs-full` (complete rebuild to catch issues)

### 4. Before Submitting PR
**Required checklist:**
- [ ] `make preview` works without errors
- [ ] `make docs-full` builds successfully  
- [ ] All links work (click through your changes)
- [ ] Code blocks execute properly
- [ ] No absolute paths or time-dependent content

**Test your code:**
```bash
# Verify tangled code imports correctly
conda activate geoAI
python -c "import geogfm; print('✅ Package imports work')"
```

### 5. Submit Pull Request
```bash
git add .
git commit -m "Add MAE implementation to Week 4"
git push origin week-4-mae-implementation
```

**PR Description Template:**
```markdown
## Changes
- Added MAE implementation section to Session 4
- Fixed broken links in patch embedding examples
- Updated visualization code to use fixed random seeds

## Testing
- [x] Local preview works
- [x] Full build passes  
- [x] Code imports correctly

## Screenshots
[Include screenshots of any UI/content changes]
```

---

## 📋 Content Guidelines

### ✅ Good Practices
- **Clear section headings** with file paths: `### Data Loader → geogfm/data/loaders.py`
- **Small, focused code blocks** - one concept per block
- **Executable examples** that students can run immediately
- **Reproducible outputs** (use fixed seeds: `SEED = 42`)
- **Relative paths** for data/images (`../../data/sample.tif`)

### ❌ Things to Avoid  
- **Large monolithic code blocks** (break into smaller pieces)
- **Absolute paths** (`/Users/kelly/Desktop/...` ← Never!)
- **Time references** ("this week", "today", "last Friday")
- **Installation commands** in content (use environment instead)
- **Editing generated files** (anything in `geogfm/` or `docs/`)

### Content Structure Pattern
```markdown
---
title: "Session Title"
subtitle: "Week N: Specific Focus"  
jupyter: geoai
---

## Overview
Brief introduction to what students will learn.

### Component Name → `geogfm/path/to/file.py`

```{python}
#| tangle: geogfm/path/to/file.py
# Clear, focused code that demonstrates one concept
def example_function():
    return "Students can understand this!"
```

Explanation of what the code does and why it matters.
```

---

## 🏗️ Understanding Our Build System

### The Big Picture
1. **Content** (`.qmd` files) contains instructional material + code
2. **Tangle filter** extracts code → creates Python package (`geogfm/`)
3. **Quarto** renders content → creates website (`docs/`)
4. **GitHub Pages** serves website to students

### Why This Matters
- Students get both **learning materials** (website) and **working code** (Python package)
- Everything stays in sync automatically
- One source of truth for content and implementation

### File Flow
```
book/chapters/c01-*.qmd  →  [Quarto + Tangle]  →  docs/chapters/c01-*.html
                        →                      →  geogfm/data/loaders.py
                        →                      →  geogfm/models/vit.py
```

---

## 🐛 Troubleshooting Common Issues

### Build Fails
```bash
# Nuclear option - start fresh
make clean
make docs-full

# If still failing
conda activate geoAI
pip install -e .
make kernelspec
```

### Kernel/Import Errors
```bash
# Fix Jupyter kernel binding
make kernelspec

# Ensure package is editable-installed
pip install -e .
```

### Preview Not Updating
- Check for syntax errors in your `.qmd` files
- Look for broken tangle paths (`#| tangle: wrong/path.py`)
- Verify all Python imports work

### Git Issues
```bash
# Your branch is behind main
git checkout main
git pull origin main
git checkout your-branch
git merge main

# Resolve conflicts, then continue
```

---

## 📝 PR Guidelines

### Good PR Titles
- `[Week 4] Add MAE pretraining implementation`
- `[Session 2] Fix broken attention mechanism examples`  
- `[Infra] Update environment dependencies`
- `[Docs] Clarify tangle usage in authoring guide`

### What to Include
- **Clear description** of what changed
- **Screenshots** for visual/content changes
- **Testing notes** ("Verified full build works")
- **Context** if fixing bugs or addressing issues

### Review Process
1. **Automated checks** must pass (build succeeds)
2. **Manual review** by Kelly/Anna 
3. **Address feedback** promptly
4. **Squash merge** after approval

---

## 🎯 Types of Contributions

### Content Updates (Most Common)
- Adding new examples or explanations
- Fixing typos, broken links, or errors
- Improving code clarity or adding comments
- Updating visualizations or plots

### Infrastructure Changes  
- Environment/dependency updates
- Build system improvements
- New make targets or automation
- Documentation improvements

### Major Content Additions
- New weekly sessions or sections
- Additional cheatsheets or examples
- Project templates or resources
- Coordinate with Kelly/Anna first

---

## 💡 Tips for Success

### Efficient Workflow
- Use `make preview` while editing (auto-rebuilds)
- Keep browser dev tools open to catch errors
- Test imports in Python/Jupyter before committing
- Make small, focused commits with clear messages

### Communication
- **Ask questions early** rather than struggling
- **Share screenshots** when describing visual issues  
- **Tag relevant people** in PR discussions
- **Be specific** about what's not working

### Quality Checks
```bash
# Before every PR
make docs-full                    # Verify clean build
python -c "import geogfm"        # Test package imports  
grep -r "/Users\|/home" book/    # Check for absolute paths
```

---

## 📚 Resources

- **Content Editing**: [AUTHORING_GUIDE.md](AUTHORING_GUIDE.md)
- **Technical Issues**: [installation/TROUBLESHOOTING.md](installation/TROUBLESHOOTING.md)
- **Quarto Docs**: [quarto.org/docs/guide](https://quarto.org/docs/guide/)
- **Course Questions**: Kelly, Anna, or course Slack
- **Build System**: `book/build_docs.py --help`

---

## ⚡ Command Cheat Sheet

```bash
# Setup (first time)
make setup

# Daily workflow
conda activate geoAI
git checkout -b my-feature-branch
make preview                    # Edit and preview
make docs-full                  # Verify before PR

# Troubleshooting  
make clean
make kernelspec
pip install -e .

# Submit changes
git add . && git commit -m "Clear description"
git push origin my-feature-branch
# Create PR on GitHub
```

**Remember**: Focus on creating clear, executable educational content. The build system handles the technical complexity!
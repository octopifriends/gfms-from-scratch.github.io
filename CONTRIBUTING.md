# Contributing to GEOG 288KC (geoAI)

Thank you for helping build and maintain GEOG 288KC: Building Geospatial Foundation Models. This document defines the rules and best practices for contributing course materials, code, and infrastructure.

The key goals are: high-quality instructional content, reproducible builds, and a smooth developer workflow that matches our Quarto setup and AI Sandbox environment.

## Table of Contents
- Scope and Roles
- Branching and PR Workflow
- Content Authoring Guidelines (Quarto)
- Code Style and Linting
- Data and Assets
- Environment and Builds
- File/Path Conventions
- Review Checklist
- FAQ

## Scope and Roles
- Instructors maintain course structure, learning outcomes, and major content edits
- Contributors may add lessons, examples, fixes, or improvements following these rules
- GRIT/IT focuses on AI Sandbox setup and performance; coordinate via installation/ updates

## Branching and PR Workflow
1. Create a feature branch: `git checkout -b feature/<short-description>`
2. Make focused edits; keep diffs small and scoped
3. Build locally (see Environment and Builds). Fix all errors before PR
4. Open a PR against `main` with a concise summary and screenshots (if UI changes)
5. Request review from maintainers (Kelly, Anna). Address feedback promptly
6. Squash-merge after approval; avoid force-pushing to `main`

Naming PRs: `[Week N] <topic>`, `[Session N] <topic>`, `[Infra] build_docs sync`, `[Fix] broken link`

## Content Authoring Guidelines (Quarto)
- Only include executable code blocks for functional, instructive examples; narrative demos or pseudo-code should use Quarto callouts/tables, not code fences
- Use `qmd` for lessons; organize by `chapters/` structure
- Use callouts (`note`, `tip`, `warning`, `important`) to convey narrative content
- Keep code chunks minimal and runnable; prefer smaller, composable examples over monoliths
- Avoid time/scheduling in weekly docs; class meets Fridays 9–12; office hours 2–5 (only in syllabus)
- Use relative links within `chapters/`; prefer root `/images/...` for images that are shared
- Do not reference external frameworks by name repeatedly; keep content course-native

Quarto front-matter tips:
- Provide clear `title` and `subtitle`
- Set `code-fold: true|show` thoughtfully
- Prefer `toc: true` and `toc-depth: 3` for longer pages

## Code Style and Linting
- Python: write clear, readable code; follow PEP8 and repo’s explicit patterns
- Prefer explicit names (no 1–2 char vars); small functions; early returns
- Keep notebooks minimal; prefer `qmd` with Python chunks
- Validate imports and runtime paths locally before committing

## Data and Assets
- Place reusable assets in `images/` and small sample datasets in `data/`
- For new example data, add a brief `data/README.md` entry with source and license
- Use raw GitHub URLs for small static assets when necessary
- Do not commit large datasets or model weights to the repo

## Environment and Builds
- Local dev: `conda activate geoAI` (environment from `environment.yml`)
- Install the package for development (editable): `pip install -e .`
- Optional: register the Jupyter kernel used by Quarto: `make kernelspec` (kernel name `geoai`)
- Build site locally: `make docs` (incremental) or `make docs-full` (clean build)
- Preview locally: `make preview`
- When running full builds, caches are cleared to avoid freeze errors
- The build process must mirror `_quarto.yml` render rules. Exclusions include:
  - `nbs/`, `installation/`, `example_course/`, `LLMs-from-scratch/`
  - Non-course internal docs like `COURSE_*.md`, `*_PLAN.md`
- Never render or link to excluded folders in Quarto navigation

### Rationale: conda + editable install
- Heavy geospatial/DL deps (GDAL/PROJ/PyTorch) are installed via conda to ensure binary compatibility
- `pyproject.toml` uses setuptools with empty `dependencies` to avoid conflicts with conda
- `pip install -e .` makes code tangled into `geogfm/` immediately importable during Quarto execution

## File/Path Conventions
- Weekly lessons: `chapters/c0N-*.qmd` (Week N) with Stage title and Week subtitle
- Interactive sessions: within `chapters/` as weekly files (c0N-*.qmd)
- Lectures: `chapters/extras/lectures/lectureN_<topic>.qmd`
- Projects/templates: `chapters/extras/projects/`
- Images: `images/`; reference as `images/<file>` in docs
- Sample data: `data/`; reference via relative path or raw GitHub URL

## Review Checklist
Before submitting a PR:
- Content
  - [ ] Titles/subtitles match course structure
  - [ ] Executable code is minimal, functional, and necessary for instruction
  - [ ] Narrative is in callouts/tables, not code blocks
- Links & Paths
  - [ ] All links resolve; images render
  - [ ] Paths are relative and respect repo structure
- Build
  - [ ] Local build passes: `python build_docs.py --serve` (or `--full`)
  - [ ] No content from excluded directories is rendered
- Style
  - [ ] Variable names and functions are descriptive
  - [ ] No scheduling times in weekly docs; timing only in syllabus

## FAQ
- Q: Where do I put new rules or guidelines?
  - A: Here in `CONTRIBUTING.md`; instructional-team-only docs in `COURSE_*.md` (excluded from site)
- Q: Can I add new datasets?
  - A: Yes; keep them small, documented in `data/README.md`, and attribute sources
- Q: How do I add a new interactive session?
  - A: Create a new weekly page under `chapters/` (e.g., `chapters/c0N-<topic>.qmd`) and link it from `_quarto.yml` navigation

Thanks for contributing to a high-clarity, reproducible course experience!

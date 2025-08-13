# Authoring Guide for GEOG 288KC (geoAI)

This guide outlines best practices and code patterns for developing interactive sessions, cheatsheets, lectures, and other chapters in this repository.

Goals: high-clarity pedagogy, reproducible execution, fast local rendering, and consistent outputs across environments (AI Sandbox, local laptops, CI).

## Contents
- Content types and structure
- Quarto authoring guidelines
- Deterministic and reproducible code
- Data handling and sample datasets
- Device and environment patterns (CPU/MPS/CUDA)
- Visualization and logging
- Performance and execution time
- Common pitfalls and anti-patterns

---

## Content Types and Structure

- Interactive sessions (hands-on): weekly pages under `chapters/` (e.g., `chapters/c0N-<topic>.qmd`)
  - Sections: Introduction → Setup → Data → Core tasks → Wrap-up
  - Include runnable, minimal examples with clear printed outputs
- Cheatsheets: `chapters/extras/cheatsheets/<topic>.qmd`
  - Quick reference, copy-paste friendly; prioritize short code blocks and tables
- Weekly materials: `chapters/c0N-*.qmd`
  - Link to the corresponding interactive session; no timing/scheduling
- Lectures: `chapters/extras/lectures/lectureN_<topic>.qmd`
  - Conceptual content; minimal execution; use figures/tables/callouts

---

## Quarto Authoring Guidelines

- Executable code blocks are for functional, instructive code only; use callouts/tables for narrative or pseudo-code
- Prefer:
  - `toc: true`, `toc-depth: 3` on longer pages
  - `code-fold: true|show` based on context
- Use callouts for emphasis:
  - `::: {.callout-note}`, `.tip`, `.warning`, `.important`
- Keep code chunks small and focused; avoid long, monolithic cells
- Avoid embedding meeting times/schedules in weekly docs; timing lives only in the syllabus
- Follow site navigation and file organization; don’t craft ad‑hoc menus inside pages

### Tangling code to repo files (literate export)

To keep pedagogy first while avoiding copy‑paste into `geogfm/`, you can export code blocks from `.qmd` during render using the `tangle` filter (already enabled in `book/_quarto.yml`).

Two ways to opt‑in per code chunk:

1) Attributes in the fence header (preferred for clarity):

````
```{python tangle="../geogfm/data/tiles.py" tangle-append="true"}
from pathlib import Path

def tiles_from_aoi(aoi_path: str, patch_size: int):
    ...
```
````

2) Hash‑pipe directives inside the cell:

```python
#| tangle: ../geogfm/data/tiles.py
#| tangle-append: true
from pathlib import Path

def tiles_from_aoi(aoi_path: str, patch_size: int):
    ...
```

Options:
- `tangle`: relative path (from the current `.qmd` directory)
- `tangle-append`: `true|false` (append vs overwrite)
- `tangle-marker`: custom first-line header comment

Best practices:
- Keep exported cells small and composable; mirror import boundaries you’d expect in modules
- Use a reset block (no append) first, then append in subsequent blocks for stable ordering
- Printed outputs in the page should remain minimal; exports are independent of visible output
- Never tangle to absolute paths; keep targets within this repo (e.g., `../geogfm/...`)

---

## Deterministic and Reproducible Code

Always write code to generate consistent output across environments:

- Set fixed seeds at the start of executable sections:

```python
# Reproducibility seeds
import os, random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # Enable (best-effort) determinism where feasible
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
except ImportError:
    pass
```

- Avoid time-dependent randomness; do not rely on clock or dataset order without fixing seeds
- For demos that rely on randomness (e.g., masks), show the seed and print key shapes/summaries
- If using parallel data loaders, fix `num_workers` or document differences

---

## Data Handling and Sample Datasets

- Prefer small sample data in `data/` and load via `Path("../../data")` from within chapters
- If downloading, use stable URLs (e.g., raw GitHub) and cache to `data/`:

```python
from pathlib import Path
import urllib.request

data_dir = Path("../../data")
data_dir.mkdir(exist_ok=True)
url = "https://raw.githubusercontent.com/kellycaylor/geoAI/main/data/landcover_sample.tif"
sample_file = data_dir / "landcover_sample.tif"
if not sample_file.exists():
    urllib.request.urlretrieve(url, sample_file)
```

- When realistic data is too large/unavailable, generate synthetic but plausible data and document the assumptions
- Never write outside the repo or to absolute paths; keep all outputs ephemeral or under `data/` (small) or `docs/` (build targets)
- Attribute data sources in `data/README.md` when adding new assets

---

## Device and Environment Patterns (CPU/MPS/CUDA)

Make code run anywhere; degrade gracefully:

```python
try:
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except ImportError:
    device = "cpu"

print(f"Using device: {device}")
```

- Avoid hard failures on missing GPU; reduce batch sizes when on CPU
- Keep memory usage low; prefer small tensors/examples in executed cells

---

## Visualization and Logging

- Fix figure sizes and styles for consistent rendering:

```python
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (6, 4), "figure.dpi": 110})
```

- Print concise summaries (shapes, dtypes, min/max) instead of entire arrays
- Use deterministic colormaps and limits when appropriate (e.g., `vmin`, `vmax`)

---

## Performance and Execution Time

- Keep individual chunk runtime under ~10–15 seconds where possible
- Prefer precomputed constants or small subsamples for demonstration
- Avoid network calls inside many separate chunks; download once then reuse
- Do not spawn background jobs or servers in executed chunks within course pages

---

## Common Pitfalls and Anti‑patterns

- Hard‑coded absolute paths (e.g., `/Users/...`) — use `Path` and repo‑relative paths
- Non‑deterministic examples — always fix seeds and parameters
- Large outputs in the page — summarize; hide verbose logs
- Installing packages at render time — rely on environment
- Writing artifacts outside `docs/` during render — let Quarto manage outputs
- Ignoring site excludes — builds must respect `_quarto.yml` render rules

---

## Quick Checklist (per page)

- [ ] Titles/subtitles match course structure
- [ ] Functional code only in executable chunks; narrative in callouts/tables
- [ ] Seeds set; outputs deterministic across runs
- [ ] Data loaded from `data/` or generated reproducibly
- [ ] Device handling works on CPU-only machines
- [ ] Figures have fixed sizes and reproducible appearance
- [ ] Chunk runtimes reasonable; no heavy downloads in loops
- [ ] Links and paths are relative; images resolve


Happy authoring! Keep materials clear, minimal, and reproducible.

#!/usr/bin/env python3
"""
Bootstrap renderer for foundational library code.

Renders a prescribed list of Quarto pages in order so that code blocks
are executed and tangled into the `geogfm/` package before other pages
that import these modules are rendered.

Assumptions:
- Run from the `book/` directory (same as build_docs.py), or provide
  the `--book-dir` argument to change working directory.
- Quarto is installed and available on PATH.
- The active environment is already set (e.g., `conda activate geoAI`).
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List


# Ordered list of pages that tangle foundational library modules
BOOTSTRAP_PAGES: List[str] = [
    # Week 1: core, data, transforms
    "chapters/c01-geospatial-data-foundations.qmd",
    # Week 2: embeddings, attention, blocks
    "chapters/c02-spatial-temporal-attention-mechanisms.qmd",
    # Week 3: models/gfm_vit.py + reconstruction head
    "chapters/c03-complete-gfm-architecture.qmd",
    # Week 4: models/gfm_mae.py + mae_loss used by quick-start and later pages
    "chapters/c04-pretraining-implementation.qmd",
]


def run_quarto_render(qmd_path: str) -> None:
    try:
        # Stream Quarto output directly so errors are visible
        subprocess.run(["quarto", "render", qmd_path], check=True)
        print(f"✅ Rendered {qmd_path}")
    except subprocess.CalledProcessError as e:
        # Provide a clearer message; Quarto error details will have already printed
        raise SystemExit(f"❌ Failed rendering {qmd_path} (exit code {e.returncode})")


def bootstrap(book_dir: Path, pages: List[str] | None = None) -> None:
    if not (book_dir / "_quarto.yml").exists():
        raise FileNotFoundError(f"_quarto.yml not found in {book_dir}")

    ordered_pages = pages or BOOTSTRAP_PAGES
    print("📚 Bootstrapping library pages in order:")
    for p in ordered_pages:
        print(f" - {p}")

    cwd = Path.cwd()
    try:
        os.chdir(book_dir)
        # Ensure destination package paths exist prior to tangling
        repo_root = Path(__file__).resolve().parents[2]
        (repo_root / "geogfm" / "models").mkdir(parents=True, exist_ok=True)
        init_models = repo_root / "geogfm" / "models" / "__init__.py"
        if not init_models.exists():
            init_models.write_text("# geogfm.models\n")
        for rel_path in ordered_pages:
            print(f"🔨 Rendering {rel_path}...")
            run_quarto_render(rel_path)
        print("✅ Bootstrap complete")
    finally:
        os.chdir(cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap geogfm library by rendering pages in order")
    parser.add_argument(
        "--book-dir",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Path to the book directory containing _quarto.yml (default: book/)",
    )
    parser.add_argument(
        "--pages",
        nargs="*",
        help="Optional explicit list of .qmd pages to render in order (relative to book dir)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bootstrap(args.book_dir, pages=args.pages)



#!/usr/bin/env python3
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
BOOK_DIR = ROOT / "book"
COURSE_DIR = BOOK_DIR / "chapters"

CODE_FENCE_START_RE = re.compile(r"^```\{python(?P<attrs>[^}]*)\}\s*$")
CODE_FENCE_END_RE = re.compile(r"^```\s*$")

ATTR_RE = re.compile(r"(\w[\w-]*)\s*=\s*(?:\"([^\"]*)\"|([^\s]+))")


def parse_attrs(attr_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for m in ATTR_RE.finditer(attr_text.strip()):
        key = m.group(1)
        val = m.group(2) if m.group(2) is not None else m.group(3)
        attrs[key] = val
    return attrs


def tangle_from_file(qmd_path: Path) -> List[Tuple[Path, str, bool]]:
    """Return list of (target_path, code, append) for tangle blocks in a qmd file."""
    results: List[Tuple[Path, str, bool]] = []
    with qmd_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        m = CODE_FENCE_START_RE.match(line)
        if not m:
            i += 1
            continue
        # Check for Quarto-style directives in the next lines
        i += 1
        tangle = None
        append_flag = False
        code_lines: List[str] = []

        # Look for #| tangle: directives at the start of the code block
        while i < len(lines) and not CODE_FENCE_END_RE.match(lines[i]):
            line = lines[i]
            if line.strip().startswith("#| tangle:"):
                tangle = line.strip().split(":", 1)[1].strip()
            elif line.strip().startswith("#| tangle-append:") or line.strip().startswith("#| mode: append"):
                append_flag = True
            elif not line.strip().startswith("#|"):
                # Regular code line, add it to code_lines
                code_lines.append(line)
            i += 1

        if not tangle:
            # No tangle directive found, skip this block
            continue

        code = "".join(code_lines)
        target = (qmd_path.parent / tangle).resolve()
        # Rebase paths tangled to book/geogfm -> repo_root/geogfm
        book_geogfm = BOOK_DIR / "geogfm"
        if str(target).startswith(str(book_geogfm)):
            rel = target.relative_to(book_geogfm)
            target = (ROOT / "geogfm" / rel).resolve()
        # Ensure target is inside repo root
        try:
            target.relative_to(ROOT)
        except Exception:
            print(f"Skipping unsafe tangle path outside repo: {target}")
            continue
        results.append((target, code, append_flag))
    return results


def main() -> int:
    if not COURSE_DIR.exists():
        print(f"Chapters directory not found: {COURSE_DIR}")
        return 1

    # Collect all QMD files under chapters
    qmd_files = sorted(COURSE_DIR.rglob("*.qmd"))
    if not qmd_files:
        print("No .qmd files found to tangle.")
        return 0

    # Track which files have been initialized (for overwrite semantics)
    initialized: Dict[Path, bool] = {}

    total_blocks = 0
    total_files = set()

    for qmd in qmd_files:
        blocks = tangle_from_file(qmd)
        if not blocks:
            continue
        for target, code, append_flag in blocks:
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "a"
            header = ""
            if not append_flag or target not in initialized:
                # First write without append resets the file
                mode = "w"
                header = f"# Generated from {qmd.relative_to(ROOT)}\n"
                initialized[target] = True
            with target.open(mode, encoding="utf-8") as out:
                if header:
                    out.write(header)
                out.write(code)
                if not code.endswith("\n"):
                    out.write("\n")
            total_blocks += 1
            total_files.add(target)
            print(f"Tangled -> {target.relative_to(ROOT)} ({'append' if mode=='a' else 'write'})")

    print(f"\nTangled {total_blocks} code blocks into {len(total_files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
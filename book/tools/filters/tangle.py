#!/usr/bin/env python
import os
import io
import re
import panflute as pf
from pathlib import Path
from datetime import datetime

# Quarto-style directive lines: `#| key: value`
directive_re = re.compile(r"^\s*#\|\s*([\w-]+)\s*:\s*(.+?)\s*$")


def prepare(doc):
    """
    Initialize per-document state.
    - tangle_root: base directory for output files (default: '.')
    - tangle: aggregation dict used by 'concat' mode
    - timestamp: ISO-8601 used in the header when requested
    """
    doc.tangle_root = Path(doc.get_metadata(
        'tangle-root', default='.')).resolve()
    doc.tangle = {}
    doc.timestamp = datetime.now().isoformat(timespec='seconds')


# --- helpers -------------------------------------------------------

def parse_directives(text: str, attrs: dict) -> dict:
    """
    Merge code block attributes with any leading Quarto-style cell directives
    of the form '#| key: value'. Only the leading contiguous block of such
    lines is considered.
    """
    merged = dict(attrs)
    for line in text.splitlines():
        m = directive_re.match(line.strip())
        if m:
            merged[m.group(1).strip()] = m.group(2).strip()
        else:
            # stop at the first non-directive line
            break
    return merged


def render_header(header_text: str, ts: str) -> str:
    """
    Render the optional header text as Python comments plus a standard timestamp trailer.
    """
    commented = ""
    if header_text:
        # Prefix each line with '# '
        commented_lines = [f"# {line}" for line in header_text.splitlines() if line.strip() != ""]
        commented = "\n".join(commented_lines) + "\n"
    commented += f"# Tangled on {ts}"
    return commented.rstrip()


# --- filter core ---------------------------------------------------

def action(elem, doc):
    """
    Handle both:
    - Python CodeBlocks that contain Quarto `#|` directives at the top
    - Div-wrapped cells where Quarto has lifted directives into Div attributes
      (e.g., attributes {"tangle": "geogfm/file.py", "mode": "concat", "header": "..."})
    """
    # Case 1: Div element with tangle attributes, containing a Python CodeBlock
    if isinstance(elem, pf.Div):
        attrs = dict(elem.attributes or {})
        tangle_attr = attrs.get('tangle') or attrs.get('file')
        if tangle_attr is None:
            return
        # Find first inner Python CodeBlock
        inner_block = None
        for child in elem.content:
            if isinstance(child, pf.CodeBlock):
                classes_lower = {c.lower() for c in (child.classes or [])}
                if {"python", "py"} & classes_lower:
                    inner_block = child
                    break
        if inner_block is None:
            return

        # Resolve target path
        normalized = str(tangle_attr).strip().lstrip(os.sep)
        target = (doc.tangle_root / normalized).resolve()
        if not str(target).startswith(str(doc.tangle_root)):
            raise RuntimeError(f"Tangle path escapes project: {target}")

        mode = (attrs.get('mode') or 'concat').strip().lower()
        header_text = attrs.get('header', '')
        chunk = inner_block.text if inner_block.text.endswith('\n') else inner_block.text + '\n'

        if mode == 'append':
            target.parent.mkdir(parents=True, exist_ok=True)
            with io.open(target, 'a', encoding='utf-8') as f:
                f.write(chunk)
            return
        if mode == 'overwrite':
            target.parent.mkdir(parents=True, exist_ok=True)
            out = render_header(header_text, doc.timestamp) + "\n\n" + chunk
            with io.open(target, 'w', encoding='utf-8') as f:
                f.write(out)
            return
        # concat (default)
        entry = doc.tangle.setdefault(
            str(target),
            {
                'chunks': [],
                'header': render_header(header_text, doc.timestamp),
            },
        )
        entry['chunks'].append(chunk)
        return

    # Case 2: Plain Python CodeBlock with inline `#|` directives
    if not isinstance(elem, pf.CodeBlock):
        return

    # Determine language; only tangle Python code blocks
    classes_lower = {c.lower() for c in (elem.classes or [])}
    is_python = bool({"python", "py"} & classes_lower)
    if not is_python:
        return

    attrs = parse_directives(elem.text, elem.attributes)
    tangle_attr = attrs.get('tangle')
    file_attr = attrs.get('file')

    # Accept explicit path in `tangle:`, or allow `file:` when `tangle:` is just a flag
    tangle_target = None
    if isinstance(tangle_attr, str) and tangle_attr.strip():
        tangle_target = tangle_attr.strip()
    elif tangle_attr in ('', 'true', 'True', True) and isinstance(file_attr, str) and file_attr.strip():
        tangle_target = file_attr.strip()

    if not tangle_target:
        return

    # Normalize target path: treat leading '/' as project-relative, not filesystem root
    normalized = tangle_target.lstrip(os.sep)
    target = (doc.tangle_root / normalized).resolve()
    if not str(target).startswith(str(doc.tangle_root)):
        raise RuntimeError(f"Tangle path escapes project: {target}")

    mode = attrs.get('mode', 'concat')  # concat | append | overwrite
    header_text = attrs.get('header', '')

    # Ensure a trailing newline for clean file boundaries
    chunk = elem.text if elem.text.endswith('\n') else elem.text + '\n'

    if mode == 'append':
        target.parent.mkdir(parents=True, exist_ok=True)
        with io.open(target, 'a', encoding='utf-8') as f:
            f.write(chunk)
        return

    if mode == 'overwrite':
        target.parent.mkdir(parents=True, exist_ok=True)
        out = render_header(header_text, doc.timestamp) + "\n\n" + chunk
        with io.open(target, 'w', encoding='utf-8') as f:
            f.write(out)
        return

    # concat mode: collect to write once in finalize()
    entry = doc.tangle.setdefault(
        str(target),
        {
            'chunks': [],
            'header': render_header(header_text, doc.timestamp),
        },
    )
    entry['chunks'].append(chunk)


def finalize(doc):
    """
    Write out all files collected in 'concat' mode. We join chunks with a blank
    line between them for readability and add the header if present.
    """
    for path, info in doc.tangle.items():
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        chunks = [c.rstrip('\n') for c in info['chunks']]
        body = ("\n\n".join(chunks) + "\n") if chunks else ""
        with io.open(p, 'w', encoding='utf-8') as f:
            header = info.get('header', '').rstrip()
            if header:
                f.write(header + "\n\n")
            f.write(body)


def main(doc=None):
    return pf.run_filter(action, prepare=prepare, finalize=finalize)


if __name__ == "__main__":
    main()

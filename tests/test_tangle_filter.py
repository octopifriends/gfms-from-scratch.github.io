import importlib.util
import sys
from pathlib import Path

import pytest
pf = pytest.importorskip("panflute")


def _load_tangle_module(repo_root: Path):
    """
    Import the tangle filter module from its file path so tests don't depend
    on package/module discovery.
    """
    tangle_path = repo_root / "book" / "tools" / "filters" / "tangle.py"
    spec = importlib.util.spec_from_file_location("tangle_filter", str(tangle_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import tangle filter from {tangle_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _extract_python_blocks_from_qmd(qmd_text: str):
    """
    Minimal parser that extracts fenced Python code blocks of the form
    ```{python}
    ...
    ```
    Returns a list of raw code strings (including any leading Quarto directives).
    """
    lines = qmd_text.splitlines()
    blocks = []
    in_block = False
    current = []
    for line in lines:
        if not in_block and line.strip().startswith("```{python"):
            in_block = True
            current = []
            continue
        if in_block and line.strip() == "```":
            blocks.append("\n".join(current))
            in_block = False
            current = []
            continue
        if in_block:
            current.append(line)
    return blocks


def test_tangle_concat_from_qmd(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    tangle = _load_tangle_module(repo_root)

    # Use a temporary working directory so outputs are isolated
    monkeypatch.chdir(tmp_path)

    qmd_path = repo_root / "book" / "test_tangle.qmd"
    qmd_text = qmd_path.read_text(encoding="utf-8")
    blocks_text = _extract_python_blocks_from_qmd(qmd_text)

    # Simulate a pandoc document and run the filter
    doc = pf.Doc()  # no metadata; default tangle_root is '.' which is tmp_path now
    tangle.prepare(doc)
    for block_text in blocks_text:
        elem = pf.CodeBlock(text=block_text, classes=["python"])
        tangle.action(elem, doc)
    tangle.finalize(doc)

    # Validate the expected file and its contents
    out_file = tmp_path / "geogfm" / "test_concat.py"
    assert out_file.exists(), "Expected concatenated file was not created"
    content = out_file.read_text(encoding="utf-8")

    # Header checks
    lines = content.splitlines()
    assert lines[0] == "Test concat file"
    assert lines[1].startswith("# Tangled on ")

    # Function order and returns
    assert content.find("def part_one()") < content.find("def part_two()") < content.find("def part_three()")
    assert "return \"one\"" in content
    assert "return \"two\"" in content
    assert "return \"three\"" in content


def test_tangle_modes_append_and_overwrite(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    tangle = _load_tangle_module(repo_root)

    monkeypatch.chdir(tmp_path)

    # Build a synthetic document with explicit modes via Quarto-style directives
    append_block_1 = """
#| tangle: geogfm/test_append.py
#| mode: append
print("A")
""".strip()

    append_block_2 = """
#| tangle: geogfm/test_append.py
#| mode: append
print("B")
""".strip()

    overwrite_block_1 = """
#| tangle: geogfm/test_overwrite.py
#| mode: overwrite
#| header: Overwrite header
print("first")
""".strip()

    overwrite_block_2 = """
#| tangle: geogfm/test_overwrite.py
#| mode: overwrite
#| header: Overwrite header (new)
print("second")
""".strip()

    doc = pf.Doc()
    tangle.prepare(doc)

    # Append twice
    tangle.action(pf.CodeBlock(text=append_block_1, classes=["python"]), doc)
    tangle.action(pf.CodeBlock(text=append_block_2, classes=["python"]), doc)

    # Overwrite twice (second should replace the first)
    tangle.action(pf.CodeBlock(text=overwrite_block_1, classes=["python"]), doc)
    tangle.action(pf.CodeBlock(text=overwrite_block_2, classes=["python"]), doc)

    # Finalize to flush any concat buffers (not used here but harmless)
    tangle.finalize(doc)

    append_out = tmp_path / "geogfm" / "test_append.py"
    overwrite_out = tmp_path / "geogfm" / "test_overwrite.py"

    assert append_out.exists()
    assert overwrite_out.exists()

    append_content = append_out.read_text(encoding="utf-8")
    # Append mode writes chunks directly with no automatic header
    assert "print(\"A\")" in append_content
    assert "print(\"B\")" in append_content
    assert append_content.index("print(\"A\")") < append_content.index("print(\"B\")")

    overwrite_content = overwrite_out.read_text(encoding="utf-8")
    # Overwrite mode keeps only the second block's header and body
    assert "Overwrite header (new)" in overwrite_content
    assert "print(\"second\")" in overwrite_content
    assert "print(\"first\")" not in overwrite_content


def test_non_python_blocks_are_ignored(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    tangle = _load_tangle_module(repo_root)

    monkeypatch.chdir(tmp_path)
    doc = pf.Doc()
    tangle.prepare(doc)

    # A non-Python block should be ignored
    block = pf.CodeBlock(text="#| tangle: geogfm/ignored.txt\nprint('hi')", classes=["r"])  # not python
    tangle.action(block, doc)
    tangle.finalize(doc)

    assert not (tmp_path / "geogfm" / "ignored.txt").exists()


def test_path_escape_is_prevented(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    tangle = _load_tangle_module(repo_root)

    monkeypatch.chdir(tmp_path)
    doc = pf.Doc()
    tangle.prepare(doc)

    # Leading slash is treated as project-relative; escaping outside should error
    dangerous = pf.CodeBlock(text="#| tangle: /../../escape.py\nprint('x')", classes=["python"])
    try:
        tangle.action(dangerous, doc)
    except RuntimeError as e:
        assert "escapes project" in str(e)
    else:
        raise AssertionError("Expected RuntimeError for path escape was not raised")



def test_tangle_from_div_attributes(tmp_path, monkeypatch):
    """
    Simulate Quarto's behavior where cell-level directives (tangle/header/mode)
    are lifted into a wrapping Div around a Python CodeBlock. The CodeBlock no
    longer contains the `#|` lines. Our filter should read attributes from the
    Div and tangle the inner Python code.
    """
    repo_root = Path(__file__).resolve().parents[1]
    tangle = _load_tangle_module(repo_root)

    # Work in temp dir so outputs are isolated
    monkeypatch.chdir(tmp_path)

    # Build a Div that mimics Quarto's structure:
    # <div class="cell" data-tangle="geogfm/test_div.py" data-header="Header Text">
    #   ```python
    #   def foo():
    #       return "ok"
    #   ```
    # </div>
    code = "\n".join([
        "def foo():",
        "    return \"ok\"",
    ])
    inner_block = pf.CodeBlock(text=code, classes=["python"])  # no #| directives here
    cell_div = pf.Div(
        inner_block,
        classes=["cell"],
        attributes={
            "tangle": "geogfm/test_div.py",
            "header": "Header from Div",
            # note: rely on default mode = concat
        },
    )

    # Run filter lifecycle
    doc = pf.Doc()
    tangle.prepare(doc)
    # Current implementation only handles CodeBlock; this should fail until we add Div handling
    tangle.action(cell_div, doc)
    tangle.finalize(doc)

    out_path = tmp_path / "geogfm" / "test_div.py"
    assert out_path.exists(), "Expected file was not created from Div attributes"
    content = out_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert lines[0] == "Header from Div"
    assert lines[1].startswith("# Tangled on ")
    assert "def foo():" in content and "return \"ok\"" in content


def test_div_two_different_targets_created(tmp_path, monkeypatch):
    """Two Div cells, each with different tangle targets, both files created."""
    repo_root = Path(__file__).resolve().parents[1]
    tangle = _load_tangle_module(repo_root)

    monkeypatch.chdir(tmp_path)

    code_a = "def a():\n    return 'A'\n"
    code_b = "def b():\n    return 'B'\n"

    div_a = pf.Div(
        pf.CodeBlock(text=code_a, classes=["python"]),
        classes=["cell"],
        attributes={"tangle": "geogfm/file_a.py", "header": "Header A"},
    )
    div_b = pf.Div(
        pf.CodeBlock(text=code_b, classes=["python"]),
        classes=["cell"],
        attributes={"tangle": "geogfm/file_b.py", "header": "Header B"},
    )

    doc = pf.Doc()
    tangle.prepare(doc)
    tangle.action(div_a, doc)
    tangle.action(div_b, doc)
    tangle.finalize(doc)

    out_a = (tmp_path / "geogfm" / "file_a.py").read_text(encoding="utf-8")
    out_b = (tmp_path / "geogfm" / "file_b.py").read_text(encoding="utf-8")
    assert "Header A" in out_a and "def a():" in out_a
    assert "Header B" in out_b and "def b():" in out_b


def test_div_alternating_targets_concat(tmp_path, monkeypatch):
    """
    Alternation: A -> B -> A. Concat mode should route blocks correctly and
    merge multiple A blocks in order, and create B separately.
    """
    repo_root = Path(__file__).resolve().parents[1]
    tangle = _load_tangle_module(repo_root)

    monkeypatch.chdir(tmp_path)

    a1 = pf.Div(
        pf.CodeBlock(text="def a1():\n    return 1\n", classes=["python"]),
        classes=["cell"], attributes={"tangle": "geogfm/alt_a.py", "header": "A Header"},
    )
    b1 = pf.Div(
        pf.CodeBlock(text="def b1():\n    return 1\n", classes=["python"]),
        classes=["cell"], attributes={"tangle": "geogfm/alt_b.py", "header": "B Header"},
    )
    a2 = pf.Div(
        pf.CodeBlock(text="def a2():\n    return 2\n", classes=["python"]),
        classes=["cell"], attributes={"tangle": "geogfm/alt_a.py"},  # same file, no new header
    )

    doc = pf.Doc()
    tangle.prepare(doc)
    for elem in (a1, b1, a2):
        tangle.action(elem, doc)
    tangle.finalize(doc)

    out_a = (tmp_path / "geogfm" / "alt_a.py").read_text(encoding="utf-8")
    out_b = (tmp_path / "geogfm" / "alt_b.py").read_text(encoding="utf-8")
    # A contains both a1 then a2 in order, with single header at top
    assert out_a.splitlines()[0] == "A Header"
    assert out_a.find("def a1()") != -1 and out_a.find("def a2()") != -1
    assert out_a.find("def a1()") < out_a.find("def a2()")
    # B contains only b1 with its header
    assert out_b.splitlines()[0] == "B Header"
    assert "def b1():" in out_b


def test_div_modes_append_and_overwrite(tmp_path, monkeypatch):
    """Append writes cumulatively; overwrite keeps only last block (and header)."""
    repo_root = Path(__file__).resolve().parents[1]
    tangle = _load_tangle_module(repo_root)

    monkeypatch.chdir(tmp_path)

    # Append mode: two cells to same file
    app1 = pf.Div(
        pf.CodeBlock(text="print('A')\n", classes=["python"]),
        classes=["cell"], attributes={"tangle": "geogfm/div_append.py", "mode": "append"},
    )
    app2 = pf.Div(
        pf.CodeBlock(text="print('B')\n", classes=["python"]),
        classes=["cell"], attributes={"tangle": "geogfm/div_append.py", "mode": "append"},
    )

    # Overwrite mode: two cells to same file, different headers; second should win
    ovw1 = pf.Div(
        pf.CodeBlock(text="print('first')\n", classes=["python"]),
        classes=["cell"], attributes={"tangle": "geogfm/div_overwrite.py", "mode": "overwrite", "header": "H1"},
    )
    ovw2 = pf.Div(
        pf.CodeBlock(text="print('second')\n", classes=["python"]),
        classes=["cell"], attributes={"tangle": "geogfm/div_overwrite.py", "mode": "overwrite", "header": "H2"},
    )

    doc = pf.Doc()
    tangle.prepare(doc)
    for elem in (app1, app2, ovw1, ovw2):
        tangle.action(elem, doc)
    tangle.finalize(doc)

    append_out = (tmp_path / "geogfm" / "div_append.py").read_text(encoding="utf-8")
    overwrite_out = (tmp_path / "geogfm" / "div_overwrite.py").read_text(encoding="utf-8")

    # Append: both lines present in order; no header in append path
    assert append_out.index("print('A')") < append_out.index("print('B')")
    assert not append_out.splitlines()[0].startswith("# Tangled on")

    # Overwrite: only the second block content remains, with header H2
    lines = overwrite_out.splitlines()
    assert lines[0] == "H2"
    assert lines[1].startswith("# Tangled on ")
    assert "print('second')" in overwrite_out and "print('first')" not in overwrite_out


def test_div_ignores_div_without_python_block(tmp_path, monkeypatch):
    """Div with tangle attributes but without an inner Python CodeBlock should be ignored."""
    repo_root = Path(__file__).resolve().parents[1]
    tangle = _load_tangle_module(repo_root)

    monkeypatch.chdir(tmp_path)
    non_py_block = pf.CodeBlock(text="1 + 1\n", classes=["r"])  # not python
    div = pf.Div(non_py_block, classes=["cell"], attributes={"tangle": "geogfm/ignored.py"})

    doc = pf.Doc()
    tangle.prepare(doc)
    tangle.action(div, doc)
    tangle.finalize(doc)

    assert not (tmp_path / "geogfm" / "ignored.py").exists()


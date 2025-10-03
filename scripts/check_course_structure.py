#!/usr/bin/env python3
"""
Course structure validation script for geoAI repository.

This script validates that course materials follow proper structure:
1. Proper front matter in .qmd files
2. Valid chapter numbering
3. Consistent file naming
4. Required sections in course materials
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml


def validate_qmd_frontmatter(content: str, filepath: Path) -> List[Tuple[int, str]]:
    """Validate YAML front matter in Quarto documents."""
    errors = []

    # Check for front matter
    if not content.startswith('---\n'):
        errors.append((1, "Missing YAML front matter at start of file"))
        return errors

    # Extract front matter
    try:
        end_idx = content.find('\n---\n', 4)
        if end_idx == -1:
            errors.append((1, "Malformed YAML front matter (missing closing ---)"))
            return errors

        frontmatter_str = content[4:end_idx]
        frontmatter = yaml.safe_load(frontmatter_str)

        if not isinstance(frontmatter, dict):
            errors.append((1, "Front matter must be a YAML dictionary"))
            return errors

        # Check required fields for chapters
        if filepath.parts[-2] == 'chapters':
            required_fields = ['title', 'format']
            for field in required_fields:
                if field not in frontmatter:
                    errors.append((1, f"Missing required front matter field: {field}"))

            # Validate chapter numbering in title
            filename = filepath.stem
            if filename.startswith('c') and filename[1:3].isdigit():
                chapter_num = int(filename[1:3])
                title = frontmatter.get('title', '')
                if not re.search(rf'Chapter\s+{chapter_num:02d}|Week\s+{chapter_num}', title, re.IGNORECASE):
                    errors.append((1, f"Chapter title should include 'Chapter {chapter_num:02d}' or 'Week {chapter_num}'"))

    except yaml.YAMLError as e:
        errors.append((1, f"Invalid YAML in front matter: {e}"))

    return errors


def validate_chapter_content(content: str, filepath: Path) -> List[Tuple[int, str]]:
    """Validate chapter content structure."""
    errors = []
    lines = content.split('\n')

    # Check for learning objectives
    has_objectives = any(
        'objective' in line.lower() or 'learning goal' in line.lower()
        for line in lines
    )

    if filepath.parts[-2] == 'chapters' and not has_objectives:
        errors.append((1, "Chapter should include learning objectives"))

    # Check for proper heading structure
    heading_levels = []
    for i, line in enumerate(lines, 1):
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            heading_levels.append((i, level))

    # Validate heading hierarchy
    for i, (line_num, level) in enumerate(heading_levels):
        if i > 0:
            prev_level = heading_levels[i-1][1]
            if level > prev_level + 1:
                errors.append((
                    line_num,
                    f"Heading level {level} follows level {prev_level} (skip detected)"
                ))

    return errors


def validate_code_blocks(content: str, filepath: Path) -> List[Tuple[int, str]]:
    """Validate code blocks in course materials."""
    errors = []
    lines = content.split('\n')

    in_code_block = False
    code_block_lang = None
    code_block_start = None

    for i, line in enumerate(lines, 1):
        if line.startswith('```'):
            if not in_code_block:
                # Starting code block
                in_code_block = True
                code_block_start = i
                lang_match = re.match(r'```(\w+)', line)
                code_block_lang = lang_match.group(1) if lang_match else None

                # Check for proper language specification for Python blocks
                if code_block_lang == 'python':
                    # This is good
                    pass
                elif not code_block_lang and any(keyword in content[content.find(line):]
                                               for keyword in ['import ', 'def ', 'class ', 'print(']):
                    errors.append((i, "Python code block missing language specification"))

            else:
                # Ending code block
                in_code_block = False
                code_block_lang = None
                code_block_start = None

    if in_code_block:
        errors.append((code_block_start, "Unclosed code block"))

    return errors


def check_file(filepath: Path) -> List[Tuple[int, str]]:
    """Check a single course material file."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        return [(1, f"Failed to read file: {e}")]

    errors = []

    if filepath.suffix == '.qmd':
        errors.extend(validate_qmd_frontmatter(content, filepath))
        errors.extend(validate_chapter_content(content, filepath))
        errors.extend(validate_code_blocks(content, filepath))

    return errors


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate course material structure"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Course material files to check"
    )

    args = parser.parse_args()

    if not args.files:
        # If no files specified, check all course materials
        root = Path(__file__).parent.parent
        files = list(root.glob("book/**/*.qmd"))
        files.extend(root.glob("book/**/*.ipynb"))
    else:
        files = [Path(f) for f in args.files]

    total_errors = 0

    for filepath in files:
        if not filepath.exists():
            print(f"Warning: {filepath} does not exist")
            continue

        # Skip certain directories
        if any(skip_dir in str(filepath) for skip_dir in ['_quarto', '_site', 'docs']):
            continue

        errors = check_file(filepath)
        if errors:
            print(f"\n{filepath}:")
            for line_num, message in sorted(errors):
                print(f"  {line_num}: {message}")
                total_errors += 1

    if total_errors > 0:
        print(f"\nFound {total_errors} course structure issue(s)")
        # Don't fail commit for course structure issues, just warn
        sys.exit(0)
    else:
        print("Course structure validation passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Anti-pattern detection script for geoAI repository.

This script detects critical code anti-patterns that should block commits:
1. Code-as-strings patterns (functions defined in triple quotes)
2. Use of exec(), eval(), compile() for code generation
3. Bare except clauses
4. Missing type hints on functions
5. SQL injection vulnerabilities
6. Hardcoded secrets/credentials
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set


class AntiPatternDetector(ast.NodeVisitor):
    """AST visitor to detect code anti-patterns."""

    def __init__(self, filename: str):
        self.filename = filename
        self.errors: List[Tuple[int, str]] = []
        self.function_names: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function definitions for type hints and other issues."""
        self.function_names.add(node.name)

        # Check for missing type hints on public functions
        if not node.name.startswith('_'):
            has_return_annotation = node.returns is not None
            has_arg_annotations = any(arg.annotation is not None for arg in node.args.args)

            if not has_return_annotation and not has_arg_annotations:
                self.errors.append((
                    node.lineno,
                    f"Function '{node.name}' missing type hints"
                ))

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check async function definitions for type hints."""
        self.visit_FunctionDef(node)  # Reuse the logic

    def visit_Call(self, node: ast.Call) -> None:
        """Check for dangerous function calls."""
        if isinstance(node.func, ast.Name):
            # Check for exec(), eval(), compile()
            if node.func.id in ('exec', 'eval', 'compile'):
                self.errors.append((
                    node.lineno,
                    f"Dangerous function call: {node.func.id}()"
                ))

        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Check for bare except clauses."""
        if node.type is None:
            self.errors.append((
                node.lineno,
                "Bare except clause detected - specify exception type"
            ))

        self.generic_visit(node)


def check_code_as_strings(content: str, filename: str) -> List[Tuple[int, str]]:
    """Check for code-as-strings patterns using regex."""
    errors = []
    lines = content.split('\n')

    # Look for very specific problematic patterns only
    # Pattern 1: Multi-line strings containing complete function definitions
    multiline_func_pattern = re.compile(
        r'(code|template|script|function_def)\s*=\s*["\'].*?def\s+\w+\(.*?\):.*?["\']',
        re.DOTALL | re.IGNORECASE
    )

    for match in multiline_func_pattern.finditer(content):
        line_num = content[:match.start()].count('\n') + 1
        errors.append((
            line_num,
            "Function definition found in code string variable (code-as-string pattern)"
        ))

    # Pattern 2: exec/eval with string literals containing function definitions
    for i, line in enumerate(lines, 1):
        if re.search(r'(exec|eval)\s*\(\s*["\'].*def\s+\w+\s*\(', line):
            errors.append((
                i,
                "Dynamic function definition detected in exec/eval call"
            ))

    # Pattern 3: Triple quoted strings used for code generation (not docstrings)
    in_function = False
    brace_depth = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track if we're inside a function
        if stripped.startswith('def '):
            in_function = True
            brace_depth = 0

        # Count braces/indentation to track function scope
        if in_function:
            brace_depth += line.count('{') - line.count('}')
            if stripped and not line.startswith(' ') and not line.startswith('\t') and not stripped.startswith('#'):
                if not stripped.startswith('def ') and not stripped.startswith('class '):
                    in_function = False

        # Look for suspicious triple quotes containing functions (not at start of function)
        if in_function and '"""' in line and 'def ' in content[content.find(line):content.find(line) + 200]:
            # Check if this looks like code generation rather than documentation
            next_lines = '\n'.join(lines[i:i+5])
            if re.search(r'""".*def\s+\w+\(.*?\):.*"""', next_lines, re.DOTALL):
                if not any(word in next_lines.lower() for word in ['example', 'usage', 'note:', 'warning:', 'see also']):
                    errors.append((
                        i,
                        "Suspicious function definition in triple quotes (possible code-as-string)"
                    ))

    return errors


def check_sql_injection(content: str, filename: str) -> List[Tuple[int, str]]:
    """Check for potential SQL injection vulnerabilities."""
    errors = []
    lines = content.split('\n')

    # Common SQL injection patterns
    sql_patterns = [
        re.compile(r'execute\s*\(\s*["\'].*%s.*["\']', re.IGNORECASE),
        re.compile(r'query\s*\(\s*["\'].*%s.*["\']', re.IGNORECASE),
        re.compile(r'(SELECT|INSERT|UPDATE|DELETE).*%s', re.IGNORECASE),
        re.compile(r'(SELECT|INSERT|UPDATE|DELETE).*\+.*["\']', re.IGNORECASE),
    ]

    for i, line in enumerate(lines, 1):
        for pattern in sql_patterns:
            if pattern.search(line):
                errors.append((
                    i,
                    "Potential SQL injection vulnerability detected"
                ))
                break

    return errors


def check_hardcoded_secrets(content: str, filename: str) -> List[Tuple[int, str]]:
    """Check for hardcoded secrets and credentials."""
    errors = []
    lines = content.split('\n')

    # Patterns for common secrets
    secret_patterns = [
        (re.compile(r'password\s*=\s*["\'][^"\']{8,}["\']', re.IGNORECASE), "password"),
        (re.compile(r'api_key\s*=\s*["\'][^"\']{20,}["\']', re.IGNORECASE), "API key"),
        (re.compile(r'secret_key\s*=\s*["\'][^"\']{20,}["\']', re.IGNORECASE), "secret key"),
        (re.compile(r'token\s*=\s*["\'][^"\']{20,}["\']', re.IGNORECASE), "token"),
        (re.compile(r'aws_access_key_id\s*=\s*["\'][^"\']+["\']', re.IGNORECASE), "AWS access key"),
        (re.compile(r'aws_secret_access_key\s*=\s*["\'][^"\']+["\']', re.IGNORECASE), "AWS secret key"),
    ]

    for i, line in enumerate(lines, 1):
        for pattern, secret_type in secret_patterns:
            if pattern.search(line):
                # Skip if it's clearly a placeholder or example
                if any(placeholder in line.lower() for placeholder in
                       ['example', 'placeholder', 'your_', 'xxx', '***', 'dummy']):
                    continue

                errors.append((
                    i,
                    f"Hardcoded {secret_type} detected"
                ))

    return errors


def check_file(filepath: Path) -> List[Tuple[int, str]]:
    """Check a single Python file for anti-patterns."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        return [(1, f"Failed to read file: {e}")]

    errors = []

    # AST-based checks
    try:
        tree = ast.parse(content, filename=str(filepath))
        detector = AntiPatternDetector(str(filepath))
        detector.visit(tree)
        errors.extend(detector.errors)
    except SyntaxError as e:
        errors.append((e.lineno or 1, f"Syntax error: {e.msg}"))

    # Regex-based checks
    errors.extend(check_code_as_strings(content, str(filepath)))
    errors.extend(check_sql_injection(content, str(filepath)))
    errors.extend(check_hardcoded_secrets(content, str(filepath)))

    return errors


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect critical anti-patterns in Python code"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Python files to check"
    )
    parser.add_argument(
        "--exclude-dirs",
        default="LLMs-from-scratch,_extensions,.claude",
        help="Comma-separated list of directories to exclude"
    )

    args = parser.parse_args()

    if not args.files:
        # If no files specified, check all Python files in geogfm/
        root = Path(__file__).parent.parent
        files = list(root.glob("geogfm/**/*.py"))
        files.extend(root.glob("book/tools/**/*.py"))
        files.extend(root.glob("tests/**/*.py"))
        files.extend(root.glob("scripts/**/*.py"))
    else:
        files = []
        for f in args.files:
            path = Path(f)
            if path.is_dir():
                # If it's a directory, find all Python files in it
                files.extend(path.glob("**/*.py"))
            elif path.is_file():
                files.append(path)
            else:
                files.append(path)  # Let it fail later with a proper error

    # Filter out excluded directories
    exclude_dirs = args.exclude_dirs.split(",")
    filtered_files = []
    for file in files:
        if not any(exclude_dir in str(file) for exclude_dir in exclude_dirs):
            filtered_files.append(file)

    total_errors = 0

    for filepath in filtered_files:
        if not filepath.exists():
            print(f"Warning: {filepath} does not exist")
            continue

        errors = check_file(filepath)
        if errors:
            print(f"\n{filepath}:")
            for line_num, message in sorted(errors):
                print(f"  {line_num}: {message}")
                total_errors += 1

    if total_errors > 0:
        print(f"\nFound {total_errors} critical anti-pattern(s)")
        sys.exit(1)
    else:
        print("No critical anti-patterns detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
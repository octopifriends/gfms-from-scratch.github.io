# Pre-Commit Hooks Setup and Usage

This repository uses comprehensive pre-commit hooks to maintain code quality and prevent critical anti-patterns from being committed. The hooks are designed to catch issues early and enforce consistent coding standards across the project.

## Quick Setup

1. **Activate the environment:**
   ```bash
   conda activate geoAI
   ```

2. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

3. **Run on all files (optional):**
   ```bash
   pre-commit run --all-files
   ```

## What Gets Checked

### Critical Anti-Patterns (BLOCKS COMMIT)

These patterns will **completely block** your commit:

1. **Code-as-strings patterns** - Functions defined in triple quotes
2. **Dynamic code execution** - Use of `exec()`, `eval()`, `compile()` for code generation
3. **Bare except clauses** - `except:` without specifying exception type
4. **Missing type hints** - Public functions without type annotations
5. **SQL injection vulnerabilities** - Unsafe SQL query construction
6. **Hardcoded secrets** - API keys, passwords, tokens in code

### Quality Standards (FAILS COMMIT)

These issues will fail the commit with fixable errors:

1. **PEP 8 violations** - Code formatting issues (auto-fixed by Black)
2. **Import sorting** - Incorrect import order (auto-fixed by isort)
3. **Type checking failures** - MyPy type errors
4. **Security issues** - Bandit security warnings
5. **Missing docstrings** - Public functions without documentation
6. **Line length violations** - Lines longer than 88 characters

### File Quality Checks

1. **Trailing whitespace** - Automatically removed
2. **End-of-file issues** - Ensures newline at end
3. **Large files** - Prevents files >10MB from being committed
4. **Merge conflicts** - Detects unresolved conflicts
5. **Private keys** - Prevents accidental key commits
6. **YAML/JSON validation** - Syntax checking for config files

## Excluded Directories

The following directories are excluded from most checks:
- `LLMs-from-scratch/` - External course materials
- `_extensions/` - Quarto extensions
- `.claude/` - Claude AI configuration
- `tests/` - Test files (relaxed rules)

## Tool Configurations

### Black (Code Formatting)
- Line length: 88 characters
- Target: Python 3.11
- Profile: Compatible with other tools

### isort (Import Sorting)
- Profile: Black-compatible
- Multi-line imports with trailing commas
- Ensures consistent import organization

### MyPy (Type Checking)
- Strict mode enabled
- Requires type hints on all functions
- Ignores missing imports for external libraries

### Bandit (Security Scanning)
- Scans for common security issues
- Excludes test-related security warnings
- Focuses on production code vulnerabilities

### Flake8 (Linting)
- PEP 8 compliance checking
- Additional plugins for better code quality
- Reasonable ignore list for documentation

## Manual Usage

### Run all hooks on staged files:
```bash
pre-commit run
```

### Run specific hook:
```bash
pre-commit run black
pre-commit run mypy
pre-commit run check-antipatterns
```

### Run on all files:
```bash
pre-commit run --all-files
```

### Skip hooks (emergency use only):
```bash
git commit --no-verify -m "Emergency commit"
```

### Update hook versions:
```bash
pre-commit autoupdate
```

## Custom Scripts

### Anti-Pattern Detection (`scripts/check_antipatterns.py`)

This custom script detects critical patterns that should never appear in the codebase:

**Usage:**
```bash
python scripts/check_antipatterns.py                    # Check all relevant files
python scripts/check_antipatterns.py file1.py file2.py  # Check specific files
```

**Detected Patterns:**
- Functions defined inside triple-quoted strings
- Dynamic code execution patterns
- Bare except clauses without exception types
- Missing type hints on public functions
- SQL injection vulnerabilities
- Hardcoded secrets and credentials

### Course Structure Validation (`scripts/check_course_structure.py`)

Validates course material structure for educational content:

**Usage:**
```bash
python scripts/check_course_structure.py              # Check all course materials
python scripts/check_course_structure.py chapter.qmd  # Check specific file
```

**Validates:**
- Proper YAML front matter in .qmd files
- Consistent chapter numbering
- Required sections in course materials
- Proper code block formatting

## Troubleshooting

### Common Issues and Solutions

1. **"pre-commit command not found"**
   ```bash
   conda activate geoAI
   conda install pre-commit
   ```

2. **Black formatting conflicts with existing code**
   ```bash
   black .
   git add .
   git commit
   ```

3. **MyPy type errors**
   - Add type hints to functions
   - Use `# type: ignore` for unavoidable issues
   - Check the MyPy configuration in `pyproject.toml`

4. **Import sorting issues**
   ```bash
   isort .
   git add .
   git commit
   ```

5. **Anti-pattern script false positives**
   - Review the detected pattern carefully
   - If legitimate, consider refactoring the code
   - For false positives, update the detection script

### Skipping Specific Checks

If you need to skip a specific check for a valid reason:

```bash
# Skip a specific hook
SKIP=mypy git commit -m "Commit message"

# Skip multiple hooks
SKIP=mypy,bandit git commit -m "Commit message"

# Skip all hooks (emergency only)
git commit --no-verify -m "Emergency commit"
```

### Pre-commit in CI/CD

The configuration includes CI settings for automatic fixes:
- Runs on pull requests
- Auto-fixes formatting issues
- Updates weekly
- Skips local-only hooks

## Best Practices

1. **Run pre-commit early and often**
   - Set up hooks immediately after cloning
   - Run `pre-commit run --all-files` after major changes

2. **Keep commits focused**
   - Fix quality issues in separate commits
   - Don't mix feature changes with formatting fixes

3. **Understand the errors**
   - Read error messages carefully
   - Learn from quality tool feedback
   - Improve code quality over time

4. **Configure your editor**
   - Set up Black formatting on save
   - Enable MyPy in your IDE
   - Use isort import organization

5. **Team coordination**
   - Ensure all team members use the same setup
   - Update pre-commit configuration together
   - Discuss quality standards and exceptions

## Environment Integration

The pre-commit setup is integrated with the course environment:

- Tools are included in `environment.yml`
- Configurations are in `pyproject.toml`
- Scripts work with the conda environment
- Compatible with Jupyter notebooks and course materials

## Support

For issues with pre-commit hooks:

1. Check this documentation first
2. Review the error messages carefully
3. Consult tool-specific documentation
4. Ask for help with specific error messages

The goal is to maintain high code quality while keeping the development process smooth and educational for course participants.
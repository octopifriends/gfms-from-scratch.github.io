# Pre-Commit Quick Start

## Setup (One-time)

```bash
# 1. Activate environment
conda activate geoAI

# 2. Run setup script
./scripts/setup_precommit.sh
```

## Daily Usage

Pre-commit hooks run automatically on `git commit`. They will:

- **BLOCK commits** with critical anti-patterns (exec/eval, bare except, missing type hints, SQL injection, hardcoded secrets)
- **AUTO-FIX** formatting issues (Black, isort)
- **FAIL commits** with quality issues (MyPy errors, Bandit security warnings, linting violations)

## Manual Testing

```bash
# Test before committing
pre-commit run

# Test all files
pre-commit run --all-files

# Test specific tool
pre-commit run black
pre-commit run mypy
```

## Emergency Override

```bash
# Skip all hooks (use sparingly)
git commit --no-verify -m "Emergency commit"

# Skip specific hook
SKIP=mypy git commit -m "Skip type checking"
```

## Full Documentation

See `docs/PRE_COMMIT_SETUP.md` for complete setup instructions, troubleshooting, and tool configurations.
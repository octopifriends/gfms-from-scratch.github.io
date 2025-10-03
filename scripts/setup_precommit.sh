#!/bin/bash
# Pre-commit setup script for geoAI repository
# This script sets up pre-commit hooks and validates the installation

set -e  # Exit on error

echo "🔧 Setting up pre-commit hooks for geoAI repository..."

# Check if we're in the right directory
if [ ! -f "environment.yml" ] || [ ! -f ".pre-commit-config.yaml" ]; then
    echo "❌ Error: Run this script from the root of the geoAI repository"
    exit 1
fi

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "geoAI" ]; then
    echo "⚠️  Warning: geoAI conda environment not activated"
    echo "   Please run: conda activate geoAI"
    echo "   Then rerun this script"
    exit 1
fi

echo "✅ Environment check passed"

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    conda install -c conda-forge pre-commit -y
else
    echo "✅ pre-commit already installed"
fi

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Create scripts directory if it doesn't exist
mkdir -p scripts
chmod +x scripts/check_antipatterns.py
chmod +x scripts/check_course_structure.py

# Test the installation
echo "🧪 Testing pre-commit installation..."

# Run a quick test on a sample file
echo "# Test file" > temp_test.py
echo "def test_function():" >> temp_test.py
echo "    pass" >> temp_test.py

# Run pre-commit on the test file
if pre-commit run --files temp_test.py; then
    echo "✅ Pre-commit hooks working correctly"
else
    echo "⚠️  Pre-commit hooks detected issues (this is expected for unformatted code)"
fi

# Clean up test file
rm -f temp_test.py

# Run anti-pattern detection test
echo "🔍 Testing anti-pattern detection..."
if python scripts/check_antipatterns.py --exclude-dirs="LLMs-from-scratch,_extensions,.claude,tests" geogfm; then
    echo "✅ Anti-pattern detection working"
else
    echo "⚠️  Anti-pattern detection found issues (review output above)"
fi

# Test course structure validation
echo "📚 Testing course structure validation..."
if python scripts/check_course_structure.py book/index.qmd; then
    echo "✅ Course structure validation working"
else
    echo "⚠️  Course structure validation found issues (review output above)"
fi

echo ""
echo "🎉 Pre-commit setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Run 'pre-commit run --all-files' to check all existing files"
echo "   2. Make a test commit to verify everything works"
echo "   3. Read docs/PRE_COMMIT_SETUP.md for detailed usage information"
echo ""
echo "💡 Tips:"
echo "   - Hooks run automatically on 'git commit'"
echo "   - Use 'pre-commit run' to test before committing"
echo "   - See docs/PRE_COMMIT_SETUP.md for troubleshooting"
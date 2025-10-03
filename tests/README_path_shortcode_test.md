# Path Shortcode Filter Tests

This directory contains comprehensive tests for the `{{< path >}}` shortcode filter used throughout the GEOG 288KC course site.

## What the Filter Does

The path shortcode filter resolves `{{< path filename >}}` shortcodes to absolute paths (`/filename`) that work consistently across all page locations in the site, including:
- Root pages (`docs/index.html`)  
- Chapter pages (`docs/chapters/c01-*.html`)
- Other subdirectories (`docs/extras/cheatsheets/*.html`)

## Running the Tests

### Quick Test (Shortcode Filter Only)
```bash
make test-shortcode
```

### Full Test Suite (All Tests)
```bash  
make test
```

### Specific Test File
```bash
pytest tests/test_path_shortcode_filter.py -v
```

### Run Individual Test Classes
```bash
# Test just the filter functionality
pytest tests/test_path_shortcode_filter.py::TestPathShortcodeFilter -v

# Test Quarto integration
pytest tests/test_path_shortcode_filter.py::TestQuartoIntegration -v

# Test footer image resolution
pytest tests/test_path_shortcode_filter.py::TestFooterImageResolution -v
```

## Test Coverage

### Unit Tests (`TestPathShortcodeFilter`)
- ✅ Basic shortcode replacement
- ✅ Multiple shortcodes in one string
- ✅ Different panflute element types (Str, RawInline, RawBlock)
- ✅ Whitespace variations in shortcodes
- ✅ Quoted paths (single and double quotes)
- ✅ No-shortcode passthrough
- ✅ Nested directory paths
- ✅ Complex HTML contexts

### Integration Tests (`TestQuartoIntegration`)  
- ✅ Quarto render without warnings
- ✅ Shortcodes resolved in output HTML
- ✅ Proper path resolution across contexts

### Footer Tests (`TestFooterImageResolution`)
- ✅ _quarto.yml contains expected shortcodes
- ✅ Regex pattern matches footer content
- ✅ Filter can resolve footer shortcodes

### Cross-Site Tests (`TestCrossSitePathResolution`)
- ✅ Referenced image files exist
- ✅ Absolute paths work from subdirectories
- ✅ Path format is correct for web serving

## Expected Test Results

When all tests pass, you should see:
- ✅ All shortcode patterns correctly resolved to absolute paths
- ✅ No "Shortcode 'path' not found" warnings during Quarto builds
- ✅ Footer images render correctly across all site pages
- ✅ Filter handles edge cases (quotes, whitespace, multiple shortcodes)

## Troubleshooting

### Test Import Errors
If tests fail to import the filter:
```bash
# Ensure the filter exists and is executable
ls -la book/tools/filters/path_shortcode.py
chmod +x book/tools/filters/path_shortcode.py
```

### Quarto Integration Test Failures  
If Quarto integration tests fail:
```bash
# Verify Quarto is installed and accessible
quarto --version

# Check that filter is registered in _quarto.yml
grep -A 5 "filters:" book/_quarto.yml
```

### Missing Test Dependencies
```bash
# Install test dependencies
pip install pytest panflute
```

## Test Files

- `test_path_shortcode_filter.py` - Main test suite
- `fixtures/test_shortcode_integration.qmd` - Sample Quarto file for integration testing

## Integration with CI/CD

These tests can be run in continuous integration to verify:
1. Filter functionality remains correct across code changes
2. Quarto builds complete without shortcode warnings  
3. Site images render properly from all page locations

Add to your CI pipeline:
```yaml
- name: Test Path Shortcode Filter
  run: make test-shortcode
```
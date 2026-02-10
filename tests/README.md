# utils_cmip7 Tests

This directory contains tests for the utils_cmip7 package.

## Test Structure

```
tests/
├── run_smoke_tests.py    # Main test runner
├── test_imports.py        # Import resolution tests
├── test_config.py         # Configuration validation tests
└── README.md              # This file
```

## Smoke Tests

**Smoke tests** verify basic functionality without requiring sample data:
- ✓ Import resolution (all public API functions can be imported)
- ✓ Configuration loading (VAR_CONVERSIONS, RECCAP regions, etc.)
- ✓ Validation functions (error handling for missing/unreadable files)
- ✓ Backward compatibility (legacy imports still work with deprecation warnings)

### Running Smoke Tests

```bash
# From repository root
cd tests
python3 run_smoke_tests.py

# Or run individual test suites
python3 test_imports.py
python3 test_config.py
```

### Expected Output

If dependencies are not installed (numpy, iris, etc.), tests will fail with import errors. This is expected. Install the package first:

```bash
cd ..
pip install -e .
cd tests
python3 run_smoke_tests.py
```

## Unit Tests

Unit tests covering core modules (added in v0.3.0-v0.4.0):

- [x] `tests/test_io/test_stash.py` - STASH code mappings (100% coverage)
- [x] `tests/test_io/test_file_discovery.py` - Month code decoding, file patterns (98% coverage)
- [x] `tests/test_io/test_extract.py` - Cube extraction with STASH handling
- [x] `tests/test_processing/test_spatial.py` - Global aggregation
- [x] `tests/test_processing/test_temporal.py` - Temporal aggregation (92% coverage)
- [x] `tests/test_processing/test_metrics.py` - Metric definitions (70% coverage)
- [x] `tests/test_diagnostics/test_extraction.py` - Extraction workflows (56% coverage)
- [x] `tests/test_validation/test_compare.py` - Bias and RMSE computation (99% coverage)
- [x] `tests/test_validation/test_outputs.py` - Validation output bundles (100% coverage)
- [x] `tests/test_plotting/test_maps.py` - Spatial map extraction and plotting
- [x] `tests/test_v04_breaking_changes.py` - v0.4.0 breaking change verification
- [ ] `tests/test_processing/test_regional.py` - Regional masking

## Continuous Integration

CI is configured via GitHub Actions (`.github/workflows/tests.yml`):

- [x] GitHub Actions workflow
- [x] Test on Python 3.9, 3.10, 3.11, 3.12
- [x] Linting (flake8, black, isort)
- [x] Coverage reporting (Codecov)

## Test Data

Sample data for integration tests is not yet available. Options:

1. Create synthetic NetCDF files with expected structure
2. Use subset of real data (coordinate with research group)
3. Mock iris.Cube objects for unit tests

## Contributing Tests

When adding tests:

1. Follow existing test naming conventions (`test_*.py`)
2. Add docstrings explaining what each test verifies
3. Use descriptive assertion messages
4. Keep tests independent (no shared state)
5. Update this README with new test categories

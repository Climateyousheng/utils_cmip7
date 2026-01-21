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

## Unit Tests (TODO)

Unit tests for individual functions are planned for v0.2.2-v0.3.0:

- [ ] `tests/test_io/test_stash.py` - STASH code mappings
- [ ] `tests/test_io/test_file_discovery.py` - Month code decoding, file patterns
- [ ] `tests/test_processing/test_spatial.py` - Global aggregation
- [ ] `tests/test_processing/test_temporal.py` - Temporal aggregation
- [ ] `tests/test_processing/test_regional.py` - Regional masking

These will require synthetic test fixtures (small NetCDF files).

## Continuous Integration (TODO)

CI setup is planned for v0.2.2-v0.3.0:

- [ ] GitHub Actions workflow
- [ ] Test on Python 3.8, 3.9, 3.10, 3.11
- [ ] Linting (flake8, black, isort)
- [ ] Coverage reporting

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

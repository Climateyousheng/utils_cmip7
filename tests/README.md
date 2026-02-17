# utils_cmip7 Tests

This directory contains tests for the utils_cmip7 package.

## Test Structure

```
tests/
├── run_smoke_tests.py              # Basic smoke test runner
├── test_imports.py                 # Import resolution tests
├── test_config.py                  # Configuration validation tests
├── test_canonical_variables.py     # Canonical variable registry tests
├── test_cli_helpers.py             # CLI helper function tests
├── test_cli_auto_detection.py      # Auto-detection from ensemble logs tests
├── test_v04_breaking_changes.py    # v0.4.0 breaking change verification
├── test_extract_raw_overview.py    # Raw extraction + overview table integration
├── test_extraction_fix.py          # Extraction pipeline fix tests
├── test_metrics_pre_extracted.py   # Pre-extracted metrics tests
├── test_overview_upsert.py         # Overview table upsert tests
├── test_plot_fix.py                # Plotting fix tests
├── test_soil_params_parser.py      # Soil parameter parser tests
├── test_tau_fix.py                 # Tau computation fix tests
├── test_warnings_suppressed.py     # Warning suppression tests
├── test_io/                        # I/O module tests
├── test_processing/                # Processing module tests
├── test_diagnostics/               # Diagnostics module tests
├── test_validation/                # Validation module tests
├── test_plotting/                  # Plotting module tests
└── README.md                       # This file
```

## Smoke Tests

**Smoke tests** verify basic functionality without requiring sample data:
- ✓ Import resolution (all public API functions can be imported)
- ✓ Configuration loading (VAR_CONVERSIONS, RECCAP regions, etc.)
- ✓ Validation functions (error handling for missing/unreadable files)

### Running Smoke Tests

```bash
# From repository root
python tests/run_smoke_tests.py
```

### Running the full test suite

```bash
.venv/bin/python -m pytest tests/ -v
```

## Unit Tests

Unit tests covering core modules (added in v0.3.0-v0.4.x):

- [x] `tests/test_io/test_stash.py` - STASH code mappings (100% coverage)
- [x] `tests/test_io/test_file_discovery.py` - Month code decoding, file patterns (98% coverage)
- [x] `tests/test_io/test_extract.py` - Cube extraction with STASH handling
- [x] `tests/test_processing/test_spatial.py` - Global aggregation
- [x] `tests/test_processing/test_temporal.py` - Temporal aggregation (92% coverage)
- [x] `tests/test_processing/test_metrics.py` - Metric definitions (70% coverage)
- [x] `tests/test_diagnostics/test_extraction.py` - Extraction workflows
- [x] `tests/test_validation/test_compare.py` - Bias and RMSE computation (99% coverage)
- [x] `tests/test_validation/test_outputs.py` - Validation output bundles (100% coverage)
- [x] `tests/test_plotting/test_maps.py` - Spatial map extraction and plotting
- [x] `tests/test_plotting/test_ppe_scatter.py` - PPE scatter plot functions (26 tests)
- [x] `tests/test_v04_breaking_changes.py` - v0.4.0 breaking change verification
- [x] `tests/test_cli_helpers.py` - CLI helper functions
- [x] `tests/test_cli_auto_detection.py` - Auto-detection from ensemble logs
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

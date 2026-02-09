# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.3.1] - 2026-02-09

### Added

#### Canonical Name Resolution in `try_extract()`
- `try_extract()` now accepts canonical variable names (`'CVeg'`, `'GPP'`) and aliases (`'VegCarb'`, `'soilResp'`) directly, resolving them to STASH codes via `CANONICAL_VARIABLES`
- Users no longer need to know stash short names or MSI codes for extraction
- 15 new tests in `tests/test_io/test_extract.py`

#### Optional Unit Conversion in Map Field Extraction
- `extract_map_field()` and `extract_anomaly_field()` accept an optional `variable` parameter
- When provided, applies `conversion_factor` and overrides `units`/`name` from `CANONICAL_VARIABLES`
- No conversion by default (fully backward compatible)
- 7 new tests in `tests/test_plotting/test_maps.py`

#### Raw Extraction Validation
- `--validate` flag for `scripts/extract_raw.py` - Validate extracted annual means against observations
- `--validate` flag for `utils-cmip7-extract-raw` CLI command
- `--validation-outdir` option to customize validation output directory
- Global-only validation against CMIP6 and RECCAP2 observations
- CSV outputs with bias statistics, three-way comparison plots, console summary

### Fixed
- Colorbar overlapping x-axis labels in multi-panel subplot layouts (2x3 grids) — increased pad from 0.05 to 0.12

### Documentation
- Updated `docs/API.md` with `try_extract()`, `extract_map_field()`, `extract_anomaly_field()`, `combine_fields()`, `plot_spatial_map()`, `plot_spatial_anomaly()`
- Fixed inaccurate raises/return docs for `get_variable_config()` and `get_conversion_key()`
- Fixed `CANONICAL_VARIABLES` structure example

### Statistics
- **Tests**: 275 (was 174 in v0.3.0)
- **Coverage**: 29% (was 24% in v0.3.0)

---

## [0.3.0] - 2026-01-26

### Added

#### Testing Infrastructure (Sprint 1)
- Comprehensive test suite with **174 tests** covering core functionality
- GitHub Actions CI/CD workflow for Python 3.8, 3.9, 3.10, 3.11
- Code quality tools: flake8, black, isort, mypy configurations
- Coverage reporting with pytest-cov (24% overall coverage)
- Test files:
  - `tests/test_io/test_stash.py` - 70+ tests for STASH code mapping (100% coverage)
  - `tests/test_io/test_file_discovery.py` - 50+ tests for file discovery (98% coverage)
  - `tests/test_processing/test_metrics.py` - 60+ tests for metrics (70% coverage)
  - `tests/test_processing/test_spatial.py` - 15+ tests for spatial aggregation (48% coverage)
  - `tests/test_diagnostics/test_extraction.py` - 17 tests for extraction (56% coverage)
  - `tests/test_processing/test_temporal.py` - 14 tests for temporal aggregation (92% coverage)
  - `tests/test_warnings_suppressed.py` - Iris warning suppression tests

#### API Stabilization
- Public API definition with `__all__` exports in 6 modules:
  - `io/stash.py`, `io/file_discovery.py`
  - `processing/temporal.py`, `processing/spatial.py`
  - `diagnostics/extraction.py`
- API stability matrix establishing clear guarantees:
  - **Stable**: Core extraction, processing, config, STASH, file discovery
  - **Provisional**: Regional aggregation, raw extraction
  - **Unstable**: Validation, plotting
  - **Experimental**: CLI, soil params

#### Documentation
- `docs/API.md` - Comprehensive public API reference (527 lines)
- `docs/API_STABILIZATION_PLAN.md` - API stabilization strategy
- `docs/API_STABILIZATION_COMPLETE.md` - Completion summary
- `docs/SPRINT1_COMPLETE.md` - Sprint 1 testing achievements
- `docs/SPRINT2_PROGRESS.md` - Sprint 2 coverage expansion
- `docs/CODEMAPS/` - Code structure documentation

#### Features
- `compute_latlon_box_mean()` - Area-weighted regional box extraction
- Enhanced error messages with detailed troubleshooting guidance
- Improved RECCAP mask loading with environment variable support

### Changed

#### Backward Compatible Changes
- **Deprecation warnings** now reference v0.4.0 instead of v0.3.0:
  - `var_mapping` parameter in `extract_annual_means()` deprecated (use canonical names)
  - Legacy variable names deprecated (e.g., 'VegCarb' → 'CVeg', 'soilResp' → 'Rh')
- Version bumped from 0.2.1 to 0.3.0
- Package structure reorganized for better clarity

#### Dependency Updates
- Pinned `numpy<1.25` for Python 3.9 binary compatibility
- Pinned `iris<3.9` to maintain Python 3.9 support
- Pinned `cf-units<4.0` and `cftime<1.7` for stability
- Added coverage files to `.gitignore`

### Fixed

#### CI/CD Fixes
- Python 3.8 compatibility: Fixed cf-units/cftime AttributeError
- Python 3.9 compatibility: Fixed numpy binary incompatibility
- Test collection errors: Converted `test_warnings_suppressed.py` to proper pytest format
- Area weights test: Added coordinate bounds to fix ValueError
- Import errors: Fixed legacy path manipulation in test files
- Filename patterns: Corrected test filenames with invalid spaces

#### Test Fixes
- Fixed `decode_month()` test expectations for edge cases
- Fixed STASH edge case tests (None/numeric input handling)
- Mocked `load_reccap_mask()` to avoid FileNotFoundError in tests
- Fixed iris coordinate constraints in temporal tests

### Deprecated

The following features are **deprecated in v0.3.0** and will be **removed in v0.4.0**:

1. **`var_mapping` parameter** in `extract_annual_means()`:
   ```python
   # Deprecated:
   extract_annual_means(expts, var_mapping=['gpp', 'npp'])

   # Use instead:
   extract_annual_means(expts, var_list=['GPP', 'NPP'])
   ```

2. **Legacy variable names** (use canonical names):
   - `VegCarb` → `CVeg`
   - `soilResp` → `Rh`
   - `soilCarbon` → `CSoil`
   - `temp` → `tas`
   - `precip` → `pr`

**Migration period**: These features will continue to work (with warnings) through v0.3.x and be removed in v0.4.0.

### Security

- No security issues identified or fixed in this release

### Infrastructure

#### CI/CD
- GitHub Actions workflow with matrix testing (Python 3.8-3.11)
- Automated coverage reporting (Codecov ready)
- Code quality checks on every push/PR
- Test artifacts uploaded for debugging

#### Development Tools
- `.flake8` configuration for linting
- `pyproject.toml` enhancements for black, isort, pytest
- Coverage HTML reports in `htmlcov/`

### Performance

- No significant performance changes in this release

### Breaking Changes

**None** - v0.3.0 is fully backward compatible with v0.2.x.

All deprecated features continue to work with deprecation warnings.

### Migration Guide

#### From v0.2.x to v0.3.0

No code changes required! However, you should update your code to use canonical variable names to avoid deprecation warnings:

**Before (v0.2.x):**
```python
from utils_cmip7 import extract_annual_means

data = extract_annual_means(
    ['xqhuc'],
    var_list=['VegCarb', 'soilResp', 'soilCarbon'],  # Legacy names
    var_mapping=['vegcarb', 'soilresp', 'soilcarbon']  # No longer needed
)
```

**After (v0.3.0):**
```python
from utils_cmip7 import extract_annual_means

data = extract_annual_means(
    ['xqhuc'],
    var_list=['CVeg', 'Rh', 'CSoil']  # Canonical names
    # var_mapping removed - automatic lookup
)
```

See `docs/API.md` for complete migration guide.

### Statistics

- **Lines of code**: ~3000 statements
- **Test coverage**: 24% (704/2998 statements covered)
- **Tests**: 174 passing, 1 skipped
- **Python support**: 3.8, 3.9, 3.10, 3.11
- **CI status**: ✅ All platforms passing

### Contributors

- Yousheng Li (@Climateyousheng) - Project maintainer
- Claude Sonnet 4.5 - AI assistant for testing and documentation

### Acknowledgments

Special thanks to:
- The SciTools/Iris team for the excellent climate data library
- The pytest community for outstanding testing tools
- GitHub Actions for reliable CI/CD infrastructure

---

## [0.2.1] - 2025-XX-XX

### Added
- Canonical variable registry in `config.py`
- Soil parameter module
- Validation improvements

### Changed
- Package structure reorganization
- Enhanced observation data handling

---

## [0.2.0] - 2025-XX-XX

### Added
- Initial stable release
- Core extraction and processing functionality
- CMIP6 and RECCAP2 data support

---

## Release Tags

- `v0.3.1` - Canonical name resolution, unit conversion, plotting fixes (2026-02-09)
- `v0.3.0` - API stabilization and testing foundation (2026-01-26)
- `v0.2.1` - Canonical variables and validation
- `v0.2.0` - Initial stable release

---

For detailed information about each release, see the corresponding documentation in `docs/`.

For bug reports and feature requests, please visit:
https://github.com/Climateyousheng/utils_cmip7/issues

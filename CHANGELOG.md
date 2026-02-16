# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Auto-detection of ensemble parameters in `validate-experiment`

The `utils-cmip7-validate-experiment` CLI now automatically detects soil parameters from ensemble-generator logs, eliminating the need for manual parameter specification in common workflows.

**How it works:**

1. **Automatic detection**: When no explicit parameter source is provided, the CLI checks the default log directory (`~/scripts/hadcm3b-ensemble-generator/logs`) for matching ensemble parameters
2. **Priority order**:
   - Explicit flags (`--soil-param-file`, `--soil-log-file`, `--soil-params`, `--use-default-soil-params`) take precedence
   - Auto-detection from logs (when available)
   - Error with helpful guidance (when neither is available)
3. **Graceful degradation**: Warning if default directory doesn't exist, then requires explicit parameter source

**New behavior:**

```bash
# ✨ NEW: Auto-detect from ensemble logs (no flags needed!)
utils-cmip7-validate-experiment xqjca
# ✓ Auto-detected soil parameters from: .../xqjc_updated_parameters_20260128.json
#   Ensemble: xqjc, Member: xqjca

# Custom log directory
utils-cmip7-validate-experiment xqjca --log-dir /custom/path/logs

# Explicit source still works (overrides auto-detection)
utils-cmip7-validate-experiment xqjca --use-default-soil-params
```

**Benefits:**

- **Prevents accidental overwrites**: Before, running `validate-experiment xqjca --use-default-soil-params` would overwrite parameters loaded from logs. Now auto-detection prevents this.
- **Convenience**: No need to specify parameter source for ensemble experiments
- **Backward compatible**: All existing workflows continue to work

**New CLI argument:**

- `--log-dir DIR`: Custom log directory (default: `~/scripts/hadcm3b-ensemble-generator/logs`)

**Implementation details:**

- New helper function: `_extract_ensemble_prefix(expt_id)` — extracts 4-character prefix from 5-character experiment IDs
- 3-phase parameter loading logic: explicit source → auto-detection → error with guidance
- Comprehensive test coverage: 12 new tests in `test_cli_helpers.py` and `test_cli_auto_detection.py`

---

## [0.4.0] - 2026-02-09

### Breaking Changes

This is a **breaking release**. Deprecated features from v0.3.x have been removed.

#### Removed: Legacy variable names

The following legacy variable names now raise `ValueError` instead of resolving silently:

| Legacy Name | Canonical Name |
|-------------|---------------|
| `VegCarb` | `CVeg` |
| `soilResp` | `Rh` |
| `soilCarbon` | `CSoil` |
| `temp` | `tas` |
| `precip` | `pr` |
| `fracPFTs` | `frac` |

**Migration:** Replace all legacy names with canonical names in your scripts.

#### Removed: `var_mapping` parameter

`extract_annual_means(var_mapping=...)` now raises `TypeError`. Use `var_list` with canonical variable names instead.

#### Removed: `var_dict` alias

`from utils_cmip7.config import var_dict` no longer works. Use `VAR_CONVERSIONS` directly.

#### Removed: `DEFAULT_VAR_MAPPING`

`config.DEFAULT_VAR_MAPPING` has been deleted. Use `DEFAULT_VAR_LIST` instead.

#### Removed: Legacy `VAR_CONVERSIONS` keys

Internal keys like `'V carb'`, `'S resp'`, `'S carb'`, `'Ocean flux'`, `'Air flux'`, etc. have been removed from `VAR_CONVERSIONS`. Only canonical keys and internal dispatch keys (`'Others'`, `'precip'`, `'Total co2'`) remain.

#### Dropped: Python 3.8 support

Python 3.8 reached end-of-life in October 2024. Minimum supported version is now Python 3.9.

### Added

#### Level selection for multi-dimensional cubes
- `extract_map_field()` and `extract_anomaly_field()` accept an optional `level` parameter for cubes with extra dimensions (e.g., PFT fractions with 9 levels)
- New internal helper `_select_level()` in `processing/map_fields.py`

#### Validation module tests
- `tests/test_validation/test_compare.py` — 20 tests for `compute_bias`, `compute_rmse`, `compare_single_metric`, `compare_metrics`, `summarize_comparison`, `print_comparison_table` (99% coverage)
- `tests/test_validation/test_outputs.py` — 4 tests for `write_single_validation_bundle` (100% coverage)

#### Breaking change verification tests
- `tests/test_v04_breaking_changes.py` — Comprehensive tests documenting and verifying all v0.4.0 breaking changes

### Changed

- Internal `METRIC_DEFINITIONS` conversion keys updated to canonical names
- Internal `diagnostics/raw.py` variable tuples updated to canonical names
- Internal `diagnostics/metrics.py` mappings cleaned of legacy entries
- `try_extract()` no longer resolves legacy aliases — only canonical names and STASH codes
- `scripts/extract_raw.py` updated to use canonical variable names
- CI matrix updated: Python 3.9, 3.10, 3.11, 3.12

### Migration Guide

**From v0.3.x to v0.4.0:**

```python
# Before (v0.3.x — deprecated, with warnings):
data = extract_annual_means(
    ['xqhuc'],
    var_list=['VegCarb', 'soilResp', 'soilCarbon'],
    var_mapping=['vegcarb', 'soilresp', 'soilcarbon']
)

# After (v0.4.0):
data = extract_annual_means(
    ['xqhuc'],
    var_list=['CVeg', 'Rh', 'CSoil']
)
```

### Statistics
- **Tests**: ~300 (was 275 in v0.3.1)
- **Coverage**: 29%
- **Python support**: 3.9, 3.10, 3.11, 3.12

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

- `v0.4.0` - Breaking: remove deprecated features, drop Python 3.8, add level selection (2026-02-09)
- `v0.3.1` - Canonical name resolution, unit conversion, plotting fixes (2026-02-09)
- `v0.3.0` - API stabilization and testing foundation (2026-01-26)
- `v0.2.1` - Canonical variables and validation
- `v0.2.0` - Initial stable release

---

For detailed information about each release, see the corresponding documentation in `docs/`.

For bug reports and feature requests, please visit:
https://github.com/Climateyousheng/utils_cmip7/issues

# Migration Guide

This guide covers upgrading between versions and the planned roadmap.

---

## Quick Reference: Import Changes

| Old Import | New Import (v0.2.2+) |
|------------|---------------------|
| `from analysis import extract_annual_means` | `from utils_cmip7.diagnostics import extract_annual_means` |
| `from plot import plot_timeseries_grouped` | `from utils_cmip7.plotting import plot_timeseries_grouped` |
| `from analysis import try_extract, stash` | `from utils_cmip7.io import try_extract, stash` |

## Quick Reference: Variable Name Changes (v0.4.0)

| Legacy Name | Canonical Name |
|-------------|---------------|
| `VegCarb` | `CVeg` |
| `soilResp` | `Rh` |
| `soilCarbon` | `CSoil` |
| `temp` | `tas` |
| `precip` | `pr` |
| `fracPFTs` | `frac` |
| `Total co2` | `co2` |

---

## Upgrading to v0.4.0 (from v0.3.x)

v0.4.0 is a **breaking release**. All deprecated features from v0.3.x have been removed.

### Variable names

Legacy variable names now raise `ValueError` instead of resolving silently.
Replace all legacy names with canonical CMIP-style names:

```python
# Before (v0.3.x -- deprecated, with warnings):
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

### Removed APIs

| Removed | Replacement |
|---------|-------------|
| `extract_annual_means(var_mapping=...)` | Use `var_list` with canonical names |
| `from utils_cmip7.config import var_dict` | Use `VAR_CONVERSIONS` directly |
| `config.DEFAULT_VAR_MAPPING` | Use `DEFAULT_VAR_LIST` |
| Legacy `VAR_CONVERSIONS` keys (`'V carb'`, `'S resp'`, etc.) | Use canonical keys (`'CVeg'`, `'Rh'`, etc.) |

### Python version

Python 3.8 is no longer supported (EOL Oct 2024). Minimum is Python 3.9.

---

## Upgrading to v0.3.0 (from v0.2.x)

No code changes required. v0.3.0 was fully backward compatible with v0.2.x.

Deprecation warnings were added for legacy variable names and the `var_mapping`
parameter. These warnings guided migration ahead of the v0.4.0 breaking release.

### New in v0.3.0

- API stabilization with `__all__` exports and stability matrix
- 174 tests, 24% coverage
- GitHub Actions CI/CD (Python 3.8-3.11)

---

## Upgrading to v0.2.2 (from v0.2.1)

### Plotting imports changed (backward-compatible)

```python
# Old (still works with DeprecationWarning):
from plot import plot_timeseries_grouped

# New:
from utils_cmip7.plotting import plot_timeseries_grouped
```

All plotting functions now accept an `ax` parameter for custom axes.

### CLI tools added

```bash
utils-cmip7-extract-preprocessed xqhuc
utils-cmip7-extract-raw xqhuj --start-year 2000 --end-year 2010
utils-cmip7-validate-experiment xqhuc --use-default-soil-params
utils-cmip7-validate-ppe xqhuc --top-n 20
```

---

## Deprecation Timeline

| Version | Status |
|---------|--------|
| v0.2.0 | Legacy imports deprecated (warnings added) |
| v0.2.2 | New package structure available, backward-compatible wrappers |
| v0.3.0 | API frozen, legacy wrappers remain with warnings |
| **v0.4.0** | **Legacy names and APIs removed (breaking)** |
| v1.0.0 | Legacy import shims (`analysis.py`, `plot.py`) removed |

---

## Module Organization (v0.4.0)

```
utils_cmip7/
    io/                  # NetCDF loading, STASH handling, file discovery
        stash.py
        file_discovery.py
        extract.py
        obs_loader.py
    processing/          # Temporal/spatial aggregation, unit conversions
        spatial.py
        temporal.py
        regional.py
        metrics.py
    diagnostics/         # Carbon-cycle diagnostics
        extraction.py
        raw.py
        metrics.py
    validation/          # Model validation against observations
        compare.py
        visualize.py
        overview_table.py
        outputs.py
    plotting/            # Visualization (requires cartopy)
        timeseries.py
        spatial.py
        maps.py
        styles.py
        ppe_viz.py
        ppe_param_viz.py
    soil_params/         # Soil parameter management
        params.py
        parsers.py
    config.py            # Configuration, variable registry, constants
    cli.py               # Command-line entry points (Experimental)
```

---

## Common Issues

### ImportError: No module named 'utils_cmip7'

Install in editable mode:
```bash
cd ~/path/to/utils_cmip7
pip install -e .
```

### ValueError: Variable name 'VegCarb' was removed in v0.4.0

Replace legacy names with canonical names (see Quick Reference above).

### TypeError: extract_annual_means() got an unexpected keyword argument 'var_mapping'

Remove the `var_mapping` parameter. Use `var_list` with canonical names instead.

---

## Roadmap: v0.4.x Series

Based on an expert review of the v0.4.0 codebase (2026-02-10).

### Phase 1: Scientific Correctness Hardening -- COMPLETE

Commit `f8f3d37` (2026-02-10). Files: `config.py`, tests.

- Derived `VAR_CONVERSIONS` from `CANONICAL_VARIABLES` (eliminated dual-maintenance drift risk)
- Added `units_in` field to every canonical variable (for future input validation)
- Added `time_handling` field (`"mean_rate"`, `"state"`, `"already_integral"`)
- Documented fgco2 assumption: conversion factor assumes `molC/m2/yr` input

### Phase 2: Documentation Fixes

Effort: Trivial (1 PR). Files: `docs/API.md`.

1. Replace stale "v0.3.x" references with "v0.4.x" (lines 85 and 318)
2. Document longitude convention (0-360 internally, auto-wrapped for plotting)
3. Document `extract_annual_means()` output invariants (dict keys, shapes, dtypes)

### Phase 3: Provenance Tracking

Effort: Medium (1 PR). Files: `validation/`, `config.py`.

1. Add `get_provenance()` returning dict with package version, git SHA, mask path
2. Write `provenance.json` alongside validation outputs for reproducibility

### Phase 4: Internal Refactor of Regional Aggregation

Effort: Medium (1 PR). Files: `processing/regional.py`, `config.py`.

1. Refactor `compute_regional_annual_mean()` to look up aggregation from `CANONICAL_VARIABLES` directly
2. Remove string-based dispatch on `Others`/`precip`/`Total co2` protocol keys
3. Deprecate `get_conversion_key()` (becomes dead code after refactor)

### Lower-Priority Items

| Item | Risk | Notes |
|------|------|-------|
| RECCAP region 5 (Africa merge) undocumented | Low | Works but fragile -- document or add explicit ID |
| `try_extract()` return type ambiguity | Low | Pick CubeList-or-None contract, document it |
| `plot_map_field()` convenience wrapper | Deferred | Plotting API is "Unstable" |
| Migrate linting to ruff | Nice-to-have | CI works fine with flake8 + black + isort |
| pre-commit hooks | Nice-to-have | CI catches issues already |
| Output schema versioning | Premature | Validation module still "Provisional" |

---

## Getting Help

- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- [API.md](API.md) for public API reference
- [CHANGELOG.md](../CHANGELOG.md) for full release history

---

Last updated: v0.4.0 (2026-02-10)

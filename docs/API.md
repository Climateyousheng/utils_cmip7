# utils_cmip7 Public API Reference

**Version:** v0.3.0
**Status:** Stable
**Last Updated:** 2026-01-26

---

## Introduction

This document defines the **public, stable API** for `utils_cmip7` v0.3.0. All functions and classes listed here are considered stable and will not have breaking changes in the v0.3.x series.

For information on API stability guarantees, see the [API Stability Matrix](#api-stability-matrix) below.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [I/O Functions](#io-functions)
- [Processing Functions](#processing-functions)
- [Diagnostics Functions](#diagnostics-functions)
- [Configuration](#configuration)
- [API Stability Matrix](#api-stability-matrix)

---

## Installation

```bash
pip install -e .
```

For development with testing dependencies:
```bash
pip install -e .[dev]
```

---

## Quick Start

```python
import utils_cmip7

# Extract annual means for experiments
data = utils_cmip7.extract_annual_means(
    expts_list=['xqhuc', 'xqhsh'],
    var_list=['GPP', 'NPP', 'Rh', 'CSoil', 'CVeg'],  # Use canonical names
    regions=['global', 'Europe', 'Africa']
)

# Access results
gpp_global = data['xqhuc']['global']['GPP']
print(gpp_global['years'])  # Array of years
print(gpp_global['data'])   # Annual GPP values in PgC/yr
```

---

## I/O Functions

### STASH Code Mapping

#### `stash(variable_name)`

Map variable name to MSI-format STASH code.

**Parameters:**
- `variable_name` (str): Short variable name (e.g., 'gpp', 'npp', 'tas')

**Returns:**
- str: MSI-format STASH code (e.g., 'm01s03i261') or "nothing" if not found

**Example:**
```python
from utils_cmip7 import stash

code = stash('gpp')  # Returns 'm01s03i261'
```

**Stability:** ✅ **Stable** - No breaking changes in v0.3.x

---

#### `stash_nc(variable_name)`

Map variable name to numeric STASH code.

**Parameters:**
- `variable_name` (str): Short variable name

**Returns:**
- int: Numeric STASH code (e.g., 3261) or "nothing" if not found

**Example:**
```python
from utils_cmip7.io import stash_nc

code = stash_nc('gpp')  # Returns 3261
```

**Stability:** ✅ **Stable**

---

### File Discovery

#### `find_matching_files(expt, model, stream, start_year=None, end_year=None, base_dir='~')`

Find UM output files matching experiment and year criteria.

**Parameters:**
- `expt` (str): Experiment name (e.g., 'xqhuj')
- `model` (str): Model identifier ('a' for atmosphere, 'o' for ocean)
- `stream` (str): Stream identifier (e.g., 'pi', 'da')
- `start_year` (int, optional): Filter files from this year onwards
- `end_year` (int, optional): Filter files up to this year
- `base_dir` (str): Base directory containing experiment data

**Returns:**
- list: List of (year, month, filepath) tuples, sorted by year and month

**Example:**
```python
from utils_cmip7 import find_matching_files

files = find_matching_files('xqhuj', 'a', 'pi', start_year=1850, end_year=1860)
```

**Stability:** ✅ **Stable**

---

#### `decode_month(mon_code)`

Decode UM-style two-character month codes.

**Parameters:**
- `mon_code` (str): Two-character month code ('ja', '11', 'a1', etc.)

**Returns:**
- int: Month number (1-12) or 0 if invalid

**Example:**
```python
from utils_cmip7.io.file_discovery import decode_month

month = decode_month('ja')  # Returns 1 (January)
month = decode_month('dc')  # Returns 12 (December)
```

**Stability:** ✅ **Stable**

---

## Processing Functions

### Spatial Aggregation

#### `global_total_pgC(cube, var)`

Compute area-weighted global total in PgC or PgC/yr.

**Parameters:**
- `cube` (iris.cube.Cube): Input cube with lat/lon dimensions
- `var` (str): Variable name for unit conversion (e.g., 'GPP', 'CSoil')

**Returns:**
- iris.cube.Cube: Spatially collapsed cube with global total

**Example:**
```python
from utils_cmip7 import global_total_pgC
import iris

cube = iris.load_cube('gpp.nc')
global_gpp = global_total_pgC(cube, 'GPP')
```

**Stability:** ✅ **Stable**

---

#### `global_mean_pgC(cube, var)`

Compute area-weighted global mean.

**Parameters:**
- `cube` (iris.cube.Cube): Input cube
- `var` (str): Variable name for unit conversion

**Returns:**
- iris.cube.Cube: Spatially collapsed cube with global mean

**Stability:** ✅ **Stable**

---

### Temporal Aggregation

#### `merge_monthly_results(results, require_full_year=False)`

Merge multiple monthly outputs into annual mean time series.

**Parameters:**
- `results` (list of dict): Each dict has 'years' (fractional years) and 'data' keys
- `require_full_year` (bool): If True, only return years with all 12 months

**Returns:**
- dict: Dictionary with 'years' (integer years) and 'data' (annual means) keys

**Example:**
```python
from utils_cmip7 import merge_monthly_results

result1 = {"years": [1850.0, 1850.083, ...], "data": [10.0, 12.0, ...]}
result2 = {"years": [1850.25, 1850.333, ...], "data": [16.0, 18.0, ...]}

annual = merge_monthly_results([result1, result2], require_full_year=True)
```

**Stability:** ✅ **Stable**

---

#### `compute_monthly_mean(cube, var)`

Compute area-weighted global total for each month.

**Parameters:**
- `cube` (iris.cube.Cube): Input cube with time dimension
- `var` (str): Variable name for unit conversion

**Returns:**
- dict: Dictionary with 'years' (fractional years), 'data', 'name', and 'units' keys

**Stability:** ✅ **Stable**

---

#### `compute_annual_mean(cube, var)`

Compute area-weighted annual means from monthly data.

**Parameters:**
- `cube` (iris.cube.Cube): Input cube with time dimension
- `var` (str): Variable name ('Others' uses global mean, otherwise global total)

**Returns:**
- dict: Dictionary with 'years', 'data', 'name', and 'units' keys

**Stability:** ✅ **Stable**

---

### Regional Aggregation

#### `compute_regional_annual_mean(cube, conversion_key, region_name)`

Compute regional annual mean using RECCAP2 mask.

**Parameters:**
- `cube` (iris.cube.Cube): Input cube
- `conversion_key` (str): Conversion key from VAR_CONVERSIONS
- `region_name` (str): Region name or 'global'

**Returns:**
- dict: Dictionary with 'years', 'data', 'region', and other metadata

**Stability:** ⚠️ **Provisional** - Minor changes possible in v0.3.x

---

## Diagnostics Functions

### High-Level Extraction

#### `extract_annual_means(expts_list, var_list=None, var_mapping=None, regions=None, base_dir='~/annual_mean')`

Extract annual means from pre-processed NetCDF files.

**Main entry point** for extracting carbon cycle variables.

**Parameters:**
- `expts_list` (list of str): Experiment names (e.g., ['xqhuc', 'xqhsh'])
- `var_list` (list of str, optional): Variable names (canonical preferred)
  Default: ['Rh', 'CSoil', 'CVeg', 'frac', 'GPP', 'NPP', 'fgco2', 'tas', 'pr', 'co2']
- `var_mapping` (list of str, optional): **DEPRECATED** - Removed in v0.4.0
- `regions` (list of str, optional): Region names. Default: all RECCAP2 regions + 'Africa' + 'global'
- `base_dir` (str): Base directory containing experiment subdirectories

**Returns:**
- dict: Nested dictionary `{expt: {region: {var: {years, data, units, ...}}}}`

**Example:**
```python
from utils_cmip7 import extract_annual_means

# Basic usage with canonical names
data = extract_annual_means(['xqhuc'])
gpp = data['xqhuc']['global']['GPP']['data']

# Specific regions
data = extract_annual_means(
    ['xqhuc', 'xqhsh'],
    regions=['global', 'Europe', 'Africa']
)

# Custom variables
data = extract_annual_means(
    ['xqhuc'],
    var_list=['GPP', 'NPP', 'Rh', 'CSoil', 'CVeg']
)
```

**Stability:** ✅ **Stable**

**Deprecations:**
- `var_mapping` parameter deprecated, removed in v0.4.0
- Legacy variable names ('VegCarb', 'soilResp', etc.) deprecated, removed in v0.4.0

---

#### `compute_latlon_box_mean(cube, lon_bounds, lat_bounds)`

Compute area-weighted mean for a lat/lon bounding box.

**Parameters:**
- `cube` (iris.cube.Cube): Input cube
- `lon_bounds` (tuple): (lon_min, lon_max) in degrees East (0-360)
- `lat_bounds` (tuple): (lat_min, lat_max) in degrees North (-90 to 90)

**Returns:**
- iris.cube.Cube: Spatially collapsed cube

**Example:**
```python
from utils_cmip7.diagnostics import compute_latlon_box_mean

# Amazon region
amazon = compute_latlon_box_mean(cube, lon_bounds=(290, 320), lat_bounds=(-15, 5))
```

**Stability:** ✅ **Stable**

---

## Configuration

### Canonical Variable Registry

#### `CANONICAL_VARIABLES`

Dictionary mapping canonical variable names to their configuration.

**Structure:**
```python
{
    'GPP': {
        'stash_name': 'gpp',
        'stash_code': 'm01s03i261',
        'units': 'PgC/yr',
        'long_name': 'Gross Primary Productivity',
        ...
    },
    ...
}
```

**Usage:**
```python
from utils_cmip7 import CANONICAL_VARIABLES

gpp_config = CANONICAL_VARIABLES['GPP']
print(gpp_config['units'])  # 'PgC/yr'
```

**Stability:** ✅ **Stable**

---

#### `DEFAULT_VAR_LIST`

Default list of variables for extraction.

**Value:**
```python
['Rh', 'CSoil', 'CVeg', 'frac', 'GPP', 'NPP', 'fgco2', 'tas', 'pr', 'co2']
```

**Stability:** ✅ **Stable**

---

### Configuration Functions

#### `resolve_variable_name(var_name)`

Resolve variable name to canonical name.

**Parameters:**
- `var_name` (str): Variable name (canonical or legacy alias)

**Returns:**
- str: Canonical variable name

**Raises:**
- ValueError: If variable name not recognized

**Example:**
```python
from utils_cmip7 import resolve_variable_name

canonical = resolve_variable_name('VegCarb')  # Returns 'CVeg' (with deprecation warning)
canonical = resolve_variable_name('GPP')      # Returns 'GPP'
```

**Stability:** ✅ **Stable**

---

#### `get_variable_config(var_name)`

Get configuration dictionary for a variable.

**Parameters:**
- `var_name` (str): Canonical variable name

**Returns:**
- dict: Variable configuration

**Raises:**
- KeyError: If variable not in registry

**Stability:** ✅ **Stable**

---

#### `get_conversion_key(var_name)`

Get unit conversion key for a variable.

**Parameters:**
- `var_name` (str): Canonical variable name

**Returns:**
- str: Conversion key (e.g., 'gC/m2/s->PgC/yr')

**Stability:** ✅ **Stable**

---

## API Stability Matrix

| Component | Status | Breaking Changes? | Notes |
|-----------|--------|-------------------|-------|
| **Core Extraction** | **Stable** | ❌ No | `extract_annual_means()` and related |
| **Processing Functions** | **Stable** | ❌ No | Spatial/temporal aggregation |
| **Configuration API** | **Stable** | ❌ No | Canonical variables, config helpers |
| **STASH Mapping** | **Stable** | ❌ No | `stash()`, `stash_nc()` |
| **File Discovery** | **Stable** | ❌ No | `find_matching_files()`, `decode_month()` |
| **Regional Aggregation** | **Provisional** | ⚠️ Minor only | May add features, no breaking changes |
| **Raw Extraction** | **Provisional** | ⚠️ Minor only | Still evolving |
| **Validation Module** | **Unstable** | ✅ Yes | Breaking changes possible |
| **Plotting Module** | **Unstable** | ✅ Yes | Breaking changes possible |
| **CLI Commands** | **Experimental** | ✅ Yes | Interface may change |

### Stability Definitions

- **Stable**: No breaking changes in v0.3.x series. Safe to use in production.
- **Provisional**: Minor additions allowed, but no breaking changes to existing functionality.
- **Unstable**: Breaking changes possible in minor versions (v0.3.x → v0.3.y).
- **Experimental**: No stability guarantees. May change significantly or be removed.

---

## Migration from v0.2.x

### Breaking Changes

**None** - All v0.2.x code should work in v0.3.0 (with deprecation warnings).

### Deprecated (Removed in v0.4.0)

1. **`var_mapping` parameter** in `extract_annual_means()`:
   ```python
   # OLD (deprecated):
   extract_annual_means(expts, var_mapping=['gpp', 'npp'])

   # NEW:
   extract_annual_means(expts, var_list=['GPP', 'NPP'])
   ```

2. **Legacy variable names**:
   ```python
   # OLD (deprecated):
   var_list=['VegCarb', 'soilResp', 'soilCarbon']

   # NEW:
   var_list=['CVeg', 'Rh', 'CSoil']
   ```

### Migration Guide

See `docs/MIGRATION_v0.3.md` for detailed migration instructions.

---

## Version History

- **v0.3.0** (2026-01-26): API stabilization, deprecation grace period, comprehensive testing
- **v0.2.1** (2025-XX-XX): Canonical variable registry, validation improvements
- **v0.2.0** (2025-XX-XX): Major refactoring, package structure cleanup

---

## Support & Feedback

- **Issues**: https://github.com/Climateyousheng/utils_cmip7/issues
- **Discussions**: https://github.com/Climateyousheng/utils_cmip7/discussions

---

**Last updated:** 2026-01-26
**Maintainer:** Yousheng Li

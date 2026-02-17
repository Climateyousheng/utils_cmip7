# utils_cmip7 Public API Reference

**Version:** v0.4.0
**Status:** Stable
**Last Updated:** 2026-02-09

---

## Introduction

This document defines the **public, stable API** for `utils_cmip7` v0.4.0. All functions and classes listed here are considered stable and will not have breaking changes in the v0.4.x series.

For information on API stability guarantees, see the [API Stability Matrix](#api-stability-matrix) below.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [I/O Functions](#io-functions)
- [Processing Functions](#processing-functions)
- [Map Field Extraction](#map-field-extraction)
- [Plotting Functions](#plotting-functions)
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

**Stability:** ✅ **Stable** - No breaking changes in v0.4.x

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

### Cube Extraction

#### `try_extract(cubes, code, stash_lookup_func=None, debug=False)`

Extract cubes matching a variable identifier from a CubeList. Accepts multiple code formats and resolves them all to MSI strings for matching.

**Parameters:**
- `cubes` (iris.cube.CubeList): Collection of cubes to search
- `code` (str, int): Variable identifier. Accepted formats:
  - Canonical variable name: `'CVeg'`, `'GPP'` (from `CANONICAL_VARIABLES`)
  - MSI string: `'m01s03i261'`
  - Short name: `'gpp'` (requires `stash_lookup_func`)
  - Numeric stash_code: `3261`
- `stash_lookup_func` (callable, optional): Function to map short names to MSI strings (e.g., `stash()`)
- `debug` (bool): Print debug information during extraction

**Returns:**
- iris.cube.CubeList or None: Matching cubes, or empty CubeList / None if not found

**Example:**
```python
from utils_cmip7.io.extract import try_extract
import iris

cubes = iris.load('data.nc')

# Using canonical variable name (recommended)
gpp = try_extract(cubes, 'GPP')

# Using MSI string
gpp = try_extract(cubes, 'm01s03i261')

# Using numeric stash code
gpp = try_extract(cubes, 3261)

# Using short name with lookup function
from utils_cmip7.io.stash import stash
gpp = try_extract(cubes, 'gpp', stash_lookup_func=stash)
```

**Stability:** ⚠️ **Provisional**

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

**Stability:** ⚠️ **Provisional** - Minor changes possible in v0.4.x

---

## Map Field Extraction

Functions for extracting 2D spatial fields from iris Cubes, ready for plotting. Located in `utils_cmip7.processing.map_fields`.

### `extract_map_field(cube, time=None, time_index=None, variable=None, level=None)`

Extract a 2D spatial field from an iris Cube. Returns a dict ready for `plot_spatial_map()`.

**Parameters:**
- `cube` (iris.cube.Cube): Input cube with latitude and longitude (and optional time)
- `time` (int, optional): Year to select. Mutually exclusive with `time_index`
- `time_index` (int, optional): Positional index along the time dimension
- `variable` (str, optional): Canonical variable name (e.g., `'GPP'`, `'CVeg'`).
  When provided, applies `conversion_factor` and overrides `units` and `name` from `CANONICAL_VARIABLES`. No conversion by default.
- `level` (int, optional): Index along extra (non-time, non-lat, non-lon) dimension. Use for cubes with PFT or pseudo-level dimensions (e.g., `frac` with 9 PFT types).

**Returns:**
- dict with keys: `'data'` (2D ndarray), `'lons'` (1D ndarray), `'lats'` (1D ndarray), `'name'` (str), `'units'` (str), `'year'` (int or None), `'title'` (str)

**Example:**
```python
from utils_cmip7.processing.map_fields import extract_map_field
import iris

cube = iris.load_cube('gpp.nc')

# Basic extraction (raw units)
field = extract_map_field(cube, time=2000)

# With unit conversion (kgC/m2/s -> PgC/yr)
field = extract_map_field(cube, time=2000, variable='GPP')
print(field['units'])  # 'PgC/yr'
print(field['name'])   # 'GPP'
```

**Stability:** ⚠️ **Unstable** (Plotting API)

---

### `extract_anomaly_field(cube, time_a=None, time_index_a=None, time_b=None, time_index_b=None, symmetric=True, variable=None)`

Extract anomaly (data_a - data_b) between two time slices. Returns a dict ready for `plot_spatial_anomaly()`.

**Parameters:**
- `cube` (iris.cube.Cube): Input cube with time, latitude, and longitude
- `time_a` (int, optional): Year for the "after" field. Default: last timestep
- `time_index_a` (int, optional): Positional index for the "after" field
- `time_b` (int, optional): Year for the "before" / baseline field. Default: first timestep
- `time_index_b` (int, optional): Positional index for the "before" field
- `symmetric` (bool, default True): Auto-compute symmetric vmin/vmax centred at zero
- `variable` (str, optional): Canonical variable name or alias. When provided, applies `conversion_factor` to each slice before subtraction.

**Returns:**
- dict with keys: `'data'`, `'lons'`, `'lats'`, `'name'`, `'units'`, `'year_a'`, `'year_b'`, `'vmin'`, `'vmax'`, `'title'`

**Example:**
```python
from utils_cmip7.processing.map_fields import extract_anomaly_field

anomaly = extract_anomaly_field(cube, time_a=2000, time_b=1850, variable='CVeg')
print(anomaly['title'])  # 'CVeg anomaly (2000 - 1850)'
print(anomaly['units'])  # 'PgC'
```

**Stability:** ⚠️ **Unstable** (Plotting API)

---

### `combine_fields(fields, operation='sum', name=None, units=None)`

Combine multiple extracted fields element-wise.

**Parameters:**
- `fields` (list of dict): Each dict from `extract_map_field()` with `'data'`, `'lons'`, `'lats'`
- `operation` (str): N-ary: `'sum'`, `'mean'`. Binary (exactly 2 fields): `'subtract'`, `'multiply'`, `'divide'`
- `name` (str, optional): Override name. Auto-generated if None (e.g., `'GPP + NPP'`)
- `units` (str, optional): Override units. Inherited from first field if None

**Returns:**
- dict with keys: `'data'`, `'lons'`, `'lats'`, `'name'`, `'units'`, `'year'`

**Example:**
```python
from utils_cmip7.processing.map_fields import extract_map_field, combine_fields

gpp = extract_map_field(gpp_cube, time=2000, variable='GPP')
npp = extract_map_field(npp_cube, time=2000, variable='NPP')

total = combine_fields([gpp, npp], operation='sum')
ratio = combine_fields([npp, gpp], operation='divide', name='CUE', units='1')
```

**Stability:** ⚠️ **Unstable** (Plotting API)

---

## Plotting Functions

Geographic map plotting for 2D spatial fields. Requires `cartopy`. Located in `utils_cmip7.plotting.maps`.

### `plot_spatial_map(data, lons, lats, *, region=None, lon_bounds=None, lat_bounds=None, projection=None, cmap='viridis', vmin=None, vmax=None, title=None, units=None, add_coastlines=True, add_gridlines=True, colorbar=True, ax=None)`

Plot a 2D field on a cartopy map projection. Accepts pre-extracted numpy arrays (not iris Cubes).

**Parameters:**
- `data` (array, shape (n_lat, n_lon)): 2D field to plot
- `lons` (array, shape (n_lon,)): Longitude values in degrees
- `lats` (array, shape (n_lat,)): Latitude values in degrees
- `region` (str, optional): RECCAP2 region name (e.g., `'Europe'`). Mutually exclusive with bounds
- `lon_bounds` / `lat_bounds` (tuple, optional): Explicit (min, max) for map extent. Must be provided together
- `projection` (cartopy.crs.Projection, optional): Default: Robinson (global), PlateCarree (regional)
- `cmap` (str): Colormap. Default: `'viridis'`
- `vmin`, `vmax` (float, optional): Colour scale bounds
- `title` (str, optional): Plot title
- `units` (str, optional): Colorbar label
- `add_coastlines`, `add_gridlines`, `colorbar` (bool): Toggle decorations
- `ax` (GeoAxes, optional): Pre-existing axes to plot on

**Returns:**
- `(fig, ax)`: matplotlib Figure and cartopy GeoAxes

**Example:**
```python
from utils_cmip7.processing.map_fields import extract_map_field
from utils_cmip7.plotting.maps import plot_spatial_map

field = extract_map_field(cube, time=2000, variable='GPP')
fig, ax = plot_spatial_map(
    field['data'], field['lons'], field['lats'],
    title=field['title'], units=field['units'],
)

# Regional view
fig, ax = plot_spatial_map(
    field['data'], field['lons'], field['lats'],
    region='Europe', title='GPP — Europe',
)

# Multi-panel layout (2x3 subplots)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                         subplot_kw={'projection': ccrs.PlateCarree()})
for ax, var_field in zip(axes.flat, fields):
    plot_spatial_map(var_field['data'], var_field['lons'], var_field['lats'],
                     ax=ax, title=var_field['name'], units=var_field['units'])
```

**Stability:** ⚠️ **Unstable** (Plotting API)

---

### `plot_spatial_anomaly(data, lons, lats, *, region=None, lon_bounds=None, lat_bounds=None, projection=None, cmap='RdBu_r', vmin=None, vmax=None, title=None, units=None, add_coastlines=True, add_gridlines=True, colorbar=True, ax=None)`

Plot an anomaly field on a cartopy map projection. Same parameters as `plot_spatial_map()` with a diverging colormap default.

**Example:**
```python
from utils_cmip7.processing.map_fields import extract_anomaly_field
from utils_cmip7.plotting.maps import plot_spatial_anomaly

a = extract_anomaly_field(cube, time_a=2000, time_b=1850, variable='GPP')
fig, ax = plot_spatial_anomaly(
    a['data'], a['lons'], a['lats'],
    vmin=a['vmin'], vmax=a['vmax'],
    title=a['title'], units=a['units'],
)
```

**Stability:** ⚠️ **Unstable** (Plotting API)

---

## PPE Validation Plotting

Functions for visualising Perturbed Physics Ensemble (PPE) validation tables. Located in `utils_cmip7.plotting.ppe_viz` (all exported from `utils_cmip7.plotting`). Input is a CSV overview table — no NetCDF I/O.

---

### `plot_param_scatter(df, param_cols, y_col, *, id_col, obs_values, ncols, highlight_col, highlight_label, sentinel, sentinel_replacement, title)` → `Figure`

Multi-panel scatter plot: one panel per parameter (X axis) against a metric or score (Y axis). Panels are arranged in a grid (`ncols=4` default). Unused cells are hidden.

**Parameters:**
- `df` (DataFrame): Ensemble overview table
- `param_cols` (sequence of str): Parameter columns to use as X axes
- `y_col` (str): Metric or score column for the Y axis
- `id_col` (str, optional): Column used to label highlighted points. Default: `"ID"`
- `obs_values` (dict, optional): `{varname: observed_value}` — draws a dashed reference line via `add_observation_lines`
- `ncols` (int): Subplot columns. Default: `4`
- `highlight_col` (str, optional): Boolean column; matching rows are over-plotted with diamond markers
- `highlight_label` (bool): Add ID labels to highlighted points. Default: `True`
- `sentinel` (float): Sentinel value to replace in parameter columns. Default: `-9999`
- `sentinel_replacement` (float): Replacement for sentinels. Default: `0.5`
- `title` (str, optional): Figure suptitle

**Returns:** `matplotlib.figure.Figure`

**Raises:** `ValueError` if none of `param_cols` exist in `df`.

**Example:**
```python
from utils_cmip7.plotting import plot_param_scatter
import pandas as pd

df = pd.read_csv("ensemble_table.csv")
fig = plot_param_scatter(
    df,
    param_cols=["ALPHA", "G_AREA", "LAI_MIN", "NL0", "R_GROW", "TLOW", "V_CRIT"],
    y_col="overall_score",
    highlight_col="_highlight",
    title="Overall skill vs parameters",
)
fig.savefig("scatter.pdf")
```

**Stability:** ⚠️ **Unstable** (Plotting API)

---

### `add_observation_lines(ax, varname, obs_values)`

Draw a dashed horizontal reference line at the observed value for `varname`. No-op if `obs_values` is `None` or `varname` is not a key.

**Parameters:**
- `ax` (Axes): Target matplotlib axes
- `varname` (str): Variable name to look up
- `obs_values` (dict or None): `{varname: scalar}`

**Stability:** ⚠️ **Unstable** (Plotting API)

---

### `save_overall_skill_param_scatter_pdf(df, out_pdf, *, param_cols, score_col, id_col, highlight_col, highlight_label, ncols, title)`

Save a multi-panel scatter of **overall skill score vs each parameter** to a PDF. TUPP is excluded automatically.

**Typical output filename:** `{ensemble_name}_overall_skill_core_param_scatter.pdf`

**Raises:** `ValueError` if no valid parameter columns are found.

**Stability:** ⚠️ **Unstable** (Plotting API)

---

### `save_param_scatter_pdf(df, metric, out_pdf, *, param_cols, obs_values, id_col, highlight_col, highlight_label, ncols, title)`

Save a multi-panel scatter of **a climatological metric vs each parameter** to a PDF. TUPP is excluded automatically. Draws observation reference lines when `obs_values` is provided.

**Typical output filename:** `{ensemble_name}_{metric}_param_scatter.pdf`

**Raises:** `KeyError` if `metric` is not a column in `df`.

**Stability:** ⚠️ **Unstable** (Plotting API)

---

### `generate_ppe_validation_report(csv_path, ensemble_name, ...)` (updated)

Now produces additional scatter PDFs in the output directory:

```
validation_outputs/ppe_{name}/
    ├── ...existing files...
    ├── {name}_overall_skill_core_param_scatter.pdf
    ├── {name}_GPP_param_scatter.pdf
    ├── {name}_NPP_param_scatter.pdf
    ├── {name}_CVeg_param_scatter.pdf
    ├── {name}_CSoil_param_scatter.pdf
    ├── {name}_GM_BL_param_scatter.pdf
    ├── {name}_GM_NL_param_scatter.pdf
    ├── {name}_GM_C3_param_scatter.pdf
    ├── {name}_GM_C4_param_scatter.pdf
    └── {name}_GM_BS_param_scatter.pdf
```

Per-metric PDFs are only generated for metrics that exist as columns in the input CSV.

---

## Diagnostics Functions

### High-Level Extraction

#### `extract_annual_means(expts_list, var_list=None, regions=None, base_dir='~/annual_mean')`

Extract annual means from pre-processed NetCDF files.

**Main entry point** for extracting carbon cycle variables.

**⚡ Performance**: Optimized with module-level mask caching (3× speedup). RECCAP2 regional mask loaded once per extraction and cached in memory.

**Parameters:**
- `expts_list` (list of str): Experiment names (e.g., ['xqhuc', 'xqhsh'])
- `var_list` (list of str, optional): Variable names (canonical names only).
  Default: ['Rh', 'CSoil', 'CVeg', 'frac', 'GPP', 'NPP', 'fgco2', 'tas', 'pr', 'co2']
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
        'description': 'Gross Primary Production',
        'stash_name': 'gpp',
        'stash_code': 'm01s03i261',
        'aggregation': 'SUM',
        'conversion_factor': 3600*24*360*1e-12,  # kgC/m2/s -> PgC/yr
        'units': 'PgC/yr',
        'category': 'flux',
        'aliases': [],
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
- `var_name` (str): Canonical variable name (e.g., 'CVeg', 'Rh', 'tas')

**Returns:**
- str: Canonical variable name

**Raises:**
- ValueError: If variable name not recognized or a removed legacy alias is used

**Example:**
```python
from utils_cmip7 import resolve_variable_name

canonical = resolve_variable_name('GPP')      # Returns 'GPP'
canonical = resolve_variable_name('CVeg')     # Returns 'CVeg'

# Legacy aliases raise ValueError:
resolve_variable_name('VegCarb')  # ValueError: removed in v0.4.0
```

**Stability:** ✅ **Stable**

---

#### `get_variable_config(var_name)`

Get configuration dictionary for a variable.

**Parameters:**
- `var_name` (str): Canonical variable name

**Returns:**
- dict: Variable configuration with keys: `description`, `stash_name`, `stash_code`, `aggregation`, `conversion_factor`, `units`, `category`, `aliases`, `canonical_name`

**Raises:**
- ValueError: If variable name not recognized

**Stability:** ✅ **Stable**

---

#### `get_conversion_key(var_name)`

Get unit conversion key for `compute_regional_annual_mean()`.

**Parameters:**
- `var_name` (str): Canonical variable name

**Returns:**
- str: Conversion key for use with `compute_regional_annual_mean()` (e.g., `'GPP'`, `'Others'`, `'precip'`, `'Total co2'`)

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

- **Stable**: No breaking changes in v0.4.x series. Safe to use in production.
- **Provisional**: Minor additions allowed, but no breaking changes to existing functionality.
- **Unstable**: Breaking changes possible in minor versions (v0.4.x → v0.4.y).
- **Experimental**: No stability guarantees. May change significantly or be removed.

---

## Migration from v0.3.x to v0.4.0

### Breaking Changes

1. **`var_mapping` parameter removed** from `extract_annual_means()`:
   ```python
   # v0.3.x (deprecated, raised warning):
   extract_annual_means(expts, var_mapping=['gpp', 'npp'])

   # v0.4.0 (raises TypeError):
   extract_annual_means(expts, var_list=['GPP', 'NPP'])
   ```

2. **Legacy variable names raise ValueError**:
   ```python
   # v0.3.x (deprecated, resolved with warning):
   resolve_variable_name('VegCarb')  # returned 'CVeg'

   # v0.4.0 (raises ValueError):
   resolve_variable_name('VegCarb')  # ValueError: removed in v0.4.0. Use 'CVeg' instead.
   ```

3. **`var_dict` and `DEFAULT_VAR_MAPPING` removed** from config module

4. **Python 3.8 no longer supported** — minimum is Python 3.9

### Migration Guide

Replace all legacy variable names with canonical equivalents:
- `VegCarb` -> `CVeg`
- `soilResp` -> `Rh`
- `soilCarbon` -> `CSoil`
- `temp` -> `tas`
- `precip` -> `pr`
- `fracPFTs` -> `frac`

---

## Version History

- **v0.4.0** (2026-02-09): Breaking — remove deprecated features, drop Python 3.8
- **v0.3.1** (2026-02-09): Canonical name resolution, unit conversion, plotting fixes
- **v0.3.0** (2026-01-26): API stabilization, deprecation grace period, comprehensive testing
- **v0.2.1** (2025-XX-XX): Canonical variable registry, validation improvements
- **v0.2.0** (2025-XX-XX): Major refactoring, package structure cleanup

---

## Support & Feedback

- **Issues**: https://github.com/Climateyousheng/utils_cmip7/issues
- **Discussions**: https://github.com/Climateyousheng/utils_cmip7/discussions

---

**Last updated:** 2026-02-09 (v0.4.0)
**Maintainer:** Yousheng Li

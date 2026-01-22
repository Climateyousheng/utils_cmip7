# utils_cmip7 — API Reference

This document provides the **developer-facing API reference** for `utils_cmip7`.

It is **descriptive**, not normative.
Architectural constraints and stability guarantees are defined in `CLAUDE.md`.

---

## Package Layout (v0.2.1)

```
utils_cmip7/
├── io/                 # NetCDF loading, STASH handling, file discovery
│   ├── stash.py       # STASH code mappings
│   ├── file_discovery.py  # UM file pattern matching
│   ├── extract.py     # Cube extraction with STASH handling
│   └── obs_loader.py  # Observational data loader (CMIP6, RECCAP2)
├── processing/        # Temporal/spatial aggregation, unit conversions
│   ├── spatial.py     # Global aggregation (SUM/MEAN)
│   ├── temporal.py    # Monthly → annual aggregation
│   ├── regional.py    # RECCAP2 regional masking
│   └── metrics.py     # Metric definitions and canonical schema validation
├── diagnostics/       # Carbon-cycle diagnostics
│   ├── extraction.py  # Pre-processed NetCDF extraction
│   ├── raw.py         # Raw monthly file extraction
│   └── metrics.py     # Metrics computation from annual means
├── validation/        # Model validation against observations
│   ├── compare.py     # Bias, RMSE, and uncertainty checks
│   └── visualize.py   # Three-way comparison plots and visualizations
├── plotting/          # Visualisation utilities (no I/O)
│   └── [TODO]         # To be split from plot.py
├── soil_params/       # Soil parameter analysis
│   └── [TODO]         # To be migrated from root
├── config.py          # Configuration and constants
└── cli.py             # [TODO] Command-line entry points
```

**Current Status (v0.2.1):**
- ✅ `io/` - Complete (4 modules including obs_loader)
- ✅ `processing/` - Complete (4 modules including metrics)
- ✅ `diagnostics/` - Complete (3 modules including metrics)
- ✅ `validation/` - Complete (2 modules: compare, visualize)
- ⚠️ `plotting/` - Exists in root `plot.py`, needs migration
- ⚠️ `soil_params/` - Exists in root, needs migration
- ❌ `cli.py` - Not yet implemented

---

## I/O Layer (`utils_cmip7.io`)

### STASH Code Functions

#### `stash(var_name: str) -> str`
Map short variable name to MSI-format STASH code string.

**Parameters:**
- `var_name` (str): Short name (e.g., 'gpp', 'npp', 'tas')

**Returns:**
- str: MSI code (e.g., 'm01s03i261') or "nothing" if not found

**Example:**
```python
from utils_cmip7.io import stash
code = stash('gpp')  # Returns 'm01s03i261'
```

**Supported Variables:**
tas, pr, gpp, npp, rh, cv, cs, dist, frac, ocn, emiss, co2, tos, sal, tco2, alk, nut, phy, zoo, detn, detc, pco2, fgco2, rlut, rlutcs, rsdt, rsut, rsutcs

---

#### `stash_nc(var_name: str) -> int | str`
Map short variable name to numeric STASH code.

**Parameters:**
- `var_name` (str): Short name

**Returns:**
- int: Numeric code (e.g., 3261) or "nothing" if not found

**Example:**
```python
from utils_cmip7.io import stash_nc
code = stash_nc('gpp')  # Returns 3261
```

---

### File Discovery

#### `find_matching_files(expt_name, model, up, start_year=None, end_year=None, base_dir='~/dump2hold')`
Find raw UM output files matching experiment pattern.

**Parameters:**
- `expt_name` (str): Experiment name (e.g., 'xqhuj')
- `model` (str): Model identifier (e.g., 'a', 'o')
- `up` (str): Stream identifier (e.g., 'pi', 'da')
- `start_year` (int, optional): Filter from this year onwards
- `end_year` (int, optional): Filter up to this year
- `base_dir` (str): Base directory (default: '~/dump2hold')

**Returns:**
- list of tuple: Sorted list of `(year, month, filepath)` tuples

**Example:**
```python
from utils_cmip7.io import find_matching_files
files = find_matching_files('xqhuj', 'a', 'pi', start_year=1850, end_year=1900)
```

**Supported Month Codes:**
- Alpha: ja (Jan), fb (Feb), mr (Mar), ar (Apr), my (May), jn (Jun), jl (Jul), ag (Aug), sp (Sep), ot (Oct), nv (Nov), dc (Dec)
- Numeric: 11-91 (Jan-Sep), a1 (Oct), b1 (Nov), c1 (Dec)

---

#### `decode_month(mon_code: str) -> int`
Decode UM-style two-letter month code to month number.

**Parameters:**
- `mon_code` (str): Two-character month code

**Returns:**
- int: Month number (1-12) or 0 if unparseable

---

### Cube Extraction

#### `try_extract(cubes, code, stash_lookup_func=None, debug=False)`
Robust extraction of cubes by STASH code with flexible format handling.

**Parameters:**
- `cubes` (iris.cube.CubeList): Collection of cubes to search
- `code` (str | int): STASH code in various formats:
  - MSI string: 'm01s03i261'
  - Short name: 'gpp' (requires `stash_lookup_func`)
  - Numeric code: 3261
- `stash_lookup_func` (callable, optional): Function to map short names to MSI
- `debug` (bool): Print debug information

**Returns:**
- iris.cube.CubeList: Matching cubes or empty list

**Example:**
```python
from utils_cmip7.io import try_extract, stash
import iris

cubes = iris.load('data.nc')
gpp = try_extract(cubes, 'gpp', stash_lookup_func=stash)
```

---

### Observational Data Loading

#### `load_cmip6_metrics(metrics=None, regions=None, include_errors=False)`
Load CMIP6 ensemble metrics from CSV files in canonical schema format.

**Parameters:**
- `metrics` (list of str, optional): Metrics to load (default: all available)
- `regions` (list of str, optional): Regions to load (default: all available)
- `include_errors` (bool): Load error/uncertainty data (default: False)

**Returns:**
```python
dict[metric][region] -> {
    'years': np.ndarray,      # Empty array for time-aggregated data
    'data': np.ndarray,       # Mean values
    'units': str,             # Units (e.g., 'PgC/yr')
    'source': 'CMIP6',
    'dataset': 'CMIP6',
    'errors': np.ndarray      # Only if include_errors=True
}
```

**Input Files:**
- `obs/stores_vs_fluxes_cmip6.csv` - Mean values
- `obs/stores_vs_fluxes_cmip6_err.csv` - Error/uncertainty values

**Example:**
```python
from utils_cmip7.io import load_cmip6_metrics

# Load specific metrics and regions
cmip6 = load_cmip6_metrics(
    metrics=['GPP', 'NPP', 'CVeg'],
    regions=['global', 'Europe', 'Africa'],
    include_errors=True
)

print(cmip6['GPP']['global']['data'])  # CMIP6 ensemble mean
print(cmip6['GPP']['global']['errors'])  # CMIP6 ensemble std dev
```

---

#### `load_reccap_metrics(metrics=None, regions=None, include_errors=False)`
Load RECCAP2 observational metrics from CSV files in canonical schema format.

**Parameters:**
- `metrics` (list of str, optional): Metrics to load (default: all available)
- `regions` (list of str, optional): Regions to load (default: all available)
- `include_errors` (bool): Load error/uncertainty data (default: False)

**Returns:**
```python
dict[metric][region] -> {
    'years': np.ndarray,      # Empty array for time-aggregated data
    'data': np.ndarray,       # Observational best estimate
    'units': str,             # Units (e.g., 'PgC/yr')
    'source': 'RECCAP2',
    'dataset': 'RECCAP2',
    'errors': np.ndarray      # Only if include_errors=True
}
```

**Input Files:**
- `obs/stores_vs_fluxes_reccap.csv` - Best estimates
- `obs/stores_vs_fluxes_reccap_err.csv` - Uncertainty values

**Example:**
```python
from utils_cmip7.io import load_reccap_metrics

# Load all available metrics
reccap = load_reccap_metrics(include_errors=True)

# Access specific metric/region
gpp_global = reccap['GPP']['global']
print(f"GPP: {gpp_global['data'][0]} ± {gpp_global['errors'][0]} {gpp_global['units']}")
```

---

## Processing Layer (`utils_cmip7.processing`)

### Spatial Aggregation

#### `global_total_pgC(cube, var)`
Compute area-weighted global total with unit conversion.

**Parameters:**
- `cube` (iris.cube.Cube | iris.cube.CubeList): Input cube
- `var` (str): Variable name for unit conversion

**Returns:**
- iris.cube.Cube: Collapsed cube with global total

**Aggregation:** SUM with area weights

**Example:**
```python
from utils_cmip7.processing import global_total_pgC
import iris

gpp_cube = iris.load_cube('gpp.nc')
gpp_global = global_total_pgC(gpp_cube, 'GPP')  # Result in PgC/year
```

---

#### `global_mean_pgC(cube, var)`
Compute area-weighted global mean with unit conversion.

**Parameters:**
- `cube` (iris.cube.Cube | iris.cube.CubeList): Input cube
- `var` (str): Variable name for unit conversion

**Returns:**
- iris.cube.Cube: Collapsed cube with global mean

**Aggregation:** MEAN with area weights

**Example:**
```python
from utils_cmip7.processing import global_mean_pgC
import iris

precip_cube = iris.load_cube('precip.nc')
precip_global = global_mean_pgC(precip_cube, 'precip')  # Result in mm/day
```

---

### Temporal Aggregation

#### `compute_monthly_mean(cube, var)`
Compute area-weighted global total for each month.

**Parameters:**
- `cube` (iris.cube.Cube): Input cube with time dimension
- `var` (str): Variable name for unit conversion

**Returns:**
- dict: `{'years': fractional_years, 'data': monthly_values, 'name': str, 'units': str}`

**Example:**
```python
from utils_cmip7.processing import compute_monthly_mean
monthly = compute_monthly_mean(gpp_cube, 'GPP')
print(monthly['years'])  # e.g., [1850.0, 1850.083, 1850.167, ...]
```

---

#### `compute_annual_mean(cube, var)`
Compute area-weighted annual means from monthly data.

**Parameters:**
- `cube` (iris.cube.Cube): Input cube with time dimension
- `var` (str): Variable name (use 'Others' for MEAN aggregation)

**Returns:**
- dict: `{'years': integer_years, 'data': annual_values, 'name': str, 'units': str}`

**Example:**
```python
from utils_cmip7.processing import compute_annual_mean
annual = compute_annual_mean(gpp_cube, 'GPP')
print(annual['years'])  # e.g., [1850, 1851, 1852, ...]
```

---

#### `merge_monthly_results(results, require_full_year=False)`
Merge multiple monthly outputs into annual means.

**Parameters:**
- `results` (list of dict): List of monthly result dictionaries
- `require_full_year` (bool): Only return years with 12 months

**Returns:**
- dict: `{'years': integer_years, 'data': annual_means}`

---

### Regional Aggregation

#### `load_reccap_mask()`
Load RECCAP2 regional mask.

**Returns:**
- tuple: `(reccap_mask_cube, regions_dict)`

**Environment Variable:**
- `UTILS_CMIP7_RECCAP_MASK`: Override default mask path

---

#### `region_mask(region: str)`
Generate binary mask for specific RECCAP2 region.

**Parameters:**
- `region` (str): Region name (e.g., 'Europe', 'Africa', 'North_America')

**Returns:**
- iris.cube.Cube: Binary mask (1=region, 0=elsewhere)

**Special Case:** 'Africa' combines regions 4 and 5

---

#### `compute_regional_annual_mean(cube, var, region)`
Compute area-weighted regional annual means.

**Parameters:**
- `cube` (iris.cube.Cube): Input cube
- `var` (str): Variable name
- `region` (str): Region name or 'global'

**Returns:**
- dict: `{'years': array, 'data': array, 'name': str, 'units': str, 'region': str}`

**Aggregation:**
- SUM: Most variables (fluxes, stocks)
- MEAN: var in ('Others', 'precip')

**Example:**
```python
from utils_cmip7.processing import compute_regional_annual_mean
europe_gpp = compute_regional_annual_mean(gpp_cube, 'GPP', 'Europe')
```

---

### Metric Definitions and Validation

#### `METRIC_DEFINITIONS`
Dictionary defining all supported carbon cycle metrics.

**Type:** dict[str, dict]

**Structure:**
```python
{
    'GPP': {
        'long_name': 'Gross Primary Production',
        'units': 'PgC/yr',
        'aggregation': 'sum',
        'required_vars': ['GPP']
    },
    'Tau': {
        'long_name': 'Ecosystem Turnover Time',
        'units': 'years',
        'aggregation': 'derived',
        'required_vars': ['CVeg', 'NPP']
    },
    ...
}
```

**Supported Metrics:**
- GPP, NPP, CVeg, CSoil, Tau, NEP, CTotal, TreeTotal
- Includes units, aggregation method, and required variables

---

#### `validate_metric_output(metric_data, metric_name=None)`
Validate metric output conforms to canonical schema.

**Parameters:**
- `metric_data` (dict): Metric dictionary to validate
- `metric_name` (str, optional): Metric name for error messages

**Returns:**
- bool: True if valid

**Raises:**
- ValueError: If validation fails

**Canonical Schema:**
```python
{
    'years': np.ndarray,      # Integer years or empty for time-aggregated
    'data': np.ndarray,       # Same length as years (or length 1 if empty years)
    'units': str,             # Required
    'source': str,            # 'UM', 'CMIP6', 'RECCAP2'
    'dataset': str            # Dataset identifier
}
```

**Special Case:** Time-aggregated data (observational data)
- `years` can be empty array `np.array([])`
- `data` must have exactly one element

**Example:**
```python
from utils_cmip7.processing.metrics import validate_metric_output

metric = {
    'years': np.array([1850, 1851, 1852]),
    'data': np.array([120.5, 121.3, 119.8]),
    'units': 'PgC/yr',
    'source': 'UM',
    'dataset': 'xqhuc'
}

validate_metric_output(metric, 'GPP')  # Returns True or raises ValueError
```

---

## Diagnostics Layer (`utils_cmip7.diagnostics`)

### High-Level Extraction

#### `extract_annual_means(expts_list, var_list=None, var_mapping=None, regions=None)`
Extract annual means from pre-processed NetCDF files.

**Main entry point** for carbon cycle diagnostics from annual mean files.

**Parameters:**
- `expts_list` (list of str): Experiment names (e.g., ['xqhuc', 'xqhsh'])
- `var_list` (list of str, optional): Variables to extract (default: 9 standard vars)
- `var_mapping` (list of str, optional): Unit conversion mappings
- `regions` (list of str, optional): Regions to process (default: all RECCAP2 + global)

**Returns:**
```python
dict[expt][region][variable] -> {
    'years': np.ndarray,
    'data': np.ndarray,
    'units': str,
    'name': str,
    'region': str
}
```

**Default Variables:**
soilResp, soilCarbon, VegCarb, fracPFTs, GPP, NPP, fgco2, temp, precip

**Derived Variables (auto-computed):**
- NEP = NPP - soilResp
- Land Carbon = soilCarbon + VegCarb + NEP
- Trees Total = PFT1 + PFT2

**Input Files (expected):**
- `~/annual_mean/{expt}/{expt}_pt_annual_mean.nc` (TRIFFID)
- `~/annual_mean/{expt}/{expt}_pd_annual_mean.nc` (atmosphere)
- `~/annual_mean/{expt}/{expt}_pf_annual_mean.nc` (ocean)

**Example:**
```python
from utils_cmip7 import extract_annual_means

# Extract for single experiment
ds = extract_annual_means(['xqhuc'])
gpp_global = ds['xqhuc']['global']['GPP']['data']

# Extract specific regions
ds = extract_annual_means(['xqhuc'], regions=['global', 'Europe', 'Africa'])
europe_npp = ds['xqhuc']['Europe']['NPP']['data']
```

---

#### `extract_annual_mean_raw(expt, base_dir='~/dump2hold', start_year=None, end_year=None)`
Extract annual means from raw monthly UM output files.

**Use case:** Process raw monthly files directly without pre-processed annual means.

**Parameters:**
- `expt` (str): Experiment name
- `base_dir` (str): Base directory containing raw files
- `start_year` (int, optional): First year to process
- `end_year` (int, optional): Last year to process

**Returns:**
```python
dict[variable] -> {
    'years': np.ndarray,
    'data': np.ndarray,
    'units': str,
    'name': str
}
```

**Variables Extracted:**
GPP, NPP, soilResp, VegCarb, soilCarbon

**Derived Variable:**
- NEP = NPP - soilResp

**Input Files (expected):**
- `~/dump2hold/{expt}/datam/{expt}a#pi00000{YYYY}{MM}+`

**Example:**
```python
from utils_cmip7 import extract_annual_mean_raw

# Extract all years
data = extract_annual_mean_raw('xqhuj')
gpp_data = data['GPP']['data']

# Extract specific range
data = extract_annual_mean_raw('xqhuj', start_year=1850, end_year=1900)
```

---

### Metrics Computation

#### `compute_metrics_from_annual_means(expt_name, metrics=None, regions=None, base_dir=None)`
Compute carbon cycle metrics from annual mean NetCDF files for all RECCAP2 regions.

**Main entry point** for validation workflows requiring regional metrics.

**Parameters:**
- `expt_name` (str): Experiment name (e.g., 'xqhuc')
- `metrics` (list of str, optional): Metrics to compute (default: GPP, NPP, CVeg, CSoil, Tau, NEP)
- `regions` (list of str, optional): Regions to compute (default: global + 11 RECCAP2 regions)
- `base_dir` (str, optional): Base directory for annual mean files (default: ~/annual_mean)

**Returns:**
```python
dict[metric][region] -> {
    'years': np.ndarray,      # Integer years
    'data': np.ndarray,       # Metric values
    'units': str,             # Units (e.g., 'PgC/yr')
    'source': 'UM',
    'dataset': str            # Experiment name
}
```

**Supported Metrics:**
- **GPP**: Gross Primary Production (PgC/yr)
- **NPP**: Net Primary Production (PgC/yr)
- **CVeg**: Vegetation Carbon (PgC)
- **CSoil**: Soil Carbon (PgC)
- **Tau**: Ecosystem Turnover Time (years) = CVeg + CSoil / NPP
- **NEP**: Net Ecosystem Production (PgC/yr) = NPP - soilResp

**Supported Regions:**
global, North_America, South_America, Europe, Africa, North_Asia, Central_Asia, East_Asia, South_Asia, South_East_Asia, Oceania

**Input Files (expected):**
- `{base_dir}/{expt_name}/{expt_name}_pt_annual_mean.nc` - TRIFFID land carbon

**Graceful Degradation:**
- Missing variables result in missing metrics (not errors)
- Missing regions are skipped silently
- Derived metrics (Tau, NEP) computed only when components available

**Example:**
```python
from utils_cmip7.diagnostics import compute_metrics_from_annual_means

# Compute all metrics for all regions
um_metrics = compute_metrics_from_annual_means('xqhuc')

# Access specific metric/region
gpp_global = um_metrics['GPP']['global']
print(f"Years: {gpp_global['years']}")
print(f"GPP: {gpp_global['data']} {gpp_global['units']}")

# Compute specific metrics and regions only
um_metrics = compute_metrics_from_annual_means(
    'xqhuc',
    metrics=['GPP', 'NPP'],
    regions=['global', 'Europe', 'Africa']
)
```

**Use Cases:**
- Model validation against CMIP6/RECCAP2
- Regional carbon cycle analysis
- Multi-experiment comparison
- Export to CSV for external analysis

---

## Validation Layer (`utils_cmip7.validation`)

### Comparison Functions

#### `compute_bias(model_data, obs_data)`
Compute absolute and percentage bias of model relative to observations.

**Parameters:**
- `model_data` (dict): Model metric in canonical schema
- `obs_data` (dict): Observational metric in canonical schema

**Returns:**
- tuple: `(bias, bias_percent)`
  - `bias` (float): Absolute bias (model - obs)
  - `bias_percent` (float): Percentage bias (100 * bias / obs)

**Example:**
```python
from utils_cmip7.validation import compute_bias

um = {'data': np.array([134.8]), 'units': 'PgC/yr'}
reccap = {'data': np.array([124.0]), 'units': 'PgC/yr'}

bias, bias_pct = compute_bias(um, reccap)
print(f"Bias: {bias:.2f} PgC/yr ({bias_pct:.1f}%)")
```

---

#### `compute_rmse(model_data, obs_data)`
Compute root mean square error between model and observations.

**Parameters:**
- `model_data` (dict): Model metric with time series
- `obs_data` (dict): Observational metric (time-aggregated or time series)

**Returns:**
- float: RMSE value

**Note:** If obs is time-aggregated (single value), compares against all model time steps.

---

#### `compare_single_metric(model_metrics, obs_metrics, metric, region)`
Compare model vs observation for a single metric and region.

**Parameters:**
- `model_metrics` (dict): Model metrics dictionary
- `obs_metrics` (dict): Observational metrics dictionary
- `metric` (str): Metric name (e.g., 'GPP')
- `region` (str): Region name (e.g., 'global')

**Returns:**
```python
{
    'metric': str,
    'region': str,
    'model_mean': float,
    'obs_mean': float,
    'bias': float,
    'bias_percent': float,
    'rmse': float,
    'within_uncertainty': bool   # If obs has errors
}
```

---

#### `compare_metrics(model_metrics, obs_metrics, metrics=None, regions=None)`
Compare model vs observations for multiple metrics and regions.

**Parameters:**
- `model_metrics` (dict): Model metrics
- `obs_metrics` (dict): Observational metrics
- `metrics` (list of str, optional): Metrics to compare (default: all common)
- `regions` (list of str, optional): Regions to compare (default: all common)

**Returns:**
- list of dict: Comparison results for each metric/region combination

**Example:**
```python
from utils_cmip7.validation import compare_metrics
from utils_cmip7.diagnostics import compute_metrics_from_annual_means
from utils_cmip7.io import load_reccap_metrics

um = compute_metrics_from_annual_means('xqhuc')
reccap = load_reccap_metrics(include_errors=True)

comparisons = compare_metrics(um, reccap,
                               metrics=['GPP', 'NPP'],
                               regions=['global', 'Europe'])

for comp in comparisons:
    print(f"{comp['metric']} {comp['region']}: bias = {comp['bias_percent']:.1f}%")
```

---

#### `summarize_comparison(comparisons, reference_comparisons=None)`
Generate text summary comparing model vs observations with optional reference model.

**Parameters:**
- `comparisons` (list of dict): Comparison results (e.g., UM vs RECCAP2)
- `reference_comparisons` (list of dict, optional): Reference comparisons (e.g., CMIP6 vs RECCAP2)

**Returns:**
- str: Multi-line text summary

**Example:**
```python
from utils_cmip7.validation import summarize_comparison

summary = summarize_comparison(um_vs_reccap, cmip6_vs_reccap)
print(summary)
# Outputs comparison table showing UM vs CMIP6 performance
```

---

### Visualization Functions

#### `plot_three_way_comparison(um_metrics, cmip6_metrics, reccap_metrics, metric, outdir=None, ax=None)`
Create three-way comparison plot showing UM, CMIP6, and RECCAP2 with uncertainty.

**Main visualization** for model validation showing relative performance.

**Parameters:**
- `um_metrics` (dict): UM metrics from `compute_metrics_from_annual_means()`
- `cmip6_metrics` (dict): CMIP6 metrics from `load_cmip6_metrics()`
- `reccap_metrics` (dict): RECCAP2 metrics from `load_reccap_metrics()`
- `metric` (str): Metric to plot (e.g., 'GPP')
- `outdir` (str, optional): Output directory for saved plot
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on (if None, creates figure)

**Returns:**
- matplotlib.axes.Axes: Plot axes

**Plot Features:**
- Bar chart with UM, CMIP6, RECCAP2 for each region
- Error bars showing observational uncertainty
- Automatic region intersection (only plots common regions)
- Color-coded for easy comparison

**Example:**
```python
from utils_cmip7.validation import plot_three_way_comparison

plot_three_way_comparison(
    um_metrics, cmip6_metrics, reccap_metrics,
    metric='GPP',
    outdir='./validation'
)
# Saves: ./validation/GPP_three_way.png
```

---

#### `plot_regional_bias_heatmap(comparisons, metrics, regions, title, outdir=None, ax=None)`
Create heatmap showing regional bias patterns across multiple metrics.

**Parameters:**
- `comparisons` (list of dict): Comparison results
- `metrics` (list of str): Metrics to include in heatmap
- `regions` (list of str): Regions to include
- `title` (str): Plot title
- `outdir` (str, optional): Output directory
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on

**Returns:**
- matplotlib.axes.Axes: Plot axes

**Plot Features:**
- Color-coded bias (red=positive, blue=negative)
- Regions on x-axis, metrics on y-axis
- Useful for identifying regional patterns

---

#### `plot_timeseries_with_obs(model_metrics, obs_metrics, metric, region, outdir=None, ax=None)`
Plot model time series with observational constraint and uncertainty.

**Parameters:**
- `model_metrics` (dict): Model metrics with time series
- `obs_metrics` (dict): Observational metrics with uncertainty
- `metric` (str): Metric name
- `region` (str): Region name
- `outdir` (str, optional): Output directory
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on

**Returns:**
- matplotlib.axes.Axes: Plot axes

**Plot Features:**
- Model time series as line
- Observational best estimate as horizontal line
- Uncertainty range as shaded region

---

#### `create_validation_report(model_metrics, obs_metrics, metrics, regions, outdir)`
Create comprehensive validation report with all plots and statistics.

**Parameters:**
- `model_metrics` (dict): Model metrics
- `obs_metrics` (dict): Observational metrics
- `metrics` (list of str): Metrics to include
- `regions` (list of str): Regions to include
- `outdir` (str): Output directory

**Creates:**
- Comparison plots for each metric
- Regional bias heatmap
- Time series plots
- Summary statistics CSV

**Example:**
```python
from utils_cmip7.validation import create_validation_report

create_validation_report(
    um_metrics, reccap_metrics,
    metrics=['GPP', 'NPP', 'CVeg'],
    regions=['global', 'Europe', 'Africa'],
    outdir='./validation/xqhuc'
)
```

---

## Configuration (`utils_cmip7.config`)

### Constants

#### `VAR_CONVERSIONS`
Dictionary mapping variable names to unit conversion factors.

**Type:** dict[str, float]

**Example:**
```python
from utils_cmip7 import VAR_CONVERSIONS
print(VAR_CONVERSIONS['GPP'])  # 3.11e-05 (kgC/m²/s → PgC/yr)
```

**Key Conversions:**
- GPP, NPP, S resp: kgC/m²/s → PgC/yr
- V carb, S carb: kgC/m² → PgC
- Ocean flux: kgCO2/m²/s → PgC/yr
- Air flux: molC/m²/yr → PgC/yr
- precip: kg/m²/s → mm/day

---

#### `RECCAP_MASK_PATH`
Path to RECCAP2 regional mask file.

**Type:** str

**Default:** `~/scripts/hadcm3b-ensemble-validator/observations/RECCAP_AfricaSplit_MASK11_Mask_regridded.hadcm3bl_grid.nc`

**Override:** Set environment variable `UTILS_CMIP7_RECCAP_MASK`

**Example:**
```bash
export UTILS_CMIP7_RECCAP_MASK=/path/to/custom/mask.nc
python script.py
```

---

#### `RECCAP_REGIONS`
Dictionary of RECCAP2 region IDs and names.

**Type:** dict[int, str]

**Regions:**
1. North_America
2. South_America
3. Europe
4. Africa
6. North_Asia
7. Central_Asia
8. East_Asia
9. South_Asia
10. South_East_Asia
11. Oceania

---

## Plotting Layer

> **Status:** Not yet migrated to `src/utils_cmip7/plotting/`
>
> Current implementation in root `plot.py` file.

Plotting functions are documented in `plot.py` and will be migrated in v0.2.2.

**Target API (post-migration):**
- Functions must accept `ax` parameter (matplotlib Axes)
- Functions must not perform data loading or aggregation
- Data must be passed as input from diagnostics layer

---

## CLI (Experimental)

**Status:** Not yet implemented

**Planned Entry Points (pyproject.toml):**
- `utils-cmip7-extract-raw` → Extract from raw monthly files
- `utils-cmip7-extract-preprocessed` → Extract from annual mean NetCDF

**Planned Implementation:**
`src/utils_cmip7/cli.py` with `extract_raw_cli()` and `extract_preprocessed_cli()`

---

## Stability Notes

Refer to `CLAUDE.md` Section 3: API Stability Matrix for stability guarantees.

**Current Stability (v0.2.1):**
- io/ - **Provisional** (includes obs_loader)
- processing/ - **Provisional** (includes metrics)
- diagnostics/ - **Provisional** (includes metrics computation)
- validation/ - **Provisional** (compare, visualize)
- plotting/ - **Unstable** (not yet migrated)
- cli/ - **Experimental** (not yet implemented)

---

## Version History

- **v0.2.1** (Current): Model validation framework complete
  - Added `validation/` module (compare, visualize)
  - Added `io/obs_loader.py` for CMIP6/RECCAP2 data loading
  - Added `processing/metrics.py` for metric definitions and validation
  - Added `diagnostics/metrics.py` for regional metrics computation
  - Added `scripts/validate_experiment.py` for high-level validation workflow
  - Three-way comparison visualization (UM vs CMIP6 vs RECCAP2)
  - CSV export in observational format
  - Graceful handling of missing metrics and regions
- **v0.2.0**: Core extraction functionality complete, package structure established
- **v0.1.0**: Initial scripts and functions

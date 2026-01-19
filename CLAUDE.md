# CLAUDE.md

This file provides guidance to Claude Code when working with the `utils_cmip7` Python package for analyzing Unified Model (UM) climate model outputs.

## Overview

`utils_cmip7` is a Python toolkit for carbon cycle analysis from UM climate model outputs. It provides:
- STASH code mapping and variable extraction
- Regional and global spatial aggregation
- Temporal processing (monthly → annual means)
- Publication-quality visualization

## Module: analysis.py

Core data processing module (~830 lines) for extracting and computing carbon cycle variables from UM output files.

### Key Functions

#### STASH Code Management
- `stash(var_name)` - Maps short names to PP-format STASH codes
  - Example: `'gpp'` → `'m01s03i261'`
- `stash_nc(var_name)` - Maps short names to NetCDF numeric STASH codes
  - Example: `'gpp'` → `3261`
- Covers 30+ variables: carbon cycle, ocean biogeochemistry, atmosphere, radiation

#### File Discovery
- `find_matching_files(expt_name, model, up, start_year=None, end_year=None, base_dir="~/dump2hold")`
  - Locates UM output files for a given experiment
  - Supports both alpha month codes (`ja`, `fb`, `mr`, etc.) and numeric codes (`11`-`91`, `a1`-`c1`)
  - Returns sorted list of `(year, month, filepath)` tuples

- `decode_month(mon_code)` - Parses UM two-letter month codes
  - Alpha: `ja`(Jan), `fb`(Feb), `mr`(Mar), `ar`(Apr), `my`(May), `jn`(Jun), `jl`(Jul), `ag`(Aug), `sp`(Sep), `ot`(Oct), `nv`(Nov), `dc`(Dec)
  - Numeric: `11`-`91` (Jan-Sep), `a1`(Oct), `b1`(Nov), `c1`(Dec)

#### Unit Conversions (`var_dict`)
Automatic unit conversions applied to standard variables:

| Variable | Conversion | From → To |
|----------|------------|-----------|
| Ocean flux (`m01s00i250`) | `(12/44)*3600*24*360*1e-12` | kgCO2/m²/s → PgC/yr |
| GPP, NPP, Rh | `3600*24*360*1e-12` | kgC/m²/s → PgC/yr |
| CV, CS | `1e-12` | kgC/m² → PgC |
| Air flux (`m02s30i249`) | `12/1000*1e-12` | molC/m²/yr → PgC/yr |
| Precipitation | `86400` | kg/m²/s → mm/day |
| Total CO2 | `28.97/44.01*1e6` | mmr → ppmv |

#### Spatial Processing

- `global_total_pgC(cube, var)` - Area-weighted sum (for extensive variables like fluxes)
- `global_mean_pgC(cube, var)` - Area-weighted mean (for intensive variables)
- `compute_regional_annual_mean(cube, var, region)` - Regional analysis using RECCAP mask
  - Handles pfts dimension (4D cubes)
  - Applies region-specific masking
  - Returns dict with `years`, `data`, `units`, `name`, `region`

- `load_reccap_mask()` - Loads RECCAP2 regional mask
  - 11 RECCAP2 regions: North_America, South_America, Europe, Africa (combines 4+5), North_Asia, Central_Asia, East_Asia, South_Asia, South_East_Asia, Oceania
  - Plus special regions: `Africa` (combined), `global`

- `region_mask(region)` - Creates binary mask for a specific region

#### Temporal Processing

- `compute_annual_mean(cube, var)` - Converts monthly data to annual means
  - Handles 360-day calendars
  - Adds missing time bounds automatically
  - Groups by year and averages
  - Returns dict with `years`, `data`, `name`, `units`

- `compute_monthly_mean(cube, var)` - Converts to monthly means (fractional years)
  - For TRIFFID variables with special time coordinates
  - Returns fractional year format (e.g., 1850.083 for Feb 1850)

- `merge_monthly_results(results, require_full_year=False)` - Merges multiple monthly outputs
  - Combines results from multiple files
  - Averages duplicates at same (year, month)
  - Optionally filters to complete years only

#### Variable Extraction

- `try_extract(cubes, code, stash_lookup_func=None, debug=False)` - Robust cube extraction
  - Works with both PP STASH objects and NetCDF numeric stash_codes
  - Accepts MSI strings, short names, or numeric codes
  - Handles multiple STASH formats automatically

#### High-Level Analysis

- `extract_annual_means(expts_list, var_list=None, var_mapping=None, regions=None)` - **Main workhorse function**

  **Default Variables Processed:**
  - `soilResp` (S resp) - Soil respiration
  - `soilCarbon` (S carb) - Soil carbon storage
  - `VegCarb` (V carb) - Vegetation carbon storage
  - `fracPFTs` (Others) - Plant functional type fractions (1-9)
  - `GPP` - Gross primary production
  - `NPP` - Net primary production
  - `fgco2` - Ocean CO2 flux (global only)
  - `temp` - Surface air temperature
  - `precip` - Precipitation

  **Computed Derived Variables:**
  - `NEP = NPP - soilResp` (Net ecosystem production)
  - `Land Carbon = soilCarbon + VegCarb + NEP` (Total land carbon)
  - `Trees Total = PFT1 + PFT2` (Total tree cover fraction)

  **Returns:**
  ```python
  dict[expt][region][var] -> {
      "years": np.array,
      "data": np.array,
      "units": str,
      "name": str,
      "region": str
  }
  ```

  **Example:**
  ```python
  ds = extract_annual_means(
      expts_list=['xqhuc'],
      regions=['global', 'Africa']
  )
  # Access: ds['xqhuc']['global']['GPP']['data']
  ```

---

### High-Level Analysis (Raw Monthly Files)

- `extract_annual_mean_raw(expt, base_dir='~/dump2hold', start_year=None, end_year=None)` - **Extract from raw monthly files**

  **Purpose:** Process raw monthly UM output files directly (not pre-processed annual mean NetCDF files).

  **Use case:** When you have raw monthly files in `~/dump2hold/expt/datam/` but no pre-processed annual means.

  **Variables Extracted:**
  - `GPP` - Gross Primary Production
  - `NPP` - Net Primary Production
  - `soilResp` - Soil respiration
  - `VegCarb` - Vegetation carbon
  - `soilCarbon` - Soil carbon
  - `NEP` - Net Ecosystem Production (derived: NPP - soilResp)

  **Workflow:**
  1. Finds raw monthly files using `find_matching_files()`
  2. Extracts variables using `try_extract()` with STASH lookup
  3. Computes monthly means with `compute_monthly_mean()`
  4. Merges into annual means with `merge_monthly_results()`
  5. Returns dict with years, data, units for each variable

  **Returns:**
  ```python
  {
      'GPP': {'years': array, 'data': array, 'units': 'PgC/year', 'name': str},
      'NPP': {'years': array, 'data': array, 'units': 'PgC/year', 'name': str},
      'soilResp': {'years': array, 'data': array, 'units': 'PgC/year', 'name': str},
      'VegCarb': {'years': array, 'data': array, 'units': 'PgC', 'name': str},
      'soilCarbon': {'years': array, 'data': array, 'units': 'PgC', 'name': str},
      'NEP': {'years': array, 'data': array, 'units': 'PgC/year', 'name': str},
  }
  ```

  **Example:**
  ```python
  # Extract from raw monthly files
  data = extract_annual_mean_raw('xqhuj')

  # Plot GPP time series
  import matplotlib.pyplot as plt
  plt.plot(data['GPP']['years'], data['GPP']['data'])
  plt.xlabel('Year')
  plt.ylabel('GPP (PgC/year)')
  plt.show()

  # Access NEP
  nep_years = data['NEP']['years']
  nep_values = data['NEP']['data']
  ```

  **Comparison with `extract_annual_means()`:**

  | Feature | `extract_annual_mean_raw()` | `extract_annual_means()` |
  |---------|----------------------------|-------------------------|
  | Input files | Raw monthly UM output | Pre-processed annual mean NetCDF |
  | Location | `~/dump2hold/expt/datam/` | `~/annual_mean/expt/` |
  | Regional analysis | No (global only) | Yes (RECCAP2 regions) |
  | Processing speed | Slower (reads many files) | Faster (reads 3 files) |
  | Use case | Direct from UM runs | After CDO post-processing |

---

## Module: plot.py

Visualization module (~431 lines) for publication-quality carbon cycle plots.

### Configuration

**Legend Labels & Colors:**
```python
legend_labels = {
    "xqhsh": "PI LU COU spinup",
    "xqhuc": "PI HadCM3 spinup",
}

color_map = {
    "xqhsh": "k",
    "xqhuc": "r",
}
```

### Plotting Functions

#### 1. `plot_timeseries_grouped(data, expts_list, region, outdir, ...)`
Multi-variable time series in a grid layout.

**Parameters:**
- `data` - Dict from `extract_annual_means()`
- `expts_list` - List of experiment names to plot
- `region` - Region name (e.g., 'global', 'Africa')
- `outdir` - Output directory for plots
- `legend_labels` - Optional dict for custom labels
- `color_map` - Optional dict for custom colors
- `exclude` - Tuple of variable prefixes to exclude (default: `("fracPFTs",)`)
- `ncols` - Number of columns in grid (default: 3)
- `show` - Display plot interactively (default: False)

**Features:**
- Auto-groups variables by prefix
- Multiple experiments overlaid per panel
- Drops first year of data (spinup)
- Grid layout with shared x-axis
- Auto-generated titles with units

**Output:** `allvars_{region}_{expts}_timeseries.png` (300 DPI)

**Example:**
```python
plot_timeseries_grouped(
    ds,
    expts_list=['xqhuc', 'xqhsh'],
    region='global',
    outdir='./plots/',
    exclude=('fracPFTs', 'temp')
)
```

---

#### 2. `plot_pft_timeseries(data, expts_list, region, outdir, ...)`
PFT fraction time series for one region.

**Parameters:**
- `pfts` - Tuple of PFT numbers to plot (default: `(1,2,3,4,5)`)
- Other parameters same as `plot_timeseries_grouped`

**Features:**
- 2×3 subplot layout for PFT 1-5
- Multiple experiments overlaid per PFT
- Drops first year of data
- Y-axis in fractions (0-1)

**Output:** `fracPFTs_{region}_timeseries.png` (300 DPI)

**Example:**
```python
plot_pft_timeseries(
    ds,
    expts_list=['xqhuc'],
    region='global',
    outdir='./plots/',
    pfts=(1,2,3,4,5)
)
```

---

#### 3. `plot_regional_pie(data, varname, expt, year, outdir, ...)`
Single pie chart showing regional distribution.

**Parameters:**
- `varname` - Variable name (e.g., 'soilResp', 'GPP')
- `expt` - Experiment name
- `year` - Year to plot
- Other parameters as above

**Features:**
- Shows value and percentage for each region
- Donut chart style (wedge width = 0.45)
- Auto-calculates totals

**Output:** `{var}_regional_pie_{expt}_{year}.png` (300 DPI)

**Example:**
```python
plot_regional_pie(
    ds,
    varname='GPP',
    expt='xqhuc',
    year=2000,
    outdir='./plots/'
)
```

---

#### 4. `plot_regional_pies(data, varname, expts_list, year, outdir, ...)`
Side-by-side pie charts for multiple experiments.

**Parameters:**
- Same as `plot_regional_pie` but accepts `expts_list` instead of single `expt`

**Features:**
- Compares experiments for same variable and year
- Side-by-side layout (1 row × N experiments)
- Consistent region ordering across experiments

**Output:** `{var}_regional_pies_{year}.png` (300 DPI)

**Example:**
```python
plot_regional_pies(
    ds,
    varname='NPP',
    expts_list=['xqhuc', 'xqhsh'],
    year=2000,
    outdir='./plots/'
)
```

---

#### 5. `plot_pft_grouped_bars(data, expts_list, year, outdir, ...)`
Grouped bar charts for PFT fractions by region.

**Parameters:**
- `year` - Year to plot
- `pfts` - Tuple of PFT numbers (default: `(1,2,3,4,5)`)
- Other parameters as above

**Features:**
- 2×3 subplot layout for PFT 1-5
- X-axis: regions (consistent ordering across all subplots)
- Grouped bars: one bar per experiment
- Y-axis: fraction (0-1 scale)
- Color-coded by experiment

**Output:** `fracPFTs_1to5_grouped_bars_{year}.png` (300 DPI)

**Example:**
```python
plot_pft_grouped_bars(
    ds,
    expts_list=['xqhuc', 'xqhsh'],
    year=2000,
    outdir='./plots/',
    pfts=(1,2,3,4,5)
)
```

---

## Typical Workflow

### 1. Extract Annual Means from UM Output
```python
import os
import sys
sys.path.append(os.path.expanduser('~/scripts/utils_cmip7'))
from analysis import extract_annual_means

# Extract data for single experiment (all regions)
ds = extract_annual_means(expts_list=['xqhuc'])

# Extract for multiple experiments with specific regions
ds = extract_annual_means(
    expts_list=['xqhuc', 'xqhsh'],
    regions=['global', 'Africa', 'South_America']
)
```

### 2. Generate Visualizations
```python
from plot import (
    plot_timeseries_grouped,
    plot_pft_timeseries,
    plot_regional_pies,
    plot_pft_grouped_bars
)

outdir = './plots/'

# Time series for all variables
plot_timeseries_grouped(
    ds,
    expts_list=['xqhuc'],
    region='global',
    outdir=outdir
)

# PFT fractions over time
plot_pft_timeseries(
    ds,
    expts_list=['xqhuc'],
    region='global',
    outdir=outdir
)

# Regional distribution comparison
plot_regional_pies(
    ds,
    varname='GPP',
    expts_list=['xqhuc', 'xqhsh'],
    year=2000,
    outdir=outdir
)

# PFT regional bar chart
plot_pft_grouped_bars(
    ds,
    expts_list=['xqhuc', 'xqhsh'],
    year=2000,
    outdir=outdir
)
```

### 3. Access Data Directly
```python
# Access global GPP time series for xqhuc
gpp_years = ds['xqhuc']['global']['GPP']['years']
gpp_values = ds['xqhuc']['global']['GPP']['data']
gpp_units = ds['xqhuc']['global']['GPP']['units']  # 'PgC/year'

# Access regional soil carbon for Africa
africa_soilc = ds['xqhuc']['Africa']['soilCarbon']['data']

# Access specific PFT fraction
pft1_global = ds['xqhuc']['global']['fracPFTs']['PFT 1']['data']
```

---

## Input Data Requirements

### Directory Structure
The `extract_annual_means()` function expects annual mean NetCDF files in:
```
~/annual_mean/{expt}/
├── {expt}_pa_annual_mean.nc  (atmosphere - temp, precip)
├── {expt}_pt_annual_mean.nc  (TRIFFID - GPP, NPP, soil resp, carbon stocks, PFTs)
└── {expt}_pf_annual_mean.nc  (ocean - fgco2)
```

### Pre-processing
Use the `annual_mean_cdo.sh` script (from parent project) to generate these files from monthly UM output:
```bash
./annual_mean_cdo.sh "xqhuc" ~/annual_mean pt pd pf
```

---

## STASH Codes Reference

Common variables used in carbon cycle analysis:

| Variable | STASH Code | Description |
|----------|------------|-------------|
| tas | m01s03i236 | Surface air temperature |
| pr | m01s05i216 | Precipitation |
| gpp | m01s03i261 | Gross Primary Production |
| npp | m01s03i262 | Net Primary Production |
| rh | m01s03i293 | Soil respiration |
| cv | m01s19i002 | Vegetation carbon |
| cs | m01s19i016 | Soil carbon |
| frac | m01s19i013 | PFT fractions |
| fgco2 | m02s30i249 | Ocean CO2 flux |
| co2 | m01s00i252 | Atmospheric CO2 concentration |

---

## Notes

- All time series plots drop the first year of data (spinup artifact)
- PFT numbering: 1=Broadleaf tree, 2=Needleleaf tree, 3=C3 grass, 4=C4 grass, 5=Shrub
- Ocean flux (fgco2) is only computed for global region
- All plots use 300 DPI for publication quality
- Default calendar: 360-day (UM standard)
- Regional analysis uses RECCAP2 mask at HadCM3 grid resolution

---

## Troubleshooting

### Problem: Plots only show Trees Total (or other subset of variables)

**Symptom:** When calling `plot_timeseries_grouped()`, only one or two variables appear instead of the expected 9+ variables.

**Root Cause:** Variables are missing from the data structure because `extract_annual_means()` couldn't find them in the NetCDF files.

**Diagnosis:** The improved error handling (added 2026-01) now provides detailed diagnostics during extraction:

```python
from analysis import extract_annual_means

ds = extract_annual_means(expts_list=['xqhuc'])
```

**Output example:**
```
============================================================
Extracting data for experiment: xqhuc
============================================================
Looking in: /home/user/annual_mean/xqhuc/
NetCDF files found: 1
  - xqhuc_pt_annual_mean.nc

Total cubes loaded: 12

Extracting variables...
  ❌ soilResp (rh, m01s03i293): NOT FOUND
  ❌ soilCarbon (cs, m01s19i016): NOT FOUND
  ❌ VegCarb (cv, m01s19i002): NOT FOUND
  ✓ fracPFTs (frac): Found (via stash code 3317)
  ❌ GPP (gpp, m01s03i261): NOT FOUND
  ❌ NPP (npp, m01s03i262): NOT FOUND
  ❌ fgco2 (fgco2, m02s30i249): NOT FOUND
  ❌ temp (tas, m01s03i236): NOT FOUND
  ❌ precip (pr, m01s05i216): NOT FOUND

============================================================
Extraction Summary for xqhuc
============================================================
Variables successfully extracted: 1/9
  Found: fracPFTs
  ⚠ Missing: soilResp, soilCarbon, VegCarb, GPP, NPP, fgco2, temp, precip

  These variables will NOT appear in plots!
============================================================
```

**Solutions:**

1. **Check if annual mean files exist:**
   ```bash
   ls -lh ~/annual_mean/xqhuc/
   ```
   Expected files:
   - `xqhuc_pt_annual_mean.nc` (TRIFFID variables)
   - `xqhuc_pd_annual_mean.nc` (atmosphere variables)
   - `xqhuc_pf_annual_mean.nc` (ocean variables)

2. **Generate missing annual mean files:**
   ```bash
   cd /path/to/scripts
   ./annual_mean_cdo.sh "xqhuc" ~/annual_mean pt pd pf
   ```

3. **Check file contents for STASH codes:**
   ```bash
   ncdump -h ~/annual_mean/xqhuc/xqhuc_pt_annual_mean.nc | grep STASH
   ```
   Should show variables with stash_code attribute (e.g., 3261 for GPP, 3262 for NPP)

4. **Verify source data location:**
   The `annual_mean_cdo.sh` script looks for monthly output in:
   - `~/umdata/xqhuc/pt/*.nc` (TRIFFID)
   - `~/umdata/xqhuc/pd/*.nc` (atmosphere)
   - `~/umdata/xqhuc/pf/*.nc` (ocean)

### Problem: Variables extracted but still missing from regional plots

**Symptom:** Variables show ✓ during extraction but don't appear in regional (non-global) plots.

**Cause:** `fgco2` (ocean CO2 flux) is intentionally skipped for non-global regions (line 843 in analysis.py).

**Solution:** Use `region='global'` when plotting ocean variables.

### Problem: NetCDF STASH codes show 's' suffix in ncdump

**What you see:** When checking NetCDF files:
```bash
ncdump -h file.nc | grep stash_code
```
Shows codes like:
```
stash_code = 3261s ;    ← Note the 's' suffix
stash_code = 19002s ;
```

**Clarification:** The 's' is a **NetCDF CDL type indicator** (short integer), NOT part of the actual value. When iris loads the file, `stash_code` attributes are plain Python integers (e.g., `3261`, not `'3261s'`). This is normal and requires no special handling.

### Problem: Wrong STASH codes in NetCDF files

**Symptom:** Variables exist in files but extraction reports NOT FOUND, and STASH codes don't match expected format.

**Diagnosis:** Check actual STASH codes in files:
```bash
ncdump -h file.nc | grep stash_code
```

**Solution:** If STASH codes differ significantly from expected values (not just 's' suffix), update the `stash()` or `stash_nc()` functions in analysis.py or use custom `var_list` and `var_mapping` parameters:

```python
ds = extract_annual_means(
    expts_list=['xqhuc'],
    var_list=['custom_var1', 'custom_var2'],
    var_mapping=['mapping1', 'mapping2']
)
```

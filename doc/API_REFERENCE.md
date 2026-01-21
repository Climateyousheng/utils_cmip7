# utils_cmip7 — API Reference

This document provides the **developer-facing API reference** for `utils_cmip7`.

It is **descriptive**, not normative.
Architectural constraints and stability guarantees are defined in `CLAUDE.md`.

---

## Package Layout (v0.2.0)

```
utils_cmip7/
├── io/                 # NetCDF loading, STASH handling, file discovery
│   ├── stash.py       # STASH code mappings
│   ├── file_discovery.py  # UM file pattern matching
│   └── extract.py     # Cube extraction with STASH handling
├── processing/        # Temporal/spatial aggregation, unit conversions
│   ├── spatial.py     # Global aggregation (SUM/MEAN)
│   ├── temporal.py    # Monthly → annual aggregation
│   └── regional.py    # RECCAP2 regional masking
├── diagnostics/       # Carbon-cycle diagnostics
│   ├── extraction.py  # Pre-processed NetCDF extraction
│   └── raw.py         # Raw monthly file extraction
├── plotting/          # Visualisation utilities (no I/O)
│   └── [TODO]         # To be split from plot.py
├── soil_params/       # Soil parameter analysis
│   └── [TODO]         # To be migrated from root
├── config.py          # Configuration and constants
└── cli.py             # [TODO] Command-line entry points
```

**Current Status (v0.2.0):**
- ✅ `io/` - Complete
- ✅ `processing/` - Complete
- ✅ `diagnostics/` - Complete
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

**Current Stability (v0.2.0):**
- io/ - **Provisional**
- processing/ - **Provisional**
- diagnostics/ - **Provisional**
- plotting/ - **Unstable** (not yet migrated)
- cli/ - **Experimental** (not yet implemented)

---

## Version History

- **v0.2.0** (Current): Core extraction functionality complete, package structure established
- **v0.1.0**: Initial scripts and functions

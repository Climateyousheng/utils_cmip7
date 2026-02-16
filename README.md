# utils_cmip7

Python toolkit for carbon cycle analysis from Unified Model (UM) climate model outputs.

## Installation

### Development Install (Recommended)

```bash
# Clone the repository
cd ~/path/to/utils_cmip7

# Install in editable mode
pip install -e .
```

This allows you to use `import utils_cmip7` from anywhere while still editing the source code.

### Environment Configuration

Set custom RECCAP2 mask path (optional):
```bash
export UTILS_CMIP7_RECCAP_MASK=/path/to/custom/mask.nc
```

Default mask location: `~/scripts/hadcm3b-ensemble-validator/observations/RECCAP_AfricaSplit_MASK11_Mask_regridded.hadcm3bl_grid.nc`

### Requirements

- Python ≥ 3.9
- numpy ≥ 1.22
- pandas ≥ 1.4
- matplotlib ≥ 3.5
- iris ≥ 3.2
- cartopy ≥ 0.21
- xarray ≥ 0.21
- cf-units ≥ 3.0
- netCDF4 ≥ 1.5

Dependencies are automatically installed with `pip install -e .`

## Package Structure (v0.4.0)

```
utils_cmip7/
├── src/utils_cmip7/          # Main package (src-layout)
│   ├── io/                   # NetCDF loading and file discovery
│   │   ├── stash.py          # STASH code mappings
│   │   ├── file_discovery.py # UM file pattern matching
│   │   ├── extract.py        # Cube extraction with STASH handling
│   │   └── obs_loader.py     # Observational data loader (CMIP6, RECCAP2)
│   ├── processing/           # Aggregation and unit conversions
│   │   ├── spatial.py        # Global aggregation (SUM/MEAN)
│   │   ├── temporal.py       # Monthly → annual aggregation
│   │   ├── regional.py       # RECCAP2 regional masking
│   │   ├── metrics.py        # Metric definitions and validation
│   │   └── map_fields.py     # Extract/combine 2D fields for map plotting
│   ├── diagnostics/          # High-level extraction workflows
│   │   ├── extraction.py     # Pre-processed NetCDF extraction
│   │   ├── raw.py            # Raw monthly file extraction
│   │   └── metrics.py        # Metrics computation from annual means
│   ├── validation/           # Model validation against observations (code)
│   │   ├── compare.py        # Bias and RMSE computation
│   │   └── visualize.py      # Three-way comparison plots
│   ├── data/obs/             # Observational data (packaged)
│   │   ├── stores_vs_fluxes_cmip6.csv
│   │   ├── stores_vs_fluxes_cmip6_err.csv
│   │   ├── stores_vs_fluxes_reccap.csv
│   │   └── stores_vs_fluxes_reccap_err.csv
│   ├── plotting/             # Visualization (maps, time series, PPE)
│   ├── soil_params/          # Soil parameter analysis (placeholder)
│   ├── config.py             # Configuration and constants
│   └── __init__.py           # Package API
├── validation_outputs/       # Validation results (generated, not in repo)
│   └── single_val_*/         # Per-experiment validation results
├── tests/                    # Test suite
│   ├── test_imports.py       # Import resolution tests
│   ├── test_config.py        # Configuration validation tests
│   ├── run_smoke_tests.py    # Test runner
│   └── README.md             # Test documentation
├── scripts/                  # Executable scripts
│   ├── extract_raw.sh        # Shell wrapper (generic)
│   ├── extract_raw.py        # Extract from raw monthly files
│   ├── extract_preprocessed.py # Extract from annual NetCDF files
│   ├── validate_experiment.py # Three-way validation (UM vs CMIP6 vs RECCAP2)
│   └── README.md             # Script documentation
├── examples/                 # Example scripts and notebooks
│   ├── validation_threeway_example.py  # Three-way validation example
│   ├── xqhuj_xqhuk_carbon_store.ipynb  # Carbon storage analysis
│   └── xqhul_co2_252.ipynb   # CO2 field analysis
├── dev/                      # Development/diagnostic tools
│   ├── debug_plot.py
│   └── diagnose_extraction.py
├── docs/                     # Documentation
│   ├── API_REFERENCE.md      # Comprehensive API documentation
│   ├── MIGRATION.md          # Migration guide from v0.1.x
│   ├── STASH.md              # STASH code reference
│   ├── TROUBLESHOOTING.md    # Common issues and solutions
│   ├── NAMING_ANALYSIS.md    # Variable naming analysis
│   └── REFACTORING_SUMMARY.md # Refactoring notes
├── analysis.py               # Backward-compatible wrapper (deprecated)
├── plot.py                   # Backward-compatible wrapper (deprecated)
└── pyproject.toml            # Package metadata and dependencies
```

**Status (v0.4.0):**
- ✅ `io/` - **Stable** - 4 modules (stash, file_discovery, extract, obs_loader)
- ✅ `processing/` - **Stable** - 5 modules (spatial, temporal, regional, metrics, map_fields)
- ✅ `diagnostics/` - **Stable** - 3 modules (extraction, raw, metrics)
- ✅ `validation/` - **Provisional** - 3 modules (compare, visualize, outputs)
- ✅ `tests/` - ~354 tests, 32% coverage, CI/CD across Python 3.9-3.12
- ✅ `data/obs/` - Observational data packaged
- ✅ `scripts/` - High-level validation workflows
- ✅ `cli.py` - **Experimental** - 4 CLI commands implemented
- ✅ `plotting/` - **Unstable** - maps.py for spatial map/anomaly plotting
- ⚠️ `soil_params/` - **Experimental** - Exists in root, needs migration

**Backward Compatibility:**
Existing scripts using `from analysis import ...` will continue to work during the v0.2.x series. See [Migration Guide](docs/MIGRATION_GUIDE.md) for updating to the new import style.

## Features

- **STASH code mapping** - Convert between variable names and UM STASH codes
- **File discovery** - Locate and parse UM output files with month code support
- **Spatial aggregation** - Global and regional analysis using RECCAP2 masks
- **Temporal processing** - Convert monthly data to annual means
- **Unit conversions** - Automatic conversion to standard units (PgC/yr, mm/day, etc.)
- **Model validation** - Three-way comparison (UM vs CMIP6 vs RECCAP2)
- **Observational data loading** - Load CMIP6 and RECCAP2 metrics from CSV
- **Bias and RMSE computation** - Statistical comparison against observations
- **Visualization** - Publication-quality plots for carbon cycle variables and validation
- **⚡ High performance** - Optimized extraction with intelligent caching (5-8× speedup)

## Performance

**Recent optimizations (2025)** have dramatically improved extraction performance:

### Raw Data Extraction: 5× Speedup

**File-level caching** eliminates redundant file loading:
- **Before**: Each file loaded 5 times (once per variable) = 6,000 loads for 100-year simulation
- **After**: Each file loaded once, all variables extracted in single pass = 1,200 loads
- **Result**: ~30 minutes → ~6 minutes for 100-year extraction

### Preprocessed Data Extraction: 3× Speedup

**Module-level mask caching** eliminates redundant NetCDF reads:
- **Before**: RECCAP2 mask file loaded 75+ times per extraction
- **After**: Mask file loaded once and cached in memory
- **Result**: ~9 minutes → ~3 minutes for multi-region extraction

### Technical Details

The optimizations are **completely transparent** to users:
- No API changes required
- All existing scripts work unchanged
- Memory overhead: negligible (~1-2 MB for mask cache)
- Thread-safe caching using `functools.lru_cache`

**Implementation**:
- Loop restructuring in `extract_annual_mean_raw()` (files outer, variables inner)
- `@lru_cache(maxsize=1)` on `load_reccap_mask()` and `_get_land_mask()`

See [CHANGELOG.md](CHANGELOG.md) for full performance improvement details.

## API Stability (v0.4.0)

The v0.4.0 release is a **breaking release** that removes deprecated features from v0.3.x. See [CHANGELOG.md](CHANGELOG.md) for migration guide.

- **Stable** - No breaking changes in v0.4.x series:
  - Core extraction (`extract_annual_means`, `extract_annual_mean_raw`)
  - Processing functions (spatial, temporal aggregation)
  - Configuration API (canonical variables, config helpers)
  - STASH mapping (`stash`, `stash_nc`)
  - File discovery (`find_matching_files`, `decode_month`)

- **Provisional** - Minor additions only, no breaking changes:
  - Regional aggregation (`compute_regional_annual_mean`)
  - Raw extraction workflows
  - Validation comparison (`compute_bias`, `compute_rmse`)

- **Unstable** - Breaking changes possible:
  - Plotting module
  - Validation visualization

- **Experimental** - No stability guarantees:
  - CLI commands
  - Soil parameter analysis

See [docs/API.md](docs/API.md) for the complete API reference and stability matrix.

## Core Modules

### I/O Layer (`utils_cmip7.io`)
- **stash.py** - STASH code mappings for UM variables (32 variables supported)
- **file_discovery.py** - UM file pattern matching with month code decoding
- **extract.py** - Robust cube extraction with flexible STASH handling
- **obs_loader.py** - Load CMIP6 and RECCAP2 observational data from CSV files

### Processing Layer (`utils_cmip7.processing`)
- **spatial.py** - Global aggregation (SUM/MEAN) with area weighting
- **temporal.py** - Monthly → annual aggregation, fractional year support
- **regional.py** - RECCAP2 regional masking (11 regions + global)
- **metrics.py** - Metric definitions (GPP, NPP, CVeg, CSoil, Tau, NEP) and canonical schema validation

### Diagnostics Layer (`utils_cmip7.diagnostics`)
- **extraction.py** - Main entry point for pre-processed NetCDF files
- **raw.py** - Main entry point for raw monthly UM files
- **metrics.py** - Compute metrics from annual mean files for all RECCAP2 regions

### Validation Layer (`utils_cmip7.validation`)
- **compare.py** - Bias, RMSE, and uncertainty checks against observations
- **visualize.py** - Three-way comparison plots, regional heatmaps, timeseries

### Configuration (`utils_cmip7.config`)
- **VAR_CONVERSIONS** - Unit conversion factors (kgC/m²/s → PgC/yr, etc.)
- **RECCAP_MASK_PATH** - Regional mask file location (configurable)
- **RECCAP_REGIONS** - Region ID to name mappings

## Scripts

### scripts/extract_raw.sh
Shell wrapper for extracting from raw monthly files:
```bash
./scripts/extract_raw.sh EXPERIMENT [OUTPUT_DIR]
```
Example: `./scripts/extract_raw.sh xqhuk ./plots`

### scripts/extract_raw.py
Python script for raw monthly file extraction:
```bash
python scripts/extract_raw.py xqhuj --outdir ./plots
```

### scripts/extract_preprocessed.py
Python script for pre-processed annual mean files (extracts all RECCAP2 regions):
```bash
python scripts/extract_preprocessed.py EXPERIMENT [--base-dir BASE_DIR]
```
Example: `python scripts/extract_preprocessed.py xqhuc --base-dir ~/annual_mean`

**Outputs** (in `validation_outputs/single_val_{expt}/`):
- `{expt}_extraction.csv` - Time-mean values for all variables and regions
- `plots/` - Time series plots for all regions (global, North_America, Europe, Africa, etc.)
- Automatically skips regions with no data

### scripts/validate_experiment.py
Comprehensive validation of a UM experiment against CMIP6 and RECCAP2 observations:
```bash
# Basic usage
python scripts/validate_experiment.py xqhuc

# With custom base directory
python scripts/validate_experiment.py --expt xqhuc --base-dir ~/annual_mean
```

**Outputs** (in `validation_outputs/single_val_{expt}/`):
- `{expt}_metrics.csv` - UM results in observational format
- `{expt}_bias_vs_cmip6.csv` - Bias statistics vs CMIP6
- `{expt}_bias_vs_reccap2.csv` - Bias statistics vs RECCAP2
- `comparison_summary.txt` - Text summary comparing UM vs CMIP6 performance
- `plots/` - Three-way comparison plots, regional bias heatmaps, timeseries

See [scripts/README.md](scripts/README.md) for detailed documentation.

## Command-Line Interface

CLI entry points are now available (implemented in v0.2.2, **Experimental** in v0.4.0):

```bash
# Extract from raw monthly files
utils-cmip7-extract-raw xqhuj

# Extract from pre-processed annual means
utils-cmip7-extract-preprocessed xqhuc

# Validate single experiment
utils-cmip7-validate-experiment xqhuc

# Validate perturbed parameter ensemble (PPE)
utils-cmip7-validate-ppe
```

See [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) for detailed documentation.

**Note**: CLI commands are marked **Experimental** - interfaces may change in future versions. For stable interfaces, use the Python API directly.

## Quick Start

### Option 1: From Pre-processed Annual Mean Files

```python
from utils_cmip7 import extract_annual_means

# Extract annual means for xqhuc experiment
ds = extract_annual_means(expts_list=['xqhuc'])

# Access data
gpp_global = ds['xqhuc']['global']['GPP']
print(f"GPP years: {gpp_global['years']}")
print(f"GPP data: {gpp_global['data']}")
print(f"GPP units: {gpp_global['units']}")

# Extract specific regions
ds = extract_annual_means(['xqhuc'], regions=['global', 'Europe', 'Africa'])
europe_npp = ds['xqhuc']['Europe']['NPP']['data']
```

### Option 2: From Raw Monthly UM Files

**⚡ Performance Note**: Raw extraction is now **5× faster** thanks to file-level caching (each file loaded once, all variables extracted in single pass).

**Using Package Function:**
```python
from utils_cmip7 import extract_annual_mean_raw
import matplotlib.pyplot as plt

# Extract from raw monthly files (5× faster with optimized caching)
data = extract_annual_mean_raw('xqhuj', start_year=1850, end_year=1900)

# Plot GPP
plt.plot(data['GPP']['years'], data['GPP']['data'])
plt.xlabel('Year')
plt.ylabel(f"GPP ({data['GPP']['units']})")
plt.title('Global GPP')
plt.show()
```

**Using Shell Script (for batch processing):**
```bash
./scripts/extract_raw.sh xqhuj
```

**Using Python Script:**
```bash
python scripts/extract_raw.py xqhuj --outdir ./plots
```

**With Validation (NEW in v0.3.1):**
```bash
# Script
python scripts/extract_raw.py xqhuj --validate

# CLI
utils-cmip7-extract-raw xqhuj --validate

# With custom validation output directory
utils-cmip7-extract-raw xqhuj --validate --validation-outdir ./my_validation
```

This validates the extracted annual means against CMIP6 and RECCAP2 observations (global only). Outputs include bias statistics CSVs and three-way comparison plots.

### Working with Regional Data

```python
from utils_cmip7.processing import compute_regional_annual_mean
import iris

# Load a cube
gpp_cube = iris.load_cube('gpp.nc')

# Compute regional annual mean
europe_gpp = compute_regional_annual_mean(gpp_cube, 'GPP', 'Europe')
print(f"Europe GPP: {europe_gpp['data']} {europe_gpp['units']}")

# Available regions: North_America, South_America, Europe, Africa,
# North_Asia, Central_Asia, East_Asia, South_Asia, South_East_Asia, Oceania
```

### Plotting Spatial Maps (Notebook Workflow)

Plot 2D fields on geographic map projections.  The workflow separates
**extraction** (iris cube to arrays) from **plotting** (arrays to map),
so plotting functions never touch cubes or NetCDF files directly.

#### Basic usage

```python
import iris
from utils_cmip7.processing import extract_map_field
from utils_cmip7.plotting import plot_spatial_map

# 1. Load a cube with lat/lon (and optionally time) dimensions
cube = iris.load_cube("path/to/annual_mean.nc", "gpp")

# 2. Extract a 2D field (returns a dict with data, lons, lats, title, units, ...)
field = extract_map_field(cube, time=1900)

# 3. Plot — global map with Robinson projection (default)
fig, ax = plot_spatial_map(
    field["data"], field["lons"], field["lats"],
    title=field["title"], units=field["units"],
)
```

#### Regional views

```python
# Named RECCAP2 region (auto-switches to PlateCarree)
field = extract_map_field(cube, time=1900)
fig, ax = plot_spatial_map(
    field["data"], field["lons"], field["lats"],
    region="Europe", cmap="RdYlGn",
    title=field["title"], units=field["units"],
)

# Custom bounding box
fig, ax = plot_spatial_map(
    field["data"], field["lons"], field["lats"],
    lon_bounds=(-90, -30), lat_bounds=(-60, 15),
    title="South America GPP",
)
```

#### Anomaly (difference) maps

```python
from utils_cmip7.processing import extract_anomaly_field
from utils_cmip7.plotting import plot_spatial_anomaly

anomaly = extract_anomaly_field(cube, time_a=2000, time_b=1900)
fig, ax = plot_spatial_anomaly(
    anomaly["data"], anomaly["lons"], anomaly["lats"],
    vmin=anomaly["vmin"], vmax=anomaly["vmax"],
    title=anomaly["title"], units=anomaly["units"],
)
```

#### Combining multiple variables

```python
from utils_cmip7.processing import extract_map_field, combine_fields
from utils_cmip7.plotting import plot_spatial_map

cube_gpp = iris.load_cube("gpp.nc", "gpp")
cube_npp = iris.load_cube("npp.nc", "npp")

field_gpp = extract_map_field(cube_gpp, time=1900)
field_npp = extract_map_field(cube_npp, time=1900)

# Sum (default), mean, subtract, multiply, divide
total = combine_fields([field_gpp, field_npp])
fig, ax = plot_spatial_map(
    total["data"], total["lons"], total["lats"],
    title=total["name"], units=total["units"],
)
```

#### Multi-panel figures

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

field_1900 = extract_map_field(cube, time=1900)
field_2000 = extract_map_field(cube, time=2000)

fig, axes = plt.subplots(
    1, 2, figsize=(16, 5),
    subplot_kw={"projection": ccrs.PlateCarree()},
)
plot_spatial_map(
    field_1900["data"], field_1900["lons"], field_1900["lats"],
    ax=axes[0], title="1900",
)
plot_spatial_map(
    field_2000["data"], field_2000["lons"], field_2000["lats"],
    ax=axes[1], title="2000",
)
plt.tight_layout()
```

Available RECCAP2 regions: `North_America`, `South_America`, `Europe`, `Africa`,
`North_Asia`, `Central_Asia`, `East_Asia`, `South_Asia`, `South_East_Asia`, `Oceania`.

### Model Validation (New in v0.2.1)

**Three-Way Comparison: UM vs CMIP6 vs RECCAP2**

```python
from utils_cmip7.diagnostics import compute_metrics_from_annual_means
from utils_cmip7.io import load_cmip6_metrics, load_reccap_metrics
from utils_cmip7.validation import plot_three_way_comparison

# Compute UM metrics from annual mean files
um_metrics = compute_metrics_from_annual_means(
    expt_name='xqhuc',
    metrics=['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau'],
    regions=['global', 'North_America', 'Europe']
)

# Load observational data
cmip6_metrics = load_cmip6_metrics(
    metrics=['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau'],
    regions=['global', 'North_America', 'Europe'],
    include_errors=True
)

reccap_metrics = load_reccap_metrics(
    metrics=['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau'],
    regions=['global', 'North_America', 'Europe'],
    include_errors=True
)

# Create three-way comparison plot
plot_three_way_comparison(
    um_metrics, cmip6_metrics, reccap_metrics,
    metric='GPP',
    outdir='./validation'
)
```

**High-Level Validation Workflow:**

```bash
# Validate experiment xqhuc against all observations
python scripts/validate_experiment.py xqhuc

# Outputs saved to validation_outputs/single_val_xqhuc/
# - CSV files with metrics and bias statistics
# - Plots comparing UM vs CMIP6 vs RECCAP2
# - Text summary with performance comparison
```

See [examples/validation_threeway_example.py](examples/validation_threeway_example.py) for a complete example.

## Input Data Requirements

### For Pre-processed Annual Means

Annual mean NetCDF files should be located in `~/annual_mean/{expt}/`:
- `{expt}_pa_annual_mean.nc` - Atmosphere (temp, precip)
- `{expt}_pt_annual_mean.nc` - TRIFFID (GPP, NPP, soil resp, carbon stocks, PFTs)
- `{expt}_pf_annual_mean.nc` - Ocean (fgco2)

Generate these files using:
```bash
./annual_mean_cdo.sh "xqhuj" ~/annual_mean pt pd pf
```

### For Raw Monthly Files

Raw monthly UM output files in `~/dump2hold/{expt}/datam/`:
- Files matching pattern: `{expt}a#pi00000{YYYY}{MM}+`
- Month codes: `ja`-`dc` (alpha) or `11`-`c1` (numeric)
- Example: `xqhuja#pi000018511ja+` (January 1851)

## Variables Processed

- **Carbon fluxes**: GPP, NPP, soil respiration, ocean CO2 flux
- **Carbon stocks**: Vegetation carbon, soil carbon
- **PFT fractions**: Plant functional types 1-9
- **Climate**: Temperature, precipitation
- **Derived**: NEP, Land Carbon, Tree Total

## Documentation

- **[API Reference](docs/API.md)** - Public API reference with stability guarantees (v0.4.0)
- **[Performance Guide](docs/PERFORMANCE.md)** - Performance optimization details and benchmarking
- **[CHANGELOG](CHANGELOG.md)** - Version history and release notes
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Guide for migrating from v0.1.x to v0.2.x
- **[CLI Reference](docs/CLI_REFERENCE.md)** - Command-line interface documentation
- **[STASH Codes](docs/STASH.md)** - UM STASH code reference
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[CLAUDE.md](CLAUDE.md)** - Architectural constraints and design rules (for developers/AI)

## License

MIT License

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

- Python ≥ 3.8
- numpy ≥ 1.20
- pandas ≥ 1.3
- matplotlib ≥ 3.4
- iris ≥ 3.0
- cartopy ≥ 0.20
- xarray ≥ 0.19
- cf-units ≥ 2.1
- netCDF4 ≥ 1.5

Dependencies are automatically installed with `pip install -e .`

## Package Structure (v0.2.1)

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
│   │   └── metrics.py        # Metric definitions and validation
│   ├── diagnostics/          # High-level extraction workflows
│   │   ├── extraction.py     # Pre-processed NetCDF extraction
│   │   ├── raw.py            # Raw monthly file extraction
│   │   └── metrics.py        # Metrics computation from annual means
│   ├── validation/           # Model validation against observations
│   │   ├── compare.py        # Bias and RMSE computation
│   │   └── visualize.py      # Three-way comparison plots
│   ├── plotting/             # Visualization (placeholder)
│   ├── soil_params/          # Soil parameter analysis (placeholder)
│   ├── config.py             # Configuration and constants
│   └── __init__.py           # Package API
├── obs/                      # Observational data (CMIP6, RECCAP2)
│   ├── stores_vs_fluxes_cmip6.csv
│   ├── stores_vs_fluxes_cmip6_err.csv
│   ├── stores_vs_fluxes_reccap.csv
│   └── stores_vs_fluxes_reccap_err.csv
├── validation/               # Validation results and analysis
│   └── random_sampling_combined_overview_table.csv
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
├── doc/                      # Documentation
│   ├── API_REFERENCE.md      # Comprehensive API documentation
│   ├── MIGRATION.md          # Migration guide from v0.1.x
│   ├── STASH.md              # STASH code reference
│   └── TROUBLESHOOTING.md    # Common issues and solutions
├── analysis.py               # Backward-compatible wrapper (deprecated)
├── plot.py                   # Backward-compatible wrapper (deprecated)
└── pyproject.toml            # Package metadata and dependencies
```

**Status (v0.2.1):**
- ✅ `io/` - Complete (4 modules including obs_loader)
- ✅ `processing/` - Complete (4 modules including metrics)
- ✅ `diagnostics/` - Complete (3 modules including metrics)
- ✅ `validation/` - Complete (2 modules: compare, visualize)
- ✅ `tests/` - Basic smoke tests implemented
- ✅ `obs/` - Observational data for validation (CMIP6, RECCAP2)
- ✅ `scripts/` - High-level validation workflow (validate_experiment.py)
- ⚠️ `plotting/` - Exists in root `plot.py`, needs migration
- ⚠️ `soil_params/` - Exists in root, needs migration
- ❌ `cli.py` - Not yet implemented

**Backward Compatibility:**
Existing scripts using `from analysis import ...` will continue to work during the v0.2.x series. See [Migration Guide](doc/MIGRATION.md) for updating to the new import style.

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
Python script for pre-processed annual mean files:
```bash
python scripts/extract_preprocessed.py EXPERIMENT [--outdir OUTPUT_DIR]
```
Example: `python scripts/extract_preprocessed.py xqhuc --outdir ./plots`

### scripts/validate_experiment.py
Comprehensive validation of a UM experiment against CMIP6 and RECCAP2 observations:
```bash
# Basic usage
python scripts/validate_experiment.py xqhuc

# With custom base directory
python scripts/validate_experiment.py --expt xqhuc --base-dir ~/annual_mean
```

**Outputs** (in `validation/single_val_{expt}/`):
- `{expt}_metrics.csv` - UM results in observational format
- `{expt}_bias_vs_cmip6.csv` - Bias statistics vs CMIP6
- `{expt}_bias_vs_reccap2.csv` - Bias statistics vs RECCAP2
- `comparison_summary.txt` - Text summary comparing UM vs CMIP6 performance
- `plots/` - Three-way comparison plots, regional bias heatmaps, timeseries

See [scripts/README.md](scripts/README.md) for detailed documentation.

## Command-Line Interface (Planned)

CLI entry points will be available in v0.2.2:

```bash
# Extract from raw monthly files
utils-cmip7-extract-raw xqhuj

# Extract from pre-processed annual means
utils-cmip7-extract-preprocessed xqhuc
```

Currently use the `scripts/` versions instead.

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

**Using Package Function:**
```python
from utils_cmip7 import extract_annual_mean_raw
import matplotlib.pyplot as plt

# Extract from raw monthly files
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

# Outputs saved to validation/single_val_xqhuc/
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

- **[API Reference](doc/API_REFERENCE.md)** - Comprehensive API documentation for all modules
- **[Migration Guide](doc/MIGRATION.md)** - Guide for migrating from v0.1.x to v0.2.x
- **[STASH Codes](doc/STASH.md)** - UM STASH code reference
- **[CLAUDE.md](CLAUDE.md)** - Architectural constraints and design rules (for developers/AI)

## License

MIT License

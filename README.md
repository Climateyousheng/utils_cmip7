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

## Package Structure (v0.2.0)

```
utils_cmip7/
├── src/utils_cmip7/          # Main package (src-layout)
│   ├── io/                   # NetCDF loading and file discovery
│   │   ├── stash.py          # STASH code mappings
│   │   ├── file_discovery.py # UM file pattern matching
│   │   └── extract.py        # Cube extraction with STASH handling
│   ├── processing/           # Aggregation and unit conversions
│   │   ├── spatial.py        # Global aggregation (SUM/MEAN)
│   │   ├── temporal.py       # Monthly → annual aggregation
│   │   └── regional.py       # RECCAP2 regional masking
│   ├── diagnostics/          # High-level extraction workflows
│   │   ├── extraction.py     # Pre-processed NetCDF extraction
│   │   └── raw.py            # Raw monthly file extraction
│   ├── config.py             # Configuration and constants
│   └── __init__.py           # Package API
├── scripts/                  # Executable scripts
│   ├── extract_raw.sh        # Shell wrapper (generic)
│   ├── extract_raw.py        # Extract from raw monthly files
│   └── extract_preprocessed.py # Extract from annual NetCDF files
├── examples/                 # Example notebooks
│   └── xqhuj_xqhuk_carbon_store.ipynb
├── dev/                      # Development/diagnostic tools
│   ├── debug_plot.py
│   └── diagnose_extraction.py
└── doc/                      # Documentation
    ├── API_REFERENCE.md      # Comprehensive API documentation
    ├── MIGRATION.md          # Migration guide from v0.1.x
    └── STASH.md              # STASH code reference
```

**Status (v0.2.0):**
- ✅ `io/` - Complete
- ✅ `processing/` - Complete
- ✅ `diagnostics/` - Complete
- ⚠️ `plotting/` - Exists in root `plot.py`, needs migration
- ❌ `cli.py` - Not yet implemented

**Backward Compatibility:**
Existing scripts using `from analysis import ...` will continue to work during the v0.2.x series. See [Migration Guide](doc/MIGRATION.md) for updating to the new import style.

## Features

- **STASH code mapping** - Convert between variable names and UM STASH codes
- **File discovery** - Locate and parse UM output files with month code support
- **Spatial aggregation** - Global and regional analysis using RECCAP2 masks
- **Temporal processing** - Convert monthly data to annual means
- **Unit conversions** - Automatic conversion to standard units (PgC/yr, mm/day, etc.)
- **Visualization** - Publication-quality plots for carbon cycle variables

## Core Modules

### I/O Layer (`utils_cmip7.io`)
- **stash.py** - STASH code mappings for UM variables (32 variables supported)
- **file_discovery.py** - UM file pattern matching with month code decoding
- **extract.py** - Robust cube extraction with flexible STASH handling

### Processing Layer (`utils_cmip7.processing`)
- **spatial.py** - Global aggregation (SUM/MEAN) with area weighting
- **temporal.py** - Monthly → annual aggregation, fractional year support
- **regional.py** - RECCAP2 regional masking (11 regions + global)

### Diagnostics Layer (`utils_cmip7.diagnostics`)
- **extraction.py** - Main entry point for pre-processed NetCDF files
- **raw.py** - Main entry point for raw monthly UM files

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

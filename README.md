# utils_cmip7

Python toolkit for carbon cycle analysis from Unified Model (UM) climate model outputs.

## Features

- **STASH code mapping** - Convert between variable names and UM STASH codes
- **File discovery** - Locate and parse UM output files with month code support
- **Spatial aggregation** - Global and regional analysis using RECCAP2 masks
- **Temporal processing** - Convert monthly data to annual means
- **Unit conversions** - Automatic conversion to standard units (PgC/yr, mm/day, etc.)
- **Visualization** - Publication-quality plots for carbon cycle variables

## Modules

### analysis.py
Core data processing module for extracting and computing carbon cycle variables:
- Variable extraction with STASH code support
- Regional/global spatial aggregation
- Temporal aggregation (monthly → annual)
- Derived variable computation (NEP, Land Carbon, Tree Total)

### plot.py
Visualization module with functions for:
- Multi-variable time series plots
- PFT fraction time series
- Regional distribution pie charts
- PFT grouped bar charts

## Quick Start

```python
import os
import sys
sys.path.append(os.path.expanduser('~/scripts/utils_cmip7'))
from analysis import extract_annual_means
from plot import plot_timeseries_grouped

# Extract annual means for xqhuc experiment
ds = extract_annual_means(expts_list=['xqhuc'])

# Generate time series plots for global region
plot_timeseries_grouped(ds, expts_list=['xqhuc'],
                        region='global', outdir='./plots/')
```

## Input Data Requirements

Annual mean NetCDF files should be located in `~/annual_mean/{expt}/`:
- `{expt}_pa_annual_mean.nc` - Atmosphere (temp, precip)
- `{expt}_pt_annual_mean.nc` - TRIFFID (GPP, NPP, soil resp, carbon stocks, PFTs)
- `{expt}_pf_annual_mean.nc` - Ocean (fgco2)

## Variables Processed

- **Carbon fluxes**: GPP, NPP, soil respiration, ocean CO2 flux
- **Carbon stocks**: Vegetation carbon, soil carbon
- **PFT fractions**: Plant functional types 1-9
- **Climate**: Temperature, precipitation
- **Derived**: NEP, Land Carbon, Tree Total

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed API documentation.

## License

MIT License

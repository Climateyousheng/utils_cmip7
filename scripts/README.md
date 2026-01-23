# Scripts

High-level workflow scripts for common validation and analysis tasks.

## extract_preprocessed.py

Extract annual means from pre-processed NetCDF files for all RECCAP2 regions and generate time series plots.

### Usage

```bash
# Basic usage
python scripts/extract_preprocessed.py xqhuc

# With custom base directory
python scripts/extract_preprocessed.py xqhuc --base-dir ~/annual_mean
```

### Requirements

- Annual mean NetCDF files in `~/annual_mean/{expt}/` (or custom base directory)
  - `{expt}_pa_annual_mean.nc` (atmosphere)
  - `{expt}_pt_annual_mean.nc` (TRIFFID)
  - `{expt}_pf_annual_mean.nc` (ocean)

### Outputs

Creates `validation_outputs/single_val_{expt}/plots/` containing time series plots for all regions:

```
validation_outputs/single_val_{expt}/plots/
├── allvars_global_{expt}_timeseries.png
├── allvars_Europe_{expt}_timeseries.png
├── allvars_North_America_{expt}_timeseries.png
├── allvars_South_America_{expt}_timeseries.png
├── allvars_Africa_{expt}_timeseries.png
├── allvars_North_Asia_{expt}_timeseries.png
├── allvars_Central_Asia_{expt}_timeseries.png
├── allvars_East_Asia_{expt}_timeseries.png
├── allvars_South_Asia_{expt}_timeseries.png
├── allvars_South_East_Asia_{expt}_timeseries.png
└── allvars_Oceania_{expt}_timeseries.png
```

### What it does

1. **Extracts data** for all RECCAP2 regions (global + 10 regions)
2. **Generates time series plots** for each region showing:
   - Carbon fluxes (GPP, NPP, Rh, fgco2)
   - Carbon stocks (CVeg, CSoil)
   - Climate variables (tas, pr)
   - PFT fractions (if available)
3. **Automatically skips** regions with no data

### Notes

- Only variables successfully extracted are shown in plots
- If many variables are missing, check:
  1. Annual mean files exist in base directory
  2. Files generated using `annual_mean_cdo.sh`
  3. STASH codes are correct in files

---

## validate_experiment.py

Comprehensive validation of a single UM experiment against CMIP6 and RECCAP2 observations.

### Usage

```bash
# Basic usage
python scripts/validate_experiment.py xqhuc

# With custom base directory
python scripts/validate_experiment.py --expt xqhuc --base-dir ~/annual_mean
```

### Requirements

- Annual mean NetCDF files in `~/annual_mean/{expt}/`
- Observational data (automatically loaded from package data)

### Outputs

Creates `validation_outputs/single_val_{expt}/` containing:

```
validation_outputs/single_val_{expt}/
├── {expt}_metrics.csv              # UM results (obs format)
├── {expt}_bias_vs_cmip6.csv        # Bias statistics vs CMIP6
├── {expt}_bias_vs_reccap2.csv      # Bias statistics vs RECCAP2
├── comparison_summary.txt          # Text summary with performance comparison
└── plots/
    ├── GPP_three_way.png           # Three-way comparison plots
    ├── NPP_three_way.png
    ├── CVeg_three_way.png
    ├── CSoil_three_way.png
    ├── Tau_three_way.png
    ├── bias_heatmap_vs_cmip6.png   # Regional bias heatmaps
    ├── bias_heatmap_vs_reccap2.png
    └── *_timeseries_global.png     # Time series plots
```

### What it does

1. **Computes UM metrics** for all RECCAP2 regions (global + 11 regions)
2. **Loads observational data** (CMIP6 ensemble and RECCAP2)
3. **Computes bias statistics** (bias, bias %, RMSE, within uncertainty)
4. **Exports to CSV** in standardized format matching obs/ files
5. **Creates visualizations**:
   - Three-way comparisons (UM vs CMIP6 vs RECCAP2)
   - Regional bias heatmaps
   - Time series with observational uncertainty
6. **Generates text summary** comparing UM vs CMIP6 performance

### CSV Format

**{expt}_metrics.csv**: UM results in same format as observational CSV files
```csv
,global,North_America,South_America,Europe,Africa,...
GPP,134.81,18.40,31.65,6.11,30.02,...
NPP,68.29,10.36,14.45,3.42,14.45,...
...
```

**{expt}_bias_vs_*.csv**: Detailed bias statistics
```csv
metric,region,um_mean,obs_mean,bias,bias_percent,rmse,within_uncertainty
GPP,global,134.81,124.04,10.77,8.68,10.94,False
...
```

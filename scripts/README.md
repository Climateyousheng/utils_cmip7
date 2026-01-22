# Scripts

High-level workflow scripts for common validation and analysis tasks.

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
- Observational data in `obs/` directory (CMIP6 and RECCAP2)

### Outputs

Creates `validation/single_val_{expt}/` containing:

```
validation/single_val_{expt}/
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

**{expt}_metrics.csv**: UM results in same format as `obs/stores_vs_fluxes_cmip6.csv`
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

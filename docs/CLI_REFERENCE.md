# CLI Reference

Command-line tools for utils_cmip7 package (v0.2.2+).

---

## Installation

Ensure the package is installed:

```bash
cd ~/path/to/utils_cmip7
pip install -e .
```

After installation, four CLI commands will be available:
- `utils-cmip7-extract-preprocessed`
- `utils-cmip7-extract-raw`
- `utils-cmip7-validate-experiment`
- `utils-cmip7-validate-ppe`

---

## Data Extraction Commands

### `utils-cmip7-extract-preprocessed`

Extract annual means from pre-processed NetCDF files.

**Basic Usage:**
```bash
utils-cmip7-extract-preprocessed xqhuc
```

**Advanced Usage:**
```bash
# Specify base directory
utils-cmip7-extract-preprocessed xqhuc --base-dir ~/annual_mean

# Filter specific regions
utils-cmip7-extract-preprocessed xqhuc --regions global Europe Africa

# Filter specific variables
utils-cmip7-extract-preprocessed xqhuc --var-list GPP NPP CVeg CSoil

# Save to CSV
utils-cmip7-extract-preprocessed xqhuc --output results.csv

# Verbose output
utils-cmip7-extract-preprocessed xqhuc --verbose
```

**Output:**
- By default: Prints structured data to stdout
- With `--output`: Saves time-mean values to CSV

**Variables Extracted:**
GPP, NPP, CVeg, CSoil, Tau, precip, tas, and others as available

---

### `utils-cmip7-extract-raw`

Extract annual means from raw monthly UM output files.

**Basic Usage:**
```bash
utils-cmip7-extract-raw xqhuj
```

**Advanced Usage:**
```bash
# Specify base directory
utils-cmip7-extract-raw xqhuj --base-dir ~/dump2hold

# Year range selection
utils-cmip7-extract-raw xqhuj --start-year 2000 --end-year 2010

# Save to CSV
utils-cmip7-extract-raw xqhuj --output timeseries.csv

# Verbose output
utils-cmip7-extract-raw xqhuj --verbose
```

**Output:**
- By default: Prints time series summary to stdout
- With `--output`: Saves full time series to CSV

**Variables Extracted:**
GPP, NPP, soilResp, VegCarb, soilCarbon, NEP

---

## Validation Commands

### `utils-cmip7-validate-experiment`

Validate a single UM experiment against CMIP6 and RECCAP2 observations.

**Basic Usage:**
```bash
# With default soil parameters
utils-cmip7-validate-experiment xqhuc --use-default-soil-params

# With soil parameters from log file
utils-cmip7-validate-experiment xqhuc --soil-log-file rose.log

# With soil parameters from JSON/YAML file
utils-cmip7-validate-experiment xqhuc --soil-param-file params.json

# With manual soil parameters
utils-cmip7-validate-experiment xqhuc --soil-params "ALPHA=0.08,G_AREA=0.004"
```

**Advanced Usage:**
```bash
# Specify base directory
utils-cmip7-validate-experiment xqhuc --soil-log-file rose.log --base-dir ~/annual_mean
```

**Soil Parameters (REQUIRED):**

Must provide **one** of:
- `--soil-param-file FILE` - JSON/YAML parameter file
- `--soil-log-file FILE` - UM/Rose log with &LAND_CC block
- `--soil-params KEY=VAL,...` - Manual parameters
- `--use-default-soil-params` - Use default LAND_CC values

**Output Directory:**
```
validation_outputs/single_val_{expt}/
├── {expt}_metrics.csv              # UM results (carbon + vegetation)
├── {expt}_bias_vs_cmip6.csv        # Bias statistics vs CMIP6
├── {expt}_bias_vs_reccap2.csv      # Bias statistics vs RECCAP2
├── {expt}_bias_vs_igbp.csv         # Vegetation bias vs IGBP
├── soil_params.json                # Full soil parameters + provenance
├── validation_scores.csv           # Scores for overview table
├── comparison_summary.txt          # Text summary
└── plots/                          # All comparison plots
    ├── bias_heatmap_unified.png    # Carbon + veg unified heatmap
    ├── GPP_three_way.png           # UM vs CMIP6 vs RECCAP2
    ├── BL_vs_igbp.png              # Veg fraction comparisons
    └── ...
```

**Variables Validated:**
- **Carbon cycle**: GPP, NPP, CVeg, CSoil, Tau
- **Vegetation fractions**: BL, NL, C3, C4, shrub, bare_soil

**Observational Datasets:**
- CMIP6 ensemble (mean + uncertainty)
- RECCAP2 regional observations
- IGBP vegetation fractions

---

### `utils-cmip7-validate-ppe`

Generate PPE (Perturbed Physics Ensemble) validation report.

**Basic Usage:**
```bash
utils-cmip7-validate-ppe xqhuc
```

**Advanced Usage:**
```bash
# Control plot parameters
utils-cmip7-validate-ppe xqhuc --top-n 20 --top-k 40 --q 0.15

# Highlight additional experiments
utils-cmip7-validate-ppe xqhuc --highlight xqhua,xqhub

# Custom CSV path
utils-cmip7-validate-ppe xqhuc --csv my_ensemble_table.csv

# Parameter importance analysis
utils-cmip7-validate-ppe xqhuc --param-viz
utils-cmip7-validate-ppe xqhuc --param-viz --param-viz-vars GPP NPP CVeg CSoil

# Customize highlighting
utils-cmip7-validate-ppe xqhuc --highlight-style outline --no-highlight-label
```

**Parameters:**
- `--top-n` - Number of top experiments to highlight in score plots (default: 15)
- `--top-k` - Number of experiments to show in heatmap (default: 30)
- `--q` - Quantile for parameter shift analysis (default: 0.10)
- `--csv` - Path to ensemble CSV (default: `validation_outputs/random_sampling_combined_overview_table.csv`)
- `--output-dir` - Base output directory (default: `validation_outputs`)

**Highlighting Options:**
- `--highlight` - Additional experiments to highlight (can be repeated)
- `--include-highlight` - Force-include highlighted experiments (default: True)
- `--highlight-style` - Style: outline, marker, rowcol, both (default: both)
- `--highlight-label` - Add labels to highlighted experiments (default: True)

**Parameter Importance:**
- `--param-viz` - Run Spearman + RandomForest analysis
- `--param-viz-vars` - Variables to analyze (default: all)

**Output Directory:**
```
validation_outputs/ppe_{expt}/
├── ensemble_table.csv              # Input data copy
├── score_distribution.pdf          # Histogram + ECDF with top-N labeled
├── validation_heatmap.pdf          # Normalized metrics for top-K
├── parameter_shifts.pdf            # Parameter distributions (top vs bottom)
└── top_experiments.txt             # Text summary with statistics
```

If `--param-viz` enabled:
```
validation_outputs/param_viz_{expt}/
├── expanded_parameters.csv         # Parameter matrix
├── importance_spearman_{var}.csv   # Spearman correlations
├── importance_rfperm_{var}.csv     # RF permutation importance
├── bar_spearman_{var}.png          # Importance bar charts
├── bar_rfperm_{var}.png
├── pca_{var}.png                   # PCA embeddings
└── summary.json                    # Analysis metadata
```

**Input CSV Format:**

Required columns:
- `overall_score` (or specify with `--score-col`)

Optional columns:
- `ID` (or specify with `--id-col`)
- Parameter columns: ALPHA, G_AREA, LAI_MIN, NL0, R_GROW, TLOW, TUPP, V_CRIT
- Metric columns: rmse_GPP, rmse_NPP, GM_BL, etc.

Note: RMSE metrics are automatically inverted in heatmap (higher = better)

---

## Complete Workflow Example

```bash
# 1. Extract and validate individual experiments
utils-cmip7-validate-experiment xqhuc --use-default-soil-params
utils-cmip7-validate-experiment xqhua --soil-log-file logs/xqhua.log
utils-cmip7-validate-experiment xqhub --soil-log-file logs/xqhub.log

# 2. Generate PPE validation report
utils-cmip7-validate-ppe xqhuc --top-n 20 --top-k 50 --highlight xqhua,xqhub

# 3. Run parameter importance analysis
utils-cmip7-validate-ppe xqhuc --param-viz --param-viz-vars GPP NPP CVeg CSoil

# 4. Extract data for custom analysis
utils-cmip7-extract-preprocessed xqhuc --output xqhuc_data.csv
```

---

## Tips

### Getting Help

All commands support `--help`:
```bash
utils-cmip7-extract-preprocessed --help
utils-cmip7-validate-experiment --help
```

### Shell Completion (Optional)

For bash completion, add to `~/.bashrc`:
```bash
eval "$(_UTILS_CMIP7_EXTRACT_PREPROCESSED_COMPLETE=bash_source utils-cmip7-extract-preprocessed)"
eval "$(_UTILS_CMIP7_VALIDATE_EXPERIMENT_COMPLETE=bash_source utils-cmip7-validate-experiment)"
```

### Pipeline Integration

All commands write structured output suitable for pipelines:
```bash
# Extract to CSV and analyze
utils-cmip7-extract-preprocessed xqhuc --output data.csv
python my_analysis.py data.csv

# Validate and check exit code
if utils-cmip7-validate-experiment xqhuc --use-default-soil-params; then
    echo "Validation passed"
fi
```

---

## Troubleshooting

### Command not found

Ensure package is installed:
```bash
pip install -e .
```

### Permission denied

Make sure you have write access to output directories:
```bash
mkdir -p validation_outputs
chmod u+w validation_outputs
```

### Missing data

Check that input files exist:
```bash
ls ~/annual_mean/xqhuc/
ls ~/dump2hold/xqhuj/datam/
```

For more troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

Last updated: v0.2.2 (2025-01-25)

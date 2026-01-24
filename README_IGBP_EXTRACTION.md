# IGBP Regional Means Extraction

## Overview

The IGBP vegetation fraction observations need to be extracted from the raw NetCDF file and saved as regional means before running validation.

## Quick Start

**On HPC (with iris environment activated):**

```bash
# Make sure you're in the repository root
cd /path/to/utils_cmip7

# Run the extraction script
python scripts/extract_igbp_regional_means.py
```

This will:
- Read `src/utils_cmip7/data/obs/qrparm.veg.frac_igbp.pp.hadcm3bl.nc`
- Compute regional means for all RECCAP2 regions + global
- Save to `src/utils_cmip7/data/obs/igbp_regional_means.csv`

## Output Format

The CSV file will have:
- **Rows**: Regions (North_America, South_America, Europe, Africa, North_Asia, Central_Asia, East_Asia, South_Asia, South_East_Asia, Oceania, global)
- **Columns**: PFTs (BL, NL, C3, C4, shrub, bare_soil)

Example:
```csv
region,BL,NL,C3,C4,shrub,bare_soil
North_America,0.123,0.234,0.345,0.045,0.067,0.186
South_America,0.456,0.012,0.234,0.123,0.089,0.086
...
global,0.234,0.123,0.289,0.089,0.078,0.187
```

## After Extraction

Once the CSV is generated, run validation:

```bash
python scripts/validate_experiment.py xqhuc
```

The validation will now include:
- Regional comparisons for all PFT metrics
- Unified heatmap with carbon + vegetation metrics
- Individual PFT plots (bar charts, time series)
- CSV output with veg metrics bias statistics

## Troubleshooting

**Error: IGBP file not found**
- Make sure `qrparm.veg.frac_igbp.pp.hadcm3bl.nc` exists in `src/utils_cmip7/data/obs/`

**Error: ModuleNotFoundError: No module named 'iris'**
- Activate your iris/conda environment first: `conda activate iris`

**Error: Region extraction failed**
- Check that RECCAP mask file is available and correctly configured

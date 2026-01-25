# Migration Guide: Legacy → v0.2.2+

This guide helps you update code from legacy imports to the new package structure.

---

## Quick Reference

| Old Import | New Import (v0.2.2+) |
|------------|---------------------|
| `from analysis import extract_annual_means` | `from utils_cmip7.diagnostics import extract_annual_means` |
| `from plot import plot_timeseries_grouped` | `from utils_cmip7.plotting import plot_timeseries_grouped` |
| `from analysis import try_extract, stash` | `from utils_cmip7.io import try_extract, stash` |

---

## Breaking Changes in v0.2.2

### 1. Plotting Functions Migrated

**Before (v0.2.1):**
```python
from plot import plot_timeseries_grouped, plot_regional_pie

plot_timeseries_grouped(data, ['exp1'], 'global', outdir='./plots')
plot_regional_pie(data, 'GPP', 'exp1', 2020, outdir='./plots')
```

**After (v0.2.2):**
```python
from utils_cmip7.plotting import plot_timeseries_grouped, plot_regional_pie

plot_timeseries_grouped(data, ['exp1'], 'global', outdir='./plots')
plot_regional_pie(data, 'GPP', 'exp1', 2020, outdir='./plots')
```

**New Features:**
- All plotting functions now accept `ax` parameter for custom axes
- Plotting functions split into logical modules:
  - `plotting.timeseries` - Time series plots
  - `plotting.spatial` - Regional/spatial plots
  - `plotting.styles` - Styling utilities

### 2. CLI Commands Available

**New in v0.2.2:**
```bash
# Extract from preprocessed annual means
utils-cmip7-extract-preprocessed xqhuc

# Extract from raw monthly files
utils-cmip7-extract-raw xqhuj --start-year 2000 --end-year 2010

# Validate single experiment (requires soil parameters)
utils-cmip7-validate-experiment xqhuc --use-default-soil-params

# Generate PPE validation report
utils-cmip7-validate-ppe xqhuc --top-n 20

# Get help
utils-cmip7-extract-preprocessed --help
utils-cmip7-extract-raw --help
utils-cmip7-validate-experiment --help
utils-cmip7-validate-ppe --help
```

### 3. Soil Parameters (v0.2.1.1+)

**New in v0.2.1.1:**
```python
from utils_cmip7.soil_params import SoilParamSet

# Load from default
params = SoilParamSet.from_default()

# Load from file
params = SoilParamSet.from_file('params.json')

# Load from UM log
params = SoilParamSet.from_log_file('rose_log.txt')

# Extract BL tree parameters
bl_params = params.to_overview_table_format()
```

---

## Module Organization (v0.2.2)

```
utils_cmip7/
├── io/                  # NetCDF loading, STASH handling, file discovery
│   ├── stash.py
│   ├── file_discovery.py
│   ├── extract.py
│   └── obs_loader.py
├── processing/          # Temporal/spatial aggregation, unit conversions
│   ├── spatial.py
│   ├── temporal.py
│   ├── regional.py
│   └── metrics.py
├── diagnostics/         # Carbon-cycle diagnostics
│   ├── extraction.py
│   ├── raw.py
│   └── metrics.py
├── validation/          # Model validation against observations
│   ├── compare.py
│   ├── visualize.py
│   ├── overview_table.py
│   └── outputs.py
├── plotting/            # Visualization utilities
│   ├── timeseries.py   # NEW in v0.2.2
│   ├── spatial.py      # NEW in v0.2.2
│   ├── styles.py       # NEW in v0.2.2
│   ├── ppe_viz.py
│   └── ppe_param_viz.py
├── soil_params/         # Soil parameter management (v0.2.1.1+)
│   ├── params.py
│   └── parsers.py
├── config.py            # Configuration and constants
└── cli.py               # NEW in v0.2.2: Command-line entry points
```

---

## Step-by-Step Migration

### Example: Update Analysis Script

**Before:**
```python
import sys
import os
sys.path.append(os.path.expanduser('~/scripts/utils_cmip7'))

from analysis import extract_annual_means
from plot import plot_timeseries_grouped

# Extract data
data = extract_annual_means(['exp1'], regions=['global'])

# Plot
plot_timeseries_grouped(data, ['exp1'], 'global', outdir='./plots')
```

**After:**
```python
from utils_cmip7.diagnostics import extract_annual_means
from utils_cmip7.plotting import plot_timeseries_grouped

# Extract data
data = extract_annual_means(['exp1'], regions=['global'])

# Plot
plot_timeseries_grouped(data, ['exp1'], 'global', outdir='./plots')
```

**Changes:**
1. Remove path manipulation (`sys.path.append`)
2. Update imports to use package structure
3. Ensure package is installed: `pip install -e .`

---

## Backward Compatibility

### Deprecation Timeline

| Version | Status | Notes |
|---------|--------|-------|
| v0.2.0  | Legacy imports deprecated | Warnings added |
| v0.2.2  | New structure available | Backward-compatible wrappers |
| v0.3.0  | Legacy wrappers remain | API frozen |
| v1.0.0  | Legacy imports removed | Breaking change |

### Gradual Migration

You can migrate gradually. Legacy imports still work with deprecation warnings:

```python
# Still works in v0.2.2 (with warning)
from plot import plot_timeseries_grouped

# Recommended: Migrate when convenient
from utils_cmip7.plotting import plot_timeseries_grouped
```

---

## Common Issues

### Issue: ImportError after upgrading

**Symptom:**
```
ImportError: No module named 'utils_cmip7'
```

**Solution:**
Install the package in editable mode:
```bash
cd ~/path/to/utils_cmip7
pip install -e .
```

### Issue: DeprecationWarning spam

**Symptom:**
```
DeprecationWarning: Importing from 'plot' is deprecated...
```

**Solution:**
Update your imports to the new structure (see Quick Reference above).

**Temporary workaround:**
```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
```

### Issue: Plotting functions don't accept Axes

**Symptom:**
```
TypeError: plot_timeseries_grouped() got an unexpected keyword argument 'ax'
```

**Solution:**
This is a new feature in v0.2.2. Update to the latest version:
```bash
cd ~/path/to/utils_cmip7
git pull
pip install -e . --force-reinstall
```

---

## Testing Your Migration

After migrating, verify everything works:

```bash
# Test imports
python -c "from utils_cmip7 import extract_annual_means; print('✓ Imports work')"

# Test CLI
utils-cmip7-extract-preprocessed --help

# Run your analysis scripts
python scripts/my_analysis.py
```

---

## Getting Help

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Check [API_REFERENCE.md](API_REFERENCE.md) for detailed API docs
- Report issues: https://github.com/Climateyousheng/utils_cmip7/issues

---

## Version-Specific Guides

### Upgrading from v0.2.0 → v0.2.1

- No breaking changes
- Soil parameter features added (optional)

### Upgrading from v0.2.1 → v0.2.2

- **Plotting imports changed** (backward-compatible)
- **CLI tools added** (new feature)
- Update imports to avoid deprecation warnings

### Upgrading from v0.2.2 → v0.3.0 (upcoming)

- API will be frozen
- No new breaking changes expected
- Good time to complete migration

---

Last updated: v0.2.2 (2025-01-25)

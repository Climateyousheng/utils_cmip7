# CLAUDE.md

This file provides architectural guidance and technical reference for the `utils_cmip7` Python package.

---

## Part I: Architecture Review and Development Roadmap

### 1. Purpose and Scope

`utils_cmip7` is intended to serve as a common toolkit for:

- **Extracting carbon-cycle and related variables** from Unified Model (UM) NetCDF output
- **Aggregating diagnostics** spatially (global, regional using RECCAP2) and temporally (monthly → annual)
- **Producing standardised plots** for evaluation and validation
- **Supporting intercomparison and validation workflows** across multiple UM simulations

The package is expected to be used in:
- HPC environments (bc4, bp1)
- Batch post-processing workflows
- Interactive Python analysis (Jupyter / VS Code)
- Long-term research projects with evolving diagnostics

### 2. Current Strengths

- Clear scientific intent: carbon-cycle diagnostics from UM outputs
- Practical, working scripts for real research workflows
- Minimal dependencies and straightforward Python usage
- MIT-licensed, enabling reuse and extension

These are strong foundations; the primary need is structural maturity, not a rewrite.

### 3. Key Limitations Identified

#### 3.1 Repository Structure
- Flat module layout (`analysis.py`, `plot.py`) does not scale
- Scripts, logic, and examples are mixed together
- No formal Python package structure or installer

#### 3.2 Packaging and Installation
- Cannot be installed via pip
- Imports rely on local paths or ad-hoc environment setup
- No versioning or release strategy

#### 3.3 Documentation
- README is narrative but not API-oriented
- No standard API reference or structured documentation
- Limited examples of real workflows with expected outputs

#### 3.4 Testing and Reliability
- No unit tests or regression tests
- No continuous integration (CI)
- Risk of silent breakage as code evolves

#### 3.5 Data Handling Assumptions
- Hard-coded directory layouts and filename patterns
- No validation of NetCDF contents (variables, dimensions, units)
- Limited abstraction between I/O and scientific logic

#### 3.6 Plotting and Diagnostics
- Plotting routines are tightly coupled to specific workflows
- Limited flexibility for reuse in papers or notebooks
- No standardised plotting style or configuration

### 4. Recommended Improvements

#### 4.1 Convert to a Proper Python Package (High Priority)

**Target Package Structure:**
```
utils_cmip7/                    # Repository root
├── pyproject.toml              # Modern Python packaging (PEP 621)
├── README.md                   # User-facing documentation
├── CLAUDE.md                   # Architecture and developer guide (this file)
├── LICENSE                     # MIT License
├── .gitignore
│
├── src/
│   └── utils_cmip7/           # Importable package
│       ├── __init__.py        # Package entry point, version
│       ├── io/                # I/O layer
│       │   ├── __init__.py
│       │   ├── stash.py       # STASH code mappings
│       │   ├── file_discovery.py  # find_matching_files, month codes
│       │   └── netcdf_loader.py   # Load and validate NetCDF
│       ├── processing/        # Scientific computation layer
│       │   ├── __init__.py
│       │   ├── spatial.py     # global_total_pgC, regional aggregation
│       │   ├── temporal.py    # compute_annual_mean, monthly means
│       │   ├── units.py       # var_dict, unit conversions
│       │   └── masks.py       # load_reccap_mask, region_mask
│       ├── diagnostics/       # Derived variables and metrics
│       │   ├── __init__.py
│       │   ├── carbon_cycle.py    # NEP, Land Carbon, etc.
│       │   └── extraction.py      # extract_annual_means (high-level)
│       └── plotting/          # Visualization layer
│           ├── __init__.py
│           ├── timeseries.py  # plot_timeseries_grouped, plot_pft_timeseries
│           ├── spatial.py     # plot_regional_pie(s)
│           ├── comparison.py  # plot_pft_grouped_bars
│           └── styles.py      # Matplotlib style configuration
│
├── scripts/                    # Command-line tools
│   ├── extract_raw.sh
│   ├── extract_raw.py
│   └── extract_preprocessed.py
│
├── tests/                      # pytest test suite
│   ├── __init__.py
│   ├── conftest.py            # Shared fixtures
│   ├── test_io/
│   │   ├── test_stash.py
│   │   └── test_file_discovery.py
│   ├── test_processing/
│   │   ├── test_spatial.py
│   │   ├── test_temporal.py
│   │   └── test_units.py
│   ├── test_diagnostics/
│   │   └── test_carbon_cycle.py
│   └── fixtures/              # Small synthetic NetCDF test data
│
├── examples/                   # Usage examples
│   ├── notebooks/
│   │   └── xqhuj_xqhuk_carbon_store.ipynb
│   └── sample_workflows/
│       └── basic_analysis.py
│
└── dev/                        # Development tools
    ├── debug_plot.py
    └── diagnose_extraction.py
```

**Migration Path from Current Structure:**

| Current File | New Location | Notes |
|--------------|--------------|-------|
| `analysis.py` | Split into `io/`, `processing/`, `diagnostics/` | Separate concerns |
| `plot.py` | `plotting/*.py` | One module per plot type |
| `scripts/*.py` | `scripts/*.py` (unchanged) | Keep as CLI tools |
| `examples/*.ipynb` | `examples/notebooks/` | Organize by type |
| `dev/*.py` | `dev/` (unchanged) | Development utilities |

**Example `pyproject.toml`:**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "utils_cmip7"
version = "0.1.0"
description = "Carbon cycle analysis toolkit for Unified Model outputs"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@bristol.ac.uk"}
]
keywords = ["climate", "carbon-cycle", "CMIP", "UM", "NetCDF"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20",
    "matplotlib>=3.3",
    "iris>=3.0",        # UM file handling
    "cartopy>=0.20",    # For geographical plotting
    "netCDF4>=1.5",     # NetCDF file I/O
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=3.0",
    "flake8>=4.0",
    "black>=22.0",
    "isort>=5.10",
    "mypy>=0.950",
]
docs = [
    "sphinx>=4.5",
    "sphinx-rtd-theme>=1.0",
]

[project.urls]
Homepage = "https://github.com/Climateyousheng/utils_cmip7"
Repository = "https://github.com/Climateyousheng/utils_cmip7"
Issues = "https://github.com/Climateyousheng/utils_cmip7/issues"

[project.scripts]
# Entry points for command-line tools
utils-cmip7-extract = "utils_cmip7.cli:extract_main"
utils-cmip7-plot = "utils_cmip7.cli:plot_main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=utils_cmip7 --cov-report=html --cov-report=term"

[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

**Key Improvements:**

1. **Proper src-layout:** Prevents accidental imports from repository root during development
2. **Separation of concerns:** Clear boundaries between I/O, processing, diagnostics, and plotting
3. **Dependency management:** Explicit version requirements prevent compatibility issues
4. **Entry points:** Scripts become proper CLI tools: `utils-cmip7-extract xqhuc`
5. **Test infrastructure:** pytest with fixtures and coverage tracking
6. **Development tools:** Linting, formatting, type checking configured
7. **Semantic versioning:** Clear version progression (0.1.0 → 0.2.0 → 1.0.0)

**Installation Workflow:**

```bash
# Development install (editable)
cd utils_cmip7
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# Now import works from anywhere
python -c "from utils_cmip7 import extract_annual_means"

# CLI tools available system-wide
utils-cmip7-extract xqhuc --outdir ./plots
```

**Version Management:**

- Use git tags: `git tag v0.1.0 && git push --tags`
- Update version in `pyproject.toml` for each release
- Consider `setuptools_scm` for automatic versioning from git tags
- Document breaking changes in CHANGELOG.md

**Benefits:**

- **No more path hacks:** Works across projects, clusters, and users
- **Reproducibility:** Pin versions in requirements or environment files
- **Collaboration:** Standard structure familiar to Python developers
- **Testing:** Clear separation enables comprehensive unit tests
- **Distribution:** Can publish to PyPI or internal package index
- **Documentation:** Sphinx can auto-generate API docs from docstrings

#### 4.2 Improve Documentation and Developer Guidance (High Priority)

Expand README with:
- Installation instructions
- Minimal working example
- Expected input data structure
- Typical validation workflow

Use docstrings throughout (NumPy or Google style).

Treat this file (CLAUDE.md) as:
- Architectural overview
- Design rationale
- Long-term roadmap

Optionally introduce Sphinx or MkDocs later.

**Benefit:** Reduces onboarding time and prevents design drift.

#### 4.3 Refactor Code for Separation of Concerns (Medium–High Priority)

Separate responsibilities:
- **I/O layer:** Locating files, loading NetCDF, STASH handling
- **Processing layer:** Aggregation, unit conversion, derived variables
- **Diagnostics layer:** Standard metrics and validation logic
- **Plotting layer:** Visualisation only

Avoid mixing filesystem logic with numerical computation.

Replace fragile shell scripts with Python CLIs where feasible.

**Benefit:** Easier testing, extension, and reuse.

#### 4.4 Introduce Testing and CI (High Priority)

Add pytest-based unit tests:
- Aggregation logic
- Unit conversions
- Variable selection and filtering

Include small synthetic NetCDF test fixtures.

Set up GitHub Actions to run:
- Tests
- Linting (flake8, isort)
- Optional type checks (mypy)

**Benefit:** Protects scientific correctness as the code evolves.

#### 4.5 Improve Data Validation and Robustness (Medium Priority)

Validate inputs early:
- Required variables present
- Dimensions consistent
- Units sensible

Consider lightweight CF-compliance checks.

Allow configuration via YAML/TOML instead of hard-coded paths.

**Benefit:** Fewer silent failures in large batch workflows.

#### 4.6 Make Plotting More Modular and Publication-Ready (Medium Priority)

- Allow plotting functions to accept existing Figure/Axes
- Centralise style configuration (Matplotlib stylesheets)
- Ensure plots can be reused directly in papers without rewriting code

**Benefit:** Cleaner analysis notebooks and consistent figures.

#### 4.7 Project Sustainability and Collaboration (Low–Medium Priority)

Add:
- `CONTRIBUTING.md`
- Issue and PR templates
- Define a roadmap in the README
- Include example datasets or links to public UM outputs

**Benefit:** Encourages adoption beyond the original author.

### 5. Suggested Development Roadmap

**v0.1 – Stabilisation**
- Package structure
- Installation via pip
- Basic tests
- Improved README

**v0.2 – Robustness**
- Data validation
- Refactored plotting
- Configurable workflows

**v1.0 – Community-Ready**
- Stable API
- CI with coverage
- Benchmarked diagnostics
- Used in at least one published analysis

### 6. Guiding Principles

- **Scientific correctness over convenience**
- **Explicit assumptions**
- **Minimal but extensible abstractions**
- **Designed for HPC and long-running workflows**
- **Readable by future-you in 5 years**

This document should be updated as architectural decisions are made, serving as a living design reference for `utils_cmip7`.

---

## Part II: Technical API Reference

### Overview

`utils_cmip7` is a Python toolkit for carbon cycle analysis from UM climate model outputs. It provides:
- STASH code mapping and variable extraction
- Regional and global spatial aggregation
- Temporal processing (monthly → annual means)
- Publication-quality visualization

## Module: analysis.py

Core data processing module (~830 lines) for extracting and computing carbon cycle variables from UM output files.

### Key Functions

#### STASH Code Management
- `stash(var_name)` - Maps short names to PP-format STASH codes
  - Example: `'gpp'` → `'m01s03i261'`
- `stash_nc(var_name)` - Maps short names to NetCDF numeric STASH codes
  - Example: `'gpp'` → `3261`
- Covers 30+ variables: carbon cycle, ocean biogeochemistry, atmosphere, radiation

#### File Discovery
- `find_matching_files(expt_name, model, up, start_year=None, end_year=None, base_dir="~/dump2hold")`
  - Locates UM output files for a given experiment
  - Supports both alpha month codes (`ja`, `fb`, `mr`, etc.) and numeric codes (`11`-`91`, `a1`-`c1`)
  - Returns sorted list of `(year, month, filepath)` tuples

- `decode_month(mon_code)` - Parses UM two-letter month codes
  - Alpha: `ja`(Jan), `fb`(Feb), `mr`(Mar), `ar`(Apr), `my`(May), `jn`(Jun), `jl`(Jul), `ag`(Aug), `sp`(Sep), `ot`(Oct), `nv`(Nov), `dc`(Dec)
  - Numeric: `11`-`91` (Jan-Sep), `a1`(Oct), `b1`(Nov), `c1`(Dec)

#### Unit Conversions (`var_dict`)
Automatic unit conversions applied to standard variables:

| Variable | Conversion | From → To |
|----------|------------|-----------|
| Ocean flux (`m01s00i250`) | `(12/44)*3600*24*360*1e-12` | kgCO2/m²/s → PgC/yr |
| GPP, NPP, Rh | `3600*24*360*1e-12` | kgC/m²/s → PgC/yr |
| CV, CS | `1e-12` | kgC/m² → PgC |
| Air flux (`m02s30i249`) | `12/1000*1e-12` | molC/m²/yr → PgC/yr |
| Precipitation | `86400` | kg/m²/s → mm/day |
| Total CO2 | `28.97/44.01*1e6` | mmr → ppmv |

#### Spatial Processing

- `global_total_pgC(cube, var)` - Area-weighted sum (for extensive variables like fluxes)
- `global_mean_pgC(cube, var)` - Area-weighted mean (for intensive variables)
- `compute_regional_annual_mean(cube, var, region)` - Regional analysis using RECCAP mask
  - Handles pfts dimension (4D cubes)
  - Applies region-specific masking
  - Returns dict with `years`, `data`, `units`, `name`, `region`

- `load_reccap_mask()` - Loads RECCAP2 regional mask
  - 11 RECCAP2 regions: North_America, South_America, Europe, Africa (combines 4+5), North_Asia, Central_Asia, East_Asia, South_Asia, South_East_Asia, Oceania
  - Plus special regions: `Africa` (combined), `global`

- `region_mask(region)` - Creates binary mask for a specific region

#### Temporal Processing

- `compute_annual_mean(cube, var)` - Converts monthly data to annual means
  - Handles 360-day calendars
  - Adds missing time bounds automatically
  - Groups by year and averages
  - Returns dict with `years`, `data`, `name`, `units`

- `compute_monthly_mean(cube, var)` - Converts to monthly means (fractional years)
  - For TRIFFID variables with special time coordinates
  - Returns fractional year format (e.g., 1850.083 for Feb 1850)

- `merge_monthly_results(results, require_full_year=False)` - Merges multiple monthly outputs
  - Combines results from multiple files
  - Averages duplicates at same (year, month)
  - Optionally filters to complete years only

#### Variable Extraction

- `try_extract(cubes, code, stash_lookup_func=None, debug=False)` - Robust cube extraction
  - Works with both PP STASH objects and NetCDF numeric stash_codes
  - Accepts MSI strings, short names, or numeric codes
  - Handles multiple STASH formats automatically

#### High-Level Analysis

- `extract_annual_means(expts_list, var_list=None, var_mapping=None, regions=None)` - **Main workhorse function**

  **Default Variables Processed:**
  - `soilResp` (S resp) - Soil respiration
  - `soilCarbon` (S carb) - Soil carbon storage
  - `VegCarb` (V carb) - Vegetation carbon storage
  - `fracPFTs` (Others) - Plant functional type fractions (1-9)
  - `GPP` - Gross primary production
  - `NPP` - Net primary production
  - `fgco2` - Ocean CO2 flux (global only)
  - `temp` - Surface air temperature
  - `precip` - Precipitation

  **Computed Derived Variables:**
  - `NEP = NPP - soilResp` (Net ecosystem production)
  - `Land Carbon = soilCarbon + VegCarb + NEP` (Total land carbon)
  - `Trees Total = PFT1 + PFT2` (Total tree cover fraction)

  **Returns:**
  ```python
  dict[expt][region][var] -> {
      "years": np.array,
      "data": np.array,
      "units": str,
      "name": str,
      "region": str
  }
  ```

  **Example:**
  ```python
  ds = extract_annual_means(
      expts_list=['xqhuc'],
      regions=['global', 'Africa']
  )
  # Access: ds['xqhuc']['global']['GPP']['data']
  ```

---

### High-Level Analysis (Raw Monthly Files)

- `extract_annual_mean_raw(expt, base_dir='~/dump2hold', start_year=None, end_year=None)` - **Extract from raw monthly files**

  **Purpose:** Process raw monthly UM output files directly (not pre-processed annual mean NetCDF files).

  **Use case:** When you have raw monthly files in `~/dump2hold/expt/datam/` but no pre-processed annual means.

  **Variables Extracted:**
  - `GPP` - Gross Primary Production
  - `NPP` - Net Primary Production
  - `soilResp` - Soil respiration
  - `VegCarb` - Vegetation carbon
  - `soilCarbon` - Soil carbon
  - `NEP` - Net Ecosystem Production (derived: NPP - soilResp)

  **Workflow:**
  1. Finds raw monthly files using `find_matching_files()`
  2. Extracts variables using `try_extract()` with STASH lookup
  3. Computes monthly means with `compute_monthly_mean()`
  4. Merges into annual means with `merge_monthly_results()`
  5. Returns dict with years, data, units for each variable

  **Returns:**
  ```python
  {
      'GPP': {'years': array, 'data': array, 'units': 'PgC/year', 'name': str},
      'NPP': {'years': array, 'data': array, 'units': 'PgC/year', 'name': str},
      'soilResp': {'years': array, 'data': array, 'units': 'PgC/year', 'name': str},
      'VegCarb': {'years': array, 'data': array, 'units': 'PgC', 'name': str},
      'soilCarbon': {'years': array, 'data': array, 'units': 'PgC', 'name': str},
      'NEP': {'years': array, 'data': array, 'units': 'PgC/year', 'name': str},
  }
  ```

  **Example:**
  ```python
  # Extract from raw monthly files
  data = extract_annual_mean_raw('xqhuj')

  # Plot GPP time series
  import matplotlib.pyplot as plt
  plt.plot(data['GPP']['years'], data['GPP']['data'])
  plt.xlabel('Year')
  plt.ylabel('GPP (PgC/year)')
  plt.show()

  # Access NEP
  nep_years = data['NEP']['years']
  nep_values = data['NEP']['data']
  ```

  **Comparison with `extract_annual_means()`:**

  | Feature | `extract_annual_mean_raw()` | `extract_annual_means()` |
  |---------|----------------------------|-------------------------|
  | Input files | Raw monthly UM output | Pre-processed annual mean NetCDF |
  | Location | `~/dump2hold/expt/datam/` | `~/annual_mean/expt/` |
  | Regional analysis | No (global only) | Yes (RECCAP2 regions) |
  | Processing speed | Slower (reads many files) | Faster (reads 3 files) |
  | Use case | Direct from UM runs | After CDO post-processing |

---

## Module: plot.py

Visualization module (~431 lines) for publication-quality carbon cycle plots.

### Configuration

**Legend Labels & Colors:**
```python
legend_labels = {
    "xqhsh": "PI LU COU spinup",
    "xqhuc": "PI HadCM3 spinup",
}

color_map = {
    "xqhsh": "k",
    "xqhuc": "r",
}
```

### Plotting Functions

#### 1. `plot_timeseries_grouped(data, expts_list, region, outdir, ...)`
Multi-variable time series in a grid layout.

**Parameters:**
- `data` - Dict from `extract_annual_means()`
- `expts_list` - List of experiment names to plot
- `region` - Region name (e.g., 'global', 'Africa')
- `outdir` - Output directory for plots
- `legend_labels` - Optional dict for custom labels
- `color_map` - Optional dict for custom colors
- `exclude` - Tuple of variable prefixes to exclude (default: `("fracPFTs",)`)
- `ncols` - Number of columns in grid (default: 3)
- `show` - Display plot interactively (default: False)

**Features:**
- Auto-groups variables by prefix
- Multiple experiments overlaid per panel
- Drops first year of data (spinup)
- Grid layout with shared x-axis
- Auto-generated titles with units

**Output:** `allvars_{region}_{expts}_timeseries.png` (300 DPI)

**Example:**
```python
plot_timeseries_grouped(
    ds,
    expts_list=['xqhuc', 'xqhsh'],
    region='global',
    outdir='./plots/',
    exclude=('fracPFTs', 'temp')
)
```

---

#### 2. `plot_pft_timeseries(data, expts_list, region, outdir, ...)`
PFT fraction time series for one region.

**Parameters:**
- `pfts` - Tuple of PFT numbers to plot (default: `(1,2,3,4,5)`)
- Other parameters same as `plot_timeseries_grouped`

**Features:**
- 2×3 subplot layout for PFT 1-5
- Multiple experiments overlaid per PFT
- Drops first year of data
- Y-axis in fractions (0-1)

**Output:** `fracPFTs_{region}_timeseries.png` (300 DPI)

**Example:**
```python
plot_pft_timeseries(
    ds,
    expts_list=['xqhuc'],
    region='global',
    outdir='./plots/',
    pfts=(1,2,3,4,5)
)
```

---

#### 3. `plot_regional_pie(data, varname, expt, year, outdir, ...)`
Single pie chart showing regional distribution.

**Parameters:**
- `varname` - Variable name (e.g., 'soilResp', 'GPP')
- `expt` - Experiment name
- `year` - Year to plot
- Other parameters as above

**Features:**
- Shows value and percentage for each region
- Donut chart style (wedge width = 0.45)
- Auto-calculates totals

**Output:** `{var}_regional_pie_{expt}_{year}.png` (300 DPI)

**Example:**
```python
plot_regional_pie(
    ds,
    varname='GPP',
    expt='xqhuc',
    year=2000,
    outdir='./plots/'
)
```

---

#### 4. `plot_regional_pies(data, varname, expts_list, year, outdir, ...)`
Side-by-side pie charts for multiple experiments.

**Parameters:**
- Same as `plot_regional_pie` but accepts `expts_list` instead of single `expt`

**Features:**
- Compares experiments for same variable and year
- Side-by-side layout (1 row × N experiments)
- Consistent region ordering across experiments

**Output:** `{var}_regional_pies_{year}.png` (300 DPI)

**Example:**
```python
plot_regional_pies(
    ds,
    varname='NPP',
    expts_list=['xqhuc', 'xqhsh'],
    year=2000,
    outdir='./plots/'
)
```

---

#### 5. `plot_pft_grouped_bars(data, expts_list, year, outdir, ...)`
Grouped bar charts for PFT fractions by region.

**Parameters:**
- `year` - Year to plot
- `pfts` - Tuple of PFT numbers (default: `(1,2,3,4,5)`)
- Other parameters as above

**Features:**
- 2×3 subplot layout for PFT 1-5
- X-axis: regions (consistent ordering across all subplots)
- Grouped bars: one bar per experiment
- Y-axis: fraction (0-1 scale)
- Color-coded by experiment

**Output:** `fracPFTs_1to5_grouped_bars_{year}.png` (300 DPI)

**Example:**
```python
plot_pft_grouped_bars(
    ds,
    expts_list=['xqhuc', 'xqhsh'],
    year=2000,
    outdir='./plots/',
    pfts=(1,2,3,4,5)
)
```

---

## Typical Workflow

### 1. Extract Annual Means from UM Output
```python
import os
import sys
sys.path.append(os.path.expanduser('~/scripts/utils_cmip7'))
from analysis import extract_annual_means

# Extract data for single experiment (all regions)
ds = extract_annual_means(expts_list=['xqhuc'])

# Extract for multiple experiments with specific regions
ds = extract_annual_means(
    expts_list=['xqhuc', 'xqhsh'],
    regions=['global', 'Africa', 'South_America']
)
```

### 2. Generate Visualizations
```python
from plot import (
    plot_timeseries_grouped,
    plot_pft_timeseries,
    plot_regional_pies,
    plot_pft_grouped_bars
)

outdir = './plots/'

# Time series for all variables
plot_timeseries_grouped(
    ds,
    expts_list=['xqhuc'],
    region='global',
    outdir=outdir
)

# PFT fractions over time
plot_pft_timeseries(
    ds,
    expts_list=['xqhuc'],
    region='global',
    outdir=outdir
)

# Regional distribution comparison
plot_regional_pies(
    ds,
    varname='GPP',
    expts_list=['xqhuc', 'xqhsh'],
    year=2000,
    outdir=outdir
)

# PFT regional bar chart
plot_pft_grouped_bars(
    ds,
    expts_list=['xqhuc', 'xqhsh'],
    year=2000,
    outdir=outdir
)
```

### 3. Access Data Directly
```python
# Access global GPP time series for xqhuc
gpp_years = ds['xqhuc']['global']['GPP']['years']
gpp_values = ds['xqhuc']['global']['GPP']['data']
gpp_units = ds['xqhuc']['global']['GPP']['units']  # 'PgC/year'

# Access regional soil carbon for Africa
africa_soilc = ds['xqhuc']['Africa']['soilCarbon']['data']

# Access specific PFT fraction
pft1_global = ds['xqhuc']['global']['fracPFTs']['PFT 1']['data']
```

---

## Input Data Requirements

### Directory Structure
The `extract_annual_means()` function expects annual mean NetCDF files in:
```
~/annual_mean/{expt}/
├── {expt}_pa_annual_mean.nc  (atmosphere - temp, precip)
├── {expt}_pt_annual_mean.nc  (TRIFFID - GPP, NPP, soil resp, carbon stocks, PFTs)
└── {expt}_pf_annual_mean.nc  (ocean - fgco2)
```

### Pre-processing
Use the `annual_mean_cdo.sh` script (from parent project) to generate these files from monthly UM output:
```bash
./annual_mean_cdo.sh "xqhuc" ~/annual_mean pt pd pf
```

---

## STASH Codes Reference

Common variables used in carbon cycle analysis:

| Variable | STASH Code | Description |
|----------|------------|-------------|
| tas | m01s03i236 | Surface air temperature |
| pr | m01s05i216 | Precipitation |
| gpp | m01s03i261 | Gross Primary Production |
| npp | m01s03i262 | Net Primary Production |
| rh | m01s03i293 | Soil respiration |
| cv | m01s19i002 | Vegetation carbon |
| cs | m01s19i016 | Soil carbon |
| frac | m01s19i013 | PFT fractions |
| fgco2 | m02s30i249 | Ocean CO2 flux |
| co2 | m01s00i252 | Atmospheric CO2 concentration |

---

## Notes

- All time series plots drop the first year of data (spinup artifact)
- PFT numbering: 1=Broadleaf tree, 2=Needleleaf tree, 3=C3 grass, 4=C4 grass, 5=Shrub
- Ocean flux (fgco2) is only computed for global region
- All plots use 300 DPI for publication quality
- Default calendar: 360-day (UM standard)
- Regional analysis uses RECCAP2 mask at HadCM3 grid resolution

---

## Troubleshooting

### Problem: Plots only show Trees Total (or other subset of variables)

**Symptom:** When calling `plot_timeseries_grouped()`, only one or two variables appear instead of the expected 9+ variables.

**Root Cause:** Variables are missing from the data structure because `extract_annual_means()` couldn't find them in the NetCDF files.

**Diagnosis:** The improved error handling (added 2026-01) now provides detailed diagnostics during extraction:

```python
from analysis import extract_annual_means

ds = extract_annual_means(expts_list=['xqhuc'])
```

**Output example:**
```
============================================================
Extracting data for experiment: xqhuc
============================================================
Looking in: /home/user/annual_mean/xqhuc/
NetCDF files found: 1
  - xqhuc_pt_annual_mean.nc

Total cubes loaded: 12

Extracting variables...
  ❌ soilResp (rh, m01s03i293): NOT FOUND
  ❌ soilCarbon (cs, m01s19i016): NOT FOUND
  ❌ VegCarb (cv, m01s19i002): NOT FOUND
  ✓ fracPFTs (frac): Found (via stash code 3317)
  ❌ GPP (gpp, m01s03i261): NOT FOUND
  ❌ NPP (npp, m01s03i262): NOT FOUND
  ❌ fgco2 (fgco2, m02s30i249): NOT FOUND
  ❌ temp (tas, m01s03i236): NOT FOUND
  ❌ precip (pr, m01s05i216): NOT FOUND

============================================================
Extraction Summary for xqhuc
============================================================
Variables successfully extracted: 1/9
  Found: fracPFTs
  ⚠ Missing: soilResp, soilCarbon, VegCarb, GPP, NPP, fgco2, temp, precip

  These variables will NOT appear in plots!
============================================================
```

**Solutions:**

1. **Check if annual mean files exist:**
   ```bash
   ls -lh ~/annual_mean/xqhuc/
   ```
   Expected files:
   - `xqhuc_pt_annual_mean.nc` (TRIFFID variables)
   - `xqhuc_pd_annual_mean.nc` (atmosphere variables)
   - `xqhuc_pf_annual_mean.nc` (ocean variables)

2. **Generate missing annual mean files:**
   ```bash
   cd /path/to/scripts
   ./annual_mean_cdo.sh "xqhuc" ~/annual_mean pt pd pf
   ```

3. **Check file contents for STASH codes:**
   ```bash
   ncdump -h ~/annual_mean/xqhuc/xqhuc_pt_annual_mean.nc | grep STASH
   ```
   Should show variables with stash_code attribute (e.g., 3261 for GPP, 3262 for NPP)

4. **Verify source data location:**
   The `annual_mean_cdo.sh` script looks for monthly output in:
   - `~/umdata/xqhuc/pt/*.nc` (TRIFFID)
   - `~/umdata/xqhuc/pd/*.nc` (atmosphere)
   - `~/umdata/xqhuc/pf/*.nc` (ocean)

### Problem: Variables extracted but still missing from regional plots

**Symptom:** Variables show ✓ during extraction but don't appear in regional (non-global) plots.

**Cause:** `fgco2` (ocean CO2 flux) is intentionally skipped for non-global regions (line 843 in analysis.py).

**Solution:** Use `region='global'` when plotting ocean variables.

### Problem: NetCDF STASH codes show 's' suffix in ncdump

**What you see:** When checking NetCDF files:
```bash
ncdump -h file.nc | grep stash_code
```
Shows codes like:
```
stash_code = 3261s ;    ← Note the 's' suffix
stash_code = 19002s ;
```

**Clarification:** The 's' is a **NetCDF CDL type indicator** (short integer), NOT part of the actual value. When iris loads the file, `stash_code` attributes are plain Python integers (e.g., `3261`, not `'3261s'`). This is normal and requires no special handling.

### Problem: Wrong STASH codes in NetCDF files

**Symptom:** Variables exist in files but extraction reports NOT FOUND, and STASH codes don't match expected format.

**Diagnosis:** Check actual STASH codes in files:
```bash
ncdump -h file.nc | grep stash_code
```

**Solution:** If STASH codes differ significantly from expected values (not just 's' suffix), update the `stash()` or `stash_nc()` functions in analysis.py or use custom `var_list` and `var_mapping` parameters:

```python
ds = extract_annual_means(
    expts_list=['xqhuc'],
    var_list=['custom_var1', 'custom_var2'],
    var_mapping=['mapping1', 'mapping2']
)
```

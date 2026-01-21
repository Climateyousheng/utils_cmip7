# utils_cmip7 — API Reference

This document provides the **developer-facing API reference** for `utils_cmip7`.

It is **descriptive**, not normative.
Architectural constraints and stability guarantees are defined in `CLAUDE.md`.

---

## Package Layout (Target)

utils_cmip7/
├── io/ # NetCDF loading, STASH handling, file discovery
├── processing/ # Temporal/spatial aggregation, unit conversions
├── diagnostics/ # Carbon-cycle and soil diagnostics
├── plotting/ # Visualisation utilities (no I/O)
└── cli/ # Command-line entry points


---

## I/O Layer

### `load_dataset(path, *, pattern=None)`
Load UM NetCDF output and return an `xarray.Dataset`.

**Parameters**
- `path` (str): Root directory containing UM output
- `pattern` (str, optional): Filename glob or regex

**Returns**
- `xarray.Dataset`

---

### `try_extract(cubes, code, *, debug=False)`
Robust extraction of variables by STASH code or alias.

**Parameters**
- `cubes`: Iris cube list or xarray dataset
- `code`: STASH string, numeric code, or short alias
- `debug` (bool): Verbose diagnostics

**Returns**
- Extracted cube or `None`

---

## Processing Layer

### `compute_annual_mean(dataset)`
Convert monthly data to annual means.

**Returns**
- `xarray.Dataset`

---

### `compute_monthly_mean(dataset)`
Return monthly means with fractional-year time coordinate.

**Returns**
- `xarray.Dataset`

---

### `global_total_pgC(dataset, var)`
Area-weighted global total for extensive variables.

**Returns**
- `numpy.ndarray`

---

### `global_mean_pgC(dataset, var)`
Area-weighted global mean for intensive variables.

**Returns**
- `numpy.ndarray`

---

## Diagnostics Layer

### `extract_annual_means(expts_list, *, regions=None, var_list=None)`
High-level diagnostic extraction for pre-processed annual-mean data.

**Returns**
dict[expt][region][variable] -> {
"years": np.ndarray,
"data": np.ndarray,
"units": str,
"name": str,
"region": str
}


---

### `extract_annual_mean_raw(expt, *, start_year=None, end_year=None)`
Extract diagnostics directly from raw monthly UM output.

**Returns**
- Dict of annual time series per variable

---

### Derived Diagnostics
- `NEP = NPP - soilResp`
- `LandCarbon = soilCarbon + VegCarb + NEP`
- `TreesTotal = PFT1 + PFT2`

---

## Plotting Layer

> Plotting functions **must not** load data or compute aggregates.

### `plot_timeseries_grouped(data, *, ax=None, **kwargs)`
Grouped time-series plots for multiple variables.

**Parameters**
- `data`: Output of `extract_annual_means`
- `ax` (matplotlib Axes, optional)

---

### `plot_pft_timeseries(data, *, pfts=(1,2,3,4,5), ax=None)`
PFT fraction time series.

---

### `plot_regional_pie(data, varname, *, year, ax=None)`
Regional distribution for a single experiment.

---

### `plot_regional_pies(data, varname, *, year, axes=None)`
Side-by-side regional comparisons.

---

## CLI (Experimental)

Planned entry points:
- `utils-cmip7-extract-raw`
- `utils-cmip7-extract-preprocessed`

Refer to CLI help for arguments.

---

## Stability Notes

Refer to `CLAUDE.md` for API stability guarantees.


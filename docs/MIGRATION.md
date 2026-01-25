# utils_cmip7 — Migration & Refactor Guide

This document tracks **migration steps**, **technical debt**, and **refactor tasks**
from v0.2.x toward v1.0.

---

## Scope

- Preserve scientific behaviour
- Improve structure, packaging, and maintainability
- Maintain backward compatibility during v0.2.x

---

## High Priority (v0.2.1)

### Package Structure
- Introduce `src/` layout
- Move logic into `io/`, `processing/`, `diagnostics/`, `plotting/`
- Add backward-compatible re-exports in legacy modules

### Scripts
- Update `scripts/*.py` to import from `utils_cmip7.*`
- Add fallback imports for non-installed environments

### Configuration
- Validate RECCAP mask path on first use
- Support override via environment variable
- Centralise config checks in `config.py`

### Smoke Tests
- Import resolution
- Annual-mean extraction
- Raw extraction
- Mask loading

---
## High Priority (0.2.1.1)

Intriduce obs and validation, and metrics to ensure proper comparison.

### Formalise metric computation across UM, CMIP6 obs, and validation

The core package now lives under `src/utils_cmip7`, and legacy
`analysis.py` / `plot.py` will be removed. Metric logic must therefore
be formalised within the core modules.

#### 1. Introduce explicit metric definitions in processing/

Add `processing/metrics.py` to define how supported metrics are computed:

- GPP
- NPP
- vegCarbon
- soilCarbon
- tas
- precip

This module defines:
- SUM vs MEAN behaviour
- post-aggregation units

No I/O or dataset-specific logic is permitted here.

#### 2. Add a diagnostic-level metric runner

Add `diagnostics/metrics.py` to orchestrate metric computation by
calling:

- `processing.temporal`
- `processing.spatial`
- `processing.regional`
- `processing.metrics`

This module produces the canonical metric output structure.

#### 3. Canonical metric schema (mandatory)

All metric-producing code must return:

dict[metric][region] -> {
"years": array,
"data": array,
"units": str,
"source": str,
"dataset": str
}


UM workflows, CMIP6 observational datasets, and future model sources
must conform exactly to this schema.

#### 4. Role of obs/

The `obs/` directory contains observational datasets (CSV).
These datasets represent already-aggregated metrics and must be
adapted to the canonical metric schema when loaded.

No aggregation or unit conversion logic may be added to `obs/`.

#### 5. Role of validation/

The `validation/` directory operates strictly on canonical metric
outputs.

Validation code must not:
- load NetCDF
- aggregate data
- compute metrics

It may only compare, summarise, and visualise metric results.

Any deviation from this separation is blocking technical debt.

#### 6. Add new feature, update logs, and overview table

Feature: require soil-parameter set per experiment + update PPE overview table

Goal
When validating an experiment, require an associated soil parameter set (flexible source: parsed from logs, or manually provided). After validation, upsert the experiment row into validation_outputs/random_sampling_combined_overview_table.csv, and write a per-experiment “single validation” output bundle under validation_outputs/ (directory name: single_val_<EXPT> or similar). Only store the Broadleaf (BL) tree parameter values in the overview table.

Context (current repo)

Existing overview table: validation/random_sampling_combined_overview_table.csv (rename validation/ → validation_outputs/ if/when that migration lands).

Validation entrypoint script exists: scripts/validate_experiment.py.

Package has src/utils_cmip7/validation/compare.py and src/utils_cmip7/processing/metrics.py/diagnostics/metrics.py.

Required behaviour

Soil parameter set is mandatory

Validation must fail fast unless soil params are available from one of:

--soil-params (manual override string or key=value list)

--soil-param-file (YAML/JSON; preferred)

--soil-log-file (parse UM/rose/namelist log containing &LAND_CC ... /)

default &LAND_CC block (only used if user explicitly opts in: --use-default-soil-params)

Default soil parameters (if opted in)
Use this as the default full set:

&LAND_CC
 ALPHA=0.08,0.08,0.08,0.040,0.08,
 F0=0.875,0.875,0.900,0.800,0.900,
 G_AREA=0.004,0.004,0.10,0.10,0.05,
 LAI_MIN=4.0,4.0,1.0,1.0,1.0,
 NL0=0.050,0.030,0.060,0.030,0.030,
 R_GROW=0.25,0.25,0.25,0.25,0.25,
 TLOW=-0.0,-5.0,0.0,13.0,0.0,
 TUPP=36.0,31.0,36.0,45.0,36.0,
 Q10=2.0,
 V_CRIT_ALPHA=0.343,
 KAPS=5e-009,
/


Overview table updates (upsert)

Read validation_outputs/random_sampling_combined_overview_table.csv

Identify the row by experiment id column (use existing name; do not introduce a second id column)

If experiment exists → update values

If not → append a new row

Persist in-place (atomic write: write temp file then replace)

Only keep BL-tree parameter columns

For array parameters, store only the BL value in the overview table (assume BL corresponds to index 0 unless your existing code defines otherwise; keep the BL-index mapping in one constant).

Add/update only these columns (example names; match your CSV conventions):

ALPHA_BL, F0_BL, G_AREA_BL, LAI_MIN_BL, NL0_BL, R_GROW_BL, TLOW_BL, TUPP_BL

Scalars: Q10, V_CRIT_ALPHA, KAPS

Do not write the other PFT entries to the overview table.

Per-experiment validation bundle
After validation completes, write an output directory under validation_outputs/:

Directory: validation_outputs/single_val_<EXPT>/ (or single_validation_<EXPT>/—choose one and keep stable)

Must include:

soil_params.json (full structured params + source + BL-index mapping)

metrics_global.csv and metrics_regional.csv (or whatever your canonical outputs are)

validation_scores.csv (the row that is written/updated in the overview table, plus any extra internal columns)

Naming requirement: do not name the directory exactly /single_val_{expt} if you already have conflicts; pick one stable convention and document it here.

Implementation steps (actionable)

A. Add soil parameter model + loaders

New module: src/utils_cmip7/soil_params/params.py

@dataclass SoilParamSet: stores full LAND_CC fields, plus source metadata

Methods:

from_default()

from_dict()

from_file(path)

from_log_text(text) / from_log_file(path) (parse &LAND_CC ... /)

Export helper: to_bl_subset(bl_index=0) -> dict[str, float]

New module: src/utils_cmip7/soil_params/parsers.py

Implement robust LAND_CC parser:

find block start &LAND_CC

accumulate until /

parse scalars and comma-separated lists (strip trailing commas)

Define one constant somewhere central:

BL_INDEX = 0 (or a mapping if your PFT order differs)

B. Wire into validation

Update scripts/validate_experiment.py to require soil params:

Add args: --soil-param-file, --soil-log-file, --soil-params, --use-default-soil-params

Load SoilParamSet before running validation

If missing and no default opt-in: exit non-zero with clear message

Update/extend src/utils_cmip7/validation/:

Add overview_table.py with:

load_overview_table(path)

upsert_overview_row(df, expt_id, bl_params, scores)

write_atomic_csv(df, path)

Called at the end of validation.

C. Update output writing

Add helper: src/utils_cmip7/validation/outputs.py

write_single_validation_bundle(outdir, soil_params, metrics, scores)

Ensure outdir = validation_outputs/<bundle_name> is created

D. Add minimal tests

tests/test_soil_params_parser.py

parse the provided default block and validate keys + list lengths

tests/test_overview_upsert.py

upsert updates existing row and appends new row; BL columns only

Notes / constraints

Do not hardcode repo-relative paths for overview table; accept path argument with default validation_outputs/random_sampling_combined_overview_table.csv (but keep backward-compatible fallback to current validation/random_sampling_combined_overview_table.csv for one release if needed).

Keep soil params flexible and provenance-aware (source = "default"|"manual"|"log"|"file").
## Medium Priority (v0.2.2 – v0.3.0)

### Plotting Refactor
- Split plotting into:
  - `plotting/styles.py`
  - `plotting/timeseries.py`
  - `plotting/spatial.py`
- Ensure all plotting functions accept Axes

### CLI
- Implement CLI wrappers for extraction functions
- Add `--help` documentation
- Register entry points in `pyproject.toml`

### Soil Parameters
- Move soil parameter modules into `diagnostics/soil_params/`
- Update imports and exports

### Documentation
- Create migration examples (old → new imports)
- Add troubleshooting notes

---

## Testing & CI

### Tests
- I/O: STASH mapping, file discovery
- Processing: spatial + temporal aggregation
- Diagnostics: derived variables
- Plotting: smoke tests only

### CI
- GitHub Actions
- Python 3.8–3.11
- Linting + tests
- Optional coverage

---

## Low Priority (Pre-v1.0)

- Remove commented-out code
- Add type hints
- Improve error messages
- Add example datasets
- Performance profiling
- Add `CHANGELOG.md`

---

## Completion Criteria for v1.0

- All high + medium priority items complete
- ≥80% test coverage
- CI passing on all supported versions
- Public API frozen
- Documentation complete
- At least one published analysis using the package


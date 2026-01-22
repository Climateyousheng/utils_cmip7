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


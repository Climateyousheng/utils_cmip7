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


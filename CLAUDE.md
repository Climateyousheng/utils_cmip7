# CLAUDE.md — utils_cmip7 (AI Control File)

## Purpose
This file defines **binding architectural rules and constraints** for the
`utils_cmip7` codebase.

It is intentionally concise and normative.

API documentation, tutorials, examples, and troubleshooting guides
**must not** live in this file.

---

## Project Scope (Fixed)

`utils_cmip7` is a Python toolkit for:
- Analysing carbon-cycle variables from Unified Model (UM) outputs
- Producing standard diagnostics and plots
- Supporting validation and intercomparison workflows

Target environments:
- HPC batch post-processing
- Interactive Python analysis (Jupyter / VS Code)
- Long-running, reproducible research workflows

---

## Current State (v0.3.0)

- Core extraction, processing, and plotting logic exists and is scientifically correct.
- Module structure is **stabilized** with clear public API guarantees.
- 174 tests with 24% coverage, CI/CD across Python 3.8-3.11.
- Backward compatibility with existing analysis scripts is preserved.

Scientific behaviour must not change unless explicitly documented.

---

## Binding Design Rules (MANDATORY)

1. No filesystem discovery or NetCDF I/O inside plotting functions
2. NetCDF loading must be isolated in a dedicated I/O layer
3. Diagnostics must return data objects (xarray.Dataset, dict), never plots
4. Aggregation logic must not be duplicated across modules
5. Plotting functions must accept matplotlib Axes objects
6. No hard-coded paths outside CLI or configuration layers
7. **NEVER iterate over sets when order matters** — always use sorted(set) or maintain deterministic order
8. **NEVER use positional matching (zip, indexing) when names are available** — use dict lookup instead

Violations constitute technical debt and must be recorded explicitly.

---

## API Stability Matrix

| Component                | Stability     |
|-------------------------|---------------|
| Soil carbon diagnostics | Stable        |
| Carbon flux diagnostics | Stable        |
| Annual-mean processing  | Provisional   |
| Raw data extraction     | Provisional   |
| Plotting API            | Unstable      |
| CLI interface           | Experimental  |

Only *Stable* components may be relied upon in long-lived scripts.

**As of v0.3.0 (2026-01-26)**, the API stability matrix is frozen for the v0.3.x series.
No breaking changes will be introduced to "Stable" components until v0.4.0.

---

## Standards Alignment

The project aims for:
- CF-compliant metadata where feasible
- Explicit mapping between UM STASH codes and diagnostic variables
- Unit consistency checks for all carbon pools and fluxes

Full CMIP compliance is aspirational prior to v1.0.

---

## v0.3.0 Achievements (2026-01-26)

- ✅ Stabilized package structure and imports
- ✅ Configuration-driven I/O with canonical variables
- ✅ 174 tests with CI/CD (Python 3.8-3.11)
- ✅ Public API frozen with stability guarantees
- ✅ 24% test coverage, zero breaking changes from v0.2.x

See [CHANGELOG.md](CHANGELOG.md) for full v0.3.0 release notes.

---

## Guiding Principle

**Scientific correctness > convenience**


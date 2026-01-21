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

## Current State (v0.2.x)

- Core extraction, processing, and plotting logic exists and is scientifically correct.
- Module structure is provisional and undergoing refactor.
- Backward compatibility with existing analysis scripts must be preserved during v0.2.x.

Scientific behaviour must not change unless explicitly documented.

---

## Binding Design Rules (MANDATORY)

1. No filesystem discovery or NetCDF I/O inside plotting functions  
2. NetCDF loading must be isolated in a dedicated I/O layer  
3. Diagnostics must return data objects (xarray.Dataset, dict), never plots  
4. Aggregation logic must not be duplicated across modules  
5. Plotting functions must accept matplotlib Axes objects  
6. No hard-coded paths outside CLI or configuration layers  

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

---

## Standards Alignment

The project aims for:
- CF-compliant metadata where feasible
- Explicit mapping between UM STASH codes and diagnostic variables
- Unit consistency checks for all carbon pools and fluxes

Full CMIP compliance is aspirational prior to v1.0.

---

## v0.2.x → v0.3.0 Intent

- Stabilise package structure and imports
- Introduce configuration-driven I/O
- Add tests and continuous integration
- Freeze public API at v0.3.0

---

## Guiding Principle

**Scientific correctness > convenience**


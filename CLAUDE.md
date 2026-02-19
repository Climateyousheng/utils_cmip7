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

## Current State (v0.4.0)

- Core extraction, processing, and plotting logic exists and is scientifically correct.
- Module structure is **stabilized** with clear public API guarantees.
- ~300 tests with 29% coverage, CI/CD across Python 3.9-3.12.
- Legacy variable names removed; only canonical names accepted.
- Validation comparison module has test coverage (99% compare.py, 100% outputs.py).

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

| Component                    | Stability     |
|------------------------------|---------------|
| Soil carbon diagnostics      | Stable        |
| Carbon flux diagnostics      | Stable        |
| Annual-mean processing       | Provisional   |
| Raw data extraction          | Provisional   |
| Validation comparison        | Provisional   |
| Plotting API                 | Unstable      |
| CLI interface                | Experimental  |

Only *Stable* components may be relied upon in long-lived scripts.

**As of v0.4.0 (2026-02-09)**, the API stability matrix is updated for the v0.4.x series.
No breaking changes will be introduced to "Stable" components until v0.5.0.

---

## Standards Alignment

The project aims for:
- CF-compliant metadata where feasible
- Explicit mapping between UM STASH codes and diagnostic variables
- Unit consistency checks for all carbon pools and fluxes

Full CMIP compliance is aspirational prior to v1.0.

---

## v0.4.0 Achievements (2026-02-09)

- ✅ Removed all deprecated features (var_mapping, var_dict, legacy names)
- ✅ Internal migration to canonical variable names throughout
- ✅ Level selection for multi-dimensional cubes (frac/PFT)
- ✅ Validation module test coverage (compare.py 99%, outputs.py 100%)
- ✅ ~300 tests with CI/CD (Python 3.9-3.12)
- ✅ Dropped Python 3.8 support (EOL Oct 2024)

See [CHANGELOG.md](CHANGELOG.md) for full v0.4.0 release notes.

---

## Bridge Pipeline — Transfer Rules (MANDATORY)

Before triggering any `ftp_master` transfer:

1. Check that the experiment appears in `list_runs` **or** that inidata exist
   under `~/ummodel/data/<expt>/` on silurian.
2. If **neither** condition is met, **do not transfer** — the experiment is
   not set up and the transfer will produce no useful output.

```bash
# Quick pre-transfer check (run on silurian)
list_runs | grep <expt>                          # check 1
ls ~/ummodel/data/<expt>/                        # check 2
```

---

## Guiding Principle

**Scientific correctness > convenience**


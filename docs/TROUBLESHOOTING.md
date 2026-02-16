# utils_cmip7 — Troubleshooting Guide

This document addresses **runtime issues**, **missing diagnostics**, and
**common extraction failures**.

This file is user-facing and **non-normative**.

---

## Variables Missing from Plots

### Symptom
Plots show only a subset of expected variables (e.g. only PFTs).

### Cause
Variables were not found during extraction and were silently skipped.

### Diagnosis
Run extraction with verbose output:

```python
from analysis import extract_annual_means
ds = extract_annual_means(expts_list=['xqhuc'])
Review extraction summary for missing variables.

## Missing Annual Mean Files
Expected files
{expt}_pt_annual_mean.nc
{expt}_pd_annual_mean.nc
{expt}_pf_annual_mean.nc
Fix

Re-run annual-mean preprocessing (e.g. via CDO scripts).

STASH Codes Appear Correct but Extraction Fails
Diagnosis

Check STASH codes directly:

ncdump -h file.nc | grep stash_code

Common issue

Codes appear with s suffix (e.g. 3261s).

Resolution

No action needed. This is normal NetCDF syntax.

Variables Present but Units Incorrect
Cause

Unit conversions were skipped or overridden.

Fix

Verify variable is listed in conversion dictionary.
Custom variables require explicit conversion rules.

Raw Extraction Performance
Status

**✅ Optimized (2025)**: Raw extraction is now **5× faster** thanks to file-level caching.

Explanation

Each monthly file is loaded once and all variables extracted in a single pass, rather than loading each file 5 times (once per variable).

Performance

- 100-year simulation: ~6 minutes (previously ~30 minutes)
- File loads reduced from 6,000 to 1,200 for typical workflows

Recommendation

Raw extraction is now efficient enough for routine use. Pre-processed files are still faster for repeated analyses, but raw extraction is viable when annual means haven't been generated yet.

Regional Results Look Incorrect
Possible causes

Mask resolution mismatch

Wrong RECCAP region name

Missing land/sea mask alignment

Debug steps

Verify mask file exists

Confirm region spelling

Check grid compatibility

Debugging Tips

Always inspect extraction summaries

Verify NetCDF metadata early

Test global before regional diagnostics

Use small time slices for debugging

When to Escalate

If issues persist:

Verify UM output integrity

Confirm STASH mappings

Compare with known-good experiment

# API Stabilization Plan for v0.3.0

**Date:** 2026-01-26
**Current Version:** v0.3.0.dev0
**Target Version:** v0.3.0 (stable)

---

## Goals

1. **Freeze public API** - Establish stable interfaces for v0.3.0
2. **Remove deprecations** - Clean up deprecated code paths
3. **Document public API** - Clear `__all__` exports in all modules
4. **Version consistency** - Align version strings across package
5. **API documentation** - Document what's public vs internal

---

## Current API Issues

### 1. Version String Mismatch ❌

**Problem:**
- `pyproject.toml`: `version = "0.3.0.dev0"`
- `src/utils_cmip7/__init__.py`: `__version__ = "0.2.1"`

**Action:** Update `__init__.py` to match `pyproject.toml`

### 2. Deprecation Warnings Still Active ⚠️

**In `diagnostics/extraction.py`:**
```python
# Line 169-175: var_mapping parameter deprecated
if var_mapping is not None:
    warnings.warn(
        "The 'var_mapping' parameter is deprecated and will be removed in v0.3.0.",
        DeprecationWarning,
        stacklevel=2
    )

# Lines 186-193: Legacy variable names deprecated
if var != canonical:
    warnings.warn(
        f"Variable name '{var}' is deprecated. Use canonical name '{canonical}'.",
        DeprecationWarning,
        stacklevel=2
    )
```

**Action:** Since we're releasing v0.3.0, these should:
- Option A: Be removed entirely (breaking change)
- Option B: Change warning to say "will be removed in v0.4.0" (grace period)
- **Recommendation:** Option B - keep backward compatibility one more version

### 3. Missing `__all__` Exports

**Modules without `__all__`:**
- `io/stash.py`
- `io/file_discovery.py`
- `io/extract.py`
- `processing/spatial.py`
- `processing/temporal.py`
- `processing/regional.py`
- `diagnostics/extraction.py`
- `diagnostics/raw.py`
- `validation/` (all modules)

**Action:** Add `__all__` to all public modules

### 4. Internal vs Public Functions Unclear

**Example issues:**
- `compute_latlon_box_mean()` in extraction.py - is this public?
- Helper functions in processing modules - should they be exported?
- Validation module functions - which are user-facing?

**Action:** Review each module and mark internal functions with leading underscore

---

## API Stability Matrix (Updated for v0.3.0)

| Component | Status in v0.3.0 | Breaking Changes Allowed? |
|-----------|------------------|---------------------------|
| Canonical variable registry (`config.CANONICAL_VARIABLES`) | **Stable** | ❌ No |
| Main extraction API (`extract_annual_means`) | **Stable** | ❌ No |
| Processing functions (`global_total_pgC`, etc.) | **Stable** | ❌ No |
| Configuration API (`get_variable_config`, etc.) | **Stable** | ❌ No |
| STASH mapping (`stash`, `stash_nc`) | **Stable** | ❌ No |
| Temporal aggregation | **Stable** | ❌ No |
| Regional aggregation | **Provisional** | ⚠️ Minor only |
| Raw data extraction (`extract_annual_mean_raw`) | **Provisional** | ⚠️ Minor only |
| Validation API | **Unstable** | ✅ Yes |
| Plotting API | **Unstable** | ✅ Yes |
| CLI interface | **Experimental** | ✅ Yes |
| Soil params module | **Experimental** | ✅ Yes |

---

## Proposed Changes

### Phase 1: Version Alignment

**File:** `src/utils_cmip7/__init__.py`
```python
# Line 32: Update version
__version__ = "0.3.0.dev0"  # Match pyproject.toml
```

**File:** Import from pyproject.toml dynamically
```python
# Better approach: Single source of truth
from importlib.metadata import version
__version__ = version("utils_cmip7")
```

### Phase 2: Deprecation Updates

**File:** `diagnostics/extraction.py`

**Change deprecation messages:**
```python
# Update: v0.3.0 → v0.4.0
warnings.warn(
    "The 'var_mapping' parameter is deprecated and will be removed in v0.4.0.",
    DeprecationWarning,
    stacklevel=2
)

warnings.warn(
    f"Variable name '{var}' is deprecated. "
    f"Use canonical name '{canonical}' instead. "
    f"Aliases will be removed in v0.4.0.",
    DeprecationWarning,
    stacklevel=2
)
```

**Rationale:** One more version for users to migrate

### Phase 3: Add `__all__` Exports

**Example:** `io/stash.py`
```python
__all__ = [
    'stash',
    'stash_nc',
]
```

**Example:** `processing/temporal.py`
```python
__all__ = [
    'merge_monthly_results',
    'compute_monthly_mean',
    'compute_annual_mean',
]
```

**Example:** `diagnostics/extraction.py`
```python
__all__ = [
    'extract_annual_means',
    'compute_latlon_box_mean',  # Public helper
]
```

### Phase 4: Mark Internal Functions

**Pattern:**
```python
# Public function - exported in __all__
def extract_annual_means(...):
    pass

# Internal helper - not exported
def _build_cube_map(cubes, var_list):
    pass
```

### Phase 5: Update API Documentation

**File:** `docs/API.md` (new file)
```markdown
# utils_cmip7 Public API Reference

## Stable API (v0.3.0)

### Extraction
- `extract_annual_means()` - Main entry point
- `extract_annual_mean_raw()` - Raw data extraction

### Processing
- `global_total_pgC()` - Global spatial total
- `global_mean_pgC()` - Global spatial mean
- `compute_regional_annual_mean()` - Regional processing

... (continue for all public functions)
```

---

## Implementation Steps

### Step 1: Version Sync ✅
- [ ] Update `__init__.py` __version__ to "0.3.0.dev0"
- [ ] OR use importlib.metadata for single source of truth

### Step 2: Deprecation Grace Period ✅
- [ ] Update deprecation messages: v0.3.0 → v0.4.0
- [ ] Document migration path in CHANGELOG

### Step 3: Public API Definition ✅
- [ ] Add `__all__` to all public modules
- [ ] Rename internal functions with `_` prefix
- [ ] Update imports in `__init__.py` if needed

### Step 4: API Documentation ✅
- [ ] Create `docs/API.md` with public API reference
- [ ] Update README.md with API stability guarantees
- [ ] Add examples for each stable API function

### Step 5: Prepare v0.3.0 Release ✅
- [ ] Update CHANGELOG.md with v0.3.0 changes
- [ ] Create MIGRATION.md for v0.2.x → v0.3.0 migration
- [ ] Update version to "0.3.0" (remove .dev0)
- [ ] Tag release: `git tag v0.3.0`

---

## Breaking Changes from v0.2.x → v0.3.0

### Removed

None (all deprecated features still work with warnings)

### Deprecated (with grace period until v0.4.0)

1. **Legacy variable names** (use canonical names instead):
   - `VegCarb` → `CVeg`
   - `soilResp` → `Rh`
   - `soilCarbon` → `CSoil`
   - `temp` → `tas`
   - `precip` → `pr`

2. **`var_mapping` parameter** in `extract_annual_means()`:
   - No longer needed (automatic lookup from canonical registry)

### Changed

1. **Package structure** - submodules reorganized
2. **Import paths** - use top-level imports from `utils_cmip7`
3. **Configuration** - `config.py` now has canonical variable registry

### Added

1. **Canonical variable registry** - `CANONICAL_VARIABLES` in `config.py`
2. **Helper functions** - `resolve_variable_name()`, `get_variable_config()`, etc.
3. **Comprehensive testing** - 174 tests with 23% coverage

---

## API Stability Guarantees

### For v0.3.x Series

**Stable API (No breaking changes):**
- Main extraction functions
- Core processing functions
- Configuration API
- STASH mapping

**Provisional API (Minor changes allowed):**
- Regional aggregation
- Raw data extraction

**Unstable API (Breaking changes possible):**
- Validation module
- Plotting module
- CLI commands

**Experimental (No guarantees):**
- Soil parameters module

### For v1.0.0 (Future)

All currently "Stable" and "Provisional" APIs will be frozen.

---

## Post-Stabilization Checklist

- [ ] All modules have `__all__` exports
- [ ] Version strings consistent across package
- [ ] Deprecation warnings updated to v0.4.0
- [ ] API documentation complete (`docs/API.md`)
- [ ] Migration guide written (`docs/MIGRATION_v0.3.md`)
- [ ] CHANGELOG.md updated
- [ ] README.md updated with stability matrix
- [ ] Tests passing on Python 3.8-3.11
- [ ] CI green on all platforms

---

## Timeline

**Estimated effort:** 2-3 hours

1. Phase 1 (Version sync): 15 min
2. Phase 2 (Deprecations): 15 min
3. Phase 3 (`__all__` exports): 60 min
4. Phase 4 (Internal marking): 30 min
5. Phase 5 (Documentation): 45 min

**Total:** ~3 hours

---

Last updated: 2026-01-26

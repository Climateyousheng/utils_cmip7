# Variable Naming and Configuration Refactoring Summary

## What Was Done

Successfully refactored the variable naming system in `utils_cmip7` to use a centralized canonical variable registry with CMIP-style naming conventions.

## Key Changes

### 1. Created Canonical Variable Registry (config.py)

Added `CANONICAL_VARIABLES` dictionary as the single source of truth for all variable metadata:

```python
CANONICAL_VARIABLES = {
    "CVeg": {
        "description": "Vegetation Carbon Content",
        "stash_name": "cv",                # For UM data extraction
        "stash_code": "m01s19i002",
        "aggregation": "SUM",              # Spatial aggregation method
        "conversion_factor": 1e-12,        # kgC/m2 → PgC
        "units": "PgC",
        "category": "stock",
        "aliases": ["VegCarb", "vegetation_carbon_content"],  # Backward compat
    },
    # ... 8 more variables
}
```

**Key features:**
- Encodes STASH name, conversion factor, aggregation method, and units
- Preserves the "Others" → MEAN aggregation semantic
- Supports aliases for backward compatibility
- Handles frac/fracb fallback automatically

### 2. New Helper Functions (config.py)

**`resolve_variable_name(name: str) -> str`**
- Resolves any name (canonical or alias) to canonical name
- Example: `'VegCarb'` → `'CVeg'`, `'soilResp'` → `'Rh'`

**`get_variable_config(name: str) -> dict`**
- Returns full variable configuration (resolves aliases)
- Example: `get_variable_config('VegCarb')['stash_name']` → `'cv'`

**`get_conversion_key(name: str) -> str`**
- Returns the key for `compute_regional_annual_mean()`
- Preserves backward compatibility with old var_mapping system
- Returns "Others" for MEAN aggregation, canonical name for SUM

### 3. Updated DEFAULT_VAR_LIST (config.py)

Changed from mixed naming to **canonical CMIP-style names**:

**Before:**
```python
['soilResp', 'soilCarbon', 'VegCarb', 'fracPFTs', 'GPP', 'NPP', 'fgco2', 'temp', 'precip']
```

**After:**
```python
['Rh', 'CSoil', 'CVeg', 'frac', 'GPP', 'NPP', 'fgco2', 'tas', 'pr']
```

### 4. Refactored extract_annual_means() (diagnostics/extraction.py)

**Changes:**
- ✅ Removed hardcoded `default_var_list` and `default_var_mapping`
- ✅ Added deprecation warning for `var_mapping` parameter
- ✅ Added automatic alias resolution with deprecation warnings
- ✅ Dynamic variable extraction using registry
- ✅ Automatic frac→fracb fallback handling
- ✅ Automatic conversion key lookup from registry
- ✅ Updated derived variables to use canonical names (NEP, Land Carbon, Trees Total)

**Example of improved extraction loop:**

**Before (hardcoded):**
```python
sr = try_extract(cubes, 'rh', stash_lookup_func=stash)
# Repeat 9 times for each variable...
```

**After (dynamic):**
```python
for var_name in var_list:
    var_config = get_variable_config(var_name)
    extracted = try_extract(cubes, var_config['stash_name'], stash_lookup_func=stash)
    if not extracted and var_config.get('stash_fallback'):
        extracted = try_extract(cubes, var_config['stash_fallback'])
    cube_map[var_name] = extracted
```

### 5. Updated VAR_CONVERSIONS (config.py)

Added entries for canonical variable names:

```python
VAR_CONVERSIONS = {
    # Legacy names (backward compatibility)
    'S resp': 3600*24*360*(1e-12),
    'V carb': (1e-12),
    # ... more legacy entries ...

    # NEW: Canonical CMIP-style names
    'Rh': 3600*24*360*(1e-12),
    'CVeg': (1e-12),
    'CSoil': (1e-12),
    'tas': 1,  # MEAN aggregation
    'pr': 86400,
    # ... more canonical entries
}
```

### 6. Updated Package Exports (__init__.py)

Added new functions to public API:

```python
from .config import (
    CANONICAL_VARIABLES,
    DEFAULT_VAR_LIST,
    resolve_variable_name,
    get_variable_config,
    get_conversion_key,
)
```

## Canonical Variable Names (New Standard)

| Old Name | New Canonical | STASH | Aggregation | Notes |
|----------|---------------|-------|-------------|-------|
| `soilResp` | **`Rh`** | `rh` | SUM | Heterotrophic respiration (CMIP standard) |
| `soilCarbon` | **`CSoil`** | `cs` | SUM | Soil carbon (CMIP standard) |
| `VegCarb` | **`CVeg`** | `cv` | SUM | Vegetation carbon (CMIP standard) |
| `fracPFTs` | **`frac`** | `frac` | MEAN | PFT fractions (with fracb fallback) |
| `temp` | **`tas`** | `tas` | MEAN | Near-surface air temperature |
| `precip` | **`pr`** | `pr` | MEAN | Precipitation |
| `GPP` | **`GPP`** | `gpp` | SUM | No change (already canonical) |
| `NPP` | **`NPP`** | `npp` | SUM | No change (already canonical) |
| `fgco2` | **`fgco2`** | `fgco2` | SUM | No change (already canonical) |

## Backward Compatibility

✅ **Old aliases still work** with deprecation warnings:

```python
>>> ds = extract_annual_means(['xqhuc'], var_list=['VegCarb', 'soilResp'])
DeprecationWarning: Variable name 'VegCarb' is deprecated.
Use canonical name 'CVeg' instead. Aliases will be removed in v0.3.0.
DeprecationWarning: Variable name 'soilResp' is deprecated.
Use canonical name 'Rh' instead. Aliases will be removed in v0.3.0.
```

✅ **var_mapping parameter still accepted** (but ignored with warning):

```python
>>> ds = extract_annual_means(['xqhuc'], var_mapping=['S resp', ...])
DeprecationWarning: The 'var_mapping' parameter is deprecated and will be
removed in v0.3.0. Conversion keys are now automatically looked up.
```

## Critical Feature Preserved: MEAN vs SUM Aggregation

The "Others" semantic (MEAN aggregation) is **fully preserved**:

- Variables with `aggregation: "MEAN"` → conversion key = "Others" (or "precip")
- Variables with `aggregation: "SUM"` → conversion key = canonical name
- `compute_regional_annual_mean()` checks: if key in ("Others", "precip") use MEAN, else use SUM

**Example:**
```python
>>> get_conversion_key('tas')  # MEAN aggregation
'Others'

>>> get_conversion_key('CVeg')  # SUM aggregation
'CVeg'

>>> get_conversion_key('pr')  # MEAN aggregation (special case)
'precip'
```

## Frac/Fracb Fallback Handling

Automatic fallback implemented:

```python
CANONICAL_VARIABLES = {
    "frac": {
        "stash_name": "frac",
        "stash_code": "m01s19i013",
        "stash_fallback": 19017,  # Try fracb if frac not found
        # ...
    }
}
```

**Behavior:**
1. Try to extract `frac` (m01s19i013)
2. If not found, try stash code 19017 (fracb)
3. Report which one was found in diagnostic output
4. Store result under canonical name `'frac'` regardless of source

## Migration Path for Users

### Immediate (v0.2.2)

**Option 1: Use new canonical names (recommended)**
```python
ds = extract_annual_means(['xqhuc'], var_list=['Rh', 'CVeg', 'CSoil', 'GPP'])
```

**Option 2: Continue using aliases (deprecated)**
```python
ds = extract_annual_means(['xqhuc'], var_list=['soilResp', 'VegCarb', 'soilCarbon', 'GPP'])
# Works but shows deprecation warnings
```

**Option 3: Use defaults (now canonical)**
```python
ds = extract_annual_means(['xqhuc'])  # Uses DEFAULT_VAR_LIST with canonical names
```

### Future (v0.3.0 - Breaking Changes)

- Aliases will be removed
- `var_mapping` parameter will be removed
- Only canonical names will be accepted

## Testing

Created comprehensive test suite (`test_canonical_variables.py`):

```
✓ PASS: test_canonical_variables_registry
✓ PASS: test_resolve_variable_name
✓ PASS: test_get_variable_config
✓ PASS: test_get_conversion_key
✓ PASS: test_default_var_list
✓ PASS: test_aggregation_semantics

Result: 6/6 tests passed
```

## Benefits

1. **Single source of truth** - No more separate var_list + var_mapping
2. **Self-documenting** - CMIP names are scientifically standard
3. **Easier maintenance** - Add new variables in one place
4. **Better error messages** - Clear validation and suggestions
5. **CMIP alignment** - Natural fit with observational datasets
6. **Flexible extraction** - Dynamic lookup instead of hardcoded logic
7. **Type safety** - Clear structure with required fields
8. **Extensible** - Easy to add new variables or metadata fields

## Files Modified

1. `src/utils_cmip7/config.py` - Added CANONICAL_VARIABLES, helper functions
2. `src/utils_cmip7/diagnostics/extraction.py` - Refactored to use registry
3. `src/utils_cmip7/__init__.py` - Exported new functions
4. `docs/NAMING_ANALYSIS.md` - Detailed analysis document
5. `test_canonical_variables.py` - Test suite

## Next Steps (Optional)

1. Update all example notebooks to use canonical names
2. Update plotting functions to use canonical names internally
3. Add migration guide to documentation
4. Update CHANGELOG.md for v0.2.2 release
5. Plan v0.3.0 breaking changes timeline

## Questions Answered

1. **Use Rh?** ✅ Yes, using CMIP standard `Rh` for heterotrophic respiration
2. **Handle frac/fracb?** ✅ Implemented automatic fallback in registry
3. **Timing for breaking changes?** ✅ Can do now or in v0.3.0
4. **Preserve "Others" semantic?** ✅ Fully preserved through aggregation field

## Example Usage

### Old Way (Still Works)
```python
from utils_cmip7 import extract_annual_means

ds = extract_annual_means(
    ['xqhuc'],
    var_list=['soilResp', 'VegCarb', 'GPP'],
    var_mapping=['S resp', 'V carb', 'GPP']  # No longer needed!
)
```

### New Way (Recommended)
```python
from utils_cmip7 import extract_annual_means

# Simple - uses defaults
ds = extract_annual_means(['xqhuc'])

# Explicit - canonical names
ds = extract_annual_means(['xqhuc'], var_list=['Rh', 'CVeg', 'GPP'])

# Access results with canonical names
print(ds['xqhuc']['global']['Rh']['data'])  # Heterotrophic respiration
print(ds['xqhuc']['global']['CVeg']['data'])  # Vegetation carbon
```

### Advanced - Using Registry Directly
```python
from utils_cmip7 import get_variable_config, CANONICAL_VARIABLES

# Get variable metadata
config = get_variable_config('CVeg')
print(config['description'])  # "Vegetation Carbon Content"
print(config['units'])  # "PgC"
print(config['aggregation'])  # "SUM"

# List all available variables
print(list(CANONICAL_VARIABLES.keys()))
# ['GPP', 'NPP', 'Rh', 'fgco2', 'CVeg', 'CSoil', 'tas', 'pr', 'frac']
```

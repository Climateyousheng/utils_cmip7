# Variable Naming Analysis and Recommendations

## Current Naming Mess

### The Problem: 4 Different Naming Conventions

For the same physical variable, we have up to 4 different names:

| Physical Variable | STASH Name | User Variable | CMIP Name | Conversion Key |
|------------------|------------|---------------|-----------|----------------|
| Vegetation Carbon | `cv` | `VegCarb` | `CVeg` | `V carb` |
| Soil Carbon | `cs` | `soilCarbon` | `CSoil` | `S carb` |
| Soil Respiration | `rh` | `soilResp` | `soilResp` | `S resp` |
| Gross Primary Production | `gpp` | `GPP` | `GPP` | `GPP` |
| Net Primary Production | `npp` | `NPP` | `NPP` | `NPP` |
| Surface Temperature | `tas` | `temp` | `tas` | `Others` |
| Precipitation | `pr` | `precip` | `precip` | `precip` |
| Ocean CO2 Flux | `fgco2` | `fgco2` | `fgco2` | `field646_mm_dpth` |
| PFT Fractions | `frac` | `fracPFTs` | N/A | `Others` |

### Specific Issues

1. **Inconsistent Casing**
   - `gpp` (STASH) → `GPP` (user) → `GPP` (config)
   - Makes code-reading difficult

2. **Obscure STASH Names**
   - `rh` = heterotrophic respiration (not intuitive)
   - `cv` = vegetation carbon (cryptic)
   - `cs` = soil carbon (cryptic)
   - Users shouldn't need to know UM internal codes

3. **Multiple Aliases Per Variable**
   - Vegetation: `cv`, `VegCarb`, `CVeg`, `vegetation_carbon_content`
   - Soil: `cs`, `soilCarbon`, `CSoil`
   - Temperature: `tas`, `temp`
   - Creates confusion about which to use when

4. **Cryptic Conversion Keys**
   - `'field646_mm_dpth'` - what is this?
   - `'S resp'`, `'V carb'`, `'S carb'` - arbitrary abbreviations
   - `'Others'` - meaningless catch-all
   - Not self-documenting

5. **No Clear Canonical Name**
   - Which is "correct"? `VegCarb` or `CVeg`?
   - Code uses both interchangeably
   - No clear hierarchy or preference

## Impact on Users

### Current User Pain Points

```python
# User sees this in extraction:
ds = extract_annual_means(['xqhuc'])
ds['xqhuc']['global']['VegCarb']  # Which name should I use?

# But CMIP comparison uses:
cmip_data = load_cmip6_metrics()
cmip_data['CVeg']  # Different name!

# And internally we use:
sr = try_extract(cubes, 'rh', stash_lookup_func=stash)  # What's 'rh'?

# With conversion keys like:
var_mapping = ['S resp', 'S carb', 'V carb']  # Not intuitive
```

### Code Maintenance Issues

1. Developers must remember 4 different naming schemes
2. Easy to mix up names when adding new features
3. Hard to validate whether variable names are correct
4. Documentation becomes cluttered with aliases

## Recommended Solution

### Principle: **CMIP-Style Names as Canonical**

Use CMIP6/CMIP7 variable names as the **single source of truth** throughout the public API:

- `CVeg` (not `VegCarb` or `cv`)
- `CSoil` (not `soilCarbon` or `cs`)
- `GPP` (not `gpp`)
- `NPP` (not `npp`)
- `Rh` (not `soilResp` or `rh`) - Standard CMIP abbreviation for heterotrophic respiration
- `tas` (not `temp`)
- `pr` (not `precip`)
- `fgco2` (keep as-is, already CMIP standard)

### Rationale

1. **CMIP names are the scientific standard** - widely recognized
2. **Aligns with observational data** - CMIP6/RECCAP2 use these names
3. **Clear documentation** - CMIP variable tables are authoritative
4. **Reduces cognitive load** - one name to remember, not four
5. **Future-proof** - as project aims for CMIP compliance

### Implementation Strategy

#### Phase 1: Add Canonical Mappings (v0.2.2 - Non-Breaking)

**Create canonical variable registry:**

```python
# config.py - NEW SECTION

"""
Canonical Variable Registry
---------------------------
Single source of truth for all variable names and their mappings.
"""

CANONICAL_VARIABLES = {
    # Physical variable identifier (CMIP-style)
    "CVeg": {
        "description": "Vegetation Carbon Content",
        "stash_name": "cv",                      # For data extraction
        "stash_code": "m01s19i002",
        "aliases": ["VegCarb", "vegetation_carbon_content"],  # For backward compat
        "units": "PgC",
        "category": "stock",
        "conversion_factor": 1e-12,  # kgC/m2 → PgC
    },
    "CSoil": {
        "description": "Soil Carbon Content",
        "stash_name": "cs",
        "stash_code": "m01s19i016",
        "aliases": ["soilCarbon"],
        "units": "PgC",
        "category": "stock",
        "conversion_factor": 1e-12,
    },
    "Rh": {
        "description": "Heterotrophic Respiration",
        "stash_name": "rh",
        "stash_code": "m01s03i293",
        "aliases": ["soilResp"],
        "units": "PgC/yr",
        "category": "flux",
        "conversion_factor": 3600*24*360*1e-12,  # kgC/m2/s → PgC/yr
    },
    "GPP": {
        "description": "Gross Primary Production",
        "stash_name": "gpp",
        "stash_code": "m01s03i261",
        "aliases": [],
        "units": "PgC/yr",
        "category": "flux",
        "conversion_factor": 3600*24*360*1e-12,
    },
    "NPP": {
        "description": "Net Primary Production",
        "stash_name": "npp",
        "stash_code": "m01s03i262",
        "aliases": [],
        "units": "PgC/yr",
        "category": "flux",
        "conversion_factor": 3600*24*360*1e-12,
    },
    "tas": {
        "description": "Near-Surface Air Temperature",
        "stash_name": "tas",
        "stash_code": "m01s03i236",
        "aliases": ["temp"],
        "units": "°C",
        "category": "climate",
        "conversion_factor": 1.0,
    },
    "pr": {
        "description": "Precipitation",
        "stash_name": "pr",
        "stash_code": "m01s05i216",
        "aliases": ["precip"],
        "units": "mm/day",
        "category": "climate",
        "conversion_factor": 86400,  # kg/m2/s → mm/day
    },
    "fgco2": {
        "description": "Surface Downward Mass Flux of Carbon as CO2",
        "stash_name": "fgco2",
        "stash_code": "m02s30i249",
        "aliases": [],
        "units": "PgC/yr",
        "category": "flux",
        "conversion_factor": 12/1000*1e-12,  # molC/m2/yr → PgC/yr
    },
    "frac": {
        "description": "Plant Functional Type Grid Fractions",
        "stash_name": "frac",
        "stash_code": "m01s19i013",
        "aliases": ["fracPFTs"],
        "units": "1",
        "category": "land_use",
        "conversion_factor": 1.0,
    },
}

def resolve_variable_name(name: str) -> str:
    """
    Resolve any variable name (canonical or alias) to canonical name.

    Parameters
    ----------
    name : str
        Variable name (canonical or alias)

    Returns
    -------
    str
        Canonical variable name

    Raises
    ------
    ValueError
        If variable name not recognized

    Examples
    --------
    >>> resolve_variable_name('VegCarb')  # alias
    'CVeg'
    >>> resolve_variable_name('CVeg')  # already canonical
    'CVeg'
    >>> resolve_variable_name('temp')  # alias
    'tas'
    """
    # Already canonical?
    if name in CANONICAL_VARIABLES:
        return name

    # Search aliases
    for canonical_name, config in CANONICAL_VARIABLES.items():
        if name in config.get("aliases", []):
            return canonical_name

    raise ValueError(
        f"Unknown variable name: '{name}'. "
        f"Known variables: {list(CANONICAL_VARIABLES.keys())}"
    )

def get_variable_config(name: str) -> dict:
    """
    Get full configuration for a variable (resolves aliases).

    Parameters
    ----------
    name : str
        Variable name (canonical or alias)

    Returns
    -------
    dict
        Variable configuration

    Examples
    --------
    >>> cfg = get_variable_config('VegCarb')
    >>> cfg['description']
    'Vegetation Carbon Content'
    >>> cfg['conversion_factor']
    1e-12
    """
    canonical = resolve_variable_name(name)
    return CANONICAL_VARIABLES[canonical]
```

#### Phase 2: Update Internal Code (v0.2.2 - Non-Breaking)

**Deprecation warnings when aliases used:**

```python
# diagnostics/extraction.py

def extract_annual_means(expts_list, var_list=None, var_mapping=None, regions=None):
    """
    [Updated docstring]

    Parameters
    ----------
    var_list : list of str, optional
        Variable names (CMIP-style canonical names preferred).
        Aliases supported but deprecated:
        - Use 'CVeg' not 'VegCarb'
        - Use 'CSoil' not 'soilCarbon'
        - Use 'Rh' not 'soilResp'
        - Use 'tas' not 'temp'
        - Use 'pr' not 'precip'
    """
    from ..config import DEFAULT_VAR_LIST, resolve_variable_name
    import warnings

    if var_list is None:
        var_list = DEFAULT_VAR_LIST

    # Resolve aliases with deprecation warnings
    resolved_vars = []
    for var in var_list:
        canonical = resolve_variable_name(var)
        if var != canonical:
            warnings.warn(
                f"Variable name '{var}' is deprecated. "
                f"Use canonical name '{canonical}' instead. "
                f"Aliases will be removed in v0.3.0.",
                DeprecationWarning,
                stacklevel=2
            )
        resolved_vars.append(canonical)

    # Rest of function uses canonical names...
```

#### Phase 3: Update Config Defaults (v0.2.2)

```python
# config.py - Updated

DEFAULT_VAR_LIST = [
    'Rh',          # Heterotrophic respiration (was: soilResp)
    'CSoil',       # Soil carbon (was: soilCarbon)
    'CVeg',        # Vegetation carbon (was: VegCarb)
    'frac',        # PFT fractions (was: fracPFTs)
    'GPP',         # Gross primary production
    'NPP',         # Net primary production
    'fgco2',       # Ocean CO2 flux
    'tas',         # Surface air temperature (was: temp)
    'pr'           # Precipitation (was: precip)
]

# var_mapping becomes obsolete - conversion factors come from CANONICAL_VARIABLES
```

#### Phase 4: Remove Aliases (v0.3.0 - Breaking)

- Remove `aliases` field from `CANONICAL_VARIABLES`
- Remove deprecation warnings (now errors)
- Update all examples and documentation

### Benefits

1. **One name per variable** - no more confusion
2. **Self-documenting code** - CMIP names are well-defined
3. **Simplified config** - no more separate var_mapping list
4. **Better validation** - clear error messages for invalid names
5. **CMIP alignment** - natural fit with obs data
6. **Easier maintenance** - single registry to update

## Migration Path for Users

### Step 1: Add deprecation warnings (v0.2.2)

Users get warnings but code still works:

```
DeprecationWarning: Variable name 'VegCarb' is deprecated.
Use canonical name 'CVeg' instead. Aliases will be removed in v0.3.0.
```

### Step 2: Documentation update (v0.2.2)

Add migration guide to docs:

```markdown
## Variable Name Migration Guide

utils_cmip7 is adopting CMIP-style canonical variable names for consistency
with observational datasets and scientific standards.

### Name Changes (v0.3.0)

| Old Name | New Name | Notes |
|----------|----------|-------|
| `VegCarb` | `CVeg` | CMIP6 standard name |
| `soilCarbon` | `CSoil` | CMIP6 standard name |
| `soilResp` | `Rh` | CMIP6 standard for heterotrophic respiration |
| `temp` | `tas` | CMIP6 standard for near-surface air temperature |
| `precip` | `pr` | CMIP6 standard for precipitation |
| `fracPFTs` | `frac` | Simplified, matches STASH name |

### Update Your Code

**Before (v0.2.x, deprecated):**
```python
ds = extract_annual_means(['xqhuc'], var_list=['VegCarb', 'soilCarbon', 'GPP'])
```

**After (v0.3.0+, required):**
```python
ds = extract_annual_means(['xqhuc'], var_list=['CVeg', 'CSoil', 'GPP'])
```
```

### Step 3: Hard cutover (v0.3.0)

Aliases raise clear errors with migration hints:

```python
ValueError: Unknown variable name: 'VegCarb'.
Did you mean 'CVeg'?
See migration guide: https://utils-cmip7.readthedocs.io/migration-guide
```

## Implementation Checklist

- [ ] Add `CANONICAL_VARIABLES` registry to `config.py`
- [ ] Add `resolve_variable_name()` helper
- [ ] Add `get_variable_config()` helper
- [ ] Update `extract_annual_means()` to use registry
- [ ] Add deprecation warnings for aliases
- [ ] Update all docstrings with new canonical names
- [ ] Update `DEFAULT_VAR_LIST` in config
- [ ] Deprecate `DEFAULT_VAR_MAPPING` (replaced by registry)
- [ ] Update plotting functions to use canonical names
- [ ] Update metrics.py to use canonical names consistently
- [ ] Add migration guide to documentation
- [ ] Add tests for name resolution and backward compat
- [ ] Update all examples/notebooks
- [ ] Create v0.2.2 release with deprecations
- [ ] Plan v0.3.0 breaking changes

## Example: Before and After

### Before (Current Mess)

```python
# config.py
DEFAULT_VAR_LIST = ['soilResp', 'soilCarbon', 'VegCarb', ...]
DEFAULT_VAR_MAPPING = ['S resp', 'S carb', 'V carb', ...]

# extraction.py
sr = try_extract(cubes, 'rh', stash_lookup_func=stash)  # What's rh?
dict_annual_means[expt][region]['soilResp'] = ...      # Different name!

# User code
data = ds['xqhuc']['global']['VegCarb']  # VegCarb or CVeg?

# CMIP comparison
cmip_data = load_cmip6_metrics()
cmip_data['CVeg']  # Different name again!
```

### After (Clean)

```python
# config.py
CANONICAL_VARIABLES = {
    "CVeg": {
        "stash_name": "cv",
        "conversion_factor": 1e-12,
        ...
    },
    ...
}

DEFAULT_VAR_LIST = ['Rh', 'CSoil', 'CVeg', ...]  # CMIP names

# extraction.py
var_config = get_variable_config('CVeg')
cube = try_extract(cubes, var_config['stash_name'])  # cv
dict_annual_means[expt][region]['CVeg'] = ...        # Consistent!

# User code
data = ds['xqhuc']['global']['CVeg']  # Always CVeg

# CMIP comparison
cmip_data = load_cmip6_metrics()
cmip_data['CVeg']  # Same name!
```

## Questions for Discussion

1. **Should we use `Rh` or keep `soilResp`?**
   - `Rh` is CMIP standard but less intuitive
   - `soilResp` is clearer but non-standard
   - Recommendation: Use `Rh`, add clear docs

2. **What about fracPFTs?**
   - CMIP doesn't have direct equivalent
   - Simplify to `frac` to match STASH?
   - Or keep `fracPFTs` as descriptive?
   - Recommendation: Use `frac`, it's shorter and matches UM

3. **Timing for v0.3.0 breaking changes?**
   - Need time for users to migrate
   - Suggest: 3-6 months after v0.2.2 release
   - Can be accelerated if no external users yet

"""
Configuration module for utils_cmip7.

Contains unit conversion factors, file paths, and regional definitions.
"""

import os

# Unit conversion factors for various variables
# Maps variable names/codes to conversion factors
VAR_CONVERSIONS = {
    # Legacy names (for backward compatibility)
    'Ocean flux': (12/44)*3600*24*360*(1e-12),         # kgCO2/m2/s to PgC/yr
    'm01s00i250': (12/44)*3600*24*360*(1e-12),         # same as Ocean flux
    'field1560_mm_srf': (12/44)*3600*24*360*(1e-12),   # same as Ocean flux
    'P resp': 3600*24*360*(1e-12),                     # from kgC/m2/s to PgC/yr
    'S resp': 3600*24*360*(1e-12),                     # from kgC/m2/s to PgC/yr
    'litter flux': (1e-12),                            # from kgC/m2/yr to PgC/yr
    'V carb': (1e-12),                                 # from kgC/m2 to PgC
    'vegetation_carbon_content': (1e-12),              # from kgC/m2 to PgC
    'S carb': (1e-12),                                 # from kgC/m2 to PgC
    'soilCarbon': (1e-12),                             # from kgC/m2 to PgC
    'Air flux': (12)/1000*(1e-12),                     # from molC/m2/yr to PgC/yr
    'm02s30i249': (12)/1000*(1e-12),                   # same as Air flux
    'field646_mm_dpth': (12)/1000*(1e-12),             # same as Air flux
    'Total co2': 28.97/44.01*(1e6),                    # from mmr to ppmv
    'm01s00i252': 28.97/44.01*(1e6),                   # same as Total co2
    'precip': 86400,                                   # from kg/m2/s to mm/day
    'Others': 1,                                       # no conversion (used for MEAN aggregation)

    # Canonical CMIP-style names (new standard)
    'GPP': 3600*24*360*(1e-12),                        # kgC/m2/s → PgC/yr
    'NPP': 3600*24*360*(1e-12),                        # kgC/m2/s → PgC/yr
    'Rh': 3600*24*360*(1e-12),                         # kgC/m2/s → PgC/yr
    'CVeg': (1e-12),                                   # kgC/m2 → PgC
    'CSoil': (1e-12),                                  # kgC/m2 → PgC
    'fgco2': (12)/1000*(1e-12),                        # molC/m2/yr → PgC/yr
    'tas': 1,                                          # K (no conversion, use MEAN)
    'pr': 86400,                                       # kg/m2/s → mm/day
    'frac': 1,                                         # fraction (no conversion, use MEAN)
    'co2': 28.97/44.01*1e6,                            # mmr → ppmv (MEAN aggregation)
    'Total co2': 28.97/44.01*1e6,                      # mmr → ppmv (legacy key name)
}

# RECCAP2 regional mask file path
# Can be overridden by setting UTILS_CMIP7_RECCAP_MASK environment variable
RECCAP_MASK_PATH = os.environ.get(
    'UTILS_CMIP7_RECCAP_MASK',
    os.path.expanduser(
        '~/scripts/hadcm3b-ensemble-validator/observations/'
        'RECCAP_AfricaSplit_MASK11_Mask_regridded.hadcm3bl_grid.nc'
    )
)

# RECCAP2 region definitions
RECCAP_REGIONS = {
    1: "North_America",
    2: "South_America",
    3: "Europe",
    4: "Africa",  # combine North Africa (=4) and South Africa (+5)
    6: "North_Asia",
    7: "Central_Asia",
    8: "East_Asia",
    9: "South_Asia",
    10: "South_East_Asia",
    11: "Oceania",
}

# Backward compatibility: alias for old name
var_dict = VAR_CONVERSIONS

# ============================================================================
# CANONICAL VARIABLE REGISTRY
# ============================================================================
# Single source of truth for all carbon cycle variables
# Replaces the old var_list + var_mapping dual-list pattern
#
# Each variable has:
#   - description: Human-readable description
#   - stash_name: UM STASH variable name for extraction
#   - stash_code: Full STASH code string
#   - stash_fallback: Optional fallback STASH code (for frac→fracb)
#   - aggregation: "MEAN" or "SUM" (controls spatial aggregation method)
#   - conversion_factor: Multiplier to convert to output units
#   - units: Output units after conversion
#   - category: "flux", "stock", "climate", or "land_use"
#   - aliases: List of deprecated names (for backward compatibility)

CANONICAL_VARIABLES = {
    # -------------------------------------------------------------------------
    # Carbon fluxes (PgC/yr)
    # -------------------------------------------------------------------------
    "GPP": {
        "description": "Gross Primary Production",
        "stash_name": "gpp",
        "stash_code": "m01s03i261",
        "aggregation": "SUM",
        "conversion_factor": 3600*24*360*1e-12,  # kgC/m2/s → PgC/yr
        "units": "PgC/yr",
        "category": "flux",
        "aliases": [],
    },
    "NPP": {
        "description": "Net Primary Production",
        "stash_name": "npp",
        "stash_code": "m01s03i262",
        "aggregation": "SUM",
        "conversion_factor": 3600*24*360*1e-12,  # kgC/m2/s → PgC/yr
        "units": "PgC/yr",
        "category": "flux",
        "aliases": [],
    },
    "Rh": {
        "description": "Heterotrophic Respiration",
        "stash_name": "rh",
        "stash_code": "m01s03i293",
        "aggregation": "SUM",
        "conversion_factor": 3600*24*360*1e-12,  # kgC/m2/s → PgC/yr
        "units": "PgC/yr",
        "category": "flux",
        "aliases": ["soilResp"],
    },
    "fgco2": {
        "description": "Surface Downward Mass Flux of Carbon as CO2",
        "stash_name": "fgco2",
        "stash_code": "m02s30i249",
        "aggregation": "SUM",
        "conversion_factor": 12/1000*1e-12,  # molC/m2/yr → PgC/yr
        "units": "PgC/yr",
        "category": "flux",
        "aliases": [],
    },

    # -------------------------------------------------------------------------
    # Carbon stocks (PgC)
    # -------------------------------------------------------------------------
    "CVeg": {
        "description": "Vegetation Carbon Content",
        "stash_name": "cv",
        "stash_code": "m01s19i002",
        "aggregation": "SUM",
        "conversion_factor": 1e-12,  # kgC/m2 → PgC
        "units": "PgC",
        "category": "stock",
        "aliases": ["VegCarb", "vegetation_carbon_content"],
    },
    "CSoil": {
        "description": "Soil Carbon Content",
        "stash_name": "cs",
        "stash_code": "m01s19i016",
        "aggregation": "SUM",
        "conversion_factor": 1e-12,  # kgC/m2 → PgC
        "units": "PgC",
        "category": "stock",
        "aliases": ["soilCarbon"],
    },

    # -------------------------------------------------------------------------
    # Climate variables (use MEAN aggregation)
    # -------------------------------------------------------------------------
    "tas": {
        "description": "Near-Surface Air Temperature",
        "stash_name": "tas",
        "stash_code": "m01s03i236",
        "aggregation": "MEAN",  # Note: MEAN not SUM!
        "conversion_factor": 1.0,  # K (no conversion)
        "units": "K",
        "category": "climate",
        "aliases": ["temp"],
    },
    "pr": {
        "description": "Precipitation",
        "stash_name": "pr",
        "stash_code": "m01s05i216",
        "aggregation": "MEAN",  # Note: MEAN not SUM!
        "conversion_factor": 86400,  # kg/m2/s → mm/day
        "units": "mm/day",
        "category": "climate",
        "aliases": ["precip"],
    },

    # -------------------------------------------------------------------------
    # Land use / PFT fractions (use MEAN aggregation)
    # -------------------------------------------------------------------------
    "frac": {
        "description": "Plant Functional Type Grid Fractions",
        "stash_name": "frac",
        "stash_code": "m01s19i013",
        "stash_fallback": 19017,  # Try m01s19i017 (fracb) if frac not found
        "aggregation": "MEAN",  # Note: MEAN not SUM!
        "conversion_factor": 1.0,  # fraction (no conversion)
        "units": "1",
        "category": "land_use",
        "aliases": ["fracPFTs"],
    },

    # -------------------------------------------------------------------------
    # Atmospheric composition (use MEAN aggregation for 3D fields)
    # -------------------------------------------------------------------------
    "co2": {
        "description": "Atmospheric CO2 Mass Mixing Ratio",
        "stash_name": "co2",
        "stash_code": "m01s00i252",
        "aggregation": "MEAN",  # Note: MEAN for 3D vertical average!
        "conversion_factor": 28.97/44.01*1e6,  # mmr → ppmv
        "units": "ppmv",
        "category": "climate",
        "aliases": ["Total co2"],
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
        Canonical variable name (e.g., 'CVeg', 'Rh', 'tas')

    Raises
    ------
    ValueError
        If variable name not recognized

    Examples
    --------
    >>> resolve_variable_name('VegCarb')  # alias → canonical
    'CVeg'
    >>> resolve_variable_name('CVeg')  # already canonical
    'CVeg'
    >>> resolve_variable_name('soilResp')  # alias → canonical
    'Rh'
    >>> resolve_variable_name('temp')  # alias → canonical
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
        f"Known variables: {sorted(CANONICAL_VARIABLES.keys())}. "
        f"Known aliases: {sorted(sum([cfg.get('aliases', []) for cfg in CANONICAL_VARIABLES.values()], []))}"
    )


def get_variable_config(name: str) -> dict:
    """
    Get full configuration for a variable (resolves aliases automatically).

    Parameters
    ----------
    name : str
        Variable name (canonical or alias)

    Returns
    -------
    dict
        Variable configuration with keys:
        - description: str
        - stash_name: str (for data extraction)
        - stash_code: str
        - stash_fallback: int or None
        - aggregation: "MEAN" or "SUM"
        - conversion_factor: float
        - units: str
        - category: str
        - aliases: list of str

    Examples
    --------
    >>> cfg = get_variable_config('VegCarb')  # alias
    >>> cfg['description']
    'Vegetation Carbon Content'
    >>> cfg['conversion_factor']
    1e-12
    >>> cfg['aggregation']
    'SUM'

    >>> cfg = get_variable_config('tas')  # canonical
    >>> cfg['aggregation']
    'MEAN'
    """
    canonical = resolve_variable_name(name)
    config = {**CANONICAL_VARIABLES[canonical]}  # Return a copy
    config['canonical_name'] = canonical
    return config


def get_conversion_key(name: str) -> str:
    """
    Get the conversion key for compute_regional_annual_mean().

    This preserves backward compatibility with the old var_mapping system.
    Returns a key that encodes both the aggregation method and conversion factor.

    Parameters
    ----------
    name : str
        Variable name (canonical or alias)

    Returns
    -------
    str
        Conversion key for use with compute_regional_annual_mean()
        - Returns "Others" for generic MEAN aggregation variables
        - Returns "precip" for precipitation (MEAN with special conversion)
        - Returns "Total co2" for CO2 (MEAN with special conversion)
        - Returns canonical variable name for SUM aggregation variables

    Notes
    -----
    The conversion key is used by compute_regional_annual_mean() to:
    1. Determine aggregation method: "Others", "precip", or "Total co2" → MEAN, else → SUM
    2. Look up conversion factor in VAR_CONVERSIONS

    Examples
    --------
    >>> get_conversion_key('tas')
    'Others'  # MEAN aggregation, no conversion
    >>> get_conversion_key('GPP')
    'GPP'  # SUM aggregation
    >>> get_conversion_key('pr')
    'precip'  # MEAN aggregation with conversion
    >>> get_conversion_key('co2')
    'Total co2'  # MEAN aggregation with conversion
    """
    canonical = resolve_variable_name(name)
    config = CANONICAL_VARIABLES[canonical]

    # MEAN aggregation → use special keys for variables with conversions
    if config['aggregation'] == 'MEAN':
        if canonical == 'pr':
            return 'precip'
        elif canonical == 'co2':
            return 'Total co2'  # Legacy key name for backward compatibility
        else:
            return 'Others'

    # SUM aggregation → use canonical name
    # Need to ensure VAR_CONVERSIONS has the right mapping
    return canonical


# Default variable extraction configuration
# Used by extract_annual_means() in diagnostics.extraction
DEFAULT_VAR_LIST = [
    'Rh',          # Heterotrophic respiration (CMIP: Rh)
    'CSoil',       # Soil carbon (CMIP: CSoil)
    'CVeg',        # Vegetation carbon (CMIP: CVeg)
    'frac',        # PFT fractions
    'GPP',         # Gross primary production
    'NPP',         # Net primary production
    'fgco2',       # Ocean CO2 flux
    'tas',         # Surface air temperature (CMIP: tas)
    'pr',          # Precipitation (CMIP: pr)
    'co2'          # Atmospheric CO2 (CMIP: co2)
]

# DEPRECATED: Use CANONICAL_VARIABLES instead
# This is kept for backward compatibility only
DEFAULT_VAR_MAPPING = [
    'S resp',              # Rh → conversion key
    'S carb',              # CSoil → conversion key
    'V carb',              # CVeg → conversion key
    'Others',              # frac → conversion key (MEAN aggregation)
    'GPP',                 # GPP → conversion key
    'NPP',                 # NPP → conversion key
    'field646_mm_dpth',    # fgco2 → conversion key
    'Others',              # tas → conversion key (MEAN aggregation)
    'precip',              # pr → conversion key (MEAN aggregation)
    'Total co2'            # co2 → conversion key (MEAN aggregation)
]


def validate_reccap_mask_path(path=None):
    """
    Validate that the RECCAP mask file exists and is readable.

    Parameters
    ----------
    path : str, optional
        Path to RECCAP mask file. If None, uses RECCAP_MASK_PATH from config.

    Returns
    -------
    str
        Absolute path to the validated mask file

    Raises
    ------
    FileNotFoundError
        If the mask file does not exist
    RuntimeError
        If the file exists but cannot be read

    Examples
    --------
    >>> from utils_cmip7.config import validate_reccap_mask_path
    >>> mask_path = validate_reccap_mask_path()
    """
    import os

    if path is None:
        path = RECCAP_MASK_PATH

    # Expand user home directory
    path = os.path.expanduser(path)

    # Check if file exists
    if not os.path.exists(path):
        error_msg = (
            f"\n{'='*80}\n"
            f"ERROR: RECCAP2 regional mask file not found\n"
            f"{'='*80}\n\n"
            f"Expected location: {path}\n\n"
            f"This file is required for regional carbon cycle analysis.\n\n"
            f"Solutions:\n"
            f"  1. If the file exists elsewhere, set the environment variable:\n"
            f"     export UTILS_CMIP7_RECCAP_MASK=/path/to/your/mask.nc\n\n"
            f"  2. If you don't have the file, you may need to:\n"
            f"     - Obtain it from your research group\n"
            f"     - Download from RECCAP2 data repository\n"
            f"     - Create a custom regional mask matching your analysis grid\n\n"
            f"  3. For global-only analysis (no regional breakdown), this may\n"
            f"     indicate a code path issue - please report as a bug.\n"
            f"{'='*80}\n"
        )
        raise FileNotFoundError(error_msg)

    # Check if file is readable
    if not os.access(path, os.R_OK):
        error_msg = (
            f"\n{'='*80}\n"
            f"ERROR: RECCAP2 mask file exists but is not readable\n"
            f"{'='*80}\n\n"
            f"File location: {path}\n\n"
            f"The file exists but cannot be read (permission denied).\n"
            f"Please check file permissions.\n"
            f"{'='*80}\n"
        )
        raise RuntimeError(error_msg)

    return os.path.abspath(path)


def get_config_info():
    """
    Print current configuration information.

    Useful for debugging configuration issues.

    Examples
    --------
    >>> from utils_cmip7.config import get_config_info
    >>> get_config_info()
    """
    import os

    print("=" * 80)
    print("utils_cmip7 Configuration")
    print("=" * 80)
    print(f"\nRECCAP Mask Path:")
    print(f"  Environment variable: {os.environ.get('UTILS_CMIP7_RECCAP_MASK', '(not set)')}")
    print(f"  Current path: {RECCAP_MASK_PATH}")
    print(f"  Exists: {os.path.exists(os.path.expanduser(RECCAP_MASK_PATH))}")

    try:
        validated_path = validate_reccap_mask_path()
        print(f"  Validated: ✓ {validated_path}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  Validated: ✗ Error")
        print(f"\n{e}")

    print(f"\nRECCAP Regions: {len(RECCAP_REGIONS)} regions defined")
    for region_id, region_name in RECCAP_REGIONS.items():
        print(f"  {region_id:2d}. {region_name}")

    print(f"\nUnit Conversions: {len(VAR_CONVERSIONS)} conversion factors defined")
    print("=" * 80)

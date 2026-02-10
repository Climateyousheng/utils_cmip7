"""
Configuration module for utils_cmip7.

Contains unit conversion factors, file paths, and regional definitions.
"""

import os

# Legacy protocol keys (used by compute_regional_annual_mean dispatch)
# These 3 keys have no direct canonical variable equivalent
_LEGACY_PROTOCOL_KEYS = {
    'Others': 1,           # no conversion (generic MEAN aggregation)
    'precip': 86400,       # kg/m2/s → mm/day (MEAN aggregation)
    'Total co2': 28.97/44.01*1e6,  # mmr → ppmv (MEAN aggregation)
}

# VAR_CONVERSIONS is derived from CANONICAL_VARIABLES to prevent drift.
# Only the 3 legacy protocol keys above are maintained separately.
# This dict is populated after CANONICAL_VARIABLES is defined (see below).
VAR_CONVERSIONS = {}

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

# RECCAP2 approximate geographic bounding boxes (lon_min, lon_max, lat_min, lat_max)
# Used by plot_spatial_map() for regional subsetting
RECCAP_REGION_BOUNDS = {
    "North_America":   (-170, -50,   10,  80),
    "South_America":   ( -90, -30,  -60,  15),
    "Europe":          ( -15,  45,   35,  75),
    "Africa":          ( -20,  55,  -40,  40),
    "North_Asia":      (  40, 180,   50,  80),
    "Central_Asia":    (  45, 100,   25,  55),
    "East_Asia":       (  95, 150,   15,  55),
    "South_Asia":      (  60, 100,    5,  40),
    "South_East_Asia": (  90, 160,  -15,  25),
    "Oceania":         ( 110, 180,  -50,  -5),
}


def get_region_bounds(region_name):
    """
    Look up geographic bounding box for a RECCAP2 region.

    Parameters
    ----------
    region_name : str
        Region name, must match a key in RECCAP_REGION_BOUNDS
        (e.g. 'Europe', 'North_America').

    Returns
    -------
    tuple of (float, float, float, float)
        (lon_min, lon_max, lat_min, lat_max)

    Raises
    ------
    ValueError
        If region_name is not recognized.
    """
    if region_name not in RECCAP_REGION_BOUNDS:
        raise ValueError(
            f"Unknown region: '{region_name}'. "
            f"Available regions: {sorted(RECCAP_REGION_BOUNDS.keys())}"
        )
    return RECCAP_REGION_BOUNDS[region_name]


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
#   - units_in: Input units from UM (for validation)
#   - time_handling: "mean_rate" | "state" | "already_integral"
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
        "units_in": "kgC m-2 s-1",
        "time_handling": "mean_rate",
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
        "units_in": "kgC m-2 s-1",
        "time_handling": "mean_rate",
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
        "units_in": "kgC m-2 s-1",
        "time_handling": "mean_rate",
        "category": "flux",
        "aliases": ["soilResp"],  # Removed in v0.4.0, kept for error messages
    },
    "fgco2": {
        "description": "Surface Downward Mass Flux of Carbon as CO2",
        "stash_name": "fgco2",
        "stash_code": "m02s30i249",
        "aggregation": "SUM",
        # Assumes UM STASH output is molC/m2/yr (already per-year).
        # If the actual output is molC/m2/s, this factor is wrong by ~3.15e7.
        "conversion_factor": 12/1000*1e-12,  # molC/m2/yr → PgC/yr
        "units": "PgC/yr",
        "units_in": "molC m-2 yr-1",
        "time_handling": "already_integral",
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
        "units_in": "kgC m-2",
        "time_handling": "state",
        "category": "stock",
        "aliases": ["VegCarb"],  # Removed in v0.4.0, kept for error messages
    },
    "CSoil": {
        "description": "Soil Carbon Content",
        "stash_name": "cs",
        "stash_code": "m01s19i016",
        "aggregation": "SUM",
        "conversion_factor": 1e-12,  # kgC/m2 → PgC
        "units": "PgC",
        "units_in": "kgC m-2",
        "time_handling": "state",
        "category": "stock",
        "aliases": ["soilCarbon"],  # Removed in v0.4.0, kept for error messages
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
        "units_in": "K",
        "time_handling": "state",
        "category": "climate",
        "aliases": ["temp"],  # Removed in v0.4.0, kept for error messages
    },
    "pr": {
        "description": "Precipitation",
        "stash_name": "pr",
        "stash_code": "m01s05i216",
        "aggregation": "MEAN",  # Note: MEAN not SUM!
        "conversion_factor": 86400,  # kg/m2/s → mm/day
        "units": "mm/day",
        "units_in": "kg m-2 s-1",
        "time_handling": "mean_rate",
        "category": "climate",
        "aliases": ["precip"],  # Removed in v0.4.0, kept for error messages
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
        "units_in": "1",
        "time_handling": "state",
        "category": "land_use",
        "aliases": ["fracPFTs"],  # Removed in v0.4.0, kept for error messages
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
        "units_in": "kg kg-1",
        "time_handling": "state",
        "category": "climate",
        "aliases": ["Total co2"],  # Removed in v0.4.0, kept for error messages
    },
}

# Populate VAR_CONVERSIONS from CANONICAL_VARIABLES (single source of truth)
VAR_CONVERSIONS = dict(_LEGACY_PROTOCOL_KEYS)
for _name, _cfg in CANONICAL_VARIABLES.items():
    VAR_CONVERSIONS[_name] = float(_cfg["conversion_factor"])


def resolve_variable_name(name: str) -> str:
    """
    Resolve a canonical variable name.

    Parameters
    ----------
    name : str
        Canonical variable name (e.g., 'CVeg', 'Rh', 'tas')

    Returns
    -------
    str
        Canonical variable name

    Raises
    ------
    ValueError
        If variable name not recognized, or if a removed alias is used.

    Examples
    --------
    >>> resolve_variable_name('CVeg')
    'CVeg'
    >>> resolve_variable_name('GPP')
    'GPP'
    """
    # Already canonical?
    if name in CANONICAL_VARIABLES:
        return name

    # Search aliases — raise ValueError with migration message
    for canonical_name, config in CANONICAL_VARIABLES.items():
        if name in config.get("aliases", []):
            raise ValueError(
                f"Variable name '{name}' was removed in v0.4.0. "
                f"Use '{canonical_name}' instead."
            )

    raise ValueError(
        f"Unknown variable name: '{name}'. "
        f"Known variables: {sorted(CANONICAL_VARIABLES.keys())}"
    )


def get_variable_config(name: str) -> dict:
    """
    Get full configuration for a variable.

    Parameters
    ----------
    name : str
        Canonical variable name (e.g., 'CVeg', 'Rh', 'tas')

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
        - units: str (output units after conversion)
        - units_in: str (input units from UM)
        - time_handling: "mean_rate", "state", or "already_integral"
        - category: str
        - aliases: list of str

    Raises
    ------
    ValueError
        If variable name not recognized or a removed alias is used.

    Examples
    --------
    >>> cfg = get_variable_config('CVeg')
    >>> cfg['conversion_factor']
    1e-12
    >>> cfg['aggregation']
    'SUM'

    >>> cfg = get_variable_config('tas')
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

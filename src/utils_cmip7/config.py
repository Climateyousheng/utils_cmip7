"""
Configuration module for utils_cmip7.

Contains unit conversion factors, file paths, and regional definitions.
"""

import os

# Unit conversion factors for various variables
# Maps variable names/codes to conversion factors
VAR_CONVERSIONS = {
    'Ocean flux': (12/44)*3600*24*360*(1e-12),         # kgCO2/m2/s to PgC/yr
    'm01s00i250': (12/44)*3600*24*360*(1e-12),         # same as Ocean flux
    'field1560_mm_srf': (12/44)*3600*24*360*(1e-12),   # same as Ocean flux
    'GPP': 3600*24*360*(1e-12),                        # from kgC/m2/s to PgC/yr
    'NPP': 3600*24*360*(1e-12),                        # from kgC/m2/s to PgC/yr
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
    'Others': 1,                                       # no conversion
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

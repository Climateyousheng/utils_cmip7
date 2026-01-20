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

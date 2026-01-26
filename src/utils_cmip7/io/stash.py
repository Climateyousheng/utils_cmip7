"""
STASH code mapping utilities for Unified Model variables.

Maps short variable names to STASH codes in both MSI string format (e.g., 'm01s03i261')
and numeric format (e.g., 3261).
"""

__all__ = [
    'stash',
    'stash_nc',
]


def stash(s):
    """
    Map short variable name to MSI-format STASH code string.

    Parameters
    ----------
    s : str
        Short variable name (e.g., 'gpp', 'npp', 'tas')

    Returns
    -------
    str
        MSI-format STASH code (e.g., 'm01s03i261') or "nothing" if not found

    Examples
    --------
    >>> stash('gpp')
    'm01s03i261'
    >>> stash('tas')
    'm01s03i236'
    """
    switcher = {
        'tas': 'm01s03i236',
        'pr': 'm01s05i216',
        'gpp': 'm01s03i261',
        'npp': 'm01s03i262',
        'rh': 'm01s03i293',
        'landcflx': 'm01s03i326',
        'totcflx': 'm01s03i327',
        'cv': 'm01s19i002',
        'cs': 'm01s19i016',
        'dist': 'm01s19i012',
        'frac': 'm01s19i013',
        'ocn': 'm01s00i250',
        'emiss': 'm01s00i251',
        'co2': 'm01s00i252',
        'tos': 'm02s00i101',
        'sal': 'm02s00i102',
        'tco2': 'm02s00i103',
        'alk': 'm02s00i104',
        'nut': 'm02s00i105',
        'phy': 'm02s00i106',
        'zoo': 'm02s00i107',
        'detn': 'm02s00i108',
        'detc': 'm02s00i109',
        'pco2': 'm02s30i248',
        'fgco2': 'm02s30i249',
        'rlut': 'm01s02i205',
        'rlutcs': 'm01s02i206',
        'rsdt': 'm01s01i207',
        'rsut': 'm01s01i208',
        'rsutcs': 'm01s01i209',
    }

    return switcher.get(s, "nothing")


def stash_nc(s):
    """
    Map short variable name to numeric STASH code.

    Parameters
    ----------
    s : str
        Short variable name (e.g., 'gpp', 'npp', 'tas')

    Returns
    -------
    int or str
        Numeric STASH code (e.g., 3261) or "nothing" if not found

    Examples
    --------
    >>> stash_nc('gpp')
    3261
    >>> stash_nc('tas')
    3236
    """
    switcher = {
        'tas': 3236,
        'pr': 5216,
        'gpp': 3261,
        'npp': 3262,
        'rh': 3293,
        'landcflx': 3326,
        'totcflx': 3327,
        'cv': 19002,
        'cs': 19016,
        'dist': 19012,
        'frac': 19013,
        'ocn': 250,
        'emiss': 251,
        'co2': 252,
        'tos': 101,
        'sal': 102,
        'tco2': 103,
        'alk': 104,
        'nut': 105,
        'phy': 106,
        'zoo': 107,
        'detn': 108,
        'detc': 109,
        'pco2': 30248,
        'fgco2': 30249,
        'rlut': 2205,
        'rlutcs': 2206,
        'rsdt': 1207,
        'rsut': 1208,
        'rsutcs': 1209,
    }

    return switcher.get(s, "nothing")

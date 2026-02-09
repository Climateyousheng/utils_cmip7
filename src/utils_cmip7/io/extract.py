"""
Cube extraction utilities with STASH code handling.

Provides flexible extraction of iris cubes based on STASH codes in both
PP format (STASH objects) and NetCDF format (numeric stash_code attributes).
"""

import numpy as np
import iris
from iris import Constraint


def _msi_from_stash_obj(st):
    """
    Convert STASH object to MSI string format.

    Parameters
    ----------
    st : iris STASH object or similar
        STASH object with model, section, item attributes

    Returns
    -------
    str or None
        MSI string like 'm01s03i261' or None if conversion fails

    Examples
    --------
    >>> # STASH(model=1, section=3, item=261) -> 'm01s03i261'
    """
    if st is None:
        return None
    try:
        return f"m{int(st.model):02d}s{int(st.section):02d}i{int(st.item):03d}"
    except Exception:
        pass
    try:
        # some iris versions expose .msi
        s = str(st.msi).strip()
        if s.startswith("m") and "s" in s and "i" in s:
            return s
    except Exception:
        pass
    return None


def _msi_from_numeric_stash_code(code):
    """
    Convert numeric stash_code to MSI string format.

    Uses heuristic for model number:
    - section >= 30 => model 2 (ocean/CO2 flux)
    - else => model 1 (atmosphere)

    Parameters
    ----------
    code : int or numeric
        Numeric stash code like 3261

    Returns
    -------
    str or None
        MSI string like 'm01s03i261' or None if conversion fails

    Examples
    --------
    >>> _msi_from_numeric_stash_code(3261)
    'm01s03i261'
    >>> _msi_from_numeric_stash_code(30249)
    'm02s30i249'
    """
    if code is None:
        return None
    try:
        n = int(code)  # handles np.int16 etc
    except Exception:
        return None
    section = n // 1000
    item = n % 1000
    model = 2 if section >= 30 else 1
    return f"m{model:02d}s{section:02d}i{item:03d}"


def _msi_from_any_attr(attrs):
    """
    Extract MSI from either STASH object or numeric stash_code attribute.

    Parameters
    ----------
    attrs : dict
        Cube attributes dictionary

    Returns
    -------
    str or None
        MSI string or None if not found
    """
    if not attrs:
        return None
    # PP-style
    if "STASH" in attrs:
        msi = _msi_from_stash_obj(attrs.get("STASH"))
        if msi:
            return msi
    # NetCDF-style numeric
    if "stash_code" in attrs:
        msi = _msi_from_numeric_stash_code(attrs.get("stash_code"))
        if msi:
            return msi
    return None


def try_extract(cubes, code, stash_lookup_func=None, debug=False):
    """
    Extract cubes matching STASH codes from a CubeList.

    Handles both PP-style STASH objects and NetCDF-style numeric stash_code
    attributes. Supports flexible code formats.

    Parameters
    ----------
    cubes : iris.cube.CubeList
        Collection of cubes to search
    code : str, int, or numeric
        Can be:
        - Canonical variable name: 'CVeg', 'GPP' (from CANONICAL_VARIABLES)
        - Alias: 'VegCarb', 'soilResp' (resolved via CANONICAL_VARIABLES)
        - MSI string: 'm01s03i261'
        - Short name: 'gpp' (requires stash_lookup_func)
        - Numeric stash_code: 3261
    stash_lookup_func : callable, optional
        Function to map short names to MSI strings (e.g., stash())
    debug : bool, default False
        Print debug information during extraction

    Returns
    -------
    iris.cube.CubeList
        Cubes matching the code, or empty CubeList if none found

    Examples
    --------
    >>> from utils_cmip7.io.stash import stash
    >>> cubes = iris.load('data.nc')
    >>> gpp = try_extract(cubes, 'gpp', stash_lookup_func=stash)
    >>> gpp = try_extract(cubes, 'm01s03i261')
    >>> gpp = try_extract(cubes, 3261)

    Notes
    -----
    The function normalizes all code formats to MSI strings internally,
    then matches against cube attributes. This provides consistent behavior
    regardless of whether the data files use PP or NetCDF conventions.
    """
    candidates = [code]

    # Resolve canonical variable names to stash_code (MSI)
    if isinstance(code, str):
        from ..config import CANONICAL_VARIABLES
        _resolved_config = None
        if code in CANONICAL_VARIABLES:
            _resolved_config = CANONICAL_VARIABLES[code]
        if _resolved_config is not None:
            candidates.append(_resolved_config["stash_name"])
            candidates.append(_resolved_config["stash_code"])

    # Expand short-name -> MSI using your mapping
    if stash_lookup_func is not None and isinstance(code, str):
        msi = stash_lookup_func(code)
        if msi and msi != "nothing":
            candidates.append(msi)

    # Add coercions
    try:
        candidates.append(str(code))
    except Exception:
        pass

    # If numeric-like, include int form
    if isinstance(code, (int, np.integer)) or (isinstance(code, str) and code.isdigit()):
        try:
            candidates.append(int(code))
        except Exception:
            pass

    # Normalise candidate MSIs
    cand_msi = set()
    for c in candidates:
        # MSI strings pass through
        if isinstance(c, str) and c.startswith("m") and "s" in c and "i" in c:
            cand_msi.add(c.strip())
            continue
        # numeric stash_code -> MSI
        msi = _msi_from_numeric_stash_code(c)
        if msi:
            cand_msi.add(msi)

    if debug:
        print(f"Trying to extract cube for candidates: {candidates}")
        print(f"Normalized candidate MSIs: {cand_msi}")

    def _match(c):
        attrs = getattr(c, "attributes", {}) or {}
        cube_msi = _msi_from_any_attr(attrs)

        if debug:
            print(f"Cube: {c.name()} attrs keys={list(attrs.keys())} -> MSI={cube_msi}")

        return (cube_msi in cand_msi)

    try:
        return cubes.extract(Constraint(cube_func=_match))
    except Exception:
        return iris.cube.CubeList([])

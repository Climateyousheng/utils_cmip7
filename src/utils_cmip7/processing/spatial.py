"""
Spatial aggregation functions for global and regional analysis.

Provides area-weighted summation and averaging over latitude/longitude dimensions.
"""

import numpy as np
import iris
import iris.analysis
import iris.analysis.cartography as cart
from iris.analysis.cartography import area_weights

from ..config import VAR_CONVERSIONS


def compute_terrestrial_area(cube):
    """
    Compute total terrestrial area from a masked cube.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube with lat/lon coordinates and land/sea mask

    Returns
    -------
    float
        Total terrestrial area in m²

    Notes
    -----
    This function is currently unused in the main workflows but kept for
    potential future use.
    """
    if cube.ndim > 2:
        cube2d = cube[0]
    else:
        cube2d = cube
    for name in ("latitude", "longitude"):
        coord = cube2d.coord(name)
        if not coord.has_bounds():
            coord.guess_bounds()
    cellarea = cart.area_weights(cube2d)
    mask = np.ma.getmaskarray(cube2d.data)
    total_area = np.ma.array(cellarea, mask=mask).sum()
    return float(total_area)


def global_total_pgC(cube, var):
    """
    Compute area-weighted global total with unit conversion.

    Collapses latitude and longitude dimensions using SUM aggregation
    with area weights, then applies variable-specific unit conversion.

    Parameters
    ----------
    cube : iris.cube.Cube or iris.cube.CubeList
        Input cube with lat/lon coordinates. If CubeList, uses first cube.
    var : str
        Variable name for unit conversion lookup (e.g., 'GPP', 'NPP', 'S resp')

    Returns
    -------
    iris.cube.Cube
        Collapsed cube with global total in converted units

    Raises
    ------
    ValueError
        If cube is None or CubeList is empty

    Examples
    --------
    >>> gpp_cube = iris.load_cube('gpp.nc')
    >>> gpp_global = global_total_pgC(gpp_cube, 'GPP')
    >>> # Result is in PgC/year

    Notes
    -----
    Uses VAR_CONVERSIONS dict for unit conversion factors.
    Common conversions:
    - 'GPP', 'NPP', 'S resp': kgC/m²/s → PgC/yr
    - 'V carb', 'S carb': kgC/m² → PgC
    """
    # ensure cube is not None
    if cube is None:
        raise ValueError("None cube passed to global_total_pgC()")
    # Handle CubeList input
    if isinstance(cube, iris.cube.CubeList):
        if not cube:
            raise ValueError("Empty CubeList passed to global_total_pgC()")
        cube = cube[0]  # Get first cube in CubeList
    for name in ("latitude", "longitude"):
        coord = cube.coord(name)
        if not coord.has_bounds():
            coord.guess_bounds()
    weights = cart.area_weights(cube)
    gm = cube.collapsed(["latitude", "longitude"], iris.analysis.SUM, weights=weights)
    gm.data = gm.data * VAR_CONVERSIONS[var]
    return gm


def global_mean_pgC(cube, var):
    """
    Compute area-weighted global mean with unit conversion.

    Collapses latitude and longitude dimensions using MEAN aggregation
    with area weights, then applies variable-specific unit conversion.

    Parameters
    ----------
    cube : iris.cube.Cube or iris.cube.CubeList
        Input cube with lat/lon coordinates. If CubeList, uses first cube.
    var : str
        Variable name for unit conversion lookup (e.g., 'precip', 'Others')

    Returns
    -------
    iris.cube.Cube
        Collapsed cube with global mean in converted units

    Raises
    ------
    ValueError
        If cube is None or CubeList is empty

    Examples
    --------
    >>> precip_cube = iris.load_cube('precip.nc')
    >>> precip_global = global_mean_pgC(precip_cube, 'precip')
    >>> # Result is in mm/day

    Notes
    -----
    Uses VAR_CONVERSIONS dict for unit conversion factors.
    Common conversions:
    - 'precip': kg/m²/s → mm/day
    - 'Others': no conversion (factor = 1)
    """
    # ensure cube is not None
    if cube is None:
        raise ValueError("None cube passed to global_mean_pgC()")
    # Handle CubeList input
    if isinstance(cube, iris.cube.CubeList):
        if not cube:
            raise ValueError("Empty CubeList passed to global_mean_pgC()")
        cube = cube[0]  # Get first cube in CubeList
    for name in ("latitude", "longitude"):
        coord = cube.coord(name)
        if not coord.has_bounds():
            coord.guess_bounds()
    weights = cart.area_weights(cube)
    gm = cube.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=weights)
    gm.data = gm.data * VAR_CONVERSIONS[var]
    return gm

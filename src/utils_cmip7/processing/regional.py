"""
Regional aggregation and masking functions.

Provides RECCAP2 regional masking and area-weighted regional aggregation.
"""

import warnings
import numpy as np
from functools import lru_cache

# Configure Iris and suppress warnings before importing
try:
    import iris
    iris.FUTURE.date_microseconds = True
except AttributeError:
    import iris

# Suppress Iris warnings
warnings.filterwarnings('ignore', message='.*date precision.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*DEFAULT_SPHERICAL_EARTH_RADIUS.*')

import iris.analysis
import cf_units
import iris.analysis.cartography as cart
from iris.analysis.cartography import area_weights

from ..config import RECCAP_MASK_PATH, RECCAP_REGIONS, VAR_CONVERSIONS, validate_reccap_mask_path


@lru_cache(maxsize=1)
def load_reccap_mask():
    """
    Load RECCAP2 regional mask from configured path.

    **Performance**: Result is cached in memory - file loaded only once per session.

    Returns
    -------
    tuple
        (reccap_mask, regions) where:
        - reccap_mask: iris.cube.Cube with regional mask data
        - regions: dict mapping region IDs to names

    Raises
    ------
    FileNotFoundError
        If the mask file does not exist
    RuntimeError
        If the file exists but cannot be read

    Notes
    -----
    Mask path is configured via RECCAP_MASK_PATH in config.py.
    Override with environment variable: UTILS_CMIP7_RECCAP_MASK

    Region definitions:
    - 1: North_America
    - 2: South_America
    - 3: Europe
    - 4: Africa (combines 4 and 5)
    - 6: North_Asia
    - 7: Central_Asia
    - 8: East_Asia
    - 9: South_Asia
    - 10: South_East_Asia
    - 11: Oceania

    Examples
    --------
    >>> mask, regions = load_reccap_mask()
    >>> print(regions)
    {1: 'North_America', 2: 'South_America', ...}
    """
    # Validate mask file exists and is readable
    mask_path = validate_reccap_mask_path()

    # Load the mask
    reccap_mask = iris.load_cube(mask_path)
    return reccap_mask, RECCAP_REGIONS


def region_mask(region):
    """
    Generate binary mask for a specific RECCAP2 region.

    Parameters
    ----------
    region : str
        Region name (e.g., 'North_America', 'Europe', 'Africa')

    Returns
    -------
    iris.cube.Cube
        Binary mask cube (1 for region, 0 elsewhere)

    Raises
    ------
    ValueError
        If region name not found in RECCAP regions

    Examples
    --------
    >>> mask = region_mask('Europe')
    >>> mask.data  # Binary array (1=Europe, 0=elsewhere)

    Notes
    -----
    Special case: 'Africa' combines North Africa (ID=4) and South Africa (ID=5)
    """
    reccap_mask, regions = load_reccap_mask()

    # Find correct ID(s) for the region
    if region == "Africa":
        target_ids = [4, 5]
    else:
        target_ids = [k for k, v in regions.items() if v == region]
        if not target_ids:
            raise ValueError(f"Region '{region}' not found in RECCAP regions.")

    # Build binary mask
    mask = reccap_mask.copy()
    mask.data = np.isin(mask.data, target_ids).astype(int)
    return mask


@lru_cache(maxsize=1)
def _get_land_mask():
    """
    Get binary land mask from RECCAP2 regions.

    Any grid cell belonging to a RECCAP2 region (value > 0) is land.

    **Performance**: Result is cached in memory - computed only once per session.

    Returns
    -------
    numpy.ndarray
        2D binary array (1 = land, 0 = ocean)
    """
    reccap_mask, _ = load_reccap_mask()
    return (np.squeeze(np.asarray(reccap_mask.data)) > 0).astype(int)


def compute_regional_annual_mean(cube, var, region, land_only=False):
    """
    Compute area-weighted regional annual means.

    Applies regional masking and computes annual means for the specified
    region using appropriate aggregation (SUM or MEAN based on variable type).

    Parameters
    ----------
    cube : iris.cube.Cube or iris.cube.CubeList
        Input cube with time, latitude, longitude dimensions.
        If CubeList, uses first cube.
    var : str
        Variable name for unit conversion and aggregation type selection
    region : str
        Region name ('global' or RECCAP2 region name)
    land_only : bool, optional
        If True and region is 'global', restrict the spatial average to
        land grid cells (derived from RECCAP2 mask). Has no effect for
        named regions, which are already land-only by construction.
        Default: False

    Returns
    -------
    dict
        Dictionary with keys:
        - 'years': unique years as numpy array
        - 'data': annual mean values
        - 'name': variable long name
        - 'units': units string
        - 'region': region name

    Raises
    ------
    ValueError
        If cube is empty, mask shape doesn't match grid, or no time coordinate

    Examples
    --------
    >>> gpp_cube = iris.load_cube('gpp.nc')
    >>> europe_gpp = compute_regional_annual_mean(gpp_cube, 'GPP', 'Europe')
    >>> europe_gpp['data']  # Annual GPP for Europe in PgC/year

    >>> global_gpp = compute_regional_annual_mean(gpp_cube, 'GPP', 'global')
    >>> global_gpp['data']  # Global annual GPP in PgC/year

    >>> # Land-only global mean (excludes ocean cells)
    >>> frac_pft = iris.load_cube('frac.nc')
    >>> global_frac = compute_regional_annual_mean(frac_pft, 'Others', 'global',
    ...                                           land_only=True)

    Notes
    -----
    Aggregation method selection:
    - MEAN: var in ('Others', 'precip', 'Total co2')
    - SUM: all other variables (fluxes and stocks)

    Regional masking:
    - Applies mask to area weights (not to data directly)
    - Handles 4D cubes with pfts dimension
    - 'global' region uses no masking (unless land_only=True)

    Time handling:
    - Handles 360-day calendars
    - Automatically sets calendar if missing or 'unknown'
    - Averages duplicates within each year using np.nanmean
    """
    # --- Handle CubeList input ---
    if isinstance(cube, iris.cube.CubeList):
        if not cube:
            raise ValueError("Empty CubeList passed to compute_regional_annual_mean()")
        cube = cube[0]

    cube = cube.copy()

    # --- Ensure bounds for area weights ---
    for name in ("latitude", "longitude"):
        coord = cube.coord(name)
        if not coord.has_bounds():
            coord.guess_bounds()

    # --- Build weights, and mask weights for region ---
    weights = cart.area_weights(cube)

    if region != "global":
        m_obj = region_mask(region)
        m2d = np.asarray(m_obj.data) if isinstance(m_obj, iris.cube.Cube) else np.asarray(m_obj)
        m2d = np.squeeze(m2d)
        # Sanity check: mask must match the horizontal grid
        if m2d.shape != weights.shape[-2:]:
            raise ValueError(f"Mask shape {m2d.shape} != weights lat/lon shape {weights.shape[-2:]}")
        # Handle pfts broadcasting
        if cube.ndim == 4:
            m = m2d[None, None, :, :]
        else:
            m = m2d[None, :, :]
        w = weights * m
    elif land_only:
        m2d = _get_land_mask()
        if m2d.shape != weights.shape[-2:]:
            raise ValueError(f"Land mask shape {m2d.shape} != weights lat/lon shape {weights.shape[-2:]}")
        if cube.ndim == 4:
            m = m2d[None, None, :, :]
        else:
            m = m2d[None, :, :]
        w = weights * m
    else:
        w = weights

    # --- Compute regional mean or total based on variable ---
    # IMPORTANT: do the collapse HERE, using w, not inside other functions.
    # Variables using MEAN aggregation: Others (generic), precip, Total co2
    if var in ("Others", "precip", "Total co2"):
        gm = cube.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=w)
    else:
        gm = cube.collapsed(["latitude", "longitude"], iris.analysis.SUM, weights=w)

    # Apply scaling after collapse (cheap)
    gm.data = gm.data * VAR_CONVERSIONS[var]

    # --- Get time coordinate robustly ---
    time_coords = [c for c in gm.coords() if c.standard_name == "time" or c.name() in ("t", "time", "TIME")]
    if not time_coords:
        raise ValueError("No valid time coordinate found in collapsed cube.")
    tcoord = time_coords[0]

    # --- Ensure calendar and units are valid ---
    if not getattr(tcoord.units, "calendar", None) or str(tcoord.units).startswith("unknown"):
        tcoord.units = cf_units.Unit("days since 1850-12-01 00:00:00", calendar="360_day")

    # --- Convert time → years ---
    times = tcoord.units.num2date(tcoord.points)
    years = np.array([t.year for t in times])

    unique_years, idx = np.unique(years, return_inverse=True)

    # Force realisation only on the 1D time series (prevents "background" lazy compute later)
    gm_series = np.asarray(gm.data)

    if gm_series.ndim == 0:
        annual_means = np.repeat(gm_series.item(), len(unique_years))
    else:
        annual_means = np.array([np.nanmean(gm_series[idx == i]) for i in range(len(unique_years))])

    return {
        "years": unique_years,
        "data": annual_means,
        "name": cube.long_name or cube.standard_name or var,
        "units": str(gm.units),
        "region": region,
    }

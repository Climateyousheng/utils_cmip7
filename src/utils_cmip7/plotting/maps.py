"""
Geographic map plotting for 2D spatial fields.

Renders iris Cubes on cartopy map projections with configurable time
selection and regional subsetting (by RECCAP2 name or explicit bounds).

All functions accept matplotlib Axes objects for flexible composition.
No filesystem discovery or NetCDF I/O is performed inside this module.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    import iris
    HAS_IRIS = True
except ImportError:
    HAS_IRIS = False

from ..config import get_region_bounds


def _select_time_slice(cube, time=None, time_index=None):
    """
    Extract a 2D (lat, lon) slice from a cube, optionally selecting by time.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube with at least latitude and longitude dimensions.
        May optionally have a time dimension.
    time : int, optional
        Year to select.  If the cube has multiple timesteps in that year
        (e.g. monthly data), the mean over the year is returned.
        Mutually exclusive with *time_index*.
    time_index : int, optional
        Positional index along the time dimension.
        Mutually exclusive with *time*.

    Returns
    -------
    data : numpy.ndarray
        2D array of shape (lat, lon).
    year : int or None
        The year associated with the selected slice, or None if the
        cube has no time dimension.

    Raises
    ------
    ValueError
        If both *time* and *time_index* are specified, or if the
        requested year is not present in the cube's time coordinate.
    """
    if time is not None and time_index is not None:
        raise ValueError(
            "Cannot specify both 'time' and 'time_index'; they are "
            "mutually exclusive."
        )

    # Identify time dimension (if any)
    time_coords = cube.coords("time", dim_coords=True)
    has_time = len(time_coords) > 0

    # 2D cube — no time dimension
    if not has_time:
        return cube.data, None

    time_coord = time_coords[0]
    time_dim = cube.coord_dims(time_coord)[0]

    if time is not None:
        # Select by year — may need to average across months
        cell_years = np.array([
            cell.point.year for cell in time_coord.cells()
        ])
        mask = cell_years == time
        if not np.any(mask):
            available = sorted(set(cell_years.tolist()))
            raise ValueError(
                f"Year {time} not found in cube time coordinate. "
                f"Available years: {available}"
            )
        indices = np.where(mask)[0]
        slices = [slice(None)] * cube.ndim
        slices[time_dim] = indices
        subset = cube[tuple(slices)]
        data_2d = np.mean(subset.data, axis=time_dim)
        return data_2d, time

    if time_index is not None:
        idx = time_index
    else:
        # Default: first timestep
        idx = 0

    slices = [slice(None)] * cube.ndim
    slices[time_dim] = idx
    data_2d = cube[tuple(slices)].data
    year = time_coord.cell(idx).point.year
    return data_2d, year


def plot_spatial_map(
    cube,
    time=None,
    time_index=None,
    region=None,
    lon_bounds=None,
    lat_bounds=None,
    projection=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    title=None,
    units=None,
    add_coastlines=True,
    add_gridlines=True,
    colorbar=True,
    ax=None,
):
    """
    Plot a 2D field from an iris Cube on a cartopy map projection.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube with latitude and longitude dimensions (and optional
        time dimension).
    time : int, optional
        Year to plot.  If the cube contains multiple timesteps for that
        year the mean is used.  Mutually exclusive with *time_index*.
    time_index : int, optional
        Positional index along the time dimension.
        Mutually exclusive with *time*.
    region : str, optional
        RECCAP2 region name (e.g. ``'Europe'``).  Sets the map extent
        automatically.  Mutually exclusive with *lon_bounds*/*lat_bounds*.
    lon_bounds : tuple of (float, float), optional
        ``(lon_min, lon_max)`` for the map extent.
        Mutually exclusive with *region*.
    lat_bounds : tuple of (float, float), optional
        ``(lat_min, lat_max)`` for the map extent.
        Mutually exclusive with *region*.
    projection : cartopy.crs.Projection, optional
        Map projection.  Default: Robinson for global views,
        PlateCarree for regional views.
    cmap : str or matplotlib.colors.Colormap, default ``'viridis'``
        Colormap for the field.
    vmin : float, optional
        Lower bound for the colour scale.
    vmax : float, optional
        Upper bound for the colour scale.
    title : str, optional
        Plot title.  Auto-generated from cube metadata if *None*.
    units : str, optional
        Colorbar label.  Auto-detected from cube if *None*.
    add_coastlines : bool, default True
        Whether to draw coastlines.
    add_gridlines : bool, default True
        Whether to draw gridlines with labels.
    colorbar : bool, default True
        Whether to add a horizontal colorbar.
    ax : cartopy.mpl.geoaxes.GeoAxes, optional
        Pre-existing GeoAxes to plot on.  If *None*, a new figure is
        created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : cartopy.mpl.geoaxes.GeoAxes

    Raises
    ------
    ImportError
        If cartopy or iris is not installed.
    TypeError
        If *cube* is not an iris Cube, or *ax* is not a GeoAxes.
    ValueError
        If mutually exclusive arguments are both supplied, or a named
        region is not recognized.
    """
    if not HAS_CARTOPY:
        raise ImportError(
            "cartopy is required for plot_spatial_map(). "
            "Install it with: pip install cartopy"
        )
    if not HAS_IRIS:
        raise ImportError(
            "iris is required for plot_spatial_map(). "
            "Install it with: pip install scitools-iris"
        )

    # ---- input validation ------------------------------------------------
    if not isinstance(cube, iris.cube.Cube):
        raise TypeError(
            f"Expected an iris.cube.Cube, got {type(cube).__name__}"
        )

    if region is not None and (lon_bounds is not None or lat_bounds is not None):
        raise ValueError(
            "Cannot specify both 'region' and 'lon_bounds'/'lat_bounds'; "
            "they are mutually exclusive."
        )

    if ax is not None and not hasattr(ax, "projection"):
        raise TypeError(
            "When providing 'ax', it must be a cartopy GeoAxes "
            "(created with subplot_kw={'projection': ...})."
        )

    if (lon_bounds is None) != (lat_bounds is None):
        raise ValueError(
            "Both 'lon_bounds' and 'lat_bounds' must be provided together."
        )

    # ---- time selection --------------------------------------------------
    data_2d, selected_year = _select_time_slice(cube, time=time, time_index=time_index)

    # ---- coordinate extraction -------------------------------------------
    lon_coord = cube.coord("longitude")
    lat_coord = cube.coord("latitude")
    lons = lon_coord.points
    lats = lat_coord.points

    # ---- region / extent resolution --------------------------------------
    is_regional = False
    extent = None

    if region is not None:
        lon_min, lon_max, lat_min, lat_max = get_region_bounds(region)
        extent = (lon_min, lon_max, lat_min, lat_max)
        is_regional = True

    if lon_bounds is not None and lat_bounds is not None:
        extent = (lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1])
        is_regional = True

    # ---- projection ------------------------------------------------------
    if projection is not None:
        proj = projection
    elif is_regional:
        proj = ccrs.PlateCarree()
    else:
        proj = ccrs.Robinson()

    # ---- figure / axes ---------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(12, 6),
            subplot_kw={"projection": proj},
        )
    else:
        fig = ax.get_figure()

    # ---- plot data -------------------------------------------------------
    # Use contourf rather than pcolormesh to avoid a known cartopy 0.25 /
    # shapely 2.x bug with non-PlateCarree projections.
    lon2d, lat2d = np.meshgrid(lons, lats)
    contourf_kwargs = {
        "transform": ccrs.PlateCarree(),
        "cmap": cmap,
        "extend": "both",
    }
    if vmin is not None and vmax is not None:
        contourf_kwargs["levels"] = np.linspace(vmin, vmax, 21)
    else:
        contourf_kwargs["levels"] = 20
    mesh = ax.contourf(lon2d, lat2d, data_2d, **contourf_kwargs)

    # ---- map decorations -------------------------------------------------
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    elif not is_regional:
        ax.set_global()

    if add_coastlines:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    if add_gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

    # ---- colorbar --------------------------------------------------------
    if colorbar:
        cbar_label = units if units is not None else str(cube.units)
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal",
                            pad=0.05, shrink=0.7)
        cbar.set_label(cbar_label)

    # ---- title -----------------------------------------------------------
    if title is not None:
        ax.set_title(title)
    else:
        name = cube.name() or "field"
        year_str = f" ({selected_year})" if selected_year is not None else ""
        ax.set_title(f"{name}{year_str}")

    return fig, ax


__all__ = [
    "plot_spatial_map",
]

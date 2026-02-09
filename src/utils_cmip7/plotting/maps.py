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
    from cartopy.util import add_cyclic_point
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    import iris
    HAS_IRIS = True
except ImportError:
    HAS_IRIS = False

from ..config import get_region_bounds


def _get_cell_years(time_coord):
    """Return an array of integer years for every cell in *time_coord*.

    Works for both datetime-decoded and raw-numeric time coordinates.
    For numeric coordinates, converts via ``time_coord.units.num2date``.

    Returns *None* if year extraction is not possible (e.g. the time
    coordinate has no calendar metadata).
    """
    cells = list(time_coord.cells())
    # Fast path: datetime-like points
    first_pt = cells[0].point
    if hasattr(first_pt, "year"):
        return np.array([c.point.year for c in cells])

    # Slow path: numeric points — convert via the coordinate's units
    try:
        dates = time_coord.units.num2date(time_coord.points)
        return np.array([d.year for d in dates])
    except Exception:
        return None


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

    cell_years = _get_cell_years(time_coord)

    if time is not None:
        # Select by year — may need to average across months
        if cell_years is None:
            raise ValueError(
                "Cannot select by year: the time coordinate has no "
                "calendar metadata.  Use 'time_index' instead."
            )
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
    year = int(cell_years[idx]) if cell_years is not None else None
    return data_2d, year


def _require_cartopy_iris(func_name):
    """Raise ImportError if cartopy or iris is not available."""
    if not HAS_CARTOPY:
        raise ImportError(
            f"cartopy is required for {func_name}(). "
            "Install it with: pip install cartopy"
        )
    if not HAS_IRIS:
        raise ImportError(
            f"iris is required for {func_name}(). "
            "Install it with: pip install scitools-iris"
        )


def _validate_map_inputs(cube, region, lon_bounds, lat_bounds, ax):
    """Common input validation for public map-plotting functions."""
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


def _squeeze_to_2d(data, cube, label="data"):
    """Squeeze length-1 extra dims and raise if result is not 2D."""
    data = np.squeeze(data)
    if data.ndim != 2:
        extra = [
            c.name() for c in cube.coords(dim_coords=True)
            if c.name() not in ("time", "latitude", "longitude")
        ]
        msg = (
            f"After time selection the {label} has {data.ndim} dimensions "
            f"but 2 (lat, lon) are required."
        )
        if extra:
            msg += (
                f"  The cube has extra dimension(s): {extra}.  "
                f"Slice or collapse them first."
            )
        raise ValueError(msg)
    return data


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
    _require_cartopy_iris("plot_spatial_map")
    _validate_map_inputs(cube, region, lon_bounds, lat_bounds, ax)

    # ---- time selection --------------------------------------------------
    data_2d, selected_year = _select_time_slice(cube, time=time, time_index=time_index)
    data_2d = _squeeze_to_2d(data_2d, cube)

    # ---- coordinate extraction -------------------------------------------
    lon_coord = cube.coord("longitude")
    lat_coord = cube.coord("latitude")
    lons = lon_coord.points
    lats = lat_coord.points

    # ---- auto-title ------------------------------------------------------
    if title is None:
        name = cube.name() or "field"
        year_str = f" ({selected_year})" if selected_year is not None else ""
        title = f"{name}{year_str}"

    cube_units = str(cube.units)

    return _render_map(
        data_2d=data_2d,
        lons=lons,
        lats=lats,
        cube_units=cube_units,
        region=region,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        projection=projection,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=title,
        units=units,
        add_coastlines=add_coastlines,
        add_gridlines=add_gridlines,
        colorbar=colorbar,
        ax=ax,
    )


def _render_map(
    data_2d,
    lons,
    lats,
    cube_units,
    region,
    lon_bounds,
    lat_bounds,
    projection,
    cmap,
    vmin,
    vmax,
    title,
    units,
    add_coastlines,
    add_gridlines,
    colorbar,
    ax,
):
    """Shared rendering logic: contourf + map decorations.

    Called by both :func:`plot_spatial_map` and :func:`plot_spatial_anomaly`.

    Returns ``(fig, ax)``.
    """
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

    # Close the longitude gap at 0°/360° for near-global grids.
    lon_range = float(lons[-1] - lons[0])
    if lon_range > 350:
        data_2d, lons = add_cyclic_point(data_2d, coord=lons)

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
        cbar_label = units if units is not None else cube_units
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal",
                            pad=0.05, shrink=0.7)
        cbar.set_label(cbar_label)

    # ---- title -----------------------------------------------------------
    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_spatial_anomaly(
    cube,
    time_a=None,
    time_index_a=None,
    time_b=None,
    time_index_b=None,
    region=None,
    lon_bounds=None,
    lat_bounds=None,
    projection=None,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    symmetric=True,
    title=None,
    units=None,
    add_coastlines=True,
    add_gridlines=True,
    colorbar=True,
    ax=None,
):
    """
    Plot the spatial difference between two time slices of an iris Cube.

    Computes ``data_a - data_b`` and renders the anomaly on a cartopy map.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube with time, latitude, and longitude dimensions.
    time_a : int, optional
        Year for the "after" field.  Default: last timestep.
    time_index_a : int, optional
        Positional index for the "after" field.
        Mutually exclusive with *time_a*.
    time_b : int, optional
        Year for the "before" / baseline field.  Default: first timestep.
    time_index_b : int, optional
        Positional index for the "before" field.
        Mutually exclusive with *time_b*.
    region : str, optional
        RECCAP2 region name (e.g. ``'Europe'``).
    lon_bounds : tuple of (float, float), optional
        ``(lon_min, lon_max)`` for the map extent.
    lat_bounds : tuple of (float, float), optional
        ``(lat_min, lat_max)`` for the map extent.
    projection : cartopy.crs.Projection, optional
        Map projection.  Default: Robinson for global, PlateCarree for regional.
    cmap : str or matplotlib.colors.Colormap, default ``'RdBu_r'``
        Colormap.  Diverging colormaps are recommended for anomaly maps.
    vmin : float, optional
        Lower bound for the colour scale.
    vmax : float, optional
        Upper bound for the colour scale.
    symmetric : bool, default True
        When *True* and *vmin*/*vmax* are not explicitly set, auto-centres
        the colour scale at zero so that ``vmin = -abs_max`` and
        ``vmax = +abs_max``.
    title : str, optional
        Plot title.  Auto-generated if *None*.
    units : str, optional
        Colorbar label.  Auto-detected from cube if *None*.
    add_coastlines : bool, default True
    add_gridlines : bool, default True
    colorbar : bool, default True
    ax : cartopy.mpl.geoaxes.GeoAxes, optional
        Pre-existing GeoAxes to plot on.

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
        If the cube has no time dimension, or if mutually exclusive
        arguments are both supplied.
    """
    _require_cartopy_iris("plot_spatial_anomaly")
    _validate_map_inputs(cube, region, lon_bounds, lat_bounds, ax)

    # Require a time dimension for anomaly computation
    time_coords = cube.coords("time", dim_coords=True)
    if len(time_coords) == 0:
        raise ValueError(
            "plot_spatial_anomaly() requires a cube with a time dimension, "
            "but the supplied cube has no time coordinate."
        )

    # ---- time selection --------------------------------------------------
    time_coord = time_coords[0]
    n_times = time_coord.shape[0]

    # "after" field — default to last timestep
    if time_a is None and time_index_a is None:
        time_index_a = n_times - 1
    data_a, year_a = _select_time_slice(
        cube, time=time_a, time_index=time_index_a,
    )

    # "before" / baseline field — default to first timestep
    if time_b is None and time_index_b is None:
        time_index_b = 0
    data_b, year_b = _select_time_slice(
        cube, time=time_b, time_index=time_index_b,
    )

    # ---- compute anomaly -------------------------------------------------
    anomaly = np.asarray(data_a, dtype=float) - np.asarray(data_b, dtype=float)
    anomaly = _squeeze_to_2d(anomaly, cube, label="anomaly")

    # ---- symmetric colour scale ------------------------------------------
    if symmetric and vmin is None and vmax is None:
        abs_max = float(np.nanmax(np.abs(anomaly)))
        if np.isfinite(abs_max) and abs_max > 0:
            vmin = -abs_max
            vmax = abs_max

    # ---- coordinate extraction -------------------------------------------
    lons = cube.coord("longitude").points
    lats = cube.coord("latitude").points

    # ---- auto-title ------------------------------------------------------
    if title is None:
        name = cube.name() or "field"
        if year_a is not None and year_b is not None:
            title = f"{name} anomaly ({year_a} \u2212 {year_b})"
        else:
            title = f"{name} anomaly"

    cube_units = str(cube.units)

    return _render_map(
        data_2d=anomaly,
        lons=lons,
        lats=lats,
        cube_units=cube_units,
        region=region,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        projection=projection,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=title,
        units=units,
        add_coastlines=add_coastlines,
        add_gridlines=add_gridlines,
        colorbar=colorbar,
        ax=ax,
    )


__all__ = [
    "plot_spatial_map",
    "plot_spatial_anomaly",
]

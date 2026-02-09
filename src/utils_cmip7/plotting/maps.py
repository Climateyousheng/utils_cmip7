"""
Geographic map plotting for 2D spatial fields.

Renders pre-extracted numpy arrays on cartopy map projections with
configurable regional subsetting (by RECCAP2 name or explicit bounds).

All functions accept matplotlib Axes objects for flexible composition.
No filesystem discovery, NetCDF I/O, or data extraction is performed
inside this module.

Use :func:`~utils_cmip7.processing.map_fields.extract_map_field` and
:func:`~utils_cmip7.processing.map_fields.extract_anomaly_field` to
prepare data for these plotting functions.
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

from ..config import get_region_bounds


def _require_cartopy(func_name):
    """Raise ImportError if cartopy is not available."""
    if not HAS_CARTOPY:
        raise ImportError(
            f"cartopy is required for {func_name}(). "
            "Install it with: pip install cartopy"
        )


def _validate_map_inputs(data, lons, lats, region, lon_bounds, lat_bounds, ax):
    """Common input validation for public map-plotting functions."""
    data = np.asarray(data)
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    if data.ndim != 2:
        raise ValueError(
            f"'data' must be a 2D array, got {data.ndim}D"
        )
    if lons.ndim != 1:
        raise ValueError(
            f"'lons' must be a 1D array, got {lons.ndim}D"
        )
    if lats.ndim != 1:
        raise ValueError(
            f"'lats' must be a 1D array, got {lats.ndim}D"
        )
    if data.shape != (len(lats), len(lons)):
        raise ValueError(
            f"'data' shape {data.shape} does not match "
            f"(len(lats), len(lons)) = ({len(lats)}, {len(lons)})"
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


def plot_spatial_map(
    data,
    lons,
    lats,
    *,
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
    Plot a 2D field on a cartopy map projection.

    Accepts pre-extracted numpy arrays (not iris Cubes).  Use
    :func:`~utils_cmip7.processing.map_fields.extract_map_field` to
    prepare data from an iris Cube.

    Parameters
    ----------
    data : array_like, shape (n_lat, n_lon)
        2D field to plot.
    lons : array_like, shape (n_lon,)
        Longitude values in degrees.
    lats : array_like, shape (n_lat,)
        Latitude values in degrees.
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
        Plot title.
    units : str, optional
        Colorbar label.
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
        If cartopy is not installed.
    ValueError
        If array shapes are inconsistent, or mutually exclusive
        arguments are both supplied.
    TypeError
        If *ax* is not a GeoAxes.
    """
    _require_cartopy("plot_spatial_map")
    _validate_map_inputs(data, lons, lats, region, lon_bounds, lat_bounds, ax)

    return _render_map(
        data_2d=np.asarray(data),
        lons=np.asarray(lons),
        lats=np.asarray(lats),
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


def plot_spatial_anomaly(
    data,
    lons,
    lats,
    *,
    region=None,
    lon_bounds=None,
    lat_bounds=None,
    projection=None,
    cmap="RdBu_r",
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
    Plot an anomaly field on a cartopy map projection.

    Thin wrapper around :func:`_render_map` with diverging defaults.
    Accepts pre-extracted numpy arrays (not iris Cubes).  Use
    :func:`~utils_cmip7.processing.map_fields.extract_anomaly_field` to
    prepare data from an iris Cube.

    Parameters
    ----------
    data : array_like, shape (n_lat, n_lon)
        2D anomaly field to plot.
    lons : array_like, shape (n_lon,)
        Longitude values in degrees.
    lats : array_like, shape (n_lat,)
        Latitude values in degrees.
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
    title : str, optional
        Plot title.
    units : str, optional
        Colorbar label.
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
        If cartopy is not installed.
    ValueError
        If array shapes are inconsistent, or mutually exclusive
        arguments are both supplied.
    TypeError
        If *ax* is not a GeoAxes.
    """
    _require_cartopy("plot_spatial_anomaly")
    _validate_map_inputs(data, lons, lats, region, lon_bounds, lat_bounds, ax)

    return _render_map(
        data_2d=np.asarray(data),
        lons=np.asarray(lons),
        lats=np.asarray(lats),
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
        cbar_label = units if units is not None else ""
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal",
                            pad=0.12, shrink=0.7, aspect=30)
        cbar.set_label(cbar_label)

    # ---- title -----------------------------------------------------------
    if title is not None:
        ax.set_title(title)

    return fig, ax


__all__ = [
    "plot_spatial_map",
    "plot_spatial_anomaly",
]

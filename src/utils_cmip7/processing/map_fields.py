"""
Extract 2D spatial fields from iris Cubes for map plotting.

Handles time selection, squeezing extra dimensions, coordinate
extraction, and auto-title generation.  The returned dicts are
ready to pass straight to :func:`~utils_cmip7.plotting.maps.plot_spatial_map`
or :func:`~utils_cmip7.plotting.maps.plot_spatial_anomaly`.
"""

import numpy as np

try:
    import iris
    import iris.cube
    HAS_IRIS = True
except ImportError:
    HAS_IRIS = False


def _require_iris(func_name):
    """Raise ImportError if iris is not available."""
    if not HAS_IRIS:
        raise ImportError(
            f"iris is required for {func_name}(). "
            "Install it with: pip install scitools-iris"
        )


def _get_cell_years(time_coord):
    """Return an array of integer years for every cell in *time_coord*.

    Works for both datetime-decoded and raw-numeric time coordinates.
    For numeric coordinates, converts via ``time_coord.units.num2date``.

    Returns *None* if year extraction is not possible (e.g. the time
    coordinate has no calendar metadata).
    """
    cells = list(time_coord.cells())
    first_pt = cells[0].point
    if hasattr(first_pt, "year"):
        return np.array([c.point.year for c in cells])

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

    time_coords = cube.coords("time", dim_coords=True)
    has_time = len(time_coords) > 0

    if not has_time:
        return cube.data, None

    time_coord = time_coords[0]
    time_dim = cube.coord_dims(time_coord)[0]
    cell_years = _get_cell_years(time_coord)

    if time is not None:
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
        idx = 0

    slices = [slice(None)] * cube.ndim
    slices[time_dim] = idx
    data_2d = cube[tuple(slices)].data
    year = int(cell_years[idx]) if cell_years is not None else None
    return data_2d, year


def _squeeze_to_2d(data, cube, label="data"):
    """Squeeze length-1 extra dims and raise if result is not 2D."""
    data = np.squeeze(data)
    if data.ndim != 2:
        # Identify extra dimensions by elimination (handles anonymous dims)
        known = {"time", "latitude", "longitude"}
        lat_dims = set(cube.coord_dims(cube.coord("latitude")))
        lon_dims = set(cube.coord_dims(cube.coord("longitude")))
        time_coords = cube.coords("time", dim_coords=True)
        time_dims = (
            {cube.coord_dims(tc)[0] for tc in time_coords}
            if time_coords
            else set()
        )
        known_dims = lat_dims | lon_dims | time_dims
        extra_dim_indices = sorted(set(range(cube.ndim)) - known_dims)

        # Build a list of names for extra dimensions
        extra_names = []
        for dim_idx in extra_dim_indices:
            coords = cube.coords(dimensions=dim_idx)
            if coords:
                extra_names.append(coords[0].name())
            else:
                extra_names.append(f"<anonymous dim {dim_idx}>")

        msg = (
            f"After time selection the {label} has {data.ndim} dimensions "
            f"but 2 (lat, lon) are required."
        )
        if extra_names:
            msg += (
                f"  The cube has extra dimension(s): {extra_names}.  "
                f"Pass level=<int> to select a single slice, "
                f"or slice/collapse the dimension first."
            )
        raise ValueError(msg)
    return data


def _masked_to_nan(data):
    """Convert a masked array to a float array with NaN for masked cells.

    If *data* is already an unmasked ndarray it is returned as float with
    no copy unless a dtype change is needed.
    """
    if isinstance(data, np.ma.MaskedArray):
        out = np.where(data.mask, np.nan, data.data).astype(float)
        return out
    return np.asarray(data, dtype=float)


def _select_level(data, cube, level):
    """Index into an extra (non-time, non-lat, non-lon) dimension.

    Finds the extra dimension by **elimination** — any cube dimension that
    is not latitude, longitude, or time is considered "extra".  This works
    regardless of whether the dimension has a DimCoord, an AuxCoord, or is
    completely anonymous.

    Parameters
    ----------
    data : numpy.ndarray
        Array after time selection, potentially 3-D+ (e.g. PFT, lat, lon).
    cube : iris.cube.Cube
        Original cube, used to identify which axis is the extra dimension.
    level : int
        Index along the extra dimension (e.g. PFT index 0–8).

    Returns
    -------
    numpy.ndarray
        Array with the extra dimension removed by indexing.

    Raises
    ------
    ValueError
        If no extra dimension is found or *level* is out of range.
    """
    # Identify cube dimension indices for lat, lon, time
    lat_dims = set(cube.coord_dims(cube.coord("latitude")))
    lon_dims = set(cube.coord_dims(cube.coord("longitude")))
    time_coords = cube.coords("time", dim_coords=True)
    time_dims = (
        {cube.coord_dims(tc)[0] for tc in time_coords}
        if time_coords
        else set()
    )

    known_dims = lat_dims | lon_dims | time_dims
    extra_dims = sorted(set(range(cube.ndim)) - known_dims)

    if not extra_dims:
        raise ValueError(
            "No extra (non-time, non-lat, non-lon) dimension found "
            "to apply 'level' selection."
        )

    extra_cube_dim = extra_dims[0]

    # After time selection the time dimension was removed from *data*.
    # Adjust the axis index accordingly.
    if time_dims and extra_cube_dim > min(time_dims):
        axis_in_data = extra_cube_dim - 1
    else:
        axis_in_data = extra_cube_dim

    n = data.shape[axis_in_data]
    if level < 0 or level >= n:
        # Try to find a name for the dimension for the error message
        dim_name = "unknown"
        for coord in cube.coords(dimensions=extra_cube_dim):
            dim_name = coord.name()
            break
        raise ValueError(
            f"level={level} is out of range for dimension "
            f"'{dim_name}' with size {n}."
        )

    slices = [slice(None)] * data.ndim
    slices[axis_in_data] = level
    return data[tuple(slices)]


def extract_map_field(cube, time=None, time_index=None, variable=None, level=None):
    """Extract a 2D spatial field from an iris Cube.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube with latitude and longitude dimensions (and optional
        time dimension).
    time : int, optional
        Year to select.  Mutually exclusive with *time_index*.
    time_index : int, optional
        Positional index along the time dimension.
        Mutually exclusive with *time*.
    variable : str, optional
        Canonical variable name or alias (e.g. ``'GPP'``, ``'VegCarb'``).
        When provided, applies the ``conversion_factor`` and overrides
        ``units`` and ``name`` from :data:`~utils_cmip7.config.CANONICAL_VARIABLES`.
        No conversion is applied by default.
    level : int, optional
        Index along an extra (non-time, non-lat, non-lon) dimension.
        For cubes with a PFT / pseudo-level dimension (e.g. ``frac``),
        this selects a single level before squeezing to 2D.

    Returns
    -------
    dict
        Keys: ``'data'`` (2D ndarray), ``'lons'`` (1D ndarray),
        ``'lats'`` (1D ndarray), ``'name'`` (str), ``'units'`` (str),
        ``'year'`` (int or None), ``'title'`` (str).

    Raises
    ------
    ImportError
        If iris is not installed.
    TypeError
        If *cube* is not an iris Cube.
    ValueError
        If mutually exclusive arguments are both supplied, if the
        requested year is not found, if *variable* is not recognised,
        or if *level* is out of range.
    """
    _require_iris("extract_map_field")
    if not isinstance(cube, iris.cube.Cube):
        raise TypeError(
            f"Expected an iris.cube.Cube, got {type(cube).__name__}"
        )

    data_2d, year = _select_time_slice(cube, time=time, time_index=time_index)
    if level is not None:
        data_2d = _select_level(data_2d, cube, level)
    data_2d = _squeeze_to_2d(data_2d, cube)
    data_2d = _masked_to_nan(data_2d)

    lons = cube.coord("longitude").points
    lats = cube.coord("latitude").points

    if variable is not None:
        from ..config import get_variable_config
        var_config = get_variable_config(variable)
        data_2d = data_2d * var_config["conversion_factor"]
        units = var_config["units"]
        name = var_config["canonical_name"]
    else:
        name = cube.name() or "field"
        units = str(cube.units)

    year_str = f" ({year})" if year is not None else ""
    title = f"{name}{year_str}"

    return {
        "data": data_2d,
        "lons": lons,
        "lats": lats,
        "name": name,
        "units": units,
        "year": year,
        "title": title,
    }


def extract_anomaly_field(
    cube,
    time_a=None,
    time_index_a=None,
    time_b=None,
    time_index_b=None,
    symmetric=True,
    variable=None,
    level=None,
):
    """Extract anomaly (data_a - data_b) between two time slices.

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
    symmetric : bool, default True
        When *True*, auto-compute symmetric vmin/vmax centred at zero.
    variable : str, optional
        Canonical variable name or alias (e.g. ``'GPP'``, ``'VegCarb'``).
        When provided, applies the ``conversion_factor`` to each slice
        before computing the anomaly, and overrides ``units`` and ``name``
        from :data:`~utils_cmip7.config.CANONICAL_VARIABLES`.
        No conversion is applied by default.
    level : int, optional
        Index along an extra (non-time, non-lat, non-lon) dimension.
        Applied to both slices before computing the anomaly.

    Returns
    -------
    dict
        Keys: ``'data'`` (2D ndarray), ``'lons'`` (1D ndarray),
        ``'lats'`` (1D ndarray), ``'name'`` (str), ``'units'`` (str),
        ``'year_a'`` (int or None), ``'year_b'`` (int or None),
        ``'vmin'`` (float or None), ``'vmax'`` (float or None),
        ``'title'`` (str).

    Raises
    ------
    ImportError
        If iris is not installed.
    TypeError
        If *cube* is not an iris Cube.
    ValueError
        If the cube has no time dimension, if mutually exclusive
        arguments are both supplied, if *variable* is not recognised,
        or if *level* is out of range.
    """
    _require_iris("extract_anomaly_field")
    if not isinstance(cube, iris.cube.Cube):
        raise TypeError(
            f"Expected an iris.cube.Cube, got {type(cube).__name__}"
        )

    time_coords = cube.coords("time", dim_coords=True)
    if len(time_coords) == 0:
        raise ValueError(
            "extract_anomaly_field() requires a cube with a time dimension, "
            "but the supplied cube has no time coordinate."
        )

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

    # Select level from extra dimension (e.g. PFT) before squeeze
    if level is not None:
        data_a = _select_level(data_a, cube, level)
        data_b = _select_level(data_b, cube, level)

    data_a = _masked_to_nan(data_a)
    data_b = _masked_to_nan(data_b)

    # Apply unit conversion to each slice before subtraction
    if variable is not None:
        from ..config import get_variable_config
        var_config = get_variable_config(variable)
        factor = var_config["conversion_factor"]
        data_a = data_a * factor
        data_b = data_b * factor

    anomaly = data_a - data_b
    anomaly = _squeeze_to_2d(anomaly, cube, label="anomaly")

    vmin = None
    vmax = None
    if symmetric:
        abs_max = float(np.nanmax(np.abs(anomaly)))
        if np.isfinite(abs_max) and abs_max > 0:
            vmin = -abs_max
            vmax = abs_max

    lons = cube.coord("longitude").points
    lats = cube.coord("latitude").points

    if variable is not None:
        units = var_config["units"]
        name = var_config["canonical_name"]
    else:
        name = cube.name() or "field"
        units = str(cube.units)

    if year_a is not None and year_b is not None:
        title = f"{name} anomaly ({year_a} \u2212 {year_b})"
    else:
        title = f"{name} anomaly"

    return {
        "data": anomaly,
        "lons": lons,
        "lats": lats,
        "name": name,
        "units": units,
        "year_a": year_a,
        "year_b": year_b,
        "vmin": vmin,
        "vmax": vmax,
        "title": title,
    }


_OP_SYMBOLS = {
    "sum": "+",
    "mean": "+",
    "subtract": "\u2212",
    "multiply": "\u00d7",
    "divide": "/",
}

_NARY_OPS = {"sum", "mean"}
_BINARY_OPS = {"subtract", "multiply", "divide"}


def combine_fields(fields, operation="sum", name=None, units=None):
    """Combine multiple extracted fields element-wise.

    Parameters
    ----------
    fields : list of dict
        Each dict from :func:`extract_map_field` with ``'data'``,
        ``'lons'``, ``'lats'``, ``'name'``, ``'units'``, ``'year'``.
    operation : str, default ``'sum'``
        N-ary operations: ``'sum'``, ``'mean'``.
        Binary operations (exactly 2 fields): ``'subtract'``,
        ``'multiply'``, ``'divide'``.
    name : str, optional
        Override name for the combined field.
        Auto-generated if *None* (e.g. ``"GPP + NPP"``).
    units : str, optional
        Override units for the combined field.
        Inherited from first field if *None*.

    Returns
    -------
    dict
        Keys: ``'data'``, ``'lons'``, ``'lats'``, ``'name'``,
        ``'units'``, ``'year'``.

    Raises
    ------
    ValueError
        If *fields* is empty, grids don't match, *operation* is unknown,
        or a binary operation is given with != 2 fields.
    """
    if not fields:
        raise ValueError("'fields' must be a non-empty list of field dicts.")

    all_ops = _NARY_OPS | _BINARY_OPS
    if operation not in all_ops:
        raise ValueError(
            f"Unknown operation '{operation}'. "
            f"Supported: {sorted(all_ops)}"
        )

    if operation in _BINARY_OPS and len(fields) != 2:
        raise ValueError(
            f"Binary operation '{operation}' requires exactly 2 fields, "
            f"got {len(fields)}."
        )

    ref_lons = np.asarray(fields[0]["lons"])
    ref_lats = np.asarray(fields[0]["lats"])
    for i, f in enumerate(fields[1:], 1):
        if not (np.array_equal(np.asarray(f["lons"]), ref_lons)
                and np.array_equal(np.asarray(f["lats"]), ref_lats)):
            raise ValueError(
                f"Grid mismatch: field 0 and field {i} have different "
                f"lon/lat coordinates."
            )

    arrays = [np.asarray(f["data"], dtype=float) for f in fields]
    symbol = _OP_SYMBOLS[operation]

    if operation == "sum":
        result = sum(arrays)
    elif operation == "mean":
        result = sum(arrays) / len(arrays)
    elif operation == "subtract":
        result = arrays[0] - arrays[1]
    elif operation == "multiply":
        result = arrays[0] * arrays[1]
    elif operation == "divide":
        result = arrays[0] / arrays[1]

    if name is None:
        field_names = [f.get("name", "field") for f in fields]
        name = f" {symbol} ".join(field_names)

    if units is None:
        units = fields[0].get("units", "")

    year = fields[0].get("year")

    return {
        "data": result,
        "lons": ref_lons,
        "lats": ref_lats,
        "name": name,
        "units": units,
        "year": year,
    }


__all__ = [
    "extract_map_field",
    "extract_anomaly_field",
    "combine_fields",
]

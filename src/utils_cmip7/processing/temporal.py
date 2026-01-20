"""
Temporal aggregation functions for monthly and annual means.

Provides functions to convert monthly data to annual means and handle
fractional year formats used in TRIFFID variables.
"""

import numpy as np
import pandas as pd
import cf_units

from .spatial import global_total_pgC, global_mean_pgC


def merge_monthly_results(results, require_full_year=False):
    """
    Merge multiple monthly outputs into annual mean time series.

    Takes multiple dictionaries with fractional year data (output from
    compute_monthly_mean) and merges them into a single annual mean series.

    Parameters
    ----------
    results : list of dict
        Each dict has keys: 'years' (fractional years) and 'data'.
        Output format from compute_monthly_mean().
    require_full_year : bool, default False
        If True, only return years with all 12 months present.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'years': integer years as numpy array
        - 'data': annual mean values as numpy array

    Examples
    --------
    >>> # Merge results from multiple files
    >>> results = [compute_monthly_mean(cube1, 'GPP'),
    ...            compute_monthly_mean(cube2, 'GPP')]
    >>> annual = merge_monthly_results(results, require_full_year=True)
    >>> annual['years']  # Array of years with complete data
    >>> annual['data']   # Corresponding annual means

    Notes
    -----
    - Handles duplicate (year, month) entries by averaging
    - Reconstructs integer year and month from fractional years
    - Groups by year and computes mean across months
    """
    all_years, all_data = [], []
    for r in results:
        all_years.extend(r["years"])
        all_data.extend(r["data"])

    df = pd.DataFrame({"year_frac": all_years, "value": all_data})

    # Reconstruct integer year and month
    df["year"] = df["year_frac"].astype(int)
    df["month"] = np.round((df["year_frac"] - df["year"]) * 12).astype(int) + 1
    df.loc[df["month"] < 1, "month"] = 1
    df.loc[df["month"] > 12, "month"] = 12

    # 1) Average duplicates at (year, month)
    monthly = df.groupby(["year", "month"], as_index=False)["value"].mean()

    # 2) Annual mean across months
    annual = monthly.groupby("year")["value"].agg(["mean", "count"]).reset_index()

    if require_full_year:
        annual = annual[annual["count"] == 12]

    return {
        "years": annual["year"].to_numpy(dtype=int),
        "data": annual["mean"].to_numpy(),
    }


def compute_monthly_mean(cube, var):
    """
    Compute area-weighted global total for each month.

    Computes global total with area weighting, then extracts monthly
    values in fractional year format. Used for TRIFFID variables.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube with time, latitude, longitude dimensions
    var : str
        Variable name for unit conversion (e.g., 'GPP', 'NPP')

    Returns
    -------
    dict
        Dictionary with keys:
        - 'years': fractional years (e.g., 1850.083 for Feb 1850)
        - 'data': monthly values in converted units
        - 'name': variable long name
        - 'units': units string

    Raises
    ------
    ValueError
        If no valid time coordinate found in cube

    Examples
    --------
    >>> gpp_cube = iris.load_cube('gpp.nc')
    >>> monthly = compute_monthly_mean(gpp_cube, 'GPP')
    >>> monthly['years']  # Fractional years
    >>> monthly['data']   # Monthly GPP values in PgC/year

    Notes
    -----
    - Handles 360-day calendars
    - Automatically sets calendar if missing or 'unknown'
    - Averages duplicate (year, month) entries
    - Returns fractional years: year + (month - 1) / 12
    """
    gm = global_total_pgC(cube, var)

    # --- Get time coordinate robustly ---
    time_coords = [c for c in gm.coords() if c.standard_name == 'time' or c.name() in ('t', 'time', 'TIME')]
    if not time_coords:
        raise ValueError("❌ No valid time coordinate found in cube.")
    tcoord = time_coords[0]

    # --- Ensure calendar and units are valid ---
    if not getattr(tcoord.units, "calendar", None):
        tcoord.units = cf_units.Unit("days since 1850-12-01 00:00:00", calendar="360_day")
    elif str(tcoord.units).startswith("unknown"):
        tcoord.units = cf_units.Unit("days since 1850-12-01 00:00:00", calendar="360_day")

    # --- Convert time → datetimes ---
    times = tcoord.units.num2date(tcoord.points)

    # --- Build DataFrame to group by year/month ---
    df = pd.DataFrame({
        "year": [t.year for t in times],
        "month": [t.month for t in times],
        "value": gm.data
    })

    # --- Average duplicates within each (year, month) ---
    df_monthly = df.groupby(["year", "month"], as_index=False)["value"].mean()

    # --- Convert to fractional year ---
    df_monthly["year_frac"] = df_monthly["year"] + (df_monthly["month"] - 1) / 12

    return {
        "years": df_monthly['year_frac'].to_numpy(),
        "data": df_monthly['value'].to_numpy(),
        "name": cube.long_name or cube.standard_name or var,
        "units": str(gm.units)
    }


def compute_annual_mean(cube, var):
    """
    Compute area-weighted annual means from monthly data.

    Aggregates monthly data to annual means, handling both extensive
    (sum) and intensive (mean) variables.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube with time, latitude, longitude dimensions
    var : str
        Variable name for unit conversion. Special case: 'Others' uses MEAN.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'years': unique years as numpy array
        - 'data': annual mean values
        - 'name': variable long name
        - 'units': units string

    Raises
    ------
    ValueError
        If no valid time coordinate found in cube

    Examples
    --------
    >>> gpp_cube = iris.load_cube('gpp.nc')
    >>> annual = compute_annual_mean(gpp_cube, 'GPP')
    >>> annual['years']  # Array of years
    >>> annual['data']   # Annual GPP values in PgC/year

    Notes
    -----
    - Uses global_mean_pgC for var='Others', global_total_pgC otherwise
    - Handles 360-day calendars
    - Automatically sets calendar if missing
    - Handles scalar (0D) data by repeating value for each year
    - Uses np.nanmean to handle missing data
    """
    if var == 'Others':
        gm = global_mean_pgC(cube, var)
    else:
        gm = global_total_pgC(cube, var)

    # --- Get time coordinate robustly ---
    time_coords = [c for c in gm.coords() if c.standard_name == 'time' or c.name() in ('t', 'time', 'TIME')]
    if not time_coords:
        raise ValueError("❌ No valid time coordinate found in cube.")
    tcoord = time_coords[0]

    # --- Ensure calendar and units are valid ---
    if not getattr(tcoord.units, "calendar", None):
        tcoord.units = cf_units.Unit("days since 1850-12-01 00:00:00", calendar="360_day")
    elif str(tcoord.units).startswith("unknown"):
        tcoord.units = cf_units.Unit("days since 1850-12-01 00:00:00", calendar="360_day")

    # --- Convert time → datetimes & extract years ---
    times = tcoord.units.num2date(tcoord.points)
    years = np.array([t.year for t in times])

    # --- Compute annual mean ---
    unique_years, idx = np.unique(years, return_inverse=True)
    if gm.data.ndim == 0:  # scalar (no time dimension)
        annual_means = np.repeat(gm.data.item(), len(unique_years))
    else:
        annual_means = np.array([
            np.nanmean(gm.data[idx == i]) for i in range(len(unique_years))
        ])

    return {
        "years": unique_years,
        "data": annual_means,
        "name": cube.long_name or cube.standard_name or var,
        "units": str(gm.units)
    }

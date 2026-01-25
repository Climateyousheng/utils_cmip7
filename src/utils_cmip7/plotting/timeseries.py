"""
Time series plotting functions for carbon cycle analysis.

All functions accept matplotlib Axes objects for flexible composition.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from .styles import DEFAULT_LEGEND_LABELS, DEFAULT_COLOR_MAP, group_vars_by_prefix


def plot_timeseries_grouped(
    data,
    expts_list,
    region,
    outdir=None,
    legend_labels=None,
    color_map=None,
    show=False,
    exclude=("fracPFTs", "frac"),
    ncols=3,
    ax=None,
):
    """
    Plot time series of all variables grouped by prefix.

    Creates a multi-panel figure with time series for all variables in the
    specified region. Variables are grouped by prefix (e.g., all GPP_ variables
    together). Excludes nested PFT structures by default.

    Parameters
    ----------
    data : dict
        Nested dictionary: dict[expt][region][var] -> {"years":..., "data":..., "units":...}
    expts_list : list of str
        List of experiment names to plot
    region : str
        Region name (e.g., 'global', 'Europe', 'North_America')
    outdir : str, optional
        Output directory for saved figure. If None and ax is None, does not save.
    legend_labels : dict, optional
        Custom labels for experiments {expt: label}
    color_map : dict, optional
        Custom colors for experiments {expt: color}
    show : bool, default=False
        Whether to display the figure interactively
    exclude : tuple, default=('fracPFTs', 'frac')
        Variable prefixes to exclude (nested structures)
    ncols : int, default=3
        Number of columns in subplot grid
    ax : matplotlib.axes.Axes or array-like, optional
        Pre-existing axes to plot on. If provided, outdir and show are ignored.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of Axes objects

    Notes
    -----
    Excludes 'fracPFTs' and 'frac' by default (nested PFT structure).
    Skips first year in time series (spinup convention).
    """
    legend_labels = legend_labels or DEFAULT_LEGEND_LABELS
    color_map = color_map or DEFAULT_COLOR_MAP

    grouped = group_vars_by_prefix(
        data, expts_list=expts_list, region=region, exclude=exclude
    )

    # Flatten all grouped variables for this region into one figure
    all_varnames = [var for group in grouped.values() for var in group]
    n_vars = len(all_varnames)

    if n_vars == 0:
        return None, None

    # Create figure if axes not provided
    if ax is None:
        nrows = (n_vars + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), sharex=True, dpi=100
        )
        axes = np.atleast_1d(axes).flatten()
        own_figure = True
    else:
        axes = np.atleast_1d(ax).flatten()
        fig = axes[0].get_figure() if len(axes) > 0 else None
        own_figure = False

    for ax_idx, var in enumerate(all_varnames):
        if ax_idx >= len(axes):
            break

        current_ax = axes[ax_idx]
        any_series = None  # capture one valid series for units/title

        for exp in expts_list:
            series = data.get(exp, {}).get(region, {}).get(var)
            if not series:
                continue

            # Check if series has the expected structure
            if not isinstance(series, dict):
                continue
            if "years" not in series or "data" not in series:
                # Skip variables with nested structure
                continue

            any_series = series
            label = legend_labels.get(exp, exp)

            try:
                years = np.asarray(series["years"])
                vals = np.asarray(series["data"])

                # Skip first year (spinup convention)
                current_ax.plot(
                    years[1:],
                    vals[1:],
                    label=label,
                    color=color_map.get(exp, "0.5"),
                    lw=0.8,
                )
            except (KeyError, IndexError, ValueError) as e:
                print(f"Warning: Skipping {var} for {exp}: {e}")
                continue

        units = (any_series or {}).get("units", "")
        title = f"{var} ({units}) {region}" if units else var
        current_ax.set_title(title, fontsize=9)
        current_ax.set_ylabel(units or "Value", fontsize=8)
        current_ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        current_ax.tick_params(labelsize=7)
        current_ax.legend(frameon=False, fontsize=6, loc="upper left")

    # Remove unused subplots
    if own_figure:
        for k in range(len(all_varnames), len(axes)):
            fig.delaxes(axes[k])

    if len(all_varnames) > 0 and len(axes) > 0:
        axes[0].set_xlabel("Year", fontsize=9)

    if own_figure:
        plt.tight_layout()

        # Save if outdir provided
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(
                outdir, f"allvars_{region}_{'_'.join(expts_list)}_timeseries.png"
            )
            plt.savefig(outpath, dpi=300)

        if show:
            plt.show()
        elif not outdir and not show:
            # Close if not saving or showing
            plt.close(fig)

    return fig, axes


def plot_pft_timeseries(
    data,
    expts_list,
    region,
    outdir=None,
    legend_labels=None,
    color_map=None,
    show=False,
    pfts=(1, 2, 3, 4, 5),
    ax=None,
):
    """
    Plot PFT fraction time series for one region.

    Creates one figure per region containing subplots for each PFT (1-5 by default),
    with multiple experiments overlaid in each subplot.

    Parameters
    ----------
    data : dict
        Nested dict: dict[expt][region]["fracPFTs"]["PFT n"] -> {"years","data","units",...}
    expts_list : list of str
        List of experiment names to plot
    region : str
        Region name
    outdir : str, optional
        Output directory for saved figure. If None and ax is None, does not save.
    legend_labels : dict, optional
        Custom labels for experiments
    color_map : dict, optional
        Custom colors for experiments
    show : bool, default=False
        Whether to display the figure interactively
    pfts : tuple, default=(1, 2, 3, 4, 5)
        PFT indices to plot
    ax : matplotlib.axes.Axes or array-like, optional
        Pre-existing axes to plot on. If provided, outdir and show are ignored.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of Axes objects
    """
    legend_labels = legend_labels or DEFAULT_LEGEND_LABELS
    color_map = color_map or DEFAULT_COLOR_MAP

    # Create figure if axes not provided
    if ax is None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, dpi=100)
        axes = axes.flatten()
        own_figure = True
    else:
        axes = np.atleast_1d(ax).flatten()
        fig = axes[0].get_figure() if len(axes) > 0 else None
        own_figure = False

    # Plot first 5 PFTs (use first 5 axes, delete 6th if own figure)
    for ax_idx, p in enumerate(pfts):
        if ax_idx >= len(axes):
            break

        current_ax = axes[ax_idx]
        pft_key = f"PFT {p}"
        any_series = None

        for exp in expts_list:
            series = (
                data.get(exp, {})
                .get(region, {})
                .get("fracPFTs", {})
                .get(pft_key)
            )
            if not series:
                continue

            any_series = series
            years = np.asarray(series["years"])
            vals = np.asarray(series["data"])
            label = legend_labels.get(exp, exp)

            # Skip first year (spinup convention)
            current_ax.plot(
                years[1:],
                vals[1:],
                label=label,
                color=color_map.get(exp, "0.5"),
                lw=0.9,
            )

        units = (any_series or {}).get("units", "")
        title = f"{pft_key} ({units})" if units else pft_key
        current_ax.set_title(title, fontsize=9)
        current_ax.set_ylabel(units or "Fraction", fontsize=8)
        current_ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        current_ax.tick_params(labelsize=7)
        current_ax.legend(frameon=False, fontsize=6, loc="upper left")

    # Remove extra subplot (6th) if own figure
    if own_figure and len(axes) > len(pfts):
        fig.delaxes(axes[len(pfts)])

    if len(axes) > 0:
        axes[0].set_xlabel("Year", fontsize=9)

    if own_figure:
        plt.suptitle(f"PFT fractions (first 5) – {region}", fontsize=11, y=1.02)
        plt.tight_layout()

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(outdir, f"fracPFTs_{region}_timeseries.png")
            plt.savefig(outpath, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        elif not outdir and not show:
            plt.close(fig)

    return fig, axes


__all__ = [
    'plot_timeseries_grouped',
    'plot_pft_timeseries',
]

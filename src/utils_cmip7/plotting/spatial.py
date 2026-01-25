"""
Spatial distribution plotting functions (regional pie charts, bars).

All functions accept matplotlib Axes objects for flexible composition.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from .styles import DEFAULT_LEGEND_LABELS, DEFAULT_COLOR_MAP


def plot_regional_pie(
    data,
    varname,
    expt,
    year,
    outdir=None,
    legend_labels=None,
    show=False,
    ax=None,
):
    """
    Plot pie chart of a variable across regions for one experiment and year.

    Parameters
    ----------
    data : dict
        Nested dictionary: dict[expt][region][var] -> series dict
    varname : str
        Variable name (e.g., 'soilResp', 'GPP')
    expt : str
        Experiment name
    year : int
        Year to plot
    outdir : str, optional
        Output directory for saved figure. If None and ax is None, does not save.
    legend_labels : dict, optional
        Custom labels for experiments {expt: label}
    show : bool, default=False
        Whether to display the figure interactively
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to plot on. If provided, outdir and show are ignored.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object

    Raises
    ------
    ValueError
        If no data found for experiment or no valid values for the specified year
    """
    legend_labels = legend_labels or DEFAULT_LEGEND_LABELS

    expt_block = data.get(expt, {})
    if not expt_block:
        raise ValueError(f"No data found for experiment '{expt}'")

    labels, values = [], []
    units = None

    for region, region_block in expt_block.items():
        series = region_block.get(varname)
        if not series:
            continue

        years = np.asarray(series["years"])
        vals = np.asarray(series["data"])
        if year not in years:
            continue

        idx = np.where(years == year)[0][0]
        val = vals[idx]
        if val is None or np.isnan(val):
            continue

        units = units or series.get("units", "")
        labels.append(region.replace("_", " "))
        values.append(val)

    if not values:
        raise ValueError(
            f"No valid values found for {varname} in year {year} for {expt}"
        )

    # Calculate percentages
    total = float(np.sum(values))
    labels_with_values = [
        f"{lab}\n{v:.2f} ({(v/total*100):.1f}%)" for lab, v in zip(labels, values)
    ]

    # Create figure if axes not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        own_figure = True
    else:
        fig = ax.get_figure()
        own_figure = False

    ax.pie(
        values,
        startangle=90,
        counterclock=False,
        wedgeprops={'width': 0.45},
        labels=labels_with_values,
        textprops={'color': "black", 'fontsize': 8},
    )

    title_label = legend_labels.get(expt, expt)
    unit_str = f" ({units})" if units else ""
    ax.set_title(
        f"{varname}{unit_str} distribution in {year}\n{title_label}", fontsize=10
    )

    if own_figure:
        plt.tight_layout()

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(
                outdir, f"{varname}_regional_pie_{expt}_{year}.png"
            )
            plt.savefig(outpath, dpi=300)

        if show:
            plt.show()
        elif not outdir and not show:
            plt.close(fig)

    return fig, ax


def plot_regional_pies(
    data,
    varname,
    expts_list,
    year,
    outdir=None,
    legend_labels=None,
    show=False,
    ax=None,
):
    """
    Plot side-by-side pie charts for multiple experiments.

    Shows regional distribution of a variable across multiple experiments
    in a given year.

    Parameters
    ----------
    data : dict
        Nested dictionary: dict[expt][region][var] -> series dict
    varname : str
        Variable name (e.g., 'soilResp')
    expts_list : list of str
        List of experiment names
    year : int
        Year to plot
    outdir : str, optional
        Output directory for saved figure. If None and ax is None, does not save.
    legend_labels : dict, optional
        Custom labels for experiments
    show : bool, default=False
        Whether to display the figure interactively
    ax : array-like of matplotlib.axes.Axes, optional
        Pre-existing axes to plot on. If provided, outdir and show are ignored.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of Axes objects
    """
    legend_labels = legend_labels or DEFAULT_LEGEND_LABELS

    n = len(expts_list)

    # Create figure if axes not provided
    if ax is None:
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7), squeeze=False, dpi=100)
        axes = axes[0]
        own_figure = True
    else:
        axes = np.atleast_1d(ax).flatten()
        fig = axes[0].get_figure() if len(axes) > 0 else None
        own_figure = False

    for ax_idx, expt in enumerate(expts_list):
        if ax_idx >= len(axes):
            break

        current_ax = axes[ax_idx]
        expt_block = data.get(expt, {})

        if not expt_block:
            current_ax.set_title(f"{expt}\n(no data)")
            current_ax.axis("off")
            continue

        labels, values = [], []
        units = None

        for region, region_block in expt_block.items():
            series = region_block.get(varname)
            if not series:
                continue

            years = np.asarray(series["years"])
            vals = np.asarray(series["data"])

            if year not in years:
                continue

            idx = np.where(years == year)[0][0]
            val = vals[idx]
            if val is None or np.isnan(val):
                continue

            units = units or series.get("units", "")
            labels.append(region.replace("_", " "))
            values.append(val)

        if not values:
            current_ax.set_title(f"{legend_labels.get(expt, expt)}\n(no valid data)")
            current_ax.axis("off")
            continue

        total = float(np.sum(values))
        labels_with_values = [
            f"{lab}\n{v:.2f} ({(v / total * 100):.1f}%)"
            for lab, v in zip(labels, values)
        ]

        current_ax.pie(
            values,
            startangle=90,
            counterclock=False,
            wedgeprops={'width': 0.45},
            labels=labels_with_values,
            textprops={'color': "black", 'fontsize': 8},
        )

        title_label = legend_labels.get(expt, expt)
        unit_str = f" ({units})" if units else ""
        current_ax.set_title(f"{title_label}\n{varname}{unit_str}\n{year}", fontsize=10)

    if own_figure:
        plt.tight_layout()

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(outdir, f"{varname}_regional_pies_{year}.png")
            plt.savefig(outpath, dpi=300)

        if show:
            plt.show()
        elif not outdir and not show:
            plt.close(fig)

    return fig, axes


def plot_pft_grouped_bars(
    data,
    expts_list,
    year,
    outdir=None,
    legend_labels=None,
    color_map=None,
    pfts=(1, 2, 3, 4, 5),
    show=False,
    ax=None,
):
    """
    Plot grouped bar charts for PFT fractions by region.

    Creates one figure with subplots for PFT 1-5. Each subplot shows all regions
    on x-axis with grouped bars for different experiments.

    Parameters
    ----------
    data : dict
        Nested dict: dict[expt][region]["fracPFTs"]["PFT n"] -> series dict
    expts_list : list of str
        List of experiment names
    year : int
        Year to plot
    outdir : str, optional
        Output directory for saved figure
    legend_labels : dict, optional
        Custom labels for experiments
    color_map : dict, optional
        Custom colors for experiments
    pfts : tuple, default=(1, 2, 3, 4, 5)
        PFT indices to plot
    show : bool, default=False
        Whether to display the figure interactively
    ax : array-like of matplotlib.axes.Axes, optional
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

    # Determine consistent region ordering (deterministic sort)
    all_regions = sorted(
        {region for expt in expts_list for region in data.get(expt, {}).keys()}
    )

    n_regions = len(all_regions)
    n_expts = len(expts_list)
    bar_width = 0.8 / n_expts
    x = np.arange(n_regions)

    # Create figure if axes not provided
    if ax is None:
        fig, axes = plt.subplots(
            2, 3, figsize=(1.2 * n_regions + 6, 7), dpi=100
        )
        axes = axes.flatten()
        own_figure = True
    else:
        axes = np.atleast_1d(ax).flatten()
        fig = axes[0].get_figure() if len(axes) > 0 else None
        own_figure = False

    for ax_idx, p in enumerate(pfts):
        if ax_idx >= len(axes):
            break

        current_ax = axes[ax_idx]
        pft_key = f"PFT {p}"

        for i, expt in enumerate(expts_list):
            values = []

            for region in all_regions:
                series = (
                    data.get(expt, {})
                    .get(region, {})
                    .get("fracPFTs", {})
                    .get(pft_key)
                )

                if not series:
                    values.append(np.nan)
                    continue

                years = np.asarray(series["years"])
                vals = np.asarray(series["data"])
                if year not in years:
                    values.append(np.nan)
                    continue

                idx = np.where(years == year)[0][0]
                values.append(vals[idx])

            current_ax.bar(
                x + i * bar_width,
                values,
                width=bar_width,
                label=legend_labels.get(expt, expt),
                color=color_map.get(expt, None),
            )

        current_ax.set_title(pft_key, fontsize=10)
        current_ax.set_ylim(0, 1)
        current_ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.5)

        current_ax.set_xticks(x + bar_width * (n_expts - 1) / 2)
        current_ax.set_xticklabels(
            [r.replace("_", " ") for r in all_regions],
            rotation=45,
            ha="right",
            fontsize=8,
        )

    # Remove unused subplot (6th) if own figure
    if own_figure:
        for k in range(len(pfts), len(axes)):
            fig.delaxes(axes[k])

    # Shared labels and legend
    if len(axes) > 0:
        axes[0].set_ylabel("Fraction", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()

        if own_figure:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(expts_list),
                frameon=False,
                fontsize=9,
            )
            fig.suptitle(f"PFT fractions by region ({year})", fontsize=12, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.93])

            if outdir:
                os.makedirs(outdir, exist_ok=True)
                outpath = os.path.join(outdir, f"fracPFTs_1to5_grouped_bars_{year}.png")
                plt.savefig(outpath, dpi=300, bbox_inches="tight")

            if show:
                plt.show()
            elif not outdir and not show:
                plt.close(fig)

    return fig, axes


__all__ = [
    'plot_regional_pie',
    'plot_regional_pies',
    'plot_pft_grouped_bars',
]

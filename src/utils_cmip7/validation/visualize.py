"""
Visualization functions for metric validation.

Plots UM vs observational comparisons, regional differences, and time series.
NO NetCDF loading, aggregation, or metric computation permitted.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Any


def plot_metric_comparison(
    comparison: Dict[str, Dict[str, Any]],
    metric: str,
    ax: Optional[plt.Axes] = None,
    outdir: Optional[str] = None,
    filename: Optional[str] = None
) -> plt.Axes:
    """
    Plot UM vs obs for a specific metric across regions.

    Bar chart with error bars showing UM mean and observational values.

    Parameters
    ----------
    comparison : dict
        Comparison results for single metric: {region: comparison_dict}
    metric : str
        Metric name for labeling
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    outdir : str, optional
        Output directory for saving plot
    filename : str, optional
        Filename for saving. Default: {metric}_comparison.png

    Returns
    -------
    plt.Axes
        The axes object with the plot

    Examples
    --------
    >>> comparison_gpp = comparison['GPP']
    >>> plot_metric_comparison(comparison_gpp, 'GPP', outdir='./plots')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    regions = sorted(comparison.keys())
    x_pos = np.arange(len(regions))

    um_means = [comparison[r]['um_mean'] for r in regions]
    um_stds = [comparison[r]['um_std'] for r in regions]
    obs_means = [comparison[r]['obs_mean'] for r in regions]
    obs_errors = [comparison[r]['obs_error'] if comparison[r]['obs_error'] is not None else 0
                  for r in regions]

    # Plot bars
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, um_means, width, yerr=um_stds,
                   label='UM', alpha=0.8, capsize=5, color='steelblue')
    bars2 = ax.bar(x_pos + width/2, obs_means, width, yerr=obs_errors,
                   label='Observations', alpha=0.8, capsize=5, color='coral')

    # Color bars based on whether within uncertainty
    for i, region in enumerate(regions):
        if comparison[region]['within_uncertainty']:
            bars1[i].set_edgecolor('green')
            bars1[i].set_linewidth(2)

    # Formatting
    ax.set_xlabel('Region', fontsize=10)
    ax.set_ylabel(f"{metric} ({comparison[regions[0]].get('um_mean', 0)})", fontsize=10)
    ax.set_title(f'{metric} Comparison: UM vs Observations', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # Add green border legend
    green_patch = mpatches.Patch(edgecolor='green', facecolor='steelblue',
                                  label='Within obs uncertainty', linewidth=2)
    ax.legend(handles=[bars1, bars2, green_patch], frameon=False, loc='upper right')

    plt.tight_layout()

    # Save if requested
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        fname = filename if filename else f'{metric}_comparison.png'
        filepath = os.path.join(outdir, fname)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")

    return ax


def plot_regional_bias_heatmap(
    comparison: Dict[str, Dict[str, Dict[str, Any]]],
    metrics: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    value_type: str = 'bias_percent',
    ax: Optional[plt.Axes] = None,
    outdir: Optional[str] = None,
    filename: Optional[str] = None
) -> plt.Axes:
    """
    Plot heatmap of bias or RMSE across metrics and regions.

    Parameters
    ----------
    comparison : dict
        Full comparison results: {metric: {region: comparison_dict}}
    metrics : list of str, optional
        Metrics to include. Default: all
    regions : list of str, optional
        Regions to include. Default: all
    value_type : str, default='bias_percent'
        What to plot: 'bias', 'bias_percent', or 'rmse'
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    outdir : str, optional
        Output directory for saving
    filename : str, optional
        Filename for saving

    Returns
    -------
    plt.Axes
        The axes object with the plot

    Examples
    --------
    >>> plot_regional_bias_heatmap(comparison, value_type='bias_percent',
    ...                            outdir='./plots')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    if metrics is None:
        metrics = sorted(comparison.keys())
    if regions is None:
        # Get all unique regions
        all_regions = set()
        for m in comparison.values():
            all_regions.update(m.keys())
        regions = sorted(all_regions)

    # Build data matrix
    data = np.zeros((len(metrics), len(regions)))
    for i, metric in enumerate(metrics):
        for j, region in enumerate(regions):
            if metric in comparison and region in comparison[metric]:
                data[i, j] = comparison[metric][region][value_type]
            else:
                data[i, j] = np.nan

    # Plot heatmap
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-50, vmax=50)

    # Set ticks
    ax.set_xticks(np.arange(len(regions)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.set_yticklabels(metrics)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if value_type == 'bias_percent':
        cbar.set_label('Bias (%)', rotation=270, labelpad=20)
        ax.set_title('Regional Bias Heatmap: UM vs Observations (%)', fontweight='bold')
    elif value_type == 'bias':
        cbar.set_label('Absolute Bias', rotation=270, labelpad=20)
        ax.set_title('Regional Absolute Bias Heatmap', fontweight='bold')
    else:
        cbar.set_label('RMSE', rotation=270, labelpad=20)
        ax.set_title('Regional RMSE Heatmap', fontweight='bold')

    # Add values as text
    for i in range(len(metrics)):
        for j in range(len(regions)):
            if not np.isnan(data[i, j]):
                text = ax.text(j, i, f'{data[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()

    # Save if requested
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        fname = filename if filename else f'{value_type}_heatmap.png'
        filepath = os.path.join(outdir, fname)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")

    return ax


def plot_timeseries_with_obs(
    um_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    obs_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    metric: str,
    region: str,
    ax: Optional[plt.Axes] = None,
    outdir: Optional[str] = None,
    filename: Optional[str] = None
) -> plt.Axes:
    """
    Plot UM time series with observational mean and uncertainty band.

    Parameters
    ----------
    um_metrics : dict
        UM metrics in canonical schema
    obs_metrics : dict
        Obs metrics in canonical schema
    metric : str
        Metric to plot
    region : str
        Region to plot
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    outdir : str, optional
        Output directory for saving
    filename : str, optional
        Filename for saving

    Returns
    -------
    plt.Axes
        The axes object with the plot

    Examples
    --------
    >>> plot_timeseries_with_obs(um_metrics, cmip6_metrics, 'GPP', 'global',
    ...                          outdir='./plots')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Get data
    um_data = um_metrics[metric][region]
    obs_data = obs_metrics[metric][region]

    years = um_data['years']
    um_values = um_data['data']
    units = um_data['units']

    obs_mean = obs_data['data'][0]
    obs_error = obs_data.get('error', [None])[0]

    # Plot UM time series
    ax.plot(years, um_values, label='UM', color='steelblue', linewidth=1.5)

    # Plot obs mean as horizontal line
    ax.axhline(obs_mean, label='Obs mean', color='coral', linewidth=2, linestyle='--')

    # Add uncertainty band if available
    if obs_error is not None:
        ax.axhspan(obs_mean - obs_error, obs_mean + obs_error,
                  alpha=0.3, color='coral', label='Obs uncertainty')

    # Plot UM mean
    um_mean = np.mean(um_values)
    ax.axhline(um_mean, label=f'UM mean ({um_mean:.1f})', color='steelblue',
              linewidth=1.5, linestyle=':')

    # Formatting
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(f'{metric} ({units})', fontsize=10)
    ax.set_title(f'{metric} Time Series: {region}', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, loc='best')
    ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    # Save if requested
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        fname = filename if filename else f'{metric}_{region}_timeseries.png'
        filepath = os.path.join(outdir, fname)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")

    return ax


def create_validation_report(
    comparison: Dict[str, Dict[str, Dict[str, Any]]],
    um_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    obs_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    outdir: str = './validation_report',
    metrics: Optional[List[str]] = None
):
    """
    Create a complete validation report with multiple plots.

    Generates:
    1. Comparison bar charts for each metric
    2. Regional bias heatmap
    3. Time series plots for key metrics
    4. Summary statistics table

    Parameters
    ----------
    comparison : dict
        Full comparison results
    um_metrics : dict
        UM metrics in canonical schema
    obs_metrics : dict
        Obs metrics in canonical schema
    outdir : str, default='./validation_report'
        Output directory for report files
    metrics : list of str, optional
        Metrics to include. Default: all

    Examples
    --------
    >>> create_validation_report(comparison, um_metrics, cmip6_metrics,
    ...                          outdir='./validation_report')
    """
    os.makedirs(outdir, exist_ok=True)

    if metrics is None:
        metrics = sorted(comparison.keys())

    print(f"\nGenerating validation report in: {outdir}")
    print("=" * 80)

    # 1. Comparison bar charts for each metric
    print("\n1. Creating comparison bar charts...")
    for metric in metrics:
        if metric in comparison:
            plot_metric_comparison(comparison[metric], metric, outdir=outdir)

    # 2. Regional bias heatmap
    print("\n2. Creating bias heatmap...")
    plot_regional_bias_heatmap(comparison, metrics=metrics,
                               value_type='bias_percent', outdir=outdir)

    # 3. Time series for each metric and global region
    print("\n3. Creating time series plots...")
    for metric in metrics:
        if metric in um_metrics and 'global' in um_metrics[metric]:
            if metric in obs_metrics and 'global' in obs_metrics[metric]:
                plot_timeseries_with_obs(um_metrics, obs_metrics, metric, 'global',
                                        outdir=outdir)

    print("\n" + "=" * 80)
    print(f"✓ Validation report complete! Files saved to: {outdir}/")
    print("=" * 80)


def plot_three_way_comparison(
    um_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    cmip6_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    reccap_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    metric: str,
    ax: Optional[plt.Axes] = None,
    outdir: Optional[str] = None,
    filename: Optional[str] = None
) -> plt.Axes:
    """
    Plot UM and CMIP6 compared against RECCAP2 reference.

    Shows all three datasets with RECCAP2 as observational reference,
    using different colors to distinguish UM vs CMIP6 performance.

    Parameters
    ----------
    um_metrics : dict
        UM metrics in canonical schema
    cmip6_metrics : dict
        CMIP6 metrics in canonical schema
    reccap_metrics : dict
        RECCAP2 metrics in canonical schema (reference)
    metric : str
        Metric to plot
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    outdir : str, optional
        Output directory for saving
    filename : str, optional
        Filename for saving

    Returns
    -------
    plt.Axes
        The axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    # Get regions
    regions = sorted(reccap_metrics[metric].keys())
    x_pos = np.arange(len(regions))

    # Extract data
    um_means = [np.mean(um_metrics[metric][r]['data']) for r in regions]
    um_stds = [np.std(um_metrics[metric][r]['data']) for r in regions]

    cmip6_means = [cmip6_metrics[metric][r]['data'][0] for r in regions]
    cmip6_errors = [cmip6_metrics[metric][r].get('error', [0])[0] for r in regions]

    reccap_means = [reccap_metrics[metric][r]['data'][0] for r in regions]
    reccap_errors = [reccap_metrics[metric][r].get('error', [0])[0] for r in regions]

    # Plot bars with different colors
    width = 0.25
    bars_um = ax.bar(x_pos - width, um_means, width, yerr=um_stds,
                     label='UM (this study)', alpha=0.8, capsize=5, color='steelblue')
    bars_cmip6 = ax.bar(x_pos, cmip6_means, width, yerr=cmip6_errors,
                        label='CMIP6 ensemble', alpha=0.8, capsize=5, color='coral')
    bars_reccap = ax.bar(x_pos + width, reccap_means, width, yerr=reccap_errors,
                         label='RECCAP2 (obs)', alpha=0.8, capsize=5, color='forestgreen')

    # Color bars based on whether within RECCAP2 uncertainty
    for i, region in enumerate(regions):
        um_val = um_means[i]
        cmip6_val = cmip6_means[i]
        reccap_val = reccap_means[i]
        reccap_err = reccap_errors[i]

        # Check if within uncertainty
        um_within = abs(um_val - reccap_val) <= reccap_err
        cmip6_within = abs(cmip6_val - reccap_val) <= reccap_err

        if um_within:
            bars_um[i].set_edgecolor('green')
            bars_um[i].set_linewidth(2.5)
        if cmip6_within:
            bars_cmip6[i].set_edgecolor('green')
            bars_cmip6[i].set_linewidth(2.5)

    # Formatting
    ax.set_xlabel('Region', fontsize=11)
    units = um_metrics[metric][regions[0]]['units']
    ax.set_ylabel(f"{metric} ({units})", fontsize=11)
    ax.set_title(f'{metric}: UM vs CMIP6 vs RECCAP2', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    # Save if requested
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        fname = filename if filename else f'{metric}_three_way_comparison.png'
        filepath = os.path.join(outdir, fname)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")

    return ax


__all__ = [
    'plot_metric_comparison',
    'plot_regional_bias_heatmap',
    'plot_timeseries_with_obs',
    'create_validation_report',
    'plot_three_way_comparison',
]

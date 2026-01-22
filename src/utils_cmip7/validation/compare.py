"""
Metric comparison functions for validation.

Compares UM model outputs against observational datasets (CMIP6, RECCAP2).
NO NetCDF loading, aggregation, or metric computation permitted.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


def compute_bias(
    um_mean: float,
    obs_mean: float,
    obs_error: Optional[float] = None
) -> Tuple[float, float, bool]:
    """
    Compute absolute and percent bias between UM and observations.

    Parameters
    ----------
    um_mean : float
        UM model mean value
    obs_mean : float
        Observational mean value
    obs_error : float, optional
        Observational uncertainty

    Returns
    -------
    tuple
        (absolute_bias, percent_bias, within_uncertainty)
        - absolute_bias: um_mean - obs_mean
        - percent_bias: 100 * (um_mean - obs_mean) / obs_mean
        - within_uncertainty: True if |bias| <= obs_error

    Examples
    --------
    >>> compute_bias(120.0, 123.16, 9.61)
    (-3.16, -2.57, True)
    """
    absolute_bias = um_mean - obs_mean

    if obs_mean != 0:
        percent_bias = 100.0 * (um_mean - obs_mean) / obs_mean
    else:
        percent_bias = np.nan

    within_uncertainty = False
    if obs_error is not None:
        within_uncertainty = abs(absolute_bias) <= obs_error

    return absolute_bias, percent_bias, within_uncertainty


def compute_rmse(
    um_data: np.ndarray,
    obs_mean: float
) -> float:
    """
    Compute root mean square error between UM time series and obs mean.

    Parameters
    ----------
    um_data : np.ndarray
        UM model time series
    obs_mean : float
        Observational mean value (single value)

    Returns
    -------
    float
        RMSE value

    Examples
    --------
    >>> um_ts = np.array([120.0, 121.0, 122.0])
    >>> compute_rmse(um_ts, 123.16)
    2.57
    """
    differences = um_data - obs_mean
    rmse = np.sqrt(np.mean(differences ** 2))
    return float(rmse)


def compare_single_metric(
    um_metric_data: Dict[str, Any],
    obs_metric_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare a single UM metric against observational data for one region.

    Parameters
    ----------
    um_metric_data : dict
        UM data in canonical schema (with 'years', 'data', etc.)
    obs_metric_data : dict
        Obs data in canonical schema (with 'data', 'error', etc.)

    Returns
    -------
    dict
        Comparison results with keys:
        - um_mean: float
        - um_std: float
        - obs_mean: float
        - obs_error: float or None
        - bias: float
        - bias_percent: float
        - rmse: float
        - within_uncertainty: bool
        - n_years: int (UM time series length)

    Examples
    --------
    >>> um_data = {'years': np.array([1850, 1851]), 'data': np.array([120.0, 122.0]), ...}
    >>> obs_data = {'data': np.array([123.16]), 'error': np.array([9.61]), ...}
    >>> result = compare_single_metric(um_data, obs_data)
    >>> result['bias']
    -2.16
    """
    um_values = um_metric_data['data']
    obs_value = obs_metric_data['data'][0]  # Obs is single value
    obs_error = obs_metric_data.get('error', [None])[0]

    um_mean = float(np.mean(um_values))
    um_std = float(np.std(um_values))

    bias, bias_pct, within_unc = compute_bias(um_mean, obs_value, obs_error)
    rmse = compute_rmse(um_values, obs_value)

    return {
        'um_mean': um_mean,
        'um_std': um_std,
        'obs_mean': float(obs_value),
        'obs_error': float(obs_error) if obs_error is not None else None,
        'bias': bias,
        'bias_percent': bias_pct,
        'rmse': rmse,
        'within_uncertainty': within_unc,
        'n_years': len(um_values),
    }


def compare_metrics(
    um_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    obs_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    metrics: Optional[List[str]] = None,
    regions: Optional[List[str]] = None
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compare UM and observational metrics across multiple metrics and regions.

    Parameters
    ----------
    um_metrics : dict
        UM metrics in canonical schema: {metric: {region: {...}}}
    obs_metrics : dict
        Obs metrics in canonical schema: {metric: {region: {...}}}
    metrics : list of str, optional
        Metrics to compare. Default: all metrics present in both datasets
    regions : list of str, optional
        Regions to compare. Default: all regions present in both datasets

    Returns
    -------
    dict
        Comparison results: {metric: {region: comparison_dict}}

    Examples
    --------
    >>> comparison = compare_metrics(um_metrics, cmip6_metrics,
    ...                              metrics=['GPP', 'NPP'],
    ...                              regions=['global', 'Europe'])
    >>> comparison['GPP']['global']['bias']
    -3.16
    >>> comparison['GPP']['global']['within_uncertainty']
    True
    """
    # Determine metrics to compare
    if metrics is None:
        um_metric_set = set(um_metrics.keys())
        obs_metric_set = set(obs_metrics.keys())
        metrics = sorted(um_metric_set & obs_metric_set)

    # Determine regions to compare
    if regions is None:
        # Find common regions across all metrics
        common_regions = None
        for metric in metrics:
            if metric in um_metrics and metric in obs_metrics:
                um_regions = set(um_metrics[metric].keys())
                obs_regions = set(obs_metrics[metric].keys())
                metric_regions = um_regions & obs_regions

                if common_regions is None:
                    common_regions = metric_regions
                else:
                    common_regions &= metric_regions

        regions = sorted(common_regions) if common_regions else []

    # Perform comparisons
    result = {}
    for metric in metrics:
        result[metric] = {}

        if metric not in um_metrics or metric not in obs_metrics:
            continue

        for region in regions:
            if region not in um_metrics[metric] or region not in obs_metrics[metric]:
                continue

            um_data = um_metrics[metric][region]
            obs_data = obs_metrics[metric][region]

            comparison = compare_single_metric(um_data, obs_data)
            result[metric][region] = comparison

    return result


def summarize_comparison(
    comparison: Dict[str, Dict[str, Dict[str, Any]]],
    metric: Optional[str] = None
) -> Dict[str, Any]:
    """
    Summarize comparison results across regions or metrics.

    Parameters
    ----------
    comparison : dict
        Comparison results from compare_metrics()
    metric : str, optional
        If provided, summarize only this metric across regions.
        If None, summarize across all metrics and regions.

    Returns
    -------
    dict
        Summary statistics with keys:
        - n_comparisons: int
        - mean_bias: float
        - mean_bias_percent: float
        - mean_rmse: float
        - fraction_within_uncertainty: float
        - regions: list (if metric specified)
        - metrics: list (if metric not specified)

    Examples
    --------
    >>> summary = summarize_comparison(comparison, metric='GPP')
    >>> summary['mean_bias']
    -2.5
    >>> summary['fraction_within_uncertainty']
    0.9
    """
    biases = []
    bias_percents = []
    rmses = []
    within_unc = []
    items = []

    if metric is not None:
        # Summarize single metric across regions
        if metric in comparison:
            for region, comp in comparison[metric].items():
                biases.append(comp['bias'])
                bias_percents.append(comp['bias_percent'])
                rmses.append(comp['rmse'])
                within_unc.append(comp['within_uncertainty'])
                items.append(region)

        return {
            'n_comparisons': len(biases),
            'mean_bias': float(np.mean(biases)) if biases else np.nan,
            'mean_bias_percent': float(np.mean(bias_percents)) if bias_percents else np.nan,
            'mean_rmse': float(np.mean(rmses)) if rmses else np.nan,
            'fraction_within_uncertainty': float(np.mean(within_unc)) if within_unc else np.nan,
            'regions': items,
        }
    else:
        # Summarize all metrics and regions
        for m in comparison:
            for region, comp in comparison[m].items():
                biases.append(comp['bias'])
                bias_percents.append(comp['bias_percent'])
                rmses.append(comp['rmse'])
                within_unc.append(comp['within_uncertainty'])
                items.append(f"{m}.{region}")

        return {
            'n_comparisons': len(biases),
            'mean_bias': float(np.mean(biases)) if biases else np.nan,
            'mean_bias_percent': float(np.mean(bias_percents)) if bias_percents else np.nan,
            'mean_rmse': float(np.mean(rmses)) if rmses else np.nan,
            'fraction_within_uncertainty': float(np.mean(within_unc)) if within_unc else np.nan,
            'metrics_regions': items,
        }


def print_comparison_table(
    comparison: Dict[str, Dict[str, Dict[str, Any]]],
    metrics: Optional[List[str]] = None,
    regions: Optional[List[str]] = None
):
    """
    Print formatted comparison table to stdout.

    Parameters
    ----------
    comparison : dict
        Comparison results from compare_metrics()
    metrics : list of str, optional
        Metrics to include. Default: all
    regions : list of str, optional
        Regions to include. Default: all

    Examples
    --------
    >>> print_comparison_table(comparison, metrics=['GPP', 'NPP'], regions=['global', 'Europe'])
    """
    if metrics is None:
        metrics = sorted(comparison.keys())

    print("=" * 100)
    print(f"{'Metric':<12} {'Region':<15} {'UM Mean':>10} {'Obs Mean':>10} {'Bias':>8} {'Bias %':>8} {'RMSE':>8} {'In Unc?':>8}")
    print("=" * 100)

    for metric in metrics:
        if metric not in comparison:
            continue

        metric_regions = sorted(comparison[metric].keys())
        if regions is not None:
            metric_regions = [r for r in metric_regions if r in regions]

        for region in metric_regions:
            comp = comparison[metric][region]

            unc_status = "Yes" if comp['within_uncertainty'] else "No"
            if comp['obs_error'] is None:
                unc_status = "N/A"

            print(f"{metric:<12} {region:<15} {comp['um_mean']:>10.2f} {comp['obs_mean']:>10.2f} "
                  f"{comp['bias']:>8.2f} {comp['bias_percent']:>7.1f}% {comp['rmse']:>8.2f} {unc_status:>8}")

    print("=" * 100)


__all__ = [
    'compute_bias',
    'compute_rmse',
    'compare_single_metric',
    'compare_metrics',
    'summarize_comparison',
    'print_comparison_table',
]

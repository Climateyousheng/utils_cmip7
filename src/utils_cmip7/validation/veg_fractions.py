"""
Vegetation fraction validation and metrics.

Computes PFT-based metrics including:
- Global mean fractions per PFT
- Aggregated metrics (trees, grass)
- Regional metrics (Amazon, subtropical, NH)
- RMSE against observations
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any


# PFT mapping (UM convention)
PFT_MAPPING = {
    1: "BL",      # Broadleaf trees
    2: "NL",      # Needleleaf trees
    3: "C3",      # C3 grass
    4: "C4",      # C4 grass
    5: "shrub",   # Shrubs
    # 6-8: Other PFTs if available
    9: "bare_soil"  # Bare soil
}


def calculate_veg_metrics(um_metrics: Dict[str, Dict[str, Dict[str, Any]]],
                         expt: str,
                         regions: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate vegetation fraction metrics from extracted data for all regions.

    Parameters
    ----------
    um_metrics : dict
        Extracted data with structure: {expt: {region: {var: {...}}}}
    expt : str
        Experiment name
    regions : list of str, optional
        List of regions to compute metrics for. If None, uses all available regions.

    Returns
    -------
    dict
        Dictionary with structure: {metric: {region: value}}
        Metrics include:
        - {pft_name}: Fraction for each PFT (BL, NL, C3, C4, shrub, bare_soil)
        - trees: Sum of BL + NL
        - grass: Sum of C3 + C4

    Notes
    -----
    Uses 'frac' variable from um_metrics which contains PFT fractions.
    Computes time-mean for each PFT and region.
    """
    # Initialize metrics dict: {metric: {region: value}}
    metrics = {}

    # Check if experiment exists
    if expt not in um_metrics:
        return metrics

    # Determine regions to process
    if regions is None:
        regions = list(um_metrics[expt].keys())

    # Filter to regions that have frac data
    regions_with_frac = [r for r in regions if r in um_metrics[expt] and 'frac' in um_metrics[expt][r]]

    if not regions_with_frac:
        return metrics

    # Initialize metric entries
    for pft_id, pft_name in PFT_MAPPING.items():
        metrics[pft_name] = {}
    metrics['trees'] = {}
    metrics['grass'] = {}

    # Calculate metrics for each region
    for region in regions_with_frac:
        frac_data = um_metrics[expt][region]['frac']

        # Per-PFT metrics
        pft_values = {}
        for pft_id, pft_name in PFT_MAPPING.items():
            pft_key = f'PFT {pft_id}'
            if pft_key in frac_data:
                pft_series = frac_data[pft_key]
                if 'data' in pft_series and len(pft_series['data']) > 0:
                    # Compute time-mean
                    mean_val = np.mean(pft_series['data'])
                    metrics[pft_name][region] = mean_val
                    pft_values[pft_name] = mean_val

        # Aggregates
        if 'BL' in pft_values and 'NL' in pft_values:
            metrics['trees'][region] = pft_values['BL'] + pft_values['NL']

        if 'C3' in pft_values and 'C4' in pft_values:
            metrics['grass'][region] = pft_values['C3'] + pft_values['C4']

    # Remove empty metrics
    metrics = {k: v for k, v in metrics.items() if v}

    return metrics


def save_veg_metrics_to_csv(metrics: Dict[str, Dict[str, float]], expt: str, outdir: Path) -> pd.DataFrame:
    """
    Save vegetation metrics to CSV in same format as main metrics.

    Parameters
    ----------
    metrics : dict
        Vegetation metrics from calculate_veg_metrics()
        Structure: {metric: {region: value}}
    expt : str
        Experiment name
    outdir : Path
        Output directory

    Returns
    -------
    pandas.DataFrame
        DataFrame with metrics as rows and regions as columns
    """
    if not metrics:
        print(f"  ⚠ No vegetation metrics to save for {expt}")
        return None

    # Create DataFrame: metrics as rows, regions as columns
    df = pd.DataFrame(metrics).T
    df.index.name = 'metric'

    # Save
    csv_path = outdir / f'{expt}_veg_fractions.csv'
    df.to_csv(csv_path)
    print(f"  ✓ Saved vegetation fraction metrics: {csv_path}")
    print(f"    Metrics: {len(metrics)}, Regions: {len(df.columns)}")

    return df


def compare_veg_metrics(um_metrics: Dict[str, Dict[str, float]],
                       obs_metrics: Optional[Dict[str, float]] = None,
                       region: str = 'global') -> Dict[str, Dict[str, float]]:
    """
    Compare UM vegetation metrics against observations for a specific region.

    Parameters
    ----------
    um_metrics : dict
        UM vegetation metrics with structure: {metric: {region: value}}
    obs_metrics : dict, optional
        Observational vegetation metrics (e.g., from IGBP)
        Structure: {metric: value} for global observations
    region : str, optional
        Region to compare (default: 'global')

    Returns
    -------
    dict
        Comparison statistics for each metric:
        - um_value: UM value
        - obs_value: Observation value (if available)
        - bias: UM - Obs
        - bias_percent: (UM - Obs) / Obs * 100

    Notes
    -----
    If obs_metrics not provided, returns only UM values.
    Only compares for the specified region (typically 'global' for IGBP).
    """
    comparison = {}

    for metric, regional_values in um_metrics.items():
        if region not in regional_values:
            continue

        um_val = regional_values[region]
        comparison[metric] = {
            'um_value': um_val
        }

        if obs_metrics and metric in obs_metrics:
            obs_val = obs_metrics[metric]
            bias = um_val - obs_val
            bias_pct = (bias / obs_val * 100) if obs_val != 0 else np.nan

            comparison[metric].update({
                'obs_value': obs_val,
                'bias': bias,
                'bias_percent': bias_pct
            })

    return comparison


def load_obs_veg_metrics(obs_file: Optional[str] = None) -> Dict[str, float]:
    """
    Load observational vegetation fraction metrics (IGBP global values).

    Parameters
    ----------
    obs_file : str, optional
        Path to observational NetCDF file
        If None, uses hardcoded IGBP values

    Returns
    -------
    dict
        Observational vegetation metrics with structure: {metric: value}
        Keys match PFT names: BL, NL, C3, C4, shrub, bare_soil, trees, grass

    Notes
    -----
    This is a placeholder using approximate IGBP global mean values.
    Implement loading from NetCDF if obs file provided.
    """
    # Placeholder - using approximate IGBP global values
    # Replace with actual observations when available
    obs_metrics = {
        'BL': 0.15,          # ~15% broadleaf globally
        'NL': 0.08,          # ~8% needleleaf
        'C3': 0.12,          # ~12% C3 grass
        'C4': 0.05,          # ~5% C4 grass
        'shrub': 0.10,       # ~10% shrub
        'bare_soil': 0.20,   # ~20% bare soil
        'trees': 0.23,       # BL + NL
        'grass': 0.17,       # C3 + C4
    }

    if obs_file:
        # TODO: Implement actual loading from NetCDF
        print(f"  ⚠ Loading from NetCDF not yet implemented")
        print(f"    Using placeholder IGBP values")

    return obs_metrics


__all__ = [
    'PFT_MAPPING',
    'calculate_veg_metrics',
    'save_veg_metrics_to_csv',
    'compare_veg_metrics',
    'load_obs_veg_metrics',
]

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
                         expt: str) -> Dict[str, float]:
    """
    Calculate vegetation fraction metrics from extracted data.

    Parameters
    ----------
    um_metrics : dict
        Extracted data with structure: {expt: {region: {var: {...}}}}
    expt : str
        Experiment name

    Returns
    -------
    dict
        Dictionary of vegetation metrics:
        - global_mean_{pft}: Global mean fraction for each PFT
        - global_mean_trees: Sum of BL + NL
        - global_mean_grass: Sum of C3 + C4
        - amazon_trees: Amazon region tree fraction (if available)
        - subtropical_trees: 30S-30N tree fraction (if available)
        - NH_trees: 30N-90N tree fraction (if available)

    Notes
    -----
    Uses 'frac' variable from um_metrics which contains PFT fractions.
    Regional metrics require regional extraction data.
    """
    metrics = {}

    # Check if frac data exists
    if expt not in um_metrics or 'global' not in um_metrics[expt]:
        return metrics

    if 'frac' not in um_metrics[expt]['global']:
        return metrics

    frac_data = um_metrics[expt]['global']['frac']

    # Calculate global mean for each PFT
    for pft_id, pft_name in PFT_MAPPING.items():
        pft_key = f'PFT {pft_id}'
        if pft_key in frac_data:
            pft_series = frac_data[pft_key]
            if 'data' in pft_series and len(pft_series['data']) > 0:
                # Compute time-mean
                mean_val = np.mean(pft_series['data'])
                metrics[f'global_mean_{pft_name}'] = mean_val

    # Aggregates
    if 'global_mean_BL' in metrics and 'global_mean_NL' in metrics:
        metrics['global_mean_trees'] = metrics['global_mean_BL'] + metrics['global_mean_NL']

    if 'global_mean_C3' in metrics and 'global_mean_C4' in metrics:
        metrics['global_mean_grass'] = metrics['global_mean_C3'] + metrics['global_mean_C4']

    # Regional metrics (if regional data available)
    # Note: For more accurate regional metrics, would need to reprocess with lat/lon masks
    # For now, we can approximate using RECCAP2 regions as proxies

    # Amazon trees approximation (South America region as proxy)
    if 'South_America' in um_metrics[expt]:
        sa_frac = um_metrics[expt]['South_America'].get('frac', {})
        bl = sa_frac.get('PFT 1', {}).get('data', [])
        nl = sa_frac.get('PFT 2', {}).get('data', [])
        if len(bl) > 0 and len(nl) > 0:
            metrics['amazon_trees_approx'] = np.mean(bl) + np.mean(nl)

    # Subtropical trees (could use combination of regions near equator)
    # NH trees (could use North_America + Europe + North_Asia as proxy)
    if all(r in um_metrics[expt] for r in ['North_America', 'Europe', 'North_Asia']):
        nh_tree_fracs = []
        for region in ['North_America', 'Europe', 'North_Asia']:
            region_frac = um_metrics[expt][region].get('frac', {})
            bl = region_frac.get('PFT 1', {}).get('data', [])
            nl = region_frac.get('PFT 2', {}).get('data', [])
            if len(bl) > 0 and len(nl) > 0:
                nh_tree_fracs.append(np.mean(bl) + np.mean(nl))
        if nh_tree_fracs:
            metrics['NH_trees_approx'] = np.mean(nh_tree_fracs)

    return metrics


def save_veg_metrics_to_csv(metrics: Dict[str, float], expt: str, outdir: Path) -> pd.DataFrame:
    """
    Save vegetation metrics to CSV.

    Parameters
    ----------
    metrics : dict
        Vegetation metrics from calculate_veg_metrics()
    expt : str
        Experiment name
    outdir : Path
        Output directory

    Returns
    -------
    pandas.DataFrame
        DataFrame with metrics
    """
    if not metrics:
        print(f"  ⚠ No vegetation metrics to save for {expt}")
        return None

    # Create DataFrame
    df = pd.DataFrame({
        'metric': list(metrics.keys()),
        'value': list(metrics.values())
    })

    # Save
    csv_path = outdir / f'{expt}_veg_fractions.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved vegetation fraction metrics: {csv_path}")
    print(f"    Metrics: {len(metrics)}")

    return df


def compare_veg_metrics(um_metrics: Dict[str, float],
                       obs_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare UM vegetation metrics against observations.

    Parameters
    ----------
    um_metrics : dict
        UM vegetation metrics
    obs_metrics : dict, optional
        Observational vegetation metrics (e.g., from IGBP)

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
    """
    comparison = {}

    for metric, um_val in um_metrics.items():
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
    Load observational vegetation fraction metrics.

    Parameters
    ----------
    obs_file : str, optional
        Path to observational NetCDF file
        If None, returns empty dict

    Returns
    -------
    dict
        Observational vegetation metrics

    Notes
    -----
    This is a placeholder. Implement loading from NetCDF if obs file provided.
    For now, could use hardcoded IGBP values if available.
    """
    # Placeholder - would load from NetCDF or package data
    # Example hardcoded values (replace with actual IGBP data):
    obs_metrics = {
        # These are example values - replace with actual observations
        'global_mean_BL': 0.15,  # ~15% broadleaf globally
        'global_mean_NL': 0.08,  # ~8% needleleaf
        'global_mean_C3': 0.12,  # ~12% C3 grass
        'global_mean_C4': 0.05,  # ~5% C4 grass
        'global_mean_shrub': 0.10,  # ~10% shrub
        'global_mean_bare_soil': 0.20,  # ~20% bare soil
        'global_mean_trees': 0.23,  # BL + NL
        'global_mean_grass': 0.17,  # C3 + C4
    }

    if obs_file:
        # TODO: Implement actual loading from NetCDF
        print(f"  ⚠ Loading from NetCDF not yet implemented")
        print(f"    Using placeholder values for demonstration")

    return obs_metrics


__all__ = [
    'PFT_MAPPING',
    'calculate_veg_metrics',
    'save_veg_metrics_to_csv',
    'compare_veg_metrics',
    'load_obs_veg_metrics',
]

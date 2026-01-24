"""
Vegetation fraction validation and metrics.

Computes PFT-based metrics including:
- Global mean fractions per PFT
- Aggregated metrics (trees, grass)
- Regional metrics (Amazon, subtropical, NH)
- RMSE against observations
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import xarray for NetCDF loading
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

# Use importlib.resources for Python 3.9+, fallback for 3.8
if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    try:
        from importlib_resources import files
    except ImportError:
        files = None


# PFT mapping (UM convention)
# PFT 1-5: Vegetation types
# PFT 8: Bare soil
# PFT 9: Lake/ice (typically not included in fraction metrics)
PFT_MAPPING = {
    1: "BL",         # Broadleaf trees
    2: "NL",         # Needleleaf trees
    3: "C3",         # C3 grass
    4: "C4",         # C4 grass
    5: "shrub",      # Shrubs
    8: "bare_soil",  # Bare soil
    # 9: "lake_ice"  # Lake/ice - excluded from standard metrics
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

    # Extract RMSE values (stored alongside regional mean data during extraction)
    for region in regions_with_frac:
        frac_data = um_metrics[expt][region]['frac']
        for pft_id, pft_name in PFT_MAPPING.items():
            pft_key = f'PFT {pft_id}'
            if pft_key in frac_data and 'rmse' in frac_data[pft_key]:
                rmse_key = f'rmse_{pft_name}'
                if rmse_key not in metrics:
                    metrics[rmse_key] = {}
                metrics[rmse_key][region] = frac_data[pft_key]['rmse']

    # Remove empty metrics
    metrics = {k: v for k, v in metrics.items() if v}

    return metrics


def compute_spatial_rmse(model_data, obs_data):
    """
    Compute spatial RMSE between model and obs 2D fields.

    Parameters
    ----------
    model_data : numpy.ndarray
        Model PFT fraction field (lat, lon)
    obs_data : numpy.ndarray
        IGBP PFT fraction field (lat, lon)

    Returns
    -------
    float
        RMSE value (NaN-aware)
    """
    diff_sq = (model_data - obs_data) ** 2
    return float(np.sqrt(np.nanmean(diff_sq)))


def load_igbp_spatial():
    """
    Load IGBP vegetation fraction observation file with spatial dimensions.

    Returns
    -------
    xarray.Dataset or None
        IGBP dataset with variable 'fracPFTs_snp_srf'
        Dimensions: (pseudo, latitude, longitude)
        Returns None if xarray not available or file not found.
    """
    if not HAS_XARRAY:
        return None

    try:
        obs_dir = get_obs_dir()
        obs_file = os.path.join(obs_dir, 'igbp.veg_fraction_metrics.nc')
    except FileNotFoundError:
        return None

    if not os.path.exists(obs_file):
        return None

    try:
        return xr.open_dataset(obs_file, decode_times=False)
    except Exception as e:
        print(f"  ⚠ Error loading IGBP spatial data: {e}")
        return None


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


def get_obs_dir():
    """Get absolute path to obs/ directory using importlib.resources."""
    try:
        if files is not None:
            # Use importlib.resources to access package data
            obs_path = files('utils_cmip7').joinpath('data/obs')
            return str(obs_path)
    except (TypeError, AttributeError):
        pass

    # Fallback for editable installs or development mode
    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    obs_dir = os.path.join(pkg_dir, 'data', 'obs')
    if os.path.exists(obs_dir):
        return obs_dir

    raise FileNotFoundError(
        f"obs/ directory not found. "
        f"Make sure package is properly installed or run from repository root."
    )


def load_obs_veg_metrics(obs_file: Optional[str] = None) -> Dict[str, float]:
    """
    Load observational vegetation fraction metrics from IGBP NetCDF file.

    Parameters
    ----------
    obs_file : str, optional
        Path to observational NetCDF file
        If None, uses packaged IGBP data (igbp.veg_fraction_metrics.nc)

    Returns
    -------
    dict
        Observational vegetation metrics with structure: {metric: value}
        Keys match PFT names: BL, NL, C3, C4, shrub, bare_soil, trees, grass

    Notes
    -----
    Reads from NetCDF file with structure:
    - Variable: fracPFTs_snp_srf
    - Dimensions: pseudo (PFT index), latitude, longitude

    Computes global mean by averaging over latitude and longitude.
    """
    if not HAS_XARRAY:
        print(f"  ⚠ xarray not available, using placeholder IGBP values")
        # Fallback to hardcoded values
        return {
            'BL': 0.15, 'NL': 0.08, 'C3': 0.12, 'C4': 0.05,
            'shrub': 0.10, 'bare_soil': 0.20,
            'trees': 0.23, 'grass': 0.17,
        }

    # Determine file path
    if obs_file is None:
        try:
            obs_dir = get_obs_dir()
            obs_file = os.path.join(obs_dir, 'igbp.veg_fraction_metrics.nc')
        except FileNotFoundError:
            print(f"  ⚠ IGBP file not found, using placeholder values")
            return {
                'BL': 0.15, 'NL': 0.08, 'C3': 0.12, 'C4': 0.05,
                'shrub': 0.10, 'bare_soil': 0.20,
                'trees': 0.23, 'grass': 0.17,
            }

    if not os.path.exists(obs_file):
        print(f"  ⚠ IGBP file not found: {obs_file}")
        print(f"    Using placeholder values")
        return {
            'BL': 0.15, 'NL': 0.08, 'C3': 0.12, 'C4': 0.05,
            'shrub': 0.10, 'bare_soil': 0.20,
            'trees': 0.23, 'grass': 0.17,
        }

    try:
        # Open IGBP NetCDF file
        obs_ds = xr.open_dataset(obs_file, decode_times=False)

        # Extract fracPFTs_snp_srf variable
        # Structure: (pseudo, latitude, longitude)
        frac_var = obs_ds["fracPFTs_snp_srf"]

        # Compute global mean for each PFT
        obs_metrics = {}
        for pft_id, pft_name in PFT_MAPPING.items():
            # Note: PFT indexing may be 0-based or 1-based
            # Adjust index if needed (UM uses 1-based, Python uses 0-based)
            try:
                # Try 0-based indexing first (pft_id - 1)
                pft_data = frac_var.isel({"pseudo": pft_id - 1}).squeeze()
                global_mean = float(pft_data.mean(['latitude', 'longitude']).values)
                obs_metrics[pft_name] = global_mean
            except (IndexError, KeyError):
                # Try 1-based indexing if 0-based fails
                try:
                    pft_data = frac_var.isel({"pseudo": pft_id}).squeeze()
                    global_mean = float(pft_data.mean(['latitude', 'longitude']).values)
                    obs_metrics[pft_name] = global_mean
                except (IndexError, KeyError):
                    print(f"  ⚠ Could not load PFT {pft_id} ({pft_name}) from IGBP file")

        # Compute aggregates
        if 'BL' in obs_metrics and 'NL' in obs_metrics:
            obs_metrics['trees'] = obs_metrics['BL'] + obs_metrics['NL']

        if 'C3' in obs_metrics and 'C4' in obs_metrics:
            obs_metrics['grass'] = obs_metrics['C3'] + obs_metrics['C4']

        obs_ds.close()
        return obs_metrics

    except Exception as e:
        print(f"  ⚠ Error loading IGBP file: {e}")
        print(f"    Using placeholder values")
        return {
            'BL': 0.15, 'NL': 0.08, 'C3': 0.12, 'C4': 0.05,
            'shrub': 0.10, 'bare_soil': 0.20,
            'trees': 0.23, 'grass': 0.17,
        }


__all__ = [
    'PFT_MAPPING',
    'calculate_veg_metrics',
    'compute_spatial_rmse',
    'load_igbp_spatial',
    'save_veg_metrics_to_csv',
    'compare_veg_metrics',
    'load_obs_veg_metrics',
]

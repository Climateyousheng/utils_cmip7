"""
Observational data loader for CMIP6 and RECCAP2 datasets.

Loads CSV files from obs/ directory into canonical metric schema.
NO aggregation or unit conversion logic permitted.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from ..processing.metrics import METRIC_DEFINITIONS, validate_metric_output


# Path to obs/ directory (relative to package root)
def get_obs_dir():
    """Get absolute path to obs/ directory."""
    # Assume obs/ is at repository root, two levels up from src/utils_cmip7
    pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    obs_dir = os.path.join(pkg_dir, 'obs')
    if not os.path.exists(obs_dir):
        # Try alternative: obs/ sibling to src/
        alt_obs_dir = os.path.join(os.path.dirname(pkg_dir), 'obs')
        if os.path.exists(alt_obs_dir):
            return alt_obs_dir
        raise FileNotFoundError(
            f"obs/ directory not found. Tried:\n"
            f"  {obs_dir}\n"
            f"  {alt_obs_dir}\n"
            f"Make sure obs/ directory exists at repository root."
        )
    return obs_dir


def load_cmip6_metrics(
    metrics: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    include_errors: bool = True
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load CMIP6 observational data into canonical schema.

    Reads data from obs/stores_vs_fluxes_cmip6.csv and
    obs/stores_vs_fluxes_cmip6_err.csv

    Parameters
    ----------
    metrics : list of str, optional
        Metrics to load. Default: all available (GPP, NPP, CVeg, CSoil, Tau)
    regions : list of str, optional
        Regions to load. Default: all available regions
    include_errors : bool, default=True
        Include error/uncertainty values

    Returns
    -------
    dict
        Nested dictionary with structure:
        {metric: {region: {
            'years': np.array([]),  # Empty for time-aggregated obs
            'data': np.array([value]),
            'units': str,
            'source': 'CMIP6',
            'dataset': 'CMIP6-ensemble-mean',
            'error': np.array([error_value])  # if include_errors=True
        }}}

    Examples
    --------
    >>> cmip6 = load_cmip6_metrics(metrics=['GPP', 'NPP'], regions=['global', 'Europe'])
    >>> cmip6['GPP']['global']['data']
    array([123.16])
    >>> cmip6['GPP']['global']['error']
    array([9.61])

    Notes
    -----
    CMIP6 data represents multi-year ensemble means (typically 1995-2014).
    The 'years' array is empty to indicate time-aggregated data.
    Use the single value in 'data' array for comparison with model time series means.
    """
    obs_dir = get_obs_dir()
    values_file = os.path.join(obs_dir, 'stores_vs_fluxes_cmip6.csv')
    errors_file = os.path.join(obs_dir, 'stores_vs_fluxes_cmip6_err.csv')

    # Load CSV files
    values_df = pd.read_csv(values_file, index_col=0)
    if include_errors:
        errors_df = pd.read_csv(errors_file, index_col=0)

    # Get available metrics and regions
    available_metrics = values_df.index.tolist()
    available_regions = values_df.columns.tolist()

    # Filter to requested metrics/regions
    if metrics is None:
        metrics = available_metrics
    else:
        # Validate requested metrics
        missing = [m for m in metrics if m not in available_metrics]
        if missing:
            raise ValueError(
                f"Metrics not found in CMIP6 data: {missing}. "
                f"Available: {available_metrics}"
            )

    if regions is None:
        regions = available_regions
    else:
        # Validate requested regions
        missing = [r for r in regions if r not in available_regions]
        if missing:
            raise ValueError(
                f"Regions not found in CMIP6 data: {missing}. "
                f"Available: {available_regions}"
            )

    # Build canonical structure
    result = {}
    for metric in metrics:
        result[metric] = {}
        for region in regions:
            value = values_df.loc[metric, region]

            # Get units from METRIC_DEFINITIONS
            units = METRIC_DEFINITIONS.get(metric, {}).get('output_units', 'unknown')

            data_dict = {
                'years': np.array([]),  # Empty for time-aggregated obs
                'data': np.array([float(value)]),
                'units': units,
                'source': 'CMIP6',
                'dataset': 'CMIP6-ensemble-mean'
            }

            if include_errors:
                error = errors_df.loc[metric, region]
                data_dict['error'] = np.array([float(error)])

            # Validate canonical schema
            validate_metric_output(data_dict, f"CMIP6.{metric}.{region}")

            result[metric][region] = data_dict

    return result


def load_reccap_metrics(
    metrics: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    include_errors: bool = True
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load RECCAP2 observational data into canonical schema.

    Reads data from obs/stores_vs_fluxes_reccap.csv and
    obs/stores_vs_fluxes_reccap_err.csv

    Parameters
    ----------
    metrics : list of str, optional
        Metrics to load. Default: all available (GPP, NPP, CVeg, CSoil, Tau)
    regions : list of str, optional
        Regions to load. Default: all available regions
    include_errors : bool, default=True
        Include error/uncertainty values

    Returns
    -------
    dict
        Nested dictionary with structure:
        {metric: {region: {
            'years': np.array([]),  # Empty for time-aggregated obs
            'data': np.array([value]),
            'units': str,
            'source': 'RECCAP2',
            'dataset': 'RECCAP2-synthesis',
            'error': np.array([error_value])  # if include_errors=True
        }}}

    Examples
    --------
    >>> reccap = load_reccap_metrics(metrics=['GPP'], regions=['global'])
    >>> reccap['GPP']['global']['data']
    array([124.04])
    >>> reccap['GPP']['global']['source']
    'RECCAP2'

    Notes
    -----
    RECCAP2 data represents synthesis estimates from multiple data streams.
    The 'years' array is empty to indicate time-aggregated data.
    """
    obs_dir = get_obs_dir()
    values_file = os.path.join(obs_dir, 'stores_vs_fluxes_reccap.csv')
    errors_file = os.path.join(obs_dir, 'stores_vs_fluxes_reccap_err.csv')

    # Load CSV files
    values_df = pd.read_csv(values_file, index_col=0)
    if include_errors:
        errors_df = pd.read_csv(errors_file, index_col=0)

    # Get available metrics and regions
    available_metrics = values_df.index.tolist()
    available_regions = values_df.columns.tolist()

    # Filter to requested metrics/regions
    if metrics is None:
        metrics = available_metrics
    else:
        # Validate requested metrics
        missing = [m for m in metrics if m not in available_metrics]
        if missing:
            raise ValueError(
                f"Metrics not found in RECCAP2 data: {missing}. "
                f"Available: {available_metrics}"
            )

    if regions is None:
        regions = available_regions
    else:
        # Validate requested regions
        missing = [r for r in regions if r not in available_regions]
        if missing:
            raise ValueError(
                f"Regions not found in RECCAP2 data: {missing}. "
                f"Available: {available_regions}"
            )

    # Build canonical structure
    result = {}
    for metric in metrics:
        result[metric] = {}
        for region in regions:
            value = values_df.loc[metric, region]

            # Get units from METRIC_DEFINITIONS
            units = METRIC_DEFINITIONS.get(metric, {}).get('output_units', 'unknown')

            data_dict = {
                'years': np.array([]),  # Empty for time-aggregated obs
                'data': np.array([float(value)]),
                'units': units,
                'source': 'RECCAP2',
                'dataset': 'RECCAP2-synthesis'
            }

            if include_errors:
                error = errors_df.loc[metric, region]
                data_dict['error'] = np.array([float(error)])

            # Validate canonical schema
            validate_metric_output(data_dict, f"RECCAP2.{metric}.{region}")

            result[metric][region] = data_dict

    return result


def list_available_obs_metrics(source: str = 'both') -> List[str]:
    """
    List metrics available in observational datasets.

    Parameters
    ----------
    source : str, default='both'
        Which source to check: 'cmip6', 'reccap2', or 'both'

    Returns
    -------
    list of str
        Available metric names

    Examples
    --------
    >>> list_available_obs_metrics('cmip6')
    ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']
    """
    obs_dir = get_obs_dir()

    if source in ('cmip6', 'both'):
        cmip6_file = os.path.join(obs_dir, 'stores_vs_fluxes_cmip6.csv')
        cmip6_df = pd.read_csv(cmip6_file, index_col=0)
        cmip6_metrics = cmip6_df.index.tolist()

        if source == 'cmip6':
            return cmip6_metrics

    if source in ('reccap2', 'both'):
        reccap_file = os.path.join(obs_dir, 'stores_vs_fluxes_reccap.csv')
        reccap_df = pd.read_csv(reccap_file, index_col=0)
        reccap_metrics = reccap_df.index.tolist()

        if source == 'reccap2':
            return reccap_metrics

    # Return intersection for 'both'
    return sorted(set(cmip6_metrics) & set(reccap_metrics))


def list_available_obs_regions(source: str = 'both') -> List[str]:
    """
    List regions available in observational datasets.

    Parameters
    ----------
    source : str, default='both'
        Which source to check: 'cmip6', 'reccap2', or 'both'

    Returns
    -------
    list of str
        Available region names

    Examples
    --------
    >>> list_available_obs_regions()
    ['global', 'North_America', 'South_America', 'Europe', ...]
    """
    obs_dir = get_obs_dir()

    if source in ('cmip6', 'both'):
        cmip6_file = os.path.join(obs_dir, 'stores_vs_fluxes_cmip6.csv')
        cmip6_df = pd.read_csv(cmip6_file, index_col=0)
        cmip6_regions = cmip6_df.columns.tolist()

        if source == 'cmip6':
            return cmip6_regions

    if source in ('reccap2', 'both'):
        reccap_file = os.path.join(obs_dir, 'stores_vs_fluxes_reccap.csv')
        reccap_df = pd.read_csv(reccap_file, index_col=0)
        reccap_regions = reccap_df.columns.tolist()

        if source == 'reccap2':
            return reccap_regions

    # Return intersection for 'both'
    return sorted(set(cmip6_regions) & set(reccap_regions))


__all__ = [
    'load_cmip6_metrics',
    'load_reccap_metrics',
    'list_available_obs_metrics',
    'list_available_obs_regions',
]

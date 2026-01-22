"""
Metric orchestration for UM model outputs.

Computes canonical metrics from UM data using processing layer functions.
This module bridges extraction and validation by producing standardized outputs.
"""

import numpy as np
from typing import Dict, List, Optional, Any

from .extraction import extract_annual_means
from .raw import extract_annual_mean_raw
from ..processing.metrics import (
    get_metric_config,
    compute_derived_metric,
    validate_canonical_structure,
)


# Mapping from extraction variable names to canonical metric names
VARIABLE_TO_METRIC = {
    'GPP': 'GPP',
    'NPP': 'NPP',
    'VegCarb': 'CVeg',
    'soilCarbon': 'CSoil',
    'soilResp': 'soilResp',
    'temp': 'tas',
    'precip': 'precip',
    'fgco2': 'fgco2',
    # Aliases
    'CVeg': 'CVeg',
    'CSoil': 'CSoil',
    'tas': 'tas',
}


def compute_metrics_from_annual_means(
    expt_name: str,
    metrics: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    base_dir: str = '~/annual_mean'
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute canonical metrics from pre-processed annual mean NetCDF files.

    Parameters
    ----------
    expt_name : str
        Experiment name (e.g., 'xqhuc')
    metrics : list of str, optional
        Metrics to compute. Default: ['GPP', 'NPP', 'CVeg', 'CSoil', 'NEP', 'Tau']
    regions : list of str, optional
        Regions to process. Default: all RECCAP2 regions + global
    base_dir : str, default='~/annual_mean'
        Base directory containing annual mean files

    Returns
    -------
    dict
        Nested dictionary with canonical schema:
        {metric: {region: {
            'years': np.ndarray,
            'data': np.ndarray,
            'units': str,
            'source': 'UM',
            'dataset': str (expt_name)
        }}}

    Examples
    --------
    >>> metrics = compute_metrics_from_annual_means(
    ...     'xqhuc',
    ...     metrics=['GPP', 'NPP', 'CVeg', 'NEP'],
    ...     regions=['global', 'Europe']
    ... )
    >>> metrics['GPP']['global']['data']
    array([123.5, 124.2, ...])
    >>> metrics['NEP']['global']['units']
    'PgC/yr'

    Notes
    -----
    This function:
    1. Calls extract_annual_means() to get raw extraction data
    2. Transforms to canonical schema
    3. Computes derived metrics (NEP, Tau)
    4. Validates output against canonical schema
    """
    if metrics is None:
        metrics = ['GPP', 'NPP', 'CVeg', 'CSoil', 'NEP', 'Tau', 'tas', 'precip']

    # Determine which variables to extract
    # We need component variables for derived metrics
    # IMPORTANT: Use set for deduplication, then convert to sorted list for deterministic order
    required_vars_set = set()
    for metric in metrics:
        config = get_metric_config(metric)
        if config['aggregation'] == 'DERIVED':
            # Add component variables
            if metric == 'NEP':
                required_vars_set.update(['NPP', 'soilResp'])
            elif metric == 'Tau':
                required_vars_set.update(['CSoil', 'NPP'])
        else:
            # Map metric name to extraction variable name
            for var_name, metric_name in VARIABLE_TO_METRIC.items():
                if metric_name == metric:
                    required_vars_set.add(var_name)
                    break

    # Convert to sorted list for deterministic iteration order
    # This prevents variable/mapping misalignment due to non-deterministic set iteration
    required_vars = sorted(required_vars_set)

    # Convert to extraction variable list
    var_list_for_extraction = []
    var_mapping_for_extraction = []

    # Build extraction variable list
    extraction_var_map = {
        'GPP': ('GPP', 'GPP'),
        'NPP': ('NPP', 'NPP'),
        'soilResp': ('soilResp', 'S resp'),
        'VegCarb': ('VegCarb', 'V carb'),
        'soilCarbon': ('soilCarbon', 'S carb'),
        'temp': ('temp', 'Others'),
        'precip': ('precip', 'precip'),
        'fgco2': ('fgco2', 'field646_mm_dpth'),
    }

    for req_var in required_vars:
        if req_var in extraction_var_map:
            var_name, var_mapping = extraction_var_map[req_var]
            var_list_for_extraction.append(var_name)
            var_mapping_for_extraction.append(var_mapping)

    # Extract data using existing function
    raw_data = extract_annual_means(
        [expt_name],
        var_list=var_list_for_extraction,
        var_mapping=var_mapping_for_extraction,
        regions=regions
    )

    # Transform to canonical schema
    result = {}

    # Get data for this experiment
    expt_data = raw_data.get(expt_name, {})

    # Process each requested metric
    for metric in metrics:
        result[metric] = {}
        config = get_metric_config(metric)

        # Determine which regions to process
        if regions is None:
            target_regions = list(expt_data.keys())
        else:
            target_regions = regions

        for region in target_regions:
            region_data = expt_data.get(region, {})

            if config['aggregation'] == 'DERIVED':
                # Compute derived metric
                if metric == 'NEP':
                    npp_data = region_data.get('NPP', {})
                    soil_resp_data = region_data.get('soilResp', {})

                    if npp_data and soil_resp_data:
                        years = npp_data['years']
                        nep_values = compute_derived_metric(
                            'NEP',
                            {'NPP': npp_data['data'], 'soilResp': soil_resp_data['data']}
                        )

                        result[metric][region] = {
                            'years': years,
                            'data': nep_values,
                            'units': config['output_units'],
                            'source': 'UM',
                            'dataset': expt_name
                        }

                elif metric == 'Tau':
                    # Map CSoil and NPP
                    csoil_data = region_data.get('soilCarbon', {})
                    npp_data = region_data.get('NPP', {})

                    if csoil_data and npp_data:
                        years = csoil_data['years']
                        tau_values = compute_derived_metric(
                            'Tau',
                            {'CSoil': csoil_data['data'], 'NPP': npp_data['data']}
                        )

                        result[metric][region] = {
                            'years': years,
                            'data': tau_values,
                            'units': config['output_units'],
                            'source': 'UM',
                            'dataset': expt_name
                        }

            else:
                # Direct metric (not derived)
                # Find corresponding variable in extraction output
                var_name = None
                for v, m in VARIABLE_TO_METRIC.items():
                    if m == metric and v in region_data:
                        var_name = v
                        break

                if var_name and var_name in region_data:
                    var_data = region_data[var_name]

                    result[metric][region] = {
                        'years': var_data['years'],
                        'data': var_data['data'],
                        'units': config['output_units'],
                        'source': 'UM',
                        'dataset': expt_name
                    }

    # Validate canonical structure
    validate_canonical_structure(result)

    return result


def compute_metrics_from_raw(
    expt_name: str,
    metrics: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    base_dir: str = '~/dump2hold'
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute canonical metrics from raw monthly UM files.

    Parameters
    ----------
    expt_name : str
        Experiment name (e.g., 'xqhuj')
    metrics : list of str, optional
        Metrics to compute. Default: ['GPP', 'NPP', 'CVeg', 'CSoil', 'NEP']
    start_year : int, optional
        First year to process
    end_year : int, optional
        Last year to process
    base_dir : str, default='~/dump2hold'
        Base directory containing raw monthly files

    Returns
    -------
    dict
        Nested dictionary with canonical schema (global region only):
        {metric: {'global': {
            'years': np.ndarray,
            'data': np.ndarray,
            'units': str,
            'source': 'UM',
            'dataset': str (expt_name)
        }}}

    Examples
    --------
    >>> metrics = compute_metrics_from_raw(
    ...     'xqhuj',
    ...     metrics=['GPP', 'NPP', 'NEP'],
    ...     start_year=1850,
    ...     end_year=1900
    ... )
    >>> metrics['GPP']['global']['data']
    array([110.5, 111.2, ...])

    Notes
    -----
    Raw extraction only provides global totals (no regional breakdown).
    Derived metrics (NEP) are computed automatically if components available.
    """
    if metrics is None:
        metrics = ['GPP', 'NPP', 'CVeg', 'CSoil', 'NEP']

    # Extract raw data (global only)
    raw_data = extract_annual_mean_raw(
        expt_name,
        base_dir=base_dir,
        start_year=start_year,
        end_year=end_year
    )

    # Transform to canonical schema
    result = {}

    # Mapping from raw extraction names to canonical metrics
    raw_to_metric = {
        'GPP': 'GPP',
        'NPP': 'NPP',
        'VegCarb': 'CVeg',
        'soilCarbon': 'CSoil',
        'soilResp': 'soilResp',
        'NEP': 'NEP',  # NEP is already computed by extract_annual_mean_raw
    }

    for metric in metrics:
        result[metric] = {}
        config = get_metric_config(metric)

        # Find corresponding variable in raw data
        raw_var = None
        for raw_name, canonical_name in raw_to_metric.items():
            if canonical_name == metric and raw_name in raw_data:
                raw_var = raw_name
                break

        if raw_var and raw_var in raw_data:
            var_data = raw_data[raw_var]

            result[metric]['global'] = {
                'years': var_data['years'],
                'data': var_data['data'],
                'units': config['output_units'],
                'source': 'UM',
                'dataset': expt_name
            }

    # Validate canonical structure
    validate_canonical_structure(result)

    return result


def merge_um_and_obs_metrics(
    um_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    obs_metrics: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Merge UM and observational metrics into a single structure.

    Parameters
    ----------
    um_metrics : dict
        UM metrics in canonical schema
    obs_metrics : dict
        Observational metrics in canonical schema

    Returns
    -------
    dict
        Combined structure: {source: {metric: {region: canonical_schema}}}
        where source is 'UM', 'CMIP6', or 'RECCAP2'

    Examples
    --------
    >>> combined = merge_um_and_obs_metrics(um_metrics, cmip6_metrics)
    >>> combined['UM']['GPP']['global']['data']
    array([...])
    >>> combined['CMIP6']['GPP']['global']['data']
    array([123.16])
    """
    result = {}

    # Add UM metrics
    if um_metrics:
        source = um_metrics[next(iter(um_metrics))][next(iter(um_metrics[next(iter(um_metrics))]))]['source']
        result['UM'] = um_metrics

    # Add observational metrics
    if obs_metrics:
        source = obs_metrics[next(iter(obs_metrics))][next(iter(obs_metrics[next(iter(obs_metrics))]))]['source']
        result[source] = obs_metrics

    return result


__all__ = [
    'compute_metrics_from_annual_means',
    'compute_metrics_from_raw',
    'merge_um_and_obs_metrics',
]

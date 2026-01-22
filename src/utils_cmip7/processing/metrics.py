"""
Metric definitions and validation for carbon cycle diagnostics.

This module defines how supported metrics are computed, including:
- Aggregation type (SUM vs MEAN)
- Output units
- Unit conversion keys

NO I/O or dataset-specific logic permitted.
"""

import numpy as np
from typing import Dict, List, Optional, Any


# Canonical metric definitions
METRIC_DEFINITIONS = {
    "GPP": {
        "aggregation": "SUM",
        "output_units": "PgC/yr",
        "conversion_key": "GPP",
        "description": "Gross Primary Production",
        "category": "flux",
    },
    "NPP": {
        "aggregation": "SUM",
        "output_units": "PgC/yr",
        "conversion_key": "NPP",
        "description": "Net Primary Production",
        "category": "flux",
    },
    "CVeg": {
        "aggregation": "SUM",
        "output_units": "PgC",
        "conversion_key": "V carb",
        "description": "Vegetation Carbon",
        "category": "stock",
    },
    "CSoil": {
        "aggregation": "SUM",
        "output_units": "PgC",
        "conversion_key": "S carb",
        "description": "Soil Carbon",
        "category": "stock",
    },
    "soilResp": {
        "aggregation": "SUM",
        "output_units": "PgC/yr",
        "conversion_key": "S resp",
        "description": "Soil Respiration",
        "category": "flux",
    },
    "NEP": {
        "aggregation": "DERIVED",
        "output_units": "PgC/yr",
        "formula": "NPP - soilResp",
        "description": "Net Ecosystem Production",
        "category": "flux",
    },
    "Tau": {
        "aggregation": "DERIVED",
        "output_units": "years",
        "formula": "CSoil / NPP",
        "description": "Ecosystem turnover time (soil)",
        "category": "diagnostic",
    },
    "tas": {
        "aggregation": "MEAN",
        "output_units": "°C",
        "conversion_key": "Others",
        "description": "Surface air temperature (1.5m)",
        "category": "climate",
    },
    "temp": {  # Alias for tas
        "aggregation": "MEAN",
        "output_units": "°C",
        "conversion_key": "Others",
        "description": "Surface air temperature (1.5m)",
        "category": "climate",
    },
    "precip": {
        "aggregation": "MEAN",
        "output_units": "mm/day",
        "conversion_key": "precip",
        "description": "Precipitation",
        "category": "climate",
    },
    "fgco2": {
        "aggregation": "SUM",
        "output_units": "PgC/yr",
        "conversion_key": "field646_mm_dpth",
        "description": "Ocean CO2 flux",
        "category": "flux",
    },
    "VegCarb": {  # Alias for CVeg
        "aggregation": "SUM",
        "output_units": "PgC",
        "conversion_key": "V carb",
        "description": "Vegetation Carbon",
        "category": "stock",
    },
    "soilCarbon": {  # Alias for CSoil
        "aggregation": "SUM",
        "output_units": "PgC",
        "conversion_key": "S carb",
        "description": "Soil Carbon",
        "category": "stock",
    },
}


# Canonical schema structure
CANONICAL_SCHEMA = {
    "years": "np.ndarray - Time series of years",
    "data": "np.ndarray - Corresponding data values",
    "units": "str - Physical units (e.g., 'PgC/yr')",
    "source": "str - Data source ('UM', 'CMIP6', 'RECCAP2', etc.)",
    "dataset": "str - Specific dataset/experiment name",
}

OPTIONAL_SCHEMA_FIELDS = {
    "error": "np.ndarray - Uncertainty/error values (optional)",
    "metadata": "dict - Additional metadata (optional)",
}


def get_metric_config(metric_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific metric.

    Parameters
    ----------
    metric_name : str
        Metric name (e.g., 'GPP', 'NPP', 'CVeg', 'CSoil', 'Tau')

    Returns
    -------
    dict
        Metric configuration with keys:
        - aggregation: 'SUM', 'MEAN', or 'DERIVED'
        - output_units: str
        - conversion_key: str (if not derived)
        - formula: str (if derived)
        - description: str
        - category: str

    Raises
    ------
    ValueError
        If metric_name not found in METRIC_DEFINITIONS

    Examples
    --------
    >>> config = get_metric_config('GPP')
    >>> config['aggregation']
    'SUM'
    >>> config['output_units']
    'PgC/yr'
    """
    if metric_name not in METRIC_DEFINITIONS:
        available = ', '.join(METRIC_DEFINITIONS.keys())
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available metrics: {available}"
        )
    return METRIC_DEFINITIONS[metric_name].copy()


def list_metrics(category: Optional[str] = None) -> List[str]:
    """
    List available metrics, optionally filtered by category.

    Parameters
    ----------
    category : str, optional
        Filter by category: 'flux', 'stock', 'climate', 'diagnostic'

    Returns
    -------
    list of str
        Metric names

    Examples
    --------
    >>> list_metrics('flux')
    ['GPP', 'NPP', 'soilResp', 'NEP', 'fgco2']
    >>> list_metrics('stock')
    ['CVeg', 'CSoil', 'VegCarb', 'soilCarbon']
    """
    if category is None:
        return list(METRIC_DEFINITIONS.keys())

    return [
        name for name, config in METRIC_DEFINITIONS.items()
        if config.get('category') == category
    ]


def validate_metric_output(metric_data: Dict[str, Any], metric_name: str = None) -> bool:
    """
    Validate that metric output conforms to canonical schema.

    Parameters
    ----------
    metric_data : dict
        Metric data to validate
    metric_name : str, optional
        Metric name for validation error messages

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValueError
        If metric_data does not conform to canonical schema

    Examples
    --------
    >>> data = {
    ...     'years': np.array([1850, 1851]),
    ...     'data': np.array([123.0, 124.0]),
    ...     'units': 'PgC/yr',
    ...     'source': 'UM',
    ...     'dataset': 'xqhuc'
    ... }
    >>> validate_metric_output(data, 'GPP')
    True
    """
    prefix = f"Metric '{metric_name}'" if metric_name else "Metric data"

    # Check required fields
    required_fields = ['years', 'data', 'units', 'source', 'dataset']
    for field in required_fields:
        if field not in metric_data:
            raise ValueError(f"{prefix}: Missing required field '{field}'")

    # Check types
    if not isinstance(metric_data['years'], np.ndarray):
        raise ValueError(f"{prefix}: 'years' must be numpy array, got {type(metric_data['years'])}")

    if not isinstance(metric_data['data'], np.ndarray):
        raise ValueError(f"{prefix}: 'data' must be numpy array, got {type(metric_data['data'])}")

    if not isinstance(metric_data['units'], str):
        raise ValueError(f"{prefix}: 'units' must be string, got {type(metric_data['units'])}")

    if not isinstance(metric_data['source'], str):
        raise ValueError(f"{prefix}: 'source' must be string, got {type(metric_data['source'])}")

    if not isinstance(metric_data['dataset'], str):
        raise ValueError(f"{prefix}: 'dataset' must be string, got {type(metric_data['dataset'])}")

    # Check array shapes match
    if metric_data['years'].shape != metric_data['data'].shape:
        raise ValueError(
            f"{prefix}: 'years' and 'data' must have same shape. "
            f"Got years: {metric_data['years'].shape}, data: {metric_data['data'].shape}"
        )

    # Check for NaN/Inf in data
    if np.any(np.isnan(metric_data['data'])):
        raise ValueError(f"{prefix}: 'data' contains NaN values")

    if np.any(np.isinf(metric_data['data'])):
        raise ValueError(f"{prefix}: 'data' contains Inf values")

    # Validate optional fields if present
    if 'error' in metric_data:
        if not isinstance(metric_data['error'], np.ndarray):
            raise ValueError(f"{prefix}: 'error' must be numpy array if present")
        if metric_data['error'].shape != metric_data['data'].shape:
            raise ValueError(f"{prefix}: 'error' must have same shape as 'data'")

    return True


def validate_canonical_structure(metrics_dict: Dict[str, Dict[str, Dict[str, Any]]]) -> bool:
    """
    Validate that a complete metrics dictionary conforms to canonical structure.

    Expected structure: dict[metric][region] -> canonical_schema

    Parameters
    ----------
    metrics_dict : dict
        Nested dictionary with structure:
        {metric_name: {region_name: canonical_schema}}

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValueError
        If structure does not conform

    Examples
    --------
    >>> metrics = {
    ...     'GPP': {
    ...         'global': {'years': ..., 'data': ..., 'units': ..., 'source': ..., 'dataset': ...},
    ...         'Europe': {'years': ..., 'data': ..., 'units': ..., 'source': ..., 'dataset': ...}
    ...     }
    ... }
    >>> validate_canonical_structure(metrics)
    True
    """
    if not isinstance(metrics_dict, dict):
        raise ValueError(f"metrics_dict must be dict, got {type(metrics_dict)}")

    for metric_name, regions_dict in metrics_dict.items():
        if not isinstance(regions_dict, dict):
            raise ValueError(
                f"Metric '{metric_name}': regions data must be dict, got {type(regions_dict)}"
            )

        for region_name, region_data in regions_dict.items():
            validate_metric_output(region_data, f"{metric_name}[{region_name}]")

    return True


def compute_derived_metric(
    metric_name: str,
    component_data: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Compute derived metrics from component data.

    Parameters
    ----------
    metric_name : str
        Name of derived metric ('NEP', 'Tau', etc.)
    component_data : dict
        Dictionary mapping component names to data arrays
        e.g., {'NPP': array(...), 'soilResp': array(...)}

    Returns
    -------
    np.ndarray
        Computed derived metric data

    Raises
    ------
    ValueError
        If metric is not derived or components missing

    Examples
    --------
    >>> npp = np.array([56.0, 57.0, 58.0])
    >>> soil_resp = np.array([50.0, 51.0, 52.0])
    >>> nep = compute_derived_metric('NEP', {'NPP': npp, 'soilResp': soil_resp})
    >>> nep
    array([6.0, 6.0, 6.0])
    """
    config = get_metric_config(metric_name)

    if config['aggregation'] != 'DERIVED':
        raise ValueError(f"Metric '{metric_name}' is not a derived metric")

    # Compute based on formula
    if metric_name == 'NEP':
        if 'NPP' not in component_data or 'soilResp' not in component_data:
            raise ValueError("NEP requires 'NPP' and 'soilResp' components")
        return component_data['NPP'] - component_data['soilResp']

    elif metric_name == 'Tau':
        if 'CSoil' not in component_data or 'NPP' not in component_data:
            raise ValueError("Tau requires 'CSoil' and 'NPP' components")
        # Avoid division by zero
        npp = component_data['NPP']
        with np.errstate(divide='ignore', invalid='ignore'):
            tau = component_data['CSoil'] / npp
            tau[npp == 0] = np.nan
        return tau

    else:
        raise ValueError(f"Unknown derived metric '{metric_name}'")


__all__ = [
    'METRIC_DEFINITIONS',
    'CANONICAL_SCHEMA',
    'get_metric_config',
    'list_metrics',
    'validate_metric_output',
    'validate_canonical_structure',
    'compute_derived_metric',
]

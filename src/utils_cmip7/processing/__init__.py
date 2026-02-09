"""
Processing module for spatial, temporal, and regional aggregation.

Provides functions for global/regional aggregation and temporal averaging.
"""

from .spatial import compute_terrestrial_area, global_total_pgC, global_mean_pgC
from .temporal import merge_monthly_results, compute_monthly_mean, compute_annual_mean
from .regional import load_reccap_mask, region_mask, compute_regional_annual_mean
from .metrics import (
    METRIC_DEFINITIONS,
    get_metric_config,
    list_metrics,
    validate_metric_output,
    validate_canonical_structure,
    compute_derived_metric,
)

# Map field extraction (requires iris)
try:
    from .map_fields import extract_map_field, extract_anomaly_field, combine_fields
except ImportError:
    pass

__all__ = [
    # Spatial aggregation
    'compute_terrestrial_area',
    'global_total_pgC',
    'global_mean_pgC',
    # Temporal aggregation
    'merge_monthly_results',
    'compute_monthly_mean',
    'compute_annual_mean',
    # Regional aggregation
    'load_reccap_mask',
    'region_mask',
    'compute_regional_annual_mean',
    # Metric definitions and validation
    'METRIC_DEFINITIONS',
    'get_metric_config',
    'list_metrics',
    'validate_metric_output',
    'validate_canonical_structure',
    'compute_derived_metric',
    # Map field extraction
    'extract_map_field',
    'extract_anomaly_field',
    'combine_fields',
]

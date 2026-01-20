"""
Processing module for spatial, temporal, and regional aggregation.

Provides functions for global/regional aggregation and temporal averaging.
"""

from .spatial import compute_terrestrial_area, global_total_pgC, global_mean_pgC
from .temporal import merge_monthly_results, compute_monthly_mean, compute_annual_mean
from .regional import load_reccap_mask, region_mask, compute_regional_annual_mean

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
]

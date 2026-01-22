"""
Diagnostics module for high-level extraction workflows.

Provides main entry points for extracting carbon cycle variables from
UM output files and computing canonical metrics for validation.
"""

from .extraction import extract_annual_means
from .raw import extract_annual_mean_raw
from .metrics import (
    compute_metrics_from_annual_means,
    compute_metrics_from_raw,
    merge_um_and_obs_metrics,
)

__all__ = [
    # Raw extraction functions
    'extract_annual_means',
    'extract_annual_mean_raw',
    # Canonical metric computation
    'compute_metrics_from_annual_means',
    'compute_metrics_from_raw',
    'merge_um_and_obs_metrics',
]

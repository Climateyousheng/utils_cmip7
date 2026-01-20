"""
Processing module for spatial and temporal aggregation.

Provides functions for global/regional aggregation and temporal averaging.
"""

from .spatial import compute_terrestrial_area, global_total_pgC, global_mean_pgC

__all__ = [
    'compute_terrestrial_area',
    'global_total_pgC',
    'global_mean_pgC',
]

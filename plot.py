"""
Backward-compatible wrapper for plot.py

This module provides backward compatibility for code using the old import pattern:
    from plot import plot_timeseries_grouped

DEPRECATED: This import path is deprecated as of v0.2.0.
As of v0.2.2, plotting functions have been migrated to utils_cmip7.plotting.

New import path (recommended):
    from utils_cmip7.plotting import plot_timeseries_grouped

This legacy import path will be removed in v1.0.
"""

import warnings

# Issue deprecation warning on first import
warnings.warn(
    "Importing from 'plot' is deprecated as of v0.2.0. "
    "Use 'from utils_cmip7.plotting import ...' instead. "
    "This legacy import path will be removed in v1.0.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new package locations (v0.2.2+)
try:
    from utils_cmip7.plotting import (
        group_vars_by_prefix,
        plot_timeseries_grouped,
        plot_regional_pie,
        plot_pft_timeseries,
        plot_regional_pies,
        plot_pft_grouped_bars,
    )
except ImportError:
    # Fallback to legacy file if package not installed
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from plot_legacy import (
        group_vars_by_prefix,
        plot_timeseries_grouped,
        plot_regional_pie,
        plot_pft_timeseries,
        plot_regional_pies,
        plot_pft_grouped_bars,
    )

__all__ = [
    'group_vars_by_prefix',
    'plot_timeseries_grouped',
    'plot_regional_pie',
    'plot_pft_timeseries',
    'plot_regional_pies',
    'plot_pft_grouped_bars',
]

"""
Backward-compatible wrapper for plot.py

This module provides backward compatibility for code using the old import pattern:
    from plot import plot_timeseries_grouped

DEPRECATED: This import path is deprecated as of v0.2.0.
Plotting functions have not yet been migrated to the new package structure.

Migration is planned for v0.2.2, after which the import path will be:
    from utils_cmip7.plotting import plot_timeseries_grouped

This legacy import path will be removed in v1.0.
"""

import warnings
import sys
import os

# Issue deprecation warning on first import
warnings.warn(
    "Importing from 'plot' is deprecated as of v0.2.0. "
    "Plotting functions have not yet been migrated to utils_cmip7.plotting. "
    "This legacy import path will be removed in v1.0.",
    DeprecationWarning,
    stacklevel=2
)

# Import from legacy file
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

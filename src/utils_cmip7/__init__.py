"""
utils_cmip7: Carbon cycle analysis toolkit for Unified Model outputs.

This package provides tools for extracting, processing, and visualizing
carbon cycle variables from UM climate model output files.
"""

__version__ = "0.2.0"

# Core extraction and processing functions
from .io import stash, try_extract, find_matching_files
from .processing import (
    global_total_pgC, 
    global_mean_pgC,
    compute_regional_annual_mean,
    merge_monthly_results,
    compute_monthly_mean,
    compute_annual_mean,
)
from .diagnostics import extract_annual_means, extract_annual_mean_raw
from .config import VAR_CONVERSIONS, RECCAP_MASK_PATH, validate_reccap_mask_path, get_config_info

__all__ = [
    # Version
    '__version__',
    # I/O functions
    'stash',
    'try_extract',
    'find_matching_files',
    # Processing functions
    'global_total_pgC',
    'global_mean_pgC',
    'compute_regional_annual_mean',
    'merge_monthly_results',
    'compute_monthly_mean',
    'compute_annual_mean',
    # High-level diagnostics (main entry points)
    'extract_annual_means',
    'extract_annual_mean_raw',
    # Configuration
    'VAR_CONVERSIONS',
    'RECCAP_MASK_PATH',
    'validate_reccap_mask_path',
    'get_config_info',
]

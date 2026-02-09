"""
Backward-compatible wrapper for analysis.py

This module provides backward compatibility for code using the old import pattern:
    from analysis import extract_annual_means

DEPRECATED: This import path is deprecated as of v0.2.0.
Please update to:
    from utils_cmip7 import extract_annual_means

This wrapper will be removed in v1.0.
"""

import warnings
import sys
import os

# Add src to path for local imports
_src_path = os.path.join(os.path.dirname(__file__), 'src')
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Try importing from new package structure
try:
    # Import all public functions from new package
    from utils_cmip7.io import stash, stash_nc, decode_month, find_matching_files, try_extract
    from utils_cmip7.processing import (
        compute_terrestrial_area,
        global_total_pgC,
        global_mean_pgC,
        merge_monthly_results,
        compute_monthly_mean,
        compute_annual_mean,
        load_reccap_mask,
        region_mask,
        compute_regional_annual_mean,
    )
    from utils_cmip7.diagnostics import extract_annual_means, extract_annual_mean_raw
    from utils_cmip7.config import VAR_CONVERSIONS

    # Issue deprecation warning on first import
    warnings.warn(
        "Importing from 'analysis' is deprecated as of v0.2.0. "
        "Please update your imports to:\n"
        "    from utils_cmip7 import extract_annual_means\n"
        "This legacy import path will be removed in v1.0.",
        DeprecationWarning,
        stacklevel=2
    )

    # Import private helper functions for backward compatibility
    from utils_cmip7.io.extract import (
        _msi_from_stash_obj,
        _msi_from_numeric_stash_code,
        _msi_from_any_attr,
    )

    __all__ = [
        'stash',
        'stash_nc',
        'decode_month',
        'find_matching_files',
        'try_extract',
        'compute_terrestrial_area',
        'global_total_pgC',
        'global_mean_pgC',
        'merge_monthly_results',
        'compute_monthly_mean',
        'compute_annual_mean',
        'load_reccap_mask',
        'region_mask',
        'compute_regional_annual_mean',
        'extract_annual_means',
        'extract_annual_mean_raw',
        'VAR_CONVERSIONS',
        '_msi_from_stash_obj',
        '_msi_from_numeric_stash_code',
        '_msi_from_any_attr',
    ]

except ImportError as e:
    # Fall back to legacy implementation if new package not available
    warnings.warn(
        f"Could not import from utils_cmip7 package ({e}). "
        "Falling back to legacy analysis_legacy.py. "
        "Install package with 'pip install -e .' to use new structure.",
        ImportWarning,
        stacklevel=2
    )

    # Import from legacy file
    sys.path.insert(0, os.path.dirname(__file__))
    from analysis_legacy import *

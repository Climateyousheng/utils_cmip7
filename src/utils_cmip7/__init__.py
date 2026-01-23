"""
utils_cmip7: Carbon cycle analysis toolkit for Unified Model outputs.

This package provides tools for extracting, processing, validating, and visualizing
carbon cycle variables from UM climate model output files.

Submodules:
    io: STASH codes, file discovery, cube extraction, observational data loading
    processing: Spatial/temporal/regional aggregation, metric definitions
    diagnostics: High-level extraction and canonical metric computation
    validation: Model-observation comparison and visualization
    config: Configuration constants and validation
"""

__version__ = "0.2.1"

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
from .config import (
    VAR_CONVERSIONS,
    RECCAP_MASK_PATH,
    CANONICAL_VARIABLES,
    DEFAULT_VAR_LIST,
    validate_reccap_mask_path,
    get_config_info,
    resolve_variable_name,
    get_variable_config,
    get_conversion_key,
)

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
    'CANONICAL_VARIABLES',
    'DEFAULT_VAR_LIST',
    'validate_reccap_mask_path',
    'get_config_info',
    'resolve_variable_name',
    'get_variable_config',
    'get_conversion_key',
]

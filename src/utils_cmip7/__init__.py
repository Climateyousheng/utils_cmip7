"""
utils_cmip7: Carbon cycle analysis toolkit for Unified Model outputs.

This package provides tools for extracting, processing, and visualizing
carbon cycle variables from UM climate model output files.
"""

__version__ = "0.2.0"

# Core extraction and processing functions
from .io import stash, try_extract, find_matching_files
from .processing import global_total_pgC, global_mean_pgC
from .config import VAR_CONVERSIONS, RECCAP_MASK_PATH

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
    # Configuration
    'VAR_CONVERSIONS',
    'RECCAP_MASK_PATH',
]

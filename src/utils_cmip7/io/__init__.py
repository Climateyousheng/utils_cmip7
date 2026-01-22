"""
Input/Output module for Unified Model data files and observational datasets.

Provides STASH code mappings, file discovery, cube extraction utilities,
and observational data loaders.
"""

from .stash import stash, stash_nc
from .file_discovery import decode_month, find_matching_files, MONTH_MAP_ALPHA
from .extract import try_extract
from .obs_loader import (
    load_cmip6_metrics,
    load_reccap_metrics,
    list_available_obs_metrics,
    list_available_obs_regions,
)

__all__ = [
    # STASH codes
    'stash',
    'stash_nc',
    # File discovery
    'decode_month',
    'find_matching_files',
    'MONTH_MAP_ALPHA',
    # Cube extraction
    'try_extract',
    # Observational data
    'load_cmip6_metrics',
    'load_reccap_metrics',
    'list_available_obs_metrics',
    'list_available_obs_regions',
]

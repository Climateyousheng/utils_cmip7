"""
Input/Output module for Unified Model data files.

Provides STASH code mappings, file discovery, and cube extraction utilities.
"""

from .stash import stash, stash_nc
from .file_discovery import decode_month, find_matching_files, MONTH_MAP_ALPHA
from .extract import try_extract

__all__ = [
    'stash',
    'stash_nc',
    'decode_month',
    'find_matching_files',
    'MONTH_MAP_ALPHA',
    'try_extract',
]

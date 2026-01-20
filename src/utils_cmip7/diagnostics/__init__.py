"""
Diagnostics module for high-level extraction workflows.

Provides main entry points for extracting carbon cycle variables from
UM output files.
"""

from .extraction import extract_annual_means
from .raw import extract_annual_mean_raw

__all__ = [
    'extract_annual_means',
    'extract_annual_mean_raw',
]

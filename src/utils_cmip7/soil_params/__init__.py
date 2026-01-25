"""
Soil parameter handling for UM TRIFFID experiments.

Provides structured representation and loading of LAND_CC namelist parameters.
"""

from .params import SoilParamSet, BL_INDEX, DEFAULT_LAND_CC
from .parsers import parse_land_cc_block

__all__ = [
    'SoilParamSet',
    'BL_INDEX',
    'DEFAULT_LAND_CC',
    'parse_land_cc_block',
]

"""
Soil parameter data structures and loaders for UM TRIFFID experiments.

Defines SoilParamSet for storing and manipulating &LAND_CC namelist parameters.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# BL (Broadleaf trees) is at index 0 in all PFT arrays
BL_INDEX = 0

# Default soil parameters from MIGRATION.md
DEFAULT_LAND_CC = {
    'ALPHA': [0.08, 0.08, 0.08, 0.040, 0.08],
    'F0': [0.875, 0.875, 0.900, 0.800, 0.900],
    'G_AREA': [0.004, 0.004, 0.10, 0.10, 0.05],
    'LAI_MIN': [4.0, 4.0, 1.0, 1.0, 1.0],
    'NL0': [0.050, 0.030, 0.060, 0.030, 0.030],
    'R_GROW': [0.25, 0.25, 0.25, 0.25, 0.25],
    'TLOW': [-0.0, -5.0, 0.0, 13.0, 0.0],
    'TUPP': [36.0, 31.0, 36.0, 45.0, 36.0],
    'Q10': 2.0,
    'V_CRIT_ALPHA': 0.343,
    'KAPS': 5e-009,
}


@dataclass
class SoilParamSet:
    """
    Structured representation of UM LAND_CC soil parameters.

    Attributes
    ----------
    ALPHA : list of float
        Quantum efficiency (mol CO2 per mol PAR)
    F0 : list of float
        CI/CA for DQ=0
    G_AREA : list of float
        Leaf turnover rate (/360days)
    LAI_MIN : list of float
        Minimum LAI
    NL0 : list of float
        Top leaf nitrogen concentration (kg N/kg C)
    R_GROW : list of float
        Growth respiration fraction
    TLOW : list of float
        Lower temperature for photosynthesis (°C)
    TUPP : list of float
        Upper temperature for photosynthesis (°C)
    Q10 : float
        Q10 for soil respiration
    V_CRIT_ALPHA : float
        Critical stem volume (m3)
    KAPS : float
        Specific hydraulic conductivity (kg m-2 MPa-1 s-1)
    source : str
        Origin of parameters ('default', 'manual', 'file', 'log')
    metadata : dict
        Additional provenance information
    """
    ALPHA: List[float] = field(default_factory=lambda: DEFAULT_LAND_CC['ALPHA'].copy())
    F0: List[float] = field(default_factory=lambda: DEFAULT_LAND_CC['F0'].copy())
    G_AREA: List[float] = field(default_factory=lambda: DEFAULT_LAND_CC['G_AREA'].copy())
    LAI_MIN: List[float] = field(default_factory=lambda: DEFAULT_LAND_CC['LAI_MIN'].copy())
    NL0: List[float] = field(default_factory=lambda: DEFAULT_LAND_CC['NL0'].copy())
    R_GROW: List[float] = field(default_factory=lambda: DEFAULT_LAND_CC['R_GROW'].copy())
    TLOW: List[float] = field(default_factory=lambda: DEFAULT_LAND_CC['TLOW'].copy())
    TUPP: List[float] = field(default_factory=lambda: DEFAULT_LAND_CC['TUPP'].copy())
    Q10: float = DEFAULT_LAND_CC['Q10']
    V_CRIT_ALPHA: float = DEFAULT_LAND_CC['V_CRIT_ALPHA']
    KAPS: float = DEFAULT_LAND_CC['KAPS']
    source: str = 'unknown'
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_default(cls) -> 'SoilParamSet':
        """
        Create SoilParamSet with default LAND_CC values.

        Returns
        -------
        SoilParamSet
            Parameters from DEFAULT_LAND_CC with source='default'
        """
        return cls(source='default', metadata={'note': 'Default UM TRIFFID soil parameters'})

    @classmethod
    def from_dict(cls, data: Dict[str, Any], source: str = 'manual') -> 'SoilParamSet':
        """
        Create SoilParamSet from dictionary.

        Parameters
        ----------
        data : dict
            Parameter dictionary with keys matching field names
        source : str, default='manual'
            Origin identifier

        Returns
        -------
        SoilParamSet
        """
        # Extract known fields
        params = {}
        for key in ['ALPHA', 'F0', 'G_AREA', 'LAI_MIN', 'NL0', 'R_GROW', 'TLOW', 'TUPP',
                    'Q10', 'V_CRIT_ALPHA', 'KAPS']:
            if key in data:
                params[key] = data[key]

        # Extract metadata if present
        metadata = data.get('metadata', {})

        return cls(**params, source=source, metadata=metadata)

    @classmethod
    def from_file(cls, path: str) -> 'SoilParamSet':
        """
        Load SoilParamSet from JSON or YAML file.

        Parameters
        ----------
        path : str
            Path to parameter file (.json or .yaml/.yml)

        Returns
        -------
        SoilParamSet

        Raises
        ------
        FileNotFoundError
            If file doesn't exist
        ValueError
            If file format is unsupported
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Soil parameter file not found: {path}")

        suffix = path_obj.suffix.lower()

        if suffix == '.json':
            with open(path) as f:
                data = json.load(f)
        elif suffix in ['.yaml', '.yml']:
            if not HAS_YAML:
                raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .json or .yaml/.yml")

        metadata = data.get('metadata', {})
        metadata['source_file'] = str(path)

        return cls.from_dict(data, source='file')

    @classmethod
    def from_log_file(cls, path: str) -> 'SoilParamSet':
        """
        Parse SoilParamSet from UM/Rose namelist log file.

        Parameters
        ----------
        path : str
            Path to log file containing &LAND_CC ... /

        Returns
        -------
        SoilParamSet

        Raises
        ------
        FileNotFoundError
            If file doesn't exist
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Log file not found: {path}")

        with open(path) as f:
            text = f.read()

        return cls.from_log_text(text, source_file=str(path))

    @classmethod
    def from_log_text(cls, text: str, source_file: Optional[str] = None) -> 'SoilParamSet':
        """
        Parse SoilParamSet from log text containing &LAND_CC block.

        Parameters
        ----------
        text : str
            Log text containing &LAND_CC namelist
        source_file : str, optional
            Path to source file for metadata

        Returns
        -------
        SoilParamSet

        Raises
        ------
        ValueError
            If &LAND_CC block not found or parsing fails
        """
        from .parsers import parse_land_cc_block

        data = parse_land_cc_block(text)
        metadata = {'source_file': source_file} if source_file else {}

        return cls.from_dict(data, source='log').with_metadata(**metadata)

    def to_bl_subset(self, bl_index: int = BL_INDEX) -> Dict[str, float]:
        """
        Extract BL-tree parameter subset for overview table.

        Parameters
        ----------
        bl_index : int, default=BL_INDEX (0)
            Index of BL tree in PFT arrays

        Returns
        -------
        dict
            Dictionary with keys like 'ALPHA_BL', 'F0_BL', ..., plus scalars
        """
        bl_subset = {}

        # Array parameters: extract BL index
        for key in ['ALPHA', 'F0', 'G_AREA', 'LAI_MIN', 'NL0', 'R_GROW', 'TLOW', 'TUPP']:
            arr = getattr(self, key)
            bl_subset[f'{key}_BL'] = arr[bl_index]

        # Scalar parameters: include as-is
        bl_subset['Q10'] = self.Q10
        bl_subset['V_CRIT_ALPHA'] = self.V_CRIT_ALPHA
        bl_subset['KAPS'] = self.KAPS

        return bl_subset

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary (for JSON/YAML export).

        Returns
        -------
        dict
        """
        return asdict(self)

    def to_file(self, path: str):
        """
        Save parameters to JSON or YAML file.

        Parameters
        ----------
        path : str
            Output path (.json or .yaml/.yml)
        """
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()

        data = self.to_dict()

        if suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif suffix in ['.yaml', '.yml']:
            if not HAS_YAML:
                raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .json or .yaml/.yml")

    def with_metadata(self, **kwargs) -> 'SoilParamSet':
        """
        Update metadata and return self (for method chaining).

        Parameters
        ----------
        **kwargs
            Metadata key-value pairs

        Returns
        -------
        SoilParamSet
            Self (modified in-place)
        """
        self.metadata.update(kwargs)
        return self


__all__ = [
    'SoilParamSet',
    'BL_INDEX',
    'DEFAULT_LAND_CC',
]

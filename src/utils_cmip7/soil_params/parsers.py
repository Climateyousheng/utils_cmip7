"""
Parsers for UM namelist soil parameters from log files.

Extracts &LAND_CC blocks from UM/Rose namelists.
"""

import re
from typing import Dict, Any, List


def parse_land_cc_block(text: str) -> Dict[str, Any]:
    """
    Parse &LAND_CC namelist block from UM log text.

    Parameters
    ----------
    text : str
        Log file content containing &LAND_CC ... /

    Returns
    -------
    dict
        Parsed parameters with keys matching SoilParamSet fields

    Raises
    ------
    ValueError
        If &LAND_CC block not found or parsing fails

    Examples
    --------
    >>> text = '''
    ... &LAND_CC
    ...  ALPHA=0.08,0.08,0.08,0.040,0.08,
    ...  Q10=2.0,
    ... /
    ... '''
    >>> params = parse_land_cc_block(text)
    >>> params['ALPHA']
    [0.08, 0.08, 0.08, 0.04, 0.08]
    >>> params['Q10']
    2.0
    """
    # Find &LAND_CC block
    pattern = r'&LAND_CC\s+(.*?)\s+/'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if not match:
        raise ValueError("&LAND_CC block not found in log text")

    block_text = match.group(1)

    # Parse key=value pairs
    params = {}

    # Split by comma (but be careful with array values)
    # Pattern: KEY=value1,value2,..., or KEY=value,
    lines = block_text.split('\n')

    for line in lines:
        line = line.strip()
        if not line or '=' not in line:
            continue

        # Split on first = only
        key, value_str = line.split('=', 1)
        key = key.strip()
        value_str = value_str.strip()

        # Remove trailing comma
        value_str = value_str.rstrip(',')

        # Parse value(s)
        if ',' in value_str:
            # Array parameter
            values = []
            for v in value_str.split(','):
                v = v.strip()
                if v:
                    try:
                        values.append(float(v))
                    except ValueError:
                        # Skip non-numeric values
                        pass
            params[key] = values
        else:
            # Scalar parameter
            try:
                params[key] = float(value_str)
            except ValueError:
                # Keep as string if not numeric
                params[key] = value_str

    if not params:
        raise ValueError("No parameters found in &LAND_CC block")

    return params


__all__ = [
    'parse_land_cc_block',
]

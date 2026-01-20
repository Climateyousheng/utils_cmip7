"""
File discovery utilities for Unified Model output files.

Handles UM-style month codes and file pattern matching.
"""

import os
import re
import glob


# UM-style two-letter month codes (alpha format)
MONTH_MAP_ALPHA = {
    "ja": 1,   # January
    "fb": 2,   # February
    "mr": 3,   # March
    "ar": 4,   # April
    "my": 5,   # May
    "jn": 6,   # June
    "jl": 7,   # July
    "ag": 8,   # August
    "sp": 9,   # September
    "ot": 10,  # October
    "nv": 11,  # November
    "dc": 12,  # December
}


def decode_month(mon_code: str) -> int:
    """
    Decode UM-style month codes.

    Supports both alpha codes ('ja', 'fb', 'dc', etc.) and numeric codes
    ('11'-'91' for Jan-Sep, 'a1' for Oct, 'b1' for Nov, 'c1' for Dec).

    Parameters
    ----------
    mon_code : str
        Two-character month code

    Returns
    -------
    int
        Month number (1-12), or 0 if unparseable

    Examples
    --------
    >>> decode_month('ja')
    1
    >>> decode_month('dc')
    12
    >>> decode_month('11')
    1
    >>> decode_month('a1')
    10
    """
    if not mon_code:
        return 0
    s = mon_code.lower()

    # Alpha codes: 'dc', 'sp', etc.
    if s.isalpha():
        return MONTH_MAP_ALPHA.get(s, 0)

    # Numeric/hex-ish codes: '11'..'91','a1','b1','c1'
    # Rule: first char encodes month index; second char is typically '1' (ignore)
    # '1'..'9' => 1..9, 'a' => 10, 'b' => 11, 'c' => 12
    if len(s) == 2:
        first = s[0]
        if first.isdigit():
            m = int(first)
            return m if 1 <= m <= 9 else 0
        if first in ("a", "b", "c"):
            return {"a": 10, "b": 11, "c": 12}[first]

    return 0


def find_matching_files(expt_name, model, up, start_year=None, end_year=None, base_dir="~/dump2hold"):
    """
    Find matching raw UM output files for a given experiment.

    Locates files matching the pattern: {expt_name}{model}#{up}00000{YYYY}{MM}+
    where {MM} is a two-character month code (alpha or numeric).

    Parameters
    ----------
    expt_name : str
        Experiment name (e.g., 'xqhuj', 'xqhuk')
    model : str
        Model identifier (e.g., 'a', 'o')
    up : str
        Stream identifier (e.g., 'pi', 'da')
    start_year : int, optional
        Filter files from this year onwards
    end_year : int, optional
        Filter files up to and including this year
    base_dir : str, default '~/dump2hold'
        Base directory containing experiment subdirectories

    Returns
    -------
    list of tuple
        Sorted list of (year, month, filepath) tuples

    Examples
    --------
    >>> files = find_matching_files('xqhuj', 'a', 'pi', start_year=1850, end_year=1852)
    >>> files[0]
    (1850, 1, '/path/to/xqhuja#pi000001850ja+')

    Notes
    -----
    Supports two month code formats:
    - Alpha: 'ja', 'fb', 'mr', ..., 'dc'
    - Numeric: '11'-'91' (Jan-Sep), 'a1' (Oct), 'b1' (Nov), 'c1' (Dec)

    Examples of valid filenames:
    - xqhuja#pi000001853dc+ (December 1853)
    - xqhujo#da00000185511+ (January 1855)
    """
    base_dir = os.path.expanduser(base_dir)
    datam_path = os.path.join(base_dir, expt_name, "datam")
    if not os.path.isdir(datam_path):
        datam_path = base_dir

    # Month token can be either:
    #   - two letters: [a-zA-Z]{2}
    #   - two chars: [0-9a-cA-C][0-9]
    pattern = (
        fr"{re.escape(expt_name)}[{model}]\#{re.escape(up)}00000"
        fr"(\d{{4}})"                 # year
        fr"([a-zA-Z]{{2}}|[0-9a-cA-C][0-9])"  # month token
        fr"\+"
    )
    regex = re.compile(pattern)

    files = glob.glob(os.path.join(datam_path, "**"), recursive=True)
    matching_files = []

    for f in files:
        match = regex.search(os.path.basename(f)) or regex.search(f)
        if not match:
            continue

        year = int(match.group(1))
        mon_code = match.group(2)
        month = decode_month(mon_code)

        if month == 0:
            continue  # skip unparseable months

        if (start_year is None or year >= start_year) and (end_year is None or year <= end_year):
            matching_files.append((year, month, f))

    matching_files.sort(key=lambda x: (x[0], x[1]))
    return matching_files

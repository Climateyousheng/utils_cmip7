"""
Styling utilities and constants for plotting.

Defines default colors, labels, and helper functions for data grouping.
"""

# Default legend labels for common experiments
DEFAULT_LEGEND_LABELS = {
    "xqhsh": "PI LU COU spinup",
    "xqhuc": "PI HadCM3 spinup",
}

# Default color map for experiments
DEFAULT_COLOR_MAP = {
    "xqhsh": "k",
    "xqhuc": "r",
}


def group_vars_by_prefix(data, expts_list=None, region="global", exclude=("fracPFTs", "frac")):
    """
    Group variables by their common prefix.

    Useful for organizing plot layout where variables with similar prefixes
    should be grouped together.

    Parameters
    ----------
    data : dict
        Nested dictionary: dict[expt][region][var] -> series dict
    expts_list : list, optional
        List of experiments to consider. If None, uses all experiments in data.
    region : str, default='global'
        Region to extract variables from
    exclude : tuple, default=('fracPFTs', 'frac')
        Variable prefixes to exclude (typically nested structures)

    Returns
    -------
    dict
        Dictionary mapping prefix -> sorted list of variable names

    Examples
    --------
    >>> data = {'exp1': {'global': {'GPP_flux': {...}, 'GPP_mean': {...}, 'NPP': {...}}}}
    >>> group_vars_by_prefix(data)
    {'GPP': ['GPP_flux', 'GPP_mean'], 'NPP': ['NPP']}

    Notes
    -----
    Excludes 'fracPFTs' and 'frac' by default since they have nested structure
    """
    grouped = {}

    if expts_list is None:
        expts_list = list(data.keys())

    for expt in expts_list:
        expt_block = data.get(expt, {})
        region_block = expt_block.get(region, {})
        for var in region_block.keys():
            if any(var.startswith(p) for p in exclude):
                continue
            prefix = var.split("_")[0] if "_" in var else var
            grouped.setdefault(prefix, set()).add(var)

    # Return sorted lists for deterministic ordering
    return {k: sorted(v) for k, v in sorted(grouped.items())}


__all__ = [
    'DEFAULT_LEGEND_LABELS',
    'DEFAULT_COLOR_MAP',
    'group_vars_by_prefix',
]

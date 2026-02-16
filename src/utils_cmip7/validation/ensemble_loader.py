"""
Load ensemble parameters from hadcm3b-ensemble-generator logs.

Reads parameter sets from ensemble generator JSON logs and populates
the overview table with soil parameters.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..soil_params import SoilParamSet


def load_ensemble_params_from_logs(
    log_dir: str,
    ensemble_prefix: str
) -> Dict[str, SoilParamSet]:
    """
    Load ensemble parameters from generator log files.

    Parameters
    ----------
    log_dir : str
        Path to ensemble-generator logs directory
        (e.g., /user/home/nd20983/scripts/hadcm3b-ensemble-generator/logs)
    ensemble_prefix : str
        Ensemble name prefix (e.g., 'xqjc')

    Returns
    -------
    dict
        Dictionary mapping experiment ID to SoilParamSet
        {
            'xqjca': SoilParamSet(...),
            'xqjcb': SoilParamSet(...),
            ...
        }

    Raises
    ------
    FileNotFoundError
        If log directory or parameter files not found
    ValueError
        If parameter format is invalid

    Examples
    --------
    >>> params = load_ensemble_params_from_logs(
    ...     '/user/home/nd20983/scripts/hadcm3b-ensemble-generator/logs',
    ...     'xqjc'
    ... )
    >>> params['xqjca'].ALPHA[0]  # BL ALPHA value
    0.08

    Notes
    -----
    Log file naming convention:
    - {prefix}_updated_parameters_YYYYMMDD.json
    - {prefix}_updated_parameters_additional_YYYYMMDD.json

    Parameter mapping:
    - ALPHA, G_AREA, LAI_MIN, NL0, R_GROW, TLOW, TUPP → arrays[5]
    - V_CRIT_ALPHA → scalar
    - F0 → included in log but not in overview table
    - Q10, KAPS → not in log, use defaults
    """
    log_path = Path(log_dir).expanduser()
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    # Find all matching log files
    pattern = f"{ensemble_prefix}_updated_parameters*.json"
    log_files = sorted(log_path.glob(pattern))

    if not log_files:
        raise FileNotFoundError(
            f"No parameter log files found matching {pattern} in {log_dir}"
        )

    # Load all parameter sets
    params_dict = {}
    for log_file in log_files:
        entries = _load_parameter_log(log_file)
        for entry in entries:
            expt_id = entry['ensemble_id']
            param_set = _create_param_set_from_entry(entry, source=str(log_file))
            params_dict[expt_id] = param_set

    return params_dict


def _load_parameter_log(log_file: Path) -> List[Dict[str, Any]]:
    """
    Load parameter entries from a single log file.

    Parameters
    ----------
    log_file : Path
        Path to JSON log file

    Returns
    -------
    list of dict
        List of parameter entries
    """
    with open(log_file, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in {log_file}, got {type(data)}")

    return data


def _create_param_set_from_entry(
    entry: Dict[str, Any],
    source: str
) -> SoilParamSet:
    """
    Create SoilParamSet from log entry.

    Parameters
    ----------
    entry : dict
        Parameter entry from log file with keys:
        ALPHA, G_AREA, F0, LAI_MIN, NL0, R_GROW, TLOW, TUPP, V_CRIT_ALPHA, ensemble_id
    source : str
        Source file path for metadata

    Returns
    -------
    SoilParamSet
        Parameter set with correct field mapping

    Notes
    -----
    - Validates array parameters have length 5 (PFT count)
    - Maps V_CRIT_ALPHA to V_CRIT_ALPHA field
    - Includes F0 even though it's not used in overview table
    - Uses default values for Q10 and KAPS (not in log)
    """
    expt_id = entry.get('ensemble_id', 'unknown')

    # Extract array parameters (must have 5 values for PFTs)
    array_params = {}
    for key in ['ALPHA', 'G_AREA', 'F0', 'LAI_MIN', 'NL0', 'R_GROW', 'TLOW', 'TUPP']:
        if key not in entry:
            raise ValueError(f"Missing required parameter {key} in entry for {expt_id}")

        values = entry[key]
        if not isinstance(values, list):
            raise ValueError(f"Parameter {key} must be a list, got {type(values)}")

        if len(values) != 5:
            raise ValueError(
                f"Parameter {key} must have 5 PFT values, got {len(values)} for {expt_id}"
            )

        array_params[key] = values

    # Extract scalar parameter
    if 'V_CRIT_ALPHA' not in entry:
        raise ValueError(f"Missing V_CRIT_ALPHA in entry for {expt_id}")

    v_crit_alpha = entry['V_CRIT_ALPHA']

    # Handle single-element list (common in logs)
    if isinstance(v_crit_alpha, list):
        if len(v_crit_alpha) != 1:
            raise ValueError(
                f"V_CRIT_ALPHA should be scalar or single-element list, "
                f"got {len(v_crit_alpha)} elements for {expt_id}"
            )
        v_crit_alpha = v_crit_alpha[0]

    # Create parameter set (Q10 and KAPS will use defaults)
    params = SoilParamSet(
        ALPHA=array_params['ALPHA'],
        F0=array_params['F0'],
        G_AREA=array_params['G_AREA'],
        LAI_MIN=array_params['LAI_MIN'],
        NL0=array_params['NL0'],
        R_GROW=array_params['R_GROW'],
        TLOW=array_params['TLOW'],
        TUPP=array_params['TUPP'],
        V_CRIT_ALPHA=v_crit_alpha,
        source='ensemble_generator',
        metadata={
            'experiment_id': expt_id,
            'log_file': source,
        }
    )

    return params


def populate_overview_table_from_logs(
    log_dir: str,
    ensemble_prefix: str,
    overview_csv: str,
    experiment_ids: Optional[List[str]] = None
) -> None:
    """
    Populate overview table with parameters from ensemble generator logs.

    Updates or creates rows in the overview table CSV with soil parameters
    from the ensemble generator logs. Only updates the parameter columns,
    leaving validation metrics (GPP, CVeg, etc.) as NaN until validation runs.

    Parameters
    ----------
    log_dir : str
        Path to ensemble-generator logs directory
    ensemble_prefix : str
        Ensemble name prefix (e.g., 'xqjc')
    overview_csv : str
        Path to overview table CSV file
    experiment_ids : list of str, optional
        Specific experiment IDs to update. If None, updates all found in logs.

    Examples
    --------
    >>> populate_overview_table_from_logs(
    ...     '/user/home/nd20983/scripts/hadcm3b-ensemble-generator/logs',
    ...     'xqjc',
    ...     'validation_outputs/random_sampling_combined_overview_table.csv'
    ... )
    Updated 12 experiments in overview table

    >>> populate_overview_table_from_logs(
    ...     '/user/home/nd20983/scripts/hadcm3b-ensemble-generator/logs',
    ...     'xqjc',
    ...     'validation_outputs/random_sampling_combined_overview_table.csv',
    ...     experiment_ids=['xqjca', 'xqjcb']
    ... )
    Updated 2 experiments in overview table

    Notes
    -----
    - Creates overview CSV if it doesn't exist
    - Preserves existing validation metrics
    - Only updates parameter columns (ALPHA, G_AREA, LAI_MIN, etc.)
    - Uses atomic write to prevent data loss
    """
    from .overview_table import load_overview_table, upsert_overview_row, write_atomic_csv

    # Load parameters from logs
    all_params = load_ensemble_params_from_logs(log_dir, ensemble_prefix)

    # Filter if specific IDs requested
    if experiment_ids is not None:
        params_to_update = {
            exp_id: all_params[exp_id]
            for exp_id in experiment_ids
            if exp_id in all_params
        }
        if len(params_to_update) < len(experiment_ids):
            missing = set(experiment_ids) - set(params_to_update.keys())
            print(f"Warning: {len(missing)} experiments not found in logs: {missing}")
    else:
        params_to_update = all_params

    # Load or create overview table
    overview_path = Path(overview_csv).expanduser()
    overview_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_overview_table(str(overview_path))

    # Update rows
    for exp_id, param_set in params_to_update.items():
        bl_params = param_set.to_overview_table_format()
        # Empty scores dict - will be filled by validation
        scores = {}
        df = upsert_overview_row(df, exp_id, bl_params, scores)

    # Write atomically
    write_atomic_csv(df, str(overview_path))

    print(f"✓ Updated {len(params_to_update)} experiments in overview table: {overview_path}")


__all__ = [
    'load_ensemble_params_from_logs',
    'populate_overview_table_from_logs',
]

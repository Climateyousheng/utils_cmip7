"""
Overview table management for PPE validation tracking.

Handles upserting experiment results into the combined overview table.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


def load_overview_table(path: str) -> pd.DataFrame:
    """
    Load PPE overview table from CSV.

    Parameters
    ----------
    path : str
        Path to overview table CSV

    Returns
    -------
    pd.DataFrame
        Overview table with experiment rows

    Notes
    -----
    If file doesn't exist, returns empty DataFrame with proper structure.
    """
    path_obj = Path(path)

    if not path_obj.exists():
        # Return empty DataFrame with expected columns
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"  ⚠ Warning: Failed to load overview table from {path}: {e}")
        return pd.DataFrame()


def upsert_overview_row(
    df: pd.DataFrame,
    expt_id: str,
    bl_params: Dict[str, float],
    scores: Optional[Dict[str, float]] = None,
    id_column: str = 'ID'
) -> pd.DataFrame:
    """
    Update or insert experiment row in overview table.

    Parameters
    ----------
    df : pd.DataFrame
        Existing overview table
    expt_id : str
        Experiment identifier
    bl_params : dict
        BL-tree soil parameters (keys: ALPHA_BL, F0_BL, ..., Q10, V_CRIT_ALPHA, KAPS)
    scores : dict, optional
        Validation scores/metrics to include
    id_column : str, default='ID'
        Name of experiment ID column

    Returns
    -------
    pd.DataFrame
        Updated overview table

    Notes
    -----
    - If experiment exists: updates row in-place
    - If experiment new: appends new row
    - Only BL parameters + scalars are stored
    """
    # Build row data
    row_data = {id_column: expt_id}
    row_data.update(bl_params)

    if scores:
        row_data.update(scores)

    # Check if experiment already exists
    if not df.empty and id_column in df.columns:
        existing_mask = df[id_column] == expt_id

        if existing_mask.any():
            # Update existing row
            for key, value in row_data.items():
                if key not in df.columns:
                    df[key] = None
                df.loc[existing_mask, key] = value
            return df

    # Append new row
    row_df = pd.DataFrame([row_data])

    if df.empty:
        return row_df
    else:
        return pd.concat([df, row_df], ignore_index=True)


def write_atomic_csv(df: pd.DataFrame, path: str):
    """
    Write DataFrame to CSV atomically (temp file + rename).

    Parameters
    ----------
    df : pd.DataFrame
        Data to write
    path : str
        Destination path

    Notes
    -----
    Uses atomic write pattern:
    1. Write to temporary file
    2. Rename to final path (atomic on POSIX)

    This prevents corruption if process is interrupted.
    """
    path_obj = Path(path)

    # Ensure parent directory exists
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file in same directory
    fd, temp_path = tempfile.mkstemp(
        dir=path_obj.parent,
        prefix=f'.{path_obj.name}.',
        suffix='.tmp'
    )

    try:
        # Close the file descriptor (we'll use pandas to write)
        os.close(fd)

        # Write DataFrame with 5 decimal precision
        df.to_csv(temp_path, index=False, float_format='%.5f')

        # Atomic rename
        shutil.move(temp_path, path)

    except Exception:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


__all__ = [
    'load_overview_table',
    'upsert_overview_row',
    'write_atomic_csv',
]

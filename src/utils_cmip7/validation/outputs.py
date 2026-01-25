"""
Single-experiment validation bundle writing.

Creates structured output directory with all validation artifacts.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


def write_single_validation_bundle(
    outdir: Path,
    expt_id: str,
    soil_params: Any,  # SoilParamSet
    metrics: Optional[Dict] = None,
    scores: Optional[Dict] = None
):
    """
    Write complete single-experiment validation bundle.

    Parameters
    ----------
    outdir : Path
        Base output directory (validation_outputs/)
    expt_id : str
        Experiment identifier
    soil_params : SoilParamSet
        Soil parameters used for this experiment
    metrics : dict, optional
        Computed metrics (already written by main validation)
    scores : dict, optional
        Validation scores for overview table

    Creates
    -------
    {outdir}/single_val_{expt_id}/
        ├── soil_params.json       # Full structured params + provenance
        ├── validation_scores.csv  # Scores for overview table
        └── (metrics CSVs already created by main validation)

    Notes
    -----
    Main validation should have already created:
    - {expt}_metrics.csv
    - {expt}_bias_vs_*.csv
    - comparison_summary.txt
    - plots/

    This function only adds soil_params.json and validation_scores.csv.
    """
    bundle_dir = outdir / f'single_val_{expt_id}'
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Write soil parameters
    soil_params_path = bundle_dir / 'soil_params.json'
    with open(soil_params_path, 'w') as f:
        json.dump(soil_params.to_dict(), f, indent=2)

    print(f"  ✓ Saved soil parameters: {soil_params_path}")

    # Write validation scores if provided
    if scores:
        scores_path = bundle_dir / 'validation_scores.csv'
        scores_df = pd.DataFrame([scores])
        scores_df.to_csv(scores_path, index=False)
        print(f"  ✓ Saved validation scores: {scores_path}")


__all__ = [
    'write_single_validation_bundle',
]

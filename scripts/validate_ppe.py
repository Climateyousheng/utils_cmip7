#!/usr/bin/env python3
"""
High-level validation script for PPE (Perturbed Physics Ensemble) experiments.

Generates comprehensive validation report with visualizations comparing multiple
ensemble members based on their overall performance scores.

Usage:
    python scripts/validate_ppe.py --csv ensemble_results.csv --name soil_tuning_2026
    python scripts/validate_ppe.py --csv results.csv --name my_ensemble --top-n 20 --top-k 40

Outputs:
    - validation_outputs/ppe_{name}/
        ├── ensemble_table.csv          # Copy of input data
        ├── score_distribution.pdf      # Histogram + ECDF of scores
        ├── validation_heatmap.pdf      # Normalized metrics heatmap
        ├── parameter_shifts.pdf        # Parameter distribution shifts
        └── top_experiments.txt         # Text summary
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

from utils_cmip7.plotting import generate_ppe_validation_report


def main():
    """Main workflow for PPE validation report generation."""

    parser = argparse.ArgumentParser(
        description='Generate PPE validation report with comprehensive visualizations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_ppe.py --csv ensemble.csv --name soil_tuning_2026
  python scripts/validate_ppe.py --csv results.csv --name my_ppe --top-n 20 --top-k 40 --q 0.15

Output Structure:
  validation_outputs/ppe_{name}/
    ├── ensemble_table.csv          # Input data copy
    ├── score_distribution.pdf      # Score histogram + ECDF with top-N labels
    ├── validation_heatmap.pdf      # Normalized metrics for top-K experiments
    ├── parameter_shifts.pdf        # Parameter distributions (top vs bottom)
    └── top_experiments.txt         # Text summary with statistics

Input CSV Format:
  Required columns:
    - overall_score (or specify --score-col)

  Optional columns:
    - ID (or specify --id-col) - Experiment identifier
    - Parameter columns (e.g., ALPHA, G_AREA, LAI_MIN, ...)
    - Metric columns (numeric, e.g., rmse_GPP, rmse_NPP, ...)

  RMSE metrics (prefixed with "rmse_") are automatically inverted in heatmap
  so that higher normalized values = better performance.
        """
    )

    # Required arguments
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to ensemble results CSV file'
    )
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Ensemble name for output directory (e.g., soil_tuning_2026)'
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation_outputs',
        help='Base output directory (default: validation_outputs)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=15,
        help='Number of top experiments to highlight in score plots (default: 15)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=30,
        help='Number of experiments to show in heatmap (default: 30)'
    )
    parser.add_argument(
        '--q',
        type=float,
        default=0.10,
        help='Quantile for parameter shift analysis (default: 0.10 = top/bottom 10%%)'
    )
    parser.add_argument(
        '--score-col',
        type=str,
        default='overall_score',
        help='Column name for ranking score (default: overall_score)'
    )
    parser.add_argument(
        '--id-col',
        type=str,
        default='ID',
        help='Column name for experiment ID (default: ID)'
    )
    parser.add_argument(
        '--param-cols',
        type=str,
        default='ALPHA,G_AREA,LAI_MIN,NL0,R_GROW,TLOW,TUPP,V_CRIT',
        help='Comma-separated parameter column names (default: soil params)'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=40,
        help='Number of bins for histograms (default: 40)'
    )

    # Experiment highlighting arguments
    highlight_group = parser.add_argument_group('experiment highlighting')
    highlight_group.add_argument(
        '--highlight',
        type=str,
        action='append',
        help='Experiment(s) to highlight (can be comma-separated or repeated). Example: --highlight xqhuc --highlight xqhua'
    )
    highlight_group.add_argument(
        '--include-highlight',
        action='store_true',
        default=True,
        help='Force-include highlighted experiments even if filtered out (default: True)'
    )
    highlight_group.add_argument(
        '--no-include-highlight',
        dest='include_highlight',
        action='store_false',
        help='Do not force-include highlighted experiments'
    )
    highlight_group.add_argument(
        '--highlight-style',
        choices=['outline', 'marker', 'rowcol', 'both'],
        default='both',
        help='Highlight style for heatmaps (default: both)'
    )
    highlight_group.add_argument(
        '--highlight-label',
        action='store_true',
        default=True,
        help='Add labels to highlighted experiments (default: True)'
    )
    highlight_group.add_argument(
        '--no-highlight-label',
        dest='highlight_label',
        action='store_false',
        help='Disable labels for highlighted experiments'
    )

    args = parser.parse_args()

    # Validate inputs
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    if not (0 < args.q < 0.5):
        print(f"ERROR: --q must be between 0 and 0.5 (got {args.q})")
        sys.exit(1)

    # Parse parameter columns
    param_cols = [c.strip() for c in args.param_cols.split(',') if c.strip()]

    # Parse highlight experiments
    highlight_expts = []
    if args.highlight:
        for h in args.highlight:
            # Support comma-separated lists
            highlight_expts.extend([e.strip() for e in h.split(',') if e.strip()])

    # Generate report
    generate_ppe_validation_report(
        csv_path=str(csv_path),
        ensemble_name=args.name,
        output_dir=args.output_dir,
        top_n=args.top_n,
        top_k=args.top_k,
        q=args.q,
        score_col=args.score_col,
        id_col=args.id_col if args.id_col else None,
        param_cols=param_cols,
        bins=args.bins,
        highlight_expts=highlight_expts if highlight_expts else None,
        include_highlight=args.include_highlight,
        highlight_style=args.highlight_style,
        highlight_label=args.highlight_label,
    )


if __name__ == '__main__':
    main()

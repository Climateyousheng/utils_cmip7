#!/usr/bin/env python3
"""
High-level validation script for PPE (Perturbed Physics Ensemble) experiments.

Generates comprehensive validation report with visualizations comparing multiple
ensemble members based on their overall performance scores.

Usage:
    python scripts/validate_ppe.py xqhuc
    python scripts/validate_ppe.py xqhuc --top-n 20 --top-k 40
    python scripts/validate_ppe.py xqhuc --highlight xqhua,xqhub
    python scripts/validate_ppe.py xqhuc --param-viz --param-viz-vars GPP NPP CVeg

Outputs:
    - validation_outputs/ppe_{expt}/
        ├── ensemble_table.csv          # Copy of input data
        ├── score_distribution.pdf      # Histogram + ECDF of scores
        ├── validation_heatmap.pdf      # Normalized metrics heatmap
        ├── parameter_shifts.pdf        # Parameter distribution shifts
        └── top_experiments.txt         # Text summary
    - validation_outputs/param_viz_{expt}/ (if --param-viz)
        ├── expanded_parameters.csv     # Expanded parameter matrix
        ├── importance_spearman_{var}.csv  # Spearman correlations
        ├── importance_rfperm_{var}.csv    # RF permutation importance
        ├── bar_spearman_{var}.png         # Importance bar charts
        ├── bar_rfperm_{var}.png
        ├── pca_{var}.png                  # PCA embeddings
        └── summary.json                   # Analysis metadata

The specified experiment will automatically be highlighted in all plots.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

from utils_cmip7.plotting import generate_ppe_validation_report, run_param_importance_suite


def main():
    """Main workflow for PPE validation report generation."""

    parser = argparse.ArgumentParser(
        description='Generate PPE validation report with comprehensive visualizations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_ppe.py xqhuc
  python scripts/validate_ppe.py xqhuc --top-n 20 --top-k 40 --q 0.15
  python scripts/validate_ppe.py xqhuc --highlight xqhua,xqhub
  python scripts/validate_ppe.py xqhuc --param-viz --param-viz-vars GPP NPP CVeg CSoil

Output Structure:
  validation_outputs/ppe_{expt}/
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

The specified experiment will be automatically highlighted in all plots.
        """
    )

    # Positional argument for experiment to validate
    parser.add_argument(
        'expt',
        type=str,
        help='Experiment ID to validate (used for output dir and auto-highlighted in plots)'
    )

    # CSV input (optional with default)
    parser.add_argument(
        '--csv',
        type=str,
        default='validation_outputs/random_sampling_combined_overview_table.csv',
        help='Path to ensemble results CSV file (default: validation_outputs/random_sampling_combined_overview_table.csv)'
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
        help='Additional experiment(s) to highlight (can be comma-separated or repeated). Example: --highlight xqhua,xqhub'
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

    # Parameter importance analysis
    param_group = parser.add_argument_group('parameter importance analysis')
    param_group.add_argument(
        '--param-viz',
        action='store_true',
        help='Run parameter importance analysis (Spearman + RandomForest)'
    )
    param_group.add_argument(
        '--param-viz-vars',
        type=str,
        nargs='+',
        help='Variables to analyze (e.g., GPP NPP CVeg CSoil). If not specified, analyzes all skill columns.'
    )
    param_group.add_argument(
        '--param-viz-method',
        choices=['spearman', 'rf', 'both'],
        default='both',
        help='Importance method: spearman (fast), rf (slow, captures nonlinear), or both (default: both)'
    )
    param_group.add_argument(
        '--param-viz-outdir',
        type=str,
        help='Output directory for parameter importance results (default: validation_outputs/param_viz_{expt})'
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
    # Automatically include the target experiment
    highlight_expts = [args.expt]

    # Add any additional highlighted experiments
    if args.highlight:
        for h in args.highlight:
            # Support comma-separated lists
            highlight_expts.extend([e.strip() for e in h.split(',') if e.strip()])

    # Generate report
    generate_ppe_validation_report(
        csv_path=str(csv_path),
        ensemble_name=args.expt,
        output_dir=args.output_dir,
        top_n=args.top_n,
        top_k=args.top_k,
        q=args.q,
        score_col=args.score_col,
        id_col=args.id_col if args.id_col else None,
        param_cols=param_cols,
        bins=args.bins,
        highlight_expts=highlight_expts,
        include_highlight=args.include_highlight,
        highlight_style=args.highlight_style,
        highlight_label=args.highlight_label,
    )

    # Run parameter importance analysis if requested
    if args.param_viz:
        param_viz_outdir = args.param_viz_outdir or f'{args.output_dir}/param_viz_{args.expt}'
        run_param_importance_suite(
            overview_csv=str(csv_path),
            outdir=param_viz_outdir,
            variables=args.param_viz_vars,
            id_col=args.id_col if args.id_col else None,
            param_cols=param_cols,
            method=args.param_viz_method
        )


if __name__ == '__main__':
    main()

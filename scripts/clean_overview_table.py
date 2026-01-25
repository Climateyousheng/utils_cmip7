#!/usr/bin/env python3
"""
Clean overview table by removing extra columns from old code versions.

Keeps only the standard columns:
- ID, soil params, metrics, scores

Removes:
- *_BL parameter duplicates (ALPHA_BL, F0_BL, etc.)
- Scalar params (Q10, V_CRIT_ALPHA, KAPS)
- Bias percentages (*_bias_pct)
- Any other non-standard columns

Usage:
    python scripts/clean_overview_table.py
    python scripts/clean_overview_table.py --csv path/to/table.csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))


# Standard columns that should be kept
STANDARD_COLUMNS = [
    'ID',
    # Soil parameters (BL-tree values)
    'ALPHA', 'G_AREA', 'LAI_MIN', 'NL0', 'R_GROW', 'TLOW', 'TUPP', 'V_CRIT',
    # Carbon cycle metrics
    'GPP', 'NPP', 'CVeg', 'CSoil',
    # Regional tree metrics
    'Tr30SN', 'Tr30-90N', 'AMZTrees',
    # Global mean vegetation fractions
    'GM_BL', 'GM_NL', 'GM_C3', 'GM_C4', 'GM_BS',
    # RMSE values
    'rmse_BL', 'rmse_NL', 'rmse_C3', 'rmse_C4', 'rmse_BS',
    # Overall score
    'overall_score',
]


def clean_overview_table(csv_path: str, output_path: str = None, dry_run: bool = False):
    """
    Clean overview table by removing non-standard columns.

    Parameters
    ----------
    csv_path : str
        Path to overview table CSV
    output_path : str, optional
        Output path (default: overwrite input)
    dry_run : bool
        If True, show what would be removed but don't modify file
    """
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. Install with: pip install pandas")
        sys.exit(1)

    # Read table
    df = pd.read_csv(csv_path)
    print(f"Loaded table from: {csv_path}")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")

    # Find columns to remove
    extra_cols = [c for c in df.columns if c not in STANDARD_COLUMNS]

    if not extra_cols:
        print("✓ Table is clean - no extra columns found")
        return

    print(f"\nFound {len(extra_cols)} extra columns:")
    for col in sorted(extra_cols):
        print(f"  - {col}")

    if dry_run:
        print("\n[DRY RUN] Would remove these columns")
        return

    # Remove extra columns
    df_clean = df[STANDARD_COLUMNS]

    # Write output
    out_path = output_path or csv_path
    df_clean.to_csv(out_path, index=False, float_format='%.5f')

    print(f"\n✓ Cleaned table saved to: {out_path}")
    print(f"  Removed {len(extra_cols)} columns")
    print(f"  Kept {len(STANDARD_COLUMNS)} standard columns")


def main():
    parser = argparse.ArgumentParser(
        description='Clean overview table by removing extra columns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--csv',
        default='validation_outputs/random_sampling_combined_overview_table.csv',
        help='Path to overview table CSV (default: validation_outputs/random_sampling_combined_overview_table.csv)'
    )
    parser.add_argument(
        '--output',
        help='Output path (default: overwrite input file)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without modifying file'
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    clean_overview_table(
        str(csv_path),
        output_path=args.output,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Extract annual means from raw monthly files (pp files) and create plots
Currently only extract carbon cycle variables.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import argparse
import pandas as pd

# Try importing from installed package first, fall back to legacy path
try:
    from utils_cmip7 import extract_annual_mean_raw
    print("✓ Using utils_cmip7 package")
except ImportError:
    # Fall back to legacy path-based import
    sys.path.append(os.path.expanduser('~/scripts/utils_cmip7'))
    from analysis import extract_annual_mean_raw
    print("⚠ Using legacy imports (install package with 'pip install -e .' for new imports)")


def _export_comparison_csv(comparison, output_path):
    """Export comparison dictionary to CSV."""
    rows = []
    for metric, regions_dict in comparison.items():
        for region, comp_data in regions_dict.items():
            row = {
                'metric': metric,
                'region': region,
                **comp_data  # Flatten comparison data
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.5f')
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract annual means from raw monthly UM files and create plots'
    )
    parser.add_argument('expt', help='Experiment name (e.g., xqhuj)')
    parser.add_argument('--outdir', default='./plots', help='Output directory for plots')
    parser.add_argument('--base-dir', default='~/dump2hold', help='Base directory for raw files')
    parser.add_argument('--start-year', type=int, default=None, help='Start year')
    parser.add_argument('--end-year', type=int, default=None, help='End year')
    parser.add_argument('--validate', action='store_true',
                        help='Validate extracted data against observations (global only)')
    parser.add_argument('--validation-outdir', default=None,
                        help='Output directory for validation results (default: validation_outputs/single_val_{expt})')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Extract data
    print(f"\n{'='*80}")
    print(f"EXTRACTING ANNUAL MEANS FROM RAW MONTHLY FILES")
    print(f"{'='*80}\n")

    data = extract_annual_mean_raw(
        expt=args.expt,
        base_dir=args.base_dir,
        start_year=args.start_year,
        end_year=args.end_year
    )

    if not data:
        print("\n❌ No data extracted. Exiting.")
        sys.exit(1)

    # Validation workflow (if requested)
    if args.validate:
        print(f"\n{'='*80}")
        print(f"VALIDATING AGAINST OBSERVATIONS")
        print(f"{'='*80}\n")

        try:
            from utils_cmip7.diagnostics import compute_metrics_from_raw
            from utils_cmip7.io import load_cmip6_metrics, load_reccap_metrics
            from utils_cmip7.validation import compare_metrics, summarize_comparison
            from utils_cmip7.validation import plot_three_way_comparison
        except ImportError as e:
            print(f"❌ Error importing validation modules: {e}")
            print("   Ensure utils_cmip7 is installed with: pip install -e .")
            sys.exit(1)

        # 1. Transform raw data to canonical schema
        print("→ Transforming data to canonical schema...")
        metrics = ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']
        um_metrics = compute_metrics_from_raw(
            expt_name=args.expt,
            metrics=metrics,
            start_year=args.start_year,
            end_year=args.end_year,
            base_dir=args.base_dir
        )

        # Check available metrics
        available_metrics = [m for m in metrics if m in um_metrics and um_metrics[m]]
        if not available_metrics:
            print("❌ No metrics available for validation")
            sys.exit(1)

        print(f"✓ Available metrics: {', '.join(available_metrics)}")

        # 2. Load observational data (global only)
        regions = ['global']
        print("\n→ Loading observational data...")

        try:
            cmip6_metrics = load_cmip6_metrics(available_metrics, regions, include_errors=True)
            print("✓ Loaded CMIP6 metrics")
        except Exception as e:
            print(f"⚠ Warning: Could not load CMIP6 data: {e}")
            cmip6_metrics = None

        try:
            reccap_metrics = load_reccap_metrics(available_metrics, regions, include_errors=True)
            print("✓ Loaded RECCAP2 metrics")
        except Exception as e:
            print(f"⚠ Warning: Could not load RECCAP2 data: {e}")
            reccap_metrics = None

        if not cmip6_metrics and not reccap_metrics:
            print("❌ No observational data loaded. Cannot validate.")
            sys.exit(1)

        # 3. Compare metrics
        print("\n→ Comparing against observations...")
        comparison_cmip6 = None
        comparison_reccap = None

        if cmip6_metrics:
            comparison_cmip6 = compare_metrics(um_metrics, cmip6_metrics, available_metrics, regions)
            print("✓ Compared against CMIP6")

        if reccap_metrics:
            comparison_reccap = compare_metrics(um_metrics, reccap_metrics, available_metrics, regions)
            print("✓ Compared against RECCAP2")

        # 4. Save results to CSV
        outdir = args.validation_outdir or f'validation_outputs/single_val_{args.expt}'
        os.makedirs(outdir, exist_ok=True)

        print(f"\n→ Saving validation results to: {outdir}")

        if comparison_cmip6:
            csv_path = os.path.join(outdir, f'{args.expt}_bias_vs_cmip6.csv')
            _export_comparison_csv(comparison_cmip6, csv_path)

        if comparison_reccap:
            csv_path = os.path.join(outdir, f'{args.expt}_bias_vs_reccap2.csv')
            _export_comparison_csv(comparison_reccap, csv_path)

        # 5. Create validation plots
        plot_dir = os.path.join(outdir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        print(f"\n→ Creating validation plots...")
        for metric in available_metrics:
            if metric in um_metrics:
                try:
                    plot_three_way_comparison(
                        um_metrics, cmip6_metrics, reccap_metrics,
                        metric=metric,
                        outdir=plot_dir
                    )
                    print(f"✓ Created plot for {metric}")
                except Exception as e:
                    print(f"⚠ Warning: Could not create plot for {metric}: {e}")

        # 6. Print summary
        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY (GLOBAL)")
        print(f"{'='*80}\n")

        if comparison_cmip6:
            summary_cmip6 = summarize_comparison(comparison_cmip6)
            print(f"CMIP6 Comparison:")
            print(f"  - Comparisons: {summary_cmip6.get('n_comparisons', 0)}")
            print(f"  - Within uncertainty: {summary_cmip6.get('fraction_within_uncertainty', 0)*100:.1f}%")
            print(f"  - Mean bias: {summary_cmip6.get('mean_bias', 0):.2f}")
            print(f"  - Mean RMSE: {summary_cmip6.get('mean_rmse', 0):.2f}")

        if comparison_reccap:
            summary_reccap = summarize_comparison(comparison_reccap)
            print(f"\nRECCAP2 Comparison:")
            print(f"  - Comparisons: {summary_reccap.get('n_comparisons', 0)}")
            print(f"  - Within uncertainty: {summary_reccap.get('fraction_within_uncertainty', 0)*100:.1f}%")
            print(f"  - Mean bias: {summary_reccap.get('mean_bias', 0):.2f}")
            print(f"  - Mean RMSE: {summary_reccap.get('mean_rmse', 0):.2f}")

        print(f"\n✓ Validation outputs saved to: {os.path.abspath(outdir)}/")
        print(f"{'='*80}\n")

    # Create plots
    print(f"\n{'='*80}")
    print(f"CREATING PLOTS")
    print(f"{'='*80}\n")

    # Plot 1: All variables in separate subplots
    variables = ['GPP', 'NPP', 'Rh', 'CVeg', 'CSoil']
    available_vars = [v for v in variables if v in data]

    if available_vars:
        n_vars = len(available_vars)
        fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3*n_vars), sharex=True)

        if n_vars == 1:
            axes = [axes]

        for ax, var in zip(axes, available_vars):
            years = data[var]['years'][1:]  # Drop first year (spinup)
            values = data[var]['data'][1:]

            ax.plot(years, values, linewidth=0.8, color='steelblue')
            ax.set_ylabel(f"{var}\n({data[var]['units']})", fontsize=9)
            ax.set_title(f"{var} - {args.expt}", fontsize=10)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.tick_params(labelsize=8)

        axes[-1].set_xlabel('Year', fontsize=9)
        plt.tight_layout()

        outfile1 = os.path.join(args.outdir, f'{args.expt}_carbon_cycle_timeseries.png')
        plt.savefig(outfile1, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {outfile1}")
        plt.close()

    # Plot 2: NEP if available
    if 'NEP' in data:
        fig, ax = plt.subplots(figsize=(10, 4))

        years = data['NEP']['years'][1:]
        values = data['NEP']['data'][1:]

        ax.plot(years, values, linewidth=0.8, color='darkgreen')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel(f"NEP ({data['NEP']['units']})", fontsize=9)
        ax.set_title(f"Net Ecosystem Production - {args.expt}", fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)
        plt.tight_layout()

        outfile2 = os.path.join(args.outdir, f'{args.expt}_NEP_timeseries.png')
        plt.savefig(outfile2, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {outfile2}")
        plt.close()

    # Plot 3: Carbon stocks comparison
    if 'CVeg' in data and 'CSoil' in data:
        fig, ax = plt.subplots(figsize=(10, 5))

        years_vc = data['CVeg']['years'][1:]
        values_vc = data['CVeg']['data'][1:]

        years_sc = data['CSoil']['years'][1:]
        values_sc = data['CSoil']['data'][1:]

        ax.plot(years_vc, values_vc, label='Vegetation Carbon', linewidth=0.8, color='green')
        ax.plot(years_sc, values_sc, label='Soil Carbon', linewidth=0.8, color='brown')

        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Carbon (PgC)', fontsize=9)
        ax.set_title(f"Carbon Stocks - {args.expt}", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)
        plt.tight_layout()

        outfile3 = os.path.join(args.outdir, f'{args.expt}_carbon_stocks.png')
        plt.savefig(outfile3, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {outfile3}")
        plt.close()

    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")
    print(f"Plots saved to: {os.path.abspath(args.outdir)}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

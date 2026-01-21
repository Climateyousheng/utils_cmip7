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

# Try importing from installed package first, fall back to legacy path
try:
    from utils_cmip7 import extract_annual_mean_raw
    print("✓ Using utils_cmip7 package")
except ImportError:
    # Fall back to legacy path-based import
    sys.path.append(os.path.expanduser('~/scripts/utils_cmip7'))
    from analysis import extract_annual_mean_raw
    print("⚠ Using legacy imports (install package with 'pip install -e .' for new imports)")


def main():
    parser = argparse.ArgumentParser(
        description='Extract annual means from raw monthly UM files and create plots'
    )
    parser.add_argument('expt', help='Experiment name (e.g., xqhuj)')
    parser.add_argument('--outdir', default='./plots', help='Output directory for plots')
    parser.add_argument('--base-dir', default='~/dump2hold', help='Base directory for raw files')
    parser.add_argument('--start-year', type=int, default=None, help='Start year')
    parser.add_argument('--end-year', type=int, default=None, help='End year')

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

    # Create plots
    print(f"\n{'='*80}")
    print(f"CREATING PLOTS")
    print(f"{'='*80}\n")

    # Plot 1: All variables in separate subplots
    variables = ['GPP', 'NPP', 'soilResp', 'VegCarb', 'soilCarbon']
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
    if 'VegCarb' in data and 'soilCarbon' in data:
        fig, ax = plt.subplots(figsize=(10, 5))

        years_vc = data['VegCarb']['years'][1:]
        values_vc = data['VegCarb']['data'][1:]

        years_sc = data['soilCarbon']['years'][1:]
        values_sc = data['soilCarbon']['data'][1:]

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

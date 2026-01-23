#!/usr/bin/env python3
"""
High-level validation script for UM experiments.

Compares a single UM experiment against CMIP6 and RECCAP2 observational data
for all regions, generates visualizations, and exports results to CSV.

Usage:
    python scripts/validate_experiment.py xqhuc
    python scripts/validate_experiment.py --expt xqhuc --base-dir ~/annual_mean

Outputs:
    - validation_outputs/single_val_{expt}/
        ├── {expt}_metrics.csv           # UM results in obs format
        ├── {expt}_bias_vs_cmip6.csv     # Bias statistics vs CMIP6
        ├── {expt}_bias_vs_reccap2.csv   # Bias statistics vs RECCAP2
        ├── comparison_summary.txt       # Text summary
        └── plots/                       # All comparison plots
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

from utils_cmip7.diagnostics import compute_metrics_from_annual_means
from utils_cmip7.diagnostics.extraction import extract_annual_means
from utils_cmip7.io import load_cmip6_metrics, load_reccap_metrics
from utils_cmip7.validation import (
    compare_metrics,
    summarize_comparison,
    plot_three_way_comparison,
    plot_regional_bias_heatmap,
    plot_timeseries_with_obs,
)
from utils_cmip7.validation.veg_fractions import (
    calculate_veg_metrics,
    save_veg_metrics_to_csv,
    compare_veg_metrics,
    load_obs_veg_metrics,
)
from utils_cmip7.config import RECCAP_REGIONS


def get_all_regions():
    """Get all RECCAP2 regions plus global and Africa."""
    regions = ['global'] + list(RECCAP_REGIONS.values())
    # Ensure Africa is included (it's region 4 in RECCAP_REGIONS)
    if 'Africa' not in regions:
        regions.append('Africa')
    return regions


def save_um_metrics_to_csv(um_metrics, expt, outdir):
    """
    Save UM metrics to CSV in observational data format.

    Parameters
    ----------
    um_metrics : dict
        UM metrics in canonical schema
    expt : str
        Experiment name
    outdir : Path
        Output directory
    """
    regions = get_all_regions()
    metrics = ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']

    # Build dataframe
    data = {}
    for region in regions:
        regional_data = []
        for metric in metrics:
            if metric in um_metrics and region in um_metrics[metric]:
                # Compute time-mean
                mean_val = np.mean(um_metrics[metric][region]['data'])
                regional_data.append(mean_val)
            else:
                regional_data.append(np.nan)
        data[region] = regional_data

    df = pd.DataFrame(data, index=metrics)

    # Save
    csv_path = outdir / f'{expt}_metrics.csv'
    df.to_csv(csv_path)
    print(f"  ✓ Saved UM metrics: {csv_path}")

    return df


def save_bias_statistics(comparison, obs_name, expt, outdir):
    """
    Save bias statistics to CSV.

    Parameters
    ----------
    comparison : dict
        Comparison results from compare_metrics()
    obs_name : str
        Observational dataset name ('CMIP6' or 'RECCAP2')
    expt : str
        Experiment name
    outdir : Path
        Output directory
    """
    regions = get_all_regions()
    metrics = ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']

    rows = []
    for metric in metrics:
        if metric not in comparison:
            continue
        for region in regions:
            if region not in comparison[metric]:
                continue

            comp = comparison[metric][region]
            rows.append({
                'metric': metric,
                'region': region,
                'um_mean': comp['um_mean'],
                'obs_mean': comp['obs_mean'],
                'bias': comp['bias'],
                'bias_percent': comp['bias_percent'],
                'rmse': comp['rmse'],
                'within_uncertainty': comp['within_uncertainty'],
            })

    df = pd.DataFrame(rows)
    csv_path = outdir / f'{expt}_bias_vs_{obs_name.lower()}.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved bias statistics vs {obs_name}: {csv_path}")

    return df


def save_comparison_summary(
    um_metrics,
    cmip6_metrics,
    reccap_metrics,
    comparison_cmip6,
    comparison_reccap,
    expt,
    outdir
):
    """
    Save text summary of validation results.

    Parameters
    ----------
    um_metrics, cmip6_metrics, reccap_metrics : dict
        Metric dictionaries
    comparison_cmip6, comparison_reccap : dict
        Comparison results
    expt : str
        Experiment name
    outdir : Path
        Output directory
    """
    summary_path = outdir / 'comparison_summary.txt'

    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"VALIDATION SUMMARY: {expt}\n")
        f.write("="*80 + "\n\n")

        # UM vs CMIP6
        f.write("UM vs CMIP6 ENSEMBLE\n")
        f.write("-"*80 + "\n")
        for metric in ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']:
            if metric in comparison_cmip6 and metric in um_metrics:
                # Get any available region to determine units
                available_regions = list(um_metrics[metric].keys())
                if available_regions:
                    summary = summarize_comparison(comparison_cmip6, metric=metric)
                    units = um_metrics[metric][available_regions[0]]['units']
                    f.write(f"\n{metric}:\n")
                    f.write(f"  Mean bias: {summary['mean_bias']:.2f} {units}\n")
                    f.write(f"  Mean bias %: {summary['mean_bias_percent']:.1f}%\n")
                    f.write(f"  Fraction within uncertainty: {summary['fraction_within_uncertainty']:.1%}\n")

        # UM vs RECCAP2
        f.write("\n" + "="*80 + "\n")
        f.write("UM vs RECCAP2 OBSERVATIONS\n")
        f.write("-"*80 + "\n")
        for metric in ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']:
            if metric in comparison_reccap and metric in um_metrics:
                # Get any available region to determine units
                available_regions = list(um_metrics[metric].keys())
                if available_regions:
                    summary = summarize_comparison(comparison_reccap, metric=metric)
                    units = um_metrics[metric][available_regions[0]]['units']
                    f.write(f"\n{metric}:\n")
                    f.write(f"  Mean bias: {summary['mean_bias']:.2f} {units}\n")
                    f.write(f"  Mean bias %: {summary['mean_bias_percent']:.1f}%\n")
                    f.write(f"  Fraction within uncertainty: {summary['fraction_within_uncertainty']:.1%}\n")

        # UM vs CMIP6 performance comparison
        f.write("\n" + "="*80 + "\n")
        f.write("UM vs CMIP6 PERFORMANCE (against RECCAP2)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<10} {'Region':<20} {'UM Bias %':<12} {'CMIP6 Bias %':<15} {'Winner':<10}\n")
        f.write("-"*80 + "\n")

        regions = get_all_regions()
        for metric in ['GPP', 'NPP', 'CVeg', 'CSoil']:
            for region in regions:
                if (metric in um_metrics and region in um_metrics[metric] and
                    metric in cmip6_metrics and region in cmip6_metrics[metric] and
                    metric in reccap_metrics and region in reccap_metrics[metric]):

                    um_val = np.mean(um_metrics[metric][region]['data'])
                    cmip6_val = cmip6_metrics[metric][region]['data'][0]
                    reccap_val = reccap_metrics[metric][region]['data'][0]

                    um_bias_pct = 100 * (um_val - reccap_val) / reccap_val
                    cmip6_bias_pct = 100 * (cmip6_val - reccap_val) / reccap_val

                    if abs(um_bias_pct) < abs(cmip6_bias_pct):
                        winner = "UM"
                    elif abs(um_bias_pct) > abs(cmip6_bias_pct):
                        winner = "CMIP6"
                    else:
                        winner = "Tie"

                    f.write(f"{metric:<10} {region:<20} {um_bias_pct:>10.1f}%  {cmip6_bias_pct:>12.1f}%  {winner:<10}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"  ✓ Saved comparison summary: {summary_path}")


def create_all_plots(
    um_metrics,
    cmip6_metrics,
    reccap_metrics,
    comparison_cmip6,
    comparison_reccap,
    outdir
):
    """
    Create all validation plots.

    Parameters
    ----------
    um_metrics, cmip6_metrics, reccap_metrics : dict
        Metric dictionaries
    comparison_cmip6, comparison_reccap : dict
        Comparison results
    outdir : Path
        Output directory for plots
    """
    plots_dir = outdir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    metrics = ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']

    print("\n  Creating plots...")

    # 1. Three-way comparison plots
    print("    - Three-way comparisons")
    for metric in metrics:
        # Check if metric exists in all three datasets with at least one common region
        if (metric in um_metrics and metric in cmip6_metrics and metric in reccap_metrics and
            um_metrics[metric] and cmip6_metrics[metric] and reccap_metrics[metric]):
            plot_three_way_comparison(
                um_metrics,
                cmip6_metrics,
                reccap_metrics,
                metric=metric,
                outdir=plots_dir,
                filename=f'{metric}_three_way.png'
            )
        else:
            print(f"      ⚠ Skipping {metric} (not available in all datasets)")

    # 2. Bias heatmaps
    print("    - Bias heatmaps")
    # Filter to metrics that exist in comparisons
    available_metrics = [m for m in metrics if m in comparison_cmip6 and comparison_cmip6[m]]
    if available_metrics:
        plot_regional_bias_heatmap(
            comparison_cmip6,
            metrics=available_metrics,
            value_type='bias_percent',
            outdir=plots_dir,
            filename='bias_heatmap_vs_cmip6.png'
        )

    available_metrics_reccap = [m for m in metrics if m in comparison_reccap and comparison_reccap[m]]
    if available_metrics_reccap:
        plot_regional_bias_heatmap(
            comparison_reccap,
            metrics=available_metrics_reccap,
            value_type='bias_percent',
            outdir=plots_dir,
            filename='bias_heatmap_vs_reccap2.png'
        )

    # 3. Time series plots (global only)
    print("    - Time series")
    for metric in metrics:
        if (metric in um_metrics and 'global' in um_metrics[metric] and
            metric in reccap_metrics and 'global' in reccap_metrics[metric]):
            plot_timeseries_with_obs(
                um_metrics,
                reccap_metrics,
                metric=metric,
                region='global',
                outdir=plots_dir,
                filename=f'{metric}_timeseries_global.png'
            )
        elif metric in um_metrics and um_metrics[metric]:
            # Try with any available region if global doesn't exist
            available_regions = list(um_metrics[metric].keys())
            if (available_regions and metric in reccap_metrics and
                available_regions[0] in reccap_metrics[metric]):
                plot_timeseries_with_obs(
                    um_metrics,
                    reccap_metrics,
                    metric=metric,
                    region=available_regions[0],
                    outdir=plots_dir,
                    filename=f'{metric}_timeseries_{available_regions[0]}.png'
                )


def main():
    """Main validation workflow."""
    parser = argparse.ArgumentParser(
        description='Validate UM experiment against CMIP6 and RECCAP2 observations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'expt',
        nargs='?',
        help='Experiment name (e.g., xqhuc)'
    )
    parser.add_argument(
        '--expt',
        dest='expt_flag',
        help='Experiment name (alternative flag syntax)'
    )
    parser.add_argument(
        '--base-dir',
        default='~/annual_mean',
        help='Base directory containing annual mean files (default: ~/annual_mean)'
    )

    args = parser.parse_args()

    # Get experiment name from positional or flag argument
    expt = args.expt or args.expt_flag
    if not expt:
        parser.error("Experiment name required (e.g., python validate_experiment.py xqhuc)")

    print("\n" + "="*80)
    print(f"VALIDATION WORKFLOW: {expt}")
    print("="*80)

    # Create output directory
    outdir = Path('validation_outputs') / f'single_val_{expt}'
    outdir.mkdir(parents=True, exist_ok=True)

    # Get all regions
    regions = get_all_regions()
    metrics = ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']

    # =========================================================================
    # Step 1: Compute UM metrics for all regions
    # =========================================================================
    print(f"\n[1/5] Computing UM metrics from {args.base_dir}/{expt}/...")
    print("-"*80)

    um_metrics = compute_metrics_from_annual_means(
        expt_name=expt,
        metrics=metrics,
        regions=regions,
        base_dir=args.base_dir
    )

    print(f"✓ Computed {len(um_metrics)} metrics for {len(regions)} regions")

    # =========================================================================
    # Step 1b: Extract raw data for vegetation fractions
    # =========================================================================
    print(f"\n[1b/6] Extracting vegetation fraction data...")
    print("-"*80)

    # Extract raw data including frac variable
    raw_data = extract_annual_means(
        expts_list=[expt],
        var_list=['frac'],  # Only need frac for vegetation metrics
        regions=regions,
        base_dir=args.base_dir
    )

    # Calculate vegetation fraction metrics
    veg_metrics = calculate_veg_metrics(raw_data, expt)
    if veg_metrics:
        print(f"✓ Computed {len(veg_metrics)} vegetation fraction metrics")
    else:
        print(f"⚠ No vegetation fraction data available")

    # =========================================================================
    # Step 2: Load observational data
    # =========================================================================
    print("\n[2/6] Loading observational data...")
    print("-"*80)

    cmip6_metrics = load_cmip6_metrics(
        metrics=metrics,
        regions=regions,
        include_errors=True
    )

    reccap_metrics = load_reccap_metrics(
        metrics=metrics,
        regions=regions,
        include_errors=True
    )

    print(f"✓ Loaded CMIP6 ensemble data")
    print(f"✓ Loaded RECCAP2 observational data")

    # Load vegetation fraction observations (IGBP)
    obs_veg_metrics = load_obs_veg_metrics()
    if obs_veg_metrics:
        print(f"✓ Loaded IGBP vegetation fraction observations")

    # =========================================================================
    # Step 3: Compare UM vs CMIP6 and UM vs RECCAP2
    # =========================================================================
    print("\n[3/6] Computing comparison statistics...")
    print("-"*80)

    comparison_cmip6 = compare_metrics(
        um_metrics,
        cmip6_metrics,
        metrics=metrics,
        regions=regions
    )

    comparison_reccap = compare_metrics(
        um_metrics,
        reccap_metrics,
        metrics=metrics,
        regions=regions
    )

    print(f"✓ Computed bias statistics vs CMIP6")
    print(f"✓ Computed bias statistics vs RECCAP2")

    # Compare vegetation fractions vs IGBP
    veg_comparison = None
    if veg_metrics and obs_veg_metrics:
        veg_comparison = compare_veg_metrics(veg_metrics, obs_veg_metrics)
        print(f"✓ Computed vegetation fraction bias vs IGBP")

    # =========================================================================
    # Step 4: Export to CSV
    # =========================================================================
    print("\n[4/6] Exporting results to CSV...")
    print("-"*80)

    save_um_metrics_to_csv(um_metrics, expt, outdir)
    save_bias_statistics(comparison_cmip6, 'CMIP6', expt, outdir)
    save_bias_statistics(comparison_reccap, 'RECCAP2', expt, outdir)

    # Save vegetation fraction metrics
    if veg_metrics:
        save_veg_metrics_to_csv(veg_metrics, expt, outdir)

    # Save vegetation fraction comparison
    if veg_comparison:
        veg_df = pd.DataFrame(veg_comparison).T
        veg_df.index.name = 'metric'
        veg_csv_path = outdir / f'{expt}_veg_bias_vs_igbp.csv'
        veg_df.to_csv(veg_csv_path)
        print(f"  ✓ Saved: {veg_csv_path.name}")

    save_comparison_summary(
        um_metrics,
        cmip6_metrics,
        reccap_metrics,
        comparison_cmip6,
        comparison_reccap,
        expt,
        outdir
    )

    # =========================================================================
    # Step 5: Create plots
    # =========================================================================
    print("\n[5/6] Creating validation plots...")
    print("-"*80)

    create_all_plots(
        um_metrics,
        cmip6_metrics,
        reccap_metrics,
        comparison_cmip6,
        comparison_reccap,
        outdir
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {outdir}/")
    print(f"  - {expt}_metrics.csv            (UM results)")
    print(f"  - {expt}_bias_vs_cmip6.csv      (Bias vs CMIP6)")
    print(f"  - {expt}_bias_vs_reccap2.csv    (Bias vs RECCAP2)")
    if veg_metrics:
        print(f"  - {expt}_veg_fractions.csv      (Vegetation fraction metrics)")
    if veg_comparison:
        print(f"  - {expt}_veg_bias_vs_igbp.csv   (Vegetation bias vs IGBP)")
    print(f"  - comparison_summary.txt        (Text summary)")
    print(f"  - plots/                        (All visualizations)")

    # Quick summary
    print(f"\nKey findings vs RECCAP2:")
    if 'GPP' in comparison_reccap:
        gpp_summary = summarize_comparison(comparison_reccap, 'GPP')
        print(f"  GPP: {gpp_summary['mean_bias']:+.1f} PgC/yr ({gpp_summary['mean_bias_percent']:+.1f}%), "
              f"{gpp_summary['fraction_within_uncertainty']:.0%} within uncertainty")
    if 'NPP' in comparison_reccap:
        npp_summary = summarize_comparison(comparison_reccap, 'NPP')
        print(f"  NPP: {npp_summary['mean_bias']:+.1f} PgC/yr ({npp_summary['mean_bias_percent']:+.1f}%), "
              f"{npp_summary['fraction_within_uncertainty']:.0%} within uncertainty")

    if veg_metrics:
        print(f"\nVegetation fractions vs IGBP:")
        if veg_comparison:
            # Show trees
            if 'global_mean_trees' in veg_comparison:
                trees = veg_comparison['global_mean_trees']
                print(f"  Trees (BL+NL): UM={trees['um_value']:.3f}, Obs={trees.get('obs_value', 'N/A'):.3f}, "
                      f"Bias={trees.get('bias', 0):+.3f} ({trees.get('bias_percent', 0):+.1f}%)")
            # Show grass
            if 'global_mean_grass' in veg_comparison:
                grass = veg_comparison['global_mean_grass']
                print(f"  Grass (C3+C4): UM={grass['um_value']:.3f}, Obs={grass.get('obs_value', 'N/A'):.3f}, "
                      f"Bias={grass.get('bias', 0):+.3f} ({grass.get('bias_percent', 0):+.1f}%)")
            # Show shrub
            if 'global_mean_shrub' in veg_comparison:
                shrub = veg_comparison['global_mean_shrub']
                print(f"  Shrub: UM={shrub['um_value']:.3f}, Obs={shrub.get('obs_value', 'N/A'):.3f}, "
                      f"Bias={shrub.get('bias', 0):+.3f} ({shrub.get('bias_percent', 0):+.1f}%)")
        else:
            # Fallback if no comparison
            if 'global_mean_trees' in veg_metrics:
                print(f"  Trees (BL+NL): {veg_metrics['global_mean_trees']:.3f}")
            if 'global_mean_grass' in veg_metrics:
                print(f"  Grass (C3+C4): {veg_metrics['global_mean_grass']:.3f}")
            if 'global_mean_shrub' in veg_metrics:
                print(f"  Shrub: {veg_metrics['global_mean_shrub']:.3f}")

    print("="*80 + "\n")


if __name__ == '__main__':
    main()

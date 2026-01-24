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
    plot_two_way_comparison,
    plot_regional_bias_heatmap,
    plot_timeseries_with_obs,
)
from utils_cmip7.validation.veg_fractions import (
    PFT_MAPPING,
    calculate_veg_metrics,
    load_obs_veg_metrics,
)

# Vegetation metric names for plotting
VEG_METRICS = ['BL', 'NL', 'C3', 'C4', 'shrub', 'bare_soil']
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
    # Include all metrics that have canonical structure (dict with 'data' key)
    metrics = []
    for m in sorted(um_metrics.keys()):
        # Check this is canonical structure (not scalar veg metrics)
        sample_region = next(iter(um_metrics[m].values()), None)
        if isinstance(sample_region, dict) and 'data' in sample_region:
            metrics.append(m)

    # Build dataframe
    data = {}
    for region in regions:
        regional_data = []
        for metric in metrics:
            if metric in um_metrics and region in um_metrics[metric]:
                mean_val = np.mean(um_metrics[metric][region]['data'])
                regional_data.append(mean_val)
            else:
                regional_data.append(np.nan)
        data[region] = regional_data

    df = pd.DataFrame(data, index=metrics)

    # Save
    csv_path = outdir / f'{expt}_metrics.csv'
    df.to_csv(csv_path)
    print(f"  ✓ Saved UM metrics ({len(metrics)} variables): {csv_path}")

    return df


def save_bias_statistics(comparison, obs_name, expt, outdir):
    """
    Save bias statistics to CSV.

    Parameters
    ----------
    comparison : dict
        Comparison results from compare_metrics()
    obs_name : str
        Observational dataset name ('CMIP6', 'RECCAP2', or 'IGBP')
    expt : str
        Experiment name
    outdir : Path
        Output directory
    """
    rows = []
    for metric in sorted(comparison.keys()):
        for region in sorted(comparison[metric].keys()):
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
    comparison_igbp,
    expt,
    outdir
):
    """
    Save text summary of validation results.

    Parameters
    ----------
    um_metrics, cmip6_metrics, reccap_metrics : dict
        Metric dictionaries
    comparison_cmip6, comparison_reccap, comparison_igbp : dict
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

        # UM vs IGBP (vegetation fractions)
        if comparison_igbp:
            f.write("\n" + "="*80 + "\n")
            f.write("UM vs IGBP VEGETATION FRACTIONS\n")
            f.write("-"*80 + "\n")
            for metric in sorted(comparison_igbp.keys()):
                if 'global' in comparison_igbp[metric]:
                    comp = comparison_igbp[metric]['global']
                    f.write(f"\n{metric}:\n")
                    f.write(f"  UM mean: {comp['um_mean']:.4f}\n")
                    f.write(f"  IGBP obs: {comp['obs_mean']:.4f}\n")
                    f.write(f"  Bias: {comp['bias']:+.4f} ({comp['bias_percent']:+.1f}%)\n")

        f.write("\n" + "="*80 + "\n")

    print(f"  ✓ Saved comparison summary: {summary_path}")


def create_all_plots(
    um_metrics,
    cmip6_metrics,
    reccap_metrics,
    comparison_cmip6,
    comparison_reccap,
    comparison_igbp,
    igbp_metrics,
    outdir
):
    """
    Create all validation plots.

    Parameters
    ----------
    um_metrics, cmip6_metrics, reccap_metrics : dict
        Metric dictionaries in canonical schema
    comparison_cmip6, comparison_reccap, comparison_igbp : dict
        Comparison results
    igbp_metrics : dict
        IGBP obs metrics in canonical schema
    outdir : Path
        Output directory for plots
    """
    plots_dir = outdir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    carbon_metrics = ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']
    all_metrics = carbon_metrics + VEG_METRICS

    print("\n  Creating plots...")

    # 1. Three-way comparison plots (carbon metrics: UM vs CMIP6 vs RECCAP2)
    print("    - Three-way comparisons (carbon)")
    for metric in carbon_metrics:
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

    # 1b. Two-way comparison plots (veg metrics: UM vs IGBP)
    print("    - Veg fraction comparisons (UM vs IGBP)")
    for metric in VEG_METRICS:
        if (metric in um_metrics and um_metrics[metric] and
            metric in igbp_metrics and igbp_metrics[metric]):
            plot_two_way_comparison(
                um_metrics,
                igbp_metrics,
                metric=metric,
                outdir=plots_dir,
                filename=f'{metric}_vs_igbp.png'
            )
        else:
            print(f"      ⚠ Skipping {metric} (not available)")

    # 2. Bias heatmaps
    print("    - Bias heatmaps")
    # Carbon metrics vs CMIP6
    available_metrics = [m for m in carbon_metrics if m in comparison_cmip6 and comparison_cmip6[m]]
    if available_metrics:
        plot_regional_bias_heatmap(
            comparison_cmip6,
            metrics=available_metrics,
            value_type='bias_percent',
            outdir=plots_dir,
            filename='bias_heatmap_vs_cmip6.png'
        )

    # Carbon metrics vs RECCAP2
    available_metrics_reccap = [m for m in carbon_metrics if m in comparison_reccap and comparison_reccap[m]]
    if available_metrics_reccap:
        plot_regional_bias_heatmap(
            comparison_reccap,
            metrics=available_metrics_reccap,
            value_type='bias_percent',
            outdir=plots_dir,
            filename='bias_heatmap_vs_reccap2.png'
        )

    # Veg metrics vs IGBP
    available_veg = [m for m in VEG_METRICS if m in comparison_igbp and comparison_igbp[m]]
    if available_veg:
        plot_regional_bias_heatmap(
            comparison_igbp,
            metrics=available_veg,
            value_type='bias_percent',
            outdir=plots_dir,
            filename='bias_heatmap_vs_igbp.png'
        )

    # 3. Time series plots
    print("    - Time series (carbon)")
    for metric in carbon_metrics:
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

    # 3b. Time series for veg metrics (UM vs IGBP)
    print("    - Time series (veg fractions)")
    for metric in VEG_METRICS:
        if (metric in um_metrics and 'global' in um_metrics[metric] and
            metric in igbp_metrics and 'global' in igbp_metrics[metric]):
            plot_timeseries_with_obs(
                um_metrics,
                igbp_metrics,
                metric=metric,
                region='global',
                outdir=plots_dir,
                filename=f'{metric}_timeseries_global.png'
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
    # Step 1: Compute UM metrics (including vegetation fractions) for all regions
    # =========================================================================
    print(f"\n[1/5] Computing UM metrics from {args.base_dir}/{expt}/...")
    print("-"*80)

    # Compute standard carbon cycle metrics
    um_metrics = compute_metrics_from_annual_means(
        expt_name=expt,
        metrics=metrics,
        regions=regions,
        base_dir=args.base_dir
    )
    print(f"✓ Computed {len(um_metrics)} standard metrics for {len(regions)} regions")

    # Extract raw data for vegetation fractions (preserves time series)
    print(f"  Extracting vegetation fraction data...")
    raw_data = extract_annual_means(
        expts_list=[expt],
        var_list=['frac'],
        regions=regions,
        base_dir=args.base_dir
    )

    # Promote PFT time series to um_metrics with canonical structure
    veg_count = 0
    if expt in raw_data:
        for pft_id, pft_name in sorted(PFT_MAPPING.items()):
            pft_key = f'PFT {pft_id}'
            um_metrics[pft_name] = {}
            for region in regions:
                if (region in raw_data[expt] and
                    'frac' in raw_data[expt][region] and
                    pft_key in raw_data[expt][region]['frac']):
                    pft_data = raw_data[expt][region]['frac'][pft_key]
                    um_metrics[pft_name][region] = {
                        'years': pft_data['years'],
                        'data': pft_data['data'],
                        'units': 'fraction',
                        'source': 'UM',
                        'dataset': expt
                    }
            if um_metrics[pft_name]:
                veg_count += 1
            else:
                del um_metrics[pft_name]

    # Also compute scalar veg metrics (for CSV export and RMSE)
    veg_metrics = calculate_veg_metrics(raw_data, expt, regions=regions)

    if veg_count > 0:
        print(f"✓ Promoted {veg_count} PFT time series to metrics")
        print(f"✓ Total metrics: {len(um_metrics)}")
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

    # Load vegetation fraction observations (IGBP) in canonical schema
    obs_veg_metrics = load_obs_veg_metrics()
    igbp_metrics = {}
    if obs_veg_metrics:
        for pft_name in VEG_METRICS:
            if pft_name in obs_veg_metrics:
                igbp_metrics[pft_name] = {
                    'global': {
                        'data': np.array([obs_veg_metrics[pft_name]]),
                        'units': 'fraction',
                        'source': 'IGBP',
                    }
                }
        print(f"✓ Loaded IGBP vegetation fraction observations ({len(igbp_metrics)} PFTs)")

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

    # Compare vegetation fractions vs IGBP using canonical comparison
    comparison_igbp = {}
    if igbp_metrics:
        comparison_igbp = compare_metrics(
            um_metrics, igbp_metrics,
            metrics=VEG_METRICS,
            regions=['global']
        )
        print(f"✓ Computed vegetation fraction bias vs IGBP ({len(comparison_igbp)} PFTs)")

    # =========================================================================
    # Step 4: Export to CSV
    # =========================================================================
    print("\n[4/6] Exporting results to CSV...")
    print("-"*80)

    save_um_metrics_to_csv(um_metrics, expt, outdir)
    save_bias_statistics(comparison_cmip6, 'CMIP6', expt, outdir)
    save_bias_statistics(comparison_reccap, 'RECCAP2', expt, outdir)

    # Save vegetation fraction comparison (global only)
    if comparison_igbp:
        save_bias_statistics(comparison_igbp, 'IGBP', expt, outdir)

    save_comparison_summary(
        um_metrics,
        cmip6_metrics,
        reccap_metrics,
        comparison_cmip6,
        comparison_reccap,
        comparison_igbp,
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
        comparison_igbp,
        igbp_metrics,
        outdir
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {outdir}/")
    print(f"  - {expt}_metrics.csv            (UM results including veg fractions)")
    print(f"  - {expt}_bias_vs_cmip6.csv      (Bias vs CMIP6)")
    print(f"  - {expt}_bias_vs_reccap2.csv    (Bias vs RECCAP2)")
    if comparison_igbp:
        print(f"  - {expt}_bias_vs_IGBP.csv       (Vegetation bias vs IGBP)")
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

    if comparison_igbp:
        print(f"\nVegetation fractions (global) vs IGBP:")
        for pft_name in VEG_METRICS:
            if pft_name in comparison_igbp and 'global' in comparison_igbp[pft_name]:
                c = comparison_igbp[pft_name]['global']
                print(f"  {pft_name}: UM={c['um_mean']:.3f}, Obs={c['obs_mean']:.3f}, "
                      f"Bias={c['bias']:+.3f} ({c['bias_percent']:+.1f}%)")

        # Show spatial RMSE per PFT
        if veg_metrics:
            rmse_keys = sorted([k for k in veg_metrics if k.startswith('rmse_')])
            if rmse_keys:
                print(f"\n  Spatial RMSE vs IGBP (per PFT):")
                for rk in rmse_keys:
                    if 'global' in veg_metrics[rk]:
                        pft_label = rk.replace('rmse_', '')
                        print(f"    {pft_label}: {veg_metrics[rk]['global']:.4f}")

    print("="*80 + "\n")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Complete validation workflow example for utils_cmip7.

This script demonstrates how to:
1. Compute canonical metrics from UM model output
2. Load observational data (CMIP6, RECCAP2)
3. Compare UM vs observations
4. Visualize validation results

Requirements:
- utils_cmip7 package installed (pip install -e .)
- Annual mean files in ~/annual_mean/xqhuc/
- Observational data in obs/ directory
"""

import numpy as np
import matplotlib.pyplot as plt

# Import utils_cmip7 modules
from utils_cmip7.diagnostics import compute_metrics_from_annual_means
from utils_cmip7.io import load_cmip6_metrics, load_reccap_metrics
from utils_cmip7.validation import (
    compare_metrics,
    summarize_comparison,
    print_comparison_table,
    create_validation_report,
)


def main():
    """Run complete validation workflow."""

    print("\n" + "=" * 80)
    print("UTILS_CMIP7 VALIDATION WORKFLOW EXAMPLE")
    print("=" * 80)

    # =========================================================================
    # Step 1: Compute canonical metrics from UM model output
    # =========================================================================
    print("\n[1/5] Computing UM metrics from annual mean files...")
    print("-" * 80)

    um_metrics = compute_metrics_from_annual_means(
        expt_name='xqhuc',
        metrics=['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau'],
        regions=['global', 'North_America', 'South_America', 'Europe', 'Africa']
    )

    print(f"✓ Computed {len(um_metrics)} metrics for {len(um_metrics['GPP'])} regions")
    print(f"  Example: GPP global mean = {np.mean(um_metrics['GPP']['global']['data']):.2f} PgC/yr")

    # =========================================================================
    # Step 2: Load observational data
    # =========================================================================
    print("\n[2/5] Loading observational data...")
    print("-" * 80)

    cmip6_metrics = load_cmip6_metrics(
        metrics=['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau'],
        regions=['global', 'North_America', 'South_America', 'Europe', 'Africa'],
        include_errors=True
    )

    reccap_metrics = load_reccap_metrics(
        metrics=['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau'],
        regions=['global', 'North_America', 'South_America', 'Europe', 'Africa'],
        include_errors=True
    )

    print(f"✓ Loaded CMIP6 data: {len(cmip6_metrics)} metrics")
    print(f"  Example: CMIP6 GPP global = {cmip6_metrics['GPP']['global']['data'][0]:.2f} ± "
          f"{cmip6_metrics['GPP']['global']['error'][0]:.2f} PgC/yr")
    print(f"✓ Loaded RECCAP2 data: {len(reccap_metrics)} metrics")
    print(f"  Example: RECCAP2 GPP global = {reccap_metrics['GPP']['global']['data'][0]:.2f} ± "
          f"{reccap_metrics['GPP']['global']['error'][0]:.2f} PgC/yr")

    # =========================================================================
    # Step 3: Compare UM vs CMIP6
    # =========================================================================
    print("\n[3/5] Comparing UM vs CMIP6...")
    print("-" * 80)

    comparison_cmip6 = compare_metrics(
        um_metrics,
        cmip6_metrics,
        metrics=['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau'],
        regions=['global', 'North_America', 'South_America', 'Europe', 'Africa']
    )

    # Print comparison table
    print_comparison_table(comparison_cmip6, metrics=['GPP', 'NPP'])

    # Summarize GPP comparison
    gpp_summary = summarize_comparison(comparison_cmip6, metric='GPP')
    print(f"\nGPP Summary:")
    print(f"  Mean bias: {gpp_summary['mean_bias']:.2f} PgC/yr")
    print(f"  Mean bias %: {gpp_summary['mean_bias_percent']:.1f}%")
    print(f"  Fraction within uncertainty: {gpp_summary['fraction_within_uncertainty']:.1%}")

    # =========================================================================
    # Step 4: Compare UM vs RECCAP2
    # =========================================================================
    print("\n[4/5] Comparing UM vs RECCAP2...")
    print("-" * 80)

    comparison_reccap = compare_metrics(
        um_metrics,
        reccap_metrics,
        metrics=['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']
    )

    # Print comparison table
    print_comparison_table(comparison_reccap, metrics=['GPP', 'NPP'])

    # =========================================================================
    # Step 5: Create validation report with plots
    # =========================================================================
    print("\n[5/5] Creating validation report...")
    print("-" * 80)

    # Generate CMIP6 validation report
    create_validation_report(
        comparison_cmip6,
        um_metrics,
        cmip6_metrics,
        outdir='./validation_report_cmip6',
        metrics=['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']
    )

    # Generate RECCAP2 validation report
    create_validation_report(
        comparison_reccap,
        um_metrics,
        reccap_metrics,
        outdir='./validation_report_reccap2',
        metrics=['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']
    )

    print("\n" + "=" * 80)
    print("VALIDATION WORKFLOW COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - ./validation_report_cmip6/   (UM vs CMIP6 plots)")
    print("  - ./validation_report_reccap2/ (UM vs RECCAP2 plots)")
    print("\nKey findings:")
    print(f"  - UM GPP mean bias vs CMIP6: {gpp_summary['mean_bias']:.2f} PgC/yr "
          f"({gpp_summary['mean_bias_percent']:.1f}%)")
    print(f"  - {gpp_summary['fraction_within_uncertainty']:.0%} of regions within obs uncertainty")
    print("\n")


if __name__ == '__main__':
    main()

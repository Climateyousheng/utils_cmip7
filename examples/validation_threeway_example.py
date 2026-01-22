#!/usr/bin/env python3
"""
Three-way validation workflow: UM vs CMIP6 vs RECCAP2.

Compares both UM and CMIP6 against RECCAP2 observational reference,
showing relative performance of UM compared to CMIP6 ensemble.

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
from utils_cmip7.validation import plot_three_way_comparison


def main():
    """Run three-way validation workflow."""

    print("\n" + "="*80)
    print("THREE-WAY VALIDATION: UM vs CMIP6 vs RECCAP2")
    print("="*80)

    # =========================================================================
    # Step 1: Compute UM metrics
    # =========================================================================
    print("\n[1/3] Computing UM metrics from annual mean files...")
    print("-"*80)

    um_metrics = compute_metrics_from_annual_means(
        expt_name='xqhuc',
        metrics=['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau'],
        regions=['global', 'North_America', 'South_America', 'Europe', 'Africa']
    )

    print(f"✓ Computed {len(um_metrics)} metrics for {len(um_metrics['GPP'])} regions")

    # =========================================================================
    # Step 2: Load observational data (CMIP6 and RECCAP2)
    # =========================================================================
    print("\n[2/3] Loading observational data...")
    print("-"*80)

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
    print(f"✓ Loaded RECCAP2 data: {len(reccap_metrics)} metrics")

    # =========================================================================
    # Step 3: Create three-way comparison plots
    # =========================================================================
    print("\n[3/3] Creating three-way comparison plots...")
    print("-"*80)

    outdir = './validation_threeway'

    for metric in ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']:
        plot_three_way_comparison(
            um_metrics,
            cmip6_metrics,
            reccap_metrics,
            metric=metric,
            outdir=outdir
        )

    # Create summary comparison table
    print("\n" + "="*80)
    print("SUMMARY: UM vs CMIP6 Performance Against RECCAP2")
    print("="*80)
    print(f"{'Metric':<10} {'Region':<20} {'UM Bias %':<12} {'CMIP6 Bias %':<15} {'Winner':<10}")
    print("="*80)

    for metric in ['GPP', 'NPP']:
        for region in ['global', 'North_America', 'South_America', 'Europe', 'Africa']:
            um_val = np.mean(um_metrics[metric][region]['data'])
            cmip6_val = cmip6_metrics[metric][region]['data'][0]
            reccap_val = reccap_metrics[metric][region]['data'][0]

            um_bias_pct = 100 * (um_val - reccap_val) / reccap_val
            cmip6_bias_pct = 100 * (cmip6_val - reccap_val) / reccap_val

            # Determine winner (smaller absolute bias)
            if abs(um_bias_pct) < abs(cmip6_bias_pct):
                winner = "UM"
            elif abs(um_bias_pct) > abs(cmip6_bias_pct):
                winner = "CMIP6"
            else:
                winner = "Tie"

            print(f"{metric:<10} {region:<20} {um_bias_pct:>10.1f}%  {cmip6_bias_pct:>12.1f}%  {winner:<10}")

    print("="*80)
    print("\n✓ Validation complete! Plots saved to: ./validation_threeway/")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

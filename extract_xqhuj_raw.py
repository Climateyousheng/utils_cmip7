#!/usr/bin/env python3
"""
Extract annual means for xqhuj from raw monthly UM output files
Uses files from ~/dump2hold/xqhuj/datam/ (not pre-processed annual means)
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.expanduser('~/scripts/utils_cmip7'))
from analysis import (
    find_matching_files,
    try_extract,
    compute_monthly_mean,
    merge_monthly_results,
    stash
)
import iris

# Variables to extract
variables = {
    'gpp': 'GPP',
    'npp': 'NPP',
    'rh': 'S resp',
    'cv': 'V carb',
    'cs': 'S carb',
}

expt_name = 'xqhuj'

print(f"{'='*60}")
print(f"Extracting annual means for {expt_name}")
print(f"{'='*60}")

# Find raw monthly output files
print(f"\nSearching for files in ~/dump2hold/{expt_name}/datam/...")
files = find_matching_files(
    expt_name=expt_name,
    model='a',
    up='pi',
    start_year=None,
    end_year=None,
    base_dir='~/dump2hold',
)
print(f"Found {len(files)} monthly files")

# Dictionary to store results
annual_means = {}

# Process each variable
for var_code, var_name in variables.items():
    print(f"\n{'='*60}")
    print(f"Processing {var_name} ({var_code})")
    print(f"{'='*60}")

    monthly_results = []
    files_processed = 0
    files_failed = 0

    for y, m, f in files:
        try:
            # Load cubes from file
            cubes = iris.load(f)

            # Extract the variable
            cube = try_extract(cubes, var_code, stash_lookup_func=stash)

            if not cube:
                files_failed += 1
                continue

            # Compute monthly mean
            mm = compute_monthly_mean(cube[0], var_name)
            monthly_results.append(mm)
            files_processed += 1

        except Exception as e:
            print(f"  ⚠ Error processing {y}-{m:02d}: {e}")
            files_failed += 1
            continue

    if monthly_results:
        # Merge monthly results into annual means
        annual_data = merge_monthly_results(monthly_results)
        annual_means[var_code] = annual_data

        print(f"  ✓ Successfully processed {files_processed} files")
        print(f"  ✓ Got {len(annual_data['years'])} years of data")
        print(f"  Years: {annual_data['years'][0]} - {annual_data['years'][-1]}")
        if files_failed > 0:
            print(f"  ⚠ Failed: {files_failed} files")
    else:
        print(f"  ❌ No data extracted for {var_name}")

# Summary
print(f"\n{'='*60}")
print(f"EXTRACTION SUMMARY")
print(f"{'='*60}")
print(f"Variables successfully extracted: {len(annual_means)}/{len(variables)}")
for var_code, var_name in variables.items():
    if var_code in annual_means:
        print(f"  ✓ {var_name}: {len(annual_means[var_code]['data'])} years")
    else:
        print(f"  ❌ {var_name}: Failed")

# Create plots
if annual_means:
    print(f"\n{'='*60}")
    print("Creating plots...")
    print(f"{'='*60}")

    n_vars = len(annual_means)
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 4*n_vars), sharex=True)

    if n_vars == 1:
        axes = [axes]

    for ax, (var_code, var_name) in zip(axes, variables.items()):
        if var_code not in annual_means:
            continue

        data = annual_means[var_code]
        years = data['years']
        values = data['data']

        ax.plot(years[1:], values[1:], linewidth=0.8)  # Drop first year (spinup)
        ax.set_ylabel(var_name)
        ax.set_title(f"{var_name} - {expt_name}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Year')
    plt.tight_layout()

    output_file = f'{expt_name}_carbon_cycle_timeseries.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {output_file}")
else:
    print(f"\n⚠ No data to plot")

print(f"\n{'='*60}")
print("Done!")
print(f"{'='*60}")

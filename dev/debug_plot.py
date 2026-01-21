#!/usr/bin/env python3
"""
Debug script for plot_timeseries_grouped function
"""

import os
import sys

# Try importing from installed package first, fall back to legacy path
try:
    from utils_cmip7 import extract_annual_means
    from plot import group_vars_by_prefix  # plot.py not yet migrated
    print("✓ Using utils_cmip7 package")
except ImportError:
    # Fall back to legacy path-based import
    sys.path.append(os.path.expanduser('~/scripts/utils_cmip7'))
    from analysis import extract_annual_means
    from plot import group_vars_by_prefix
    print("⚠ Using legacy imports (install package with 'pip install -e .' for new imports)")

# Extract data
print("Extracting annual means...")
ds = extract_annual_means(expts_list=['xqhuc'])

# Check what's in the data structure
print("\n=== Data Structure ===")
for expt in ds.keys():
    print(f"\nExperiment: {expt}")
    for region in ds[expt].keys():
        print(f"  Region: {region}")
        print(f"    Variables: {list(ds[expt][region].keys())}")

# Test the grouping function
print("\n=== Testing group_vars_by_prefix ===")
expts_list = ['xqhuc']
region = 'global'

grouped = group_vars_by_prefix(ds, expts_list=expts_list, region=region, exclude=("fracPFTs",))
print(f"\nGrouped variables for region '{region}':")
for prefix, vars_list in grouped.items():
    print(f"  {prefix}: {vars_list}")

# Flatten to see what would be plotted
all_varnames = [var for group in grouped.values() for var in group]
print(f"\nTotal variables to plot: {len(all_varnames)}")
print(f"Variables: {all_varnames}")

# Check a specific variable's data
print("\n=== Checking data availability ===")
for var in all_varnames[:5]:  # Check first 5 variables
    series = ds.get('xqhuc', {}).get(region, {}).get(var)
    if series:
        print(f"✓ {var}: {len(series.get('data', []))} data points, units: {series.get('units', 'N/A')}")
    else:
        print(f"✗ {var}: NO DATA")

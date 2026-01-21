#!/usr/bin/env python3
"""
Diagnostic script to check data extraction issues
"""

import os
import sys

# Try importing from installed package first, fall back to legacy path
try:
    from utils_cmip7 import extract_annual_means
    print("✓ Using utils_cmip7 package")
except ImportError:
    # Fall back to legacy path-based import
    sys.path.append(os.path.expanduser('~/scripts/utils_cmip7'))
    from analysis import extract_annual_means
    print("⚠ Using legacy imports (install package with 'pip install -e .' for new imports)")

# Extract data
print("=" * 60)
print("Extracting annual means for xqhuc...")
print("=" * 60)

ds = extract_annual_means(expts_list=['xqhuc'])

# Check the structure
print("\n" + "=" * 60)
print("DATA STRUCTURE ANALYSIS")
print("=" * 60)

for expt in ds.keys():
    print(f"\nExperiment: {expt}")
    for region in ds[expt].keys():
        print(f"\n  Region: {region}")
        print(f"  Variables present: {list(ds[expt][region].keys())}")
        print(f"  Number of variables: {len(ds[expt][region].keys())}")

        # Check which variables have data
        for var in ds[expt][region].keys():
            if var == 'fracPFTs':
                pfts = ds[expt][region][var].keys()
                print(f"    - {var}: {list(pfts)}")
            else:
                series = ds[expt][region][var]
                if isinstance(series, dict) and 'data' in series:
                    n_points = len(series['data'])
                    print(f"    ✓ {var}: {n_points} data points, units: {series.get('units', 'N/A')}")
                else:
                    print(f"    ✗ {var}: Unexpected structure")

# Expected variables
expected_vars = ['soilResp', 'soilCarbon', 'VegCarb', 'fracPFTs',
                 'GPP', 'NPP', 'fgco2', 'temp', 'precip',
                 'NEP', 'Land Carbon', 'Trees Total']

print("\n" + "=" * 60)
print("MISSING VARIABLES CHECK")
print("=" * 60)

for region in ds.get('xqhuc', {}).keys():
    print(f"\nRegion: {region}")
    present_vars = set(ds['xqhuc'][region].keys())
    missing_vars = set(expected_vars) - present_vars

    if missing_vars:
        print(f"  ⚠ Missing variables: {sorted(missing_vars)}")
    else:
        print(f"  ✓ All expected variables present")

# Check file paths
print("\n" + "=" * 60)
print("FILE PATH CHECK")
print("=" * 60)

import glob
base_dir = os.path.expanduser('~/annual_mean/xqhuc/')
print(f"\nLooking in: {base_dir}")
print(f"Directory exists: {os.path.isdir(base_dir)}")

if os.path.isdir(base_dir):
    filenames = glob.glob(os.path.join(base_dir, "**/*.nc"), recursive=True)
    print(f"\nNetCDF files found: {len(filenames)}")
    for f in filenames:
        print(f"  - {os.path.basename(f)}")
else:
    print("  ✗ Directory does not exist!")

print("\n" + "=" * 60)

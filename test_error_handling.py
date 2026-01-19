#!/usr/bin/env python3
"""
Test script to demonstrate improved error handling in extract_annual_means
"""

import os
import sys
sys.path.append(os.path.expanduser('~/scripts/utils_cmip7'))

from analysis import extract_annual_means
from plot import plot_timeseries_grouped

print("Testing improved error handling for data extraction...")
print("=" * 80)

# Extract annual means - will now show detailed diagnostics
ds = extract_annual_means(expts_list=['xqhuc'])

print("\n" + "=" * 80)
print("WHAT TO LOOK FOR IN THE OUTPUT ABOVE:")
print("=" * 80)
print("""
1. File Discovery Section:
   - Shows which NetCDF files were found
   - Warns if directory is empty

2. Variable Extraction Section:
   - Shows ✓ for successfully extracted variables
   - Shows ❌ for missing variables with STASH codes
   - Helps identify which files are missing or incomplete

3. Extraction Summary:
   - Lists all found variables
   - Lists all missing variables
   - Warns that missing variables won't appear in plots

If you see many ❌, you need to:
   a) Check if annual mean files exist in ~/annual_mean/xqhuc/
   b) Generate them using annual_mean_cdo.sh script
   c) Verify STASH codes are correct in the files
""")

print("\n" + "=" * 80)
print("GENERATING PLOTS WITH AVAILABLE DATA...")
print("=" * 80)

# Generate plots with whatever data is available
plot_timeseries_grouped(
    ds,
    expts_list=['xqhuc'],
    region='global',
    outdir='./plots/'
)

print("\n✓ Plots generated successfully!")
print(f"Check: ./plots/allvars_global_xqhuc_timeseries.png")
print("\nNote: Plot will only show variables that were successfully extracted.")

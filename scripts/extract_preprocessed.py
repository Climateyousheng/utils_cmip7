#!/usr/bin/env python3
"""
Extract annual means from pre-processed NetCDF files and generate plots
"""

import os
import sys
import argparse

sys.path.append(os.path.expanduser('~/scripts/utils_cmip7'))

from analysis import extract_annual_means
from plot import plot_timeseries_grouped

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Extract annual means from pre-processed NetCDF files')
parser.add_argument('expt', type=str, help='Experiment name (e.g., xqhuc)')
parser.add_argument('--outdir', type=str, default='./plots', help='Output directory for plots (default: ./plots)')
args = parser.parse_args()

print(f"Extracting data for experiment: {args.expt}")
print("=" * 80)

# Extract annual means - will now show detailed diagnostics
ds = extract_annual_means(expts_list=[args.expt])

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
   a) Check if annual mean files exist in ~/annual_mean/{args.expt}/
   b) Generate them using annual_mean_cdo.sh script
   c) Verify STASH codes are correct in the files
""")

print("\n" + "=" * 80)
print("GENERATING PLOTS WITH AVAILABLE DATA...")
print("=" * 80)

# Generate plots with whatever data is available
plot_timeseries_grouped(
    ds,
    expts_list=[args.expt],
    region='global',
    outdir=args.outdir
)

print("\n✓ Plots generated successfully!")
print(f"Check: {args.outdir}/allvars_global_{args.expt}_timeseries.png")
print("\nNote: Plot will only show variables that were successfully extracted.")

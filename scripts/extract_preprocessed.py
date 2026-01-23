#!/usr/bin/env python3
"""
Extract annual means from pre-processed NetCDF files for all regions and generate plots.

Extracts data for all RECCAP2 regions plus global, saves structured output to
validation_outputs/single_val_{expt}/ for consistency with validation workflow.
"""

import os
import sys
import argparse
from pathlib import Path

# Add repository root and scripts directory to path
repo_root = Path(__file__).parent.parent
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(scripts_dir))

# Try importing from installed package first, fall back to legacy path
try:
    from utils_cmip7 import extract_annual_means
    from utils_cmip7.config import RECCAP_REGIONS
    print("✓ Using utils_cmip7 package imports")
    package_imports = True
except ImportError:
    # Fall back to legacy path-based import
    sys.path.append(os.path.expanduser('~/scripts/utils_cmip7'))
    from analysis import extract_annual_means
    print("⚠ Using legacy imports (install package with 'pip install -e .' for new imports)")
    package_imports = False

    # Define fallback RECCAP_REGIONS if not available
    RECCAP_REGIONS = {
        1: "North_America",
        2: "South_America",
        3: "Europe",
        4: "Africa",
        6: "North_Asia",
        7: "Central_Asia",
        8: "East_Asia",
        9: "South_Asia",
        10: "South_East_Asia",
        11: "Oceania",
    }

# Import plotting from repository root (works in both cases)
try:
    from plot_legacy import plot_timeseries_grouped
except ImportError:
    # Fallback to old name
    from plot import plot_timeseries_grouped

# Import helper function from validate_experiment
try:
    from validate_experiment import get_all_regions
except ImportError:
    # Fallback: define locally if import fails
    def get_all_regions():
        """Get all RECCAP2 regions plus global."""
        regions = ['global'] + list(RECCAP_REGIONS.values())
        # Ensure Africa is included (it's region 4 in RECCAP_REGIONS)
        if 'Africa' not in regions:
            regions.append('Africa')
        return regions


def save_extraction_to_csv(ds, expt, outdir):
    """
    Save extracted data to CSV in observational data format.

    Parameters
    ----------
    ds : dict
        Extracted data from extract_annual_means() with structure:
        {expt: {region: {var: {'years': array, 'data': array, 'units': str}}}}
    expt : str
        Experiment name
    outdir : Path
        Output directory

    Returns
    -------
    pandas.DataFrame
        DataFrame with variables as rows and regions as columns
    """
    import numpy as np
    import pandas as pd

    regions = get_all_regions()

    # Get all available variables from the data
    all_vars = set()
    if expt in ds:
        for region in regions:
            if region in ds[expt]:
                all_vars.update(ds[expt][region].keys())

    # Filter out nested structures (like frac which has PFT sub-dictionaries)
    # Keep only variables with 'years' and 'data' keys
    simple_vars = []
    if expt in ds and 'global' in ds[expt]:
        for var in all_vars:
            var_data = ds[expt]['global'].get(var)
            if isinstance(var_data, dict) and 'years' in var_data and 'data' in var_data:
                simple_vars.append(var)

    # Sort variables alphabetically
    variables = sorted(simple_vars)

    if not variables:
        print(f"  ⚠ No variables found to save for {expt}")
        return None

    # Build dataframe: variables as rows, regions as columns
    data = {}
    for region in regions:
        regional_data = []
        for var in variables:
            if expt in ds and region in ds[expt] and var in ds[expt][region]:
                var_data = ds[expt][region][var]
                if 'data' in var_data and len(var_data['data']) > 0:
                    # Compute time-mean
                    mean_val = np.mean(var_data['data'])
                    regional_data.append(mean_val)
                else:
                    regional_data.append(np.nan)
            else:
                regional_data.append(np.nan)
        data[region] = regional_data

    df = pd.DataFrame(data, index=variables)

    # Save
    csv_path = outdir / f'{expt}_extraction.csv'
    df.to_csv(csv_path)
    print(f"  ✓ Saved extraction data: {csv_path}")
    print(f"    Variables: {len(variables)}, Regions: {len(regions)}")

    return df


def main():
    """Main extraction and plotting workflow."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Extract annual means from pre-processed NetCDF files for all regions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/extract_preprocessed.py xqhuc
  python scripts/extract_preprocessed.py xqhuc --base-dir ~/annual_mean

Output Structure:
  validation_outputs/single_val_{expt}/
    ├── {expt}_extraction.csv           # Time-mean values for all regions
    └── plots/
        ├── allvars_global_{expt}_timeseries.png
        ├── allvars_Europe_{expt}_timeseries.png
        ├── allvars_North_America_{expt}_timeseries.png
        └── ...
        """
    )
    parser.add_argument('expt', type=str, help='Experiment name (e.g., xqhuc)')
    parser.add_argument(
        '--base-dir',
        type=str,
        default='~/annual_mean',
        help='Base directory containing annual mean files (default: ~/annual_mean)'
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(f"EXTRACTION WORKFLOW: {args.expt}")
    print("=" * 80)

    # Create output directory structure
    outdir = Path('validation_outputs') / f'single_val_{args.expt}'
    plots_dir = outdir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {outdir}/")
    print(f"Plots directory:  {plots_dir}/")

    # Get all regions
    regions = get_all_regions()
    print(f"\nRegions to extract: {len(regions)} regions")
    print(f"  {', '.join(regions[:4])}, ...")

    print("\n" + "=" * 80)
    print(f"EXTRACTING DATA FOR EXPERIMENT: {args.expt}")
    print("=" * 80)

    # Extract annual means for ALL regions
    ds = extract_annual_means(
        expts_list=[args.expt],
        regions=regions,
        base_dir=args.base_dir
    )

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
   a) Check if annual mean files exist in {args.base_dir}/{args.expt}/
   b) Generate them using annual_mean_cdo.sh script
   c) Verify STASH codes are correct in the files
    """)

    # Save extracted data to CSV
    print("\n" + "=" * 80)
    print("SAVING EXTRACTED DATA TO CSV...")
    print("=" * 80)

    save_extraction_to_csv(ds, args.expt, outdir)

    print("\n" + "=" * 80)
    print("GENERATING PLOTS FOR ALL REGIONS...")
    print("=" * 80)

    # Generate plots for each region
    successfully_plotted = []
    failed_regions = []

    for region in regions:
        # Check if this region has any data
        region_has_data = False
        if args.expt in ds:
            if region in ds[args.expt]:
                region_data = ds[args.expt][region]
                # Check if there are any non-empty variables
                region_has_data = any(
                    isinstance(v, dict) and 'years' in v and len(v['years']) > 0
                    for v in region_data.values()
                )

        if not region_has_data:
            print(f"  ⚠ Skipping {region}: no data available")
            failed_regions.append(region)
            continue

        try:
            plot_timeseries_grouped(
                ds,
                expts_list=[args.expt],
                region=region,
                outdir=str(plots_dir)
            )
            print(f"  ✓ Generated plots for {region}")
            successfully_plotted.append(region)
        except Exception as e:
            print(f"  ❌ Failed to plot {region}: {e}")
            failed_regions.append(region)

    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {outdir}/")
    print(f"\nOutputs:")
    print(f"  - CSV: {args.expt}_extraction.csv (time-mean values for all regions)")
    print(f"  - Plots: {len(successfully_plotted)}/{len(regions)} regional time series")

    if successfully_plotted:
        print("\n✓ Successfully plotted regions:")
        for region in successfully_plotted[:5]:
            print(f"  - {region}")
        if len(successfully_plotted) > 5:
            print(f"  ... and {len(successfully_plotted) - 5} more")

    if failed_regions:
        print(f"\n⚠ Skipped regions (no data or errors): {len(failed_regions)}")
        for region in failed_regions[:5]:
            print(f"  - {region}")
        if len(failed_regions) > 5:
            print(f"  ... and {len(failed_regions) - 5} more")

    print(f"\nData summary: {outdir}/{args.expt}_extraction.csv")
    print(f"Plot directory: {plots_dir}/")
    print("\nNote: CSV contains time-mean values. Plots show full time series.")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()

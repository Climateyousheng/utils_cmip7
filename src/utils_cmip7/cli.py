"""
Command-line interface for utils_cmip7 extraction tools.

Provides CLI entry points registered in pyproject.toml:
- utils-cmip7-extract-raw
- utils-cmip7-extract-preprocessed
"""

import sys
import argparse
from pathlib import Path

from .diagnostics import extract_annual_means, extract_annual_mean_raw
from .config import RECCAP_REGIONS


def get_all_regions():
    """Get all RECCAP2 regions plus global."""
    regions = ['global'] + list(RECCAP_REGIONS.values())
    # Ensure Africa is included
    if 'Africa' not in regions:
        regions.append('Africa')
    return regions


def extract_preprocessed_cli():
    """
    CLI entry point for extracting from pre-processed annual mean NetCDF files.

    Registered as: utils-cmip7-extract-preprocessed
    """
    parser = argparse.ArgumentParser(
        prog='utils-cmip7-extract-preprocessed',
        description='Extract annual means from pre-processed NetCDF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  utils-cmip7-extract-preprocessed xqhuc
  utils-cmip7-extract-preprocessed xqhuc --base-dir ~/annual_mean
  utils-cmip7-extract-preprocessed xqhuc --regions global Europe Africa

Output:
  Extracts carbon cycle variables for specified regions and prints
  structured data to stdout. Use --output to save to CSV.

Variables extracted:
  GPP, NPP, CVeg, CSoil, Tau, precip, tas, and others as available
        """
    )

    parser.add_argument(
        'expt',
        type=str,
        help='Experiment name (e.g., xqhuc)'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default='~/annual_mean',
        help='Base directory containing annual mean files (default: ~/annual_mean)'
    )

    parser.add_argument(
        '--regions',
        nargs='+',
        default=None,
        help='Regions to extract (default: all RECCAP2 regions + global)'
    )

    parser.add_argument(
        '--var-list',
        nargs='+',
        default=None,
        help='Variables to extract (default: all available)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: print to stdout)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Determine regions
    regions = args.regions if args.regions else get_all_regions()

    if args.verbose or not args.output:
        print("=" * 80, file=sys.stderr)
        print(f"EXTRACT PREPROCESSED: {args.expt}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"Base directory: {args.base_dir}", file=sys.stderr)
        print(f"Regions: {', '.join(regions)}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

    # Extract data
    try:
        data = extract_annual_means(
            expts_list=[args.expt],
            regions=regions,
            var_list=args.var_list,
            base_dir=args.base_dir
        )
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)

    # Save or print results
    if args.output:
        _save_extraction_to_csv(data, args.expt, regions, Path(args.output))
        print(f"✓ Saved to {args.output}", file=sys.stderr)
    else:
        # Print structured output to stdout
        _print_extraction_data(data, args.expt, regions)

    if args.verbose or not args.output:
        print("=" * 80, file=sys.stderr)
        print("EXTRACTION COMPLETE", file=sys.stderr)
        print("=" * 80, file=sys.stderr)


def extract_raw_cli():
    """
    CLI entry point for extracting from raw monthly UM files.

    Registered as: utils-cmip7-extract-raw
    """
    parser = argparse.ArgumentParser(
        prog='utils-cmip7-extract-raw',
        description='Extract annual means from raw monthly UM output files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  utils-cmip7-extract-raw xqhuj
  utils-cmip7-extract-raw xqhuj --base-dir ~/dump2hold
  utils-cmip7-extract-raw xqhuj --start-year 2000 --end-year 2010
  utils-cmip7-extract-raw xqhuj --output results.csv

Output:
  Extracts global carbon cycle variables from raw monthly files and
  prints time series data to stdout. Use --output to save to CSV.

Variables extracted:
  GPP, NPP, soilResp, VegCarb, soilCarbon, NEP
        """
    )

    parser.add_argument(
        'expt',
        type=str,
        help='Experiment name (e.g., xqhuj)'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default='~/dump2hold',
        help='Base directory containing raw monthly files (default: ~/dump2hold)'
    )

    parser.add_argument(
        '--start-year',
        type=int,
        default=None,
        help='Start year (default: process all available years)'
    )

    parser.add_argument(
        '--end-year',
        type=int,
        default=None,
        help='End year (default: process all available years)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: print to stdout)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    if args.verbose or not args.output:
        print("=" * 80, file=sys.stderr)
        print(f"EXTRACT RAW: {args.expt}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"Base directory: {args.base_dir}", file=sys.stderr)
        if args.start_year or args.end_year:
            print(f"Year range: {args.start_year or 'start'} - {args.end_year or 'end'}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

    # Extract data
    try:
        data = extract_annual_mean_raw(
            expt=args.expt,
            base_dir=args.base_dir,
            start_year=args.start_year,
            end_year=args.end_year
        )
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print("No data extracted", file=sys.stderr)
        sys.exit(1)

    # Save or print results
    if args.output:
        _save_raw_to_csv(data, Path(args.output))
        print(f"✓ Saved to {args.output}", file=sys.stderr)
    else:
        # Print structured output to stdout
        _print_raw_data(data)

    if args.verbose or not args.output:
        print("=" * 80, file=sys.stderr)
        print("EXTRACTION COMPLETE", file=sys.stderr)
        print("=" * 80, file=sys.stderr)


def _save_extraction_to_csv(data, expt, regions, output_path):
    """
    Save preprocessed extraction data to CSV.

    Parameters
    ----------
    data : dict
        Extracted data from extract_annual_means()
    expt : str
        Experiment name
    regions : list
        Regions extracted
    output_path : Path
        Output CSV file path
    """
    import pandas as pd
    import numpy as np

    # Get all variables
    all_vars = set()
    if expt in data:
        for region in regions:
            if region in data[expt]:
                all_vars.update(data[expt][region].keys())

    # Filter to simple variables (have 'years' and 'data' keys)
    simple_vars = []
    if expt in data and regions and regions[0] in data[expt]:
        for var in all_vars:
            var_data = data[expt][regions[0]].get(var)
            if isinstance(var_data, dict) and 'years' in var_data and 'data' in var_data:
                simple_vars.append(var)

    variables = sorted(simple_vars)

    # Build DataFrame: variables as rows, regions as columns
    df_data = {}
    for region in regions:
        regional_data = []
        for var in variables:
            if expt in data and region in data[expt] and var in data[expt][region]:
                var_data = data[expt][region][var]
                if 'data' in var_data and len(var_data['data']) > 0:
                    mean_val = np.mean(var_data['data'])
                    regional_data.append(mean_val)
                else:
                    regional_data.append(np.nan)
            else:
                regional_data.append(np.nan)
        df_data[region] = regional_data

    df = pd.DataFrame(df_data, index=variables)
    df.to_csv(output_path, float_format='%.5f')


def _save_raw_to_csv(data, output_path):
    """
    Save raw extraction data to CSV.

    Parameters
    ----------
    data : dict
        Extracted data from extract_annual_mean_raw()
    output_path : Path
        Output CSV file path
    """
    import pandas as pd

    # Build DataFrame: years as rows, variables as columns
    variables = sorted(data.keys())

    # Get years from first variable
    years = data[variables[0]]['years']

    df_data = {'year': years}
    for var in variables:
        df_data[var] = data[var]['data']

    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False, float_format='%.5f')


def _print_extraction_data(data, expt, regions):
    """Print preprocessed extraction data to stdout in structured format."""
    import numpy as np

    if expt not in data:
        print("No data found for experiment", file=sys.stderr)
        return

    # Get variables from first region
    first_region = regions[0] if regions else 'global'
    if first_region not in data[expt]:
        print(f"No data found for region {first_region}", file=sys.stderr)
        return

    variables = sorted([
        var for var in data[expt][first_region].keys()
        if isinstance(data[expt][first_region][var], dict)
        and 'years' in data[expt][first_region][var]
        and 'data' in data[expt][first_region][var]
    ])

    # Print header
    print(f"# Experiment: {expt}")
    print(f"# Variables: {', '.join(variables)}")
    print(f"# Regions: {', '.join(regions)}")
    print()

    # Print time-mean values
    print("# Time-mean values:")
    for var in variables:
        print(f"\n{var}:")
        for region in regions:
            if region in data[expt] and var in data[expt][region]:
                var_data = data[expt][region][var]
                if 'data' in var_data and len(var_data['data']) > 0:
                    mean_val = np.mean(var_data['data'])
                    units = var_data.get('units', '')
                    print(f"  {region:20s}: {mean_val:10.5f} {units}")


def _print_raw_data(data):
    """Print raw extraction data to stdout in structured format."""
    import numpy as np

    variables = sorted(data.keys())

    # Print header
    print(f"# Variables: {', '.join(variables)}")
    print()

    # Print time series
    for var in variables:
        years = data[var]['years']
        values = data[var]['data']
        units = data[var].get('units', '')

        print(f"\n{var} ({units}):")
        print(f"  Years: {years[0]} - {years[-1]}")
        print(f"  Mean: {np.mean(values):.5f}")
        print(f"  Min: {np.min(values):.5f}")
        print(f"  Max: {np.max(values):.5f}")


__all__ = [
    'extract_preprocessed_cli',
    'extract_raw_cli',
]

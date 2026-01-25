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


def validate_experiment_cli():
    """
    CLI entry point for validating a single UM experiment.

    Registered as: utils-cmip7-validate-experiment
    """
    # Import here to avoid heavy dependencies if not needed
    import numpy as np
    import pandas as pd
    from pathlib import Path as PathLib

    from .diagnostics import compute_metrics_from_annual_means, extract_annual_means
    from .io import load_cmip6_metrics, load_reccap_metrics
    from .validation import (
        compare_metrics,
        summarize_comparison,
        plot_three_way_comparison,
        plot_two_way_comparison,
        plot_regional_bias_heatmap,
        plot_timeseries_with_obs,
        load_overview_table,
        upsert_overview_row,
        write_atomic_csv,
        write_single_validation_bundle,
    )
    from .validation.veg_fractions import (
        PFT_MAPPING,
        calculate_veg_metrics,
        load_obs_veg_metrics,
    )
    from .soil_params import SoilParamSet

    parser = argparse.ArgumentParser(
        prog='utils-cmip7-validate-experiment',
        description='Validate UM experiment against CMIP6 and RECCAP2 observations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  utils-cmip7-validate-experiment xqhuc --use-default-soil-params
  utils-cmip7-validate-experiment xqhuc --soil-log-file rose.log
  utils-cmip7-validate-experiment xqhuc --soil-param-file params.json --base-dir ~/annual_mean

Output:
  Creates validation_outputs/single_val_{expt}/ containing:
  - Metrics CSV files
  - Bias statistics vs CMIP6/RECCAP2/IGBP
  - Comparison plots
  - Soil parameters JSON
  - Summary text file

Soil Parameters (REQUIRED):
  One of these options must be provided:
  --soil-param-file FILE       JSON/YAML parameter file
  --soil-log-file FILE         UM/Rose log with &LAND_CC block
  --soil-params KEY=VAL,...    Manual parameters
  --use-default-soil-params    Use default LAND_CC values
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

    # Soil parameter arguments (REQUIRED unless --use-default-soil-params)
    soil_group = parser.add_argument_group('soil parameters')
    soil_group.add_argument(
        '--soil-param-file',
        help='Path to soil parameter file (JSON or YAML)'
    )
    soil_group.add_argument(
        '--soil-log-file',
        help='Path to UM/Rose log file containing &LAND_CC block'
    )
    soil_group.add_argument(
        '--soil-params',
        help='Manual soil parameters as key=value pairs (e.g., "ALPHA=0.08,F0=0.875")'
    )
    soil_group.add_argument(
        '--use-default-soil-params',
        action='store_true',
        help='Use default LAND_CC soil parameters (opt-in required)'
    )

    args = parser.parse_args()

    # Check that at least one soil parameter source is provided
    soil_param_sources = [
        args.soil_param_file,
        args.soil_log_file,
        args.soil_params,
        args.use_default_soil_params
    ]

    if not any(soil_param_sources):
        parser.error(
            "Soil parameters required for validation tracking.\n"
            "Provide one of:\n"
            "  --soil-param-file FILE       (JSON/YAML parameter file)\n"
            "  --soil-log-file FILE         (UM/Rose log with &LAND_CC block)\n"
            "  --soil-params KEY=VAL,...    (Manual parameters)\n"
            "  --use-default-soil-params    (Use default LAND_CC values)"
        )

    # Load soil parameters
    soil_params = None

    if args.soil_param_file:
        soil_params = SoilParamSet.from_file(args.soil_param_file)
        print(f"✓ Loaded soil parameters from file: {args.soil_param_file}")
    elif args.soil_log_file:
        soil_params = SoilParamSet.from_log_file(args.soil_log_file)
        print(f"✓ Parsed soil parameters from log: {args.soil_log_file}")
    elif args.soil_params:
        # Parse manual key=value pairs
        manual_params = {}
        for pair in args.soil_params.split(','):
            key, value = pair.split('=')
            manual_params[key.strip()] = float(value.strip())
        soil_params = SoilParamSet.from_dict(manual_params, source='manual')
        print(f"✓ Loaded manual soil parameters ({len(manual_params)} values)")
    elif args.use_default_soil_params:
        soil_params = SoilParamSet.from_default()
        print(f"✓ Using default LAND_CC soil parameters")

    print("\n" + "="*80)
    print(f"VALIDATION WORKFLOW: {args.expt}")
    print("="*80)

    # Delegate to the script's main workflow
    # Import the actual validation workflow
    script_path = PathLib(__file__).parent.parent.parent / 'scripts' / 'validate_experiment.py'

    # For now, print instructions
    print(f"\nRunning validation for experiment: {args.expt}")
    print(f"Base directory: {args.base_dir}")
    print(f"Soil parameters: {soil_params.source}")
    print("\nValidation workflow is complex - delegating to scripts/validate_experiment.py")
    print(f"Please run: python scripts/validate_experiment.py {args.expt} " +
          f"--base-dir {args.base_dir} " +
          ("--use-default-soil-params" if args.use_default_soil_params else
           f"--soil-param-file {args.soil_param_file}" if args.soil_param_file else
           f"--soil-log-file {args.soil_log_file}" if args.soil_log_file else
           f"--soil-params {args.soil_params}"))

    sys.exit(0)


def validate_ppe_cli():
    """
    CLI entry point for validating PPE ensemble.

    Registered as: utils-cmip7-validate-ppe
    """
    from pathlib import Path as PathLib
    from .plotting import generate_ppe_validation_report, run_param_importance_suite

    parser = argparse.ArgumentParser(
        prog='utils-cmip7-validate-ppe',
        description='Generate PPE validation report with comprehensive visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  utils-cmip7-validate-ppe xqhuc
  utils-cmip7-validate-ppe xqhuc --top-n 20 --top-k 40
  utils-cmip7-validate-ppe xqhuc --highlight xqhua,xqhub
  utils-cmip7-validate-ppe xqhuc --param-viz --param-viz-vars GPP NPP CVeg

Output Structure:
  validation_outputs/ppe_{expt}/
    ├── ensemble_table.csv          # Input data copy
    ├── score_distribution.pdf      # Score histogram + ECDF
    ├── validation_heatmap.pdf      # Normalized metrics heatmap
    ├── parameter_shifts.pdf        # Parameter distribution shifts
    └── top_experiments.txt         # Text summary

Input CSV:
  Default: validation_outputs/random_sampling_combined_overview_table.csv
  Required column: overall_score
  Optional: ID, parameter columns, metric columns
        """
    )

    parser.add_argument(
        'expt',
        type=str,
        help='Experiment ID to validate (highlighted in all plots)'
    )

    parser.add_argument(
        '--csv',
        type=str,
        default='validation_outputs/random_sampling_combined_overview_table.csv',
        help='Path to ensemble results CSV (default: validation_outputs/random_sampling_combined_overview_table.csv)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation_outputs',
        help='Base output directory (default: validation_outputs)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=15,
        help='Number of top experiments to highlight in score plots (default: 15)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=30,
        help='Number of experiments to show in heatmap (default: 30)'
    )

    parser.add_argument(
        '--q',
        type=float,
        default=0.10,
        help='Quantile for parameter shift analysis (default: 0.10)'
    )

    parser.add_argument(
        '--score-col',
        type=str,
        default='overall_score',
        help='Column name for ranking score (default: overall_score)'
    )

    parser.add_argument(
        '--id-col',
        type=str,
        default='ID',
        help='Column name for experiment ID (default: ID)'
    )

    parser.add_argument(
        '--param-cols',
        type=str,
        default='ALPHA,G_AREA,LAI_MIN,NL0,R_GROW,TLOW,TUPP,V_CRIT',
        help='Comma-separated parameter column names (default: soil params)'
    )

    parser.add_argument(
        '--bins',
        type=int,
        default=40,
        help='Number of bins for histograms (default: 40)'
    )

    # Highlighting options
    highlight_group = parser.add_argument_group('experiment highlighting')
    highlight_group.add_argument(
        '--highlight',
        type=str,
        action='append',
        help='Additional experiments to highlight (can be repeated)'
    )
    highlight_group.add_argument(
        '--include-highlight',
        action='store_true',
        default=True,
        help='Force-include highlighted experiments (default: True)'
    )
    highlight_group.add_argument(
        '--no-include-highlight',
        dest='include_highlight',
        action='store_false',
        help='Do not force-include highlighted experiments'
    )
    highlight_group.add_argument(
        '--highlight-style',
        choices=['outline', 'marker', 'rowcol', 'both'],
        default='both',
        help='Highlight style for heatmaps (default: both)'
    )
    highlight_group.add_argument(
        '--highlight-label',
        action='store_true',
        default=True,
        help='Add labels to highlighted experiments (default: True)'
    )
    highlight_group.add_argument(
        '--no-highlight-label',
        dest='highlight_label',
        action='store_false',
        help='Disable labels for highlighted experiments'
    )

    # Parameter importance analysis
    param_group = parser.add_argument_group('parameter importance analysis')
    param_group.add_argument(
        '--param-viz',
        action='store_true',
        help='Run parameter importance analysis (Spearman + RandomForest)'
    )
    param_group.add_argument(
        '--param-viz-vars',
        type=str,
        nargs='+',
        help='Variables to analyze (e.g., GPP NPP CVeg). If not specified, analyzes all.'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"PPE VALIDATION REPORT: {args.expt}")
    print("="*80)
    print(f"Input CSV: {args.csv}")
    print(f"Output directory: {args.output_dir}/ppe_{args.expt}/")
    print("="*80 + "\n")

    # Parse parameter columns
    param_cols = [c.strip() for c in args.param_cols.split(',')]

    # Parse highlight experiments
    highlight_list = [args.expt]  # Always highlight the main experiment
    if args.highlight:
        for h in args.highlight:
            highlight_list.extend([x.strip() for x in h.split(',')])

    try:
        # Generate main validation report
        generate_ppe_validation_report(
            csv_path=args.csv,
            expt=args.expt,
            output_dir=args.output_dir,
            top_n=args.top_n,
            top_k=args.top_k,
            q=args.q,
            score_col=args.score_col,
            id_col=args.id_col,
            param_cols=param_cols,
            bins=args.bins,
            highlight=highlight_list,
            include_highlight=args.include_highlight,
            highlight_style=args.highlight_style,
            highlight_label=args.highlight_label,
        )

        # Run parameter importance analysis if requested
        if args.param_viz:
            print("\n" + "="*80)
            print("PARAMETER IMPORTANCE ANALYSIS")
            print("="*80 + "\n")

            run_param_importance_suite(
                csv_path=args.csv,
                expt=args.expt,
                output_dir=args.output_dir,
                param_cols=param_cols,
                skill_vars=args.param_viz_vars,
            )

        print("\n" + "="*80)
        print("PPE VALIDATION COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {args.output_dir}/ppe_{args.expt}/")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\nError during PPE validation: {e}", file=sys.stderr)
        sys.exit(1)


__all__ = [
    'extract_preprocessed_cli',
    'extract_raw_cli',
    'validate_experiment_cli',
    'validate_ppe_cli',
]

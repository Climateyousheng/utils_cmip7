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

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate extracted data against observations (global only)'
    )

    parser.add_argument(
        '--validation-outdir',
        type=str,
        default=None,
        help='Output directory for validation results (default: validation_outputs/single_val_{expt})'
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
            end_year=args.end_year,
            verbose=args.verbose
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

    # Validation workflow (if requested)
    if args.validate:
        print("=" * 80, file=sys.stderr)
        print("VALIDATING AGAINST OBSERVATIONS", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        try:
            from utils_cmip7.diagnostics import compute_metrics_from_raw
            from utils_cmip7.io import load_cmip6_metrics, load_reccap_metrics
            from utils_cmip7.validation import compare_metrics, summarize_comparison
            from utils_cmip7.validation import plot_three_way_comparison
            import pandas as pd
        except ImportError as e:
            print(f"Error importing validation modules: {e}", file=sys.stderr)
            sys.exit(1)

        # 1. Transform raw data to canonical schema
        print("→ Transforming data to canonical schema...", file=sys.stderr)
        metrics = ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']
        um_metrics = compute_metrics_from_raw(
            expt_name=args.expt,
            metrics=metrics,
            start_year=args.start_year,
            end_year=args.end_year,
            base_dir=args.base_dir,
            verbose=args.verbose
        )

        # Check available metrics
        available_metrics = [m for m in metrics if m in um_metrics and um_metrics[m]]
        if not available_metrics:
            print("Error: No metrics available for validation", file=sys.stderr)
            sys.exit(1)

        print(f"✓ Available metrics: {', '.join(available_metrics)}", file=sys.stderr)

        # 2. Load observational data (global only)
        regions = ['global']
        print("→ Loading observational data...", file=sys.stderr)

        cmip6_metrics = None
        reccap_metrics = None

        try:
            cmip6_metrics = load_cmip6_metrics(available_metrics, regions, include_errors=True)
            print("✓ Loaded CMIP6 metrics", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not load CMIP6 data: {e}", file=sys.stderr)

        try:
            reccap_metrics = load_reccap_metrics(available_metrics, regions, include_errors=True)
            print("✓ Loaded RECCAP2 metrics", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not load RECCAP2 data: {e}", file=sys.stderr)

        if not cmip6_metrics and not reccap_metrics:
            print("Error: No observational data loaded. Cannot validate.", file=sys.stderr)
            sys.exit(1)

        # 3. Compare metrics
        print("→ Comparing against observations...", file=sys.stderr)
        comparison_cmip6 = None
        comparison_reccap = None

        if cmip6_metrics:
            comparison_cmip6 = compare_metrics(um_metrics, cmip6_metrics, available_metrics, regions)
            print("✓ Compared against CMIP6", file=sys.stderr)

        if reccap_metrics:
            comparison_reccap = compare_metrics(um_metrics, reccap_metrics, available_metrics, regions)
            print("✓ Compared against RECCAP2", file=sys.stderr)

        # 4. Save results to CSV
        outdir = args.validation_outdir or f'validation_outputs/single_val_{args.expt}'
        os.makedirs(outdir, exist_ok=True)

        print(f"→ Saving validation results to: {outdir}", file=sys.stderr)

        # Helper function for CSV export
        def _export_comparison_csv(comparison, output_path):
            rows = []
            for metric, regions_dict in comparison.items():
                for region, comp_data in regions_dict.items():
                    row = {'metric': metric, 'region': region, **comp_data}
                    rows.append(row)
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False, float_format='%.5f')
            print(f"✓ Saved: {output_path}", file=sys.stderr)

        if comparison_cmip6:
            csv_path = os.path.join(outdir, f'{args.expt}_bias_vs_cmip6.csv')
            _export_comparison_csv(comparison_cmip6, csv_path)

        if comparison_reccap:
            csv_path = os.path.join(outdir, f'{args.expt}_bias_vs_reccap2.csv')
            _export_comparison_csv(comparison_reccap, csv_path)

        # 5. Create validation plots
        plot_dir = os.path.join(outdir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        print("→ Creating validation plots...", file=sys.stderr)
        for metric in available_metrics:
            if metric in um_metrics:
                try:
                    plot_three_way_comparison(
                        um_metrics, cmip6_metrics, reccap_metrics,
                        metric=metric,
                        outdir=plot_dir
                    )
                    print(f"✓ Created plot for {metric}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Could not create plot for {metric}: {e}", file=sys.stderr)

        # 6. Print summary
        print("=" * 80, file=sys.stderr)
        print("VALIDATION SUMMARY (GLOBAL)", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        if comparison_cmip6:
            summary_cmip6 = summarize_comparison(comparison_cmip6)
            print(f"CMIP6 Comparison:", file=sys.stderr)
            print(f"  - Comparisons: {summary_cmip6.get('n_comparisons', 0)}", file=sys.stderr)
            print(f"  - Within uncertainty: {summary_cmip6.get('fraction_within_uncertainty', 0)*100:.1f}%", file=sys.stderr)
            print(f"  - Mean bias: {summary_cmip6.get('mean_bias', 0):.2f}", file=sys.stderr)
            print(f"  - Mean RMSE: {summary_cmip6.get('mean_rmse', 0):.2f}", file=sys.stderr)

        if comparison_reccap:
            summary_reccap = summarize_comparison(comparison_reccap)
            print(f"RECCAP2 Comparison:", file=sys.stderr)
            print(f"  - Comparisons: {summary_reccap.get('n_comparisons', 0)}", file=sys.stderr)
            print(f"  - Within uncertainty: {summary_reccap.get('fraction_within_uncertainty', 0)*100:.1f}%", file=sys.stderr)
            print(f"  - Mean bias: {summary_reccap.get('mean_bias', 0):.2f}", file=sys.stderr)
            print(f"  - Mean RMSE: {summary_reccap.get('mean_rmse', 0):.2f}", file=sys.stderr)

        print(f"✓ Validation outputs saved to: {os.path.abspath(outdir)}/", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

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


def _extract_ensemble_prefix(expt_id: str) -> str:
    """
    Extract ensemble prefix from experiment ID.

    Convention: 5-character IDs have 4-character prefix (xqjca → xqjc)

    Parameters
    ----------
    expt_id : str
        Experiment identifier

    Returns
    -------
    str
        Ensemble prefix for log file matching

    Examples
    --------
    >>> _extract_ensemble_prefix('xqjca')
    'xqjc'
    >>> _extract_ensemble_prefix('xqhuc')
    'xqhuc'
    """
    return expt_id[:4] if len(expt_id) == 5 else expt_id


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
  # Auto-detect from ensemble logs (new!)
  utils-cmip7-validate-experiment xqjca

  # Custom log directory
  utils-cmip7-validate-experiment xqjca --log-dir /custom/path/logs

  # Explicit source (overrides auto-detection)
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

Soil Parameters:
  Auto-detected from ensemble logs when available, or provide one of:
  --soil-param-file FILE       JSON/YAML parameter file
  --soil-log-file FILE         UM/Rose log with &LAND_CC block
  --soil-params KEY=VAL,...    Manual parameters
  --use-default-soil-params    Use default LAND_CC values
  --log-dir DIR                Custom log directory (default: ~/scripts/hadcm3b-ensemble-generator/logs)
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
    soil_group.add_argument(
        '--log-dir',
        default='~/scripts/hadcm3b-ensemble-generator/logs',
        help='Directory containing ensemble-generator logs for auto-detection '
             '(default: ~/scripts/hadcm3b-ensemble-generator/logs)'
    )

    args = parser.parse_args()

    # Phase A: Check explicit sources
    explicit_sources = [
        args.soil_param_file,
        args.soil_log_file,
        args.soil_params,
        args.use_default_soil_params
    ]
    explicit_count = sum(1 for src in explicit_sources if src)

    if explicit_count > 1:
        parser.error("Only one soil parameter source can be specified")

    # Phase B: Load from explicit source if provided
    soil_params = None

    if explicit_count == 1:
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

    # Phase C: Try auto-detection from default log directory
    else:
        import sys
        from pathlib import Path as PathLib
        from .validation import load_ensemble_params_from_logs

        log_dir_path = PathLib(args.log_dir).expanduser()

        if log_dir_path.exists():
            ensemble_prefix = _extract_ensemble_prefix(args.expt)

            try:
                params_dict = load_ensemble_params_from_logs(
                    str(log_dir_path),
                    ensemble_prefix
                )

                if args.expt in params_dict:
                    soil_params = params_dict[args.expt]
                    log_file = soil_params.metadata.get('log_file', 'ensemble logs')
                    print(f"✓ Auto-detected soil parameters from: {log_file}")
                    print(f"  Ensemble: {ensemble_prefix}, Member: {args.expt}")
                else:
                    available = ', '.join(sorted(params_dict.keys())[:5])
                    more_text = '...' if len(params_dict) > 5 else ''
                    parser.error(
                        f"Experiment '{args.expt}' not found in ensemble logs.\n"
                        f"Available: {available}{more_text}\n"
                        f"Provide explicit soil parameter source."
                    )

            except FileNotFoundError:
                # Log files don't exist - fall through to error
                parser.error(
                    f"No ensemble logs found for '{ensemble_prefix}' in {log_dir_path}\n"
                    f"Provide explicit soil parameter source:\n"
                    f"  --soil-param-file FILE\n"
                    f"  --soil-log-file FILE\n"
                    f"  --soil-params KEY=VAL,...\n"
                    f"  --use-default-soil-params"
                )

        else:
            # Default directory doesn't exist - warn and require explicit source
            print(f"⚠ Warning: Default log directory not found: {log_dir_path}",
                  file=sys.stderr)
            parser.error(
                "Soil parameters required.\n"
                "Provide one of:\n"
                "  --soil-param-file FILE\n"
                "  --soil-log-file FILE\n"
                "  --soil-params KEY=VAL,...\n"
                "  --use-default-soil-params\n"
                "  --log-dir DIR (custom log directory)"
            )

    # Import config for regions
    from .config import RECCAP_REGIONS

    def get_all_regions():
        """Get all RECCAP2 regions plus global."""
        regions = ['global'] + list(RECCAP_REGIONS.values())
        if 'Africa' not in regions:
            regions.append('Africa')
        return regions

    # Import veg metrics constants
    VEG_METRICS = ['BL', 'NL', 'C3', 'C4', 'shrub', 'bare_soil']

    # Helper functions for saving results
    def save_um_metrics_to_csv(um_metrics, expt, outdir):
        """Save UM metrics to CSV in observational data format."""
        regions = get_all_regions()
        # Include all metrics that have canonical structure (dict with 'data' key)
        metrics = []
        for m in sorted(um_metrics.keys()):
            # Check this is canonical structure (not scalar veg metrics)
            sample_region = next(iter(um_metrics[m].values()), None)
            if isinstance(sample_region, dict) and 'data' in sample_region:
                metrics.append(m)

        # Build dataframe
        data = {}
        for region in regions:
            regional_data = []
            for metric in metrics:
                if metric in um_metrics and region in um_metrics[metric]:
                    mean_val = np.mean(um_metrics[metric][region]['data'])
                    regional_data.append(mean_val)
                else:
                    regional_data.append(np.nan)
            data[region] = regional_data

        df = pd.DataFrame(data, index=metrics)

        # Save with 5 decimal precision
        csv_path = outdir / f'{expt}_metrics.csv'
        df.to_csv(csv_path, float_format='%.5f')
        print(f"  ✓ Saved UM metrics ({len(metrics)} variables): {csv_path}")

        return df

    def save_bias_statistics(comparison, obs_name, expt, outdir):
        """Save bias statistics to CSV."""
        rows = []
        for metric in sorted(comparison.keys()):
            for region in sorted(comparison[metric].keys()):
                comp = comparison[metric][region]
                rows.append({
                    'metric': metric,
                    'region': region,
                    'um_mean': comp['um_mean'],
                    'obs_mean': comp['obs_mean'],
                    'bias': comp['bias'],
                    'bias_percent': comp['bias_percent'],
                    'rmse': comp['rmse'],
                    'within_uncertainty': comp['within_uncertainty'],
                })

        df = pd.DataFrame(rows)
        csv_path = outdir / f'{expt}_bias_vs_{obs_name.lower()}.csv'
        df.to_csv(csv_path, index=False, float_format='%.5f')
        print(f"  ✓ Saved bias statistics vs {obs_name}: {csv_path}")

        return df

    def save_comparison_summary(
        um_metrics,
        cmip6_metrics,
        reccap_metrics,
        comparison_cmip6,
        comparison_reccap,
        comparison_igbp,
        expt,
        outdir
    ):
        """Save text summary of validation results."""
        summary_path = outdir / 'comparison_summary.txt'

        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"VALIDATION SUMMARY: {expt}\n")
            f.write("="*80 + "\n\n")

            # UM vs CMIP6
            f.write("UM vs CMIP6 ENSEMBLE\n")
            f.write("-"*80 + "\n")
            for metric in ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']:
                if metric in comparison_cmip6 and metric in um_metrics:
                    # Get any available region to determine units
                    available_regions = list(um_metrics[metric].keys())
                    if available_regions:
                        summary = summarize_comparison(comparison_cmip6, metric=metric)
                        units = um_metrics[metric][available_regions[0]]['units']
                        f.write(f"\n{metric}:\n")
                        f.write(f"  Mean bias: {summary['mean_bias']:.2f} {units}\n")
                        f.write(f"  Mean bias %: {summary['mean_bias_percent']:.1f}%\n")
                        f.write(f"  Fraction within uncertainty: {summary['fraction_within_uncertainty']:.1%}\n")

            # UM vs RECCAP2
            f.write("\n" + "="*80 + "\n")
            f.write("UM vs RECCAP2 OBSERVATIONS\n")
            f.write("-"*80 + "\n")
            for metric in ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']:
                if metric in comparison_reccap and metric in um_metrics:
                    # Get any available region to determine units
                    available_regions = list(um_metrics[metric].keys())
                    if available_regions:
                        summary = summarize_comparison(comparison_reccap, metric=metric)
                        units = um_metrics[metric][available_regions[0]]['units']
                        f.write(f"\n{metric}:\n")
                        f.write(f"  Mean bias: {summary['mean_bias']:.2f} {units}\n")
                        f.write(f"  Mean bias %: {summary['mean_bias_percent']:.1f}%\n")
                        f.write(f"  Fraction within uncertainty: {summary['fraction_within_uncertainty']:.1%}\n")

            # UM vs CMIP6 performance comparison
            f.write("\n" + "="*80 + "\n")
            f.write("UM vs CMIP6 PERFORMANCE (against RECCAP2)\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Metric':<10} {'Region':<20} {'UM Bias %':<12} {'CMIP6 Bias %':<15} {'Winner':<10}\n")
            f.write("-"*80 + "\n")

            regions = get_all_regions()
            for metric in ['GPP', 'NPP', 'CVeg', 'CSoil']:
                for region in regions:
                    if (metric in um_metrics and region in um_metrics[metric] and
                        metric in cmip6_metrics and region in cmip6_metrics[metric] and
                        metric in reccap_metrics and region in reccap_metrics[metric]):

                        um_val = np.mean(um_metrics[metric][region]['data'])
                        cmip6_val = cmip6_metrics[metric][region]['data'][0]
                        reccap_val = reccap_metrics[metric][region]['data'][0]

                        um_bias_pct = 100 * (um_val - reccap_val) / reccap_val
                        cmip6_bias_pct = 100 * (cmip6_val - reccap_val) / reccap_val

                        if abs(um_bias_pct) < abs(cmip6_bias_pct):
                            winner = "UM"
                        elif abs(um_bias_pct) > abs(cmip6_bias_pct):
                            winner = "CMIP6"
                        else:
                            winner = "Tie"

                        f.write(f"{metric:<10} {region:<20} {um_bias_pct:>10.1f}%  {cmip6_bias_pct:>12.1f}%  {winner:<10}\n")

            # UM vs IGBP (vegetation fractions)
            if comparison_igbp:
                f.write("\n" + "="*80 + "\n")
                f.write("UM vs IGBP VEGETATION FRACTIONS\n")
                f.write("-"*80 + "\n")
                for metric in sorted(comparison_igbp.keys()):
                    if 'global' in comparison_igbp[metric]:
                        comp = comparison_igbp[metric]['global']
                        f.write(f"\n{metric}:\n")
                        f.write(f"  UM mean: {comp['um_mean']:.4f}\n")
                        f.write(f"  IGBP obs: {comp['obs_mean']:.4f}\n")
                        f.write(f"  Bias: {comp['bias']:+.4f} ({comp['bias_percent']:+.1f}%)\n")

            f.write("\n" + "="*80 + "\n")

        print(f"  ✓ Saved comparison summary: {summary_path}")

    def create_all_plots(
        um_metrics,
        cmip6_metrics,
        reccap_metrics,
        comparison_cmip6,
        comparison_reccap,
        comparison_igbp,
        igbp_metrics,
        outdir
    ):
        """Create all validation plots."""
        plots_dir = outdir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        carbon_metrics = ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']
        all_metrics = carbon_metrics + VEG_METRICS

        print("\n  Creating plots...")

        # 1. Three-way comparison plots (carbon metrics: UM vs CMIP6 vs RECCAP2)
        print("    - Three-way comparisons (carbon)")
        for metric in carbon_metrics:
            if (metric in um_metrics and metric in cmip6_metrics and metric in reccap_metrics and
                um_metrics[metric] and cmip6_metrics[metric] and reccap_metrics[metric]):
                plot_three_way_comparison(
                    um_metrics,
                    cmip6_metrics,
                    reccap_metrics,
                    metric=metric,
                    outdir=plots_dir,
                    filename=f'{metric}_three_way.png'
                )
            else:
                print(f"      ⚠ Skipping {metric} (not available in all datasets)")

        # 1b. Two-way comparison plots (veg metrics: UM vs IGBP)
        print("    - Veg fraction comparisons (UM vs IGBP)")
        for metric in VEG_METRICS:
            if (metric in um_metrics and um_metrics[metric] and
                metric in igbp_metrics and igbp_metrics[metric]):
                plot_two_way_comparison(
                    um_metrics,
                    igbp_metrics,
                    metric=metric,
                    outdir=plots_dir,
                    filename=f'{metric}_vs_igbp.png'
                )
            else:
                print(f"      ⚠ Skipping {metric} (not available)")

        # 2. Bias heatmaps
        print("    - Bias heatmaps")
        # Separate heatmap for CMIP6 (carbon metrics only)
        available_metrics_cmip6 = [m for m in carbon_metrics if m in comparison_cmip6 and comparison_cmip6[m]]
        if available_metrics_cmip6:
            plot_regional_bias_heatmap(
                comparison_cmip6,
                metrics=available_metrics_cmip6,
                value_type='bias_percent',
                outdir=plots_dir,
                filename='bias_heatmap_vs_cmip6.png'
            )

        # UNIFIED heatmap: Carbon metrics (vs RECCAP2) + Veg metrics (vs IGBP)
        combined_comparison = {}
        available_carbon = [m for m in carbon_metrics if m in comparison_reccap and comparison_reccap[m]]
        available_veg = [m for m in VEG_METRICS if m in comparison_igbp and comparison_igbp[m]]

        # Merge comparisons into single dict
        for m in available_carbon:
            combined_comparison[m] = comparison_reccap[m]
        for m in available_veg:
            combined_comparison[m] = comparison_igbp[m]

        if combined_comparison:
            all_combined_metrics = available_carbon + available_veg
            plot_regional_bias_heatmap(
                combined_comparison,
                metrics=all_combined_metrics,
                value_type='bias_percent',
                outdir=plots_dir,
                filename='bias_heatmap_unified.png'
            )
            print(f"      ✓ Created unified heatmap ({len(all_combined_metrics)} metrics)")

        # 3. Time series plots
        print("    - Time series (carbon)")
        for metric in carbon_metrics:
            if (metric in um_metrics and 'global' in um_metrics[metric] and
                metric in reccap_metrics and 'global' in reccap_metrics[metric]):
                plot_timeseries_with_obs(
                    um_metrics,
                    reccap_metrics,
                    metric=metric,
                    region='global',
                    outdir=plots_dir,
                    filename=f'{metric}_timeseries_global.png'
                )
            elif metric in um_metrics and um_metrics[metric]:
                available_regions = list(um_metrics[metric].keys())
                if (available_regions and metric in reccap_metrics and
                    available_regions[0] in reccap_metrics[metric]):
                    plot_timeseries_with_obs(
                        um_metrics,
                        reccap_metrics,
                        metric=metric,
                        region=available_regions[0],
                        outdir=plots_dir,
                        filename=f'{metric}_timeseries_{available_regions[0]}.png'
                    )

        # 3b. Time series for veg metrics (UM vs IGBP)
        print("    - Time series (veg fractions)")
        for metric in VEG_METRICS:
            if (metric in um_metrics and 'global' in um_metrics[metric] and
                metric in igbp_metrics and 'global' in igbp_metrics[metric]):
                plot_timeseries_with_obs(
                    um_metrics,
                    igbp_metrics,
                    metric=metric,
                    region='global',
                    outdir=plots_dir,
                    filename=f'{metric}_timeseries_global.png'
                )

        print(f"  ✓ Created all plots in {plots_dir}/")

    print("\n" + "="*80)
    print(f"VALIDATION WORKFLOW: {args.expt}")
    print("="*80)

    # Create output directory
    outdir = PathLib('validation_outputs') / f'single_val_{args.expt}'
    outdir.mkdir(parents=True, exist_ok=True)

    # Get all regions and metrics
    regions = get_all_regions()
    metrics = ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']

    # Step 1: Compute UM metrics
    print(f"\n[1/7] Computing UM metrics from {args.base_dir}/{args.expt}/...")
    print("-"*80)

    um_metrics = compute_metrics_from_annual_means(
        expt_name=args.expt,
        metrics=metrics,
        regions=regions,
        base_dir=args.base_dir
    )
    print(f"✓ Computed {len(um_metrics)} standard metrics for {len(regions)} regions")

    # Extract vegetation fractions
    print(f"  Extracting vegetation fraction data...")
    raw_data = extract_annual_means(
        expts_list=[args.expt],
        var_list=['frac'],
        regions=regions,
        base_dir=args.base_dir
    )

    # Promote PFT time series to um_metrics
    veg_count = 0
    if args.expt in raw_data:
        for pft_id, pft_name in sorted(PFT_MAPPING.items()):
            pft_key = f'PFT {pft_id}'
            um_metrics[pft_name] = {}
            for region in regions:
                if (region in raw_data[args.expt] and
                    'frac' in raw_data[args.expt][region] and
                    pft_key in raw_data[args.expt][region]['frac']):
                    pft_data = raw_data[args.expt][region]['frac'][pft_key]
                    um_metrics[pft_name][region] = {
                        'years': pft_data['years'],
                        'data': pft_data['data'],
                        'units': 'fraction',
                        'source': 'UM',
                        'dataset': args.expt
                    }
            if um_metrics[pft_name]:
                veg_count += 1
            else:
                del um_metrics[pft_name]

    # Compute scalar veg metrics
    veg_metrics = calculate_veg_metrics(raw_data, args.expt, regions=regions)

    if veg_count > 0:
        print(f"✓ Promoted {veg_count} PFT time series to metrics")
        print(f"✓ Total metrics: {len(um_metrics)}")
    else:
        print(f"⚠ No vegetation fraction data available")

    # Step 2: Load observational data
    print("\n[2/7] Loading observational data...")
    print("-"*80)

    cmip6_metrics = load_cmip6_metrics(metrics=metrics, regions=regions, include_errors=True)
    reccap_metrics = load_reccap_metrics(metrics=metrics, regions=regions, include_errors=True)
    print(f"✓ Loaded CMIP6 ensemble data")
    print(f"✓ Loaded RECCAP2 observational data")

    # Load IGBP vegetation observations
    obs_veg_metrics = load_obs_veg_metrics(regions=regions)
    igbp_metrics = {}
    if obs_veg_metrics:
        for pft_name in VEG_METRICS:
            if pft_name in obs_veg_metrics:
                igbp_metrics[pft_name] = {}
                for region in obs_veg_metrics[pft_name].keys():
                    igbp_metrics[pft_name][region] = {
                        'data': np.array([obs_veg_metrics[pft_name][region]]),
                        'units': 'fraction',
                        'source': 'IGBP',
                    }
        print(f"✓ Loaded IGBP vegetation fraction observations ({len(igbp_metrics)} PFTs)")

    # Step 3: Compare UM vs observations
    print("\n[3/7] Computing comparison statistics...")
    print("-"*80)

    comparison_cmip6 = compare_metrics(um_metrics, cmip6_metrics, metrics=metrics, regions=regions)
    comparison_reccap = compare_metrics(um_metrics, reccap_metrics, metrics=metrics, regions=regions)
    print(f"✓ Computed bias statistics vs CMIP6")
    print(f"✓ Computed bias statistics vs RECCAP2")

    # Compare vegetation fractions
    comparison_igbp = {}
    if igbp_metrics:
        comparison_igbp = compare_metrics(um_metrics, igbp_metrics, metrics=VEG_METRICS, regions=regions)
        print(f"✓ Computed vegetation fraction bias vs IGBP")

    # Step 4: Export to CSV
    print("\n[4/7] Exporting results to CSV...")
    print("-"*80)

    # Save UM metrics
    save_um_metrics_to_csv(um_metrics, args.expt, outdir)
    save_bias_statistics(comparison_cmip6, 'CMIP6', args.expt, outdir)
    save_bias_statistics(comparison_reccap, 'RECCAP2', args.expt, outdir)
    if comparison_igbp:
        save_bias_statistics(comparison_igbp, 'IGBP', args.expt, outdir)
    save_comparison_summary(
        um_metrics, cmip6_metrics, reccap_metrics,
        comparison_cmip6, comparison_reccap, comparison_igbp,
        args.expt, outdir
    )

    # Step 5: Create plots
    print("\n[5/7] Creating validation plots...")
    print("-"*80)

    create_all_plots(
        um_metrics, cmip6_metrics, reccap_metrics,
        comparison_cmip6, comparison_reccap, comparison_igbp,
        igbp_metrics, outdir
    )

    # Step 6: Update overview table
    print("\n[6/7] Updating overview table and writing validation bundle...")
    print("-"*80)

    bl_params = soil_params.to_overview_table_format()

    # Prepare validation scores
    scores = {}
    for metric in ['GPP', 'NPP', 'CVeg', 'CSoil']:
        if metric in um_metrics and 'global' in um_metrics[metric]:
            scores[metric] = np.mean(um_metrics[metric]['global']['data'])

    # Regional tree metrics
    if args.expt in raw_data and 'global' in raw_data[args.expt] and 'frac' in raw_data[args.expt]['global']:
        frac_data = raw_data[args.expt]['global']['frac']
        for metric_name in ['Tr30SN', 'Tr30-90N', 'AMZTrees']:
            if metric_name in frac_data and 'data' in frac_data[metric_name]:
                scores[metric_name] = np.mean(frac_data[metric_name]['data'])
            else:
                scores[metric_name] = np.nan
    else:
        for metric_name in ['Tr30SN', 'Tr30-90N', 'AMZTrees']:
            scores[metric_name] = np.nan

    # Global mean vegetation fractions
    for pft_name in ['BL', 'NL', 'C3', 'C4', 'bare_soil']:
        gm_col = f'GM_{pft_name.replace("bare_soil", "BS")}'
        if pft_name in um_metrics and 'global' in um_metrics[pft_name]:
            scores[gm_col] = np.mean(um_metrics[pft_name]['global']['data'])
        else:
            scores[gm_col] = np.nan

    # RMSE values
    if veg_metrics:
        for pft_name in ['BL', 'NL', 'C3', 'C4', 'bare_soil']:
            rmse_key = f'rmse_{pft_name}'
            rmse_col = rmse_key.replace('bare_soil', 'BS')
            if rmse_key in veg_metrics and 'global' in veg_metrics[rmse_key]:
                scores[rmse_col] = veg_metrics[rmse_key]['global']
            else:
                scores[rmse_col] = np.nan

    # Overall score
    bias_pcts = []
    for metric in ['GPP', 'NPP', 'CVeg', 'CSoil']:
        if metric in comparison_reccap and 'global' in comparison_reccap[metric]:
            bias_pcts.append(abs(comparison_reccap[metric]['global']['bias_percent']))
    if comparison_igbp:
        for pft in ['BL', 'NL', 'C3', 'C4']:
            if pft in comparison_igbp and 'global' in comparison_igbp[pft]:
                bias_pcts.append(abs(comparison_igbp[pft]['global']['bias_percent']))
    if bias_pcts:
        mean_abs_bias = np.mean(bias_pcts)
        scores['overall_score'] = 1.0 - (mean_abs_bias / 100.0)
    else:
        scores['overall_score'] = np.nan

    # Update overview table
    overview_path = PathLib('validation_outputs') / 'random_sampling_combined_overview_table.csv'
    overview_df = load_overview_table(str(overview_path))
    overview_df = upsert_overview_row(overview_df, args.expt, bl_params, scores)
    write_atomic_csv(overview_df, str(overview_path))
    print(f"  ✓ Updated overview table: {overview_path}")

    # Write validation bundle
    write_single_validation_bundle(
        outdir=PathLib('validation_outputs'),
        expt_id=args.expt,
        soil_params=soil_params,
        metrics=um_metrics,
        scores=scores
    )
    print(f"  ✓ Wrote validation bundle: {outdir}/")

    # Step 7: Summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {outdir}/")
    print(f"  - {args.expt}_metrics.csv")
    print(f"  - {args.expt}_bias_vs_cmip6.csv")
    print(f"  - {args.expt}_bias_vs_reccap2.csv")
    if comparison_igbp:
        print(f"  - {args.expt}_bias_vs_IGBP.csv")
    print(f"  - comparison_summary.txt")
    print(f"  - plots/")

    # Quick summary
    print(f"\nKey findings vs RECCAP2:")
    if 'GPP' in comparison_reccap and 'global' in comparison_reccap['GPP']:
        gpp_summary = summarize_comparison(comparison_reccap, 'GPP')
        print(f"  GPP: {gpp_summary['mean_bias']:+.1f} PgC/yr ({gpp_summary['mean_bias_percent']:+.1f}%)")
    if 'NPP' in comparison_reccap and 'global' in comparison_reccap['NPP']:
        npp_summary = summarize_comparison(comparison_reccap, 'NPP')
        print(f"  NPP: {npp_summary['mean_bias']:+.1f} PgC/yr ({npp_summary['mean_bias_percent']:+.1f}%)")
    print("="*80 + "\n")


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
            ensemble_name=args.expt,
            output_dir=args.output_dir,
            top_n=args.top_n,
            top_k=args.top_k,
            q=args.q,
            score_col=args.score_col,
            id_col=args.id_col,
            param_cols=param_cols,
            bins=args.bins,
            highlight_expts=highlight_list,
            include_highlight=args.include_highlight,
            highlight_style=args.highlight_style,
            highlight_label=args.highlight_label,
        )

        # Run parameter importance analysis if requested
        if args.param_viz:
            print("\n" + "="*80)
            print("PARAMETER IMPORTANCE ANALYSIS")
            print("="*80 + "\n")

            param_viz_outdir = f"{args.output_dir}/param_viz_{args.expt}"
            run_param_importance_suite(
                overview_csv=args.csv,
                outdir=param_viz_outdir,
                variables=args.param_viz_vars,
                id_col=args.id_col,
                param_cols=param_cols,
            )

        print("\n" + "="*80)
        print("PPE VALIDATION COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {args.output_dir}/ppe_{args.expt}/")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\nError during PPE validation: {e}", file=sys.stderr)
        sys.exit(1)


def populate_overview_cli():
    """
    CLI entry point for populating overview table from ensemble generator logs.

    Registered as: utils-cmip7-populate-overview
    """
    parser = argparse.ArgumentParser(
        prog='utils-cmip7-populate-overview',
        description='Populate overview table with parameters from ensemble generator logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Populate all xqjc experiments
  utils-cmip7-populate-overview xqjc \\
    --log-dir /user/home/nd20983/scripts/hadcm3b-ensemble-generator/logs

  # Populate specific experiments only
  utils-cmip7-populate-overview xqjc \\
    --log-dir ~/scripts/hadcm3b-ensemble-generator/logs \\
    --experiments xqjca xqjcb xqjcc

  # Custom overview table location
  utils-cmip7-populate-overview xqjc \\
    --log-dir ~/scripts/hadcm3b-ensemble-generator/logs \\
    --overview-csv my_ensemble_table.csv

Output:
  Updates validation_outputs/random_sampling_combined_overview_table.csv
  with soil parameters from ensemble generator logs. Only fills parameter
  columns (ALPHA, G_AREA, LAI_MIN, etc.). Validation metrics remain empty
  until validate-experiment is run.

Notes:
  - Creates overview CSV if it doesn't exist
  - Preserves existing validation metrics
  - Uses atomic write to prevent data loss
  - Parameter names are automatically mapped to overview table format
        """
    )

    parser.add_argument(
        'ensemble_prefix',
        type=str,
        help='Ensemble name prefix (e.g., xqjc, xqhl, xqar)'
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        required=True,
        help='Path to ensemble-generator logs directory'
    )

    parser.add_argument(
        '--overview-csv',
        type=str,
        default='validation_outputs/random_sampling_combined_overview_table.csv',
        help='Path to overview table CSV (default: validation_outputs/random_sampling_combined_overview_table.csv)'
    )

    parser.add_argument(
        '--experiments',
        nargs='+',
        default=None,
        help='Specific experiment IDs to update (default: all found in logs)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print(f"POPULATE OVERVIEW TABLE: {args.ensemble_prefix}")
    print("=" * 80)
    print(f"Log directory: {args.log_dir}")
    print(f"Overview table: {args.overview_csv}")
    if args.experiments:
        print(f"Experiments: {', '.join(args.experiments)}")
    else:
        print("Experiments: All found in logs")
    print("=" * 80)

    try:
        from .validation import populate_overview_table_from_logs

        populate_overview_table_from_logs(
            log_dir=args.log_dir,
            ensemble_prefix=args.ensemble_prefix,
            overview_csv=args.overview_csv,
            experiment_ids=args.experiments
        )

        print("=" * 80)
        print("OVERVIEW TABLE UPDATED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Run validate-experiment for each experiment to compute metrics")
        print("  2. Run validate-ppe to visualize ensemble results")
        print("=" * 80)

    except Exception as e:
        print(f"\nError populating overview table: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


__all__ = [
    'extract_preprocessed_cli',
    'extract_raw_cli',
    'validate_experiment_cli',
    'validate_ppe_cli',
    'populate_overview_cli',
]

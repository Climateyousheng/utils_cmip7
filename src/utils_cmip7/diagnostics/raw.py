"""
High-level extraction from raw monthly UM output files.

Processes raw monthly UM output files directly (not pre-processed annual means).
"""

import warnings
import iris

from ..io import stash, try_extract, find_matching_files
from ..processing.temporal import compute_monthly_mean, merge_monthly_results


def extract_annual_mean_raw(expt, base_dir='~/dump2hold', start_year=None, end_year=None, verbose=False):
    """
    Extract annual means from raw monthly UM output files.

    Processes raw monthly files in ~/dump2hold/expt/datam/ (not pre-processed
    annual mean NetCDF files). Useful when annual means haven't been generated yet.

    Parameters
    ----------
    expt : str
        Experiment name (e.g., 'xqhuj', 'xqhuk')
    base_dir : str, default '~/dump2hold'
        Base directory containing experiment subdirectories
    start_year : int, optional
        First year to process. If None, processes all available years.
    end_year : int, optional
        Last year to process. If None, processes all available years.
    verbose : bool, default False
        If True, print detailed error messages for failed files.
        If False, suppress warnings and silently skip failed files.

    Returns
    -------
    dict
        Dictionary with structure:
        {
            'GPP': {'years': array, 'data': array, 'units': str, 'name': str},
            'NPP': {'years': array, 'data': array, 'units': str, 'name': str},
            'Rh': {'years': array, 'data': array, 'units': str, 'name': str},
            'CVeg': {'years': array, 'data': array, 'units': str, 'name': str},
            'CSoil': {'years': array, 'data': array, 'units': str, 'name': str},
            'NEP': {'years': array, 'data': array, 'units': str, 'name': str},
        }

    Examples
    --------
    >>> # Extract all years
    >>> data = extract_annual_mean_raw('xqhuj')
    >>> data['GPP']['data']  # GPP time series in PgC/year

    >>> # Extract specific year range
    >>> data = extract_annual_mean_raw('xqhuj', start_year=1850, end_year=1900)
    >>> data['NPP']['years']  # Years 1850-1900

    >>> # Plot GPP
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(data['GPP']['years'], data['GPP']['data'])
    >>> plt.xlabel('Year')
    >>> plt.ylabel('GPP (PgC/year)')
    >>> plt.show()

    Notes
    -----
    Input files expected in: ~/dump2hold/{expt}/datam/
    Filename pattern: {expt}a#pi00000{YYYY}{MM}+
    where {MM} is two-letter month code (ja, fb, mr, etc.)

    Variables extracted:
    - GPP: Gross Primary Production
    - NPP: Net Primary Production
    - Rh: Heterotrophic respiration
    - CVeg: Vegetation carbon
    - CSoil: Soil carbon

    Derived variables computed:
    - NEP = NPP - Rh (Net Ecosystem Production)

    Workflow:
    1. Find raw monthly files using find_matching_files()
    2. Load each file and extract variable using try_extract()
    3. Compute monthly mean using compute_monthly_mean()
    4. Merge monthly results into annual means using merge_monthly_results()

    Detailed diagnostic output printed to stdout showing:
    - Number of files found
    - Year range available
    - Processing status for each variable
    - Number of files processed/failed
    - Final extraction summary
    """

    # Variables to extract: (code, display_name, mapping_name)
    variables = [
        ('gpp', 'GPP', 'GPP'),
        ('npp', 'NPP', 'NPP'),
        ('rh', 'Rh', 'Rh'),
        ('cv', 'CVeg', 'CVeg'),
        ('cs', 'CSoil', 'CSoil'),
    ]

    # Suppress warnings unless verbose mode
    if not verbose:
        warnings.filterwarnings('ignore')

    print(f"\n{'='*60}")
    print(f"Extracting annual means from raw monthly files: {expt}")
    print(f"{'='*60}")

    # Find raw monthly output files
    print(f"\nSearching for files in {base_dir}/{expt}/datam/...")
    files = find_matching_files(
        expt_name=expt,
        model='a',
        up='pi',
        start_year=start_year,
        end_year=end_year,
        base_dir=base_dir,
    )
    print(f"Found {len(files)} monthly files")

    if not files:
        print(f"  ⚠ No files found!")
        return {}

    print(f"  Year range: {files[0][0]} - {files[-1][0]}")

    # Dictionary to store results
    annual_means = {}

    # Process each variable
    for var_code, var_key, var_name in variables:
        print(f"\n{'='*60}")
        print(f"Processing {var_key} ({var_code})")
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
                files_failed += 1
                if verbose:
                    print(f"  ⚠ Failed to process {f}: {type(e).__name__}: {e}")
                continue

        if monthly_results:
            # Merge monthly results into annual means
            annual_data = merge_monthly_results(monthly_results)
            annual_means[var_key] = {
                'years': annual_data['years'],
                'data': annual_data['data'],
                'units': 'PgC/year' if var_key in ('GPP', 'NPP', 'Rh') else 'PgC',
                'name': var_key,
            }

            print(f"  ✓ Successfully processed {files_processed}/{len(files)} files")
            print(f"  ✓ Got {len(annual_data['years'])} years of data")
            print(f"  Years: {annual_data['years'][0]} - {annual_data['years'][-1]}")
            if files_failed > 0:
                print(f"  ⚠ Failed: {files_failed} files")
        else:
            print(f"  ❌ No data extracted for {var_key}")

    # Compute derived variables
    if 'NPP' in annual_means and 'Rh' in annual_means:
        print(f"\n{'='*60}")
        print("Computing derived variable: NEP")
        print(f"{'='*60}")
        nep_years = annual_means['NPP']['years'].copy()
        nep_data = annual_means['NPP']['data'] - annual_means['Rh']['data']
        annual_means['NEP'] = {
            'years': nep_years,
            'data': nep_data,
            'units': 'PgC/year',
            'name': 'Net Ecosystem Production',
        }
        print(f"  ✓ NEP computed: {len(nep_years)} years")

    # Summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Variables successfully extracted: {len(annual_means)}")
    for var_key in annual_means.keys():
        n_years = len(annual_means[var_key]['data'])
        print(f"  ✓ {var_key}: {n_years} years")
    print(f"{'='*60}\n")

    return annual_means

#!/usr/bin/env python3
"""
Performance benchmark and correctness validation for extraction optimizations.

Tests both raw and preprocessed pathways, comparing results and measuring speedup.

Usage:
    python validation/benchmark_extraction.py
"""

import time
import numpy as np
from pathlib import Path
import sys

# Add src to path for development testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils_cmip7.diagnostics.raw import extract_annual_mean_raw
from utils_cmip7.diagnostics.extraction import extract_annual_means


def benchmark_raw(expt, start_year, end_year, n_runs=3):
    """
    Benchmark raw pathway extraction.

    Parameters
    ----------
    expt : str
        Experiment name (e.g., 'xqhuj')
    start_year : int
        First year to process
    end_year : int
        Last year to process
    n_runs : int
        Number of runs for averaging

    Returns
    -------
    dict
        {
            'elapsed': list of elapsed times,
            'mean_time': average time,
            'std_time': standard deviation,
            'data': last extraction result
        }
    """
    times = []
    result = None

    print(f"\n{'='*60}")
    print(f"Benchmarking RAW pathway: {expt} ({start_year}-{end_year})")
    print(f"{'='*60}")

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}...")
        start = time.time()
        result = extract_annual_mean_raw(
            expt,
            start_year=start_year,
            end_year=end_year,
            verbose=False
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Completed in {elapsed:.2f}s")

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n{'='*60}")
    print(f"RAW PATHWAY RESULTS")
    print(f"{'='*60}")
    print(f"  Mean time: {mean_time:.2f}s ± {std_time:.2f}s")
    print(f"  Min time: {min(times):.2f}s")
    print(f"  Max time: {max(times):.2f}s")

    return {
        'elapsed': times,
        'mean_time': mean_time,
        'std_time': std_time,
        'data': result
    }


def benchmark_preprocessed(expts, n_runs=3):
    """
    Benchmark preprocessed pathway extraction.

    Parameters
    ----------
    expts : list of str
        Experiment names (e.g., ['xqhuj'])
    n_runs : int
        Number of runs for averaging

    Returns
    -------
    dict
        {
            'elapsed': list of elapsed times,
            'mean_time': average time,
            'std_time': standard deviation,
            'data': last extraction result
        }
    """
    times = []
    result = None

    print(f"\n{'='*60}")
    print(f"Benchmarking PREPROCESSED pathway: {expts}")
    print(f"{'='*60}")

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}...")
        start = time.time()
        result = extract_annual_means(expts)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Completed in {elapsed:.2f}s")

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n{'='*60}")
    print(f"PREPROCESSED PATHWAY RESULTS")
    print(f"{'='*60}")
    print(f"  Mean time: {mean_time:.2f}s ± {std_time:.2f}s")
    print(f"  Min time: {min(times):.2f}s")
    print(f"  Max time: {max(times):.2f}s")

    return {
        'elapsed': times,
        'mean_time': mean_time,
        'std_time': std_time,
        'data': result
    }


def compare_results(ref, opt, tolerance=1e-10):
    """
    Compare reference and optimized extraction results.

    Parameters
    ----------
    ref : dict
        Reference extraction result
    opt : dict
        Optimized extraction result
    tolerance : float
        Absolute tolerance for np.allclose

    Returns
    -------
    bool
        True if all variables match within tolerance
    """
    print(f"\n{'='*60}")
    print("CORRECTNESS VALIDATION")
    print(f"{'='*60}")

    all_match = True

    # Check that both have the same variables
    ref_vars = set(ref.keys())
    opt_vars = set(opt.keys())

    if ref_vars != opt_vars:
        print(f"  ❌ Variable mismatch!")
        print(f"     Reference: {sorted(ref_vars)}")
        print(f"     Optimized: {sorted(opt_vars)}")
        return False

    # Compare each variable
    for var in sorted(ref_vars):
        ref_data = ref[var]['data']
        opt_data = opt[var]['data']

        # Check shapes match
        if ref_data.shape != opt_data.shape:
            print(f"  ❌ {var}: shape mismatch ({ref_data.shape} vs {opt_data.shape})")
            all_match = False
            continue

        # Check values match
        if np.allclose(ref_data, opt_data, atol=tolerance, rtol=0):
            max_diff = np.max(np.abs(ref_data - opt_data))
            print(f"  ✓ {var}: IDENTICAL (max diff: {max_diff:.2e})")
        else:
            max_diff = np.max(np.abs(ref_data - opt_data))
            mean_diff = np.mean(np.abs(ref_data - opt_data))
            print(f"  ❌ {var}: MISMATCH (max diff: {max_diff:.2e}, mean: {mean_diff:.2e})")
            all_match = False

    if all_match:
        print(f"\n✓ All variables match within tolerance ({tolerance})")
    else:
        print(f"\n❌ Some variables do not match!")

    return all_match


def main():
    """Run benchmark suite."""
    # Test configuration
    EXPT = 'xqhuj'
    START_YEAR = 1850
    END_YEAR = 1852  # Small range for quick testing
    N_RUNS = 2

    print("="*70)
    print("PERFORMANCE BENCHMARK & CORRECTNESS VALIDATION")
    print("="*70)
    print(f"Experiment: {EXPT}")
    print(f"Year range: {START_YEAR}-{END_YEAR}")
    print(f"Runs per benchmark: {N_RUNS}")
    print("="*70)

    # Benchmark raw pathway
    raw_results = benchmark_raw(EXPT, START_YEAR, END_YEAR, n_runs=N_RUNS)

    # Note: For correctness comparison, you would need to save reference
    # results from before the optimization and load them here.
    # For now, we just report the benchmark timing.

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Raw pathway: {raw_results['mean_time']:.2f}s ± {raw_results['std_time']:.2f}s")
    print(f"Variables extracted: {list(raw_results['data'].keys())}")
    for var in raw_results['data']:
        n_years = len(raw_results['data'][var]['data'])
        print(f"  {var}: {n_years} years")

    # Example: If you have reference data, uncomment this:
    # if Path('validation/reference_raw.npy').exists():
    #     ref_data = np.load('validation/reference_raw.npy', allow_pickle=True).item()
    #     match = compare_results(ref_data, raw_results['data'])
    #     if not match:
    #         sys.exit(1)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

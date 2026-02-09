"""
Smoke tests for import resolution.

Tests that all public API functions can be imported without errors.
"""

import sys
import os

# Add src to path for testing without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_import_main_package():
    """Test that main package can be imported."""
    try:
        import utils_cmip7
        assert hasattr(utils_cmip7, '__version__')
        print("✓ Main package imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import utils_cmip7: {e}")
        return False


def test_import_io_functions():
    """Test that I/O functions can be imported."""
    try:
        from utils_cmip7 import stash, try_extract, find_matching_files
        from utils_cmip7.io import stash_nc, decode_month
        print("✓ I/O functions import successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import I/O functions: {e}")
        return False


def test_import_processing_functions():
    """Test that processing functions can be imported."""
    try:
        from utils_cmip7 import (
            global_total_pgC,
            global_mean_pgC,
            compute_regional_annual_mean,
            merge_monthly_results,
            compute_monthly_mean,
            compute_annual_mean,
        )
        from utils_cmip7.processing import load_reccap_mask, region_mask
        print("✓ Processing functions import successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import processing functions: {e}")
        return False


def test_import_diagnostics_functions():
    """Test that diagnostics functions can be imported."""
    try:
        from utils_cmip7 import extract_annual_means, extract_annual_mean_raw
        print("✓ Diagnostics functions import successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import diagnostics functions: {e}")
        return False


def test_import_config():
    """Test that configuration can be imported."""
    try:
        from utils_cmip7 import (
            VAR_CONVERSIONS,
            RECCAP_MASK_PATH,
            validate_reccap_mask_path,
            get_config_info,
        )
        from utils_cmip7.config import RECCAP_REGIONS
        print("✓ Configuration imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import configuration: {e}")
        return False


def test_backward_compatible_imports():
    """Test that legacy imports still work."""
    try:
        # Add project root to path
        project_root = os.path.join(os.path.dirname(__file__), '..')
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Suppress deprecation warnings for this test
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from analysis import extract_annual_means, stash
            from plot import plot_timeseries_grouped

        print("✓ Backward-compatible imports work (with deprecation warnings)")
        return True
    except ImportError as e:
        print(f"✗ Failed to import via legacy path: {e}")
        return False


def run_all_import_tests():
    """Run all import tests and report results."""
    print("=" * 80)
    print("IMPORT RESOLUTION SMOKE TESTS")
    print("=" * 80)
    print()

    tests = [
        test_import_main_package,
        test_import_io_functions,
        test_import_processing_functions,
        test_import_diagnostics_functions,
        test_import_config,
        test_backward_compatible_imports,
    ]

    results = []
    for test in tests:
        results.append(test())
        print()

    print("=" * 80)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 80)

    return all(results)


if __name__ == '__main__':
    success = run_all_import_tests()
    sys.exit(0 if success else 1)

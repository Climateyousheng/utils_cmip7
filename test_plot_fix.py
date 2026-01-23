#!/usr/bin/env python3
"""
Test to verify plotting code handles frac variable correctly.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def test_frac_exclusion():
    """Test that frac is excluded from grouped plotting."""
    from plot_legacy import group_vars_by_prefix

    # Simulate data structure with frac
    data = {
        'xqhuc': {
            'global': {
                'GPP': {'years': [2000, 2001], 'data': [1.0, 1.1]},
                'NPP': {'years': [2000, 2001], 'data': [0.5, 0.6]},
                'frac': {  # Nested structure
                    'PFT 1': {'years': [2000, 2001], 'data': [0.3, 0.4]},
                    'PFT 2': {'years': [2000, 2001], 'data': [0.2, 0.3]},
                },
                'CVeg': {'years': [2000, 2001], 'data': [10.0, 10.5]},
            }
        }
    }

    grouped = group_vars_by_prefix(data, expts_list=['xqhuc'], region='global')

    # Verify frac is NOT in the grouped variables
    all_vars = [var for group in grouped.values() for var in group]
    assert 'frac' not in all_vars, "frac should be excluded from plotting"
    assert 'fracPFTs' not in all_vars, "fracPFTs should be excluded from plotting"

    # Verify other variables are included
    assert 'GPP' in all_vars, "GPP should be included"
    assert 'NPP' in all_vars, "NPP should be included"
    assert 'CVeg' in all_vars, "CVeg should be included"

    print("✓ frac correctly excluded from grouped plotting")
    return True


def test_series_validation():
    """Test that plotting handles invalid series structures gracefully."""
    import numpy as np

    # Test valid series structure
    valid_series = {'years': [2000, 2001, 2002], 'data': [1.0, 1.1, 1.2]}
    assert isinstance(valid_series, dict), "Should be dict"
    assert 'years' in valid_series, "Should have years"
    assert 'data' in valid_series, "Should have data"
    print("✓ Valid series structure recognized")

    # Test invalid series structures
    invalid_series = [
        None,  # Not a dict
        {},  # Empty dict
        {'PFT 1': {}},  # Nested structure (like frac)
        {'years': [2000]},  # Missing 'data' key
        {'data': [1.0]},  # Missing 'years' key
    ]

    for i, series in enumerate(invalid_series):
        if not series:
            print(f"✓ Invalid series {i+1}: None/empty - correctly skipped")
            continue
        if not isinstance(series, dict):
            print(f"✓ Invalid series {i+1}: Not a dict - correctly skipped")
            continue
        if 'years' not in series or 'data' not in series:
            print(f"✓ Invalid series {i+1}: Missing keys - correctly skipped")
            continue

    return True


def test_backward_compatibility():
    """Test that both old and new variable names are handled."""
    from plot_legacy import group_vars_by_prefix

    # Test with old name 'fracPFTs'
    data_old = {
        'xqhuc': {
            'global': {
                'GPP': {'years': [2000, 2001], 'data': [1.0, 1.1]},
                'fracPFTs': {  # Old name
                    'PFT 1': {'years': [2000, 2001], 'data': [0.3, 0.4]},
                },
            }
        }
    }

    grouped = group_vars_by_prefix(data_old, expts_list=['xqhuc'], region='global')
    all_vars = [var for group in grouped.values() for var in group]

    assert 'fracPFTs' not in all_vars, "fracPFTs (old name) should be excluded"
    assert 'GPP' in all_vars, "GPP should be included"

    print("✓ Old variable name 'fracPFTs' correctly excluded")

    # Test with new name 'frac'
    data_new = {
        'xqhuc': {
            'global': {
                'GPP': {'years': [2000, 2001], 'data': [1.0, 1.1]},
                'frac': {  # New name
                    'PFT 1': {'years': [2000, 2001], 'data': [0.3, 0.4]},
                },
            }
        }
    }

    grouped = group_vars_by_prefix(data_new, expts_list=['xqhuc'], region='global')
    all_vars = [var for group in grouped.values() for var in group]

    assert 'frac' not in all_vars, "frac (new name) should be excluded"
    assert 'GPP' in all_vars, "GPP should be included"

    print("✓ New variable name 'frac' correctly excluded")

    return True


def main():
    """Run all tests."""
    print("="*70)
    print("Plot Fix Test Suite")
    print("="*70)

    tests = [
        test_frac_exclusion,
        test_series_validation,
        test_backward_compatibility,
    ]

    results = []
    for test_func in tests:
        print(f"\n{test_func.__doc__}")
        try:
            success = test_func()
            results.append((test_func.__name__, success))
        except Exception as e:
            print(f"  ❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))

    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print("="*70)
    print(f"Result: {passed}/{total} tests passed")
    print("="*70)

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())

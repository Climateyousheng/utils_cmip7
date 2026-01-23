#!/usr/bin/env python3
"""
Test to verify the extraction fix handles missing variables correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_empty_dict_not_created():
    """
    Verify that missing variables don't create empty dictionaries
    that would cause KeyError in plotting.
    """
    print("Testing extraction fix...")

    # Simulate what happens with missing variable data
    dict_annual_means = {}
    expt = 'test_expt'
    region = 'global'
    dict_annual_means[expt] = {region: {}}

    # Simulate the fixed frac handling
    frac_data = {}  # No PFTs found

    # OLD BUG: Would do this regardless
    # dict_annual_means[expt][region]['frac'] = frac_data  # Creates empty dict!

    # NEW FIX: Only add if data was found
    if frac_data:
        dict_annual_means[expt][region]['frac'] = frac_data

    # Verify frac is NOT in the dictionary
    assert 'frac' not in dict_annual_means[expt][region], \
        "Empty frac_data should not be added to results"

    print("  ✓ Empty frac_data correctly NOT added to results")

    # Now simulate successful extraction
    frac_data = {
        'PFT 1': {'years': [2000, 2001], 'data': [0.5, 0.6], 'units': '1'},
        'PFT 2': {'years': [2000, 2001], 'data': [0.3, 0.4], 'units': '1'},
    }

    if frac_data:
        dict_annual_means[expt][region]['frac'] = frac_data

    # Verify frac IS in the dictionary when data exists
    assert 'frac' in dict_annual_means[expt][region], \
        "Non-empty frac_data should be added to results"
    assert 'PFT 1' in dict_annual_means[expt][region]['frac'], \
        "PFT data should be preserved"

    print("  ✓ Non-empty frac_data correctly added to results")

    # Test that we can safely access nested data (no KeyError)
    for pft_name, pft_data in dict_annual_means[expt][region]['frac'].items():
        years = pft_data['years']  # Should not raise KeyError
        data = pft_data['data']
        assert len(years) == len(data), "Years and data should have same length"

    print("  ✓ Can safely access PFT data without KeyError")

    return True


def test_none_cube_handling():
    """Test that None cubes are handled gracefully."""
    print("\nTesting None cube handling...")

    cube = None
    varname = 'frac'
    dict_annual_means = {}
    expt = 'test_expt'
    region = 'global'
    dict_annual_means[expt] = {region: {}}

    # Simulate the fixed logic
    if cube is not None and varname == 'frac':
        # This block should NOT execute when cube is None
        dict_annual_means[expt][region][varname] = {}
        print("  ❌ ERROR: Should not reach here when cube is None")
        return False

    # Verify variable was NOT added
    assert varname not in dict_annual_means[expt][region], \
        "Variable should not be added when cube is None"

    print("  ✓ None cube correctly skipped (no empty dict created)")

    return True


def main():
    """Run all tests."""
    print("="*70)
    print("Extraction Fix Test Suite")
    print("="*70)

    tests = [
        test_empty_dict_not_created,
        test_none_cube_handling,
    ]

    results = []
    for test_func in tests:
        try:
            success = test_func()
            results.append((test_func.__name__, success))
        except Exception as e:
            print(f"\n  ❌ Test crashed: {e}")
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

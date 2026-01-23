#!/usr/bin/env python3
"""
Verification script for repository hygiene changes.
Run this in your normal Python environment to verify all imports work correctly.
"""

import sys
import os

def test_package_import():
    """Test that utils_cmip7 imports from src/"""
    try:
        import utils_cmip7
        print(f"✓ Package imports correctly from: {utils_cmip7.__file__}")
        assert 'src/utils_cmip7' in utils_cmip7.__file__, "Package not importing from src/"
        return True
    except Exception as e:
        print(f"✗ Package import failed: {e}")
        return False


def test_validation_module():
    """Test that validation module doesn't shadow with validation_outputs/"""
    try:
        import utils_cmip7.validation
        print(f"✓ Validation module imports correctly from: {utils_cmip7.validation.__file__}")
        assert 'src/utils_cmip7/validation' in utils_cmip7.validation.__file__
        return True
    except Exception as e:
        print(f"✗ Validation module import failed: {e}")
        return False


def test_obs_loader():
    """Test that obs_loader can find data using importlib.resources"""
    try:
        from utils_cmip7.io.obs_loader import get_obs_dir
        obs_dir = get_obs_dir()
        print(f"✓ obs_loader found data at: {obs_dir}")

        # Verify CSV files exist
        import os
        csv_files = [
            'stores_vs_fluxes_cmip6.csv',
            'stores_vs_fluxes_cmip6_err.csv',
            'stores_vs_fluxes_reccap.csv',
            'stores_vs_fluxes_reccap_err.csv'
        ]
        for csv_file in csv_files:
            csv_path = os.path.join(obs_dir, csv_file)
            assert os.path.exists(csv_path), f"Missing {csv_file}"

        print(f"  ✓ All 4 CSV files present")
        return True
    except Exception as e:
        print(f"✗ obs_loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_root_init():
    """Test that root __init__.py doesn't exist"""
    repo_root = os.path.dirname(__file__)
    root_init = os.path.join(repo_root, '__init__.py')
    if not os.path.exists(root_init):
        print("✓ Root __init__.py correctly removed")
        return True
    else:
        print(f"✗ Root __init__.py still exists at {root_init}")
        return False


def test_directory_structure():
    """Test that directory structure is correct"""
    repo_root = os.path.dirname(__file__)

    checks = [
        ('docs/', 'docs/ directory exists'),
        ('tests/', 'tests/ directory exists'),
        ('validation_outputs/', 'validation_outputs/ directory exists'),
        ('src/utils_cmip7/data/obs/', 'obs/ moved to package data'),
    ]

    missing = [
        ('doc/', 'doc/ correctly removed'),
        ('obs/', 'obs/ correctly moved'),
        ('validation/', 'validation/ correctly renamed'),
    ]

    all_pass = True

    for path, desc in checks:
        full_path = os.path.join(repo_root, path)
        if os.path.exists(full_path):
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc} - NOT FOUND at {full_path}")
            all_pass = False

    for path, desc in missing:
        full_path = os.path.join(repo_root, path)
        if not os.path.exists(full_path):
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc} - STILL EXISTS at {full_path}")
            all_pass = False

    return all_pass


def main():
    print("="*70)
    print("Repository Hygiene Verification")
    print("="*70)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Root __init__.py Removal", test_no_root_init),
        ("Package Import", test_package_import),
        ("Validation Module", test_validation_module),
        ("Obs Loader", test_obs_loader),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            results.append((name, False))

    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    print("="*70)
    print(f"Result: {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("\n✅ All repository hygiene changes verified!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

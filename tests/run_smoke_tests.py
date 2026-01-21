#!/usr/bin/env python3
"""
Run all smoke tests for utils_cmip7 package.

Smoke tests verify basic functionality without requiring sample data:
- Import resolution
- Configuration loading
- Validation functions
- Backward compatibility

These tests should run quickly and catch obvious integration issues.
"""

import sys
import os

# Add src to path for testing without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def main():
    """Run all smoke test suites."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 24 + "utils_cmip7 SMOKE TESTS" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    results = {}

    # Import test modules
    try:
        import test_imports
        import test_config
    except ImportError as e:
        print(f"✗ Failed to import test modules: {e}")
        print("  Make sure you're running from the tests/ directory")
        return False

    # Run import tests
    print()
    results['imports'] = test_imports.run_all_import_tests()

    # Run config tests
    print()
    results['config'] = test_config.run_all_config_tests()

    # Final summary
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 28 + "OVERALL SUMMARY" + " " * 35 + "║")
    print("╠" + "═" * 78 + "╣")

    for suite_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"║  {suite_name.capitalize():20s} {status:56s} ║")

    print("╚" + "═" * 78 + "╝")
    print()

    all_passed = all(results.values())

    if all_passed:
        print("✓ All smoke tests passed!")
        print()
        print("Next steps:")
        print("  1. Install package: pip install -e .")
        print("  2. Run with actual data to test extraction functions")
        print("  3. Add unit tests for individual functions")
        print()
    else:
        print("✗ Some smoke tests failed.")
        print()
        print("Common issues:")
        print("  - Missing dependencies (numpy, iris, etc.)")
        print("    Solution: pip install -e .")
        print("  - Import path issues")
        print("    Solution: Run from tests/ directory or install package")
        print()

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

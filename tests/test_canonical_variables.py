#!/usr/bin/env python3
"""
Quick test script to verify canonical variable registry implementation.
"""

from utils_cmip7 import config

CANONICAL_VARIABLES = config.CANONICAL_VARIABLES
resolve_variable_name = config.resolve_variable_name
get_variable_config = config.get_variable_config
get_conversion_key = config.get_conversion_key
DEFAULT_VAR_LIST = config.DEFAULT_VAR_LIST

def test_canonical_variables_registry():
    """Test that the canonical variables registry is properly structured."""
    print("Testing CANONICAL_VARIABLES registry...")

    required_fields = {'description', 'stash_name', 'stash_code', 'aggregation',
                      'conversion_factor', 'units', 'category', 'aliases'}

    for var_name, config in CANONICAL_VARIABLES.items():
        # Check all required fields present
        missing = required_fields - set(config.keys())
        if missing:
            print(f"  ❌ {var_name} missing fields: {missing}")
            return False

        # Check aggregation is valid
        if config['aggregation'] not in ('MEAN', 'SUM'):
            print(f"  ❌ {var_name} has invalid aggregation: {config['aggregation']}")
            return False

        # Check category is valid
        if config['category'] not in ('flux', 'stock', 'climate', 'land_use'):
            print(f"  ❌ {var_name} has invalid category: {config['category']}")
            return False

    print(f"  ✓ All {len(CANONICAL_VARIABLES)} variables properly structured")
    return True


def test_resolve_variable_name():
    """Test variable name resolution."""
    print("\nTesting resolve_variable_name()...")

    test_cases = [
        ('CVeg', 'CVeg'),         # Canonical → canonical
        ('VegCarb', 'CVeg'),      # Alias → canonical
        ('Rh', 'Rh'),             # Canonical → canonical
        ('soilResp', 'Rh'),       # Alias → canonical
        ('CSoil', 'CSoil'),       # Canonical → canonical
        ('soilCarbon', 'CSoil'),  # Alias → canonical
        ('tas', 'tas'),           # Canonical → canonical
        ('temp', 'tas'),          # Alias → canonical
        ('pr', 'pr'),             # Canonical → canonical
        ('precip', 'pr'),         # Alias → canonical
        ('frac', 'frac'),         # Canonical → canonical
        ('fracPFTs', 'frac'),     # Alias → canonical
    ]

    for input_name, expected in test_cases:
        result = resolve_variable_name(input_name)
        if result != expected:
            print(f"  ❌ {input_name} → {result}, expected {expected}")
            return False
        print(f"  ✓ {input_name} → {result}")

    # Test invalid variable
    try:
        resolve_variable_name('invalid_var')
        print(f"  ❌ Should have raised ValueError for invalid variable")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError for invalid variable")

    return True


def test_get_variable_config():
    """Test getting variable configuration."""
    print("\nTesting get_variable_config()...")

    # Test canonical name
    config = get_variable_config('CVeg')
    assert config['canonical_name'] == 'CVeg'
    assert config['stash_name'] == 'cv'
    assert config['aggregation'] == 'SUM'
    print(f"  ✓ CVeg config retrieved correctly")

    # Test alias
    config = get_variable_config('VegCarb')
    assert config['canonical_name'] == 'CVeg'
    assert config['stash_name'] == 'cv'
    print(f"  ✓ VegCarb (alias) config retrieved correctly")

    # Test MEAN aggregation variable
    config = get_variable_config('tas')
    assert config['aggregation'] == 'MEAN'
    assert config['canonical_name'] == 'tas'
    print(f"  ✓ tas config retrieved correctly (MEAN aggregation)")

    return True


def test_get_conversion_key():
    """Test conversion key generation."""
    print("\nTesting get_conversion_key()...")

    test_cases = [
        # SUM aggregation → use canonical name
        ('GPP', 'GPP'),
        ('NPP', 'NPP'),
        ('Rh', 'Rh'),
        ('CVeg', 'CVeg'),
        ('CSoil', 'CSoil'),
        ('fgco2', 'fgco2'),

        # MEAN aggregation → use 'Others' (except pr and co2)
        ('tas', 'Others'),
        ('temp', 'Others'),  # alias
        ('frac', 'Others'),

        # Special case: pr → 'precip'
        ('pr', 'precip'),
        ('precip', 'precip'),  # alias

        # Special case: co2 → 'Total co2'
        ('co2', 'Total co2'),
        ('Total co2', 'Total co2'),  # alias
    ]

    for var_name, expected_key in test_cases:
        result = get_conversion_key(var_name)
        if result != expected_key:
            print(f"  ❌ {var_name} → '{result}', expected '{expected_key}'")
            return False
        print(f"  ✓ {var_name} → '{result}'")

    return True


def test_default_var_list():
    """Test that default variable list uses canonical names."""
    print("\nTesting DEFAULT_VAR_LIST...")

    for var in DEFAULT_VAR_LIST:
        try:
            canonical = resolve_variable_name(var)
            if var != canonical:
                print(f"  ⚠ DEFAULT_VAR_LIST contains alias '{var}' instead of canonical '{canonical}'")
                return False
            print(f"  ✓ {var} is canonical")
        except ValueError:
            print(f"  ❌ {var} is not a valid variable")
            return False

    return True


def test_aggregation_semantics():
    """Test that aggregation method is correctly encoded."""
    print("\nTesting aggregation semantics...")

    # MEAN aggregation variables
    mean_vars = ['tas', 'pr', 'frac', 'co2']
    for var in mean_vars:
        config = get_variable_config(var)
        if config['aggregation'] != 'MEAN':
            print(f"  ❌ {var} should have MEAN aggregation, got {config['aggregation']}")
            return False
        conv_key = get_conversion_key(var)
        if conv_key not in ('Others', 'precip', 'Total co2'):
            print(f"  ❌ {var} should map to 'Others', 'precip', or 'Total co2', got '{conv_key}'")
            return False
        print(f"  ✓ {var}: MEAN aggregation, conversion_key='{conv_key}'")

    # SUM aggregation variables
    sum_vars = ['GPP', 'NPP', 'Rh', 'CVeg', 'CSoil', 'fgco2']
    for var in sum_vars:
        config = get_variable_config(var)
        if config['aggregation'] != 'SUM':
            print(f"  ❌ {var} should have SUM aggregation, got {config['aggregation']}")
            return False
        conv_key = get_conversion_key(var)
        if conv_key == 'Others':
            print(f"  ❌ {var} (SUM) should not map to 'Others'")
            return False
        print(f"  ✓ {var}: SUM aggregation, conversion_key='{conv_key}'")

    return True


def main():
    """Run all tests."""
    print("="*70)
    print("Canonical Variables Registry Test Suite")
    print("="*70)

    tests = [
        test_canonical_variables_registry,
        test_resolve_variable_name,
        test_get_variable_config,
        test_get_conversion_key,
        test_default_var_list,
        test_aggregation_semantics,
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

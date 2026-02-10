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
                      'conversion_factor', 'units', 'units_in', 'time_handling',
                      'category', 'aliases'}

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
    """Test variable name resolution — canonical names resolve, aliases raise."""
    import pytest

    # Canonical names resolve to themselves
    canonical_cases = [
        ('CVeg', 'CVeg'),
        ('Rh', 'Rh'),
        ('CSoil', 'CSoil'),
        ('tas', 'tas'),
        ('pr', 'pr'),
        ('frac', 'frac'),
    ]

    for input_name, expected in canonical_cases:
        result = resolve_variable_name(input_name)
        assert result == expected, f"{input_name} → {result}, expected {expected}"

    # Legacy aliases raise ValueError with migration message (removed in v0.4.0)
    alias_cases = [
        ('VegCarb', 'CVeg'),
        ('soilResp', 'Rh'),
        ('soilCarbon', 'CSoil'),
        ('temp', 'tas'),
        ('precip', 'pr'),
        ('fracPFTs', 'frac'),
    ]

    for alias, canonical in alias_cases:
        with pytest.raises(ValueError, match="removed in v0.4.0"):
            resolve_variable_name(alias)

    # Unknown variable raises ValueError
    with pytest.raises(ValueError, match="Unknown variable name"):
        resolve_variable_name('invalid_var')


def test_get_variable_config():
    """Test getting variable configuration."""
    import pytest

    # Test canonical name
    cfg = get_variable_config('CVeg')
    assert cfg['canonical_name'] == 'CVeg'
    assert cfg['stash_name'] == 'cv'
    assert cfg['aggregation'] == 'SUM'

    # Test alias raises ValueError (removed in v0.4.0)
    with pytest.raises(ValueError, match="removed in v0.4.0"):
        get_variable_config('VegCarb')

    # Test MEAN aggregation variable
    cfg = get_variable_config('tas')
    assert cfg['aggregation'] == 'MEAN'
    assert cfg['canonical_name'] == 'tas'


def test_get_conversion_key():
    """Test conversion key generation — canonical names only."""
    import pytest

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
        ('frac', 'Others'),

        # Special case: pr → 'precip'
        ('pr', 'precip'),

        # Special case: co2 → 'Total co2'
        ('co2', 'Total co2'),
    ]

    for var_name, expected_key in test_cases:
        result = get_conversion_key(var_name)
        assert result == expected_key, f"{var_name} → '{result}', expected '{expected_key}'"

    # Legacy aliases raise ValueError
    with pytest.raises(ValueError, match="removed in v0.4.0"):
        get_conversion_key('temp')
    with pytest.raises(ValueError, match="removed in v0.4.0"):
        get_conversion_key('precip')


def test_units_in_field():
    """Test that every variable has a units_in field (string)."""
    for var_name, cfg in CANONICAL_VARIABLES.items():
        assert "units_in" in cfg, f"{var_name} missing units_in"
        assert isinstance(cfg["units_in"], str), f"{var_name} units_in must be str"
        assert len(cfg["units_in"]) > 0, f"{var_name} units_in must be non-empty"


def test_time_handling_field():
    """Test that every variable has a valid time_handling field."""
    valid_values = {"mean_rate", "state", "already_integral"}
    for var_name, cfg in CANONICAL_VARIABLES.items():
        assert "time_handling" in cfg, f"{var_name} missing time_handling"
        assert cfg["time_handling"] in valid_values, (
            f"{var_name} time_handling='{cfg['time_handling']}' "
            f"not in {valid_values}"
        )


def test_var_conversions_derived_from_canonical():
    """VAR_CONVERSIONS must match CANONICAL_VARIABLES conversion factors."""
    import pytest
    from utils_cmip7.config import VAR_CONVERSIONS

    for name, cfg in CANONICAL_VARIABLES.items():
        assert name in VAR_CONVERSIONS, f"{name} missing from VAR_CONVERSIONS"
        assert VAR_CONVERSIONS[name] == pytest.approx(cfg["conversion_factor"]), (
            f"{name}: VAR_CONVERSIONS={VAR_CONVERSIONS[name]} != "
            f"CANONICAL_VARIABLES={cfg['conversion_factor']}"
        )


def test_legacy_protocol_keys_in_var_conversions():
    """Legacy protocol keys must still exist in VAR_CONVERSIONS."""
    from utils_cmip7.config import VAR_CONVERSIONS

    assert "Others" in VAR_CONVERSIONS
    assert "precip" in VAR_CONVERSIONS
    assert "Total co2" in VAR_CONVERSIONS


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

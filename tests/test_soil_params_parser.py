"""
Tests for soil parameter parsing and SoilParamSet functionality.

Validates LAND_CC namelist parser and SoilParamSet loaders.
"""

import sys
import os

# Add src to path for testing without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# Default LAND_CC block from MIGRATION.md
DEFAULT_LAND_CC_TEXT = """
&LAND_CC
 ALPHA=0.08,0.08,0.08,0.040,0.08,
 F0=0.875,0.875,0.900,0.800,0.900,
 G_AREA=0.004,0.004,0.10,0.10,0.05,
 LAI_MIN=4.0,4.0,1.0,1.0,1.0,
 NL0=0.050,0.030,0.060,0.030,0.030,
 R_GROW=0.25,0.25,0.25,0.25,0.25,
 TLOW=-0.0,-5.0,0.0,13.0,0.0,
 TUPP=36.0,31.0,36.0,45.0,36.0,
 Q10=2.0,
 V_CRIT_ALPHA=0.343,
 KAPS=5e-009,
/
"""


def test_parse_default_land_cc_block():
    """Test parsing the default LAND_CC block from MIGRATION.md."""
    try:
        from utils_cmip7.soil_params.parsers import parse_land_cc_block

        params = parse_land_cc_block(DEFAULT_LAND_CC_TEXT)

        # Check that parsing returns a dict
        assert isinstance(params, dict), "Parser should return dict"
        assert len(params) > 0, "Parser should extract parameters"

        print("✓ Successfully parsed default LAND_CC block")
        print(f"  - Extracted {len(params)} parameters")
        return True
    except Exception as e:
        print(f"✗ Failed to parse default LAND_CC block: {e}")
        return False


def test_all_parameters_present():
    """Verify all 11 expected parameters are present."""
    try:
        from utils_cmip7.soil_params.parsers import parse_land_cc_block

        params = parse_land_cc_block(DEFAULT_LAND_CC_TEXT)

        expected_params = [
            'ALPHA', 'F0', 'G_AREA', 'LAI_MIN', 'NL0', 'R_GROW', 'TLOW', 'TUPP',
            'Q10', 'V_CRIT_ALPHA', 'KAPS'
        ]

        missing = [p for p in expected_params if p not in params]

        if missing:
            print(f"✗ Missing parameters: {missing}")
            return False

        print("✓ All 11 expected parameters present")
        return True
    except Exception as e:
        print(f"✗ Failed to check parameter presence: {e}")
        return False


def test_array_lengths():
    """Verify array parameters have length 5 (5 PFTs)."""
    try:
        from utils_cmip7.soil_params.parsers import parse_land_cc_block

        params = parse_land_cc_block(DEFAULT_LAND_CC_TEXT)

        array_params = ['ALPHA', 'F0', 'G_AREA', 'LAI_MIN', 'NL0', 'R_GROW', 'TLOW', 'TUPP']

        for param in array_params:
            if param not in params:
                print(f"✗ Array parameter {param} missing")
                return False
            if not isinstance(params[param], list):
                print(f"✗ {param} should be a list, got {type(params[param])}")
                return False
            if len(params[param]) != 5:
                print(f"✗ {param} should have length 5, got {len(params[param])}")
                return False

        print("✓ All array parameters have correct length (5)")
        return True
    except Exception as e:
        print(f"✗ Failed to check array lengths: {e}")
        return False


def test_scalar_values():
    """Verify scalar parameters have correct values."""
    try:
        from utils_cmip7.soil_params.parsers import parse_land_cc_block

        params = parse_land_cc_block(DEFAULT_LAND_CC_TEXT)

        expected_scalars = {
            'Q10': 2.0,
            'V_CRIT_ALPHA': 0.343,
            'KAPS': 5e-009,
        }

        for key, expected_val in expected_scalars.items():
            if key not in params:
                print(f"✗ Scalar parameter {key} missing")
                return False
            actual_val = params[key]
            if abs(actual_val - expected_val) > 1e-10:
                print(f"✗ {key} should be {expected_val}, got {actual_val}")
                return False

        print("✓ All scalar parameters have correct values")
        return True
    except Exception as e:
        print(f"✗ Failed to check scalar values: {e}")
        return False


def test_missing_block_raises():
    """Test that ValueError is raised when LAND_CC block not found."""
    try:
        from utils_cmip7.soil_params.parsers import parse_land_cc_block

        text_without_block = """
        This is some random text
        without a LAND_CC block
        """

        try:
            parse_land_cc_block(text_without_block)
            print("✗ Parser should raise ValueError for missing block")
            return False
        except ValueError as e:
            if "LAND_CC" in str(e):
                print("✓ Parser correctly raises ValueError for missing block")
                return True
            else:
                print(f"✗ ValueError message should mention LAND_CC: {e}")
                return False
    except Exception as e:
        print(f"✗ Failed to test missing block error: {e}")
        return False


def test_soilparamset_from_default():
    """Test SoilParamSet.from_default() creates valid parameter set."""
    try:
        from utils_cmip7.soil_params import SoilParamSet, DEFAULT_LAND_CC

        params = SoilParamSet.from_default()

        # Check source is marked as default
        assert params.source == 'default', "Source should be 'default'"

        # Check all arrays have correct length
        for key in ['ALPHA', 'F0', 'G_AREA', 'LAI_MIN', 'NL0', 'R_GROW', 'TLOW', 'TUPP']:
            arr = getattr(params, key)
            assert isinstance(arr, list), f"{key} should be a list"
            assert len(arr) == 5, f"{key} should have length 5"

        # Check scalars match DEFAULT_LAND_CC
        assert params.Q10 == DEFAULT_LAND_CC['Q10']
        assert params.V_CRIT_ALPHA == DEFAULT_LAND_CC['V_CRIT_ALPHA']
        assert params.KAPS == DEFAULT_LAND_CC['KAPS']

        print("✓ SoilParamSet.from_default() creates valid parameter set")
        return True
    except Exception as e:
        print(f"✗ Failed to create SoilParamSet from default: {e}")
        return False


def test_bl_subset_extraction():
    """Test to_bl_subset() and to_overview_table_format() methods."""
    try:
        from utils_cmip7.soil_params import SoilParamSet, BL_INDEX

        params = SoilParamSet.from_default()

        # Test to_bl_subset()
        bl_subset = params.to_bl_subset()

        # Check BL-suffixed keys
        expected_bl_keys = ['ALPHA_BL', 'F0_BL', 'G_AREA_BL', 'LAI_MIN_BL',
                            'NL0_BL', 'R_GROW_BL', 'TLOW_BL', 'TUPP_BL']
        for key in expected_bl_keys:
            assert key in bl_subset, f"Missing {key} in BL subset"

        # Check scalars
        assert 'Q10' in bl_subset
        assert 'V_CRIT_ALPHA' in bl_subset
        assert 'KAPS' in bl_subset

        # Test to_overview_table_format()
        overview_params = params.to_overview_table_format()

        # Check keys match overview table schema (no _BL suffix)
        expected_overview_keys = ['ALPHA', 'G_AREA', 'LAI_MIN', 'NL0',
                                   'R_GROW', 'TLOW', 'TUPP', 'V_CRIT']
        for key in expected_overview_keys:
            assert key in overview_params, f"Missing {key} in overview format"

        # F0, Q10, KAPS should NOT be in overview format (excluded from standard table)
        assert 'F0' not in overview_params, "F0 should not be in overview format"
        assert 'Q10' not in overview_params, "Q10 should not be in overview format"
        assert 'KAPS' not in overview_params, "KAPS should not be in overview format"

        print("✓ BL subset extraction works correctly")
        print(f"  - to_bl_subset() returns {len(bl_subset)} values")
        print(f"  - to_overview_table_format() returns {len(overview_params)} values")
        return True
    except Exception as e:
        print(f"✗ Failed BL subset extraction test: {e}")
        return False


def run_all_soil_params_tests():
    """Run all soil parameter tests and report results."""
    print("=" * 80)
    print("SOIL PARAMETER PARSING TESTS")
    print("=" * 80)
    print()

    tests = [
        test_parse_default_land_cc_block,
        test_all_parameters_present,
        test_array_lengths,
        test_scalar_values,
        test_missing_block_raises,
        test_soilparamset_from_default,
        test_bl_subset_extraction,
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
    success = run_all_soil_params_tests()
    sys.exit(0 if success else 1)

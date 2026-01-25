"""
Test STASH code mapping utilities.

Tests for src/utils_cmip7/io/stash.py
"""
import pytest
from utils_cmip7.io.stash import stash, stash_nc


class TestStashMSIFormat:
    """Test stash() function - MSI format STASH codes."""

    def test_gpp_mapping(self):
        """Test GPP maps to correct MSI code."""
        assert stash('gpp') == 'm01s03i261'

    def test_npp_mapping(self):
        """Test NPP maps to correct MSI code."""
        assert stash('npp') == 'm01s03i262'

    def test_tas_mapping(self):
        """Test surface temperature maps to correct MSI code."""
        assert stash('tas') == 'm01s03i236'

    def test_pr_mapping(self):
        """Test precipitation maps to correct MSI code."""
        assert stash('pr') == 'm01s05i216'

    def test_vegetation_carbon_mapping(self):
        """Test vegetation carbon maps to correct MSI code."""
        assert stash('cv') == 'm01s19i002'

    def test_soil_carbon_mapping(self):
        """Test soil carbon maps to correct MSI code."""
        assert stash('cs') == 'm01s19i016'

    def test_pft_fraction_mapping(self):
        """Test PFT fraction maps to correct MSI code."""
        assert stash('frac') == 'm01s19i013'

    def test_unknown_variable_returns_nothing(self):
        """Test unknown variable returns 'nothing'."""
        assert stash('unknown_var') == 'nothing'
        assert stash('INVALID') == 'nothing'
        assert stash('') == 'nothing'

    def test_all_carbon_cycle_variables(self):
        """Test all carbon cycle related variables."""
        carbon_vars = {
            'gpp': 'm01s03i261',
            'npp': 'm01s03i262',
            'rh': 'm01s03i293',
            'cv': 'm01s19i002',
            'cs': 'm01s19i016',
        }
        for var, expected_code in carbon_vars.items():
            assert stash(var) == expected_code, f"Failed for {var}"

    def test_ocean_variables(self):
        """Test ocean-related STASH codes."""
        ocean_vars = {
            'tos': 'm02s00i101',
            'sal': 'm02s00i102',
            'pco2': 'm02s30i248',
            'fgco2': 'm02s30i249',
        }
        for var, expected_code in ocean_vars.items():
            assert stash(var) == expected_code, f"Failed for {var}"


class TestStashNumericFormat:
    """Test stash_nc() function - numeric STASH codes."""

    def test_gpp_numeric_code(self):
        """Test GPP maps to correct numeric code."""
        assert stash_nc('gpp') == 3261

    def test_npp_numeric_code(self):
        """Test NPP maps to correct numeric code."""
        assert stash_nc('npp') == 3262

    def test_tas_numeric_code(self):
        """Test surface temperature maps to correct numeric code."""
        assert stash_nc('tas') == 3236

    def test_cv_numeric_code(self):
        """Test vegetation carbon maps to correct numeric code."""
        assert stash_nc('cv') == 19002

    def test_cs_numeric_code(self):
        """Test soil carbon maps to correct numeric code."""
        assert stash_nc('cs') == 19016

    def test_unknown_variable_returns_nothing(self):
        """Test unknown variable returns 'nothing'."""
        assert stash_nc('unknown_var') == 'nothing'
        assert stash_nc('INVALID') == 'nothing'

    def test_numeric_type(self):
        """Test that valid codes return integers, not strings."""
        result = stash_nc('gpp')
        assert isinstance(result, int)
        assert result == 3261

    def test_all_variables_have_numeric_codes(self):
        """Test that all variables have corresponding numeric codes."""
        # Test a comprehensive set
        variables = [
            'tas', 'pr', 'gpp', 'npp', 'rh', 'cv', 'cs',
            'dist', 'frac', 'tos', 'pco2', 'fgco2'
        ]
        for var in variables:
            code = stash_nc(var)
            assert code != 'nothing', f"Variable {var} should have numeric code"
            assert isinstance(code, int), f"Code for {var} should be int"


class TestStashConsistency:
    """Test consistency between MSI and numeric formats."""

    def test_msi_and_numeric_correspondence(self):
        """Test that MSI and numeric codes correspond correctly."""
        test_cases = [
            ('gpp', 'm01s03i261', 3261),
            ('npp', 'm01s03i262', 3262),
            ('tas', 'm01s03i236', 3236),
            ('cv', 'm01s19i002', 19002),
            ('cs', 'm01s19i016', 19016),
        ]

        for var, expected_msi, expected_numeric in test_cases:
            assert stash(var) == expected_msi
            assert stash_nc(var) == expected_numeric

    def test_all_variables_exist_in_both_formats(self):
        """Test that variables in one format exist in the other."""
        # These should work in both
        shared_vars = ['gpp', 'npp', 'tas', 'pr', 'cv', 'cs', 'frac']

        for var in shared_vars:
            msi_result = stash(var)
            nc_result = stash_nc(var)

            assert msi_result != 'nothing', f"{var} missing from stash()"
            assert nc_result != 'nothing', f"{var} missing from stash_nc()"


class TestStashEdgeCases:
    """Test edge cases and error handling."""

    def test_case_sensitivity(self):
        """Test that lookups are case-sensitive."""
        # Lowercase works
        assert stash('gpp') == 'm01s03i261'

        # Uppercase should fail (returns 'nothing')
        assert stash('GPP') == 'nothing'
        assert stash('Gpp') == 'nothing'

    def test_empty_string(self):
        """Test empty string handling."""
        assert stash('') == 'nothing'
        assert stash_nc('') == 'nothing'

    def test_whitespace(self):
        """Test whitespace handling."""
        assert stash(' gpp') == 'nothing'  # Leading space
        assert stash('gpp ') == 'nothing'  # Trailing space
        assert stash(' gpp ') == 'nothing'  # Both

    def test_none_input(self):
        """Test None input returns 'nothing'."""
        # dict.get() doesn't raise AttributeError for None
        assert stash(None) == 'nothing'
        assert stash_nc(None) == 'nothing'

    def test_numeric_input(self):
        """Test numeric input returns 'nothing'."""
        # dict.get() works with numeric keys, but our dict has string keys
        # so numeric input returns default value 'nothing'
        assert stash(3261) == 'nothing'
        assert stash_nc(3261) == 'nothing'

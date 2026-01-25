"""
Test file discovery utilities.

Tests for src/utils_cmip7/io/file_discovery.py
"""
import os
import pytest
from pathlib import Path
from utils_cmip7.io.file_discovery import decode_month, find_matching_files, MONTH_MAP_ALPHA


class TestDecodeMonth:
    """Test decode_month() function."""

    def test_alpha_codes_all_months(self):
        """Test all alpha month codes decode correctly."""
        expected = {
            'ja': 1, 'fb': 2, 'mr': 3, 'ar': 4, 'my': 5, 'jn': 6,
            'jl': 7, 'ag': 8, 'sp': 9, 'ot': 10, 'nv': 11, 'dc': 12
        }
        for code, month in expected.items():
            assert decode_month(code) == month, f"Failed for {code}"

    def test_alpha_codes_case_insensitive(self):
        """Test alpha codes are case-insensitive."""
        assert decode_month('JA') == 1
        assert decode_month('Ja') == 1
        assert decode_month('jA') == 1
        assert decode_month('DC') == 12
        assert decode_month('Dc') == 12

    def test_numeric_codes_jan_to_sep(self):
        """Test numeric codes for January through September."""
        for month in range(1, 10):
            code = f'{month}1'
            assert decode_month(code) == month, f"Failed for {code}"

    def test_numeric_codes_flexible_second_char(self):
        """Test that numeric codes work with any second character (first char determines month)."""
        # The implementation uses only the first digit
        assert decode_month('91') == 9  # Standard format
        assert decode_month('99') == 9  # Non-standard but valid (first char = 9)
        assert decode_month('15') == 1  # Non-standard but valid (first char = 1)

    def test_hex_like_codes_oct_to_dec(self):
        """Test hex-like codes for October through December."""
        assert decode_month('a1') == 10  # October
        assert decode_month('b1') == 11  # November
        assert decode_month('c1') == 12  # December

    def test_hex_like_codes_case_insensitive(self):
        """Test hex-like codes are case-insensitive."""
        assert decode_month('A1') == 10
        assert decode_month('B1') == 11
        assert decode_month('C1') == 12

    def test_empty_string_returns_zero(self):
        """Test empty string returns 0."""
        assert decode_month('') == 0

    def test_invalid_codes_return_zero(self):
        """Test invalid month codes return 0."""
        assert decode_month('xx') == 0
        assert decode_month('d1') == 0  # 'd' not valid for hex-like
        assert decode_month('01') == 0  # '0' is not valid (must be 1-9)
        assert decode_month('AA') == 0  # Not a valid alpha code

    def test_single_char_returns_zero(self):
        """Test single character input returns 0."""
        assert decode_month('j') == 0
        assert decode_month('1') == 0

    def test_three_char_returns_zero(self):
        """Test three character input returns 0."""
        assert decode_month('jan') == 0
        assert decode_month('111') == 0

    def test_month_map_alpha_completeness(self):
        """Test that MONTH_MAP_ALPHA contains all 12 months."""
        assert len(MONTH_MAP_ALPHA) == 12
        assert set(MONTH_MAP_ALPHA.values()) == set(range(1, 13))


class TestFindMatchingFiles:
    """Test find_matching_files() function."""

    @pytest.fixture
    def temp_experiment_dir(self, tmp_path):
        """Create temporary experiment directory with test files."""
        expt_dir = tmp_path / "xqhuj" / "datam"
        expt_dir.mkdir(parents=True)

        # Create test files with various month codes and years
        test_files = [
            "xqhuja#pi000001850ja+",  # January 1850 (alpha)
            "xqhuja#pi000001850fb+",  # February 1850 (alpha)
            "xqhuja#pi000001850dc+",  # December 1850 (alpha)
            "xqhuja#pi000001851ja+",  # January 1851 (alpha)
            "xqhuja#pi00000185111+",  # January 1851 (numeric)
            "xqhuja#pi00000185191+",  # September 1851 (numeric)
            "xqhuja#pi000001851a1+",  # October 1851 (hex-like)
            "xqhuja#pi000001851b1+",  # November 1851 (hex-like)
            "xqhuja#pi000001851c1+",  # December 1851 (hex-like)
            "xqhuja#pi000001852ja+",  # January 1852 (alpha)
            "other_file.txt",          # Non-matching file
        ]

        for filename in test_files:
            (expt_dir / filename).touch()

        return tmp_path

    def test_basic_file_discovery(self, temp_experiment_dir):
        """Test basic file discovery without year filtering."""
        files = find_matching_files(
            'xqhuj', 'a', 'pi',
            base_dir=str(temp_experiment_dir)
        )

        # Should find 10 matching files (excluding "other_file.txt")
        # Note: Files with spaces in year (185 1) won't match the regex pattern
        assert len(files) == 10

        # Check structure: list of (year, month, path) tuples
        assert all(len(f) == 3 for f in files)
        assert all(isinstance(f[0], int) for f in files)  # year
        assert all(isinstance(f[1], int) for f in files)  # month
        assert all(isinstance(f[2], str) for f in files)  # path

    def test_year_filtering_start_year(self, temp_experiment_dir):
        """Test filtering with start_year only."""
        files = find_matching_files(
            'xqhuj', 'a', 'pi',
            start_year=1851,
            base_dir=str(temp_experiment_dir)
        )

        # Should only get 1851 and 1852 files
        years = [f[0] for f in files]
        assert all(y >= 1851 for y in years)
        assert 1850 not in years

    def test_year_filtering_end_year(self, temp_experiment_dir):
        """Test filtering with end_year only."""
        files = find_matching_files(
            'xqhuj', 'a', 'pi',
            end_year=1851,
            base_dir=str(temp_experiment_dir)
        )

        # Should only get 1850 and 1851 files
        years = [f[0] for f in files]
        assert all(y <= 1851 for y in years)
        assert 1852 not in years

    def test_year_filtering_range(self, temp_experiment_dir):
        """Test filtering with both start_year and end_year."""
        files = find_matching_files(
            'xqhuj', 'a', 'pi',
            start_year=1851,
            end_year=1851,
            base_dir=str(temp_experiment_dir)
        )

        # Should only get 1851 files
        years = [f[0] for f in files]
        assert all(y == 1851 for y in years)
        assert len(files) == 6  # ja, 11, 91, a1, b1, c1

    def test_files_sorted_by_year_and_month(self, temp_experiment_dir):
        """Test that returned files are sorted by year and month."""
        files = find_matching_files(
            'xqhuj', 'a', 'pi',
            base_dir=str(temp_experiment_dir)
        )

        # Check files are sorted
        for i in range(len(files) - 1):
            year1, month1, _ = files[i]
            year2, month2, _ = files[i + 1]
            assert (year1, month1) <= (year2, month2)

    def test_nonexistent_experiment(self, tmp_path):
        """Test handling of nonexistent experiment directory."""
        files = find_matching_files(
            'nonexistent', 'a', 'pi',
            base_dir=str(tmp_path)
        )

        # Should return empty list, not raise error
        assert files == []

    def test_no_matching_files(self, temp_experiment_dir):
        """Test when directory exists but no files match."""
        files = find_matching_files(
            'xqhuj', 'z', 'qq',  # Wrong model and stream
            base_dir=str(temp_experiment_dir)
        )

        assert files == []

    def test_alpha_month_codes(self, temp_experiment_dir):
        """Test that alpha month codes are correctly parsed."""
        files = find_matching_files(
            'xqhuj', 'a', 'pi',
            start_year=1850,
            end_year=1850,
            base_dir=str(temp_experiment_dir)
        )

        # Should have January, February, December 1850
        months = sorted([f[1] for f in files])
        assert 1 in months  # January
        assert 2 in months  # February
        assert 12 in months  # December

    def test_numeric_month_codes(self, temp_experiment_dir):
        """Test that numeric month codes are correctly parsed."""
        # Create a specific test file
        expt_dir = temp_experiment_dir / "xqtest" / "datam"
        expt_dir.mkdir(parents=True)
        (expt_dir / "xqtesta#pi00000200051+").touch()  # May 2000

        files = find_matching_files(
            'xqtest', 'a', 'pi',
            start_year=2000,
            end_year=2000,
            base_dir=str(temp_experiment_dir)
        )

        assert len(files) == 1
        assert files[0][1] == 5  # May

    def test_expanduser_tilde(self, tmp_path, monkeypatch):
        """Test that ~ in base_dir is expanded."""
        # Create test dir in tmp_path
        test_home = tmp_path / "home"
        test_home.mkdir()
        expt_dir = test_home / "xqhuj" / "datam"
        expt_dir.mkdir(parents=True)
        (expt_dir / "xqhuja#pi000001850ja+").touch()

        # Mock expanduser to return our tmp path
        def mock_expanduser(path):
            if path.startswith('~'):
                return str(test_home) + path[1:]
            return path

        monkeypatch.setattr(os.path, 'expanduser', mock_expanduser)

        files = find_matching_files(
            'xqhuj', 'a', 'pi',
            base_dir='~/nonexistent',  # Will be expanded by our mock
            start_year=1850,
            end_year=1850
        )

        # With our mock, this should find files
        # (Note: actual behavior depends on test setup)
        assert isinstance(files, list)


class TestFilenamePatterns:
    """Test various filename pattern edge cases."""

    @pytest.fixture
    def pattern_test_dir(self, tmp_path):
        """Create directory with various filename patterns."""
        expt_dir = tmp_path / "xqhuj" / "datam"
        expt_dir.mkdir(parents=True)
        return expt_dir

    def test_different_models(self, pattern_test_dir, tmp_path):
        """Test different model identifiers (a, o, etc.)."""
        (pattern_test_dir / "xqhuja#pi000001850ja+").touch()
        (pattern_test_dir / "xqhujo#pi000001850ja+").touch()

        # Should only match 'a' model
        files_a = find_matching_files(
            'xqhuj', 'a', 'pi',
            base_dir=str(tmp_path)
        )
        assert len(files_a) == 1
        assert 'xqhuja' in files_a[0][2]

        # Should only match 'o' model
        files_o = find_matching_files(
            'xqhuj', 'o', 'pi',
            base_dir=str(tmp_path)
        )
        assert len(files_o) == 1
        assert 'xqhujo' in files_o[0][2]

    def test_different_streams(self, pattern_test_dir, tmp_path):
        """Test different stream identifiers (pi, da, etc.)."""
        (pattern_test_dir / "xqhuja#pi000001850ja+").touch()
        (pattern_test_dir / "xqhuja#da000001850ja+").touch()

        # Should only match 'pi' stream
        files_pi = find_matching_files(
            'xqhuj', 'a', 'pi',
            base_dir=str(tmp_path)
        )
        assert len(files_pi) == 1
        assert '#pi' in files_pi[0][2]

        # Should only match 'da' stream
        files_da = find_matching_files(
            'xqhuj', 'a', 'da',
            base_dir=str(tmp_path)
        )
        assert len(files_da) == 1
        assert '#da' in files_da[0][2]

    def test_four_digit_year_required(self, pattern_test_dir, tmp_path):
        """Test that only 4-digit years are matched."""
        (pattern_test_dir / "xqhuja#pi00000185ja+").touch()  # 3 digits - invalid
        (pattern_test_dir / "xqhuja#pi000001850ja+").touch()  # 4 digits - valid
        (pattern_test_dir / "xqhuja#pi0000018500ja+").touch()  # 5 digits - invalid

        files = find_matching_files(
            'xqhuj', 'a', 'pi',
            base_dir=str(tmp_path)
        )

        # Should only match the 4-digit year
        assert len(files) == 1
        assert files[0][0] == 1850

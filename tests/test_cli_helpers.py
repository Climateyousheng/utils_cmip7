"""
Tests for CLI helper functions.
"""

import pytest
from utils_cmip7.cli import _extract_ensemble_prefix


class TestExtractEnsemblePrefix:
    """Tests for _extract_ensemble_prefix helper function."""

    def test_five_character_id(self):
        """Test extraction from 5-character experiment ID."""
        assert _extract_ensemble_prefix('xqjca') == 'xqjc'
        assert _extract_ensemble_prefix('xqjcb') == 'xqjc'
        assert _extract_ensemble_prefix('xqhuc') == 'xqhu'

    def test_four_character_id(self):
        """Test that 4-character IDs are returned as-is."""
        assert _extract_ensemble_prefix('xqjc') == 'xqjc'
        assert _extract_ensemble_prefix('test') == 'test'

    def test_short_id(self):
        """Test that short IDs are returned as-is."""
        assert _extract_ensemble_prefix('abc') == 'abc'
        assert _extract_ensemble_prefix('ab') == 'ab'
        assert _extract_ensemble_prefix('a') == 'a'

    def test_long_id(self):
        """Test that long IDs (>5 chars) are returned as-is."""
        assert _extract_ensemble_prefix('xqjcabc') == 'xqjcabc'
        assert _extract_ensemble_prefix('longname123') == 'longname123'

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty string
        assert _extract_ensemble_prefix('') == ''

        # Exactly 5 characters
        assert _extract_ensemble_prefix('abcde') == 'abcd'

        # 6 characters (not extracted)
        assert _extract_ensemble_prefix('abcdef') == 'abcdef'

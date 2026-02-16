"""
Tests for extract-raw --validate overview table population.

Tests the new functionality that saves metrics CSV and populates
the overview table when using extract-raw --validate.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from utils_cmip7.cli import _format_metrics_for_csv


class TestFormatMetricsForCSV:
    """Test the _format_metrics_for_csv helper function."""

    def test_basic_formatting(self):
        """Test basic metrics formatting."""
        um_metrics = {
            'GPP': {
                'global': {
                    'data': np.array([120.0, 122.0, 121.0]),
                    'units': 'PgC/yr'
                }
            },
            'NPP': {
                'global': {
                    'data': np.array([60.0, 61.0, 60.5]),
                    'units': 'PgC/yr'
                }
            }
        }

        df = _format_metrics_for_csv(um_metrics)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == 'metric'
        assert 'global' in df.columns

        # Check values
        assert 'GPP' in df.index
        assert 'NPP' in df.index
        assert abs(df.loc['GPP', 'global'] - 121.0) < 0.1
        assert abs(df.loc['NPP', 'global'] - 60.5) < 0.1

    def test_all_carbon_metrics(self):
        """Test with all standard carbon metrics."""
        um_metrics = {
            'GPP': {'global': {'data': np.array([120.5]), 'units': 'PgC/yr'}},
            'NPP': {'global': {'data': np.array([60.2]), 'units': 'PgC/yr'}},
            'CVeg': {'global': {'data': np.array([550.3]), 'units': 'PgC'}},
            'CSoil': {'global': {'data': np.array([1450.7]), 'units': 'PgC'}},
            'Tau': {'global': {'data': np.array([24.1]), 'units': 'yr'}}
        }

        df = _format_metrics_for_csv(um_metrics)

        # Check all metrics present
        assert len(df) == 5
        assert set(df.index) == {'GPP', 'NPP', 'CVeg', 'CSoil', 'Tau'}

        # Check values are means
        assert abs(df.loc['GPP', 'global'] - 120.5) < 0.01
        assert abs(df.loc['CVeg', 'global'] - 550.3) < 0.01

    def test_time_averaging(self):
        """Test that time dimension is properly averaged."""
        um_metrics = {
            'GPP': {
                'global': {
                    'data': np.array([100.0, 110.0, 120.0, 130.0, 140.0]),
                    'units': 'PgC/yr'
                }
            }
        }

        df = _format_metrics_for_csv(um_metrics)

        # Mean should be 120.0
        assert abs(df.loc['GPP', 'global'] - 120.0) < 0.01

    def test_empty_metrics(self):
        """Test with empty metrics dict."""
        um_metrics = {}

        df = _format_metrics_for_csv(um_metrics)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_single_metric(self):
        """Test with single metric."""
        um_metrics = {
            'GPP': {
                'global': {
                    'data': np.array([125.5, 124.5]),
                    'units': 'PgC/yr'
                }
            }
        }

        df = _format_metrics_for_csv(um_metrics)

        assert len(df) == 1
        assert df.loc['GPP', 'global'] == 125.0


class TestOverviewTableIntegration:
    """Integration tests for overview table population."""

    def test_scores_extraction_from_metrics(self):
        """Test extracting scores dict from um_metrics for overview table."""
        um_metrics = {
            'GPP': {'global': {'data': np.array([120.5, 121.0, 119.5]), 'units': 'PgC/yr'}},
            'NPP': {'global': {'data': np.array([60.0, 61.0, 59.5]), 'units': 'PgC/yr'}},
            'CVeg': {'global': {'data': np.array([550.0, 551.0, 549.0]), 'units': 'PgC'}},
            'CSoil': {'global': {'data': np.array([1450.0, 1451.0, 1449.0]), 'units': 'PgC'}},
            'Tau': {'global': {'data': np.array([24.0, 24.1, 23.9]), 'units': 'yr'}}
        }

        # Extract scores as done in cli.py
        scores = {}
        for metric in ['GPP', 'NPP', 'CVeg', 'CSoil']:
            if metric in um_metrics and 'global' in um_metrics[metric]:
                scores[metric] = np.mean(um_metrics[metric]['global']['data'])

        # Verify
        assert len(scores) == 4
        assert 'GPP' in scores
        assert 'NPP' in scores
        assert 'CVeg' in scores
        assert 'CSoil' in scores
        assert 'Tau' not in scores  # Tau excluded from overview table

        # Check values
        assert abs(scores['GPP'] - 120.333) < 0.01
        assert abs(scores['NPP'] - 60.166) < 0.01

    def test_scores_with_missing_metrics(self):
        """Test scores extraction when some metrics are missing."""
        um_metrics = {
            'GPP': {'global': {'data': np.array([120.0]), 'units': 'PgC/yr'}},
            'NPP': {'global': {'data': np.array([60.0]), 'units': 'PgC/yr'}}
            # CVeg and CSoil missing
        }

        scores = {}
        for metric in ['GPP', 'NPP', 'CVeg', 'CSoil']:
            if metric in um_metrics and 'global' in um_metrics[metric]:
                scores[metric] = np.mean(um_metrics[metric]['global']['data'])

        # Only GPP and NPP should be in scores
        assert len(scores) == 2
        assert 'GPP' in scores
        assert 'NPP' in scores
        assert 'CVeg' not in scores
        assert 'CSoil' not in scores


class TestCSVFormat:
    """Test CSV output format matches expected structure."""

    def test_csv_index_and_columns(self):
        """Test CSV has correct index and columns."""
        um_metrics = {
            'GPP': {'global': {'data': np.array([120.0]), 'units': 'PgC/yr'}},
            'NPP': {'global': {'data': np.array([60.0]), 'units': 'PgC/yr'}}
        }

        df = _format_metrics_for_csv(um_metrics)

        # Should have 'metric' as index name
        assert df.index.name == 'metric'

        # Should have 'global' column
        assert 'global' in df.columns

        # Index should contain metric names
        assert 'GPP' in df.index
        assert 'NPP' in df.index

    def test_csv_float_precision(self):
        """Test that values maintain precision."""
        um_metrics = {
            'GPP': {
                'global': {
                    'data': np.array([120.123456789]),
                    'units': 'PgC/yr'
                }
            }
        }

        df = _format_metrics_for_csv(um_metrics)

        # Value should be precise
        assert abs(df.loc['GPP', 'global'] - 120.123456789) < 1e-6

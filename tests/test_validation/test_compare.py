"""
Tests for validation/compare.py — metric comparison functions.

All functions are pure computation on synthetic data; no I/O required.
"""

import numpy as np
import pytest

from utils_cmip7.validation.compare import (
    compute_bias,
    compute_rmse,
    compare_single_metric,
    compare_metrics,
    summarize_comparison,
    print_comparison_table,
)


# ---------------------------------------------------------------------------
# compute_bias
# ---------------------------------------------------------------------------

class TestComputeBias:
    """Tests for compute_bias()."""

    def test_basic_bias(self):
        """Absolute and percent bias computed correctly."""
        abs_bias, pct_bias, within = compute_bias(120.0, 123.16)
        assert abs_bias == pytest.approx(-3.16)
        assert pct_bias == pytest.approx(-2.565, rel=1e-2)
        # No obs_error → within_uncertainty is False
        assert within is False

    def test_within_uncertainty(self):
        """Bias within observational uncertainty returns True."""
        abs_bias, pct_bias, within = compute_bias(120.0, 123.16, obs_error=9.61)
        assert within is True

    def test_outside_uncertainty(self):
        """Bias outside observational uncertainty returns False."""
        abs_bias, pct_bias, within = compute_bias(120.0, 123.16, obs_error=1.0)
        assert within is False

    def test_zero_obs_mean(self):
        """obs_mean == 0 → percent bias is NaN."""
        abs_bias, pct_bias, within = compute_bias(5.0, 0.0)
        assert abs_bias == pytest.approx(5.0)
        assert np.isnan(pct_bias)

    def test_exact_match(self):
        """Perfect agreement → zero bias."""
        abs_bias, pct_bias, within = compute_bias(100.0, 100.0, obs_error=0.1)
        assert abs_bias == pytest.approx(0.0)
        assert pct_bias == pytest.approx(0.0)
        assert within is True


# ---------------------------------------------------------------------------
# compute_rmse
# ---------------------------------------------------------------------------

class TestComputeRmse:
    """Tests for compute_rmse()."""

    def test_basic_rmse(self):
        """RMSE for a known time series."""
        um_data = np.array([120.0, 121.0, 122.0])
        rmse = compute_rmse(um_data, 123.16)
        expected = np.sqrt(np.mean((um_data - 123.16) ** 2))
        assert rmse == pytest.approx(expected)

    def test_single_value(self):
        """RMSE with single data point equals absolute difference."""
        rmse = compute_rmse(np.array([10.0]), 12.0)
        assert rmse == pytest.approx(2.0)

    def test_perfect_match(self):
        """Constant series equal to obs → RMSE = 0."""
        rmse = compute_rmse(np.array([5.0, 5.0, 5.0]), 5.0)
        assert rmse == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compare_single_metric
# ---------------------------------------------------------------------------

class TestCompareSingleMetric:
    """Tests for compare_single_metric()."""

    def test_basic_comparison(self):
        """Standard comparison returns expected keys and values."""
        um_data = {
            'years': np.array([1850, 1851, 1852]),
            'data': np.array([120.0, 121.0, 122.0]),
        }
        obs_data = {
            'data': np.array([123.16]),
            'error': np.array([9.61]),
        }
        result = compare_single_metric(um_data, obs_data)

        assert result['um_mean'] == pytest.approx(121.0)
        assert result['obs_mean'] == pytest.approx(123.16)
        assert result['obs_error'] == pytest.approx(9.61)
        assert result['n_years'] == 3
        assert result['within_uncertainty'] == True
        assert 'bias' in result
        assert 'bias_percent' in result
        assert 'rmse' in result
        assert 'um_std' in result

    def test_no_obs_error(self):
        """Missing obs_error → obs_error is None in result."""
        um_data = {'data': np.array([10.0, 11.0])}
        obs_data = {'data': np.array([12.0])}
        result = compare_single_metric(um_data, obs_data)

        assert result['obs_error'] is None
        assert result['within_uncertainty'] is False


# ---------------------------------------------------------------------------
# compare_metrics
# ---------------------------------------------------------------------------

def _make_metric_data(values, obs_value, obs_error=None):
    """Helper to build canonical metric dicts."""
    um = {'data': np.array(values)}
    obs = {'data': np.array([obs_value])}
    if obs_error is not None:
        obs['error'] = np.array([obs_error])
    return um, obs


class TestCompareMetrics:
    """Tests for compare_metrics()."""

    def test_auto_detects_common_metrics_and_regions(self):
        """With no filters, compares intersection of metrics × regions."""
        um = {
            'GPP': {'global': {'data': np.array([120.0])}},
            'NPP': {'global': {'data': np.array([60.0])}},
        }
        obs = {
            'GPP': {'global': {'data': np.array([123.0]), 'error': np.array([10.0])}},
            'Rh':  {'global': {'data': np.array([55.0])}},
        }
        result = compare_metrics(um, obs)

        assert 'GPP' in result
        assert 'NPP' not in result  # Not in obs
        assert 'Rh' not in result   # Not in UM
        assert 'global' in result['GPP']

    def test_filter_by_metrics_and_regions(self):
        """Explicit metrics/regions filters work."""
        um = {
            'GPP': {
                'global': {'data': np.array([120.0])},
                'Europe': {'data': np.array([30.0])},
            },
        }
        obs = {
            'GPP': {
                'global': {'data': np.array([123.0])},
                'Europe': {'data': np.array([32.0])},
            },
        }
        result = compare_metrics(um, obs, metrics=['GPP'], regions=['Europe'])

        assert list(result.keys()) == ['GPP']
        assert list(result['GPP'].keys()) == ['Europe']

    def test_missing_metric_returns_empty(self):
        """Requesting a metric not in data returns empty sub-dict."""
        um = {'GPP': {'global': {'data': np.array([120.0])}}}
        obs = {'GPP': {'global': {'data': np.array([123.0])}}}
        result = compare_metrics(um, obs, metrics=['GPP', 'NPP'])

        assert result['NPP'] == {}

    def test_missing_region_skipped(self):
        """Region not present in both UM and obs is skipped."""
        um = {'GPP': {'global': {'data': np.array([120.0])}}}
        obs = {'GPP': {'Europe': {'data': np.array([32.0])}}}
        result = compare_metrics(um, obs, metrics=['GPP'], regions=['global', 'Europe'])

        assert 'global' not in result['GPP']
        assert 'Europe' not in result['GPP']


# ---------------------------------------------------------------------------
# summarize_comparison
# ---------------------------------------------------------------------------

class TestSummarizeComparison:
    """Tests for summarize_comparison()."""

    @pytest.fixture
    def sample_comparison(self):
        """Build a comparison dict with known values."""
        um = {
            'GPP': {
                'global': {'data': np.array([120.0, 121.0])},
                'Europe': {'data': np.array([30.0, 31.0])},
            },
            'NPP': {
                'global': {'data': np.array([60.0, 61.0])},
                'Europe': {'data': np.array([15.0, 16.0])},
            },
        }
        obs = {
            'GPP': {
                'global': {'data': np.array([123.0]), 'error': np.array([10.0])},
                'Europe': {'data': np.array([32.0]), 'error': np.array([5.0])},
            },
            'NPP': {
                'global': {'data': np.array([62.0]), 'error': np.array([8.0])},
                'Europe': {'data': np.array([17.0]), 'error': np.array([3.0])},
            },
        }
        return compare_metrics(um, obs)

    def test_summarize_single_metric(self, sample_comparison):
        """Summarize one metric across its regions."""
        summary = summarize_comparison(sample_comparison, metric='GPP')

        assert summary['n_comparisons'] == 2
        assert 'mean_bias' in summary
        assert 'mean_rmse' in summary
        assert 'fraction_within_uncertainty' in summary
        assert set(summary['regions']) == {'global', 'Europe'}

    def test_summarize_all(self, sample_comparison):
        """Summarize across all metrics and regions."""
        summary = summarize_comparison(sample_comparison)

        assert summary['n_comparisons'] == 4
        assert 'metrics_regions' in summary
        assert len(summary['metrics_regions']) == 4

    def test_summarize_empty_metric(self, sample_comparison):
        """Summarizing a metric not in comparison → NaN means."""
        summary = summarize_comparison(sample_comparison, metric='Rh')
        assert summary['n_comparisons'] == 0
        assert np.isnan(summary['mean_bias'])


# ---------------------------------------------------------------------------
# print_comparison_table
# ---------------------------------------------------------------------------

class TestPrintComparisonTable:
    """Tests for print_comparison_table()."""

    def test_prints_without_error(self, capsys):
        """Table prints expected header and data rows."""
        um = {'GPP': {'global': {'data': np.array([120.0])}}}
        obs = {'GPP': {'global': {'data': np.array([123.0]), 'error': np.array([10.0])}}}
        comparison = compare_metrics(um, obs)

        print_comparison_table(comparison)

        captured = capsys.readouterr().out
        assert 'Metric' in captured
        assert 'GPP' in captured
        assert 'global' in captured

    def test_filter_metrics_and_regions(self, capsys):
        """Filtering by metrics/regions works."""
        um = {
            'GPP': {'global': {'data': np.array([120.0])}, 'Europe': {'data': np.array([30.0])}},
            'NPP': {'global': {'data': np.array([60.0])}},
        }
        obs = {
            'GPP': {'global': {'data': np.array([123.0])}, 'Europe': {'data': np.array([32.0])}},
            'NPP': {'global': {'data': np.array([62.0])}},
        }
        comparison = compare_metrics(um, obs)

        print_comparison_table(comparison, metrics=['GPP'], regions=['global'])

        captured = capsys.readouterr().out
        assert 'GPP' in captured
        assert 'Europe' not in captured

    def test_no_obs_error_shows_na(self, capsys):
        """When obs_error is None, table shows N/A."""
        um = {'GPP': {'global': {'data': np.array([120.0])}}}
        obs = {'GPP': {'global': {'data': np.array([123.0])}}}
        comparison = compare_metrics(um, obs)

        print_comparison_table(comparison)

        captured = capsys.readouterr().out
        assert 'N/A' in captured

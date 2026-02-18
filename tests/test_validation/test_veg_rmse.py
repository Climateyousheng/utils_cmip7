"""
Tests for vegetation RMSE functions.

Covers:
- compute_spatial_rmse (unweighted, backward-compatible)
- compute_spatial_rmse_weighted (area-weighted)
- rmse_w propagation through calculate_veg_metrics
- shrub inclusion in overview scores
- rmse_w columns in overview scores
"""

import numpy as np
import pytest

from utils_cmip7.validation.veg_fractions import (
    compute_spatial_rmse,
    compute_spatial_rmse_weighted,
    calculate_veg_metrics,
    PFT_MAPPING,
)


# ---------------------------------------------------------------------------
# compute_spatial_rmse (unweighted — backward compatibility)
# ---------------------------------------------------------------------------

class TestComputeSpatialRmseUnweighted:
    def test_zero_for_identical_fields(self):
        field = np.ones((10, 20)) * 0.3
        assert compute_spatial_rmse(field, field) == pytest.approx(0.0)

    def test_known_scalar_bias(self):
        """Uniform +0.1 bias everywhere → RMSE = 0.1."""
        model = np.ones((4, 4)) * 0.5
        obs = np.ones((4, 4)) * 0.4
        assert compute_spatial_rmse(model, obs) == pytest.approx(0.1, rel=1e-6)

    def test_nan_aware(self):
        """NaN cells are ignored in the mean."""
        model = np.array([[0.5, np.nan], [0.5, 0.5]])
        obs = np.array([[0.4, 0.4], [0.4, 0.4]])
        # Only 3 non-NaN cells, all with diff 0.1 → RMSE = 0.1
        assert compute_spatial_rmse(model, obs) == pytest.approx(0.1, rel=1e-6)

    def test_returns_float(self):
        field = np.ones((5, 5))
        result = compute_spatial_rmse(field, field)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_spatial_rmse_weighted (area-weighted)
# ---------------------------------------------------------------------------

class TestComputeSpatialRmseWeighted:
    def test_zero_for_identical_fields(self):
        field = np.ones((10, 20)) * 0.3
        lats = np.linspace(-90, 90, 10)
        assert compute_spatial_rmse_weighted(field, field, lats) == pytest.approx(0.0)

    def test_known_scalar_bias(self):
        """Uniform +0.1 bias → weighted RMSE still = 0.1."""
        model = np.ones((4, 8)) * 0.5
        obs = np.ones((4, 8)) * 0.4
        lats = np.array([-45.0, -15.0, 15.0, 45.0])
        result = compute_spatial_rmse_weighted(model, obs, lats)
        assert result == pytest.approx(0.1, rel=1e-4)

    def test_polar_bias_weighted_less(self):
        """
        Error at high latitude contributes less than same error at equator.
        Build a case: only one row has error, compare polar vs equatorial.
        """
        nlat, nlon = 10, 20
        # All zeros except one row
        model_equator = np.zeros((nlat, nlon))
        model_polar = np.zeros((nlat, nlon))
        obs = np.zeros((nlat, nlon))

        lats = np.linspace(-85, 85, nlat)
        equator_row = np.argmin(np.abs(lats))     # closest to 0°
        polar_row = np.argmin(np.abs(lats - 80))  # closest to 80°N

        model_equator[equator_row, :] = 0.5
        model_polar[polar_row, :] = 0.5

        rmse_equator = compute_spatial_rmse_weighted(model_equator, obs, lats)
        rmse_polar = compute_spatial_rmse_weighted(model_polar, obs, lats)

        assert rmse_equator > rmse_polar, (
            "Equatorial error should produce larger weighted RMSE than polar error"
        )

    def test_nan_aware(self):
        """NaN cells should be excluded from weight and numerator."""
        model = np.array([[0.5, np.nan], [0.5, 0.5]])
        obs = np.array([[0.4, 0.4], [0.4, 0.4]])
        lats = np.array([0.0, 45.0])
        result = compute_spatial_rmse_weighted(model, obs, lats)
        assert np.isfinite(result)
        assert result > 0.0

    def test_returns_float(self):
        field = np.ones((5, 5))
        lats = np.linspace(-80, 80, 5)
        result = compute_spatial_rmse_weighted(field, field, lats)
        assert isinstance(result, float)

    def test_weighted_differs_from_unweighted_asymmetric_error(self):
        """
        For non-uniform error distribution, weighted and unweighted should differ.
        """
        nlat, nlon = 8, 16
        model = np.zeros((nlat, nlon))
        obs = np.zeros((nlat, nlon))
        lats = np.linspace(-70, 70, nlat)

        # Put error only at the highest latitude row
        model[-1, :] = 1.0

        uw = compute_spatial_rmse(model, obs)
        w = compute_spatial_rmse_weighted(model, obs, lats)

        # They should differ because the error is not uniformly distributed
        assert uw != pytest.approx(w, rel=1e-3)


# ---------------------------------------------------------------------------
# calculate_veg_metrics — rmse_w propagation
# ---------------------------------------------------------------------------

def _make_mock_raw_data(expt, pft_ids, include_rmse=True, include_rmse_w=True):
    """Build a minimal raw_data dict as returned by extract_annual_means."""
    frac_data = {}
    for pft_id in pft_ids:
        pft_key = f'PFT {pft_id}'
        entry = {
            'years': np.array([2000, 2001, 2002]),
            'data': np.array([0.2, 0.21, 0.22]),
            'units': 'fraction',
            'name': PFT_MAPPING.get(pft_id, f'PFT{pft_id}'),
            'region': 'global',
        }
        if include_rmse:
            entry['rmse'] = 0.05 + pft_id * 0.01
        if include_rmse_w:
            entry['rmse_w'] = 0.04 + pft_id * 0.01
        frac_data[pft_key] = entry

    return {expt: {'global': {'frac': frac_data}}}


class TestCalculateVegMetricsRmseW:
    def test_rmse_w_stored_for_veg_pfts(self):
        """rmse_w_ keys are propagated for BL, NL, C3, C4, shrub."""
        raw_data = _make_mock_raw_data('xtest', pft_ids=[1, 2, 3, 4, 5])
        metrics = calculate_veg_metrics(raw_data, 'xtest')

        for pft_id, pft_name in [(1, 'BL'), (2, 'NL'), (3, 'C3'), (4, 'C4'), (5, 'shrub')]:
            key = f'rmse_w_{pft_name}'
            assert key in metrics, f"Expected {key} in metrics"
            assert 'global' in metrics[key]
            assert np.isfinite(metrics[key]['global'])

    def test_rmse_stored_for_veg_pfts(self):
        """Unweighted rmse_ keys are still propagated (backward compat)."""
        raw_data = _make_mock_raw_data('xtest', pft_ids=[1, 2, 3, 4, 5])
        metrics = calculate_veg_metrics(raw_data, 'xtest')

        for pft_name in ['BL', 'NL', 'C3', 'C4', 'shrub']:
            key = f'rmse_{pft_name}'
            assert key in metrics, f"Expected {key} in metrics"
            assert 'global' in metrics[key]

    def test_no_rmse_w_when_absent_in_data(self):
        """If rmse_w not stored in raw data, it should not appear in metrics."""
        raw_data = _make_mock_raw_data('xtest', pft_ids=[1], include_rmse=True,
                                       include_rmse_w=False)
        metrics = calculate_veg_metrics(raw_data, 'xtest')
        assert 'rmse_w_BL' not in metrics


# ---------------------------------------------------------------------------
# CLI score dict — shrub and rmse_w columns (unit-level, no iris/xarray needed)
# ---------------------------------------------------------------------------

def _build_veg_metrics_for_scores():
    """Minimal veg_metrics dict as would be produced by calculate_veg_metrics."""
    veg_pfts = ['BL', 'NL', 'C3', 'C4', 'shrub']
    metrics = {}
    for i, pft in enumerate(veg_pfts):
        metrics[f'rmse_{pft}'] = {'global': 0.05 + i * 0.01}
        metrics[f'rmse_w_{pft}'] = {'global': 0.04 + i * 0.01}
    return metrics


def _build_um_metrics_for_scores():
    """Minimal um_metrics dict (PFT time series only)."""
    um = {}
    for pft in ['BL', 'NL', 'C3', 'C4', 'shrub', 'bare_soil']:
        um[pft] = {'global': {'data': np.array([0.2, 0.21, 0.22]), 'units': 'fraction'}}
    return um


def _simulate_score_block(um_metrics, veg_metrics):
    """Mirror the score-writing block from cli.py validate_experiment_cli."""
    scores = {}

    _GM_PFTS = [('BL', 'BL'), ('NL', 'NL'), ('C3', 'C3'), ('C4', 'C4'),
                ('shrub', 'shrub'), ('bare_soil', 'BS')]
    for pft_name, col_suffix in _GM_PFTS:
        gm_col = f'GM_{col_suffix}'
        if pft_name in um_metrics and 'global' in um_metrics[pft_name]:
            scores[gm_col] = float(np.mean(um_metrics[pft_name]['global']['data']))
        else:
            scores[gm_col] = np.nan

    if veg_metrics:
        _VEG_PFTS = [('BL', 'BL'), ('NL', 'NL'), ('C3', 'C3'), ('C4', 'C4'), ('shrub', 'shrub')]
        for pft_name, col_suffix in _VEG_PFTS:
            rmse_key = f'rmse_{pft_name}'
            if rmse_key in veg_metrics and 'global' in veg_metrics[rmse_key]:
                scores[f'rmse_{col_suffix}'] = veg_metrics[rmse_key]['global']
            else:
                scores[f'rmse_{col_suffix}'] = np.nan
            rmse_w_key = f'rmse_w_{pft_name}'
            if rmse_w_key in veg_metrics and 'global' in veg_metrics[rmse_w_key]:
                scores[f'rmse_w_{col_suffix}'] = veg_metrics[rmse_w_key]['global']
            else:
                scores[f'rmse_w_{col_suffix}'] = np.nan

    return scores


class TestOverviewScoreBlock:
    def setup_method(self):
        self.um = _build_um_metrics_for_scores()
        self.vm = _build_veg_metrics_for_scores()
        self.scores = _simulate_score_block(self.um, self.vm)

    def test_gm_shrub_in_scores(self):
        assert 'GM_shrub' in self.scores
        assert np.isfinite(self.scores['GM_shrub'])

    def test_gm_bs_in_scores(self):
        assert 'GM_BS' in self.scores
        assert np.isfinite(self.scores['GM_BS'])

    def test_rmse_bs_not_in_scores(self):
        """Bare soil should not have an RMSE column."""
        assert 'rmse_BS' not in self.scores

    def test_rmse_shrub_in_scores(self):
        assert 'rmse_shrub' in self.scores
        assert np.isfinite(self.scores['rmse_shrub'])

    def test_rmse_w_columns_present(self):
        for pft in ['BL', 'NL', 'C3', 'C4', 'shrub']:
            key = f'rmse_w_{pft}'
            assert key in self.scores, f"Expected {key} in scores"
            assert np.isfinite(self.scores[key])

    def test_rmse_unweighted_columns_present(self):
        for pft in ['BL', 'NL', 'C3', 'C4', 'shrub']:
            key = f'rmse_{pft}'
            assert key in self.scores, f"Expected {key} in scores"

    def test_rmse_w_differs_from_rmse(self):
        """Weighted and unweighted RMSE values should differ (our mock has distinct values)."""
        assert self.scores['rmse_BL'] != self.scores['rmse_w_BL']

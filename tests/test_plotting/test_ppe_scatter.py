"""
Tests for parameter scatter plot functions in utils_cmip7.plotting.ppe_viz.

Covers:
- _preprocess_param_col: sentinel replacement, immutability, non-numeric → NaN
- add_observation_lines: no-op when None/missing, line drawn when present
- plot_param_scatter: returns Figure, correct visible subplot count,
  unused cells hidden, highlight overplot, obs line, raises on all-missing params
- save_overall_skill_param_scatter_pdf: PDF created, TUPP excluded
- save_param_scatter_pdf: PDF created, raises KeyError on missing metric, TUPP excluded
"""

from __future__ import annotations

import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from utils_cmip7.plotting.ppe_viz import (
    _preprocess_param_col,
    add_observation_lines,
    plot_param_scatter,
    save_overall_skill_param_scatter_pdf,
    save_param_scatter_pdf,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df():
    """10-row synthetic DataFrame with parameters, metrics, and highlight column."""
    rng = np.random.default_rng(42)
    n = 10
    df = pd.DataFrame({
        "ID": [f"run{i:02d}" for i in range(n)],
        "overall_score": rng.uniform(0.3, 0.9, n),
        "ALPHA": rng.uniform(0.05, 0.15, n),
        "G_AREA": rng.uniform(0.1, 0.5, n),
        "LAI_MIN": rng.uniform(0.1, 1.0, n),
        "NL0": rng.uniform(0.01, 0.08, n),
        "R_GROW": rng.uniform(0.1, 0.4, n),
        "TLOW": rng.uniform(-5.0, 5.0, n),
        "TUPP": rng.uniform(30.0, 45.0, n),
        "V_CRIT": rng.uniform(0.01, 0.05, n),
        "GPP": rng.uniform(100.0, 160.0, n),
        "NPP": rng.uniform(40.0, 80.0, n),
        "CVeg": rng.uniform(300.0, 700.0, n),
        "CSoil": rng.uniform(1000.0, 2000.0, n),
        "_highlight": [True, False, True] + [False] * 7,
    })
    return df


# ---------------------------------------------------------------------------
# TestPreprocessParamCol
# ---------------------------------------------------------------------------

class TestPreprocessParamCol:
    def test_sentinel_replaced(self):
        s = pd.Series([0.1, -9999, 0.3, -9999])
        result = _preprocess_param_col(s)
        assert result.iloc[1] == pytest.approx(0.5)
        assert result.iloc[3] == pytest.approx(0.5)

    def test_non_sentinel_unchanged(self):
        s = pd.Series([0.1, 0.2, 0.3])
        result = _preprocess_param_col(s)
        assert list(result) == pytest.approx([0.1, 0.2, 0.3])

    def test_non_numeric_becomes_nan(self):
        s = pd.Series(["abc", "0.5", None])
        result = _preprocess_param_col(s)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(0.5)
        assert np.isnan(result.iloc[2])

    def test_immutability(self):
        """Original series must not be mutated."""
        s = pd.Series([0.1, -9999, 0.3])
        original_values = s.copy()
        _preprocess_param_col(s)
        pd.testing.assert_series_equal(s, original_values)

    def test_custom_sentinel_and_replacement(self):
        s = pd.Series([0.1, -1.0, 0.3])
        result = _preprocess_param_col(s, sentinel=-1.0, replacement=0.99)
        assert result.iloc[1] == pytest.approx(0.99)
        assert result.iloc[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# TestAddObservationLines
# ---------------------------------------------------------------------------

class TestAddObservationLines:
    def setup_method(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(0, 200)

    def teardown_method(self):
        plt.close(self.fig)

    def test_noop_when_obs_values_none(self):
        n_lines_before = len(self.ax.lines)
        add_observation_lines(self.ax, "GPP", None)
        assert len(self.ax.lines) == n_lines_before

    def test_noop_when_varname_missing(self):
        n_lines_before = len(self.ax.lines)
        add_observation_lines(self.ax, "GPP", {"NPP": 55.0})
        assert len(self.ax.lines) == n_lines_before

    def test_line_drawn_when_present(self):
        n_lines_before = len(self.ax.lines)
        add_observation_lines(self.ax, "GPP", {"GPP": 120.0})
        assert len(self.ax.lines) == n_lines_before + 1

    def test_line_at_correct_y(self):
        add_observation_lines(self.ax, "GPP", {"GPP": 120.0})
        line = self.ax.lines[-1]
        # axhline produces a line spanning x from 0 to 1 in axes coordinates
        # ydata contains the constant y value
        assert line.get_ydata()[0] == pytest.approx(120.0)

    def test_text_label_present(self):
        n_texts_before = len(self.ax.texts)
        add_observation_lines(self.ax, "GPP", {"GPP": 120.0})
        assert len(self.ax.texts) == n_texts_before + 1
        text_str = self.ax.texts[-1].get_text()
        assert "obs" in text_str
        assert "120" in text_str


# ---------------------------------------------------------------------------
# TestPlotParamScatter
# ---------------------------------------------------------------------------

class TestPlotParamScatter:
    PARAM_COLS = ["ALPHA", "G_AREA", "LAI_MIN", "NL0", "R_GROW", "TLOW", "V_CRIT"]

    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, sample_df):
        fig = plot_param_scatter(sample_df, self.PARAM_COLS, y_col="overall_score")
        assert isinstance(fig, plt.Figure)

    def test_correct_visible_subplots(self, sample_df):
        fig = plot_param_scatter(sample_df, self.PARAM_COLS, y_col="overall_score", ncols=4)
        n_visible = sum(ax.get_visible() for ax in fig.axes)
        assert n_visible == len(self.PARAM_COLS)

    def test_unused_cells_hidden(self, sample_df):
        # 7 params with ncols=4 → 2 rows × 4 cols = 8 cells; 1 must be hidden
        fig = plot_param_scatter(sample_df, self.PARAM_COLS, y_col="overall_score", ncols=4)
        n_hidden = sum(not ax.get_visible() for ax in fig.axes)
        expected_total = 4 * 2  # 2 rows × 4 cols
        assert n_hidden == expected_total - len(self.PARAM_COLS)

    def test_highlight_produces_two_collections(self, sample_df):
        """Highlighted points: 2 PathCollections (base + overplot)."""
        fig = plot_param_scatter(
            sample_df, ["ALPHA"], y_col="overall_score",
            highlight_col="_highlight",
        )
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        ax = visible_axes[0]
        collections = ax.collections
        assert len(collections) == 2

    def test_obs_line_drawn(self, sample_df):
        obs = {"overall_score": 0.5}
        fig = plot_param_scatter(
            sample_df, ["ALPHA"], y_col="overall_score", obs_values=obs
        )
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        ax = visible_axes[0]
        assert len(ax.lines) >= 1

    def test_raises_if_no_param_col_exists(self, sample_df):
        with pytest.raises(ValueError, match="None of param_cols"):
            plot_param_scatter(sample_df, ["NONEXISTENT", "ALSO_MISSING"], y_col="overall_score")

    def test_single_panel(self, sample_df):
        """Single parameter → 1×1 grid, no hidden cells."""
        fig = plot_param_scatter(sample_df, ["ALPHA"], y_col="overall_score", ncols=4)
        n_visible = sum(ax.get_visible() for ax in fig.axes)
        assert n_visible == 1

    def test_title_set(self, sample_df):
        fig = plot_param_scatter(
            sample_df, ["ALPHA"], y_col="overall_score", title="My Title"
        )
        assert fig._suptitle is not None
        assert "My Title" in fig._suptitle.get_text()


# ---------------------------------------------------------------------------
# TestSaveOverallSkillParamScatterPdf
# ---------------------------------------------------------------------------

class TestSaveOverallSkillParamScatterPdf:
    PARAM_COLS_WITH_TUPP = ["ALPHA", "G_AREA", "LAI_MIN", "NL0", "R_GROW", "TLOW", "TUPP", "V_CRIT"]

    def teardown_method(self):
        plt.close("all")

    def test_pdf_created(self, sample_df, tmp_path):
        out = str(tmp_path / "test_overall.pdf")
        save_overall_skill_param_scatter_pdf(
            sample_df, out, param_cols=self.PARAM_COLS_WITH_TUPP
        )
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_tupp_excluded(self, sample_df, tmp_path, monkeypatch):
        """TUPP must be stripped before calling plot_param_scatter."""
        seen_params = []

        original = plot_param_scatter

        def capturing_plot(df, param_cols, y_col, **kwargs):
            seen_params.extend(list(param_cols))
            return original(df, param_cols, y_col, **kwargs)

        import utils_cmip7.plotting.ppe_viz as _module
        monkeypatch.setattr(_module, "plot_param_scatter", capturing_plot)

        out = str(tmp_path / "test_no_tupp.pdf")
        save_overall_skill_param_scatter_pdf(
            sample_df, out, param_cols=self.PARAM_COLS_WITH_TUPP
        )
        assert "TUPP" not in seen_params

    def test_figure_closed_after_save(self, sample_df, tmp_path):
        """No open figures should remain after save."""
        plt.close("all")
        out = str(tmp_path / "test_closed.pdf")
        save_overall_skill_param_scatter_pdf(
            sample_df, out, param_cols=self.PARAM_COLS_WITH_TUPP
        )
        assert len(plt.get_fignums()) == 0


# ---------------------------------------------------------------------------
# TestSaveParamScatterPdf
# ---------------------------------------------------------------------------

class TestSaveParamScatterPdf:
    PARAM_COLS_WITH_TUPP = ["ALPHA", "G_AREA", "LAI_MIN", "NL0", "R_GROW", "TLOW", "TUPP", "V_CRIT"]

    def teardown_method(self):
        plt.close("all")

    def test_pdf_created(self, sample_df, tmp_path):
        out = str(tmp_path / "test_gpp.pdf")
        save_param_scatter_pdf(
            sample_df, metric="GPP", out_pdf=out,
            param_cols=self.PARAM_COLS_WITH_TUPP,
        )
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_raises_on_missing_metric(self, sample_df, tmp_path):
        out = str(tmp_path / "test_bad_metric.pdf")
        with pytest.raises(KeyError, match="NONEXISTENT"):
            save_param_scatter_pdf(
                sample_df, metric="NONEXISTENT", out_pdf=out,
                param_cols=self.PARAM_COLS_WITH_TUPP,
            )

    def test_tupp_excluded(self, sample_df, tmp_path, monkeypatch):
        """TUPP must be stripped before calling plot_param_scatter."""
        seen_params = []

        original = plot_param_scatter

        def capturing_plot(df, param_cols, y_col, **kwargs):
            seen_params.extend(list(param_cols))
            return original(df, param_cols, y_col, **kwargs)

        import utils_cmip7.plotting.ppe_viz as _module
        monkeypatch.setattr(_module, "plot_param_scatter", capturing_plot)

        out = str(tmp_path / "test_no_tupp.pdf")
        save_param_scatter_pdf(
            sample_df, metric="GPP", out_pdf=out,
            param_cols=self.PARAM_COLS_WITH_TUPP,
        )
        assert "TUPP" not in seen_params

    def test_obs_values_passed_through(self, sample_df, tmp_path, monkeypatch):
        """obs_values must be forwarded to plot_param_scatter."""
        received_obs = {}

        original = plot_param_scatter

        def capturing_plot(df, param_cols, y_col, obs_values=None, **kwargs):
            if obs_values is not None:
                received_obs.update(obs_values)
            return original(df, param_cols, y_col, obs_values=obs_values, **kwargs)

        import utils_cmip7.plotting.ppe_viz as _module
        monkeypatch.setattr(_module, "plot_param_scatter", capturing_plot)

        out = str(tmp_path / "test_obs.pdf")
        obs = {"GPP": 123.0}
        save_param_scatter_pdf(
            sample_df, metric="GPP", out_pdf=out,
            param_cols=["ALPHA"], obs_values=obs,
        )
        assert received_obs == obs

    def test_figure_closed_after_save(self, sample_df, tmp_path):
        """No open figures should remain after save."""
        plt.close("all")
        out = str(tmp_path / "test_closed.pdf")
        save_param_scatter_pdf(
            sample_df, metric="GPP", out_pdf=out,
            param_cols=["ALPHA"],
        )
        assert len(plt.get_fignums()) == 0

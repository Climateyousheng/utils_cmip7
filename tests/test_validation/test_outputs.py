"""
Tests for validation/outputs.py — validation bundle writing.
"""

import json
import pytest
import pandas as pd

from utils_cmip7.validation.outputs import write_single_validation_bundle


class MockSoilParamSet:
    """Minimal mock for SoilParamSet."""

    def __init__(self, params=None):
        self._params = params or {"b_exp": [4.0, 5.0], "sathh": [0.1, 0.2]}

    def to_dict(self):
        return self._params


class TestWriteSingleValidationBundle:
    """Tests for write_single_validation_bundle()."""

    def test_creates_bundle_directory(self, tmp_path):
        """Bundle directory is created with expected name."""
        write_single_validation_bundle(
            outdir=tmp_path,
            expt_id="test_001",
            soil_params=MockSoilParamSet(),
        )

        bundle_dir = tmp_path / "single_val_test_001"
        assert bundle_dir.is_dir()

    def test_writes_soil_params_json(self, tmp_path):
        """soil_params.json is created with correct content."""
        params = {"b_exp": [4.0, 5.0], "source": "default"}
        write_single_validation_bundle(
            outdir=tmp_path,
            expt_id="expt_A",
            soil_params=MockSoilParamSet(params),
        )

        json_path = tmp_path / "single_val_expt_A" / "soil_params.json"
        assert json_path.exists()

        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded == params

    def test_writes_scores_csv(self, tmp_path):
        """validation_scores.csv is written when scores are provided."""
        scores = {"GPP_bias": -3.16, "NPP_rmse": 2.5}
        write_single_validation_bundle(
            outdir=tmp_path,
            expt_id="scored",
            soil_params=MockSoilParamSet(),
            scores=scores,
        )

        csv_path = tmp_path / "single_val_scored" / "validation_scores.csv"
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert "GPP_bias" in df.columns
        assert df["GPP_bias"].iloc[0] == pytest.approx(-3.16)

    def test_no_scores_skips_csv(self, tmp_path):
        """Without scores, validation_scores.csv is not created."""
        write_single_validation_bundle(
            outdir=tmp_path,
            expt_id="no_scores",
            soil_params=MockSoilParamSet(),
        )

        csv_path = tmp_path / "single_val_no_scores" / "validation_scores.csv"
        assert not csv_path.exists()

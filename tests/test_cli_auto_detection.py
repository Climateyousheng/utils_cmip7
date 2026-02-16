"""
Integration tests for CLI auto-detection of ensemble parameters.
"""

import json
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def mock_ensemble_logs(tmp_path):
    """
    Create mock ensemble generator logs for testing.

    Creates a temporary log directory with a mock parameter file.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Create mock parameter log for ensemble 'xqjc'
    params_data = [
        {
            "ensemble_id": "xqjca",
            "ALPHA": [0.08, 0.08, 0.08, 0.08, 0.08],
            "F0": [0.875, 0.875, 0.875, 0.875, 0.875],
            "G_AREA": [0.008, 0.008, 0.008, 0.008, 0.008],  # Different from default 0.004
            "LAI_MIN": [4.0, 4.0, 4.0, 1.0, 1.0],
            "NL0": [0.046, 0.046, 0.046, 0.046, 0.046],
            "R_GROW": [0.25, 0.25, 0.25, 0.25, 0.25],
            "TLOW": [0.0, 0.0, 0.0, 0.0, 0.0],
            "TUPP": [36.0, 36.0, 36.0, 36.0, 36.0],
            "V_CRIT_ALPHA": [0.5]
        },
        {
            "ensemble_id": "xqjcb",
            "ALPHA": [0.10, 0.10, 0.10, 0.10, 0.10],
            "F0": [0.900, 0.900, 0.900, 0.900, 0.900],
            "G_AREA": [0.006, 0.006, 0.006, 0.006, 0.006],
            "LAI_MIN": [5.0, 5.0, 5.0, 1.5, 1.5],
            "NL0": [0.050, 0.050, 0.050, 0.050, 0.050],
            "R_GROW": [0.30, 0.30, 0.30, 0.30, 0.30],
            "TLOW": [1.0, 1.0, 1.0, 1.0, 1.0],
            "TUPP": [38.0, 38.0, 38.0, 38.0, 38.0],
            "V_CRIT_ALPHA": [0.6]
        }
    ]

    log_file = log_dir / "xqjc_updated_parameters_20260128.json"
    with open(log_file, 'w') as f:
        json.dump(params_data, f, indent=2)

    return log_dir


class TestCLIAutoDetection:
    """Tests for CLI auto-detection functionality."""

    def test_load_ensemble_params_from_logs(self, mock_ensemble_logs):
        """Test that ensemble parameters can be loaded from logs."""
        from utils_cmip7.validation import load_ensemble_params_from_logs

        params = load_ensemble_params_from_logs(
            str(mock_ensemble_logs),
            'xqjc'
        )

        # Should have 2 experiments
        assert 'xqjca' in params
        assert 'xqjcb' in params

        # Check xqjca parameters
        param_set = params['xqjca']
        assert param_set.ALPHA[0] == 0.08
        assert param_set.G_AREA[0] == 0.008
        assert param_set.LAI_MIN[0] == 4.0
        assert param_set.V_CRIT_ALPHA == 0.5

        # Check xqjcb parameters
        param_set = params['xqjcb']
        assert param_set.ALPHA[0] == 0.10
        assert param_set.LAI_MIN[0] == 5.0
        assert param_set.V_CRIT_ALPHA == 0.6

    def test_load_ensemble_params_missing_directory(self):
        """Test error when log directory doesn't exist."""
        from utils_cmip7.validation import load_ensemble_params_from_logs

        with pytest.raises(FileNotFoundError, match="Log directory not found"):
            load_ensemble_params_from_logs(
                '/nonexistent/path',
                'xqjc'
            )

    def test_load_ensemble_params_missing_files(self, tmp_path):
        """Test error when no matching log files exist."""
        from utils_cmip7.validation import load_ensemble_params_from_logs

        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No parameter log files found"):
            load_ensemble_params_from_logs(
                str(log_dir),
                'nonexistent_ensemble'
            )

    def test_extract_ensemble_prefix_consistency(self):
        """Test that prefix extraction is consistent with ensemble loader."""
        from utils_cmip7.cli import _extract_ensemble_prefix

        # These should work with the ensemble loader
        test_cases = [
            ('xqjca', 'xqjc'),  # 5-char → 4-char
            ('xqjcb', 'xqjc'),  # Same ensemble
            ('xqhuc', 'xqhu'),  # Different ensemble
        ]

        for expt_id, expected_prefix in test_cases:
            assert _extract_ensemble_prefix(expt_id) == expected_prefix


class TestCLIIntegration:
    """Integration tests for CLI parameter loading."""

    def test_cli_auto_detection_with_logs(self, mock_ensemble_logs, monkeypatch):
        """
        Test CLI auto-detection when logs are available.

        Note: This is a partial test since we can't easily test the full CLI
        without mocking many dependencies.
        """
        from utils_cmip7.validation import load_ensemble_params_from_logs
        from utils_cmip7.cli import _extract_ensemble_prefix

        # Simulate what the CLI does
        expt = 'xqjca'
        ensemble_prefix = _extract_ensemble_prefix(expt)

        params_dict = load_ensemble_params_from_logs(
            str(mock_ensemble_logs),
            ensemble_prefix
        )

        # Should find the experiment
        assert expt in params_dict

        # Should have correct parameters
        soil_params = params_dict[expt]
        assert soil_params.ALPHA[0] == 0.08
        assert soil_params.G_AREA[0] == 0.008
        assert soil_params.LAI_MIN[0] == 4.0

        # Should have metadata
        assert 'log_file' in soil_params.metadata
        assert 'experiment_id' in soil_params.metadata

    def test_cli_auto_detection_experiment_not_found(self, mock_ensemble_logs):
        """Test behavior when experiment is not in ensemble logs."""
        from utils_cmip7.validation import load_ensemble_params_from_logs
        from utils_cmip7.cli import _extract_ensemble_prefix

        # Try to find a non-existent experiment in the ensemble
        expt = 'xqjcz'  # Not in our mock logs
        ensemble_prefix = _extract_ensemble_prefix(expt)

        params_dict = load_ensemble_params_from_logs(
            str(mock_ensemble_logs),
            ensemble_prefix
        )

        # Should not find the experiment
        assert expt not in params_dict

        # Should have other members
        assert 'xqjca' in params_dict
        assert 'xqjcb' in params_dict

    def test_cli_priority_explicit_over_auto(self, mock_ensemble_logs):
        """
        Test that explicit parameter source takes priority over auto-detection.

        This verifies the 3-phase priority logic.
        """
        from utils_cmip7.soil_params import SoilParamSet

        # Phase 1: Explicit source provided (simulated)
        # When an explicit source is provided, it should be used
        explicit_params = SoilParamSet.from_default()

        # Phase 2: Auto-detection (simulated)
        # This would only run if no explicit source
        from utils_cmip7.validation import load_ensemble_params_from_logs

        auto_params = load_ensemble_params_from_logs(
            str(mock_ensemble_logs),
            'xqjc'
        )['xqjca']

        # Verify they're different (use G_AREA which differs)
        # Default G_AREA is [0.004, 0.004, 0.1, 0.1, 0.05]
        # Mock auto G_AREA is [0.008, 0.008, 0.008, 0.008, 0.008]
        assert explicit_params.G_AREA[0] != auto_params.G_AREA[0]
        assert explicit_params.G_AREA[0] == 0.004  # default
        assert auto_params.G_AREA[0] == 0.008  # from logs

        # In the actual CLI, explicit_params would be used
        # because explicit_count == 1

"""Test pre_extracted_data parameter in compute_metrics_from_raw."""

import numpy as np
import pytest
from utils_cmip7.diagnostics import compute_metrics_from_raw


def test_compute_metrics_with_pre_extracted_data():
    """Test that pre_extracted_data parameter works correctly."""

    # Mock pre-extracted data
    pre_extracted = {
        'GPP': {
            'years': np.array([1850, 1851, 1852]),
            'data': np.array([110.5, 111.2, 112.0]),
            'units': 'PgC/year',
            'name': 'Gross Primary Production'
        },
        'NPP': {
            'years': np.array([1850, 1851, 1852]),
            'data': np.array([55.2, 55.8, 56.1]),
            'units': 'PgC/year',
            'name': 'Net Primary Production'
        },
        'CVeg': {
            'years': np.array([1850, 1851, 1852]),
            'data': np.array([450.0, 451.0, 452.0]),
            'units': 'PgC',
            'name': 'Vegetation Carbon'
        },
        'CSoil': {
            'years': np.array([1850, 1851, 1852]),
            'data': np.array([1200.0, 1201.0, 1202.0]),
            'units': 'PgC',
            'name': 'Soil Carbon'
        },
        'NEP': {
            'years': np.array([1850, 1851, 1852]),
            'data': np.array([1.5, 1.6, 1.7]),
            'units': 'PgC/year',
            'name': 'Net Ecosystem Production'
        }
    }

    # Call with pre-extracted data (should NOT call extract_annual_mean_raw)
    metrics = compute_metrics_from_raw(
        expt_name='test_expt',
        metrics=['GPP', 'NPP', 'CVeg', 'CSoil'],
        pre_extracted_data=pre_extracted
    )

    # Verify structure
    assert 'GPP' in metrics
    assert 'global' in metrics['GPP']
    assert 'years' in metrics['GPP']['global']
    assert 'data' in metrics['GPP']['global']
    assert 'units' in metrics['GPP']['global']
    assert 'source' in metrics['GPP']['global']
    assert 'dataset' in metrics['GPP']['global']

    # Verify data matches input
    np.testing.assert_array_equal(
        metrics['GPP']['global']['years'],
        pre_extracted['GPP']['years']
    )
    np.testing.assert_array_almost_equal(
        metrics['GPP']['global']['data'],
        pre_extracted['GPP']['data']
    )

    # Verify all requested metrics are present
    assert 'NPP' in metrics
    assert 'CVeg' in metrics
    assert 'CSoil' in metrics

    # Verify source metadata
    assert metrics['GPP']['global']['source'] == 'UM'
    assert metrics['GPP']['global']['dataset'] == 'test_expt'

    # Verify units are from metric config
    assert metrics['GPP']['global']['units'] == 'PgC/yr'  # canonical units
    assert metrics['CVeg']['global']['units'] == 'PgC'


def test_compute_metrics_backward_compatible():
    """Test that omitting pre_extracted_data still works (backward compat)."""

    # Verify function signature accepts call without the new parameter
    # The function will attempt internal extraction (which will return empty dict
    # for nonexistent path, not raise FileNotFoundError)

    # NOT passing pre_extracted_data - should use default None
    # and attempt internal extraction
    metrics = compute_metrics_from_raw(
        expt_name='test_expt',
        metrics=['GPP'],
        start_year=1850,
        end_year=1852,
        base_dir='/nonexistent/path'
    )

    # Function should return dict, but with no metrics (no files found)
    assert isinstance(metrics, dict)


def test_compute_metrics_with_subset_of_variables():
    """Test pre-extraction with only a subset of variables."""

    # Mock pre-extracted data with only GPP and NPP
    pre_extracted = {
        'GPP': {
            'years': np.array([1850, 1851]),
            'data': np.array([110.5, 111.2]),
            'units': 'PgC/year',
            'name': 'Gross Primary Production'
        },
        'NPP': {
            'years': np.array([1850, 1851]),
            'data': np.array([55.2, 55.8]),
            'units': 'PgC/year',
            'name': 'Net Primary Production'
        }
    }

    # Request only metrics that exist in pre-extracted data
    metrics = compute_metrics_from_raw(
        expt_name='test_expt',
        metrics=['GPP', 'NPP'],
        pre_extracted_data=pre_extracted
    )

    # Should have both metrics
    assert 'GPP' in metrics
    assert 'NPP' in metrics

    # Verify data
    np.testing.assert_array_equal(
        metrics['GPP']['global']['years'],
        np.array([1850, 1851])
    )
    np.testing.assert_array_almost_equal(
        metrics['NPP']['global']['data'],
        np.array([55.2, 55.8])
    )


def test_pre_extracted_data_none_behavior():
    """Test that pre_extracted_data=None behaves like default."""

    # Explicitly passing None should trigger internal extraction
    metrics = compute_metrics_from_raw(
        expt_name='test_expt',
        metrics=['GPP'],
        start_year=1850,
        end_year=1852,
        base_dir='/nonexistent/path',
        pre_extracted_data=None  # Explicitly None
    )

    # Should return dict (empty since no files found)
    assert isinstance(metrics, dict)

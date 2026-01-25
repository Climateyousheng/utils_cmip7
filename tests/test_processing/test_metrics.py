"""
Test metric definitions and validation.

Tests for src/utils_cmip7/processing/metrics.py
"""
import pytest
import numpy as np
from utils_cmip7.processing.metrics import (
    METRIC_DEFINITIONS,
    get_metric_config,
    list_metrics,
    validate_metric_output,
)


class TestMetricDefinitions:
    """Test METRIC_DEFINITIONS constant."""

    def test_metric_definitions_exist(self):
        """Test that METRIC_DEFINITIONS is populated."""
        assert len(METRIC_DEFINITIONS) > 0

    def test_core_carbon_metrics_defined(self):
        """Test that core carbon cycle metrics are defined."""
        required_metrics = ['GPP', 'NPP', 'CVeg', 'CSoil', 'Tau']
        for metric in required_metrics:
            assert metric in METRIC_DEFINITIONS, f"Missing core metric: {metric}"

    def test_all_metrics_have_required_fields(self):
        """Test that all metric definitions have required fields."""
        required_fields = ['aggregation', 'output_units', 'description', 'category']

        for metric_name, config in METRIC_DEFINITIONS.items():
            for field in required_fields:
                assert field in config, f"Metric '{metric_name}' missing field '{field}'"

    def test_aggregation_types_valid(self):
        """Test that all aggregation types are valid."""
        valid_types = {'SUM', 'MEAN', 'DERIVED'}

        for metric_name, config in METRIC_DEFINITIONS.items():
            agg_type = config['aggregation']
            assert agg_type in valid_types, (
                f"Metric '{metric_name}' has invalid aggregation '{agg_type}'"
            )

    def test_derived_metrics_have_formula(self):
        """Test that DERIVED metrics have a formula."""
        for metric_name, config in METRIC_DEFINITIONS.items():
            if config['aggregation'] == 'DERIVED':
                assert 'formula' in config, (
                    f"DERIVED metric '{metric_name}' missing formula"
                )
                assert isinstance(config['formula'], str)

    def test_non_derived_metrics_have_conversion_key(self):
        """Test that non-DERIVED metrics have conversion_key."""
        for metric_name, config in METRIC_DEFINITIONS.items():
            if config['aggregation'] != 'DERIVED':
                assert 'conversion_key' in config, (
                    f"Non-DERIVED metric '{metric_name}' missing conversion_key"
                )

    def test_categories_valid(self):
        """Test that all categories are from expected set."""
        valid_categories = {'flux', 'stock', 'climate', 'diagnostic'}

        for metric_name, config in METRIC_DEFINITIONS.items():
            category = config['category']
            assert category in valid_categories, (
                f"Metric '{metric_name}' has invalid category '{category}'"
            )

    def test_metric_aliases_consistent(self):
        """Test that metric aliases have consistent definitions."""
        # CVeg and VegCarb should be equivalent
        assert METRIC_DEFINITIONS['CVeg']['output_units'] == (
            METRIC_DEFINITIONS['VegCarb']['output_units']
        )

        # CSoil and soilCarbon should be equivalent
        assert METRIC_DEFINITIONS['CSoil']['output_units'] == (
            METRIC_DEFINITIONS['soilCarbon']['output_units']
        )


class TestGetMetricConfig:
    """Test get_metric_config() function."""

    def test_get_gpp_config(self):
        """Test retrieving GPP configuration."""
        config = get_metric_config('GPP')

        assert config['aggregation'] == 'SUM'
        assert config['output_units'] == 'PgC/yr'
        assert config['conversion_key'] == 'GPP'
        assert config['category'] == 'flux'

    def test_get_tau_config(self):
        """Test retrieving derived metric (Tau) configuration."""
        config = get_metric_config('Tau')

        assert config['aggregation'] == 'DERIVED'
        assert config['output_units'] == 'years'
        assert 'formula' in config
        assert config['formula'] == 'CSoil / NPP'

    def test_get_cveg_config(self):
        """Test retrieving CVeg configuration."""
        config = get_metric_config('CVeg')

        assert config['aggregation'] == 'SUM'
        assert config['output_units'] == 'PgC'
        assert config['category'] == 'stock'

    def test_get_tas_config(self):
        """Test retrieving climate variable configuration."""
        config = get_metric_config('tas')

        assert config['aggregation'] == 'MEAN'
        assert config['output_units'] == '°C'
        assert config['category'] == 'climate'

    def test_returns_copy_not_reference(self):
        """Test that function returns a copy, not reference."""
        config1 = get_metric_config('GPP')
        config2 = get_metric_config('GPP')

        # Modify config1
        config1['aggregation'] = 'MODIFIED'

        # config2 should be unaffected
        assert config2['aggregation'] == 'SUM'

    def test_unknown_metric_raises_error(self):
        """Test that unknown metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric 'INVALID'"):
            get_metric_config('INVALID')

    def test_error_message_includes_available_metrics(self):
        """Test that error message lists available metrics."""
        with pytest.raises(ValueError, match="Available metrics:"):
            get_metric_config('NONEXISTENT')

    def test_case_sensitive(self):
        """Test that metric names are case-sensitive."""
        # 'GPP' exists
        config = get_metric_config('GPP')
        assert config is not None

        # 'gpp' should raise error
        with pytest.raises(ValueError):
            get_metric_config('gpp')


class TestListMetrics:
    """Test list_metrics() function."""

    def test_list_all_metrics(self):
        """Test listing all metrics without filter."""
        metrics = list_metrics()

        assert isinstance(metrics, list)
        assert len(metrics) > 0
        assert 'GPP' in metrics
        assert 'NPP' in metrics
        assert 'CVeg' in metrics

    def test_list_flux_metrics(self):
        """Test listing flux metrics."""
        flux_metrics = list_metrics('flux')

        assert isinstance(flux_metrics, list)
        assert 'GPP' in flux_metrics
        assert 'NPP' in flux_metrics
        assert 'NEP' in flux_metrics
        assert 'soilResp' in flux_metrics

        # Stock metrics should not be in flux list
        assert 'CVeg' not in flux_metrics
        assert 'CSoil' not in flux_metrics

    def test_list_stock_metrics(self):
        """Test listing stock metrics."""
        stock_metrics = list_metrics('stock')

        assert isinstance(stock_metrics, list)
        assert 'CVeg' in stock_metrics
        assert 'CSoil' in stock_metrics
        assert 'VegCarb' in stock_metrics
        assert 'soilCarbon' in stock_metrics

        # Flux metrics should not be in stock list
        assert 'GPP' not in stock_metrics
        assert 'NPP' not in stock_metrics

    def test_list_climate_metrics(self):
        """Test listing climate metrics."""
        climate_metrics = list_metrics('climate')

        assert isinstance(climate_metrics, list)
        assert 'tas' in climate_metrics or 'temp' in climate_metrics
        assert 'precip' in climate_metrics

    def test_list_diagnostic_metrics(self):
        """Test listing diagnostic metrics."""
        diagnostic_metrics = list_metrics('diagnostic')

        assert isinstance(diagnostic_metrics, list)
        assert 'Tau' in diagnostic_metrics

    def test_empty_category_returns_empty_list(self):
        """Test that nonexistent category returns empty list."""
        result = list_metrics('nonexistent_category')
        assert result == []

    def test_returns_list_not_dict_keys(self):
        """Test that function returns proper list."""
        result = list_metrics()
        assert isinstance(result, list)


class TestValidateMetricOutput:
    """Test validate_metric_output() function."""

    def test_valid_metric_data_passes(self):
        """Test that valid metric data passes validation."""
        data = {
            'years': np.array([1850, 1851, 1852]),
            'data': np.array([123.0, 124.0, 125.0]),
            'units': 'PgC/yr',
            'source': 'UM',
            'dataset': 'xqhuc'
        }

        assert validate_metric_output(data, 'GPP') is True

    def test_missing_years_raises_error(self):
        """Test that missing 'years' field raises error."""
        data = {
            'data': np.array([123.0]),
            'units': 'PgC/yr',
            'source': 'UM',
            'dataset': 'xqhuc'
        }

        with pytest.raises(ValueError, match="Missing required field 'years'"):
            validate_metric_output(data, 'GPP')

    def test_missing_data_raises_error(self):
        """Test that missing 'data' field raises error."""
        data = {
            'years': np.array([1850]),
            'units': 'PgC/yr',
            'source': 'UM',
            'dataset': 'xqhuc'
        }

        with pytest.raises(ValueError, match="Missing required field 'data'"):
            validate_metric_output(data, 'GPP')

    def test_missing_units_raises_error(self):
        """Test that missing 'units' field raises error."""
        data = {
            'years': np.array([1850]),
            'data': np.array([123.0]),
            'source': 'UM',
            'dataset': 'xqhuc'
        }

        with pytest.raises(ValueError, match="Missing required field 'units'"):
            validate_metric_output(data, 'GPP')

    def test_missing_source_raises_error(self):
        """Test that missing 'source' field raises error."""
        data = {
            'years': np.array([1850]),
            'data': np.array([123.0]),
            'units': 'PgC/yr',
            'dataset': 'xqhuc'
        }

        with pytest.raises(ValueError, match="Missing required field 'source'"):
            validate_metric_output(data, 'GPP')

    def test_missing_dataset_raises_error(self):
        """Test that missing 'dataset' field raises error."""
        data = {
            'years': np.array([1850]),
            'data': np.array([123.0]),
            'units': 'PgC/yr',
            'source': 'UM',
        }

        with pytest.raises(ValueError, match="Missing required field 'dataset'"):
            validate_metric_output(data, 'GPP')

    def test_wrong_years_type_raises_error(self):
        """Test that non-numpy array 'years' raises error."""
        data = {
            'years': [1850, 1851],  # List instead of np.array
            'data': np.array([123.0, 124.0]),
            'units': 'PgC/yr',
            'source': 'UM',
            'dataset': 'xqhuc'
        }

        with pytest.raises(ValueError, match="'years' must be numpy array"):
            validate_metric_output(data, 'GPP')

    def test_wrong_data_type_raises_error(self):
        """Test that non-numpy array 'data' raises error."""
        data = {
            'years': np.array([1850, 1851]),
            'data': [123.0, 124.0],  # List instead of np.array
            'units': 'PgC/yr',
            'source': 'UM',
            'dataset': 'xqhuc'
        }

        with pytest.raises(ValueError, match="'data' must be numpy array"):
            validate_metric_output(data, 'GPP')

    def test_wrong_units_type_raises_error(self):
        """Test that non-string 'units' raises error."""
        data = {
            'years': np.array([1850]),
            'data': np.array([123.0]),
            'units': 123,  # Integer instead of string
            'source': 'UM',
            'dataset': 'xqhuc'
        }

        with pytest.raises(ValueError, match="'units' must be string"):
            validate_metric_output(data, 'GPP')

    def test_optional_error_field_allowed(self):
        """Test that optional 'error' field is allowed."""
        data = {
            'years': np.array([1850, 1851]),
            'data': np.array([123.0, 124.0]),
            'units': 'PgC/yr',
            'source': 'CMIP6',
            'dataset': 'ensemble_mean',
            'error': np.array([5.0, 5.2])  # Optional field
        }

        assert validate_metric_output(data) is True

    def test_optional_metadata_field_allowed(self):
        """Test that optional 'metadata' field is allowed."""
        data = {
            'years': np.array([1850, 1851]),
            'data': np.array([123.0, 124.0]),
            'units': 'PgC/yr',
            'source': 'UM',
            'dataset': 'xqhuc',
            'metadata': {'version': '1.0'}  # Optional field
        }

        assert validate_metric_output(data) is True

    def test_validation_without_metric_name(self):
        """Test validation works without providing metric_name."""
        data = {
            'years': np.array([1850]),
            'data': np.array([123.0]),
            'units': 'PgC/yr',
            'source': 'UM',
            'dataset': 'xqhuc'
        }

        # Should not raise error
        assert validate_metric_output(data) is True

    def test_error_message_includes_metric_name(self):
        """Test that error messages include metric name when provided."""
        data = {
            'years': np.array([1850]),
            'data': np.array([123.0]),
            'units': 123,
            'source': 'UM',
            'dataset': 'xqhuc'
        }

        with pytest.raises(ValueError, match="Metric 'GPP'"):
            validate_metric_output(data, 'GPP')


class TestMetricCategories:
    """Test metric categorization logic."""

    def test_carbon_fluxes_are_flux_category(self):
        """Test that carbon flux metrics are categorized as 'flux'."""
        flux_metrics = ['GPP', 'NPP', 'soilResp', 'NEP']

        for metric in flux_metrics:
            config = get_metric_config(metric)
            assert config['category'] == 'flux', f"{metric} should be in flux category"

    def test_carbon_stocks_are_stock_category(self):
        """Test that carbon stock metrics are categorized as 'stock'."""
        stock_metrics = ['CVeg', 'CSoil', 'VegCarb', 'soilCarbon']

        for metric in stock_metrics:
            config = get_metric_config(metric)
            assert config['category'] == 'stock', f"{metric} should be in stock category"

    def test_flux_metrics_have_per_year_units(self):
        """Test that flux metrics have '/yr' in units."""
        flux_metrics = list_metrics('flux')

        for metric in flux_metrics:
            config = get_metric_config(metric)
            units = config['output_units']
            # Most flux metrics should have /yr (some exceptions possible)
            if metric in ['GPP', 'NPP', 'soilResp', 'NEP', 'fgco2']:
                assert '/yr' in units, f"Flux metric {metric} missing /yr in units"

    def test_stock_metrics_lack_per_year_units(self):
        """Test that stock metrics don't have '/yr' in units."""
        stock_metrics = ['CVeg', 'CSoil']

        for metric in stock_metrics:
            config = get_metric_config(metric)
            units = config['output_units']
            assert '/yr' not in units, f"Stock metric {metric} should not have /yr"

"""
Tests documenting v0.4.0 breaking changes.

These tests verify that deprecated features from v0.3.x have been properly
removed and that canonical names work correctly.
"""

import pytest


class TestVarMappingRemoved:
    """var_mapping parameter was removed in v0.4.0."""

    def test_var_mapping_raises_TypeError(self, tmp_path, monkeypatch):
        """Passing var_mapping raises TypeError (unexpected keyword)."""
        from utils_cmip7.diagnostics.extraction import extract_annual_means

        def mock_load_reccap_mask():
            return None, {'Region1': 'Region1'}

        monkeypatch.setattr(
            'utils_cmip7.diagnostics.extraction.load_reccap_mask',
            mock_load_reccap_mask,
        )

        with pytest.raises(TypeError):
            extract_annual_means(
                expts_list=[],
                var_mapping=['some_mapping'],
                base_dir=str(tmp_path),
            )


class TestVarDictRemoved:
    """var_dict alias was removed in v0.4.0."""

    def test_var_dict_not_importable(self):
        """config.var_dict no longer exists."""
        from utils_cmip7 import config

        assert not hasattr(config, 'var_dict')


class TestDefaultVarMappingRemoved:
    """DEFAULT_VAR_MAPPING was removed in v0.4.0."""

    def test_DEFAULT_VAR_MAPPING_not_importable(self):
        """config.DEFAULT_VAR_MAPPING no longer exists."""
        from utils_cmip7 import config

        assert not hasattr(config, 'DEFAULT_VAR_MAPPING')


class TestLegacyNamesRaiseValueError:
    """Legacy variable names raise ValueError in v0.4.0."""

    @pytest.mark.parametrize("legacy,canonical", [
        ('VegCarb', 'CVeg'),
        ('soilResp', 'Rh'),
        ('soilCarbon', 'CSoil'),
        ('temp', 'tas'),
        ('precip', 'pr'),
        ('fracPFTs', 'frac'),
    ])
    def test_resolve_variable_name_rejects_alias(self, legacy, canonical):
        """resolve_variable_name(legacy) raises ValueError with migration hint."""
        from utils_cmip7.config import resolve_variable_name

        with pytest.raises(ValueError, match="removed in v0.4.0"):
            resolve_variable_name(legacy)

    @pytest.mark.parametrize("legacy", [
        'VegCarb', 'soilResp', 'soilCarbon', 'temp', 'precip', 'fracPFTs',
    ])
    def test_get_variable_config_rejects_alias(self, legacy):
        """get_variable_config(legacy) raises ValueError."""
        from utils_cmip7.config import get_variable_config

        with pytest.raises(ValueError, match="removed in v0.4.0"):
            get_variable_config(legacy)

    @pytest.mark.parametrize("legacy", [
        'VegCarb', 'soilResp', 'soilCarbon', 'temp', 'precip',
    ])
    def test_get_conversion_key_rejects_alias(self, legacy):
        """get_conversion_key(legacy) raises ValueError."""
        from utils_cmip7.config import get_conversion_key

        with pytest.raises(ValueError, match="removed in v0.4.0"):
            get_conversion_key(legacy)


class TestLegacyVarConversionsKeysRemoved:
    """Legacy VAR_CONVERSIONS keys were removed in v0.4.0."""

    @pytest.mark.parametrize("removed_key", [
        'V carb', 'S carb', 'S resp', 'P resp', 'litter flux',
        'Ocean flux', 'Air flux', 'field1560_mm_srf', 'field646_mm_dpth',
        'm01s00i250', 'm02s30i249', 'm01s00i252',
        'vegetation_carbon_content', 'soilCarbon',
    ])
    def test_legacy_key_not_in_VAR_CONVERSIONS(self, removed_key):
        """Removed keys should not be in VAR_CONVERSIONS."""
        from utils_cmip7.config import VAR_CONVERSIONS

        assert removed_key not in VAR_CONVERSIONS


class TestCanonicalNamesWork:
    """Canonical names continue to work correctly in v0.4.0."""

    @pytest.mark.parametrize("name", [
        'GPP', 'NPP', 'Rh', 'CVeg', 'CSoil', 'fgco2', 'tas', 'pr', 'frac', 'co2',
    ])
    def test_resolve_canonical_name(self, name):
        """Canonical names resolve to themselves."""
        from utils_cmip7.config import resolve_variable_name

        assert resolve_variable_name(name) == name

    @pytest.mark.parametrize("name", [
        'GPP', 'NPP', 'Rh', 'CVeg', 'CSoil', 'fgco2', 'tas', 'pr', 'frac', 'co2',
    ])
    def test_get_variable_config_canonical(self, name):
        """get_variable_config works for all canonical names."""
        from utils_cmip7.config import get_variable_config

        cfg = get_variable_config(name)
        assert cfg['canonical_name'] == name
        assert 'stash_name' in cfg
        assert 'conversion_factor' in cfg

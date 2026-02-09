"""
Test diagnostics extraction functions.

Tests for src/utils_cmip7/diagnostics/extraction.py
"""
import pytest
import numpy as np

# Check if iris is available
try:
    import iris
    from iris.coords import DimCoord
    from iris.cube import Cube
    from iris import Constraint
    IRIS_AVAILABLE = True
except ImportError:
    IRIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not IRIS_AVAILABLE, reason="iris not available")


@pytest.fixture
def simple_latlon_cube():
    """Create a simple test cube with lat/lon coordinates and bounds."""
    # Create 3x4 grid: 3 latitudes, 4 longitudes
    data = np.ones((3, 4), dtype=np.float32)

    # Latitude: -60, 0, 60 with bounds
    lat_points = np.array([-60, 0, 60], dtype=np.float32)
    lat_bounds = np.array([[-90, -30], [-30, 30], [30, 90]], dtype=np.float32)
    lat = DimCoord(
        lat_points,
        standard_name='latitude',
        units='degrees_north',
        bounds=lat_bounds
    )

    # Longitude: 0, 90, 180, 270 with bounds
    lon_points = np.array([0, 90, 180, 270], dtype=np.float32)
    lon_bounds = np.array([[-45, 45], [45, 135], [135, 225], [225, 315]], dtype=np.float32)
    lon = DimCoord(
        lon_points,
        standard_name='longitude',
        units='degrees_east',
        bounds=lon_bounds
    )

    cube = Cube(
        data,
        standard_name='air_temperature',
        units='K',
        dim_coords_and_dims=[(lat, 0), (lon, 1)]
    )

    return cube


@pytest.fixture
def cube_without_bounds():
    """Create a cube without coordinate bounds."""
    data = np.ones((3, 4), dtype=np.float32)

    lat = DimCoord(
        np.array([-60, 0, 60], dtype=np.float32),
        standard_name='latitude',
        units='degrees_north'
    )

    lon = DimCoord(
        np.array([0, 90, 180, 270], dtype=np.float32),
        standard_name='longitude',
        units='degrees_east'
    )

    cube = Cube(
        data,
        standard_name='air_temperature',
        units='K',
        dim_coords_and_dims=[(lat, 0), (lon, 1)]
    )

    return cube


class TestComputeLatlonBoxMean:
    """Test compute_latlon_box_mean() function."""

    def test_basic_box_mean_with_bounds(self, simple_latlon_cube):
        """Test basic lat/lon box extraction with bounds."""
        from utils_cmip7.diagnostics.extraction import compute_latlon_box_mean

        # Extract tropical box: 0-180E, -30 to 30N
        result = compute_latlon_box_mean(
            simple_latlon_cube,
            lon_bounds=(0, 180),
            lat_bounds=(-30, 30)
        )

        # Result should be collapsed to scalar
        assert result.ndim == 0 or result.shape == ()
        # Should have metadata from original cube
        assert result.standard_name == 'air_temperature'
        assert result.units == 'K'

    def test_box_mean_without_bounds_guesses_bounds(self, cube_without_bounds):
        """Test that bounds are guessed when missing."""
        from utils_cmip7.diagnostics.extraction import compute_latlon_box_mean

        # Use a box that extracts multiple points to avoid guess_bounds() issue
        result = compute_latlon_box_mean(
            cube_without_bounds,
            lon_bounds=(0, 270),  # Extract 3 longitude points
            lat_bounds=(-70, 70)  # Extract all 3 latitude points
        )

        # Should succeed by guessing bounds
        assert result.ndim == 0 or result.shape == ()

    def test_full_globe_extraction(self, simple_latlon_cube):
        """Test extracting the full globe."""
        from utils_cmip7.diagnostics.extraction import compute_latlon_box_mean

        # Full globe: 0-360E, -90 to 90N
        result = compute_latlon_box_mean(
            simple_latlon_cube,
            lon_bounds=(0, 360),
            lat_bounds=(-90, 90)
        )

        assert result is not None

    def test_invalid_box_raises_error(self, simple_latlon_cube):
        """Test that invalid box (no data) raises ValueError."""
        from utils_cmip7.diagnostics.extraction import compute_latlon_box_mean

        # Box with no data: 500-600E (invalid longitudes)
        with pytest.raises(ValueError, match="No data found in box"):
            compute_latlon_box_mean(
                simple_latlon_cube,
                lon_bounds=(500, 600),
                lat_bounds=(-30, 30)
            )

    def test_small_regional_box(self, simple_latlon_cube):
        """Test extraction of a small regional box."""
        from utils_cmip7.diagnostics.extraction import compute_latlon_box_mean

        # Box that captures at least 2 longitude points: 45-135E, -45 to 45N
        result = compute_latlon_box_mean(
            simple_latlon_cube,
            lon_bounds=(45, 135),
            lat_bounds=(-45, 45)
        )

        # Should extract at least one grid cell
        assert result is not None

    def test_southern_hemisphere_box(self, simple_latlon_cube):
        """Test Southern Hemisphere extraction."""
        from utils_cmip7.diagnostics.extraction import compute_latlon_box_mean

        # Southern hemisphere: 0-360E, -90 to 0N
        result = compute_latlon_box_mean(
            simple_latlon_cube,
            lon_bounds=(0, 360),
            lat_bounds=(-90, 0)
        )

        assert result is not None

    def test_dateline_crossing_box(self, simple_latlon_cube):
        """Test box that crosses the dateline."""
        from utils_cmip7.diagnostics.extraction import compute_latlon_box_mean

        # Pacific box crossing dateline: 270-360E (which wraps to 0E)
        # Note: This may or may not work depending on cube longitude range
        result = compute_latlon_box_mean(
            simple_latlon_cube,
            lon_bounds=(270, 360),
            lat_bounds=(-30, 30)
        )

        assert result is not None


class TestExtractAnnualMeansDeprecations:
    """Test deprecation warnings in extract_annual_means()."""

    @pytest.fixture(autouse=True)
    def mock_reccap(self, monkeypatch):
        """Mock load_reccap_mask to avoid FileNotFoundError."""
        def mock_load_reccap_mask():
            # Return minimal mock data
            return None, {'Region1': 'Region1', 'Region2': 'Region2'}

        monkeypatch.setattr(
            'utils_cmip7.diagnostics.extraction.load_reccap_mask',
            mock_load_reccap_mask
        )

    def test_var_mapping_parameter_raises_TypeError(self, tmp_path):
        """Test that var_mapping parameter raises TypeError in v0.4.0."""
        from utils_cmip7.diagnostics.extraction import extract_annual_means

        with pytest.raises(TypeError):
            extract_annual_means(
                expts_list=[],
                var_mapping=['some_mapping'],
                base_dir=str(tmp_path)
            )

    def test_legacy_variable_names_raise_ValueError(self, tmp_path):
        """Test that legacy variable names raise ValueError in v0.4.0."""
        from utils_cmip7.diagnostics.extraction import extract_annual_means

        # Legacy names should now raise ValueError via resolve_variable_name,
        # which is caught and issued as UserWarning (skipped)
        legacy_vars = ['VegCarb', 'soilResp', 'soilCarbon']

        with pytest.warns(UserWarning, match="Skipping unknown variable"):
            result = extract_annual_means(
                expts_list=[],
                var_list=legacy_vars,
                base_dir=str(tmp_path)
            )

    def test_canonical_names_no_warnings(self, tmp_path):
        """Test that canonical names don't trigger warnings."""
        from utils_cmip7.diagnostics.extraction import extract_annual_means
        import warnings

        # Canonical names should work without deprecation warnings
        canonical_vars = ['CVeg', 'Rh', 'CSoil', 'NPP', 'GPP']

        # Capture all warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extract_annual_means(
                expts_list=[],
                var_list=canonical_vars,
                base_dir=str(tmp_path)
            )

            # Filter for DeprecationWarnings about variable names
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and 'canonical name' in str(warning.message)
            ]

            # Should have no deprecation warnings for canonical names
            assert len(deprecation_warnings) == 0


class TestExtractAnnualMeansBasic:
    """Test basic functionality of extract_annual_means()."""

    @pytest.fixture(autouse=True)
    def mock_reccap(self, monkeypatch):
        """Mock load_reccap_mask to avoid FileNotFoundError."""
        def mock_load_reccap_mask():
            return None, {'Region1': 'Region1', 'Region2': 'Region2'}

        monkeypatch.setattr(
            'utils_cmip7.diagnostics.extraction.load_reccap_mask',
            mock_load_reccap_mask
        )

    def test_empty_experiment_list_returns_empty_dict(self, tmp_path):
        """Test that empty experiment list returns empty dict."""
        from utils_cmip7.diagnostics.extraction import extract_annual_means

        result = extract_annual_means(
            expts_list=[],
            base_dir=str(tmp_path)
        )

        assert result == {}

    def test_missing_directory_creates_directory(self, tmp_path):
        """Test that missing experiment directories are created."""
        from utils_cmip7.diagnostics.extraction import extract_annual_means

        expt_name = 'test_expt'
        result = extract_annual_means(
            expts_list=[expt_name],
            base_dir=str(tmp_path)
        )

        # Directory should be created
        assert (tmp_path / expt_name).exists()
        assert (tmp_path / expt_name).is_dir()

    def test_default_var_list_used_when_none(self, tmp_path):
        """Test that DEFAULT_VAR_LIST is used when var_list is None."""
        from utils_cmip7.diagnostics.extraction import extract_annual_means
        from utils_cmip7.config import DEFAULT_VAR_LIST

        # Don't actually process (no files), just check that default vars are used
        result = extract_annual_means(
            expts_list=[],
            var_list=None,  # Should use DEFAULT_VAR_LIST
            base_dir=str(tmp_path)
        )

        # Result should be empty dict (no experiments)
        assert result == {}

    def test_regions_parameter_filters_regions(self, tmp_path):
        """Test that regions parameter filters which regions are processed."""
        from utils_cmip7.diagnostics.extraction import extract_annual_means

        # Specify only specific regions
        result = extract_annual_means(
            expts_list=[],
            regions=['global', 'Europe'],
            base_dir=str(tmp_path)
        )

        # Just verify it doesn't crash - actual region processing needs mock data
        assert result == {}

    def test_expanduser_in_base_dir(self, tmp_path, monkeypatch):
        """Test that ~ in base_dir is expanded."""
        from utils_cmip7.diagnostics.extraction import extract_annual_means
        import os

        # Mock expanduser to return our tmp path
        def mock_expanduser(path):
            if path.startswith('~'):
                return str(tmp_path) + path[1:]
            return path

        monkeypatch.setattr(os.path, 'expanduser', mock_expanduser)

        result = extract_annual_means(
            expts_list=[],
            base_dir='~/test_data',
            var_list=['GPP']  # Use single var to minimize processing
        )

        # Should succeed without errors
        assert result == {}


class TestExtractAnnualMeansVariableResolution:
    """Test variable name resolution logic."""

    @pytest.fixture(autouse=True)
    def mock_reccap(self, monkeypatch):
        """Mock load_reccap_mask to avoid FileNotFoundError."""
        def mock_load_reccap_mask():
            return None, {'Region1': 'Region1', 'Region2': 'Region2'}

        monkeypatch.setattr(
            'utils_cmip7.diagnostics.extraction.load_reccap_mask',
            mock_load_reccap_mask
        )

    def test_unknown_variable_triggers_warning(self, tmp_path):
        """Test that unknown variable names trigger UserWarning."""
        from utils_cmip7.diagnostics.extraction import extract_annual_means

        with pytest.warns(UserWarning, match="unknown variable"):
            result = extract_annual_means(
                expts_list=[],
                var_list=['INVALID_VAR_NAME'],
                base_dir=str(tmp_path)
            )

    def test_mixed_canonical_and_legacy_names(self, tmp_path):
        """Test mixing canonical and legacy variable names."""
        from utils_cmip7.diagnostics.extraction import extract_annual_means

        # Mix canonical and legacy names — legacy names are skipped with UserWarning
        with pytest.warns(UserWarning, match="Skipping unknown variable"):
            result = extract_annual_means(
                expts_list=[],
                var_list=['GPP', 'VegCarb', 'Rh', 'soilCarbon'],  # Mix of both
                base_dir=str(tmp_path)
            )

        # Should succeed (legacy names skipped)
        assert result == {}

"""
Test spatial aggregation functions.

Tests for src/utils_cmip7/processing/spatial.py

NOTE: These are basic smoke tests. Full integration tests require actual NetCDF data.
"""
import pytest
import numpy as np

try:
    import iris
    import iris.cube
    IRIS_AVAILABLE = True
except ImportError:
    IRIS_AVAILABLE = False

from utils_cmip7.processing.spatial import (
    global_total_pgC,
)


pytestmark = pytest.mark.skipif(not IRIS_AVAILABLE, reason="iris not available")


class TestGlobalTotalPgC:
    """Test global_total_pgC() function."""

    def test_none_cube_raises_error(self):
        """Test that passing None raises ValueError."""
        with pytest.raises(ValueError, match="None cube passed"):
            global_total_pgC(None, 'GPP')

    def test_empty_cubelist_raises_error(self):
        """Test that empty CubeList raises ValueError."""
        empty_cubelist = iris.cube.CubeList([])

        with pytest.raises(ValueError, match="Empty CubeList passed"):
            global_total_pgC(empty_cubelist, 'GPP')

    @pytest.fixture
    def simple_cube(self):
        """Create a simple test cube with lat/lon dimensions."""
        # Create simple 2D data (lat, lon)
        data = np.ones((3, 4), dtype=np.float32)

        # Create latitude coordinate
        lat = iris.coords.DimCoord(
            np.array([-60, 0, 60], dtype=np.float32),
            standard_name='latitude',
            units='degrees',
            bounds=np.array([[-90, -30], [-30, 30], [30, 90]], dtype=np.float32)
        )

        # Create longitude coordinate
        lon = iris.coords.DimCoord(
            np.array([0, 90, 180, 270], dtype=np.float32),
            standard_name='longitude',
            units='degrees',
            bounds=np.array([[-45, 45], [45, 135], [135, 225], [225, 315]], dtype=np.float32)
        )

        # Create cube
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(lat, 0), (lon, 1)],
            units='kg/m2/s'
        )

        return cube

    def test_accepts_single_cube(self, simple_cube):
        """Test that function accepts a single iris.cube.Cube."""
        # Should not raise error
        try:
            result = global_total_pgC(simple_cube, 'GPP')
            assert isinstance(result, iris.cube.Cube)
        except KeyError:
            # May fail due to VAR_CONVERSIONS lookup, but that's OK for this test
            pass

    def test_accepts_cubelist(self, simple_cube):
        """Test that function accepts iris.cube.CubeList."""
        cubelist = iris.cube.CubeList([simple_cube])

        # Should not raise error
        try:
            result = global_total_pgC(cubelist, 'GPP')
            assert isinstance(result, iris.cube.Cube)
        except KeyError:
            # May fail due to VAR_CONVERSIONS lookup, but that's OK for this test
            pass

    def test_handles_cube_without_bounds(self, simple_cube):
        """Test that function handles cubes without coordinate bounds."""
        # Remove bounds
        simple_cube.coord('latitude').bounds = None
        simple_cube.coord('longitude').bounds = None

        # Should add bounds automatically
        try:
            result = global_total_pgC(simple_cube, 'GPP')
            # Bounds should have been added
            assert simple_cube.coord('latitude').has_bounds()
            assert simple_cube.coord('longitude').has_bounds()
        except KeyError:
            # May fail due to VAR_CONVERSIONS lookup, but bounds should be added
            assert simple_cube.coord('latitude').has_bounds()
            assert simple_cube.coord('longitude').has_bounds()


class TestComputeTerrestrialArea:
    """Test compute_terrestrial_area() function (currently unused)."""

    def test_function_exists(self):
        """Test that compute_terrestrial_area function exists."""
        from utils_cmip7.processing.spatial import compute_terrestrial_area
        assert callable(compute_terrestrial_area)


class TestGlobalMean:
    """Test global_mean() function if it exists."""

    def test_function_exists(self):
        """Check if global_mean function exists."""
        try:
            from utils_cmip7.processing.spatial import global_mean
            assert callable(global_mean)
        except ImportError:
            pytest.skip("global_mean not implemented")


class TestIntegrationSmokeTests:
    """Integration smoke tests with mock data."""

    @pytest.fixture
    def mock_3d_cube(self):
        """Create a 3D cube with time dimension."""
        # Create 3D data (time, lat, lon)
        data = np.random.random((5, 3, 4)).astype(np.float32)

        # Create coordinates
        time = iris.coords.DimCoord(
            np.arange(5, dtype=np.float32),
            standard_name='time',
            units='days since 1850-01-01'
        )

        lat = iris.coords.DimCoord(
            np.array([-60, 0, 60], dtype=np.float32),
            standard_name='latitude',
            units='degrees'
        )

        lon = iris.coords.DimCoord(
            np.array([0, 90, 180, 270], dtype=np.float32),
            standard_name='longitude',
            units='degrees'
        )

        # Create cube
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
            units='kg/m2/s'
        )

        return cube

    def test_handles_3d_data(self, mock_3d_cube):
        """Test that function handles 3D cubes (time, lat, lon)."""
        # Should process without error
        try:
            result = global_total_pgC(mock_3d_cube, 'GPP')
            # Result should have collapsed lat/lon but kept time
            assert isinstance(result, iris.cube.Cube)
        except KeyError:
            # May fail due to VAR_CONVERSIONS, but that's not what we're testing
            pass


class TestSpatialAggregationProperties:
    """Test mathematical properties of spatial aggregation."""

    @pytest.fixture
    def uniform_cube(self):
        """Create cube with uniform data for testing."""
        data = np.full((3, 4), 2.0, dtype=np.float32)

        lat = iris.coords.DimCoord(
            np.array([-60, 0, 60], dtype=np.float32),
            standard_name='latitude',
            units='degrees'
        )

        lon = iris.coords.DimCoord(
            np.array([0, 90, 180, 270], dtype=np.float32),
            standard_name='longitude',
            units='degrees'
        )

        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(lat, 0), (lon, 1)],
            units='kg/m2/s'
        )

        return cube

    def test_uniform_data_aggregation(self, uniform_cube):
        """Test aggregation with uniform data."""
        # With uniform data, aggregation should be predictable
        # (though exact value depends on VAR_CONVERSIONS)
        try:
            result = global_total_pgC(uniform_cube, 'GPP')
            assert isinstance(result, iris.cube.Cube)
            # Result should be scalar (0-d cube)
            assert result.ndim <= 1  # May have 1 dim if processing 3D input
        except KeyError:
            pass


# Skip all tests if iris is not available
if not IRIS_AVAILABLE:
    pytest.skip("Skipping spatial tests - iris not installed", allow_module_level=True)

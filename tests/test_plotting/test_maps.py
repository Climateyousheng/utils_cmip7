"""
Tests for utils_cmip7.plotting.maps — geographic map plotting.

Uses Agg backend throughout to avoid GUI windows.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

iris = pytest.importorskip("iris")
cartopy = pytest.importorskip("cartopy")

import cartopy.crs as ccrs  # noqa: E402
import cf_units  # noqa: E402
import iris.cube  # noqa: E402
import iris.coords  # noqa: E402

from utils_cmip7.config import RECCAP_REGION_BOUNDS, get_region_bounds  # noqa: E402
from utils_cmip7.plotting.maps import _select_time_slice, plot_spatial_map  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_time_coord(years, calendar="360_day"):
    """Build an iris DimCoord representing annual or multi-step time."""
    time_units = cf_units.Unit(
        "days since 1850-01-01", calendar=calendar
    )
    # One point per year, placed at mid-year (day 180)
    points = np.array([(y - 1850) * 360 + 180 for y in years], dtype=float)
    return iris.coords.DimCoord(
        points, standard_name="time", units=time_units,
    )


@pytest.fixture
def mock_2d_cube():
    """Cube with only latitude and longitude (no time)."""
    lat = iris.coords.DimCoord(
        np.linspace(-90, 90, 10), standard_name="latitude", units="degrees",
    )
    lon = iris.coords.DimCoord(
        np.linspace(-180, 180, 20), standard_name="longitude", units="degrees",
    )
    data = np.random.default_rng(42).random((10, 20))
    cube = iris.cube.Cube(
        data,
        dim_coords_and_dims=[(lat, 0), (lon, 1)],
        standard_name="air_temperature",
        units="K",
    )
    return cube


@pytest.fixture
def mock_3d_cube():
    """Cube with time, latitude, and longitude (5 annual steps)."""
    years = [1900, 1901, 1902, 1903, 1904]
    time_coord = _make_time_coord(years)
    lat = iris.coords.DimCoord(
        np.linspace(-90, 90, 10), standard_name="latitude", units="degrees",
    )
    lon = iris.coords.DimCoord(
        np.linspace(-180, 180, 20), standard_name="longitude", units="degrees",
    )
    data = np.random.default_rng(99).random((5, 10, 20))
    cube = iris.cube.Cube(
        data,
        dim_coords_and_dims=[(time_coord, 0), (lat, 1), (lon, 2)],
        standard_name="air_temperature",
        units="K",
    )
    return cube


@pytest.fixture
def mock_monthly_cube():
    """Cube with 12 monthly timesteps in a single year (1900)."""
    time_units = cf_units.Unit("days since 1850-01-01", calendar="360_day")
    points = np.array([(1900 - 1850) * 360 + m * 30 + 15 for m in range(12)],
                      dtype=float)
    time_coord = iris.coords.DimCoord(
        points, standard_name="time", units=time_units,
    )
    lat = iris.coords.DimCoord(
        np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
    )
    lon = iris.coords.DimCoord(
        np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
    )
    data = np.random.default_rng(7).random((12, 5, 10))
    cube = iris.cube.Cube(
        data,
        dim_coords_and_dims=[(time_coord, 0), (lat, 1), (lon, 2)],
        standard_name="air_temperature",
        units="K",
    )
    return cube


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ===================================================================
# TestSelectTimeSlice
# ===================================================================

class TestSelectTimeSlice:
    """Tests for _select_time_slice()."""

    def test_2d_passthrough(self, mock_2d_cube):
        data, year = _select_time_slice(mock_2d_cube)
        assert data.shape == (10, 20)
        assert year is None

    def test_default_first_step(self, mock_3d_cube):
        data, year = _select_time_slice(mock_3d_cube)
        assert data.shape == (10, 20)
        assert year == 1900
        np.testing.assert_array_equal(data, mock_3d_cube[0].data)

    def test_by_index(self, mock_3d_cube):
        data, year = _select_time_slice(mock_3d_cube, time_index=3)
        assert year == 1903
        np.testing.assert_array_equal(data, mock_3d_cube[3].data)

    def test_by_year(self, mock_3d_cube):
        data, year = _select_time_slice(mock_3d_cube, time=1902)
        assert year == 1902
        np.testing.assert_array_equal(data, mock_3d_cube[2].data)

    def test_year_averaging(self, mock_monthly_cube):
        """Monthly data for one year should be averaged."""
        data, year = _select_time_slice(mock_monthly_cube, time=1900)
        assert year == 1900
        assert data.shape == (5, 10)
        expected = np.mean(mock_monthly_cube.data, axis=0)
        np.testing.assert_allclose(data, expected)

    def test_mutual_exclusivity_error(self, mock_3d_cube):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _select_time_slice(mock_3d_cube, time=1900, time_index=0)

    def test_missing_year_error(self, mock_3d_cube):
        with pytest.raises(ValueError, match="Year 2050 not found"):
            _select_time_slice(mock_3d_cube, time=2050)

    def test_numeric_time_no_calendar(self):
        """Cube whose time coord has no calendar (raw float points)."""
        time_coord = iris.coords.DimCoord(
            np.array([0.0, 1.0, 2.0]),
            long_name="time",
            units="1",
        )
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        data = np.random.default_rng(11).random((3, 5, 10))
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(time_coord, 0), (lat, 1), (lon, 2)],
        )
        # Default (first timestep) should work, year returned as None
        data_2d, year = _select_time_slice(cube)
        assert data_2d.shape == (5, 10)
        assert year is None

        # time_index should also work
        data_2d, year = _select_time_slice(cube, time_index=1)
        assert year is None
        np.testing.assert_array_equal(data_2d, data[1])

        # Selecting by year should raise a clear error
        with pytest.raises(ValueError, match="no calendar metadata"):
            _select_time_slice(cube, time=1900)


# ===================================================================
# TestPlotSpatialMap
# ===================================================================

class TestPlotSpatialMap:
    """Tests for plot_spatial_map()."""

    def test_returns_fig_and_ax(self, mock_2d_cube):
        fig, ax = plot_spatial_map(mock_2d_cube)
        assert isinstance(fig, plt.Figure)
        assert hasattr(ax, "projection")

    def test_works_with_2d_cube(self, mock_2d_cube):
        fig, ax = plot_spatial_map(mock_2d_cube)
        assert ax.get_title() == "air_temperature"

    def test_works_with_3d_cube(self, mock_3d_cube):
        fig, ax = plot_spatial_map(mock_3d_cube, time=1902)
        assert "1902" in ax.get_title()

    def test_ax_parameter_reused(self, mock_2d_cube):
        proj = ccrs.PlateCarree()
        fig_ext, ax_ext = plt.subplots(
            subplot_kw={"projection": proj}
        )
        fig_ret, ax_ret = plot_spatial_map(mock_2d_cube, ax=ax_ext)
        assert ax_ret is ax_ext
        assert fig_ret is fig_ext

    def test_named_region_sets_extent(self, mock_2d_cube):
        fig, ax = plot_spatial_map(mock_2d_cube, region="Europe")
        # PlateCarree should be used for regional views by default
        assert isinstance(ax.projection, ccrs.PlateCarree)

    def test_explicit_bounds(self, mock_2d_cube):
        fig, ax = plot_spatial_map(
            mock_2d_cube,
            lon_bounds=(-20, 40),
            lat_bounds=(30, 70),
        )
        assert isinstance(ax.projection, ccrs.PlateCarree)

    def test_region_and_bounds_raises(self, mock_2d_cube):
        with pytest.raises(ValueError, match="mutually exclusive"):
            plot_spatial_map(
                mock_2d_cube,
                region="Europe",
                lon_bounds=(-20, 40),
            )

    def test_lon_bounds_without_lat_bounds_raises(self, mock_2d_cube):
        with pytest.raises(ValueError, match="together"):
            plot_spatial_map(mock_2d_cube, lon_bounds=(-20, 40))

    def test_lat_bounds_without_lon_bounds_raises(self, mock_2d_cube):
        with pytest.raises(ValueError, match="together"):
            plot_spatial_map(mock_2d_cube, lat_bounds=(30, 70))

    @pytest.mark.xfail(
        reason="cartopy 0.25 + shapely 2.x bug with Mollweide projection",
        raises=TypeError,
        strict=False,
    )
    def test_custom_projection(self, mock_2d_cube):
        proj = ccrs.Mollweide()
        fig, ax = plot_spatial_map(mock_2d_cube, projection=proj)
        assert isinstance(ax.projection, ccrs.Mollweide)

    def test_colorbar_units(self, mock_2d_cube):
        fig, ax = plot_spatial_map(mock_2d_cube, units="custom units")
        # The colorbar should exist (we just verify no error)
        assert len(fig.axes) > 1  # colorbar creates extra axes

    def test_auto_title(self, mock_3d_cube):
        fig, ax = plot_spatial_map(mock_3d_cube)
        assert "air_temperature" in ax.get_title()
        assert "1900" in ax.get_title()

    def test_custom_title(self, mock_2d_cube):
        fig, ax = plot_spatial_map(mock_2d_cube, title="My Title")
        assert ax.get_title() == "My Title"

    def test_no_coastlines(self, mock_2d_cube):
        fig, ax = plot_spatial_map(mock_2d_cube, add_coastlines=False)
        # Verify it completes without error (coastlines are optional)
        assert isinstance(fig, plt.Figure)

    def test_no_gridlines(self, mock_2d_cube):
        fig, ax = plot_spatial_map(mock_2d_cube, add_gridlines=False)
        assert isinstance(fig, plt.Figure)

    def test_no_colorbar(self, mock_2d_cube):
        fig, ax = plot_spatial_map(mock_2d_cube, colorbar=False)
        # Only the main axes should be present (no colorbar axes)
        assert len(fig.axes) == 1

    def test_invalid_input_raises_typeerror(self):
        with pytest.raises(TypeError, match="iris.cube.Cube"):
            plot_spatial_map(np.zeros((5, 5)))

    def test_invalid_ax_raises_typeerror(self, mock_2d_cube):
        fig, ax = plt.subplots()  # plain Axes, not GeoAxes
        with pytest.raises(TypeError, match="GeoAxes"):
            plot_spatial_map(mock_2d_cube, ax=ax)

    def test_vmin_vmax(self, mock_2d_cube):
        fig, ax = plot_spatial_map(mock_2d_cube, vmin=0.0, vmax=1.0)
        assert isinstance(fig, plt.Figure)

    def test_custom_cmap(self, mock_2d_cube):
        fig, ax = plot_spatial_map(mock_2d_cube, cmap="plasma")
        assert isinstance(fig, plt.Figure)

    def test_time_index_on_3d(self, mock_3d_cube):
        fig, ax = plot_spatial_map(mock_3d_cube, time_index=2)
        assert "1902" in ax.get_title()

    def test_squeeze_single_extra_dim(self):
        """Cube with a length-1 pseudo-level should be auto-squeezed."""
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        level = iris.coords.DimCoord([1], long_name="pseudo_level", units="1")
        data = np.random.default_rng(8).random((1, 5, 10))
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(level, 0), (lat, 1), (lon, 2)],
            standard_name="air_temperature",
            units="K",
        )
        fig, ax = plot_spatial_map(cube)
        assert isinstance(fig, plt.Figure)

    def test_multi_level_raises(self):
        """Cube with a multi-valued extra dim should raise ValueError."""
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        level = iris.coords.DimCoord([1, 2, 3], long_name="pseudo_level", units="1")
        data = np.random.default_rng(9).random((3, 5, 10))
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(level, 0), (lat, 1), (lon, 2)],
            standard_name="air_temperature",
            units="K",
        )
        with pytest.raises(ValueError, match="extra dimension"):
            plot_spatial_map(cube)


# ===================================================================
# TestRegionBoundsConfig
# ===================================================================

class TestRegionBoundsConfig:
    """Tests for RECCAP_REGION_BOUNDS and get_region_bounds()."""

    def test_all_reccap_regions_have_bounds(self):
        from utils_cmip7.config import RECCAP_REGIONS
        for region_name in RECCAP_REGIONS.values():
            assert region_name in RECCAP_REGION_BOUNDS, (
                f"RECCAP region '{region_name}' missing from RECCAP_REGION_BOUNDS"
            )

    def test_bounds_are_4_tuples(self):
        for name, bounds in RECCAP_REGION_BOUNDS.items():
            assert len(bounds) == 4, f"{name}: expected 4-tuple, got {len(bounds)}"

    def test_lon_min_lt_lon_max(self):
        for name, (lon_min, lon_max, _, _) in RECCAP_REGION_BOUNDS.items():
            assert lon_min < lon_max, (
                f"{name}: lon_min ({lon_min}) >= lon_max ({lon_max})"
            )

    def test_lat_min_lt_lat_max(self):
        for name, (_, _, lat_min, lat_max) in RECCAP_REGION_BOUNDS.items():
            assert lat_min < lat_max, (
                f"{name}: lat_min ({lat_min}) >= lat_max ({lat_max})"
            )

    def test_get_region_bounds_returns_correct_tuple(self):
        bounds = get_region_bounds("Europe")
        assert bounds == (-15, 45, 35, 75)

    def test_get_region_bounds_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown region"):
            get_region_bounds("Atlantis")

    def test_bounds_values_in_valid_range(self):
        for name, (lon_min, lon_max, lat_min, lat_max) in RECCAP_REGION_BOUNDS.items():
            assert -180 <= lon_min <= 180, f"{name}: lon_min out of range"
            assert -180 <= lon_max <= 180, f"{name}: lon_max out of range"
            assert -90 <= lat_min <= 90, f"{name}: lat_min out of range"
            assert -90 <= lat_max <= 90, f"{name}: lat_max out of range"

"""
Tests for utils_cmip7.plotting.maps — geographic map plotting.

Uses Agg backend throughout to avoid GUI windows.
The plotting functions now accept arrays (not cubes), so tests
use extract_map_field / extract_anomaly_field for the cube → array step.
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
from utils_cmip7.processing.map_fields import (  # noqa: E402
    extract_map_field,
    extract_anomaly_field,
    combine_fields,
    _select_time_slice,
)
from utils_cmip7.plotting.maps import (  # noqa: E402
    plot_spatial_map,
    plot_spatial_anomaly,
)


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
# TestExtractMapField
# ===================================================================

class TestExtractMapField:
    """Tests for extract_map_field()."""

    def test_returns_dict_keys(self, mock_2d_cube):
        result = extract_map_field(mock_2d_cube)
        expected_keys = {"data", "lons", "lats", "name", "units", "year", "title"}
        assert set(result.keys()) == expected_keys

    def test_2d_cube(self, mock_2d_cube):
        result = extract_map_field(mock_2d_cube)
        assert result["data"].shape == (10, 20)
        assert result["year"] is None
        assert result["name"] == "air_temperature"
        assert result["units"] == "K"
        assert result["title"] == "air_temperature"

    def test_3d_cube_with_time(self, mock_3d_cube):
        result = extract_map_field(mock_3d_cube, time=1902)
        assert result["data"].shape == (10, 20)
        assert result["year"] == 1902
        assert "1902" in result["title"]

    def test_invalid_input_raises_typeerror(self):
        with pytest.raises(TypeError, match="iris.cube.Cube"):
            extract_map_field(np.zeros((5, 5)))

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
        result = extract_map_field(cube)
        assert result["data"].shape == (5, 10)

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
            extract_map_field(cube)

    def test_masked_values_become_nan(self):
        """Masked cells (missing_value) should be NaN, not fill values."""
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        raw = np.random.default_rng(42).random((5, 10))
        mask = np.zeros_like(raw, dtype=bool)
        mask[0, :3] = True  # mask some ocean cells
        data = np.ma.MaskedArray(raw, mask=mask, fill_value=1e20)
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(lat, 0), (lon, 1)],
            standard_name="air_temperature",
            units="K",
        )
        result = extract_map_field(cube)
        assert not isinstance(result["data"], np.ma.MaskedArray)
        assert np.isnan(result["data"][0, 0])
        assert np.isnan(result["data"][0, 2])
        assert not np.isnan(result["data"][1, 0])

    def test_unmasked_values_preserved(self):
        """Unmasked cells should keep their original values."""
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        raw = np.ones((5, 10)) * 300.0
        mask = np.zeros_like(raw, dtype=bool)
        mask[0, 0] = True
        data = np.ma.MaskedArray(raw, mask=mask, fill_value=1e20)
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(lat, 0), (lon, 1)],
            standard_name="air_temperature",
            units="K",
        )
        result = extract_map_field(cube)
        assert result["data"][1, 0] == 300.0
        assert np.isnan(result["data"][0, 0])

    def test_extract_map_field_with_level(self):
        """4D cube (time, pft, lat, lon) with level selection."""
        time_coord = _make_time_coord([1900])
        pft = iris.coords.DimCoord(
            np.arange(9), long_name="pseudo_level", units="1",
        )
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        rng = np.random.default_rng(42)
        data = rng.random((1, 9, 5, 10))
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[
                (time_coord, 0), (pft, 1), (lat, 2), (lon, 3),
            ],
            long_name="frac",
            units="1",
        )
        result = extract_map_field(cube, level=0)
        assert result["data"].shape == (5, 10)
        np.testing.assert_allclose(result["data"], data[0, 0], atol=1e-6)

    def test_extract_map_field_level_out_of_range(self):
        """level out of range should raise ValueError."""
        pft = iris.coords.DimCoord(
            np.arange(9), long_name="pseudo_level", units="1",
        )
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        data = np.random.default_rng(7).random((9, 5, 10))
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(pft, 0), (lat, 1), (lon, 2)],
            long_name="frac",
            units="1",
        )
        with pytest.raises(ValueError, match="out of range"):
            extract_map_field(cube, level=9)

    def test_extract_map_field_with_generic_level(self):
        """4D cube with non-standard DimCoord name ('generic')."""
        time_coord = _make_time_coord([1900])
        generic = iris.coords.DimCoord(
            np.arange(9), long_name="generic", units="1",
        )
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        rng = np.random.default_rng(42)
        data = rng.random((1, 9, 5, 10))
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[
                (time_coord, 0), (generic, 1), (lat, 2), (lon, 3),
            ],
            long_name="frac",
            units="1",
        )
        result = extract_map_field(cube, level=0)
        assert result["data"].shape == (5, 10)
        np.testing.assert_allclose(result["data"], data[0, 0], atol=1e-6)

    def test_extract_map_field_with_anonymous_level(self):
        """4D cube with anonymous extra dimension (no coord on dim 1)."""
        time_coord = _make_time_coord([1900])
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        rng = np.random.default_rng(42)
        data = rng.random((1, 9, 5, 10))
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[
                (time_coord, 0), (lat, 2), (lon, 3),
            ],
            long_name="frac",
            units="1",
        )
        # Dim 1 has no coordinate — it is anonymous
        result = extract_map_field(cube, level=0)
        assert result["data"].shape == (5, 10)
        np.testing.assert_allclose(result["data"], data[0, 0], atol=1e-6)

    def test_squeeze_error_suggests_level(self):
        """Error message should mention level= parameter."""
        generic = iris.coords.DimCoord(
            np.arange(9), long_name="generic", units="1",
        )
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        data = np.random.default_rng(42).random((9, 5, 10))
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(generic, 0), (lat, 1), (lon, 2)],
            long_name="frac",
            units="1",
        )
        with pytest.raises(ValueError, match="level="):
            extract_map_field(cube)

    def test_squeeze_error_suggests_level_anonymous(self):
        """Error message should mention level= even for anonymous dimensions."""
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        data = np.random.default_rng(42).random((9, 5, 10))
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(lat, 1), (lon, 2)],
            long_name="frac",
            units="1",
        )
        with pytest.raises(ValueError, match="level="):
            extract_map_field(cube)

    def test_anomaly_with_level(self):
        """Anomaly extraction for 4D cube with level selection."""
        years = [1900, 1901]
        time_coord = _make_time_coord(years)
        pft = iris.coords.DimCoord(
            np.arange(9), long_name="pseudo_level", units="1",
        )
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        rng = np.random.default_rng(42)
        data = rng.random((2, 9, 5, 10))
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[
                (time_coord, 0), (pft, 1), (lat, 2), (lon, 3),
            ],
            long_name="frac",
            units="1",
        )
        result = extract_anomaly_field(cube, level=0)
        expected = data[1, 0] - data[0, 0]
        assert result["data"].shape == (5, 10)
        np.testing.assert_allclose(result["data"], expected, atol=1e-6)


# ===================================================================
# TestExtractAnomalyField
# ===================================================================

class TestExtractAnomalyField:
    """Tests for extract_anomaly_field()."""

    def test_returns_dict_keys(self, mock_3d_cube):
        result = extract_anomaly_field(mock_3d_cube)
        expected_keys = {
            "data", "lons", "lats", "name", "units",
            "year_a", "year_b", "vmin", "vmax", "title",
        }
        assert set(result.keys()) == expected_keys

    def test_default_last_minus_first(self, mock_3d_cube):
        result = extract_anomaly_field(mock_3d_cube)
        expected = mock_3d_cube[-1].data - mock_3d_cube[0].data
        np.testing.assert_allclose(result["data"], expected)
        assert result["year_a"] == 1904
        assert result["year_b"] == 1900
        assert "1904" in result["title"]
        assert "1900" in result["title"]

    def test_symmetric_vmin_vmax(self, mock_3d_cube):
        result = extract_anomaly_field(mock_3d_cube)
        assert result["vmin"] == pytest.approx(-result["vmax"])

    def test_no_symmetric(self, mock_3d_cube):
        result = extract_anomaly_field(mock_3d_cube, symmetric=False)
        assert result["vmin"] is None
        assert result["vmax"] is None

    def test_by_year(self, mock_3d_cube):
        result = extract_anomaly_field(mock_3d_cube, time_a=1904, time_b=1900)
        assert result["year_a"] == 1904
        assert result["year_b"] == 1900

    def test_2d_cube_raises(self, mock_2d_cube):
        with pytest.raises(ValueError, match="time dimension"):
            extract_anomaly_field(mock_2d_cube)

    def test_invalid_input_raises_typeerror(self):
        with pytest.raises(TypeError, match="iris.cube.Cube"):
            extract_anomaly_field(np.zeros((5, 5)))

    def test_masked_anomaly_becomes_nan(self):
        """Masked cells in anomaly should be NaN, not fill values."""
        years = [1900, 1901]
        time_coord = _make_time_coord(years)
        lat = iris.coords.DimCoord(
            np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
        )
        lon = iris.coords.DimCoord(
            np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
        )
        raw = np.random.default_rng(42).random((2, 5, 10))
        mask = np.zeros_like(raw, dtype=bool)
        mask[:, 0, :3] = True  # ocean cells masked across all timesteps
        data = np.ma.MaskedArray(raw, mask=mask, fill_value=1e20)
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(time_coord, 0), (lat, 1), (lon, 2)],
            standard_name="air_temperature",
            units="K",
        )
        result = extract_anomaly_field(cube)
        assert not isinstance(result["data"], np.ma.MaskedArray)
        assert np.isnan(result["data"][0, 0])
        assert not np.isnan(result["data"][1, 0])


# ===================================================================
# TestPlotSpatialMap
# ===================================================================

class TestPlotSpatialMap:
    """Tests for plot_spatial_map() — array-based API."""

    def _field(self, cube, **kwargs):
        """Helper: extract field dict from cube."""
        return extract_map_field(cube, **kwargs)

    def test_returns_fig_and_ax(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        fig, ax = plot_spatial_map(f["data"], f["lons"], f["lats"])
        assert isinstance(fig, plt.Figure)
        assert hasattr(ax, "projection")

    def test_works_with_2d_cube(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], title=f["title"],
        )
        assert ax.get_title() == "air_temperature"

    def test_works_with_3d_cube(self, mock_3d_cube):
        f = self._field(mock_3d_cube, time=1902)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], title=f["title"],
        )
        assert "1902" in ax.get_title()

    def test_ax_parameter_reused(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        proj = ccrs.PlateCarree()
        fig_ext, ax_ext = plt.subplots(
            subplot_kw={"projection": proj}
        )
        fig_ret, ax_ret = plot_spatial_map(
            f["data"], f["lons"], f["lats"], ax=ax_ext,
        )
        assert ax_ret is ax_ext
        assert fig_ret is fig_ext

    def test_named_region_sets_extent(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], region="Europe",
        )
        assert isinstance(ax.projection, ccrs.PlateCarree)

    def test_explicit_bounds(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"],
            lon_bounds=(-20, 40),
            lat_bounds=(30, 70),
        )
        assert isinstance(ax.projection, ccrs.PlateCarree)

    def test_region_and_bounds_raises(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        with pytest.raises(ValueError, match="mutually exclusive"):
            plot_spatial_map(
                f["data"], f["lons"], f["lats"],
                region="Europe",
                lon_bounds=(-20, 40),
            )

    def test_lon_bounds_without_lat_bounds_raises(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        with pytest.raises(ValueError, match="together"):
            plot_spatial_map(f["data"], f["lons"], f["lats"], lon_bounds=(-20, 40))

    def test_lat_bounds_without_lon_bounds_raises(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        with pytest.raises(ValueError, match="together"):
            plot_spatial_map(f["data"], f["lons"], f["lats"], lat_bounds=(30, 70))

    @pytest.mark.xfail(
        reason="cartopy 0.25 + shapely 2.x bug with Mollweide projection",
        raises=TypeError,
        strict=False,
    )
    def test_custom_projection(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        proj = ccrs.Mollweide()
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], projection=proj,
        )
        assert isinstance(ax.projection, ccrs.Mollweide)

    def test_colorbar_units(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], units="custom units",
        )
        assert len(fig.axes) > 1  # colorbar creates extra axes

    def test_auto_title(self, mock_3d_cube):
        f = self._field(mock_3d_cube)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], title=f["title"],
        )
        assert "air_temperature" in ax.get_title()
        assert "1900" in ax.get_title()

    def test_custom_title(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], title="My Title",
        )
        assert ax.get_title() == "My Title"

    def test_no_coastlines(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], add_coastlines=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_no_gridlines(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], add_gridlines=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_no_colorbar(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], colorbar=False,
        )
        assert len(fig.axes) == 1

    def test_colorbar_no_overlap_in_subplots(self, mock_2d_cube):
        """Colorbar should not overlap x-axis labels in a 2x3 subplot grid."""
        f = self._field(mock_2d_cube)
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            2, 3, figsize=(18, 10),
            subplot_kw={"projection": proj},
        )
        for ax in axes.flat:
            plot_spatial_map(
                f["data"], f["lons"], f["lats"],
                ax=ax, units="K",
            )
        fig.canvas.draw()
        # Check that each colorbar's top edge is below the parent axes bottom
        geo_axes = [a for a in fig.axes if hasattr(a, "projection")]
        cbar_axes = [a for a in fig.axes if not hasattr(a, "projection")]
        assert len(cbar_axes) == 6, "Expected 6 colorbars for 2x3 grid"
        for geo_ax, cbar_ax in zip(geo_axes, cbar_axes):
            geo_bottom = geo_ax.get_position().y0
            cbar_top = cbar_ax.get_position().y1
            assert cbar_top <= geo_bottom + 0.005, (
                f"Colorbar top ({cbar_top:.4f}) overlaps axes bottom "
                f"({geo_bottom:.4f})"
            )

    def test_invalid_data_raises_valueerror(self):
        with pytest.raises(ValueError, match="2D"):
            plot_spatial_map(np.zeros((5,)), np.zeros(5), np.zeros(5))

    def test_invalid_ax_raises_typeerror(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        fig, ax = plt.subplots()  # plain Axes, not GeoAxes
        with pytest.raises(TypeError, match="GeoAxes"):
            plot_spatial_map(f["data"], f["lons"], f["lats"], ax=ax)

    def test_vmin_vmax(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], vmin=0.0, vmax=1.0,
        )
        assert isinstance(fig, plt.Figure)

    def test_custom_cmap(self, mock_2d_cube):
        f = self._field(mock_2d_cube)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], cmap="plasma",
        )
        assert isinstance(fig, plt.Figure)

    def test_time_index_on_3d(self, mock_3d_cube):
        f = self._field(mock_3d_cube, time_index=2)
        fig, ax = plot_spatial_map(
            f["data"], f["lons"], f["lats"], title=f["title"],
        )
        assert "1902" in ax.get_title()

    def test_cyclic_point_closes_gap(self):
        """Near-global lon grid (0..356.25) should get a cyclic wrap point."""
        n_lon = 288
        lons = np.linspace(0, 356.25, n_lon)
        lats = np.linspace(-90, 90, 10)
        data = np.random.default_rng(42).random((10, n_lon))
        # Use PlateCarree to avoid the cartopy 0.25 + shapely 2.x bug
        fig, ax = plot_spatial_map(
            data, lons, lats, projection=ccrs.PlateCarree(),
        )
        assert isinstance(fig, plt.Figure)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            plot_spatial_map(
                np.zeros((5, 10)),
                np.zeros(5),  # wrong: should be 10
                np.zeros(5),
            )


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


# ===================================================================
# TestPlotSpatialAnomaly
# ===================================================================

class TestPlotSpatialAnomaly:
    """Tests for plot_spatial_anomaly() — array-based API."""

    def _anomaly(self, cube, **kwargs):
        """Helper: extract anomaly dict from cube."""
        return extract_anomaly_field(cube, **kwargs)

    def test_returns_fig_and_ax(self, mock_3d_cube):
        a = self._anomaly(mock_3d_cube)
        fig, ax = plot_spatial_anomaly(a["data"], a["lons"], a["lats"])
        assert isinstance(fig, plt.Figure)
        assert hasattr(ax, "projection")

    def test_default_last_minus_first(self, mock_3d_cube):
        a = self._anomaly(mock_3d_cube)
        fig, ax = plot_spatial_anomaly(
            a["data"], a["lons"], a["lats"],
            vmin=a["vmin"], vmax=a["vmax"], title=a["title"],
        )
        assert "1904" in ax.get_title()
        assert "1900" in ax.get_title()
        # Verify symmetric limits
        cs = ax.collections[0]
        clim = cs.get_clim()
        assert clim[0] == pytest.approx(-clim[1])

    def test_by_year(self, mock_3d_cube):
        a = self._anomaly(mock_3d_cube, time_a=1904, time_b=1900)
        fig, ax = plot_spatial_anomaly(
            a["data"], a["lons"], a["lats"], title=a["title"],
        )
        assert "1904" in ax.get_title()
        assert "1900" in ax.get_title()

    def test_by_index(self, mock_3d_cube):
        a = self._anomaly(mock_3d_cube, time_index_a=4, time_index_b=0)
        fig, ax = plot_spatial_anomaly(
            a["data"], a["lons"], a["lats"], title=a["title"],
        )
        assert "1904" in ax.get_title()
        assert "1900" in ax.get_title()

    def test_symmetric_colorbar(self, mock_3d_cube):
        a = self._anomaly(mock_3d_cube)
        fig, ax = plot_spatial_anomaly(
            a["data"], a["lons"], a["lats"],
            vmin=a["vmin"], vmax=a["vmax"],
        )
        cs = ax.collections[0] if ax.collections else None
        if cs is not None:
            clim = cs.get_clim()
            assert clim[0] == pytest.approx(-clim[1]), (
                f"Expected symmetric limits, got {clim}"
            )

    def test_symmetric_false(self, mock_3d_cube):
        a = self._anomaly(mock_3d_cube, symmetric=False)
        fig, ax = plot_spatial_anomaly(a["data"], a["lons"], a["lats"])
        assert isinstance(fig, plt.Figure)

    def test_custom_title(self, mock_3d_cube):
        a = self._anomaly(mock_3d_cube)
        fig, ax = plot_spatial_anomaly(
            a["data"], a["lons"], a["lats"], title="My Anomaly",
        )
        assert ax.get_title() == "My Anomaly"

    def test_auto_title_contains_years(self, mock_3d_cube):
        a = self._anomaly(mock_3d_cube, time_a=1903, time_b=1901)
        fig, ax = plot_spatial_anomaly(
            a["data"], a["lons"], a["lats"], title=a["title"],
        )
        title = ax.get_title()
        assert "anomaly" in title
        assert "1903" in title
        assert "1901" in title

    def test_region_works(self, mock_3d_cube):
        a = self._anomaly(mock_3d_cube)
        fig, ax = plot_spatial_anomaly(
            a["data"], a["lons"], a["lats"], region="Europe",
        )
        assert isinstance(ax.projection, ccrs.PlateCarree)


# ===================================================================
# TestCombineFields
# ===================================================================

class TestCombineFields:
    """Tests for combine_fields()."""

    @pytest.fixture
    def field_a(self, mock_3d_cube):
        return extract_map_field(mock_3d_cube, time=1900)

    @pytest.fixture
    def field_b(self, mock_3d_cube):
        return extract_map_field(mock_3d_cube, time=1901)

    def test_sum(self, field_a, field_b):
        result = combine_fields([field_a, field_b])
        expected = field_a["data"] + field_b["data"]
        np.testing.assert_allclose(result["data"], expected)
        assert "+" in result["name"]

    def test_mean(self, field_a, field_b):
        result = combine_fields([field_a, field_b], operation="mean")
        expected = (field_a["data"] + field_b["data"]) / 2
        np.testing.assert_allclose(result["data"], expected)

    def test_subtract(self, field_a, field_b):
        result = combine_fields([field_a, field_b], operation="subtract")
        expected = field_a["data"] - field_b["data"]
        np.testing.assert_allclose(result["data"], expected)
        assert "\u2212" in result["name"]

    def test_multiply(self, field_a, field_b):
        result = combine_fields([field_a, field_b], operation="multiply")
        expected = field_a["data"] * field_b["data"]
        np.testing.assert_allclose(result["data"], expected)

    def test_divide(self, field_a, field_b):
        result = combine_fields([field_a, field_b], operation="divide")
        expected = field_a["data"] / field_b["data"]
        np.testing.assert_allclose(result["data"], expected)

    def test_custom_name_and_units(self, field_a, field_b):
        result = combine_fields(
            [field_a, field_b],
            operation="divide",
            name="ratio",
            units="1",
        )
        assert result["name"] == "ratio"
        assert result["units"] == "1"

    def test_inherits_units_from_first(self, field_a, field_b):
        result = combine_fields([field_a, field_b])
        assert result["units"] == field_a["units"]

    def test_inherits_year_from_first(self, field_a, field_b):
        result = combine_fields([field_a, field_b])
        assert result["year"] == field_a["year"]

    def test_returns_correct_keys(self, field_a, field_b):
        result = combine_fields([field_a, field_b])
        expected_keys = {"data", "lons", "lats", "name", "units", "year"}
        assert set(result.keys()) == expected_keys

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            combine_fields([])

    def test_unknown_operation_raises(self, field_a):
        with pytest.raises(ValueError, match="Unknown operation"):
            combine_fields([field_a], operation="power")

    def test_binary_op_with_wrong_count_raises(self, field_a):
        with pytest.raises(ValueError, match="exactly 2 fields"):
            combine_fields([field_a], operation="subtract")

    def test_grid_mismatch_raises(self, field_a):
        mismatched = {
            "data": np.zeros((5, 5)),
            "lons": np.arange(5, dtype=float),
            "lats": np.arange(5, dtype=float),
            "name": "other",
            "units": "K",
            "year": 1900,
        }
        with pytest.raises(ValueError, match="Grid mismatch"):
            combine_fields([field_a, mismatched])

    def test_sum_three_fields(self, field_a, field_b, mock_3d_cube):
        field_c = extract_map_field(mock_3d_cube, time=1902)
        result = combine_fields([field_a, field_b, field_c])
        expected = field_a["data"] + field_b["data"] + field_c["data"]
        np.testing.assert_allclose(result["data"], expected)


# ===================================================================
# TestUnitConversion
# ===================================================================

class TestUnitConversion:
    """Tests for optional unit conversion via the `variable` parameter."""

    def test_variable_converts_units(self, mock_2d_cube):
        """Pass variable='GPP'; data should be multiplied by conversion_factor."""
        from utils_cmip7.config import CANONICAL_VARIABLES
        factor = CANONICAL_VARIABLES["GPP"]["conversion_factor"]
        raw = extract_map_field(mock_2d_cube)
        converted = extract_map_field(mock_2d_cube, variable="GPP")
        np.testing.assert_allclose(converted["data"], raw["data"] * factor)
        assert converted["units"] == "PgC/yr"
        assert converted["name"] == "GPP"

    def test_no_variable_no_conversion(self, mock_2d_cube):
        """Default behaviour: no conversion applied."""
        result = extract_map_field(mock_2d_cube)
        assert result["units"] == "K"
        assert result["name"] == "air_temperature"

    def test_variable_alias_raises(self, mock_2d_cube):
        """Pass variable='VegCarb' (removed alias); should raise ValueError."""
        with pytest.raises(ValueError, match="removed in v0.4.0"):
            extract_map_field(mock_2d_cube, variable="VegCarb")

    def test_variable_unknown_raises(self, mock_2d_cube):
        """Unknown variable name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown variable"):
            extract_map_field(mock_2d_cube, variable="NotAVariable")

    def test_anomaly_variable_converts(self, mock_3d_cube):
        """Anomaly with variable param should convert both slices."""
        from utils_cmip7.config import CANONICAL_VARIABLES
        factor = CANONICAL_VARIABLES["GPP"]["conversion_factor"]
        raw = extract_anomaly_field(mock_3d_cube)
        converted = extract_anomaly_field(mock_3d_cube, variable="GPP")
        np.testing.assert_allclose(converted["data"], raw["data"] * factor)
        assert converted["units"] == "PgC/yr"
        assert converted["name"] == "GPP"

    def test_anomaly_no_variable_no_conversion(self, mock_3d_cube):
        """Default anomaly behaviour unchanged."""
        result = extract_anomaly_field(mock_3d_cube)
        assert result["units"] == "K"
        assert result["name"] == "air_temperature"

    def test_anomaly_variable_title(self, mock_3d_cube):
        """Anomaly title should use canonical name when variable is set."""
        result = extract_anomaly_field(mock_3d_cube, variable="GPP")
        assert "GPP" in result["title"]
        assert "anomaly" in result["title"]

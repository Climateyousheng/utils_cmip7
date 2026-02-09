"""
Tests for utils_cmip7.io.extract — STASH code extraction with canonical name resolution.
"""

import numpy as np
import pytest

iris = pytest.importorskip("iris")
import iris.cube  # noqa: E402
import iris.coords  # noqa: E402

from utils_cmip7.io.extract import (  # noqa: E402
    try_extract,
    _msi_from_stash_obj,
    _msi_from_numeric_stash_code,
    _msi_from_any_attr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cube_with_stash(name, stash_code):
    """Create a minimal cube with a numeric stash_code attribute."""
    lat = iris.coords.DimCoord(
        np.linspace(-90, 90, 5), standard_name="latitude", units="degrees",
    )
    lon = iris.coords.DimCoord(
        np.linspace(-180, 180, 10), standard_name="longitude", units="degrees",
    )
    data = np.random.default_rng(42).random((5, 10))
    cube = iris.cube.Cube(
        data,
        long_name=name,
        dim_coords_and_dims=[(lat, 0), (lon, 1)],
        units="kg m-2",
        attributes={"stash_code": stash_code},
    )
    return cube


def _make_cubelist(*cubes):
    """Build an iris CubeList from cubes."""
    return iris.cube.CubeList(list(cubes))


# ---------------------------------------------------------------------------
# MSI conversion helpers
# ---------------------------------------------------------------------------

class TestMsiFromNumericStashCode:

    def test_atmosphere_code(self):
        assert _msi_from_numeric_stash_code(3261) == "m01s03i261"

    def test_ocean_code(self):
        assert _msi_from_numeric_stash_code(30249) == "m02s30i249"

    def test_none_returns_none(self):
        assert _msi_from_numeric_stash_code(None) is None

    def test_non_numeric_returns_none(self):
        assert _msi_from_numeric_stash_code("abc") is None


class TestMsiFromAnyAttr:

    def test_numeric_stash_code(self):
        assert _msi_from_any_attr({"stash_code": 3261}) == "m01s03i261"

    def test_empty_dict(self):
        assert _msi_from_any_attr({}) is None

    def test_none(self):
        assert _msi_from_any_attr(None) is None


# ---------------------------------------------------------------------------
# Canonical name resolution in try_extract
# ---------------------------------------------------------------------------

class TestTryExtractCanonicalNames:
    """Tests that try_extract resolves canonical names and aliases."""

    @pytest.fixture
    def cv_cube(self):
        """Cube matching CVeg stash_code (m01s19i002 = 19002)."""
        return _make_cube_with_stash("vegetation_carbon_content", 19002)

    @pytest.fixture
    def gpp_cube(self):
        """Cube matching GPP stash_code (m01s03i261 = 3261)."""
        return _make_cube_with_stash("gross_primary_productivity", 3261)

    @pytest.fixture
    def rh_cube(self):
        """Cube matching Rh stash_code (m01s03i293 = 3293)."""
        return _make_cube_with_stash("heterotrophic_respiration", 3293)

    def test_canonical_name_resolves(self, cv_cube):
        """try_extract(cubes, 'CVeg') should find the cv cube."""
        cubes = _make_cubelist(cv_cube)
        result = try_extract(cubes, "CVeg")
        assert result is not None
        assert len(result) == 1

    def test_alias_resolves(self, cv_cube):
        """try_extract(cubes, 'VegCarb') should resolve via CVeg aliases."""
        cubes = _make_cubelist(cv_cube)
        result = try_extract(cubes, "VegCarb")
        assert result is not None
        assert len(result) == 1

    def test_another_alias_resolves(self, rh_cube):
        """try_extract(cubes, 'soilResp') should resolve via Rh aliases."""
        cubes = _make_cubelist(rh_cube)
        result = try_extract(cubes, "soilResp")
        assert result is not None
        assert len(result) == 1

    def test_stash_name_still_works(self, cv_cube):
        """try_extract(cubes, 'cv') should still work via stash_lookup_func."""
        from utils_cmip7.io.stash import stash
        cubes = _make_cubelist(cv_cube)
        result = try_extract(cubes, "cv", stash_lookup_func=stash)
        assert result is not None
        assert len(result) == 1

    def test_msi_string_still_works(self, gpp_cube):
        """try_extract(cubes, 'm01s03i261') should still work."""
        cubes = _make_cubelist(gpp_cube)
        result = try_extract(cubes, "m01s03i261")
        assert result is not None
        assert len(result) == 1

    def test_numeric_code_still_works(self, gpp_cube):
        """try_extract(cubes, 3261) should still work."""
        cubes = _make_cubelist(gpp_cube)
        result = try_extract(cubes, 3261)
        assert result is not None
        assert len(result) == 1

    def test_unrecognised_name_returns_empty(self, gpp_cube):
        """Unknown canonical name should return empty CubeList."""
        cubes = _make_cubelist(gpp_cube)
        result = try_extract(cubes, "UnknownVariable")
        assert result is None or len(result) == 0

    def test_canonical_name_among_multiple_cubes(self, cv_cube, gpp_cube, rh_cube):
        """Canonical name should pick the correct cube from a mixed CubeList."""
        cubes = _make_cubelist(cv_cube, gpp_cube, rh_cube)
        result = try_extract(cubes, "GPP")
        assert result is not None
        assert len(result) == 1
        assert result[0].long_name == "gross_primary_productivity"

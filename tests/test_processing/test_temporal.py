"""
Test temporal aggregation functions.

Tests for src/utils_cmip7/processing/temporal.py
"""
import pytest
import numpy as np
import pandas as pd


class TestMergeMonthlyResults:
    """Test merge_monthly_results() function."""

    def test_basic_merge_two_results(self):
        """Test merging two monthly results into annual means."""
        from utils_cmip7.processing.temporal import merge_monthly_results

        # Create mock results with fractional years
        result1 = {
            "years": [1850.0, 1850.083, 1850.167],  # Jan, Feb, Mar 1850
            "data": [10.0, 12.0, 14.0]
        }
        result2 = {
            "years": [1850.25, 1850.333, 1850.417],  # Apr, May, Jun 1850
            "data": [16.0, 18.0, 20.0]
        }

        merged = merge_monthly_results([result1, result2], require_full_year=False)

        # Should have year 1850
        assert 1850 in merged['years']
        # Data should be mean of all months
        assert merged['data'].shape == merged['years'].shape

    def test_merge_full_year_requirement(self):
        """Test that require_full_year filters incomplete years."""
        from utils_cmip7.processing.temporal import merge_monthly_results

        # Create complete year 1850 and incomplete year 1851
        fractional_years = [1850 + i/12 for i in range(12)]  # All 12 months of 1850
        fractional_years += [1851.0, 1851.083]  # Only Jan-Feb 1851

        result = {
            "years": fractional_years,
            "data": [float(i) for i in range(14)]
        }

        # Without requirement: should have both years
        merged_all = merge_monthly_results([result], require_full_year=False)
        assert 1850 in merged_all['years']
        assert 1851 in merged_all['years']

        # With requirement: should only have 1850
        merged_full = merge_monthly_results([result], require_full_year=True)
        assert 1850 in merged_full['years']
        assert 1851 not in merged_full['years']

    def test_merge_handles_duplicates(self):
        """Test that duplicate (year, month) entries are averaged."""
        from utils_cmip7.processing.temporal import merge_monthly_results

        # Two results with overlapping month (Jan 1850)
        result1 = {
            "years": [1850.0, 1850.083],  # Jan, Feb 1850
            "data": [10.0, 12.0]
        }
        result2 = {
            "years": [1850.0, 1850.167],  # Jan (duplicate), Mar 1850
            "data": [20.0, 14.0]
        }

        merged = merge_monthly_results([result1, result2], require_full_year=False)

        # Should average the duplicate Jan values: (10 + 20) / 2 = 15
        # Then compute annual mean: (15 + 12 + 14) / 3 = 13.67
        assert 1850 in merged['years']
        # Annual mean should be around 13.67
        year_idx = np.where(merged['years'] == 1850)[0][0]
        assert merged['data'][year_idx] == pytest.approx(13.67, rel=0.01)

    def test_merge_multiple_years(self):
        """Test merging results spanning multiple years."""
        from utils_cmip7.processing.temporal import merge_monthly_results

        # Create data for years 1850-1852
        years_1850 = [1850 + i/12 for i in range(12)]
        years_1851 = [1851 + i/12 for i in range(12)]
        years_1852 = [1852 + i/12 for i in range(12)]

        result = {
            "years": years_1850 + years_1851 + years_1852,
            "data": [float(i) for i in range(36)]
        }

        merged = merge_monthly_results([result], require_full_year=True)

        # Should have all three years
        assert len(merged['years']) == 3
        assert 1850 in merged['years']
        assert 1851 in merged['years']
        assert 1852 in merged['years']

    def test_merge_empty_results(self):
        """Test merging empty results list."""
        from utils_cmip7.processing.temporal import merge_monthly_results

        merged = merge_monthly_results([], require_full_year=False)

        # Should return empty arrays
        assert len(merged['years']) == 0
        assert len(merged['data']) == 0

    def test_fractional_year_edge_cases(self):
        """Test handling of edge cases in fractional year conversion."""
        from utils_cmip7.processing.temporal import merge_monthly_results

        # Test boundary months: month 0 and month 13 should be clamped
        result = {
            "years": [1850.0, 1850.99],  # Jan and Dec 1850
            "data": [10.0, 20.0]
        }

        merged = merge_monthly_results([result], require_full_year=False)

        # Should successfully merge without errors
        assert 1850 in merged['years']


class TestComputeMonthlyMeanMocked:
    """Test compute_monthly_mean() with mocked dependencies."""

    @pytest.fixture
    def mock_cube(self):
        """Create a mock cube for testing."""
        try:
            import iris
            from iris.coords import DimCoord, AuxCoord
            from iris.cube import Cube
            import cf_units

            # Create simple time series data
            data = np.array([10.0, 12.0, 14.0])

            # Time coordinate (3 months)
            time_points = np.array([0, 30, 60], dtype=np.float64)
            time_unit = cf_units.Unit("days since 1850-01-01", calendar="360_day")
            time_coord = DimCoord(
                time_points,
                standard_name='time',
                units=time_unit
            )

            cube = Cube(
                data,
                long_name='Gross Primary Productivity',
                units='kg m-2 s-1',
                dim_coords_and_dims=[(time_coord, 0)]
            )

            return cube
        except ImportError:
            pytest.skip("iris not available")

    def test_compute_monthly_mean_basic(self, mock_cube, monkeypatch):
        """Test basic monthly mean computation."""
        from utils_cmip7.processing.temporal import compute_monthly_mean
        import iris

        # Mock global_total_pgC to return a simple cube
        def mock_global_total(cube, var):
            # Return a 1D time series cube
            return cube

        monkeypatch.setattr(
            'utils_cmip7.processing.temporal.global_total_pgC',
            mock_global_total
        )

        result = compute_monthly_mean(mock_cube, 'GPP')

        # Should have years and data keys
        assert 'years' in result
        assert 'data' in result
        assert 'name' in result
        assert 'units' in result

        # Years should be fractional
        assert len(result['years']) == 3
        # First month should be 1850.0 (January)
        assert result['years'][0] == pytest.approx(1850.0, abs=0.01)

    def test_compute_monthly_mean_no_time_coord_raises(self, monkeypatch):
        """Test that missing time coordinate raises ValueError."""
        from utils_cmip7.processing.temporal import compute_monthly_mean
        try:
            import iris
            from iris.cube import Cube

            # Create cube without time coordinate
            cube_no_time = Cube(np.array([10.0, 12.0, 14.0]))

            def mock_global_total(cube, var):
                return cube_no_time

            monkeypatch.setattr(
                'utils_cmip7.processing.temporal.global_total_pgC',
                mock_global_total
            )

            with pytest.raises(ValueError, match="No valid time coordinate"):
                compute_monthly_mean(cube_no_time, 'GPP')
        except ImportError:
            pytest.skip("iris not available")

    def test_compute_monthly_mean_handles_duplicates(self, monkeypatch):
        """Test that duplicate (year, month) entries are averaged."""
        from utils_cmip7.processing.temporal import compute_monthly_mean
        try:
            import iris
            from iris.coords import AuxCoord
            from iris.cube import Cube
            import cf_units

            # Create data with duplicate months (use slightly different time points to avoid DimCoord error)
            data = np.array([10.0, 20.0, 15.0])  # Jan (early), Jan (late), Feb

            time_points = np.array([0, 5, 30], dtype=np.float64)  # Same month, different days
            time_unit = cf_units.Unit("days since 1850-01-01", calendar="360_day")
            # Use AuxCoord instead of DimCoord to allow non-monotonic values
            time_coord = AuxCoord(
                time_points,
                standard_name='time',
                units=time_unit
            )

            cube = Cube(
                data,
                long_name='GPP',
                units='kg m-2 s-1',
                aux_coords_and_dims=[(time_coord, 0)]
            )

            def mock_global_total(cube_in, var):
                return cube_in

            monkeypatch.setattr(
                'utils_cmip7.processing.temporal.global_total_pgC',
                mock_global_total
            )

            result = compute_monthly_mean(cube, 'GPP')

            # Should have 2 unique months after grouping
            assert len(result['years']) == 2
        except ImportError:
            pytest.skip("iris not available")


class TestComputeAnnualMeanMocked:
    """Test compute_annual_mean() with mocked dependencies."""

    @pytest.fixture
    def mock_cube_monthly(self):
        """Create a mock cube with monthly data."""
        try:
            import iris
            from iris.coords import DimCoord
            from iris.cube import Cube
            import cf_units

            # Create 24 months of data (2 years)
            data = np.arange(1, 25, dtype=np.float64)

            # Time coordinate (24 months)
            time_points = np.arange(0, 720, 30, dtype=np.float64)  # 360-day calendar
            time_unit = cf_units.Unit("days since 1850-01-01", calendar="360_day")
            time_coord = DimCoord(
                time_points,
                standard_name='time',
                units=time_unit
            )

            cube = Cube(
                data,
                long_name='GPP',
                units='kg m-2 s-1',
                dim_coords_and_dims=[(time_coord, 0)]
            )

            return cube
        except ImportError:
            pytest.skip("iris not available")

    def test_compute_annual_mean_basic(self, mock_cube_monthly, monkeypatch):
        """Test basic annual mean computation."""
        from utils_cmip7.processing.temporal import compute_annual_mean

        def mock_global_total(cube, var):
            return cube

        monkeypatch.setattr(
            'utils_cmip7.processing.temporal.global_total_pgC',
            mock_global_total
        )

        result = compute_annual_mean(mock_cube_monthly, 'GPP')

        # Should have 2 years
        assert len(result['years']) == 2
        assert 1850 in result['years']
        assert 1851 in result['years']

        # Should have corresponding data
        assert len(result['data']) == 2

    def test_compute_annual_mean_others_variable(self, mock_cube_monthly, monkeypatch):
        """Test that var='Others' uses global_mean_pgC instead."""
        from utils_cmip7.processing.temporal import compute_annual_mean

        global_mean_called = False
        global_total_called = False

        def mock_global_mean(cube, var):
            nonlocal global_mean_called
            global_mean_called = True
            return cube

        def mock_global_total(cube, var):
            nonlocal global_total_called
            global_total_called = True
            return cube

        monkeypatch.setattr(
            'utils_cmip7.processing.temporal.global_mean_pgC',
            mock_global_mean
        )
        monkeypatch.setattr(
            'utils_cmip7.processing.temporal.global_total_pgC',
            mock_global_total
        )

        # Call with var='Others'
        result = compute_annual_mean(mock_cube_monthly, 'Others')

        # Should have called global_mean_pgC, not global_total_pgC
        assert global_mean_called is True
        assert global_total_called is False

    def test_compute_annual_mean_no_time_coord_raises(self, monkeypatch):
        """Test that missing time coordinate raises ValueError."""
        from utils_cmip7.processing.temporal import compute_annual_mean
        try:
            import iris
            from iris.cube import Cube

            cube_no_time = Cube(np.array([10.0, 12.0]))

            def mock_global_total(cube, var):
                return cube_no_time

            monkeypatch.setattr(
                'utils_cmip7.processing.temporal.global_total_pgC',
                mock_global_total
            )

            with pytest.raises(ValueError, match="No valid time coordinate"):
                compute_annual_mean(cube_no_time, 'GPP')
        except ImportError:
            pytest.skip("iris not available")

    def test_compute_annual_mean_scalar_data(self, monkeypatch):
        """Test handling of scalar (0D) data."""
        from utils_cmip7.processing.temporal import compute_annual_mean
        try:
            import iris
            from iris.coords import DimCoord
            from iris.cube import Cube
            import cf_units

            # Create time series cube, but mock global_total to return scalar
            data = np.array([10.0, 12.0, 14.0])

            time_points = np.array([0, 30, 60], dtype=np.float64)
            time_unit = cf_units.Unit("days since 1850-01-01", calendar="360_day")
            time_coord = DimCoord(
                time_points,
                standard_name='time',
                units=time_unit
            )

            cube = Cube(
                data,
                long_name='GPP',
                units='kg m-2 s-1',
                dim_coords_and_dims=[(time_coord, 0)]
            )

            def mock_global_total(cube_in, var):
                # Return scalar cube (collapsed spatial dims but time coord preserved)
                scalar_cube = Cube(
                    np.array(42.0),  # Scalar data
                    long_name='GPP',
                    units='kg m-2 s-1'
                )
                # Add time as aux coord
                scalar_cube.add_aux_coord(time_coord[0])  # Just first time point
                return scalar_cube

            monkeypatch.setattr(
                'utils_cmip7.processing.temporal.global_total_pgC',
                mock_global_total
            )

            result = compute_annual_mean(cube, 'GPP')

            # Should handle scalar data
            assert len(result['years']) >= 1
            assert result['data'][0] == 42.0
        except ImportError:
            pytest.skip("iris not available")

    def test_compute_annual_mean_handles_nans(self, monkeypatch):
        """Test that np.nanmean is used to handle missing data."""
        from utils_cmip7.processing.temporal import compute_annual_mean
        try:
            import iris
            from iris.coords import DimCoord
            from iris.cube import Cube
            import cf_units

            # Create data with NaNs
            data = np.array([10.0, np.nan, 12.0, 14.0])

            time_points = np.array([0, 30, 60, 90], dtype=np.float64)
            time_unit = cf_units.Unit("days since 1850-01-01", calendar="360_day")
            time_coord = DimCoord(
                time_points,
                standard_name='time',
                units=time_unit
            )

            cube = Cube(
                data,
                long_name='GPP',
                units='kg m-2 s-1',
                dim_coords_and_dims=[(time_coord, 0)]
            )

            def mock_global_total(cube_in, var):
                return cube_in

            monkeypatch.setattr(
                'utils_cmip7.processing.temporal.global_total_pgC',
                mock_global_total
            )

            result = compute_annual_mean(cube, 'GPP')

            # Should compute annual mean ignoring NaN
            # Mean of [10, 12, 14] = 12.0
            assert result['data'][0] == pytest.approx(12.0)
        except ImportError:
            pytest.skip("iris not available")

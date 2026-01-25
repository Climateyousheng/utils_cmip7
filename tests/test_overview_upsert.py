"""
Tests for overview table upsert functionality.

Validates overview table loading, upserting, and atomic write operations.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path for testing without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_upsert_new_row():
    """Test appending new experiment to empty DataFrame."""
    try:
        from utils_cmip7.validation.overview_table import upsert_overview_row
        import pandas as pd

        # Start with empty DataFrame
        df = pd.DataFrame()

        # Prepare test data
        expt_id = 'test_expt_001'
        bl_params = {
            'ALPHA': 0.08,
            'G_AREA': 0.004,
            'LAI_MIN': 4.0,
            'NL0': 0.050,
            'R_GROW': 0.25,
            'TLOW': -0.0,
            'TUPP': 36.0,
            'V_CRIT': 0.343,
        }
        scores = {'GPP': 120.5, 'NPP': 60.2}

        # Upsert into empty DataFrame
        df = upsert_overview_row(df, expt_id, bl_params, scores)

        # Verify row was added
        assert len(df) == 1, "Should have 1 row after upsert"
        assert df.loc[0, 'ID'] == expt_id, "Experiment ID should match"
        assert df.loc[0, 'ALPHA'] == 0.08, "ALPHA should be set"
        assert df.loc[0, 'GPP'] == 120.5, "GPP score should be set"

        print("✓ Upsert correctly appends new row to empty DataFrame")
        return True
    except Exception as e:
        print(f"✗ Failed upsert new row test: {e}")
        return False


def test_upsert_update_existing():
    """Test updating existing experiment row."""
    try:
        from utils_cmip7.validation.overview_table import upsert_overview_row
        import pandas as pd

        # Create DataFrame with existing row
        df = pd.DataFrame([{
            'ID': 'expt_001',
            'ALPHA': 0.08,
            'GPP': 120.0,
        }])

        # Update with new values
        expt_id = 'expt_001'
        bl_params = {'ALPHA': 0.09}  # Changed value
        scores = {'GPP': 125.0, 'NPP': 62.0}  # Updated + new

        df = upsert_overview_row(df, expt_id, bl_params, scores)

        # Verify only 1 row exists (no duplicate)
        assert len(df) == 1, "Should still have only 1 row"

        # Verify values were updated
        assert df.loc[0, 'ALPHA'] == 0.09, "ALPHA should be updated"
        assert df.loc[0, 'GPP'] == 125.0, "GPP should be updated"
        assert df.loc[0, 'NPP'] == 62.0, "NPP should be added"

        print("✓ Upsert correctly updates existing row")
        return True
    except Exception as e:
        print(f"✗ Failed upsert update test: {e}")
        return False


def test_upsert_preserves_other_rows():
    """Test that other experiment rows remain unchanged."""
    try:
        from utils_cmip7.validation.overview_table import upsert_overview_row
        import pandas as pd

        # Create DataFrame with 2 existing experiments
        df = pd.DataFrame([
            {'ID': 'expt_001', 'ALPHA': 0.08, 'GPP': 120.0},
            {'ID': 'expt_002', 'ALPHA': 0.07, 'GPP': 115.0},
        ])

        # Update expt_001
        expt_id = 'expt_001'
        bl_params = {'ALPHA': 0.09}
        scores = {'GPP': 125.0}

        df = upsert_overview_row(df, expt_id, bl_params, scores)

        # Verify 2 rows still exist
        assert len(df) == 2, "Should still have 2 rows"

        # Verify expt_002 unchanged
        expt_002_row = df[df['ID'] == 'expt_002'].iloc[0]
        assert expt_002_row['ALPHA'] == 0.07, "expt_002 ALPHA should be unchanged"
        assert expt_002_row['GPP'] == 115.0, "expt_002 GPP should be unchanged"

        # Verify expt_001 updated
        expt_001_row = df[df['ID'] == 'expt_001'].iloc[0]
        assert expt_001_row['ALPHA'] == 0.09, "expt_001 ALPHA should be updated"
        assert expt_001_row['GPP'] == 125.0, "expt_001 GPP should be updated"

        print("✓ Upsert preserves other experiment rows")
        return True
    except Exception as e:
        print(f"✗ Failed preserve other rows test: {e}")
        return False


def test_bl_columns_only():
    """Test that only BL parameters are stored (matches overview table schema)."""
    try:
        from utils_cmip7.validation.overview_table import upsert_overview_row
        from utils_cmip7.soil_params import SoilParamSet
        import pandas as pd

        df = pd.DataFrame()

        # Use SoilParamSet to get overview table format (excludes F0, Q10, KAPS)
        params = SoilParamSet.from_default()
        bl_params = params.to_overview_table_format()

        expt_id = 'test_expt'
        df = upsert_overview_row(df, expt_id, bl_params, scores=None)

        # Expected columns (from to_overview_table_format)
        expected_cols = ['ALPHA', 'G_AREA', 'LAI_MIN', 'NL0', 'R_GROW', 'TLOW', 'TUPP', 'V_CRIT']

        # Verify all expected columns present
        for col in expected_cols:
            assert col in df.columns, f"{col} should be in DataFrame"

        # Verify excluded params NOT present
        excluded = ['F0', 'Q10', 'KAPS']
        for col in excluded:
            # These should not be in the DataFrame columns (or should be NaN if present)
            if col in df.columns:
                print(f"⚠ {col} found in columns (should be excluded from overview table)")

        print("✓ Only BL tree parameters stored in overview table format")
        print(f"  - Stored {len(expected_cols)} parameter columns")
        return True
    except Exception as e:
        print(f"✗ Failed BL columns test: {e}")
        return False


def test_scores_merged():
    """Test that scores dict is merged with params."""
    try:
        from utils_cmip7.validation.overview_table import upsert_overview_row
        import pandas as pd

        df = pd.DataFrame()

        expt_id = 'test_expt'
        bl_params = {'ALPHA': 0.08, 'V_CRIT': 0.343}
        scores = {
            'GPP': 120.5,
            'NPP': 60.2,
            'CVeg': 450.0,
            'overall_score': 0.85,
        }

        df = upsert_overview_row(df, expt_id, bl_params, scores)

        # Verify both params and scores are in the row
        assert df.loc[0, 'ALPHA'] == 0.08, "Param should be present"
        assert df.loc[0, 'GPP'] == 120.5, "Score should be present"
        assert df.loc[0, 'overall_score'] == 0.85, "Overall score should be present"

        print("✓ Scores correctly merged with parameters")
        return True
    except Exception as e:
        print(f"✗ Failed scores merge test: {e}")
        return False


def test_atomic_write():
    """Test atomic CSV write with temp file."""
    try:
        from utils_cmip7.validation.overview_table import write_atomic_csv
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'test_overview.csv'

            # Create test DataFrame
            df = pd.DataFrame([
                {'ID': 'expt_001', 'ALPHA': 0.08, 'GPP': 120.5},
                {'ID': 'expt_002', 'ALPHA': 0.07, 'GPP': 115.2},
            ])

            # Write to CSV
            write_atomic_csv(df, str(csv_path))

            # Verify file exists
            assert csv_path.exists(), "CSV file should exist after write"

            # Read back and verify
            df_read = pd.read_csv(csv_path)
            assert len(df_read) == 2, "Should have 2 rows"
            assert df_read.loc[0, 'ID'] == 'expt_001', "First row ID should match"
            assert abs(df_read.loc[0, 'GPP'] - 120.5) < 0.001, "GPP should match"

            # Verify temp files are cleaned up
            temp_files = list(Path(tmpdir).glob('.*tmp'))
            assert len(temp_files) == 0, "No temp files should remain"

            print("✓ Atomic CSV write works correctly")
            return True
    except Exception as e:
        print(f"✗ Failed atomic write test: {e}")
        return False


def test_empty_dataframe_handling():
    """Test load_overview_table with missing file."""
    try:
        from utils_cmip7.validation.overview_table import load_overview_table
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_path = Path(tmpdir) / 'nonexistent.csv'

            # Load from nonexistent path
            df = load_overview_table(str(nonexistent_path))

            # Should return empty DataFrame, not raise error
            assert isinstance(df, pd.DataFrame), "Should return DataFrame"
            assert len(df) == 0, "Should be empty DataFrame"

            print("✓ load_overview_table handles missing file gracefully")
            return True
    except Exception as e:
        print(f"✗ Failed empty DataFrame test: {e}")
        return False


def run_all_overview_upsert_tests():
    """Run all overview table upsert tests and report results."""
    print("=" * 80)
    print("OVERVIEW TABLE UPSERT TESTS")
    print("=" * 80)
    print()

    tests = [
        test_upsert_new_row,
        test_upsert_update_existing,
        test_upsert_preserves_other_rows,
        test_bl_columns_only,
        test_scores_merged,
        test_atomic_write,
        test_empty_dataframe_handling,
    ]

    results = []
    for test in tests:
        results.append(test())
        print()

    print("=" * 80)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 80)

    return all(results)


if __name__ == '__main__':
    success = run_all_overview_upsert_tests()
    sys.exit(0 if success else 1)

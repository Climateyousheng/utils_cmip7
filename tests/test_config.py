"""
Smoke tests for configuration and validation.

Tests that configuration loads correctly and validation functions work as expected.
"""

import sys
import os
import tempfile

# Add src to path for testing without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_config_loads():
    """Test that configuration module loads without errors."""
    try:
        from utils_cmip7.config import (
            VAR_CONVERSIONS,
            RECCAP_MASK_PATH,
            RECCAP_REGIONS,
            validate_reccap_mask_path,
            get_config_info,
        )

        # Check that essential config items exist
        assert isinstance(VAR_CONVERSIONS, dict)
        assert len(VAR_CONVERSIONS) > 0
        assert isinstance(RECCAP_MASK_PATH, str)
        assert isinstance(RECCAP_REGIONS, dict)
        assert len(RECCAP_REGIONS) > 0

        print("✓ Configuration loads successfully")
        print(f"  - {len(VAR_CONVERSIONS)} unit conversions defined")
        print(f"  - {len(RECCAP_REGIONS)} RECCAP regions defined")
        print(f"  - Mask path: {RECCAP_MASK_PATH[:60]}...")
        return True
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False


def test_var_conversions_content():
    """Test that VAR_CONVERSIONS has expected entries."""
    try:
        from utils_cmip7.config import VAR_CONVERSIONS

        # Check some essential conversions exist
        essential_vars = ['GPP', 'NPP', 'Rh', 'CVeg', 'CSoil', 'precip']
        missing = [v for v in essential_vars if v not in VAR_CONVERSIONS]

        if missing:
            print(f"✗ VAR_CONVERSIONS missing essential variables: {missing}")
            return False

        print("✓ VAR_CONVERSIONS contains essential variables")
        return True
    except Exception as e:
        print(f"✗ Failed to check VAR_CONVERSIONS: {e}")
        return False


def test_reccap_regions_content():
    """Test that RECCAP_REGIONS has expected entries."""
    try:
        from utils_cmip7.config import RECCAP_REGIONS

        # Check essential regions exist
        essential_regions = ['North_America', 'Europe', 'Africa', 'East_Asia']
        present_regions = list(RECCAP_REGIONS.values())
        missing = [r for r in essential_regions if r not in present_regions]

        if missing:
            print(f"✗ RECCAP_REGIONS missing essential regions: {missing}")
            return False

        print("✓ RECCAP_REGIONS contains essential regions")
        print(f"  Regions: {', '.join(present_regions)}")
        return True
    except Exception as e:
        print(f"✗ Failed to check RECCAP_REGIONS: {e}")
        return False


def test_validate_nonexistent_path():
    """Test that validation fails gracefully for nonexistent paths."""
    try:
        from utils_cmip7.config import validate_reccap_mask_path

        # Try to validate a path that definitely doesn't exist
        nonexistent_path = "/this/path/definitely/does/not/exist/mask.nc"

        try:
            validate_reccap_mask_path(nonexistent_path)
            print("✗ Validation should have raised FileNotFoundError")
            return False
        except FileNotFoundError as e:
            # This is expected - check that error message is helpful
            error_msg = str(e)
            if "RECCAP2" in error_msg and "UTILS_CMIP7_RECCAP_MASK" in error_msg:
                print("✓ Validation correctly rejects nonexistent path with helpful error")
                return True
            else:
                print("✗ Error message not helpful enough")
                print(f"  Got: {error_msg[:100]}...")
                return False
    except Exception as e:
        print(f"✗ Failed to test validation: {e}")
        return False


def test_validate_unreadable_file():
    """Test that validation fails gracefully for unreadable files."""
    try:
        from utils_cmip7.config import validate_reccap_mask_path

        # Create a temporary file with no read permissions
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
            tmp_path = tmp.name

        try:
            # Remove read permissions
            os.chmod(tmp_path, 0o000)

            try:
                validate_reccap_mask_path(tmp_path)
                print("✗ Validation should have raised RuntimeError for unreadable file")
                return False
            except RuntimeError as e:
                error_msg = str(e)
                if "not readable" in error_msg or "permission" in error_msg.lower():
                    print("✓ Validation correctly rejects unreadable file")
                    return True
                else:
                    print("✗ Error message not clear about permission issue")
                    return False
            except FileNotFoundError:
                # On some systems, chmod 000 makes the file appear to not exist
                print("⚠ Cannot test unreadable file (system limitation)")
                return True
        finally:
            # Cleanup: restore permissions and delete
            try:
                os.chmod(tmp_path, 0o644)
                os.unlink(tmp_path)
            except:
                pass

    except Exception as e:
        print(f"✗ Failed to test unreadable file validation: {e}")
        return False


def test_var_conversions_derived_from_canonical():
    """VAR_CONVERSIONS must match CANONICAL_VARIABLES conversion factors."""
    import pytest
    from utils_cmip7.config import VAR_CONVERSIONS, CANONICAL_VARIABLES

    for name, cfg in CANONICAL_VARIABLES.items():
        assert name in VAR_CONVERSIONS
        assert VAR_CONVERSIONS[name] == pytest.approx(cfg["conversion_factor"])


def test_get_config_info():
    """Test that get_config_info runs without errors."""
    try:
        from utils_cmip7.config import get_config_info
        import io
        from contextlib import redirect_stdout

        # Capture output
        f = io.StringIO()
        with redirect_stdout(f):
            get_config_info()

        output = f.getvalue()

        # Check that output contains expected sections
        if "Configuration" in output and "RECCAP" in output:
            print("✓ get_config_info() runs successfully")
            return True
        else:
            print("✗ get_config_info() output missing expected content")
            return False
    except Exception as e:
        print(f"✗ Failed to run get_config_info(): {e}")
        return False


def run_all_config_tests():
    """Run all configuration tests and report results."""
    print("=" * 80)
    print("CONFIGURATION VALIDATION SMOKE TESTS")
    print("=" * 80)
    print()

    tests = [
        test_config_loads,
        test_var_conversions_content,
        test_reccap_regions_content,
        test_validate_nonexistent_path,
        test_validate_unreadable_file,
        test_get_config_info,
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
    success = run_all_config_tests()
    sys.exit(0 if success else 1)

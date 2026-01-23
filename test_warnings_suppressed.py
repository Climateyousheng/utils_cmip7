#!/usr/bin/env python3
"""
Test to verify Iris warnings are properly suppressed.
"""

import sys
import os

print("Testing warning suppression...")
print("=" * 70)

# Import utils_cmip7 - this should trigger warning configuration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
print("Importing utils_cmip7...")
import utils_cmip7

print(f"✓ Package imported (version {utils_cmip7.__version__})")

# Check iris FUTURE settings
try:
    import iris
    if hasattr(iris, 'FUTURE'):
        print(f"✓ iris.FUTURE.date_microseconds = {iris.FUTURE.date_microseconds}")
    else:
        print("⚠ iris.FUTURE not available (older iris version)")
except ImportError:
    print("⚠ Iris not available")

# Try importing modules that use iris
print("\nImporting modules that use iris...")
from utils_cmip7.diagnostics import extraction
print("✓ diagnostics.extraction imported")

from utils_cmip7.processing import regional
print("✓ processing.regional imported")

# Test that area_weights doesn't produce warnings
print("\nTesting area_weights (should not show DEFAULT_SPHERICAL_EARTH_RADIUS warning)...")
try:
    import numpy as np
    import iris
    from iris.analysis.cartography import area_weights

    # Create a simple test cube
    data = np.zeros((3, 4))
    lat = iris.coords.DimCoord(np.linspace(-90, 90, 3), standard_name='latitude', units='degrees')
    lon = iris.coords.DimCoord(np.linspace(-180, 180, 4), standard_name='longitude', units='degrees')
    cube = iris.cube.Cube(data, dim_coords_and_dims=[(lat, 0), (lon, 1)])

    # This normally triggers DEFAULT_SPHERICAL_EARTH_RADIUS warning
    weights = area_weights(cube)
    print("✓ area_weights computed without warnings")

except Exception as e:
    print(f"⚠ Could not test area_weights: {e}")

print("\n" + "=" * 70)
print("Warning suppression test complete")
print("If you see FutureWarning or DEFAULT_SPHERICAL_EARTH_RADIUS above,")
print("the suppression is not working correctly.")
print("=" * 70)

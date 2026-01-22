#!/usr/bin/env python3
"""Debug the conversion factor application for GPP vs NPP."""

import numpy as np
from utils_cmip7.config import VAR_CONVERSIONS

print("="*80)
print("CONVERSION FACTORS")
print("="*80)

# Check what conversion factors are being used
for var in ['GPP', 'NPP', 'S resp', 'V carb', 'S carb']:
    factor = VAR_CONVERSIONS.get(var, None)
    print(f"{var:20s}: {factor}")

print("\n" + "="*80)
print("SIMULATION: What happens to GPP vs NPP")
print("="*80)

# From the cube diagnostics:
gpp_cube_mean = 1.970317e-08  # kgC/m²/s
npp_cube_mean = 1.032187e-08  # kgC/m²/s

# Number of grid cells
n_lat = 73
n_lon = 96
n_grid_cells = n_lat * n_lon
print(f"Number of grid cells: {n_grid_cells}")

# Conversion factor
conv_factor = VAR_CONVERSIONS['GPP']
print(f"Conversion factor: {conv_factor:.6e}")

# Simulate area-weighted sum
# Assuming equal area weights for simplicity (actual weights vary by latitude)
avg_area_weight = 1.0  # normalized

# After SUM collapse over lat/lon (with unit area weights):
gpp_after_sum = gpp_cube_mean * n_grid_cells * avg_area_weight
npp_after_sum = npp_cube_mean * n_grid_cells * avg_area_weight

print(f"\nAfter spatial SUM:")
print(f"  GPP: {gpp_after_sum:.6e}")
print(f"  NPP: {npp_after_sum:.6e}")

# After unit conversion
gpp_after_conv = gpp_after_sum * conv_factor
npp_after_conv = npp_after_sum * conv_factor

print(f"\nAfter unit conversion:")
print(f"  GPP: {gpp_after_conv:.2e} PgC/yr")
print(f"  NPP: {npp_after_conv:.2e} PgC/yr")

# Expected values
print(f"\nExpected (from obs):")
print(f"  GPP: ~123 PgC/yr")
print(f"  NPP: ~56 PgC/yr")

# Actual results from your run
print(f"\nActual results from your run:")
print(f"  GPP: 23,327,855,971.22 PgC/yr")
print(f"  NPP: 68.55 PgC/yr")

# Calculate discrepancy
print(f"\n" + "="*80)
print("DISCREPANCY ANALYSIS")
print("="*80)

expected_gpp = 123
actual_gpp = 23327855971.22
factor_wrong = actual_gpp / expected_gpp
print(f"GPP is {factor_wrong:.2e} times too large")

# What if generic dimension (size 1) is being incorrectly expanded?
print(f"\nIf generic dimension is causing issues:")
print(f"  Generic size: 1")
print(f"  If accidentally counted as 9 (num PFTs): {actual_gpp / 9:.2e}")

# Check if there's a cube metadata issue
print(f"\n" + "="*80)
print("HYPOTHESIS")
print("="*80)
print("The 'generic' dimension might be handled differently for GPP vs NPP")
print("Or there might be a PFT-specific extraction issue")

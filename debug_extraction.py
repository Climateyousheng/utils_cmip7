#!/usr/bin/env python3
"""Debug the extraction process for GPP to find where it goes wrong."""

import iris
import numpy as np
import os
import glob
from utils_cmip7.io import stash, try_extract
from utils_cmip7.processing.regional import compute_regional_annual_mean

# Load annual mean files
expt = 'xqhuc'
base_dir = f'~/annual_mean/{expt}/'
base_dir = os.path.expanduser(base_dir)
filenames = glob.glob(os.path.join(base_dir, "**/*.nc"), recursive=True)

cubes = iris.load(filenames)

# Extract GPP and NPP
gpp = try_extract(cubes, 'gpp', stash_lookup_func=stash)
npp = try_extract(cubes, 'npp', stash_lookup_func=stash)

print("="*80)
print("EXTRACTION RESULTS")
print("="*80)
print(f"GPP CubeList length: {len(gpp)}")
print(f"NPP CubeList length: {len(npp)}")

# Process GPP
print("\n" + "="*80)
print("PROCESSING GPP WITH COMPUTE_REGIONAL_ANNUAL_MEAN")
print("="*80)

gpp_cube = gpp[0]
print(f"Input cube shape: {gpp_cube.shape}")
print(f"Input cube has 'generic' coord: {gpp_cube.coords('generic')}")
print(f"Generic coord values: {gpp_cube.coord('generic').points if gpp_cube.coords('generic') else 'N/A'}")

# Check if there are multiple generic values
if gpp_cube.coords('generic'):
    generic_coord = gpp_cube.coord('generic')
    print(f"Generic coord shape: {generic_coord.shape}")
    print(f"Generic coord points: {generic_coord.points}")

# Now process with the actual function
print("\nCalling compute_regional_annual_mean(gpp_cube, 'GPP', 'global')...")
result_gpp = compute_regional_annual_mean(gpp_cube, 'GPP', 'global')
print(f"Result years: {result_gpp['years'][:5]}...{result_gpp['years'][-2:]}")
print(f"Result data: {result_gpp['data'][:5]}")
print(f"Result data mean: {np.mean(result_gpp['data']):.2f} {result_gpp['units']}")

print("\n" + "="*80)
print("PROCESSING NPP WITH COMPUTE_REGIONAL_ANNUAL_MEAN")
print("="*80)

npp_cube = npp[0]
print(f"Input cube shape: {npp_cube.shape}")
print(f"Input cube has 'generic' coord: {npp_cube.coords('generic')}")

# Now process with the actual function
print("\nCalling compute_regional_annual_mean(npp_cube, 'NPP', 'global')...")
result_npp = compute_regional_annual_mean(npp_cube, 'NPP', 'global')
print(f"Result years: {result_npp['years'][:5]}...{result_npp['years'][-2:]}")
print(f"Result data: {result_npp['data'][:5]}")
print(f"Result data mean: {np.mean(result_npp['data']):.2f} {result_npp['units']}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"GPP mean: {np.mean(result_gpp['data']):.2f} {result_gpp['units']}")
print(f"NPP mean: {np.mean(result_npp['data']):.2f} {result_npp['units']}")
print(f"Ratio: {np.mean(result_gpp['data']) / np.mean(result_npp['data']):.2f}")
print(f"Expected ratio: ~2.0")

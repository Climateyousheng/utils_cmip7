#!/usr/bin/env python3
"""Quick diagnostic to compare GPP vs NPP cube structure."""

import iris
import numpy as np
from utils_cmip7.io import stash, try_extract

# Load annual mean files
expt = 'xqhuc'
base_dir = f'~/annual_mean/{expt}/'
import os
base_dir = os.path.expanduser(base_dir)
import glob
filenames = glob.glob(os.path.join(base_dir, "**/*.nc"), recursive=True)

print(f"Loading cubes from: {base_dir}")
print(f"Files found: {len(filenames)}")
for f in filenames:
    print(f"  - {os.path.basename(f)}")

cubes = iris.load(filenames)
print(f"\nTotal cubes loaded: {len(cubes)}")

# Extract GPP and NPP
gpp = try_extract(cubes, 'gpp', stash_lookup_func=stash)
npp = try_extract(cubes, 'npp', stash_lookup_func=stash)

print("\n" + "="*80)
print("GPP CUBE STRUCTURE")
print("="*80)
if gpp:
    gpp_cube = gpp[0]
    print(f"Name: {gpp_cube.name()}")
    print(f"Shape: {gpp_cube.shape}")
    print(f"Dimensions: {[coord.name() for coord in gpp_cube.coords(dim_coords=True)]}")
    print(f"Units: {gpp_cube.units}")
    print(f"Data type: {gpp_cube.data.dtype}")
    print(f"Data min/max/mean: {np.min(gpp_cube.data):.6e} / {np.max(gpp_cube.data):.6e} / {np.mean(gpp_cube.data):.6e}")
    print(f"\nAll coordinates:")
    for coord in gpp_cube.coords():
        print(f"  - {coord.name()}: shape={coord.shape}, dims={gpp_cube.coord_dims(coord)}")
else:
    print("GPP NOT FOUND!")

print("\n" + "="*80)
print("NPP CUBE STRUCTURE")
print("="*80)
if npp:
    npp_cube = npp[0]
    print(f"Name: {npp_cube.name()}")
    print(f"Shape: {npp_cube.shape}")
    print(f"Dimensions: {[coord.name() for coord in npp_cube.coords(dim_coords=True)]}")
    print(f"Units: {npp_cube.units}")
    print(f"Data type: {npp_cube.data.dtype}")
    print(f"Data min/max/mean: {np.min(npp_cube.data):.6e} / {np.max(npp_cube.data):.6e} / {np.mean(npp_cube.data):.6e}")
    print(f"\nAll coordinates:")
    for coord in npp_cube.coords():
        print(f"  - {coord.name()}: shape={coord.shape}, dims={npp_cube.coord_dims(coord)}")
else:
    print("NPP NOT FOUND!")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
if gpp and npp:
    print(f"GPP shape: {gpp_cube.shape}")
    print(f"NPP shape: {npp_cube.shape}")
    print(f"Shapes match: {gpp_cube.shape == npp_cube.shape}")
    print(f"\nGPP/NPP ratio (mean): {np.mean(gpp_cube.data) / np.mean(npp_cube.data):.2f}")
    print(f"Expected GPP/NPP ratio: ~2.0-2.5 (typical for terrestrial ecosystems)")

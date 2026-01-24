#!/usr/bin/env python3
"""
Extract IGBP vegetation fraction regional means from raw NetCDF file.

Reads qrparm.veg.frac_igbp.pp.hadcm3bl.nc and computes regional means
for all RECCAP2 regions + global using existing extraction infrastructure.

Saves output to src/utils_cmip7/data/obs/igbp_regional_means.csv

Usage:
    python scripts/extract_igbp_regional_means.py
"""

import os
import sys
import warnings
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd

# Configure Iris before import
try:
    import iris
    iris.FUTURE.date_microseconds = True
except AttributeError:
    import iris

warnings.filterwarnings('ignore', message='.*date precision.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*DEFAULT_SPHERICAL_EARTH_RADIUS.*')

from iris import Constraint

from utils_cmip7.validation.veg_fractions import PFT_MAPPING, get_obs_dir
from utils_cmip7.processing.regional import load_reccap_mask, region_mask
from utils_cmip7.config import RECCAP_REGIONS

import iris.analysis
from iris.analysis.cartography import area_weights


def extract_igbp_regional_means():
    """
    Extract IGBP regional means for all PFTs and regions.

    Returns
    -------
    pd.DataFrame
        DataFrame with PFT columns and region rows
    """
    # Get IGBP file path
    obs_dir = get_obs_dir()
    igbp_file = os.path.join(obs_dir, 'qrparm.veg.frac_igbp.pp.hadcm3bl.nc')

    if not os.path.exists(igbp_file):
        raise FileNotFoundError(f"IGBP file not found: {igbp_file}")

    print(f"Loading IGBP data from: {igbp_file}")

    # Load the cube
    cube = iris.load_cube(igbp_file)
    print(f"  Shape: {cube.shape}")
    print(f"  Dimensions: {[c.name() for c in cube.coords()]}")

    # Get all regions (RECCAP2 + Africa + global)
    regions = list(RECCAP_REGIONS.values()) + ['Africa', 'global']

    # Initialize results dict: {region: {pft_name: value}}
    results = {region: {} for region in regions}

    # Extract each PFT
    for pft_id, pft_name in sorted(PFT_MAPPING.items()):
        print(f"\nProcessing PFT {pft_id} ({pft_name})...")
        success_count = 0

        try:
            # Extract this PFT using generic coordinate (pseudo dimension)
            pft_cube = cube.extract(Constraint(coord_values={'generic': pft_id}))

            if not pft_cube:
                print(f"  ⚠ PFT {pft_id} not found, skipping")
                continue

            # Squeeze out singleton dimensions to get (lat, lon)
            # Shape after extract is likely (1, lat, lon) - squeeze first dim
            if pft_cube.shape[0] == 1:
                pft_cube = pft_cube[0]

            # Add coordinate bounds if not present (required for area weighting)
            if not pft_cube.coord('latitude').has_bounds():
                pft_cube.coord('latitude').guess_bounds()
            if not pft_cube.coord('longitude').has_bounds():
                pft_cube.coord('longitude').guess_bounds()

            # Extract regional means for all regions (static field, no time dimension)
            for region in regions:
                try:
                    if region == 'global':
                        # Global mean: area-weighted mean over all grid cells
                        weights = area_weights(pft_cube)
                        mean_val = float(pft_cube.collapsed(
                            ['latitude', 'longitude'],
                            iris.analysis.MEAN,
                            weights=weights
                        ).data)
                    else:
                        # Regional mean: apply mask and compute area-weighted mean
                        mask_cube = region_mask(region)

                        # Ensure mask and data have same shape
                        if mask_cube.shape != pft_cube.shape:
                            # If shapes don't match, regrid mask to pft_cube grid
                            # For now, just check they're compatible
                            raise ValueError(f"Shape mismatch: mask {mask_cube.shape} vs data {pft_cube.shape}")

                        # Apply mask (1 = inside region, 0 = outside)
                        masked_data = np.where(mask_cube.data == 1, pft_cube.data, np.nan)
                        masked_cube = pft_cube.copy()
                        masked_cube.data = masked_data

                        # Compute area-weighted mean (NaN-aware)
                        weights = area_weights(masked_cube)
                        mean_val = float(masked_cube.collapsed(
                            ['latitude', 'longitude'],
                            iris.analysis.MEAN,
                            weights=weights
                        ).data)

                    results[region][pft_name] = mean_val
                    success_count += 1

                except Exception as e:
                    print(f"  ⚠ Failed to extract {region}: {e}")
                    continue

            if success_count > 0:
                print(f"  ✓ Extracted {pft_name} for {success_count}/{len(regions)} regions")
            else:
                print(f"  ❌ Failed to extract {pft_name} for any region")

        except Exception as e:
            print(f"  ❌ Error processing PFT {pft_id}: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(results).T
    df.index.name = 'region'

    return df


def main():
    """Main extraction workflow."""
    print("="*80)
    print("IGBP Regional Means Extraction")
    print("="*80)

    # Check input file exists
    try:
        obs_dir = get_obs_dir()
        igbp_file = os.path.join(obs_dir, 'qrparm.veg.frac_igbp.pp.hadcm3bl.nc')
        print(f"\nInput file: {igbp_file}")
        if not os.path.exists(igbp_file):
            print(f"❌ ERROR: Input file not found!")
            print(f"   Expected: {igbp_file}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)

    # Extract regional means
    df = extract_igbp_regional_means()

    if df.empty:
        print(f"\n❌ ERROR: Extraction produced no data")
        sys.exit(1)

    print(f"\n✓ Extracted {len(df.columns)} PFTs for {len(df)} regions")
    print(f"\nPreview:")
    print(df.head(10))
    print(f"\nFull dataset shape: {df.shape}")

    # Save to CSV
    output_dir = Path(__file__).parent.parent / 'src' / 'utils_cmip7' / 'data' / 'obs'
    output_file = output_dir / 'igbp_regional_means.csv'

    df.to_csv(output_file)
    print(f"\n✓ Saved to: {output_file}")
    print(f"  Rows: {len(df)} regions")
    print(f"  Cols: {len(df.columns)} PFTs")

    print("\n" + "="*80)
    print("SUCCESS! IGBP regional means extracted.")
    print("You can now run: python scripts/validate_experiment.py <expt>")
    print("="*80)


if __name__ == '__main__':
    main()

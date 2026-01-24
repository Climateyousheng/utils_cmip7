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
from utils_cmip7.processing.regional import compute_regional_annual_mean
from utils_cmip7.config import RECCAP_REGIONS


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

        try:
            # Extract this PFT using generic coordinate (pseudo dimension)
            pft_cube = cube.extract(Constraint(coord_values={'generic': pft_id}))

            if not pft_cube:
                print(f"  ⚠ PFT {pft_id} not found, skipping")
                continue

            # Extract regional means for all regions
            for region in regions:
                try:
                    # Use 'frac' as conversion key (no conversion, just fraction)
                    output = compute_regional_annual_mean(pft_cube, 'frac', region)

                    # Store time-mean value
                    mean_val = float(np.mean(output['data']))
                    results[region][pft_name] = mean_val

                except Exception as e:
                    print(f"  ⚠ Failed to extract {region}: {e}")
                    continue

            print(f"  ✓ Extracted {pft_name} for {len(results)} regions")

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

    # Extract regional means
    df = extract_igbp_regional_means()

    print(f"\n✓ Extracted {len(df.columns)} PFTs for {len(df)} regions")
    print(f"\nPreview:")
    print(df.head())

    # Save to CSV
    output_dir = Path(__file__).parent.parent / 'src' / 'utils_cmip7' / 'data' / 'obs'
    output_file = output_dir / 'igbp_regional_means.csv'

    df.to_csv(output_file)
    print(f"\n✓ Saved to: {output_file}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

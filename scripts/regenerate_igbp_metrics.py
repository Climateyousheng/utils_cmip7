#!/usr/bin/env python3
"""
Regenerate igbp.veg_fraction_metrics.nc from the N48-regridded IGBP source.

Reads qrparm.veg.frac_igbp.pp.hadcm3bl.nc (shape 1x9x73x96, N48 grid)
and produces igbp.veg_fraction_metrics.nc with:
  - Per-PFT 2D fields: {pft_name}_2D (latitude, longitude)
  - Global mean scalars: global_mean_{pft_name}, global_mean_trees, global_mean_grass
  - Regional tree metrics: amazon_trees, subtropical_trees_30S_30N, NH_trees_30N_90N

The 2D fields are consumed by load_igbp_spatial() for spatial RMSE computation.
Grid must match the model grid (N48) so subtraction works element-wise.

Usage:
    python scripts/regenerate_igbp_metrics.py
"""

import os
import sys
import warnings
from pathlib import Path

# Add src to path for editable installs
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

import numpy as np

# Configure Iris before import
try:
    import iris

    iris.FUTURE.date_microseconds = True
except AttributeError:
    import iris

warnings.filterwarnings("ignore", message=".*date precision.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*DEFAULT_SPHERICAL_EARTH_RADIUS.*")

from iris import Constraint
import iris.analysis
from iris.analysis.cartography import area_weights

from utils_cmip7.validation.veg_fractions import PFT_MAPPING, get_obs_dir

try:
    import xarray as xr
except ImportError:
    print("xarray is required: pip install xarray netcdf4")
    sys.exit(1)


def _masked_to_nan(data):
    """Convert a masked array to a float array with NaN for masked values."""
    arr = np.asarray(data, dtype=float)
    if hasattr(data, "mask") and np.any(data.mask):
        arr[data.mask] = np.nan
    return arr


def _extract_pft_cube(cube, pft_coord_name, pft_id):
    """Extract a single PFT as a 2D (lat, lon) cube."""
    pft_cube = cube.extract(Constraint(coord_values={pft_coord_name: pft_id}))
    if pft_cube is None:
        return None
    # Squeeze singleton time dimension: (1, lat, lon) -> (lat, lon)
    if pft_cube.ndim == 3 and pft_cube.shape[0] == 1:
        pft_cube = pft_cube[0]
    return pft_cube


def _detect_pft_coord(cube):
    """Detect the PFT pseudo-dimension coordinate name."""
    for candidate in ("generic", "pseudo_level", "pseudo_2"):
        try:
            cube.coord(candidate)
            return candidate
        except iris.exceptions.CoordinateNotFoundError:
            continue
    raise RuntimeError(
        f"Cannot identify PFT coordinate in cube with coords: "
        f"{[c.name() for c in cube.coords()]}"
    )


def _ensure_bounds(cube):
    """Add coordinate bounds if missing (needed for area_weights)."""
    for axis in ("latitude", "longitude"):
        coord = cube.coord(axis)
        if not coord.has_bounds():
            coord.guess_bounds()


def _land_mask(cube, pft_coord_name, threshold=0.01):
    """Create a land mask by summing all PFT fractions."""
    total = None
    for pft_id in PFT_MAPPING:
        pft_cube = _extract_pft_cube(cube, pft_coord_name, pft_id)
        if pft_cube is not None:
            data = _masked_to_nan(pft_cube.data)
            total = data if total is None else total + data
    return (np.nan_to_num(total, nan=0.0) > threshold).astype(float)


def _global_land_mean(pft_cube, land, weights):
    """Area-weighted mean over land cells."""
    data = _masked_to_nan(pft_cube.data)
    land_weights = weights * land
    total_w = np.nansum(land_weights)
    if total_w == 0:
        return np.nan
    return float(np.nansum(data * land_weights) / total_w)


def _box_mean(pft_cube, land, weights, lon_range, lat_range):
    """Area-weighted mean over a lat/lon box, land-only."""
    data = _masked_to_nan(pft_cube.data)
    lats = pft_cube.coord("latitude").points
    lons = pft_cube.coord("longitude").points
    lon_lo, lon_hi = lon_range
    lat_lo, lat_hi = lat_range

    lat_mask = (lats >= lat_lo) & (lats <= lat_hi)
    lon_mask = (lons >= lon_lo) & (lons <= lon_hi)
    box_mask = np.outer(lat_mask, lon_mask).astype(float)

    combined = weights * land * box_mask
    total_w = np.nansum(combined)
    if total_w == 0:
        return np.nan
    return float(np.nansum(data * combined) / total_w)


def regenerate():
    """Main regeneration logic."""
    obs_dir = get_obs_dir()
    src_file = os.path.join(obs_dir, "qrparm.veg.frac_igbp.pp.hadcm3bl.nc")
    out_file = os.path.join(obs_dir, "igbp.veg_fraction_metrics.nc")

    if not os.path.exists(src_file):
        print(f"Source file not found: {src_file}")
        sys.exit(1)

    print(f"Loading: {src_file}")
    cube = iris.load_cube(src_file)
    print(f"  Shape: {cube.shape}")

    pft_coord = _detect_pft_coord(cube)
    print(f"  PFT coordinate: {pft_coord}")

    # Extract a reference 2D cube for coordinates/weights
    ref_cube = _extract_pft_cube(cube, pft_coord, 1)
    _ensure_bounds(ref_cube)
    weights = area_weights(ref_cube)
    land = _land_mask(cube, pft_coord)
    n_land = int(np.sum(land))
    print(f"  Land cells: {n_land}/{land.size} ({100 * n_land / land.size:.1f}%)")

    lats = ref_cube.coord("latitude").points
    lons = ref_cube.coord("longitude").points

    # Build xarray Dataset
    ds = xr.Dataset(
        coords={
            "latitude": ("latitude", lats),
            "longitude": ("longitude", lons),
        }
    )

    global_means = {}

    for pft_id, pft_name in sorted(PFT_MAPPING.items()):
        pft_cube = _extract_pft_cube(cube, pft_coord, pft_id)
        if pft_cube is None:
            print(f"  PFT {pft_id} ({pft_name}) not found, skipping")
            continue

        _ensure_bounds(pft_cube)
        data = _masked_to_nan(pft_cube.data)

        # 2D field (set ocean to NaN)
        data_masked = np.where(land > 0, data, np.nan)
        ds[f"{pft_name}_2D"] = (("latitude", "longitude"), data_masked)

        # Global land mean
        gm = _global_land_mean(pft_cube, land, weights)
        ds[f"global_mean_{pft_name}"] = gm
        global_means[pft_name] = gm
        print(f"  {pft_name}: global mean = {gm:.4f}")

    # Aggregated global means
    bl = global_means.get("BL", 0.0)
    nl = global_means.get("NL", 0.0)
    c3 = global_means.get("C3", 0.0)
    c4 = global_means.get("C4", 0.0)
    ds["global_mean_trees"] = bl + nl
    ds["global_mean_grass"] = c3 + c4
    print(f"  trees: {bl + nl:.4f},  grass: {c3 + c4:.4f}")

    # Regional tree metrics (BL + NL)
    bl_cube = _extract_pft_cube(cube, pft_coord, 1)
    nl_cube = _extract_pft_cube(cube, pft_coord, 2)
    _ensure_bounds(bl_cube)
    _ensure_bounds(nl_cube)

    # Amazon: 290-320E, 15S-5N
    amz_bl = _box_mean(bl_cube, land, weights, (290, 320), (-15, 5))
    amz_nl = _box_mean(nl_cube, land, weights, (290, 320), (-15, 5))
    ds["amazon_trees"] = amz_bl + amz_nl
    print(f"  Amazon trees: {amz_bl + amz_nl:.4f}")

    # Subtropical trees: 0-360E, 30S-30N
    sub_bl = _box_mean(bl_cube, land, weights, (0, 360), (-30, 30))
    sub_nl = _box_mean(nl_cube, land, weights, (0, 360), (-30, 30))
    ds["subtropical_trees_30S_30N"] = sub_bl + sub_nl
    print(f"  Subtropical trees (30S-30N): {sub_bl + sub_nl:.4f}")

    # NH trees: 0-360E, 30N-90N
    nh_bl = _box_mean(bl_cube, land, weights, (0, 360), (30, 90))
    nh_nl = _box_mean(nl_cube, land, weights, (0, 360), (30, 90))
    ds["NH_trees_30N_90N"] = nh_bl + nh_nl
    print(f"  NH trees (30N-90N): {nh_bl + nh_nl:.4f}")

    # Write output
    ds.to_netcdf(out_file)
    print(f"\nWrote: {out_file}")
    print(f"  Dimensions: latitude={len(lats)}, longitude={len(lons)}")


if __name__ == "__main__":
    print("=" * 70)
    print("Regenerate igbp.veg_fraction_metrics.nc (N48)")
    print("=" * 70)
    regenerate()
    print("=" * 70)
    print("Done.")
    print("=" * 70)

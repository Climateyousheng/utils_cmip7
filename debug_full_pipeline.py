#!/usr/bin/env python3
"""Debug the full pipeline from extraction to canonical schema."""

import numpy as np
from utils_cmip7.diagnostics import extract_annual_means
from utils_cmip7.diagnostics.metrics import compute_metrics_from_annual_means

print("="*80)
print("STEP 1: EXTRACT_ANNUAL_MEANS")
print("="*80)

# Extract using the actual function
raw_data = extract_annual_means(
    ['xqhuc'],
    var_list=['GPP', 'NPP'],
    var_mapping=['GPP', 'NPP'],
    regions=['global']
)

print(f"\nExtraction complete. Checking data structure...")
print(f"Keys: {list(raw_data.keys())}")
print(f"xqhuc keys: {list(raw_data['xqhuc'].keys())}")
print(f"xqhuc/global keys: {list(raw_data['xqhuc']['global'].keys())}")

# Check GPP data
gpp_data = raw_data['xqhuc']['global']['GPP']
print(f"\nGPP data from extraction:")
print(f"  Type: {type(gpp_data)}")
print(f"  Keys: {list(gpp_data.keys())}")
print(f"  years shape: {gpp_data['years'].shape}")
print(f"  years first/last: {gpp_data['years'][:3]} ... {gpp_data['years'][-3:]}")
print(f"  data shape: {gpp_data['data'].shape}")
print(f"  data dtype: {gpp_data['data'].dtype}")
print(f"  data first 5: {gpp_data['data'][:5]}")
print(f"  data mean: {np.mean(gpp_data['data']):.2f}")
print(f"  data min/max: {np.min(gpp_data['data']):.2f} / {np.max(gpp_data['data']):.2f}")
print(f"  units: {gpp_data['units']}")

# Check NPP data
npp_data = raw_data['xqhuc']['global']['NPP']
print(f"\nNPP data from extraction:")
print(f"  years shape: {npp_data['years'].shape}")
print(f"  data shape: {npp_data['data'].shape}")
print(f"  data mean: {np.mean(npp_data['data']):.2f}")
print(f"  units: {npp_data['units']}")

print("\n" + "="*80)
print("STEP 2: COMPUTE_METRICS_FROM_ANNUAL_MEANS")
print("="*80)

# Now transform to canonical schema
canonical_data = compute_metrics_from_annual_means(
    'xqhuc',
    metrics=['GPP', 'NPP'],
    regions=['global']
)

print(f"\nCanonical schema transformation complete.")
print(f"Keys: {list(canonical_data.keys())}")
print(f"GPP keys: {list(canonical_data['GPP'].keys())}")

# Check GPP in canonical schema
gpp_canonical = canonical_data['GPP']['global']
print(f"\nGPP data in canonical schema:")
print(f"  Type: {type(gpp_canonical)}")
print(f"  Keys: {list(gpp_canonical.keys())}")
print(f"  years shape: {gpp_canonical['years'].shape}")
print(f"  years first/last: {gpp_canonical['years'][:3]} ... {gpp_canonical['years'][-3:]}")
print(f"  data shape: {gpp_canonical['data'].shape}")
print(f"  data dtype: {gpp_canonical['data'].dtype}")
print(f"  data first 5: {gpp_canonical['data'][:5]}")
print(f"  data mean: {np.mean(gpp_canonical['data']):.2f}")
print(f"  data min/max: {np.min(gpp_canonical['data']):.2f} / {np.max(gpp_canonical['data']):.2f}")
print(f"  units: {gpp_canonical['units']}")
print(f"  source: {gpp_canonical['source']}")
print(f"  dataset: {gpp_canonical['dataset']}")

# Check NPP in canonical schema
npp_canonical = canonical_data['NPP']['global']
print(f"\nNPP data in canonical schema:")
print(f"  data mean: {np.mean(npp_canonical['data']):.2f}")
print(f"  units: {npp_canonical['units']}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"GPP:")
print(f"  After extraction: {np.mean(gpp_data['data']):.2f} {gpp_data['units']}")
print(f"  After canonical:  {np.mean(gpp_canonical['data']):.2f} {gpp_canonical['units']}")
print(f"  Match: {np.array_equal(gpp_data['data'], gpp_canonical['data'])}")

print(f"\nNPP:")
print(f"  After extraction: {np.mean(npp_data['data']):.2f} {npp_data['units']}")
print(f"  After canonical:  {np.mean(npp_canonical['data']):.2f} {npp_canonical['units']}")
print(f"  Match: {np.array_equal(npp_data['data'], npp_canonical['data'])}")

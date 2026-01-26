# Validation Methods: Quick Reference Card

## Three Methods, Three Perspectives

```
                    PARAMETER SPACE ANALYSIS
                           (PPE only)

┌─────────────────────────────────────────────────────────┐
│  Question: "Which parameters drive model skill?"        │
└─────────────────────────────────────────────────────────┘

Method 1: SPEARMAN RANK              Method 2: RF PERMUTATION
├─ Fast screening (seconds)          ├─ Deep analysis (minutes)
├─ Monotonic trends only             ├─ Nonlinear + interactions
├─ No dependencies                   ├─ Requires scikit-learn
└─ Output: Correlation ρ (-1 to +1)  └─ Output: Importance (0 to 1)

         Example Results:
         ┌──────────┬─────────┬──────────┐
         │Parameter │Spearman │ RF Perm  │
         ├──────────┼─────────┼──────────┤
         │ALPHA     │ +0.78   │  0.42    │ ← Agree: Strong effect
         │G_AREA    │ -0.12   │  0.28    │ ← Disagree: Nonlinear!
         │LAI_MIN   │ +0.45   │  0.15    │ ← Agree: Moderate
         │NL0       │ +0.08   │  0.03    │ ← Agree: Weak
         └──────────┴─────────┴──────────┘

         ↓ Interpretation ↓
         • ALPHA: Clear monotonic → important (both methods)
         • G_AREA: No monotonic trend BUT important (RF finds it)
                   → Likely U-shaped or threshold effect
         • LAI_MIN: Moderate importance (both methods)
         • NL0: Not important (both methods)


Method 3: PCA EMBEDDING
├─ Visualize parameter space geometry
├─ 8D parameters → 2D plot
├─ Color by skill score
└─ Output: Scatter plot revealing structure

         Example Plot Interpretation:

              PC2
               ↑
           🔴  |  🟢
          🔴🔴 | 🟢🟢      High skill →
         ─🔴─┼─🟢─────→ PC1
             🔴|🟢
              🔴|🟢        ← Low skill

         PC1 = 0.52·ALPHA + 0.48·G_AREA + 0.30·LAI_MIN + ...
                 ↑              ↑
         Confirms: ALPHA and G_AREA together drive skill

         Insights:
         • Right side (high PC1) = optimal parameter region
         • Left side (low PC1) = poor parameter region
         • No obvious clusters → smooth parameter space
         • Green region shows optimal ALPHA + G_AREA balance


┌─────────────────────────────────────────────────────────┐
│  Combined Workflow                                       │
├─────────────────────────────────────────────────────────┤
│  1. Spearman → Identify obvious trends (30 sec)         │
│  2. RF Perm → Find complex patterns (5 min)             │
│  3. PCA     → Visualize optimal regions (30 sec)        │
│                                                          │
│  Result: Complete understanding of parameter effects    │
└─────────────────────────────────────────────────────────┘
```

---

## Observation-Based Validation (All Methods)

```
┌──────────────────────────────────────────────────────┐
│  Question: "Does model match observations?"          │
└──────────────────────────────────────────────────────┘

Used by: --validate flag (raw extraction, validate-experiment)

Metric 1: BIAS
├─ Formula: um_mean - obs_mean
├─ Units: Same as variable (e.g., PgC/yr)
├─ Sign: + overestimate, - underestimate
└─ Example: -3.16 PgC/yr (2.57% low)

Metric 2: RMSE
├─ Formula: sqrt(mean((um_timeseries - obs_mean)²))
├─ Units: Same as variable
├─ Always: Positive (0 = perfect)
└─ Example: 3.87 PgC/yr

Metric 3: WITHIN UNCERTAINTY
├─ Formula: |bias| <= obs_error
├─ Output: True/False
├─ True: Model agrees with observations
└─ Example: True (|−3.16| < 9.61 ✓)

         Example Output:
         ┌────────┬────────┬─────────┬───────┬────────┬─────────┐
         │Variable│UM Mean │Obs Mean │ Bias  │  RMSE  │Within?  │
         ├────────┼────────┼─────────┼───────┼────────┼─────────┤
         │GPP     │ 120.5  │ 123.16  │ -2.66 │  3.87  │ True ✓  │
         │NPP     │  60.2  │  56.00  │ +4.20 │  4.85  │ False ✗ │
         │CVeg    │ 450.3  │ 450.00  │ +0.30 │  5.31  │ True ✓  │
         └────────┴────────┴─────────┴───────┴────────┴─────────┘

         Validation Summary:
         • 2 out of 3 metrics within uncertainty (67%)
         • Mean absolute bias: 2.38
         • Mean RMSE: 4.68
```

---

## Decision Tree: Which Method When?

```
START
  │
  ├─ Do you have multiple experiments with different parameters?
  │   │
  │   YES → PPE ANALYSIS
  │   │     │
  │   │     ├─ Need quick screening?
  │   │     │   YES → Use Spearman only
  │   │     │   NO → Continue
  │   │     │
  │   │     ├─ Suspect nonlinear effects?
  │   │     │   YES → Use RF Permutation + PCA
  │   │     │   NO → Use Spearman + PCA
  │   │     │
  │   │     └─ Want complete analysis?
  │   │         YES → Use all three (recommended!)
  │   │
  │   NO → Continue to observation validation
  │
  └─ Want to validate against observations?
      │
      YES → OBSERVATION-BASED VALIDATION
      │     │
      │     ├─ Have raw monthly files?
      │     │   YES → Use: python scripts/extract_raw.py --validate
      │     │
      │     └─ Have preprocessed annual means?
      │         YES → Use: utils-cmip7-validate-experiment
```

---

## Command Summary

### PPE Parameter Analysis
```bash
# Full analysis (Spearman + RF + PCA)
utils-cmip7-validate-ppe --param-viz --param-viz-method both

# Spearman only (no scikit-learn needed)
utils-cmip7-validate-ppe --param-viz --param-viz-method spearman

# RF only (requires scikit-learn)
utils-cmip7-validate-ppe --param-viz --param-viz-method rf
```

### Observation Validation
```bash
# Raw extraction with validation
python scripts/extract_raw.py xqhuj --validate
utils-cmip7-extract-raw xqhuj --validate

# Preprocessed validation (with soil params)
utils-cmip7-validate-experiment xqhuc --use-default-soil-params
```

---

## Output Files Reference

### PPE Analysis (`param_viz/`)
```
├── expanded_parameters.csv          # Full parameter matrix
├── summary.json                      # Analysis metadata
│
├── importance_spearman_GPP.csv      # ρ values for each parameter
├── bar_spearman_GPP.png             # Horizontal bar chart
│
├── importance_rfperm_GPP.csv        # RF importance scores
├── bar_rfperm_GPP.png               # Horizontal bar chart
│
└── pca_GPP.png                      # 2D scatter colored by skill
```

### Observation Validation (`validation_outputs/single_val_{expt}/`)
```
├── {expt}_bias_vs_cmip6.csv         # Bias/RMSE vs CMIP6
├── {expt}_bias_vs_reccap2.csv       # Bias/RMSE vs RECCAP2
└── plots/
    ├── GPP_three_way_comparison.png
    ├── NPP_three_way_comparison.png
    └── ...
```

---

## Installation Requirements

```bash
# For observation validation (basic)
pip install utils_cmip7

# For PPE analysis with RF and PCA
pip install 'utils_cmip7[param-viz]'
# or
pip install scikit-learn
```

---

## When Methods Disagree

### Spearman vs RF: Different but Complementary

| Scenario | Spearman | RF | Interpretation |
|----------|----------|----|--------------  |
| Both high | ✓ | ✓ | Strong monotonic effect |
| Spearman high, RF low | ✓ | ✗ | Trend exists but not predictive |
| Spearman low, RF high | ✗ | ✓ | **Nonlinear or interactive effect!** |
| Both low | ✗ | ✗ | Parameter not important |

**Example:** G_AREA shows ρ=-0.12 (Spearman) but 0.28 (RF)
→ G_AREA has nonlinear effect (U-shape or threshold)
→ Check PCA plot to see the pattern visually

---

## Common Pitfalls

### ❌ Using only Spearman for complex systems
**Problem:** Misses nonlinear effects
**Solution:** Use RF permutation importance

### ❌ Ignoring low variance explained in PCA
**Problem:** 2D plot doesn't represent full structure
**Solution:** Check PC1+PC2 variance explained. If <50%, use 3D PCA or interpret with caution

### ❌ Comparing RF importance across different models
**Problem:** Importance values are relative within each model
**Solution:** Only compare importance values from the same analysis run

### ❌ Over-interpreting PCA without checking loadings
**Problem:** PC1 might not represent what you think
**Solution:** Always check PC loadings to see which parameters contribute

---

**For detailed explanations, see:** `docs/VALIDATION_EXPLAINED.md`

**Last updated:** 2026-01-26

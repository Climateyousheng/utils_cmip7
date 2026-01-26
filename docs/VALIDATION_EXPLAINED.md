# Validation Systems in utils_cmip7: Complete Guide

This document explains the two separate validation/analysis systems in the codebase.

---

## System 1: Observation-Based Validation (What --validate Does)

**Purpose:** Compare UM model output against observational benchmarks (CMIP6, RECCAP2)

**Used by:**
- `scripts/extract_raw.py --validate`
- `utils-cmip7-extract-raw --validate`
- `scripts/validate_experiment.py`
- `utils-cmip7-validate-experiment`

**Location:** `src/utils_cmip7/validation/compare.py`

### Metrics Computed

#### 1. **Bias (Absolute & Percentage)**

**Formula:**
```python
absolute_bias = um_mean - obs_mean
percent_bias = 100 * (um_mean - obs_mean) / obs_mean
```

**What it measures:**
- How far off is the model from observations?
- Positive bias = model overestimates
- Negative bias = model underestimates

**Example:**
```
UM GPP:    120.0 PgC/yr
CMIP6 GPP: 123.16 PgC/yr
→ Bias:    -3.16 PgC/yr (-2.57%)
```

#### 2. **RMSE (Root Mean Square Error)**

**Formula:**
```python
rmse = sqrt(mean((um_timeseries - obs_mean)²))
```

**What it measures:**
- Overall deviation between UM time series and observed mean
- Penalizes large errors more than small errors
- Always positive, lower is better

**Example:**
```
UM GPP: [118.0, 120.0, 122.0, 121.0] PgC/yr (4 years)
CMIP6 GPP: 123.16 PgC/yr (constant benchmark)
→ RMSE: 3.87 PgC/yr
```

#### 3. **Within Uncertainty Check**

**Logic:**
```python
within_uncertainty = |bias| <= obs_error
```

**What it measures:**
- Is the model bias smaller than observational uncertainty?
- True = model agrees with observations within error bars
- False = model differs significantly from observations

**Example:**
```
Bias: -3.16 PgC/yr
Obs error: ±9.61 PgC/yr
→ Within uncertainty: True (|-3.16| < 9.61)
```

### Output Structure

**CSV format** (`{expt}_bias_vs_cmip6.csv`):
```csv
metric,region,um_mean,um_std,obs_mean,obs_error,bias,bias_percent,rmse,within_uncertainty,n_years
GPP,global,120.5,2.3,123.16,9.61,-2.66,-2.16,3.87,True,50
NPP,global,60.2,1.5,56.00,3.40,4.20,7.50,4.85,False,50
CVeg,global,450.3,5.2,450.00,60.00,0.30,0.07,5.31,True,50
```

**Validation Summary:**
```
CMIP6 Comparison:
  - Comparisons: 5
  - Within uncertainty: 60.0%
  - Mean bias: -0.52
  - Mean RMSE: 4.16
```

---

## System 2: PPE Parameter Importance Analysis (Separate System)

**Purpose:** Identify which soil parameters drive model skill across parameter ensembles

**Used by:**
- `utils-cmip7-validate-ppe --param-viz`

**Location:** `src/utils_cmip7/plotting/ppe_param_viz.py`

**Requires:** `scikit-learn` (install with `pip install 'utils_cmip7[param-viz]'`)

### Methods

#### 1. **Spearman Rank Correlation**

**What it is:**
- Non-parametric correlation measure
- Assesses **monotonic relationships** between parameter and skill
- Range: -1 (perfect negative) to +1 (perfect positive)

**How it works:**
```python
# Step 1: Convert values to ranks
ALPHA ranks:  [1, 2, 3, 4, 5]  # Smallest to largest
GPP_skill:    [2, 1, 4, 3, 5]  # Skill scores ranked

# Step 2: Compute Pearson correlation on ranks
spearman_r = pearson(rank(ALPHA), rank(GPP_skill))
```

**What it measures:**
- If ALPHA increases, does GPP skill consistently increase (or decrease)?
- **Does NOT require linear relationship** (unlike Pearson)
- Robust to outliers

**Example interpretation:**
```
ALPHA → GPP_skill: ρ = +0.82 (strong positive)
  → Higher ALPHA → Better GPP skill (monotonic increase)

G_AREA → GPP_skill: ρ = -0.15 (weak negative)
  → Little to no monotonic relationship
```

**Use case:** Fast screening to identify parameters with clear trends

#### 2. **RandomForest Permutation Importance**

**What it is:**
- Machine learning approach to measure feature importance
- Captures **nonlinear effects** and **parameter interactions**
- Based on prediction degradation when feature is randomized

**How it works:**

```python
# Step 1: Train RandomForest to predict skill from parameters
RF model: parameters → skill score
  - ALPHA, G_AREA, LAI_MIN, ... → GPP_skill

# Step 2: Measure baseline accuracy
baseline_R² = 0.85  # Model explains 85% of variance

# Step 3: Permute each parameter and measure accuracy drop
Permute ALPHA (shuffle its values randomly):
  - New R² = 0.45
  - Importance = 0.85 - 0.45 = 0.40 (huge drop!)

Permute G_AREA:
  - New R² = 0.83
  - Importance = 0.85 - 0.83 = 0.02 (small drop)

# Step 4: Repeat 20 times and average (for stability)
```

**What it measures:**
- **How much does model skill DEPEND on this parameter?**
- High importance = parameter is critical for predictions
- Low importance = parameter doesn't matter much

**Advantages over Spearman:**
1. **Captures nonlinear relationships**
   ```
   Example: Optimal ALPHA at middle values (U-shape)
   - Spearman: ρ ≈ 0 (no monotonic trend)
   - RF importance: High (strong nonlinear effect)
   ```

2. **Captures interactions**
   ```
   Example: ALPHA effect depends on G_AREA value
   - Spearman: Tests each parameter independently
   - RF: Captures that ALPHA matters MORE when G_AREA is high
   ```

3. **Handles multiple predictors simultaneously**
   ```
   Spearman: One parameter at a time
   RF: All parameters together, accounting for redundancy
   ```

**Example interpretation:**
```
RandomForest Permutation Importance:
  ALPHA:    0.42  (Critical! Removing it destroys predictions)
  G_AREA:   0.28  (Important)
  LAI_MIN:  0.15  (Moderate)
  NL0:      0.03  (Negligible)
```

**Use case:** Comprehensive analysis of parameter influence, including complex effects

---

## Comparison: Spearman vs RF Permutation

| Aspect | Spearman Rank | RF Permutation |
|--------|---------------|----------------|
| **Speed** | Fast (seconds) | Slower (minutes) |
| **Relationships** | Monotonic only | Linear + nonlinear |
| **Interactions** | No | Yes |
| **Dependencies** | None (just numpy/pandas) | scikit-learn |
| **Interpretation** | Simple correlation | Feature importance |
| **Best for** | Quick screening | Deep analysis |
| **Robustness** | High (rank-based) | Moderate (needs enough data) |

### When to Use Which?

**Use Spearman when:**
- Quick exploratory analysis
- Checking for obvious trends
- No scikit-learn available
- Small datasets (<50 experiments)

**Use RF Permutation when:**
- Comprehensive parameter sensitivity
- Suspected nonlinear effects
- Parameter interactions expected
- Larger datasets (>50 experiments)
- Have scikit-learn installed

**Use BOTH (recommended):**
```bash
utils-cmip7-validate-ppe --param-viz --param-viz-method both
```
- Spearman finds obvious trends
- RF finds complex patterns
- Cross-validate findings (both methods agree → high confidence)

---

## Example Workflow: PPE Analysis

### 1. Run PPE validation with parameter visualization

```bash
utils-cmip7-validate-ppe \
  --overview-table random_sampling_combined_overview_table.csv \
  --param-viz \
  --param-viz-method both \
  --param-viz-variables GPP NPP CVeg
```

### 2. Outputs Generated

```
param_viz/
├── expanded_parameters.csv           # Full parameter matrix
├── summary.json                       # Analysis metadata
│
├── importance_spearman_GPP.csv       # Spearman correlations for GPP
├── bar_spearman_GPP.png              # Horizontal bar chart
│
├── importance_rfperm_GPP.csv         # RF importance for GPP
├── bar_rfperm_GPP.png                # Horizontal bar chart
│
├── pca_GPP.png                       # 2D PCA embedding colored by GPP skill
│
└── (same for NPP, CVeg, ...)
```

### 3. Interpretation Example

**Scenario:** Analyzing what drives GPP skill

**Spearman results:**
```
ALPHA:    +0.78  (strong positive correlation)
G_AREA:   -0.12  (weak correlation)
LAI_MIN:  +0.45  (moderate positive)
```
→ **Interpretation:** Higher ALPHA consistently improves GPP skill

**RF Permutation results:**
```
ALPHA:    0.42   (critical importance)
G_AREA:   0.28   (important, despite weak Spearman!)
LAI_MIN:  0.15   (moderate importance)
```
→ **Interpretation:** G_AREA has nonlinear effect not captured by Spearman

**Combined insight:**
- ALPHA has clear monotonic effect (both methods agree)
- G_AREA has complex nonlinear effect (RF found it, Spearman missed it)
- Both parameters are important, but in different ways

---

## Summary: Two Validation Systems

### Observation-Based Validation (`--validate`)
**Compares:** UM vs Observations (CMIP6/RECCAP2)
**Metrics:** Bias, RMSE, within_uncertainty
**Use for:** Checking if model matches reality
**Output:** CSV files + three-way comparison plots

### Parameter Importance Analysis (`--param-viz`)
**Compares:** Parameters vs Skill (across ensemble)
**Methods:** Spearman rank correlation, RF permutation importance
**Use for:** Understanding what parameters drive performance
**Output:** Importance rankings + bar charts + PCA embeddings

---

## References

### Spearman Rank Correlation
- **Paper:** Spearman, C. (1904). "The proof and measurement of association between two things"
- **Method:** Rank-based correlation, robust to outliers
- **Implementation:** `ppe_param_viz.py:227-247`

### RandomForest Permutation Importance
- **Paper:** Breiman, L. (2001). "Random Forests", Machine Learning
- **Method:** Prediction degradation from feature randomization
- **scikit-learn docs:** https://scikit-learn.org/stable/modules/permutation_importance.html
- **Implementation:** `ppe_param_viz.py:250-291`

### RMSE
- **Standard metric:** Root mean square error
- **Paper:** Willmott, C.J. (1982). "Some comments on the evaluation of model performance"

---

## Quick Reference

**Check raw extraction validation:**
```bash
python scripts/extract_raw.py xqhuj --validate
```

**Check PPE parameter importance:**
```bash
utils-cmip7-validate-ppe --param-viz --param-viz-method both
```

**Install scikit-learn for RF analysis:**
```bash
pip install 'utils_cmip7[param-viz]'
# or
pip install scikit-learn
```

---

**Last updated:** 2026-01-26
**Author:** utils_cmip7 development team

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

## Method 3: PCA (Principal Component Analysis) Embedding

**What it is:**
- Dimensionality reduction technique
- Projects high-dimensional parameter space onto 2D for visualization
- Reveals **clusters**, **trade-offs**, and **parameter space geometry**

**Location:** `ppe_param_viz.py:311-342`

### The Problem: High-Dimensional Parameter Space

**Example scenario:**
```
8 soil parameters (ALPHA, G_AREA, LAI_MIN, NL0, R_GROW, TLOW, TUPP, V_CRIT)
→ 8-dimensional space (impossible to visualize!)

How do we see:
- Are there clusters of similar experiments?
- Do high-skill and low-skill experiments group together?
- What trade-offs exist in parameter space?
```

### How PCA Works (Conceptual)

**Step 1: Standardize parameters** (mean=0, std=1)
```
ALPHA: [0.06, 0.08, 0.10] → [-1.0, 0.0, 1.0]
G_AREA: [0.003, 0.004, 0.005] → [-1.0, 0.0, 1.0]
... (all parameters)
```
*Why:* Parameters have different scales (ALPHA ~0.1, G_AREA ~0.004). Standardization ensures no parameter dominates just because of its scale.

**Step 2: Find principal components (PCs)**

PCs are **new axes** that capture maximum variance:

```
Original 8D space:
  - Axis 1: ALPHA
  - Axis 2: G_AREA
  - Axis 3: LAI_MIN
  - ...
  - Axis 8: V_CRIT

New 2D space:
  - PC1: 0.45·ALPHA + 0.38·G_AREA + 0.30·LAI_MIN + ... (combination of all)
  - PC2: -0.20·ALPHA + 0.52·G_AREA - 0.35·LAI_MIN + ... (different combination)
```

**Key insight:** PC1 is the direction with **most variation** in the data, PC2 is the next most important direction (perpendicular to PC1).

**Step 3: Project experiments onto PC1-PC2 plane**

Each experiment's 8D parameter vector gets compressed to 2D coordinates:
```
Experiment xqhuc: [ALPHA=0.08, G_AREA=0.004, ...] → (PC1=-1.2, PC2=0.5)
Experiment xqhsh: [ALPHA=0.10, G_AREA=0.003, ...] → (PC1=0.8, PC2=-0.3)
...
```

**Step 4: Color by skill score**

Plot (PC1, PC2) coordinates, color each point by GPP skill:
- Green = high skill
- Yellow = medium skill
- Red = low skill

### What PCA Reveals

#### 1. **Clusters of Similar Experiments**

**Example visualization:**
```
      PC2
       ↑
   *   |   *
 *  *  |  *  *    ← Cluster A: Low ALPHA, high G_AREA
---*---+---*-------→ PC1
     * | *
      *|*           ← Cluster B: High ALPHA, low G_AREA
```

**Interpretation:**
- Experiments cluster in parameter space
- Clusters might represent different "strategies" or parameter regimes
- Useful for sampling design: Are we covering the full parameter space?

#### 2. **Skill Patterns in Parameter Space**

**Example: All green dots on the right, red on the left:**
```
      PC2
       ↑
   🔴  |  🟢
  🔴🔴 | 🟢🟢    ← High-skill experiments
---🔴--+--🟢------→ PC1
     🔴|🟢
      🔴|🟢        ← Low-skill experiments
```

**Interpretation:**
- High PC1 values → Good skill
- PC1 likely represents a combination of parameters that drive performance
- Can inspect PC1 loadings to see which parameters contribute most

#### 3. **Trade-offs and Frontiers**

**Example: Skill varies along a curve:**
```
      PC2
       ↑
   🟢  |  🔴
  🟡🟢 | 🔴🔴    ← Trade-off frontier
---🟡--+--🔴------→ PC1
     🟡|🟢
      🟢|🟡
```

**Interpretation:**
- Curved pattern suggests **nonlinear relationships**
- Sweet spot might be in the middle (🟢 green region)
- Extreme parameter combinations (corners) perform poorly

#### 4. **Outliers and Anomalies**

**Example: One experiment far from others:**
```
      PC2
       ↑
   *   |   *
  * *  |  * *
--*-*--+--*-*-----→ PC1
     * | *
      *|*

       ⭐ (far away)
```

**Interpretation:**
- Outlier has unusual parameter combination
- Could be:
  - Data quality issue
  - Interesting edge case
  - Numerical instability region

### PCA Plot Interpretation Guide

#### **Variance Explained**

PCA output includes variance explained by each PC:
```
PC1: 45% of variance
PC2: 23% of variance
→ Total: 68% captured in 2D plot
```

**What this means:**
- 68% of parameter variation is visible in the plot
- 32% is in remaining 6 dimensions (hidden)
- Higher % = more reliable 2D representation

**Guidelines:**
- >70%: Excellent 2D representation
- 50-70%: Good, captures main patterns
- <50%: 2D plot misses important structure

#### **PC Loadings (What each PC represents)**

Example loadings for PC1:
```
PC1 = 0.45·ALPHA + 0.38·G_AREA + 0.30·LAI_MIN + 0.15·NL0 + ...
```

**Interpretation:**
- Large positive values: ALPHA, G_AREA dominate PC1
- If high-skill experiments have high PC1 → ALPHA and G_AREA are important
- Connects PCA back to individual parameters

### Complementary Insights: PCA vs Importance Methods

| Aspect | Spearman/RF | PCA |
|--------|-------------|-----|
| **Question** | "Which parameters matter?" | "How is parameter space structured?" |
| **Output** | Rankings/scores | Geometric visualization |
| **Shows** | Individual parameter effects | Relationships between experiments |
| **Clusters** | No | Yes ✓ |
| **Trade-offs** | No | Yes ✓ |
| **Outliers** | No | Yes ✓ |
| **Direct importance** | Yes ✓ | No (indirect via loadings) |

**Use together:**
1. **Importance methods** → Identify critical parameters (ALPHA, G_AREA)
2. **PCA embedding** → Visualize how varying ALPHA & G_AREA affects skill
3. **Combined insight** → Optimal region in (ALPHA, G_AREA) space

### Example Analysis Workflow

**Step 1: Spearman screening**
```
ALPHA:   ρ = +0.78  (strong)
G_AREA:  ρ = -0.12  (weak?)
LAI_MIN: ρ = +0.45  (moderate)
```
→ ALPHA appears important

**Step 2: RF permutation**
```
ALPHA:   0.42  (critical)
G_AREA:  0.28  (important! Spearman missed it)
LAI_MIN: 0.15  (moderate)
```
→ Both ALPHA and G_AREA matter

**Step 3: PCA visualization**
```
      PC2
       ↑
   🔴  |  🟢
  🔴🔴 | 🟢🟢
---🔴--+--🟢------→ PC1
     🔴|🟢
      🔴|🟢

PC1 loadings: 0.52·ALPHA + 0.48·G_AREA + ...
```
→ High PC1 (right side) = high skill
→ PC1 represents optimal combination of ALPHA and G_AREA

**Combined insight:**
- ALPHA and G_AREA are the two most important parameters (RF confirmed)
- They work together (high PC1 = good balance of both)
- There's a specific region in (ALPHA, G_AREA) space with high skill (PCA shows it)
- G_AREA's nonlinearity explains weak Spearman (PCA shows curved pattern)

### PCA Implementation Details

**Code snippet from `ppe_param_viz.py`:**
```python
def plot_embedding_pca(X, y, out_png, title):
    # 1. Impute missing values with median
    Xdf = pd.DataFrame(X).apply(lambda col: col.fillna(col.median()), axis=0)

    # 2. Standardize (mean=0, std=1)
    Xs = StandardScaler().fit_transform(Xdf.values)

    # 3. Reduce to 2D
    Z = PCA(n_components=2, random_state=0).fit_transform(Xs)

    # 4. Plot with skill coloring
    plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='RdYlGn', s=50)
    plt.colorbar(label='Skill Score')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
```

**Output files:**
- `pca_GPP.png` - 2D scatter plot colored by GPP skill
- `pca_GPP_biplot.png` - **NEW!** Biplot showing experiments + parameter vectors
- `pca_GPP_loadings_heatmap.png` - **NEW!** Heatmap of PC loadings
- `pca_GPP_loadings.csv` - **NEW!** PC loadings as CSV for detailed analysis
- (Same for NPP, CVeg, CSoil, etc.)

### Enhanced PCA Outputs (NEW!)

The PCA analysis now generates **four outputs** instead of just one scatter plot:

#### 1. **Scatter Plot** (`pca_GPP.png`)
- Standard PCA embedding colored by skill
- Axes show variance explained: "PC1 (45.3% variance)"
- Corner annotation shows total variance captured

#### 2. **Biplot** (`pca_GPP_biplot.png`) - **KEY FOR INTERPRETATION!**
- Shows both experiments (colored dots) AND parameter vectors (red arrows)
- **Arrow direction** = parameter contribution to PC1 and PC2
- **Arrow length** = strength of contribution
- Only top 8 most influential parameters shown (for clarity)

**How to read the biplot:**
```
Example: ALPHA arrow points right and up
→ High ALPHA → High PC1 and High PC2
→ Experiments on the right side have high ALPHA values

Example: G_AREA arrow points left and up
→ High G_AREA → Low PC1 and High PC2
→ Experiments on the left/top have high G_AREA values
```

**Connecting parameters to skill:**
```
If green (high-skill) experiments cluster in the upper-right:
→ Look at arrows pointing upper-right (e.g., ALPHA, LAI_MIN)
→ Those parameters drive high skill!

If red (low-skill) experiments cluster in the lower-left:
→ Look at arrows pointing lower-left
→ Those parameter combinations are poor
```

#### 3. **Loadings Heatmap** (`pca_GPP_loadings_heatmap.png`)
- Shows exact loading values for all parameters
- Sorted by total contribution (most important at top)
- Red = positive loading, Blue = negative loading
- Values shown in each cell

**Reading the heatmap:**
```
PC1 Column:
  ALPHA:   +0.52  (strong positive)
  G_AREA:  +0.48  (strong positive)
  LAI_MIN: +0.30  (moderate positive)
  NL0:     -0.05  (negligible)

PC2 Column:
  ALPHA:   -0.20  (moderate negative)
  G_AREA:  +0.52  (strong positive)
  ...

Interpretation:
→ PC1 ≈ 0.52·ALPHA + 0.48·G_AREA + 0.30·LAI_MIN + ...
→ PC2 ≈ -0.20·ALPHA + 0.52·G_AREA + ...
```

#### 4. **Loadings CSV** (`pca_GPP_loadings.csv`)
- Machine-readable loadings for further analysis
- Columns: PC1, PC2, PC1_abs, PC2_abs
- Rows: All parameters
- Use for quantitative analysis or custom plots

**Example CSV content:**
```csv
parameter,PC1,PC2,PC1_abs,PC2_abs
ALPHA,0.5234,-0.1985,0.5234,0.1985
G_AREA,0.4821,0.5193,0.4821,0.5193
LAI_MIN,0.3012,0.0845,0.3012,0.0845
...
```

### Common PCA Patterns and Their Meanings

#### Pattern 1: Clear Separation
```
🔴🔴🔴 | 🟢🟢🟢
```
**Meaning:** Simple linear relationship. One side of PC1 is good, other side is bad.

#### Pattern 2: Gradient
```
🔴→🟡→🟢
```
**Meaning:** Continuous improvement along PC1. Optimize in that direction.

#### Pattern 3: Sweet Spot
```
🔴 🟢 🔴
   🟡
```
**Meaning:** Optimal region in the middle. Extremes are bad. Watch for U-shaped relationships.

#### Pattern 4: Multiple Clusters
```
🟢🟢  🔴🔴
🟢🟢  🔴🔴
```
**Meaning:** Multiple distinct regimes. Different parameter combinations with different outcomes.

#### Pattern 5: No Pattern
```
🔴🟢🔴🟡
🟡🔴🟢🟢
```
**Meaning:**
- Either skill is random (parameter variations don't matter)
- Or important structure is in hidden dimensions (PC3-PC8)
- Check variance explained - if low (<50%), try 3D PCA or look at PC3

---

## Summary: Three Complementary Methods

### Spearman Rank Correlation
**Answers:** "Which parameters have monotonic effects?"
**Strength:** Fast screening, simple interpretation
**Limitation:** Misses nonlinear effects

### RandomForest Permutation Importance
**Answers:** "Which parameters are critical for predictions?"
**Strength:** Captures nonlinear effects and interactions
**Limitation:** Doesn't show spatial structure

### PCA Embedding
**Answers:** "What does parameter space look like?"
**Strength:** Reveals clusters, trade-offs, optimal regions
**Limitation:** Doesn't quantify individual parameter importance

**Use all three together** for comprehensive understanding:
1. **Spearman** → Quick screening (seconds)
2. **RF** → Deep parameter sensitivity (minutes)
3. **PCA** → Spatial structure and visualization (seconds)

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

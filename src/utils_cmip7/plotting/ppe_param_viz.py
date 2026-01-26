"""
Parameter importance analysis for PPE experiments.

Quantifies which parameters drive model skill through:
- Spearman rank correlation (monotonic sensitivity)
- RandomForest permutation importance (nonlinear + interactions)
- PCA embedding (parameter space geometry)
- Parallel coordinates (compare experiments)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


_LIST_RE = re.compile(r"^\s*\[.*\]\s*$")


def _parse_maybe_list(x):
    """
    Parse a value that might be a scalar or a list-like string.

    Handles:
    - Comma-separated: "0.08,0.08,0.04"
    - Python list string: "[0.08, 0.08]"
    - Scalar numeric: "0.08"
    """
    if pd.isna(x):
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in x]
    s = str(x).strip()

    # Comma-separated numbers: "0.08,0.08,0.04"
    if "," in s and not any(c.isalpha() for c in s):
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return None

    # Python-list-ish string: "[0.08, 0.08]"
    if _LIST_RE.match(s):
        s2 = s.strip()[1:-1]
        parts = [p.strip() for p in s2.split(",") if p.strip() != ""]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return None

    # Scalar numeric
    try:
        return float(s)
    except ValueError:
        return None


def load_overview_table(path: str) -> pd.DataFrame:
    """Load overview table CSV, removing unnamed index columns."""
    df = pd.read_csv(path)
    # Avoid accidental unnamed index col
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    return df


def infer_columns(
    df: pd.DataFrame,
    id_col: Optional[str] = None,
    param_cols: Optional[list[str]] = None,
    skill_cols: Optional[list[str]] = None
) -> tuple[str, list[str], list[str]]:
    """
    Infer experiment ID, parameter, and skill columns from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Overview table
    id_col : str, optional
        Explicit ID column name
    param_cols : list[str], optional
        Explicit parameter column names
    skill_cols : list[str], optional
        Explicit skill column names (e.g., ['GPP', 'NPP', 'CVeg'])

    Returns
    -------
    id_col : str
        Experiment ID column
    param_cols : list[str]
        Parameter columns
    skill_cols : list[str]
        Skill/metric columns
    """
    # Detect ID column
    if id_col is None:
        id_candidates = ["ID", "expt", "experiment", "run", "job", "runid"]
        lower = {c: c for c in df.columns}
        for cand in id_candidates:
            matches = [c for c in df.columns if c.lower() == cand.lower()]
            if matches:
                id_col = matches[0]
                break
        if id_col is None:
            # Fallback: first object-like col
            obj_cols = [c for c in df.columns if df[c].dtype == "object"]
            id_col = obj_cols[0] if obj_cols else df.columns[0]

    # Detect skill columns if not provided
    if skill_cols is None:
        # Known parameter columns (soil params, regional metrics)
        known_params = {
            'ALPHA', 'G_AREA', 'LAI_MIN', 'NL0', 'R_GROW', 'TLOW', 'TUPP', 'V_CRIT',
            'Tr30SN', 'Tr30-90N', 'AMZTrees'
        }
        # Known skill columns (carbon metrics, vegetation metrics, RMSE, score)
        skill_patterns = ['GPP', 'NPP', 'CVeg', 'CSoil', 'GM_', 'rmse_', 'overall_score']

        skill_cols = []
        for c in df.columns:
            if c == id_col:
                continue
            if c in known_params:
                continue
            # Check if it matches skill patterns
            if any(pat in c for pat in skill_patterns):
                # Verify it's numeric
                if pd.api.types.is_numeric_dtype(df[c]):
                    skill_cols.append(c)

    # Detect parameter columns if not provided
    if param_cols is None:
        # Default to known soil parameters
        known_params = [
            'ALPHA', 'G_AREA', 'LAI_MIN', 'NL0', 'R_GROW', 'TLOW', 'TUPP', 'V_CRIT'
        ]
        param_cols = [c for c in known_params if c in df.columns]

    return id_col, param_cols, skill_cols


def expand_param_vectors(
    df: pd.DataFrame,
    id_col: str,
    param_cols: list[str]
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Expand parameter columns that contain vector values.

    For example, ALPHA might be "0.08,0.08,0.04" across 3 PFTs,
    which gets expanded to ALPHA_0, ALPHA_1, ALPHA_2.

    Parameters
    ----------
    df : pd.DataFrame
        Overview table
    id_col : str
        Experiment ID column
    param_cols : list[str]
        Parameter columns to expand

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_experiments, n_features)
    feature_names : list[str]
        Feature names (expanded from param_cols)
    meta : pd.DataFrame
        Metadata (experiment IDs)
    """
    # First pass: determine max lengths for vector params
    max_len = {}
    for c in param_cols:
        lens = []
        for v in df[c].values[:min(len(df), 5000)]:
            parsed = _parse_maybe_list(v)
            if isinstance(parsed, list):
                lens.append(len(parsed))
        max_len[c] = max(lens) if lens else 1

    # Build feature matrix
    feat_cols = []
    for c in param_cols:
        k = max_len[c]
        if k == 1:
            feat_cols.append(c)
        else:
            feat_cols.extend([f"{c}_{i}" for i in range(k)])
    feature_names = feat_cols

    X = np.full((len(df), len(feature_names)), np.nan, dtype=float)
    for i, (_, r) in enumerate(df.iterrows()):
        col_j = 0
        for c in param_cols:
            k = max_len[c]
            parsed = _parse_maybe_list(r[c])
            if isinstance(parsed, list):
                vals = parsed[:k] + [np.nan] * max(0, k - len(parsed))
            elif isinstance(parsed, (int, float)):
                vals = [parsed] + ([np.nan] * (k - 1))
            else:
                vals = [np.nan] * k
            X[i, col_j:col_j+k] = np.asarray(vals, dtype=float)
            col_j += k

    meta = df[[id_col]].copy()
    return X, feature_names, meta


def spearman_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str]
) -> pd.Series:
    """
    Compute Spearman rank correlation between each feature and target.

    Fast screening for monotonic sensitivity.
    """
    # Compute Spearman via rank transform + Pearson (avoids scipy dependency)
    Xr = pd.DataFrame(X, columns=feature_names).rank(axis=0, na_option="keep")
    yr = pd.Series(y).rank(na_option="keep")
    corrs = {}
    for c in feature_names:
        tmp = pd.concat([Xr[c], yr], axis=1).dropna()
        if len(tmp) < 8:
            corrs[c] = np.nan
        else:
            corrs[c] = np.corrcoef(tmp.iloc[:, 0], tmp.iloc[:, 1])[0, 1]
    return pd.Series(corrs).sort_values(key=lambda s: s.abs(), ascending=False)


def rf_permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 500,
    random_state: int = 0
) -> pd.Series:
    """
    Compute RandomForest permutation importance.

    Captures nonlinear effects and interactions.
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for RandomForest importance analysis.\n"
            "Install with: pip install 'utils_cmip7[param-viz]' or pip install scikit-learn"
        )

    # Drop rows with missing y
    mask = np.isfinite(y)
    X2 = X[mask]
    y2 = y[mask]

    # Simple imputation: column median
    Xdf = pd.DataFrame(X2, columns=feature_names)
    Xdf = Xdf.apply(lambda col: col.fillna(col.median()), axis=0)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_features="sqrt",
    )
    model.fit(Xdf.values, y2)
    perm = permutation_importance(
        model, Xdf.values, y2,
        n_repeats=20,
        random_state=random_state,
        n_jobs=-1
    )
    imp = pd.Series(perm.importances_mean, index=feature_names)
    return imp.sort_values(ascending=False)


def plot_importance_bar(
    imp: pd.Series,
    title: str,
    out_png: str,
    top_k: int = 25
):
    """Plot horizontal bar chart of feature importances."""
    s = imp.dropna().iloc[:top_k][::-1]  # Reverse for bottom-to-top
    plt.figure(figsize=(10, max(4, 0.25 * len(s))))
    plt.barh(s.index, s.values, color='steelblue')
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_embedding_pca(
    X: np.ndarray,
    y: np.ndarray,
    out_png: str,
    title: str,
    feature_names: Optional[list[str]] = None
):
    """
    Plot 2D PCA embedding of parameter space, colored by skill.

    Reveals clusters and trade-offs in parameter space.

    Also saves PC loadings to CSV and creates loading visualizations.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target values (skill scores)
    out_png : str
        Output path for scatter plot
    title : str
        Plot title
    feature_names : list[str], optional
        Feature names for loading interpretation
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for PCA embedding.\n"
            "Install with: pip install 'utils_cmip7[param-viz]' or pip install scikit-learn"
        )

    # Median-impute for PCA
    Xdf = pd.DataFrame(X, columns=feature_names).apply(lambda col: col.fillna(col.median()), axis=0)
    Xs = StandardScaler().fit_transform(Xdf.values)

    # Fit PCA
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)

    # Extract loadings and variance explained
    loadings = pca.components_.T  # Shape: (n_features, 2)
    var_explained = pca.explained_variance_ratio_

    # Save loadings to CSV
    if feature_names is not None:
        out_dir = os.path.dirname(out_png)
        base_name = os.path.basename(out_png).replace('.png', '')

        loadings_df = pd.DataFrame(
            loadings,
            index=feature_names,
            columns=['PC1', 'PC2']
        )
        loadings_df['PC1_abs'] = loadings_df['PC1'].abs()
        loadings_df['PC2_abs'] = loadings_df['PC2'].abs()
        loadings_csv = os.path.join(out_dir, f'{base_name}_loadings.csv')
        loadings_df.to_csv(loadings_csv, float_format='%.4f')

    # Main scatter plot with variance explained in labels
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap='RdYlGn', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    plt.colorbar(sc, label='Skill Score', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)')
    ax.grid(True, alpha=0.3)

    # Add text showing total variance explained
    total_var = var_explained.sum() * 100
    ax.text(0.02, 0.98, f'Total: {total_var:.1f}% variance',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # Create biplot (scatter + parameter vectors)
    if feature_names is not None:
        biplot_png = out_png.replace('.png', '_biplot.png')
        _create_biplot(Z, y, loadings, feature_names, var_explained, title, biplot_png)

        # Create loading heatmap
        heatmap_png = out_png.replace('.png', '_loadings_heatmap.png')
        _create_loading_heatmap(loadings, feature_names, var_explained, title, heatmap_png)


def _create_biplot(
    Z: np.ndarray,
    y: np.ndarray,
    loadings: np.ndarray,
    feature_names: list[str],
    var_explained: np.ndarray,
    title: str,
    out_png: str,
    n_top: int = 8
):
    """
    Create PCA biplot showing both experiments and parameter vectors.

    Parameters
    ----------
    Z : np.ndarray
        PCA-transformed coordinates (n_samples, 2)
    y : np.ndarray
        Skill scores
    loadings : np.ndarray
        PC loadings (n_features, 2)
    feature_names : list[str]
        Parameter names
    var_explained : np.ndarray
        Variance explained by each PC
    title : str
        Plot title
    out_png : str
        Output path
    n_top : int
        Number of top features to show as vectors
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scale factor for arrows (to make them visible)
    scale = 3.0

    # Plot experiments
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap='RdYlGn', s=40, alpha=0.6,
                    edgecolors='k', linewidth=0.5, label='Experiments')

    # Plot parameter vectors (top contributors only)
    loading_magnitude = np.sqrt((loadings**2).sum(axis=1))
    top_idx = np.argsort(loading_magnitude)[-n_top:]

    for idx in top_idx:
        ax.arrow(0, 0, loadings[idx, 0] * scale, loadings[idx, 1] * scale,
                 head_width=0.15, head_length=0.15, fc='red', ec='darkred',
                 alpha=0.7, linewidth=1.5)
        ax.text(loadings[idx, 0] * scale * 1.15, loadings[idx, 1] * scale * 1.15,
                feature_names[idx], fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.colorbar(sc, label='Skill Score', ax=ax)
    ax.set_title(f'{title} - Biplot')
    ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

    # Add legend
    ax.text(0.02, 0.98, f'Top {n_top} parameters shown\nArrow = parameter influence',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _create_loading_heatmap(
    loadings: np.ndarray,
    feature_names: list[str],
    var_explained: np.ndarray,
    title: str,
    out_png: str
):
    """
    Create heatmap of PC loadings.

    Shows which parameters contribute most to each PC.

    Parameters
    ----------
    loadings : np.ndarray
        PC loadings (n_features, 2)
    feature_names : list[str]
        Parameter names
    var_explained : np.ndarray
        Variance explained by each PC
    title : str
        Plot title
    out_png : str
        Output path
    """
    # Sort features by total contribution (PC1^2 + PC2^2)
    loading_magnitude = np.sqrt((loadings**2).sum(axis=1))
    sorted_idx = np.argsort(loading_magnitude)[::-1]

    sorted_loadings = loadings[sorted_idx, :]
    sorted_names = [feature_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(6, max(4, len(feature_names) * 0.3)))

    im = ax.imshow(sorted_loadings, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    # Set ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'PC1\n({var_explained[0]*100:.1f}%)',
                        f'PC2\n({var_explained[1]*100:.1f}%)'])
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Loading', rotation=270, labelpad=15)

    # Add values in cells
    for i in range(len(sorted_names)):
        for j in range(2):
            text = ax.text(j, i, f'{sorted_loadings[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title(f'{title} - PC Loadings')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def run_suite(
    overview_csv: str,
    outdir: str,
    variables: Optional[list[str]] = None,
    id_col: Optional[str] = None,
    param_cols: Optional[list[str]] = None,
    method: str = "both"
):
    """
    Run complete parameter importance analysis suite.

    Parameters
    ----------
    overview_csv : str
        Path to overview table CSV
    outdir : str
        Output directory for results
    variables : list[str], optional
        Variables to analyze (e.g., ['GPP', 'NPP', 'CVeg'])
        If None, analyzes all detected skill columns
    id_col : str, optional
        Experiment ID column name
    param_cols : list[str], optional
        Parameter columns to analyze
    method : str
        'spearman', 'rf', or 'both'
    """
    # Check for scikit-learn if RF method requested
    if method in ('rf', 'both') and not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for RandomForest importance analysis.\n"
            "Install with: pip install 'utils_cmip7[param-viz]' or pip install scikit-learn\n"
            "Alternatively, use --param-viz-method spearman for correlation-only analysis."
        )

    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Parameter Importance Analysis")
    print(f"{'='*70}")

    # Load data
    df = load_overview_table(overview_csv)
    print(f"  Loaded {len(df)} experiments from {overview_csv}")

    # Infer columns
    id_col_detected, param_cols_detected, skill_cols = infer_columns(
        df, id_col=id_col, param_cols=param_cols, skill_cols=None
    )

    # Filter skill columns by variables if provided
    if variables:
        vset = [v.lower() for v in variables]
        skill_cols = [c for c in skill_cols if any(v in c.lower() for v in vset)]

    print(f"  Detected ID column: {id_col_detected}")
    print(f"  Parameter columns: {param_cols_detected}")
    print(f"  Skill columns: {skill_cols}")

    # Expand parameter vectors
    X, feature_names, meta = expand_param_vectors(df, id_col_detected, param_cols_detected)
    print(f"  Expanded to {len(feature_names)} features")

    # Persist expanded features for debugging
    feat_df = pd.concat([
        df[[id_col_detected]],
        pd.DataFrame(X, columns=feature_names)
    ], axis=1)
    expanded_path = os.path.join(outdir, "expanded_parameters.csv")
    feat_df.to_csv(expanded_path, index=False, float_format='%.5f')
    print(f"  ✓ Saved expanded parameters: {expanded_path}")

    # Summary metadata
    summary = {
        'n_experiments': len(df),
        'n_features': len(feature_names),
        'n_skill_cols': len(skill_cols),
        'id_col': id_col_detected,
        'param_cols': param_cols_detected,
        'feature_names': feature_names,
        'skill_cols': skill_cols,
        'method': method,
        'top_features': {}
    }

    # Analyze each skill column
    for sc in skill_cols:
        print(f"\n  Analyzing: {sc}")
        y = pd.to_numeric(df[sc], errors="coerce").values
        n_valid = np.sum(np.isfinite(y))
        print(f"    Valid observations: {n_valid}/{len(y)}")

        if n_valid < 10:
            print(f"    ⚠ Skipping (too few valid observations)")
            continue

        # Spearman correlation
        if method in ("spearman", "both"):
            print(f"    Computing Spearman correlations...")
            imp_s = spearman_importance(X, y, feature_names)
            imp_s.to_csv(
                os.path.join(outdir, f"importance_spearman_{sc}.csv"),
                header=['importance']
            )
            plot_importance_bar(
                imp_s,
                f"Spearman Rank Correlation | {sc}",
                os.path.join(outdir, f"bar_spearman_{sc}.png")
            )
            summary['top_features'][f'{sc}_spearman'] = imp_s.head(10).to_dict()

        # RandomForest permutation importance
        if method in ("rf", "both"):
            print(f"    Training RandomForest and computing permutation importance...")
            imp_rf = rf_permutation_importance(X, y, feature_names)
            imp_rf.to_csv(
                os.path.join(outdir, f"importance_rfperm_{sc}.csv"),
                header=['importance']
            )
            plot_importance_bar(
                imp_rf,
                f"RF Permutation Importance | {sc}",
                os.path.join(outdir, f"bar_rfperm_{sc}.png")
            )
            summary['top_features'][f'{sc}_rf'] = imp_rf.head(10).to_dict()

        # PCA embedding
        print(f"    Plotting PCA embedding...")
        plot_embedding_pca(
            X, y,
            os.path.join(outdir, f"pca_{sc}.png"),
            title=f"PCA of Parameter Space | Colored by {sc}",
            feature_names=feature_names
        )
        print(f"    ✓ Created PCA scatter, biplot, and loadings heatmap")

    # Save summary
    summary_path = os.path.join(outdir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Analysis complete. Results saved to: {outdir}")
    print(f"  ✓ Summary: {summary_path}")
    print(f"{'='*70}\n")


__all__ = [
    'load_overview_table',
    'infer_columns',
    'expand_param_vectors',
    'spearman_importance',
    'rf_permutation_importance',
    'plot_importance_bar',
    'plot_embedding_pca',
    'run_suite',
]

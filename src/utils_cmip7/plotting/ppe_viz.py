#!/usr/bin/env python3
"""
ppe_viz.py

Reusable visualisation utilities for PPE (Perturbed Physics Ensemble) validation tables.

Typical input: a CSV with columns like:
- ID (optional), overall_score (required for ranking)
- parameters: ALPHA, G_AREA, LAI_MIN, NL0, R_GROW, TLOW, TUPP, V_CRIT (optional)
- metrics: rmse_*, and others (numeric)

Usage (CLI examples):
  python soil_validation_viz.py score  --csv table.csv --out score.pdf --top-n 15
  python soil_validation_viz.py heatmap --csv table.csv --out heatmap.pdf --top-k 30
  python soil_validation_viz.py shift  --csv table.csv --out shift.pdf  --q 0.25
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------
# Configuration / defaults
# -----------------------------

DEFAULT_PARAM_COLS = ["ALPHA", "G_AREA", "LAI_MIN", "NL0", "R_GROW", "TLOW", "TUPP", "V_CRIT"]
DEFAULT_SCORE_COL = "overall_score"
DEFAULT_ID_COL = "ID"
DEFAULT_RMSE_PREFIXES = ("rmse_",)


@dataclass(frozen=True)
class NormalizeConfig:
    """Controls normalization used in the heatmap."""
    method: str = "minmax"  # "minmax" only in this implementation
    clip_quantiles: Optional[Tuple[float, float]] = None
    # Example: (0.01, 0.99) to reduce outlier influence


# -----------------------------
# Core utilities
# -----------------------------

def read_table(csv_path: str) -> pd.DataFrame:
    """Read CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def rank_by_score(
    df: pd.DataFrame,
    score_col: str = DEFAULT_SCORE_COL,
    descending: bool = True,
) -> pd.DataFrame:
    """Return df sorted by score."""
    if score_col not in df.columns:
        raise KeyError(f"score_col '{score_col}' not found in columns.")
    return df.sort_values(score_col, ascending=not descending).reset_index(drop=True)


def select_numeric_metrics(
    df: pd.DataFrame,
    exclude_cols: Sequence[str],
) -> List[str]:
    """Return numeric columns excluding exclude_cols."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in set(exclude_cols)]


def format_run_id(row: pd.Series, fallback: str, id_col: Optional[str]) -> str:
    if id_col and id_col in row.index and pd.notna(row[id_col]):
        return str(row[id_col])
    return fallback


def _minmax_normalize_series(x: pd.Series, clip_q: Optional[Tuple[float, float]]) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    if clip_q is not None:
        loq, hiq = clip_q
        lo = np.nanquantile(x.values, loq)
        hi = np.nanquantile(x.values, hiq)
        x = x.clip(lower=lo, upper=hi)

    xmin, xmax = np.nanmin(x.values), np.nanmax(x.values)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax == xmin:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - xmin) / (xmax - xmin + 1e-12)


def normalize_metrics_for_heatmap(
    df_metrics: pd.DataFrame,
    invert_prefixes: Sequence[str] = DEFAULT_RMSE_PREFIXES,
    norm_cfg: NormalizeConfig = NormalizeConfig(),
) -> pd.DataFrame:
    """
    Normalize each metric column to [0,1], optionally invert metrics (e.g., rmse_*).
    Convention returned: higher = better.
    """
    if norm_cfg.method != "minmax":
        raise ValueError(f"Unsupported normalization method: {norm_cfg.method}")

    norm = pd.DataFrame(index=df_metrics.index)
    for c in df_metrics.columns:
        x = _minmax_normalize_series(df_metrics[c], norm_cfg.clip_quantiles)
        if any(c.startswith(pfx) for pfx in invert_prefixes):
            x = 1 - x
        norm[c] = x
    return norm


# -----------------------------
# Plot functions (module API)
# -----------------------------

def plot_score_histogram(
    df: pd.DataFrame,
    score_col: str = DEFAULT_SCORE_COL,
    id_col: Optional[str] = DEFAULT_ID_COL,
    top_n: int = 15,
    bins: int = 40,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Histogram of overall_score with vertical markers and a top-N label box.
    """
    ranked = rank_by_score(df, score_col=score_col, descending=True)
    top = ranked.head(top_n)

    ax = ax or plt.gca()
    vals = pd.to_numeric(df[score_col], errors="coerce").dropna().values
    ax.hist(vals, bins=bins)
    ax.set_xlabel(score_col)
    ax.set_ylabel("Count")
    ax.set_title(title or f"Histogram of {score_col} (top-{top_n} labeled)")

    # markers
    for _, r in top.iterrows():
        x = float(r[score_col])
        ax.axvline(x, linewidth=1)

    # labels
    lines = []
    for i, r in top.iterrows():
        rid = format_run_id(r, fallback=f"rank{i+1}", id_col=id_col)
        lines.append(f"{i+1:>2}. {rid}  {float(r[score_col]):.4g}")
    ax.text(
        0.98, 0.98, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    return ax


def plot_score_ecdf(
    df: pd.DataFrame,
    score_col: str = DEFAULT_SCORE_COL,
    id_col: Optional[str] = DEFAULT_ID_COL,
    top_n: int = 15,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    ECDF of overall_score with top-N points labeled.
    """
    ranked = rank_by_score(df, score_col=score_col, descending=True)
    top = ranked.head(top_n)

    ax = ax or plt.gca()
    scores = pd.to_numeric(df[score_col], errors="coerce").dropna().values
    xs = np.sort(scores)
    ys = np.arange(1, len(xs) + 1) / len(xs)

    ax.plot(xs, ys)
    ax.set_xlabel(score_col)
    ax.set_ylabel("ECDF")
    ax.set_title(title or f"ECDF of {score_col} (top-{top_n} labeled)")

    for i, r in top.iterrows():
        x = float(r[score_col])
        y = np.searchsorted(xs, x, side="right") / len(xs)
        rid = format_run_id(r, fallback=f"rank{i+1}", id_col=id_col)
        ax.plot([x], [y], marker="o")
        ax.annotate(f"{i+1}:{rid}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    return ax


def plot_validation_heatmap(
    df: pd.DataFrame,
    score_col: str = DEFAULT_SCORE_COL,
    id_col: Optional[str] = DEFAULT_ID_COL,
    param_cols: Sequence[str] = DEFAULT_PARAM_COLS,
    top_k: int = 30,
    metrics: Optional[Sequence[str]] = None,
    invert_prefixes: Sequence[str] = DEFAULT_RMSE_PREFIXES,
    norm_cfg: NormalizeConfig = NormalizeConfig(),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Heatmap for top-k runs, normalized 0..1 per metric with 'higher=better'.
    RMSE columns are inverted by default via invert_prefixes=("rmse_",).
    """
    ranked = rank_by_score(df, score_col=score_col, descending=True)
    top = ranked.head(top_k).copy()

    existing_param_cols = [c for c in param_cols if c in df.columns]
    if metrics is None:
        metrics = select_numeric_metrics(top, exclude_cols=list(existing_param_cols) + [score_col])

    # Order metrics: rmse_* first (if present), then others
    metrics = [c for c in metrics if c in top.columns]
    rmse_first = [c for c in metrics if any(c.startswith(pfx) for pfx in invert_prefixes)]
    other = [c for c in metrics if c not in rmse_first]
    metrics_ordered = rmse_first + other

    if len(metrics_ordered) == 0:
        raise ValueError("No numeric metric columns found for heatmap.")

    mat = top[metrics_ordered].apply(pd.to_numeric, errors="coerce")
    norm = normalize_metrics_for_heatmap(mat, invert_prefixes=invert_prefixes, norm_cfg=norm_cfg)

    ax = ax or plt.gca()
    im = ax.imshow(norm.values, aspect="auto")

    ylabels = [
        format_run_id(row, fallback=f"rank{i+1}", id_col=id_col)
        for i, row in top.iterrows()
    ]
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)

    ax.set_xticks(range(len(metrics_ordered)))
    ax.set_xticklabels(metrics_ordered, rotation=90)

    ax.set_title(title or f"Top {top_k} runs: normalized validation metrics (higher=better)")
    plt.colorbar(im, ax=ax, label="Normalized goodness")
    return ax


def plot_parameter_shift(
    df: pd.DataFrame,
    score_col: str = DEFAULT_SCORE_COL,
    param_cols: Sequence[str] = DEFAULT_PARAM_COLS,
    q: float = 0.10,
    bins: int = 30,
    title_prefix: str = "Parameter shift",
    ax: Optional[plt.Axes] = None,
) -> List[plt.Figure]:
    """
    For each parameter: histogram overlay comparing top q vs bottom q by score.
    Returns a list of Figures (one per parameter) to make PDF writing easy.

    q=0.10 means top 10% vs bottom 10%.
    q=0.25 means top 25% vs bottom 25%.
    """
    if not (0 < q < 0.5):
        raise ValueError("q must be between 0 and 0.5 (exclusive).")

    existing_param_cols = [c for c in param_cols if c in df.columns]
    if len(existing_param_cols) == 0:
        raise ValueError("No parameter columns found for shift plots.")

    scores = pd.to_numeric(df[score_col], errors="coerce")
    hi_thr = scores.quantile(1 - q)
    lo_thr = scores.quantile(q)

    hi = df[scores >= hi_thr]
    lo = df[scores <= lo_thr]

    figs: List[plt.Figure] = []
    for p in existing_param_cols:
        fig = plt.figure(figsize=(9, 5))
        axp = fig.gca()
        axp.hist(pd.to_numeric(lo[p], errors="coerce").dropna().values, bins=bins, alpha=0.6, label=f"bottom {int(q*100)}%")
        axp.hist(pd.to_numeric(hi[p], errors="coerce").dropna().values, bins=bins, alpha=0.6, label=f"top {int(q*100)}%")
        axp.set_xlabel(p)
        axp.set_ylabel("Count")
        axp.set_title(f"{title_prefix}: {p} (top vs bottom {int(q*100)}% by {score_col})")
        axp.legend()
        fig.tight_layout()
        figs.append(fig)

    return figs


# -----------------------------
# Convenience writers
# -----------------------------

def save_score_plots_pdf(
    df: pd.DataFrame,
    out_pdf: str,
    score_col: str = DEFAULT_SCORE_COL,
    id_col: Optional[str] = DEFAULT_ID_COL,
    top_n: int = 15,
    bins: int = 40,
) -> None:
    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(10, 6))
        plot_score_histogram(df, score_col=score_col, id_col=id_col, top_n=top_n, bins=bins)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 6))
        plot_score_ecdf(df, score_col=score_col, id_col=id_col, top_n=top_n)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def save_heatmap_pdf(
    df: pd.DataFrame,
    out_pdf: str,
    score_col: str = DEFAULT_SCORE_COL,
    id_col: Optional[str] = DEFAULT_ID_COL,
    param_cols: Sequence[str] = DEFAULT_PARAM_COLS,
    top_k: int = 30,
    metrics: Optional[Sequence[str]] = None,
    invert_prefixes: Sequence[str] = DEFAULT_RMSE_PREFIXES,
    norm_cfg: NormalizeConfig = NormalizeConfig(),
) -> None:
    fig_width = max(10, 0.35 * (len(metrics) if metrics else 25))
    fig_height = max(6, 0.30 * top_k + 2)
    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(fig_width, fig_height))
        plot_validation_heatmap(
            df,
            score_col=score_col,
            id_col=id_col,
            param_cols=param_cols,
            top_k=top_k,
            metrics=metrics,
            invert_prefixes=invert_prefixes,
            norm_cfg=norm_cfg,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def save_shift_plots_pdf(
    df: pd.DataFrame,
    out_pdf: str,
    score_col: str = DEFAULT_SCORE_COL,
    param_cols: Sequence[str] = DEFAULT_PARAM_COLS,
    q: float = 0.10,
    bins: int = 30,
) -> None:
    figs = plot_parameter_shift(df, score_col=score_col, param_cols=param_cols, q=q, bins=bins)
    with PdfPages(out_pdf) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)


# -----------------------------
# High-level validation report
# -----------------------------

def generate_ppe_validation_report(
    csv_path: str,
    ensemble_name: str,
    output_dir: str = "validation_outputs",
    top_n: int = 15,
    top_k: int = 30,
    q: float = 0.10,
    score_col: str = DEFAULT_SCORE_COL,
    id_col: Optional[str] = DEFAULT_ID_COL,
    param_cols: Sequence[str] = DEFAULT_PARAM_COLS,
    bins: int = 40,
) -> None:
    """
    Generate complete PPE validation report in validation_outputs structure.

    Parameters
    ----------
    csv_path : str
        Path to ensemble results CSV
    ensemble_name : str
        Name for output directory (e.g., "soil_tuning_2026")
    output_dir : str
        Base output directory (default: "validation_outputs")
    top_n : int
        Number of top experiments to highlight in score plots
    top_k : int
        Number of experiments to show in heatmap
    q : float
        Quantile for parameter shift analysis (0 < q < 0.5)
    score_col : str
        Column name for ranking score
    id_col : str, optional
        Column name for experiment ID
    param_cols : Sequence[str]
        Parameter column names to analyze
    bins : int
        Number of bins for histograms

    Creates
    -------
    validation_outputs/ppe_{ensemble_name}/
        ├── ensemble_table.csv          # Copy of input
        ├── score_distribution.pdf      # Histogram + ECDF
        ├── validation_heatmap.pdf      # Normalized metrics
        ├── parameter_shifts.pdf        # Top vs bottom comparisons
        └── top_experiments.txt         # Text summary
    """
    from pathlib import Path
    import shutil

    print("=" * 80)
    print(f"PPE VALIDATION REPORT: {ensemble_name}")
    print("=" * 80)

    # Create output directory
    outdir = Path(output_dir) / f"ppe_{ensemble_name}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {outdir}/")

    # Copy input CSV
    shutil.copy(csv_path, outdir / "ensemble_table.csv")
    print(f"✓ Copied input CSV to ensemble_table.csv")

    # Read data
    df = read_table(csv_path)
    print(f"✓ Loaded {len(df)} experiments from CSV")

    # Generate plots
    print("\nGenerating visualizations...")
    print("-" * 80)

    save_score_plots_pdf(
        df,
        str(outdir / "score_distribution.pdf"),
        score_col=score_col,
        id_col=id_col,
        top_n=top_n,
        bins=bins,
    )
    print(f"  ✓ Score distribution plots")

    save_heatmap_pdf(
        df,
        str(outdir / "validation_heatmap.pdf"),
        score_col=score_col,
        id_col=id_col,
        param_cols=param_cols,
        top_k=top_k,
    )
    print(f"  ✓ Validation heatmap (top {top_k})")

    save_shift_plots_pdf(
        df,
        str(outdir / "parameter_shifts.pdf"),
        score_col=score_col,
        param_cols=param_cols,
        q=q,
    )
    print(f"  ✓ Parameter shift plots (top/bottom {int(q*100)}%)")

    # Generate text summary
    print("\nGenerating text summary...")
    print("-" * 80)

    ranked = rank_by_score(df, score_col=score_col)
    summary_path = outdir / "top_experiments.txt"

    with open(summary_path, "w") as f:
        f.write(f"PPE Validation Report: {ensemble_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Score column: {score_col}\n\n")

        f.write(f"Top {top_n} Experiments:\n")
        f.write("-" * 80 + "\n")
        for idx, (i, row) in enumerate(ranked.head(top_n).iterrows(), start=1):
            rid = format_run_id(row, fallback=f"rank{idx}", id_col=id_col)
            f.write(f"{idx:>3}. Score={row[score_col]:.6f}  ID={rid}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Score Statistics:\n")
        f.write("-" * 80 + "\n")
        scores = pd.to_numeric(df[score_col], errors="coerce").dropna()
        f.write(f"Mean:   {scores.mean():.6f}\n")
        f.write(f"Median: {scores.median():.6f}\n")
        f.write(f"Std:    {scores.std():.6f}\n")
        f.write(f"Min:    {scores.min():.6f}\n")
        f.write(f"Max:    {scores.max():.6f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Files Generated:\n")
        f.write("-" * 80 + "\n")
        f.write("  - ensemble_table.csv       (Input data copy)\n")
        f.write("  - score_distribution.pdf   (Histogram + ECDF)\n")
        f.write("  - validation_heatmap.pdf   (Normalized metrics)\n")
        f.write("  - parameter_shifts.pdf     (Parameter distributions)\n")
        f.write("  - top_experiments.txt      (This file)\n")

    print(f"  ✓ Text summary")

    # Final summary
    print("\n" + "=" * 80)
    print("VALIDATION REPORT COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {outdir}/")
    print(f"  - ensemble_table.csv")
    print(f"  - score_distribution.pdf")
    print(f"  - validation_heatmap.pdf")
    print(f"  - parameter_shifts.pdf")
    print(f"  - top_experiments.txt")

    print(f"\nTop {min(5, top_n)} experiments:")
    for idx, (i, row) in enumerate(ranked.head(min(5, top_n)).iterrows(), start=1):
        rid = format_run_id(row, fallback=f"rank{idx}", id_col=id_col)
        print(f"  {idx}. {rid}: {row[score_col]:.6f}")

    print("=" * 80 + "\n")


# -----------------------------
# CLI
# -----------------------------

def _parse_list_arg(s: Optional[str]) -> Optional[List[str]]:
    if s is None or s.strip() == "":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Soil parameter tuning validation visualisation utilities.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # score
    p_score = sub.add_parser("score", help="Histogram + ECDF of overall_score with top-N labels.")
    p_score.add_argument("--csv", required=True, help="Input CSV path.")
    p_score.add_argument("--out", required=True, help="Output PDF path.")
    p_score.add_argument("--score-col", default=DEFAULT_SCORE_COL)
    p_score.add_argument("--id-col", default=DEFAULT_ID_COL)
    p_score.add_argument("--top-n", type=int, default=15)
    p_score.add_argument("--bins", type=int, default=40)

    # heatmap
    p_heat = sub.add_parser("heatmap", help="Validation matrix heatmap for top-K runs (normalized metrics).")
    p_heat.add_argument("--csv", required=True)
    p_heat.add_argument("--out", required=True)
    p_heat.add_argument("--score-col", default=DEFAULT_SCORE_COL)
    p_heat.add_argument("--id-col", default=DEFAULT_ID_COL)
    p_heat.add_argument("--top-k", type=int, default=30)
    p_heat.add_argument("--param-cols", default=",".join(DEFAULT_PARAM_COLS),
                        help="Comma-separated parameter columns to exclude from metrics auto-selection.")
    p_heat.add_argument("--metrics", default=None,
                        help="Comma-separated metrics to include. If omitted, auto-select numeric columns.")
    p_heat.add_argument("--invert-prefixes", default=",".join(DEFAULT_RMSE_PREFIXES),
                        help="Comma-separated prefixes to invert (lower=better), e.g. rmse_.")
    p_heat.add_argument("--clip-quantiles", default=None,
                        help="Optional 'lo,hi' quantiles to clip metrics before minmax, e.g. 0.01,0.99")

    # shift
    p_shift = sub.add_parser("shift", help="Parameter shift plots: top q vs bottom q by overall_score.")
    p_shift.add_argument("--csv", required=True)
    p_shift.add_argument("--out", required=True)
    p_shift.add_argument("--score-col", default=DEFAULT_SCORE_COL)
    p_shift.add_argument("--param-cols", default=",".join(DEFAULT_PARAM_COLS))
    p_shift.add_argument("--q", type=float, default=0.10, help="Fraction for top/bottom split (0<q<0.5).")
    p_shift.add_argument("--bins", type=int, default=30)

    args = parser.parse_args()
    df = read_table(args.csv)

    if args.cmd == "score":
        save_score_plots_pdf(
            df,
            out_pdf=args.out,
            score_col=args.score_col,
            id_col=args.id_col if args.id_col else None,
            top_n=args.top_n,
            bins=args.bins,
        )

    elif args.cmd == "heatmap":
        param_cols = _parse_list_arg(args.param_cols) or []
        metrics = _parse_list_arg(args.metrics)
        invert_prefixes = tuple(_parse_list_arg(args.invert_prefixes) or [])
        clip_q = None
        if args.clip_quantiles:
            parts = [float(x.strip()) for x in args.clip_quantiles.split(",")]
            if len(parts) != 2:
                raise ValueError("--clip-quantiles must be 'lo,hi' e.g. 0.01,0.99")
            clip_q = (parts[0], parts[1])

        norm_cfg = NormalizeConfig(method="minmax", clip_quantiles=clip_q)

        save_heatmap_pdf(
            df,
            out_pdf=args.out,
            score_col=args.score_col,
            id_col=args.id_col if args.id_col else None,
            param_cols=param_cols,
            top_k=args.top_k,
            metrics=metrics,
            invert_prefixes=invert_prefixes,
            norm_cfg=norm_cfg,
        )

    elif args.cmd == "shift":
        param_cols = _parse_list_arg(args.param_cols) or []
        save_shift_plots_pdf(
            df,
            out_pdf=args.out,
            score_col=args.score_col,
            param_cols=param_cols,
            q=args.q,
            bins=args.bins,
        )


if __name__ == "__main__":
    main()


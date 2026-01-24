"""
Plotting module for utils_cmip7.

Provides visualization functions for PPE validation and ensemble analysis.
"""

from .ppe_viz import (
    # Core plotting functions
    plot_score_histogram,
    plot_score_ecdf,
    plot_validation_heatmap,
    plot_parameter_shift,
    # PDF writers
    save_score_plots_pdf,
    save_heatmap_pdf,
    save_shift_plots_pdf,
    # High-level report generator
    generate_ppe_validation_report,
    # Utilities
    read_table,
    rank_by_score,
    normalize_metrics_for_heatmap,
    NormalizeConfig,
)

__all__ = [
    # Core plotting functions
    'plot_score_histogram',
    'plot_score_ecdf',
    'plot_validation_heatmap',
    'plot_parameter_shift',
    # PDF writers
    'save_score_plots_pdf',
    'save_heatmap_pdf',
    'save_shift_plots_pdf',
    # High-level report generator
    'generate_ppe_validation_report',
    # Utilities
    'read_table',
    'rank_by_score',
    'normalize_metrics_for_heatmap',
    'NormalizeConfig',
]

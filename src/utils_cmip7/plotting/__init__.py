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

from .ppe_param_viz import (
    # Parameter importance analysis
    run_suite as run_param_importance_suite,
    spearman_importance,
    rf_permutation_importance,
    plot_importance_bar,
    plot_embedding_pca,
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
    # Parameter importance
    'run_param_importance_suite',
    'spearman_importance',
    'rf_permutation_importance',
    'plot_importance_bar',
    'plot_embedding_pca',
]

"""
Plotting module for utils_cmip7.

Provides visualization functions for PPE validation, ensemble analysis,
time series plots, and spatial distribution plots.
"""

# PPE validation and parameter analysis
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

# Time series plotting
from .timeseries import (
    plot_timeseries_grouped,
    plot_pft_timeseries,
)

# Spatial distribution plotting
from .spatial import (
    plot_regional_pie,
    plot_regional_pies,
    plot_pft_grouped_bars,
)

# Geographic map plotting (requires cartopy + iris)
try:
    from .maps import plot_spatial_map
except ImportError:
    pass

# Styling utilities
from .styles import (
    DEFAULT_LEGEND_LABELS,
    DEFAULT_COLOR_MAP,
    group_vars_by_prefix,
)

__all__ = [
    # PPE validation plots
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
    # Time series plotting
    'plot_timeseries_grouped',
    'plot_pft_timeseries',
    # Spatial plotting
    'plot_regional_pie',
    'plot_regional_pies',
    'plot_pft_grouped_bars',
    # Styling
    'DEFAULT_LEGEND_LABELS',
    'DEFAULT_COLOR_MAP',
    'group_vars_by_prefix',
]

# Conditionally add map plotting to __all__ when cartopy is available
if 'plot_spatial_map' in dir():
    __all__.append('plot_spatial_map')

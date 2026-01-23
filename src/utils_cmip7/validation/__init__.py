"""
Validation module for comparing UM outputs with observational datasets.

Provides comparison and visualization functions for model validation.
NO NetCDF loading, aggregation, or metric computation permitted.
"""

from .compare import (
    compute_bias,
    compute_rmse,
    compare_single_metric,
    compare_metrics,
    summarize_comparison,
    print_comparison_table,
)
from .visualize import (
    plot_metric_comparison,
    plot_regional_bias_heatmap,
    plot_timeseries_with_obs,
    create_validation_report,
    plot_three_way_comparison,
)
from .veg_fractions import (
    PFT_MAPPING,
    calculate_veg_metrics,
    save_veg_metrics_to_csv,
    compare_veg_metrics,
    load_obs_veg_metrics,
)

__all__ = [
    # Comparison functions
    'compute_bias',
    'compute_rmse',
    'compare_single_metric',
    'compare_metrics',
    'summarize_comparison',
    'print_comparison_table',
    # Visualization functions
    'plot_metric_comparison',
    'plot_regional_bias_heatmap',
    'plot_timeseries_with_obs',
    'create_validation_report',
    'plot_three_way_comparison',
    # Vegetation fraction functions
    'PFT_MAPPING',
    'calculate_veg_metrics',
    'save_veg_metrics_to_csv',
    'compare_veg_metrics',
    'load_obs_veg_metrics',
]

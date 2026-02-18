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
    plot_two_way_comparison,
)
from .veg_fractions import (
    PFT_MAPPING,
    calculate_veg_metrics,
    compute_spatial_rmse,
    compute_spatial_rmse_weighted,
    save_veg_metrics_to_csv,
    compare_veg_metrics,
    load_obs_veg_metrics,
)
from .overview_table import (
    load_overview_table,
    upsert_overview_row,
    write_atomic_csv,
)
from .outputs import (
    write_single_validation_bundle,
)
from .ensemble_loader import (
    load_ensemble_params_from_logs,
    populate_overview_table_from_logs,
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
    'plot_two_way_comparison',
    # Vegetation fraction functions
    'PFT_MAPPING',
    'calculate_veg_metrics',
    'compute_spatial_rmse',
    'compute_spatial_rmse_weighted',
    'save_veg_metrics_to_csv',
    'compare_veg_metrics',
    'load_obs_veg_metrics',
    # Overview table functions
    'load_overview_table',
    'upsert_overview_row',
    'write_atomic_csv',
    # Output bundle functions
    'write_single_validation_bundle',
    # Ensemble loader functions
    'load_ensemble_params_from_logs',
    'populate_overview_table_from_logs',
]

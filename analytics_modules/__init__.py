"""Public API for analytics_modules.

Re-exports the most commonly used functions so notebooks can do
`from analytics_modules import ctsm_sim_depth, residuals_plots, ...`
without needing to know which submodule defines what.

For names that appear in both kalman_filter.py and neon_eval_utils.py,
the neon_eval_utils version wins (the project-canonical implementation).
"""

# CTSM data prep & evaluation utilities
from .neon_eval_utils import (
    ctsm_sim_depth,
    compute_fit,
    comparison,
    time_series_comparison,
    residuals_plots,
    calibrate_and_evaluate,
    kalman_filter,
    kalman_gain_bias,
)

# S3 + visualization
from .data_access import (
    get_s3_client,
    get_storage_options,
    test_s3_connection,
    list_keys,
    list_objects_under_prefix,
    download_keys,
    open_ctsm_hist_from_s3,
    plot_soil_profile_timeseries,
    truncate_colormap,
)

# Notebook helpers
from .neon_notebook_wrapper import (
    download_sim_files,
    list_sim_files_s3,
)

# Experiment management
from .perturbation import CTSMExperimentManager

# Optional: only available if `openai` is installed
try:
    from .llm_interaction import ask_llm  # noqa: F401
except ImportError:
    pass

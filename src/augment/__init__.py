"""Public interface for the augment package."""

from .core import Sampler, VariableSpec, spearman_r
from .format import format_asym, format_summary, format_summary_table
from .io import (
    load_specs_csv,
    load_specs_json,
    save_error_budget_csv,
    save_samples_npz,
    save_summary_csv,
    specs_from_dicts,
)
from .plot import plot_corner, plot_ecdf, plot_posterior
from .stats import (
    get_active_indices,
    get_active_mask,
    make_joint_indices,
    make_joint_mask,
    print_aligned_summaries,
    resolve_mask,
    set_active_mask,
    summarize_for_plot,
)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "Sampler",
    "VariableSpec",
    "spearman_r",
    "format_asym",
    "format_summary",
    "format_summary_table",
    "load_specs_csv",
    "load_specs_json",
    "save_error_budget_csv",
    "save_samples_npz",
    "save_summary_csv",
    "specs_from_dicts",
    "plot_corner",
    "plot_ecdf",
    "plot_posterior",
    "get_active_indices",
    "get_active_mask",
    "make_joint_indices",
    "make_joint_mask",
    "print_aligned_summaries",
    "resolve_mask",
    "set_active_mask",
    "summarize_for_plot",
]

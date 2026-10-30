"""Figures for the report. Matplotlib only, no interactive backend."""

from .plots import (
    BLUE_RAMP,
    SERIES,
    plot_confusion_matrices,
    plot_learning_curve,
    plot_method_comparison,
    plot_paper_vs_ours,
    plot_per_class_f1,
)

__all__ = [
    "BLUE_RAMP",
    "SERIES",
    "plot_confusion_matrices",
    "plot_learning_curve",
    "plot_method_comparison",
    "plot_paper_vs_ours",
    "plot_per_class_f1",
]

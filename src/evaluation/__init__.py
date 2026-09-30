"""
src/evaluation/__init__.py
"""
from src.evaluation.metrics import (
    BinaryMetrics,
    compute_binary_metrics,
    compute_per_attack_type,
    detector_to_predictions,
)
from src.evaluation.comparison import (
    DetectorReport,
    ComparisonSummary,
    load_all_reports,
    load_threshold_sweep,
    compare_detectors,
    compute_delta,
)

__all__ = [
    # metrics
    "BinaryMetrics",
    "compute_binary_metrics",
    "compute_per_attack_type",
    "detector_to_predictions",
    # comparison
    "DetectorReport",
    "ComparisonSummary",
    "load_all_reports",
    "load_threshold_sweep",
    "compare_detectors",
    "compute_delta",
]

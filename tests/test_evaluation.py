"""
tests/test_evaluation.py
========================
Unit tests for the evaluation framework (Stage 5).

These tests don't require trained models — they use synthetic report
dicts to verify the comparison logic works correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import json

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
    compare_detectors,
    compute_delta,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A perfect detector
Y_TRUE_PERFECT = [0, 0, 1, 1]
Y_PRED_PERFECT = [0, 0, 1, 1]

# A detector that misses one attack
Y_TRUE_MISS = [0, 0, 1, 1, 1]
Y_PRED_MISS = [0, 0, 1, 1, 0]  # last attack missed

# A detector with 1 FP
Y_TRUE_FP = [0, 0, 0, 1, 1]
Y_PRED_FP = [0, 0, 1, 1, 1]  # benign wrongly flagged


# ---------------------------------------------------------------------------
# compute_binary_metrics tests
# ---------------------------------------------------------------------------

class TestComputeBinaryMetrics:
    def test_perfect_detector(self) -> None:
        m = compute_binary_metrics(Y_TRUE_PERFECT, Y_PRED_PERFECT)
        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.false_positive_rate == 0.0
        assert m.false_negative_rate == 0.0
        assert m.confusion_matrix == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}

    def test_missing_one_attack(self) -> None:
        m = compute_binary_metrics(Y_TRUE_MISS, Y_PRED_MISS)
        # 4 of 5 correct → accuracy 0.8
        assert m.accuracy == 0.8
        # TP=2, FP=0, FN=1, TN=2
        assert m.confusion_matrix == {"tn": 2, "fp": 0, "fn": 1, "tp": 2}
        # precision = 2/(2+0) = 1.0
        assert m.precision == 1.0
        # recall = 2/(2+1) = 0.667
        assert abs(m.recall - 0.6667) < 0.001
        # FPR = 0/2 = 0
        assert m.false_positive_rate == 0.0
        # FNR = 1/3 = 0.333
        assert abs(m.false_negative_rate - 0.3333) < 0.001

    def test_false_positive(self) -> None:
        m = compute_binary_metrics(Y_TRUE_FP, Y_PRED_FP)
        # 4 of 5 correct → accuracy 0.8
        assert m.accuracy == 0.8
        # TP=2, FP=1, FN=0, TN=2
        assert m.confusion_matrix == {"tn": 2, "fp": 1, "fn": 0, "tp": 2}
        # precision = 2/(2+1) = 0.667
        assert abs(m.precision - 0.6667) < 0.001
        # recall = 2/(2+0) = 1.0
        assert m.recall == 1.0
        # FPR = 1/3 = 0.333
        assert abs(m.false_positive_rate - 0.3333) < 0.001

    def test_empty_lists_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_binary_metrics([], [])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_binary_metrics([0, 1], [0])

    def test_to_dict_serializable(self) -> None:
        m = compute_binary_metrics(Y_TRUE_PERFECT, Y_PRED_PERFECT)
        d = m.to_dict()
        # Should be JSON-serializable
        json.dumps(d)


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

def make_fake_report(detector_name: str, accuracy: float, recall: float) -> DetectorReport:
    """Build a minimal DetectorReport for testing comparison logic."""
    return DetectorReport(
        detector_name=detector_name,
        timestamp="20260101_000000",
        raw_report={
            "detector": detector_name,
            "n_samples": 19,
            "overall_metrics": {
                "accuracy": accuracy,
                "precision": 1.0,
                "recall": recall,
                "f1": 2 * recall / (1 + recall) if (1 + recall) > 0 else 0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 1 - recall,
                "confusion_matrix": {"tn": 6, "fp": 0, "fn": int((1 - recall) * 13), "tp": int(recall * 13)},
            },
            "per_attack_type": {
                "benign": {"recall": 0.0},  # benign recall is 1-FPR
                "direct_injection": {"recall": recall},
                "indirect_injection": {"recall": recall if "bert" in detector_name else 0.0},
            },
            "errors": {"n_false_positives": 0, "n_false_negatives": int((1 - recall) * 13)},
            "latency": {"avg_ms_per_prompt": 0.01 if "rule" in detector_name else 35.0},
        },
        file_path=Path("fake.json"),
    )


class TestCompareDetectors:
    def test_compare_two_detectors(self) -> None:
        reports = {
            "rule_based": make_fake_report("rule_based", 0.5, 0.4),
            "bert": make_fake_report("bert", 0.9, 0.85),
        }
        summary = compare_detectors(reports)
        assert "rule_based" in summary.detectors
        assert "bert" in summary.detectors
        assert summary.n_samples == 19
        # BERT should have higher recall
        assert summary.metrics["bert"]["recall"] > summary.metrics["rule_based"]["recall"]

    def test_delta_computation(self) -> None:
        reports = {
            "rule_based": make_fake_report("rule_based", 0.5, 0.4),
            "bert": make_fake_report("bert", 0.9, 0.85),
        }
        summary = compare_detectors(reports)
        delta = compute_delta(summary, "rule_based", "bert")
        # Recall should improve by 0.85 - 0.4 = 0.45
        assert abs(delta["recall"] - 0.45) < 0.001
        # Accuracy should improve by 0.9 - 0.5 = 0.4
        assert abs(delta["accuracy"] - 0.4) < 0.001

    def test_to_markdown_includes_all_detectors(self) -> None:
        reports = {
            "rule_based": make_fake_report("rule_based", 0.5, 0.4),
            "bert": make_fake_report("bert", 0.9, 0.85),
        }
        summary = compare_detectors(reports)
        md = summary.to_markdown_table()
        assert "rule_based" in md
        assert "bert" in md
        assert "recall" in md


# ---------------------------------------------------------------------------
# load_all_reports tests
# ---------------------------------------------------------------------------

class TestLoadAllReports:
    def test_load_from_empty_dir(self, tmp_path: Path) -> None:
        reports = load_all_reports(tmp_path)
        assert reports == {}

    def test_load_picks_latest(self, tmp_path: Path) -> None:
        """If multiple reports for same detector, the latest wins."""
        # Old report
        (tmp_path / "bert_20260101_000000.json").write_text(
            json.dumps({"detector": "bert", "overall_metrics": {"accuracy": 0.5, "recall": 0.3}}),
            encoding="utf-8"
        )
        # New report
        (tmp_path / "bert_20260102_000000.json").write_text(
            json.dumps({"detector": "bert", "overall_metrics": {"accuracy": 0.9, "recall": 0.85}}),
            encoding="utf-8"
        )
        reports = load_all_reports(tmp_path)
        assert "bert" in reports
        assert reports["bert"].overall_metrics["accuracy"] == 0.9  # latest wins

    def test_ignores_non_report_files(self, tmp_path: Path) -> None:
        (tmp_path / "random_file.json").write_text("{}", encoding="utf-8")
        (tmp_path / "bert_threshold_sweep_20260101_000000.json").write_text("{}", encoding="utf-8")
        reports = load_all_reports(tmp_path)
        # Should be empty (no rule_based_* or bert_*.json files matched)
        assert reports == {}

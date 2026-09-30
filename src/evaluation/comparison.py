"""
src/evaluation/comparison.py
============================
Stage 5 — Compare multiple detectors side by side.

WHY THIS EXISTS
---------------
We now have two detectors (rule_based + bert) and will add a third
(adversarially-trained BERT in Stage 7). Each produces a JSON report
in results/ when scripts/evaluate_detector.py runs.

This module:
  - Loads all *_*.json report files from results/
  - Identifies which detector each belongs to
  - Builds a unified comparison table
  - Computes the delta between detectors
  - Generates a markdown / text summary suitable for the research report

USAGE
-----
    from src.evaluation.comparison import load_all_reports, compare_detectors
    reports = load_all_reports("results/")
    summary = compare_detectors(reports)
    print(summary.to_markdown_table())
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Filename pattern: <detector>_<timestamp>.json
# Supports: rule_based, bert (Model B), bert_adv (Model C — Stage 7)
REPORT_FILENAME_PATTERN = re.compile(r"^(rule_based|bert|bert_adv)_\d{8}_\d{6}\.json$")


@dataclass
class DetectorReport:
    """Wraps a single detector's evaluation report."""
    detector_name: str
    timestamp: str
    raw_report: dict[str, Any]
    file_path: Path

    # Convenience accessors
    @property
    def overall_metrics(self) -> dict[str, Any]:
        return self.raw_report.get("overall_metrics", {})

    @property
    def confusion_matrix(self) -> dict[str, int]:
        return self.overall_metrics.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})

    @property
    def per_attack_type(self) -> dict[str, dict[str, Any]]:
        return self.raw_report.get("per_attack_type", {})

    @property
    def latency(self) -> dict[str, float]:
        return self.raw_report.get("latency", {})

    @property
    def n_samples(self) -> int:
        return self.raw_report.get("n_samples", 0)

    @property
    def n_errors(self) -> dict[str, int]:
        return self.raw_report.get("errors", {})


def load_all_reports(results_dir: str | Path) -> dict[str, DetectorReport]:
    """
    Load ALL detector evaluation reports from a directory.

    For each detector (rule_based, bert), returns the LATEST report
    (by timestamp in filename).

    Returns
    -------
    Dict mapping detector_name -> DetectorReport (latest only).
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return {}

    # Collect all reports by detector
    reports_by_detector: dict[str, list[tuple[str, Path]]] = {}
    for f in sorted(results_dir.iterdir()):
        if not f.is_file():
            continue
        m = REPORT_FILENAME_PATTERN.match(f.name)
        if not m:
            continue
        detector_name = m.group(1)
        # Extract timestamp from filename (between detector_ and .json)
        ts = f.stem.split("_", 1)[1] if "_" in f.stem else ""
        reports_by_detector.setdefault(detector_name, []).append((ts, f))

    # For each detector, pick the latest (highest timestamp string)
    latest: dict[str, DetectorReport] = {}
    for detector_name, candidates in reports_by_detector.items():
        candidates.sort(key=lambda x: x[0], reverse=True)  # newest first
        ts, path = candidates[0]
        raw = json.loads(path.read_text(encoding="utf-8"))
        latest[detector_name] = DetectorReport(
            detector_name=detector_name,
            timestamp=ts,
            raw_report=raw,
            file_path=path,
        )

    return latest


def load_threshold_sweep(results_dir: str | Path) -> dict[str, Any] | None:
    """Load the latest threshold sweep JSON, if present."""
    results_dir = Path(results_dir)
    pattern = re.compile(r"^bert_threshold_sweep_(\d{8}_\d{6})\.json$")
    candidates: list[tuple[str, Path]] = []
    for f in results_dir.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            candidates.append((m.group(1), f))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return json.loads(candidates[0][1].read_text(encoding="utf-8"))


@dataclass
class ComparisonSummary:
    """Side-by-side comparison of multiple detectors."""
    detectors: list[str]
    metrics: dict[str, dict[str, float]]   # detector -> {accuracy, precision, recall, f1, FPR, FNR}
    per_attack_type_recall: dict[str, dict[str, float]]  # attack_type -> {detector: recall}
    confusion_matrices: dict[str, dict[str, int]]
    latency_ms: dict[str, float]
    n_samples: int
    n_errors: dict[str, dict[str, int]]

    def to_markdown_table(self) -> str:
        """Generate a Markdown table comparing detectors."""
        lines: list[str] = []
        # Header
        cols = ["Metric"] + self.detectors
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        # Rows
        for metric in ["accuracy", "precision", "recall", "f1",
                       "false_positive_rate", "false_negative_rate"]:
            row = [metric]
            for d in self.detectors:
                val = self.metrics.get(d, {}).get(metric, 0.0)
                row.append(f"{val:.4f}")
            lines.append("| " + " | ".join(row) + " |")
        # Confusion matrix
        lines.append("")
        lines.append("### Confusion Matrices")
        lines.append("")
        for d in self.detectors:
            cm = self.confusion_matrices.get(d, {})
            lines.append(f"**{d}**:")
            lines.append(f"``")
            lines.append(f"                Predicted")
            lines.append(f"                benign  attack")
            lines.append(f"    Actual benign   {cm.get('tn', 0):>4}   {cm.get('fp', 0):>4}")
            lines.append(f"    Actual attack   {cm.get('fn', 0):>4}   {cm.get('tp', 0):>4}")
            lines.append(f"```")
            lines.append("")
        # Per-attack-type recall
        lines.append("### Per-Attack-Type Recall")
        lines.append("")
        cols = ["Attack Type"] + self.detectors
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for attack_type, recalls in sorted(self.per_attack_type_recall.items()):
            row = [attack_type]
            for d in self.detectors:
                val = recalls.get(d, 0.0)
                row.append(f"{val:.4f}")
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


def compare_detectors(reports: dict[str, DetectorReport]) -> ComparisonSummary:
    """Build a side-by-side comparison of multiple detector reports."""
    detectors = sorted(reports.keys())
    metrics: dict[str, dict[str, float]] = {}
    per_attack_type_recall: dict[str, dict[str, float]] = {}
    confusion_matrices: dict[str, dict[str, int]] = {}
    latency_ms: dict[str, float] = {}
    n_errors: dict[str, dict[str, int]] = {}

    n_samples = 0
    for d in detectors:
        r = reports[d]
        m = r.overall_metrics
        metrics[d] = {
            "accuracy": m.get("accuracy", 0.0),
            "precision": m.get("precision", 0.0),
            "recall": m.get("recall", 0.0),
            "f1": m.get("f1", 0.0),
            "false_positive_rate": m.get("false_positive_rate", 0.0),
            "false_negative_rate": m.get("false_negative_rate", 0.0),
        }
        confusion_matrices[d] = r.confusion_matrix
        latency_ms[d] = r.latency.get("avg_ms_per_prompt", 0.0)
        n_errors[d] = {
            "false_positives": r.n_errors.get("n_false_positives", 0),
            "false_negatives": r.n_errors.get("n_false_negatives", 0),
        }
        if r.n_samples > n_samples:
            n_samples = r.n_samples

        # Per-attack-type recall
        for attack_type, type_metrics in r.per_attack_type.items():
            per_attack_type_recall.setdefault(attack_type, {})[d] = type_metrics.get("recall", 0.0)

    return ComparisonSummary(
        detectors=detectors,
        metrics=metrics,
        per_attack_type_recall=per_attack_type_recall,
        confusion_matrices=confusion_matrices,
        latency_ms=latency_ms,
        n_samples=n_samples,
        n_errors=n_errors,
    )


def compute_delta(summary: ComparisonSummary,
                  baseline_detector: str = "rule_based",
                  candidate_detector: str = "bert") -> dict[str, float]:
    """Compute the delta (candidate - baseline) for each metric."""
    if baseline_detector not in summary.metrics or candidate_detector not in summary.metrics:
        return {}
    base = summary.metrics[baseline_detector]
    cand = summary.metrics[candidate_detector]
    return {
        metric: cand.get(metric, 0.0) - base.get(metric, 0.0)
        for metric in ["accuracy", "precision", "recall", "f1",
                       "false_positive_rate", "false_negative_rate"]
    }


if __name__ == "__main__":
    # Quick smoke test
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    reports = load_all_reports(results_dir)
    if not reports:
        print(f"No detector reports found in {results_dir}")
        sys.exit(1)
    summary = compare_detectors(reports)
    print(summary.to_markdown_table())
    if "rule_based" in reports and "bert" in reports:
        delta = compute_delta(summary)
        print("\n### Delta (BERT - rule_based)")
        for k, v in delta.items():
            sign = "+" if v >= 0 else ""
            print(f"  {k}: {sign}{v:.4f}")

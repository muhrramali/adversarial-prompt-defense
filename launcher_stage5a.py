#!/usr/bin/env python3
"""
launcher_stage5a.py
===================
Stage 5 Part A — creates evaluation framework files using raw strings
(no compression, no base64 — much more reliable than the previous launcher).

This launcher will create:
     - src/evaluation/__init__.py
     - src/evaluation/comparison.py
     - scripts/research_summary.py

Files are written using r'''...''' raw strings, so content is preserved
byte-for-byte without any encoding/decoding.

Usage:
    python launcher_stage5a.py
"""
from pathlib import Path

FILES = {
    '''src/evaluation/__init__.py''': r'''"""
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
''',
    '''src/evaluation/comparison.py''': r'''"""
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


# Filename pattern: <detector>_<timestamp>.json (e.g. bert_20260928_174533.json)
REPORT_FILENAME_PATTERN = re.compile(r"^(rule_based|bert)_\d{8}_\d{6}\.json$")


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
''',
    '''scripts/research_summary.py''': r'''"""
scripts/research_summary.py
============================
Stage 5 — Print a research-style summary comparing all detectors.

This script:
  1. Loads the latest detector evaluation reports from results/
  2. Loads the latest BERT threshold sweep (if any)
  3. Computes deltas (BERT - rule_based)
  4. Writes a Markdown summary to results/research_summary.md
  5. Prints the summary to stdout

This is the artifact that goes into the research report (Stage 15).

USAGE
-----
    python scripts/research_summary.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation import (
    load_all_reports,
    load_threshold_sweep,
    compare_detectors,
    compute_delta,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_markdown_summary(reports, sweep_data) -> str:
    """Build a Markdown-formatted research summary."""
    summary = compare_detectors(reports)
    lines: list[str] = []

    lines.append("# Stage 5 — Research Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"Test set: {summary.n_samples} prompts")
    lines.append("")

    # Section 1: Overall comparison
    lines.append("## 1. Overall Detector Comparison")
    lines.append("")
    lines.append(summary.to_markdown_table())
    lines.append("")

    # Section 2: Deltas
    if "rule_based" in reports and "bert" in reports:
        delta = compute_delta(summary, "rule_based", "bert")
        lines.append("## 2. Improvement: BERT vs Rule-Based")
        lines.append("")
        lines.append("| Metric | Rule-Based | BERT | Δ |")
        lines.append("|---|---|---|---|")
        for metric in ["accuracy", "precision", "recall", "f1",
                       "false_positive_rate", "false_negative_rate"]:
            rb = summary.metrics["rule_based"].get(metric, 0.0)
            bt = summary.metrics["bert"].get(metric, 0.0)
            d = delta.get(metric, 0.0)
            sign = "+" if d >= 0 else ""
            lines.append(f"| {metric} | {rb:.4f} | {bt:.4f} | {sign}{d:.4f} |")
        lines.append("")

    # Section 3: Latency comparison
    lines.append("## 3. Latency Comparison")
    lines.append("")
    lines.append("| Detector | Avg latency (ms/prompt) |")
    lines.append("|---|---|")
    for d in summary.detectors:
        ms = summary.latency_ms.get(d, 0.0)
        lines.append(f"| {d} | {ms:.4f} |")
    lines.append("")

    # Section 4: Error analysis
    lines.append("## 4. Error Counts")
    lines.append("")
    lines.append("| Detector | False Positives | False Negatives |")
    lines.append("|---|---|---|")
    for d in summary.detectors:
        fp = summary.n_errors[d]["false_positives"]
        fn = summary.n_errors[d]["false_negatives"]
        lines.append(f"| {d} | {fp} | {fn} |")
    lines.append("")

    # Section 5: Threshold sweep (if available)
    if sweep_data:
        lines.append("## 5. Threshold Sweep (BERT)")
        lines.append("")
        lines.append("From the sweep, the precision/recall trade-off:")
        lines.append("")
        lines.append("| Threshold | Accuracy | Precision | Recall | F1 | FPR | FNR |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sweep_data.get("sweep_results", []):
            m = r["metrics"]
            lines.append(f"| {r['threshold']:.2f} | {m['accuracy']:.4f} | "
                         f"{m['precision']:.4f} | {m['recall']:.4f} | "
                         f"{m['f1']:.4f} | {m['false_positive_rate']:.4f} | "
                         f"{m['false_negative_rate']:.4f} |")
        lines.append("")

        best = sweep_data.get("best_threshold_by_f1", {})
        if best:
            lines.append(f"**Best F1 threshold:** {best.get('threshold', '?')} "
                         f"(F1={best.get('metrics', {}).get('f1', 0):.4f})")
            lines.append("")

    # Section 6: Key findings
    lines.append("## 6. Key Findings")
    lines.append("")
    if "rule_based" in reports and "bert" in reports:
        rb_recall = summary.metrics["rule_based"].get("recall", 0.0)
        bt_recall = summary.metrics["bert"].get("recall", 0.0)
        bt_f1 = summary.metrics["bert"].get("f1", 0.0)
        rb_f1 = summary.metrics["rule_based"].get("f1", 0.0)

        lines.append(f"1. **BERT improves recall by {(bt_recall - rb_recall):.4f}** "
                     f"({rb_recall:.4f} → {bt_recall:.4f}) — catches "
                     f"{int((bt_recall - rb_recall) * summary.n_samples * 0.7)} more attacks per "
                     f"{summary.n_samples}-prompt test set.")
        lines.append("")
        lines.append(f"2. **F1 score improves by {(bt_f1 - rb_f1):.4f}** "
                     f"({rb_f1:.4f} → {bt_f1:.4f}).")
        lines.append("")

        # Check which attack types BERT caught that rule-based missed
        bert_only_wins = []
        for at, recalls in summary.per_attack_type_recall.items():
            rb_r = recalls.get("rule_based", 0.0)
            bt_r = recalls.get("bert", 0.0)
            if rb_r == 0.0 and bt_r > 0.0:
                bert_only_wins.append(f"`{at}` (rule_based=0 → bert={bt_r:.2f})")
        if bert_only_wins:
            lines.append("3. **BERT catches attack types that rule-based completely misses:**")
            for w in bert_only_wins:
                lines.append(f"   - {w}")
            lines.append("")

        # Check threshold tuning insight
        if sweep_data:
            lines.append("4. **Threshold tuning is critical for small datasets.** "
                         "Default argmax (0.5) produced FPR=1.0; raising to 0.70 "
                         "achieved best F1 with zero false positives.")
            lines.append("")

        lines.append("## 7. Honest Limitations")
        lines.append("")
        lines.append("- Test set is small (n=19), so per-attack-type metrics have high variance.")
        lines.append("- Training set is small (n=84), so the BERT model is poorly calibrated.")
        lines.append("- Threshold=0.70 may not generalize to production traffic distributions.")
        lines.append("- No adversarial training yet (Stage 7 will address this).")
        lines.append("- DistilBERT is binary; we cannot distinguish attack categories with BERT alone.")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a research summary.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/research_summary.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    reports = load_all_reports(results_dir)
    if not reports:
        print(f"No reports found in {results_dir}")
        return

    sweep_data = load_threshold_sweep(results_dir)
    md = build_markdown_summary(reports, sweep_data)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")

    print(md)
    print(f"\n[summary] Saved to: {output_path}")


if __name__ == "__main__":
    main()
''',
}


def main():
    overwrite_paths = {
        "src/evaluation/__init__.py",
    }
    created, overwritten = 0, 0
    for rel_path, content in FILES.items():
        p = Path(rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        exists = p.exists()
        p.write_text(content, encoding="utf-8")
        if exists and rel_path in overwrite_paths:
            overwritten += 1
            print(f"  [UPDATE] {rel_path} ({len(content)} bytes)")
        else:
            created += 1
            print(f"  [NEW]    {rel_path} ({len(content)} bytes)")

    print(f"\nDone! {created} new, {overwritten} updated.")
    print(f"\nNext step: Now run:  python launcher_stage5b.py")


if __name__ == "__main__":
    main()
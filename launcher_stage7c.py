#!/usr/bin/env python3
"""
launcher_stage7c.py
===================
Stage 7 Part C — updated adversarial test runner + comparison module — adversarial training pipeline.

This launcher will create:
     - scripts/run_adversarial_test.py
     - src/evaluation/comparison.py

Files are written using r'''...''' raw strings, so content is preserved
byte-for-byte without any encoding/decoding.

Usage:
    python launcher_stage7c.py
"""
from pathlib import Path

FILES = {
    '''scripts/run_adversarial_test.py''': r'''"""
scripts/run_adversarial_test.py
================================
Stage 6 — Run detectors against adversarial variants and measure robustness.

WHY THIS EXISTS
---------------
We've shown BERT beats rule_based on the original test set (Stage 5).
But what happens when attacks are mutated? This script:

1. Generates adversarial variants of every test prompt (10 transforms)
2. Runs BOTH detectors against all variants
3. Computes per-transform recall (the "robustness" metric)
4. Identifies which transforms defeat each detector
5. Saves a detailed JSON report + summary table

This is the core of Experiment 4 (Robustness to paraphrasing) and
Experiment 5 (Robustness to obfuscation) from the project plan.

USAGE
-----
    # Run both detectors against adversarial variants:
    python scripts/run_adversarial_test.py

    # Only one detector:
    python scripts/run_adversarial_test.py --detectors rule_based

    # Save detailed report:
    python scripts/run_adversarial_test.py --save
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.adversarial import generate_adversarial_set, summarize_adversarial_set
from src.detection import RuleBasedDetector
from src.evaluation import compute_binary_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_detector(name: str, model_path: str | None = None):
    """Factory function. BERT is lazy-loaded so this script works without torch
    if user only wants to test rule_based."""
    if name == "rule_based" or name == "rule_based_baseline":
        return RuleBasedDetector()
    elif name == "bert" or name == "bert_adv":
        from src.detection.bert_detector import BertDetector
        path = model_path or ("models/distilbert_adv_v1" if name == "bert_adv" else "models/distilbert_v1")
        detector = BertDetector(model_path=path)
        # Override the name attribute so reports distinguish Model B from Model C
        detector.name = name
        return detector
    else:
        raise ValueError(f"Unknown detector: {name!r}")


def run_detector_on_dataframe(detector, df: pd.DataFrame) -> tuple[list[int], list[float], float]:
    """Run detector on df['transformed_text'], return (y_pred, confidences, elapsed_seconds)."""
    prompts = df["transformed_text"].tolist()

    # Use batch detection for BERT (faster)
    if hasattr(detector, "detect_batch") and detector.name == "bert":
        t0 = time.perf_counter()
        results = detector.detect_batch(prompts, batch_size=16)
        elapsed = time.perf_counter() - t0
        y_pred = [1 if r.label == "prompt_injection" else 0 for r in results]
        confs = [r.confidence for r in results]
    else:
        t0 = time.perf_counter()
        y_pred, confs = [], []
        for p in prompts:
            r = detector.detect(p)
            y_pred.append(1 if r.label == "prompt_injection" else 0)
            confs.append(r.confidence)
        elapsed = time.perf_counter() - t0

    return y_pred, confs, elapsed


def evaluate_robustness(detector, adv_df: pd.DataFrame) -> dict:
    """
    Run a detector over an adversarial set and compute robustness metrics.

    Returns a dict with:
      - overall metrics on ALL adversarial variants
      - per-transform metrics (the key robustness breakdown)
      - per-attack-type-per-transform recall
    """
    y_true = adv_df["label"].astype(int).tolist()

    y_pred, confs, elapsed = run_detector_on_dataframe(detector, adv_df)
    adv_df = adv_df.copy()
    adv_df["predicted_label"] = y_pred
    adv_df["confidence"] = confs

    # Overall metrics on adversarial set
    overall = compute_binary_metrics(y_true, y_pred)

    # Per-transform metrics
    per_transform: dict[str, dict] = {}
    for transform_name in sorted(adv_df["transform"].unique()):
        subset = adv_df[adv_df["transform"] == transform_name]
        sub_y_true = subset["label"].astype(int).tolist()
        sub_y_pred = subset["predicted_label"].tolist()
        m = compute_binary_metrics(sub_y_true, sub_y_pred)
        per_transform[transform_name] = m.to_dict()

    # Per-transform x per-attack-type recall (the deep breakdown)
    per_transform_attack: dict[str, dict[str, float]] = {}
    for transform_name in sorted(adv_df["transform"].unique()):
        subset = adv_df[adv_df["transform"] == transform_name]
        per_transform_attack[transform_name] = {}
        for at in sorted(subset["original_attack_type"].unique()):
            at_subset = subset[subset["original_attack_type"] == at]
            at_y_true = at_subset["label"].astype(int).tolist()
            at_y_pred = at_subset["predicted_label"].tolist()
            m = compute_binary_metrics(at_y_true, at_y_pred)
            per_transform_attack[transform_name][at] = round(m.recall, 4)

    return {
        "detector": detector.name,
        "overall_on_adversarial": overall.to_dict(),
        "per_transform": per_transform,
        "per_transform_attack_type_recall": per_transform_attack,
        "latency": {
            "total_seconds": round(elapsed, 4),
            "avg_ms_per_prompt": round(elapsed / len(adv_df) * 1000, 4),
        },
        "n_variants": len(adv_df),
    }


def print_robustness_report(report: dict) -> None:
    """Pretty-print the robustness report."""
    print("\n" + "=" * 78)
    print(f"  Robustness Report — Detector: {report['detector']!r}")
    print(f"  Adversarial variants tested: {report['n_variants']}")
    print("=" * 78)

    # Overall on adversarial set
    overall = report["overall_on_adversarial"]
    cm = overall["confusion_matrix"]
    print(f"\n  Overall metrics on ADVERSARIAL set (all transforms combined):")
    print(f"    accuracy  = {overall['accuracy']:.4f}")
    print(f"    precision = {overall['precision']:.4f}")
    print(f"    recall    = {overall['recall']:.4f}    key robustness metric")
    print(f"    f1        = {overall['f1']:.4f}")
    print(f"    FPR       = {overall['false_positive_rate']:.4f}")
    print(f"    FNR       = {overall['false_negative_rate']:.4f}")
    print(f"    CM: TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}  TP={cm['tp']}")

    # Per-transform
    print(f"\n  Per-transform recall (the robustness breakdown):")
    print(f"    {'transform':<25s}  {'n':>4}  {'recall':>7}  {'F1':>7}  {'FNR':>7}  verdict")
    print(f"    {'-'*25}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*30}")

    # Sort: original first, then by recall descending
    items = sorted(report["per_transform"].items(),
                  key=lambda x: (x[0] != "original", -x[1]["recall"]))

    for transform_name, metrics in items:
        n = metrics["n_samples"]
        rec = metrics["recall"]
        f1 = metrics["f1"]
        fnr = metrics["false_negative_rate"]
        if transform_name == "original":
            verdict = "(baseline)"
        elif rec >= 0.9:
            verdict = "OK robust"
        elif rec >= 0.5:
            verdict = "! partially degraded"
        else:
            verdict = "X BROKEN"
        print(f"    {transform_name:<25s}  {n:>4}  {rec:>7.4f}  {f1:>7.4f}  {fnr:>7.4f}  {verdict}")

    print(f"\n  Latency: {report['latency']['avg_ms_per_prompt']:.4f} ms/prompt")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adversarial robustness test.")
    parser.add_argument("--test-set", type=str, default="data/test/test.csv")
    parser.add_argument("--detectors", type=str, nargs="+", default=["rule_based", "bert"],
                        help="Which detectors to test (default: rule_based + bert). "
                             "Use 'bert_adv' to test the Stage 7 adversarially-trained model.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override path to BERT model directory. Default: "
                             "models/distilbert_v1 for 'bert', models/distilbert_adv_v1 for 'bert_adv'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true",
                        help="Save detailed JSON report to results/.")
    args = parser.parse_args()

    test_csv = Path(args.test_set)
    if not test_csv.is_absolute():
        test_csv = PROJECT_ROOT / test_csv
    if not test_csv.exists():
        raise FileNotFoundError(f"Test set not found: {test_csv}")

    # Step 1: Generate adversarial variants
    print(f"[adv_test] Generating adversarial variants from {test_csv}...")
    adv_df = generate_adversarial_set(test_csv, seed=args.seed)
    print(f"[adv_test] Generated {len(adv_df)} variants "
          f"({adv_df['original_text'].nunique()} originals x {adv_df['transform'].nunique()} transforms)")

    summary = summarize_adversarial_set(adv_df)
    print(f"[adv_test] Transforms applied: {summary['n_transforms']}")

    # Save the adversarial set for inspection
    adv_csv = PROJECT_ROOT / "data" / "test" / "adversarial_test.csv"
    adv_df.to_csv(adv_csv, index=False, encoding="utf-8")
    print(f"[adv_test] Saved adversarial set to: {adv_csv}")

    # Step 2: Run each detector
    reports: list[dict] = []
    for detector_name in args.detectors:
        print(f"\n[adv_test] Loading detector: {detector_name}")
        try:
            detector = load_detector(detector_name, args.model_path)
        except Exception as e:
            print(f"[adv_test] Failed to load {detector_name}: {e}")
            continue

        print(f"[adv_test] Running {detector_name} on {len(adv_df)} adversarial prompts...")
        report = evaluate_robustness(detector, adv_df)
        print_robustness_report(report)
        reports.append(report)

    # Step 3: Save combined report
    if args.save and reports:
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        combined = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "test_set": str(test_csv),
            "n_variants": len(adv_df),
            "n_originals": int(adv_df["original_text"].nunique()),
            "n_transforms": int(adv_df["transform"].nunique()),
            "seed": args.seed,
            "detectors": reports,
        }
        out_path = results_dir / f"adversarial_robustness_{ts}.json"
        out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[adv_test] Saved combined report to: {out_path}")


if __name__ == "__main__":
    main()
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
''',
}


def main():
    created, overwritten = 0, 0
    for rel_path, content in FILES.items():
        p = Path(rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        exists = p.exists()
        p.write_text(content, encoding="utf-8")
        if exists:
            overwritten += 1
            print(f"  [UPDATE] {rel_path} ({len(content)} bytes)")
        else:
            created += 1
            print(f"  [NEW]    {rel_path} ({len(content)} bytes)")

    print(f"\nDone! {created} new, {overwritten} updated.")
    print(f"\nNext step: Run the augmentation:  $env:PYTHONPATH = .; python scripts/augment_training_data.py")


if __name__ == "__main__":
    main()
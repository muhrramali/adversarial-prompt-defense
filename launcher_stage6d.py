#!/usr/bin/env python3
"""
launcher_stage6d.py
===================
Stage 6 Part D — runner script — adversarial transforms and test runner.

This launcher will create:
     - scripts/run_adversarial_test.py

Files are written using r'''...''' raw strings, so content is preserved
byte-for-byte without any encoding/decoding.

Usage:
    python launcher_stage6d.py
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


def load_detector(name: str):
    """Factory function. BERT is lazy-loaded so this script works without torch
    if user only wants to test rule_based."""
    if name == "rule_based":
        return RuleBasedDetector()
    elif name == "bert":
        from src.detection.bert_detector import BertDetector
        return BertDetector(model_path="models/distilbert_v1")
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

    # Per-transform × per-attack-type recall (the deep breakdown)
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
    print(f"    recall    = {overall['recall']:.4f}    ← key robustness metric")
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
            verdict = "✓ robust"
        elif rec >= 0.5:
            verdict = "⚠ partially degraded"
        else:
            verdict = "✗ BROKEN"
        print(f"    {transform_name:<25s}  {n:>4}  {rec:>7.4f}  {f1:>7.4f}  {fnr:>7.4f}  {verdict}")

    print(f"\n  Latency: {report['latency']['avg_ms_per_prompt']:.4f} ms/prompt")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adversarial robustness test.")
    parser.add_argument("--test-set", type=str, default="data/test/test.csv")
    parser.add_argument("--detectors", type=str, nargs="+", default=["rule_based", "bert"],
                        help="Which detectors to test (default: both).")
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
          f"({adv_df['original_text'].nunique()} originals × {adv_df['transform'].nunique()} transforms)")

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
            detector = load_detector(detector_name)
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
    print(f"\nNext step: Run:  python launcher_stage6e.py")


if __name__ == "__main__":
    main()
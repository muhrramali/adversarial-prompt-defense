"""
scripts/evaluate_detector.py
=============================
CLI to evaluate ANY detector against the held-out test set.

WHY THIS EXISTS
---------------
We need one script that can evaluate the rule-based detector, the BERT
detector, and the adversarially-trained detector using IDENTICAL code paths
so the comparison is fair.

USAGE
-----
    # Evaluate the rule-based baseline (Stage 3):
    python scripts/evaluate_detector.py --detector rule_based

    # Evaluate BERT (Stage 4 onwards):
    python scripts/evaluate_detector.py --detector bert --model-path models/distilbert_v1

    # Save results to results/<name>.json:
    python scripts/evaluate_detector.py --detector rule_based --save

OUTPUT
------
Prints to stdout:
  - Overall metrics (accuracy, precision, recall, F1, FPR, FNR, confusion matrix)
  - Per-attack-type breakdown
  - List of false positives and false negatives (for error analysis)
Optionally saves a JSON report to results/<detector>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.detection import RuleBasedDetector, BaseDetector
from src.evaluation import compute_binary_metrics, compute_per_attack_type


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_detector(name: str, model_path: str | None = None) -> BaseDetector:
    """Factory function. Stage 3 = rule_based, Stage 4 = bert."""
    if name == "rule_based":
        return RuleBasedDetector()
    elif name == "bert":
        # Lazy import so evaluate_detector can be used for rule_based
        # even when torch isn't installed.
        from src.detection.bert_detector import BertDetector
        if model_path is None:
            model_path = "models/distilbert_v1"
        return BertDetector(model_path=model_path)
    else:
        raise ValueError(f"Unknown detector: {name!r}. Options: rule_based, bert")


def evaluate(detector: BaseDetector, test_csv: Path) -> dict:
    """Run detector over test set, compute metrics, return report dict."""
    if not test_csv.exists():
        raise FileNotFoundError(f"Test set not found: {test_csv}")

    df = pd.read_csv(test_csv, encoding="utf-8")
    prompts = df["text"].tolist()
    y_true = df["label"].astype(int).tolist()

    # Run detection with timing
    print(f"[evaluate] Running {detector.name} on {len(prompts)} prompts...")
    t0 = time.perf_counter()

    # Use batch detection if available (BERT), else loop
    if hasattr(detector, "detect_batch") and detector.name == "bert":
        # Use overridden detect_batch for BERT — ~10x faster
        results = detector.detect_batch(prompts, batch_size=16)
        y_pred = [1 if r.label == "prompt_injection" else 0 for r in results]
        confidences = [r.confidence for r in results]
        matched_patterns_counts = [len(r.matched_patterns) for r in results]
    else:
        y_pred: list[int] = []
        confidences: list[float] = []
        matched_patterns_counts: list[int] = []
        for p in prompts:
            result = detector.detect(p)
            y_pred.append(1 if result.label == "prompt_injection" else 0)
            confidences.append(float(result.confidence))
            matched_patterns_counts.append(len(result.matched_patterns))
    elapsed = time.perf_counter() - t0

    overall = compute_binary_metrics(y_true, y_pred)
    per_type = compute_per_attack_type(df, y_pred)

    # Identify errors
    false_positives: list[dict] = []
    false_negatives: list[dict] = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        row = df.iloc[i]
        if t == 0 and p == 1:
            false_positives.append({
                "text": row["text"],
                "attack_type": row["attack_type"],
                "confidence": round(confidences[i], 4),
                "matched_patterns_count": matched_patterns_counts[i],
            })
        elif t == 1 and p == 0:
            false_negatives.append({
                "text": row["text"],
                "attack_type": row["attack_type"],
                "true_label": int(t),
                "confidence": round(confidences[i], 4),
            })

    avg_latency_ms = (elapsed / len(prompts)) * 1000 if prompts else 0.0

    report = {
        "detector": detector.name,
        "test_set": str(test_csv),
        "n_samples": len(prompts),
        "overall_metrics": overall.to_dict(),
        "per_attack_type": {k: v.to_dict() for k, v in per_type.items()},
        "errors": {
            "n_false_positives": len(false_positives),
            "n_false_negatives": len(false_negatives),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
        "latency": {
            "total_seconds": round(elapsed, 4),
            "avg_ms_per_prompt": round(avg_latency_ms, 4),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return report


def print_report(report: dict) -> None:
    """Human-readable summary printed to stdout."""
    print("\n" + "=" * 72)
    print(f"  Detector: {report['detector']!r}  |  Test set: {report['n_samples']} prompts")
    print("=" * 72)

    m = report["overall_metrics"]
    cm = m["confusion_matrix"]
    print(f"\n  Overall metrics:")
    print(f"    accuracy  = {m['accuracy']:.4f}")
    print(f"    precision = {m['precision']:.4f}")
    print(f"    recall    = {m['recall']:.4f}    (of real attacks, how many caught)")
    print(f"    f1        = {m['f1']:.4f}")
    print(f"    FPR       = {m['false_positive_rate']:.4f}    (of benign, how many wrongly blocked)")
    print(f"    FNR       = {m['false_negative_rate']:.4f}    (of attacks, how many missed)")
    print(f"\n  Confusion matrix:")
    print(f"                Predicted")
    print(f"                benign  attack")
    print(f"    Actual benign  {cm['tn']:>5}   {cm['fp']:>5}")
    print(f"    Actual attack  {cm['fn']:>5}   {cm['tp']:>5}")

    print(f"\n  Per-attack-type breakdown:")
    print(f"    {'attack_type':<28}  {'n':>4}  {'acc':>6}  {'prec':>6}  {'rec':>6}  {'f1':>6}")
    print(f"    {'-'*28}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for attack_type, metrics in sorted(report["per_attack_type"].items()):
        print(f"    {attack_type:<28}  {metrics['n_samples']:>4}  "
              f"{metrics['accuracy']:>6.4f}  {metrics['precision']:>6.4f}  "
              f"{metrics['recall']:>6.4f}  {metrics['f1']:>6.4f}")

    errs = report["errors"]
    print(f"\n  Errors:")
    print(f"    False positives (benign wrongly blocked): {errs['n_false_positives']}")
    print(f"    False negatives (attacks missed):        {errs['n_false_negatives']}")

    if errs["n_false_positives"] > 0:
        print(f"\n  False positives (first 5):")
        for fp in errs["false_positives"][:5]:
            print(f"    [{fp['attack_type']}] conf={fp['confidence']:.3f} "
                  f"matched={fp['matched_patterns_count']} | {fp['text'][:80]!r}")

    if errs["n_false_negatives"] > 0:
        print(f"\n  False negatives (first 5):")
        for fn in errs["false_negatives"][:5]:
            print(f"    [{fn['attack_type']}] conf={fn['confidence']:.3f} | {fn['text'][:80]!r}")

    print(f"\n  Latency:")
    print(f"    total: {report['latency']['total_seconds']:.4f}s")
    print(f"    avg:   {report['latency']['avg_ms_per_prompt']:.4f} ms/prompt")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a detector on the test set.")
    parser.add_argument("--detector", type=str, default="rule_based",
                        choices=["rule_based", "bert"],
                        help="Which detector to evaluate.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model directory (BERT only).")
    parser.add_argument("--detector-name", type=str, default=None,
                        help="Override the detector name in the saved report (e.g., 'bert_adv' "
                             "for the adversarially-trained model).")
    parser.add_argument("--test-set", type=str, default="data/test/test.csv",
                        help="Path to test CSV.")
    parser.add_argument("--save", action="store_true",
                        help="Save JSON report to results/<detector>_<timestamp>.json")
    args = parser.parse_args()

    detector = load_detector(args.detector, args.model_path)
    # Override detector.name if requested (used to distinguish Model B from Model C)
    if args.detector_name:
        detector.name = args.detector_name

    test_csv = Path(args.test_set)
    if not test_csv.is_absolute():
        test_csv = PROJECT_ROOT / test_csv

    report = evaluate(detector, test_csv)
    print_report(report)

    if args.save:
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"{detector.name}_{ts}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[evaluate] Saved report to: {out_path}")


if __name__ == "__main__":
    main()

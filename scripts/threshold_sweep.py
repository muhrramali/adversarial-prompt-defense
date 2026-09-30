"""
scripts/threshold_sweep.py
===========================
Quick threshold sweep for the BERT detector.

WHY THIS EXISTS
---------------
With our small training dataset (84 examples), the BERT model outputs
softmax probabilities that aren't well-calibrated. The default argmax
decision (threshold=0.5) produced FPR=1.0 (all benign wrongly blocked).

This script sweeps multiple thresholds and shows how precision, recall,
and F1 change — helping us pick a sensible operating point.

This is a preview of Experiment 7 (full threshold analysis), which will
be done properly in Stage 5.

USAGE
-----
    python scripts/threshold_sweep.py
    python scripts/threshold_sweep.py --model-path models/distilbert_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.detection import BertDetector
from src.evaluation import compute_binary_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sweep_thresholds(model_path: Path, test_csv: Path,
                     thresholds: list[float]) -> list[dict]:
    """
    Load the BERT model ONCE, run all prompts ONCE through the model
    (collecting raw confidences), then evaluate each threshold in pure Python.

    This is much faster than re-loading the model for each threshold.
    """
    print(f"[sweep] Loading model: {model_path}")
    print(f"[sweep] Loading test set: {test_csv}")

    df = pd.read_csv(test_csv, encoding="utf-8")
    prompts = df["text"].tolist()
    y_true = df["label"].astype(int).tolist()
    print(f"[sweep] Test set: {len(prompts)} prompts")

    # Load detector with threshold=0.0 (everything flagged as attack) just so we
    # can collect raw confidences. We'll evaluate thresholds manually below.
    detector = BertDetector(model_path=model_path, decision_threshold=0.0)
    print(f"[sweep] Running batch inference (one forward pass)...")
    results = detector.detect_batch(prompts, batch_size=16)
    confidences = [r.confidence for r in results]
    print(f"[sweep] Got {len(confidences)} confidence scores")

    # Save raw confidences alongside ground truth for later analysis
    raw_data = [{"text": p, "y_true": int(t), "confidence": float(c)}
                for p, t, c in zip(prompts, y_true, confidences)]

    # Sweep thresholds
    print(f"\n[sweep] Threshold sweep:")
    print(f"  {'threshold':>10}  {'accuracy':>8}  {'precision':>9}  {'recall':>7}  "
          f"{'f1':>6}  {'FPR':>6}  {'FNR':>6}  {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*3} {'-'*3} {'-'*3} {'-'*3}")

    sweep_results = []
    for thresh in thresholds:
        y_pred = [1 if c >= thresh else 0 for c in confidences]
        m = compute_binary_metrics(y_true, y_pred)
        cm = m.confusion_matrix
        print(f"  {thresh:>10.2f}  {m.accuracy:>8.4f}  {m.precision:>9.4f}  "
              f"{m.recall:>7.4f}  {m.f1:>6.4f}  {m.false_positive_rate:>6.4f}  "
              f"{m.false_negative_rate:>6.4f}  {cm['tp']:>3} {cm['fp']:>3} "
              f"{cm['fn']:>3} {cm['tn']:>3}")
        sweep_results.append({
            "threshold": thresh,
            "metrics": m.to_dict(),
        })

    return sweep_results, raw_data


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT threshold sweep.")
    parser.add_argument("--model-path", type=str, default="models/distilbert_v1")
    parser.add_argument("--test-set", type=str, default="data/test/test.csv")
    parser.add_argument("--save", action="store_true",
                        help="Save sweep results + raw confidences to results/")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    test_csv = Path(args.test_set)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not test_csv.is_absolute():
        test_csv = PROJECT_ROOT / test_csv

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    sweep_results, raw_data = sweep_thresholds(model_path, test_csv, thresholds)

    # Find optimal threshold (highest F1)
    best = max(sweep_results, key=lambda r: r["metrics"]["f1"])
    print(f"\n[sweep] Best F1 threshold: {best['threshold']:.2f}")
    print(f"         F1={best['metrics']['f1']:.4f}, "
          f"recall={best['metrics']['recall']:.4f}, "
          f"FPR={best['metrics']['false_positive_rate']:.4f}")

    if args.save:
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        sweep_path = results_dir / f"bert_threshold_sweep_{ts}.json"
        raw_path = results_dir / f"bert_raw_confidences_{ts}.json"

        sweep_path.write_text(json.dumps({
            "model_path": str(model_path),
            "test_set": str(test_csv),
            "n_prompts": len(raw_data),
            "sweep_results": sweep_results,
            "best_threshold_by_f1": best,
        }, indent=2), encoding="utf-8")
        raw_path.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")
        print(f"\n[sweep] Sweep results saved to: {sweep_path}")
        print(f"[sweep] Raw confidences saved to: {raw_path}")


if __name__ == "__main__":
    main()

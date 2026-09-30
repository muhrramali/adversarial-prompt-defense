#!/usr/bin/env python3
"""
launcher_stage7a.py
===================
Stage 7 Part A — augmentation + 3-way comparison — adversarial training pipeline.

This launcher will create:
     - scripts/augment_training_data.py
     - scripts/compare_all_models.py

Files are written using r'''...''' raw strings, so content is preserved
byte-for-byte without any encoding/decoding.

Usage:
    python launcher_stage7a.py
"""
from pathlib import Path

FILES = {
    '''scripts/augment_training_data.py''': r'''"""
scripts/augment_training_data.py
=================================
Stage 7 — Augment the training set with adversarial variants.

WHY THIS EXISTS
---------------
Stage 6 showed that BERT beats rule_based on most transforms but
STILL FAILS on 3 character-level transforms:
  - obfuscate_leetspeak    (0% recall)
  - unicode_lookalikes    (0% recall)
  - whitespace_compact     (0% recall)

Stage 7's hypothesis: if we add adversarial variants to the TRAINING
set, Model C (BERT + adversarial training) should be more robust
than Model B (BERT trained on original data only).

This script:
1. Reads data/train/train.csv (84 rows)
2. Applies ALL 10 Stage 6 transforms to each row
3. Output: data/train/augmented_train.csv (84 × 11 = 924 rows)

The labels are preserved (transforms don't change benign→attack).

USAGE
-----
    # Default: apply all transforms
    python scripts/augment_training_data.py

    # Only apply specific transforms (e.g., the ones BERT fails on):
    python scripts/augment_training_data.py --transforms obfuscate_leetspeak unicode_lookalikes whitespace_compact

    # Custom seed
    python scripts/augment_training_data.py --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.adversarial import generate_adversarial_set, summarize_adversarial_set


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment training set with adversarial variants.")
    parser.add_argument("--input", type=str, default="data/train/train.csv",
                        help="Path to original training CSV.")
    parser.add_argument("--output", type=str, default="data/train/augmented_train.csv",
                        help="Path to write augmented training CSV.")
    parser.add_argument("--transforms", type=str, nargs="+", default=None,
                        help="Specific transforms to apply. Default: all 10 transforms.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--no-original", action="store_true",
                        help="Exclude original (untransformed) prompts from output.")
    args = parser.parse_args()

    input_csv = Path(args.input)
    if not input_csv.is_absolute():
        input_csv = PROJECT_ROOT / input_csv
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv = Path(args.output)
    if not output_csv.is_absolute():
        output_csv = PROJECT_ROOT / output_csv

    # Read original to show before/after
    original_df = pd.read_csv(input_csv, encoding="utf-8")
    print(f"[augment] Input: {input_csv}")
    print(f"[augment]   {len(original_df)} rows")
    print(f"[augment]   label distribution: {original_df['label'].value_counts().to_dict()}")
    print(f"[augment]   attack types: {original_df['attack_type'].value_counts().to_dict()}")

    # Generate adversarial variants — this gives us original + all transforms
    print(f"\n[augment] Applying transforms (seed={args.seed})...")
    if args.transforms:
        print(f"[augment]   transforms: {args.transforms}")
    else:
        print(f"[augment]   transforms: ALL 10 (default)")

    # The generator expects columns [text, label, attack_type]
    # but train.csv has [row_id, text, label, attack_type, source, template_id]
    # We need to preserve the extra columns by joining back after generation
    augmented_df = generate_adversarial_set(
        input_csv,
        seed=args.seed,
        transforms=args.transforms,
        include_original=not args.no_original,
    )

    # The augmented df has columns:
    # original_text, original_label, original_attack_type, transform, transformed_text, label
    # We need to convert this back to the train.csv schema:
    # text, label, attack_type, source, template_id

    # Join back to get source and template_id from original
    original_meta = original_df[["text", "source", "template_id"]].rename(
        columns={"text": "original_text"}
    )
    augmented_df = augmented_df.merge(original_meta, on="original_text", how="left")

    # Build the final schema
    final_df = pd.DataFrame({
        "text": augmented_df["transformed_text"],
        "label": augmented_df["label"].astype(int),
        "attack_type": augmented_df["original_attack_type"],
        "source": "synthetic_adversarial",
        "template_id": augmented_df["template_id"].fillna("augmented") + "/" + augmented_df["transform"],
        "transform": augmented_df["transform"],
    })

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[augment] Output: {output_csv}")
    print(f"[augment]   {len(final_df)} rows (was {len(original_df)})")
    print(f"[augment]   {len(final_df) / len(original_df):.1f}x expansion")
    print(f"[augment]   label distribution: {final_df['label'].value_counts().to_dict()}")

    # Per-transform breakdown
    print(f"\n[augment] Per-transform breakdown:")
    for transform_name in sorted(final_df["transform"].unique()):
        subset = final_df[final_df["transform"] == transform_name]
        n_benign = int((subset["label"] == 0).sum())
        n_attack = int((subset["label"] == 1).sum())
        print(f"  {transform_name:25s}: n={len(subset):4d}  (benign={n_benign}, attack={n_attack})")

    print(f"\n[augment] Done. Next step:")
    print(f"[augment]   python scripts/train_bert.py \\")
    print(f"[augment]     --train-csv {output_csv.relative_to(PROJECT_ROOT)} \\")
    print(f"[augment]     --output-dir models/distilbert_adv_v1")


if __name__ == "__main__":
    main()
''',
    '''scripts/compare_all_models.py''': r'''"""
scripts/compare_all_models.py
==============================
Stage 7 — Compare all three models side by side.

WHY THIS EXISTS
---------------
After Stage 7, we have three detectors:
  - Model A: rule_based (Stage 3)
  - Model B: bert (Stage 4, trained on 84 original examples)
  - Model C: bert_adv (Stage 7, trained on 924 augmented examples)

This script loads the latest evaluation report for each and produces
a 3-way comparison table — the headline table for the research report.

USAGE
-----
    # After running evaluate_detector.py for all 3 models:
    python scripts/compare_all_models.py

    # Save as Markdown file:
    python scripts/compare_all_models.py --save
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


def build_three_way_comparison(reports, sweep_data=None) -> str:
    """Build a Markdown-formatted 3-way comparison."""
    summary = compare_detectors(reports)
    lines: list[str] = []

    lines.append("# Stage 7 — Three-Way Model Comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"Test set: {summary.n_samples} prompts")
    lines.append("")
    lines.append("## Models Compared")
    lines.append("")
    lines.append("- **Model A** = `rule_based` (Stage 3 baseline)")
    lines.append("- **Model B** = `bert` (Stage 4, trained on 84 original examples)")
    lines.append("- **Model C** = `bert_adv` (Stage 7, trained on 924 augmented examples)")
    lines.append("")

    # Section 1: Overall metrics table
    lines.append("## 1. Overall Metrics on Test Set")
    lines.append("")
    lines.append(summary.to_markdown_table())
    lines.append("")

    # Section 2: Improvements
    if "bert" in reports and "rule_based" in reports:
        delta_b_vs_a = compute_delta(summary, "rule_based", "bert")
        lines.append("## 2. Improvement: Model B vs Model A (rule_based)")
        lines.append("")
        lines.append("| Metric | Model A (rule_based) | Model B (bert) | Delta |")
        lines.append("|---|---|---|---|")
        for metric in ["accuracy", "precision", "recall", "f1",
                       "false_positive_rate", "false_negative_rate"]:
            a = summary.metrics["rule_based"].get(metric, 0.0)
            b = summary.metrics["bert"].get(metric, 0.0)
            d = delta_b_vs_a.get(metric, 0.0)
            sign = "+" if d >= 0 else ""
            lines.append(f"| {metric} | {a:.4f} | {b:.4f} | {sign}{d:.4f} |")
        lines.append("")

    if "bert_adv" in reports and "bert" in reports:
        delta_c_vs_b = compute_delta(summary, "bert", "bert_adv")
        lines.append("## 3. Improvement: Model C vs Model B (adversarial training)")
        lines.append("")
        lines.append("| Metric | Model B (bert) | Model C (bert_adv) | Delta |")
        lines.append("|---|---|---|---|")
        for metric in ["accuracy", "precision", "recall", "f1",
                       "false_positive_rate", "false_negative_rate"]:
            b = summary.metrics["bert"].get(metric, 0.0)
            c = summary.metrics["bert_adv"].get(metric, 0.0)
            d = delta_c_vs_b.get(metric, 0.0)
            sign = "+" if d >= 0 else ""
            lines.append(f"| {metric} | {b:.4f} | {c:.4f} | {sign}{d:.4f} |")
        lines.append("")

    if "bert_adv" in reports and "rule_based" in reports:
        delta_c_vs_a = compute_delta(summary, "rule_based", "bert_adv")
        lines.append("## 4. Improvement: Model C vs Model A (full pipeline)")
        lines.append("")
        lines.append("| Metric | Model A (rule_based) | Model C (bert_adv) | Delta |")
        lines.append("|---|---|---|---|")
        for metric in ["accuracy", "precision", "recall", "f1",
                       "false_positive_rate", "false_negative_rate"]:
            a = summary.metrics["rule_based"].get(metric, 0.0)
            c = summary.metrics["bert_adv"].get(metric, 0.0)
            d = delta_c_vs_a.get(metric, 0.0)
            sign = "+" if d >= 0 else ""
            lines.append(f"| {metric} | {a:.4f} | {c:.4f} | {sign}{d:.4f} |")
        lines.append("")

    # Section 5: Latency
    lines.append("## 5. Latency Comparison")
    lines.append("")
    lines.append("| Detector | Avg latency (ms/prompt) |")
    lines.append("|---|---|")
    for d in summary.detectors:
        ms = summary.latency_ms.get(d, 0.0)
        lines.append(f"| {d} | {ms:.4f} |")
    lines.append("")

    # Section 6: Error counts
    lines.append("## 6. Error Counts")
    lines.append("")
    lines.append("| Detector | False Positives | False Negatives |")
    lines.append("|---|---|---|")
    for d in summary.detectors:
        fp = summary.n_errors.get(d, {}).get("false_positives", 0)
        fn = summary.n_errors.get(d, {}).get("false_negatives", 0)
        lines.append(f"| {d} | {fp} | {fn} |")
    lines.append("")

    # Section 7: Key Findings
    lines.append("## 7. Key Findings")
    lines.append("")
    if "bert_adv" in reports and "bert" in reports:
        b_recall = summary.metrics["bert"].get("recall", 0.0)
        c_recall = summary.metrics["bert_adv"].get("recall", 0.0)
        b_f1 = summary.metrics["bert"].get("f1", 0.0)
        c_f1 = summary.metrics["bert_adv"].get("f1", 0.0)

        if c_recall > b_recall:
            lines.append(f"1. **Adversarial training IMPROVED recall** by {(c_recall - b_recall):.4f} "
                         f"({b_recall:.4f} to {c_recall:.4f}). "
                         f"This supports the Stage 7 hypothesis.")
        elif c_recall < b_recall:
            lines.append(f"1. **Adversarial training HURT recall** by {(b_recall - c_recall):.4f} "
                         f"({b_recall:.4f} to {c_recall:.4f}). "
                         f"Possible overfitting to attack patterns.")
        else:
            lines.append(f"1. **Adversarial training had NO EFFECT on recall** "
                         f"(both at {b_recall:.4f}).")
        lines.append("")

        if c_f1 > b_f1:
            lines.append(f"2. **F1 score IMPROVED** by {(c_f1 - b_f1):.4f} ({b_f1:.4f} to {c_f1:.4f}).")
        elif c_f1 < b_f1:
            lines.append(f"2. **F1 score DECREASED** by {(b_f1 - c_f1):.4f} ({b_f1:.4f} to {c_f1:.4f}).")
        else:
            lines.append(f"2. **F1 score unchanged** ({b_f1:.4f}).")
        lines.append("")

    # Section 8: Honest Limitations
    lines.append("## 8. Honest Limitations")
    lines.append("")
    lines.append("- Test set is small (n=19); per-attack-type metrics have high variance.")
    lines.append("- Adversarial training used the SAME transforms as Stage 6 testing — "
                 "this is methodologically optimistic (training/test contamination risk).")
    lines.append("- For true robustness evaluation, the test set should use DIFFERENT transforms "
                 "than training (e.g., paraphrase model instead of synonym substitution).")
    lines.append("- 924 augmented examples may still be too few for DistilBERT to learn "
                 "character-level robustness against leetspeak/unicode.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="3-way model comparison (A vs B vs C).")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/three_way_comparison.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    reports = load_all_reports(results_dir)
    if not reports:
        print(f"No reports found in {results_dir}")
        return

    available = list(reports.keys())
    print(f"[compare] Found reports for detectors: {available}")

    if "bert_adv" not in reports:
        print(f"\n[compare] WARNING: 'bert_adv' report not found.")
        print(f"[compare]   To generate it, train Model C and run:")
        print(f"[compare]   python scripts/evaluate_detector.py --detector bert \\")
        print(f"[compare]     --model-path models/distilbert_adv_v1 \\")
        print(f"[compare]     --detector-name bert_adv --save")

    sweep_data = load_threshold_sweep(results_dir)
    md = build_three_way_comparison(reports, sweep_data)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")

    print(md)
    print(f"\n[compare] Saved to: {output_path}")


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
    print(f"\nNext step: Run:  python launcher_stage7b.py")


if __name__ == "__main__":
    main()
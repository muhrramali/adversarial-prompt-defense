"""
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

"""
scripts/generate_eval_figures.py
=================================
Stage 5 — Generate all evaluation figures (PNG) for the research report.

WHY THIS EXISTS
---------------
Visualizations are how research findings become communicated. We need:
  - Confusion matrices for each detector (rule_based, bert)
  - Side-by-side overall metrics bar chart
  - Per-attack-type recall bar chart (THE key chart — shows where BERT wins)
  - Threshold-vs-recall/FPR curve (the precision/recall trade-off story)
  - Confidence distribution histogram (shows why threshold tuning matters)

All figures are saved to results/. They will be embedded in the final
research report (Stage 15) and the README.

USAGE
-----
    python scripts/generate_eval_figures.py
    python scripts/generate_eval_figures.py --results-dir results --output-dir results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# Font setup for cross-platform consistency
for font_path in [
    "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]:
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)

plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# Color palette
COLORS = {
    "rule_based": "#2196F3",  # blue
    "bert": "#E53935",         # red
    "benign": "#4CAF50",       # green
    "attack": "#E53935",       # red
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Figure 1: Confusion matrices (rule_based vs BERT side by side)
# ---------------------------------------------------------------------------

def plot_confusion_matrices(reports: dict, output_path: Path) -> None:
    """Two confusion matrices side by side."""
    detectors = sorted([d for d in reports.keys() if d in ("rule_based", "bert")])
    if not detectors:
        return

    fig, axes = plt.subplots(1, len(detectors), figsize=(5 * len(detectors), 4),
                              constrained_layout=True)
    if len(detectors) == 1:
        axes = [axes]

    for ax, detector_name in zip(axes, detectors):
        cm = reports[detector_name].confusion_matrix
        matrix = np.array([
            [cm.get("tn", 0), cm.get("fp", 0)],
            [cm.get("fn", 0), cm.get("tp", 0)],
        ])
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        # Annotate cells
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                color = "white" if val > matrix.max() * 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=16)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["benign", "attack"])
        ax.set_yticklabels(["benign", "attack"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{detector_name}\naccuracy={reports[detector_name].overall_metrics.get('accuracy', 0):.4f}")

    fig.suptitle("Confusion Matrices — Rule-Based vs BERT (Test Set, n=19)",
                 fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Overall metrics comparison (grouped bar chart)
# ---------------------------------------------------------------------------

def plot_overall_metrics(summary, output_path: Path) -> None:
    """Grouped bar chart of accuracy/precision/recall/F1/FPR/FNR across detectors."""
    detectors = summary.detectors
    metrics = ["accuracy", "precision", "recall", "f1",
               "false_positive_rate", "false_negative_rate"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"]

    x = np.arange(len(metrics))
    width = 0.8 / len(detectors)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for i, d in enumerate(detectors):
        vals = [summary.metrics.get(d, {}).get(m, 0.0) for m in metrics]
        offset = (i - (len(detectors) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=d,
                      color=COLORS.get(d, "#888"), edgecolor="black", linewidth=0.5)
        # Annotate each bar
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Overall Metrics — Rule-Based vs BERT (Test Set)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Per-attack-type recall (THE KEY chart)
# ---------------------------------------------------------------------------

def plot_per_attack_type_recall(summary, output_path: Path) -> None:
    """Grouped bar chart: recall per attack type for each detector.

    This is THE chart that shows where BERT wins (paraphrased, obfuscated, indirect).
    """
    detectors = summary.detectors
    attack_types = sorted(summary.per_attack_type_recall.keys())
    # Move 'benign' to the end for visual clarity (it's a special case)
    if "benign" in attack_types:
        attack_types.remove("benign")
        attack_types.append("benign")

    x = np.arange(len(attack_types))
    width = 0.8 / len(detectors)

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for i, d in enumerate(detectors):
        vals = [summary.per_attack_type_recall.get(at, {}).get(d, 0.0) for at in attack_types]
        offset = (i - (len(detectors) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=d,
                      color=COLORS.get(d, "#888"), edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(attack_types, rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall (1.0 = caught all attacks of this type)")
    ax.set_title("Per-Attack-Type Recall — Where BERT Improves Over Rule-Based",
                 fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Annotate the "BERT wins" region
    for i, at in enumerate(attack_types):
        if at in ("paraphrased", "obfuscated", "indirect_injection"):
            ax.annotate("← BERT wins", xy=(i, 0.05), fontsize=9,
                        color=COLORS["bert"], fontweight="bold", ha="center")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {output_path}")


# ---------------------------------------------------------------------------
# Figure 4: Threshold sweep curve
# ---------------------------------------------------------------------------

def plot_threshold_curve(sweep_data: dict, output_path: Path) -> None:
    """Threshold vs recall and FPR — shows the precision/recall trade-off."""
    sweep_results = sweep_data.get("sweep_results", [])
    if not sweep_results:
        return

    thresholds = [r["threshold"] for r in sweep_results]
    recalls = [r["metrics"]["recall"] for r in sweep_results]
    fprs = [r["metrics"]["false_positive_rate"] for r in sweep_results]
    f1s = [r["metrics"]["f1"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.plot(thresholds, recalls, "o-", color="#4CAF50", label="Recall (catch rate)",
            linewidth=2, markersize=8)
    ax.plot(thresholds, fprs, "s-", color="#E53935", label="FPR (false alarm rate)",
            linewidth=2, markersize=8)
    ax.plot(thresholds, f1s, "^-", color="#2196F3", label="F1 score",
            linewidth=2, markersize=8)

    # Highlight the chosen operating point (0.70)
    chosen_idx = thresholds.index(0.70) if 0.70 in thresholds else None
    if chosen_idx is not None:
        ax.axvline(0.70, color="gray", linestyle="--", alpha=0.5)
        ax.annotate("Chosen threshold = 0.70",
                    xy=(0.70, 1.05), xytext=(0.75, 1.05),
                    fontsize=10, color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel("Decision Threshold (confidence above which → attack)")
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Threshold Sweep — Precision/Recall Trade-Off for BERT",
                 fontweight="bold")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {output_path}")


# ---------------------------------------------------------------------------
# Figure 5: Confidence distribution (shows why threshold matters)
# ---------------------------------------------------------------------------

def plot_confidence_distribution(raw_data_path: Path, output_path: Path) -> None:
    """Histogram of BERT confidence scores, colored by true label."""
    if not raw_data_path.exists():
        return
    raw = json.loads(raw_data_path.read_text(encoding="utf-8"))

    benign_confs = [r["confidence"] for r in raw if r["y_true"] == 0]
    attack_confs = [r["confidence"] for r in raw if r["y_true"] == 1]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bins = np.linspace(0, 1, 21)

    ax.hist(benign_confs, bins=bins, color=COLORS["benign"], alpha=0.7,
            label=f"Benign (n={len(benign_confs)})", edgecolor="black")
    ax.hist(attack_confs, bins=bins, color=COLORS["attack"], alpha=0.6,
            label=f"Attack (n={len(attack_confs)})", edgecolor="black")

    # Draw threshold lines
    ax.axvline(0.50, color="gray", linestyle=":", linewidth=2, label="Default threshold (0.50)")
    ax.axvline(0.70, color="red", linestyle="--", linewidth=2, label="Chosen threshold (0.70)")

    ax.set_xlabel("BERT confidence (probability of attack)")
    ax.set_ylabel("Number of prompts")
    ax.set_title("Confidence Distribution by True Label — Why Threshold Tuning Matters",
                 fontweight="bold")
    ax.legend(loc="upper center")
    ax.grid(alpha=0.3)

    # Annotate the "uncertain zone"
    ax.axvspan(0.50, 0.70, alpha=0.1, color="yellow",
               label="Uncertain zone (0.5–0.7)")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all evaluation figures.")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing detector JSON reports.")
    parser.add_argument("--output-dir", type=str, default="results/figures",
                        help="Directory to save PNG figures.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[figures] Loading reports from {results_dir}")
    from src.evaluation import load_all_reports, load_threshold_sweep, compare_detectors

    reports = load_all_reports(results_dir)
    if not reports:
        print(f"[figures] No detector reports found in {results_dir}")
        print("[figures] Run these commands first:")
        print("  python scripts/evaluate_detector.py --detector rule_based --save")
        print("  python scripts/evaluate_detector.py --detector bert --save")
        return

    print(f"[figures] Found {len(reports)} detector reports: {list(reports.keys())}")

    summary = compare_detectors(reports)

    print(f"\n[figures] Generating figures:")

    # Figure 1: Confusion matrices
    plot_confusion_matrices(reports, output_dir / "fig_confusion_matrices.png")

    # Figure 2: Overall metrics
    plot_overall_metrics(summary, output_dir / "fig_overall_metrics.png")

    # Figure 3: Per-attack-type recall (KEY chart)
    plot_per_attack_type_recall(summary, output_dir / "fig_per_attack_type_recall.png")

    # Figure 4: Threshold sweep curve (if available)
    sweep_data = load_threshold_sweep(results_dir)
    if sweep_data:
        plot_threshold_curve(sweep_data, output_dir / "fig_threshold_curve.png")

    # Figure 5: Confidence distribution (if raw data available)
    raw_pattern = "bert_raw_confidences_*.json"
    raw_files = sorted(results_dir.glob(raw_pattern), reverse=True)
    if raw_files:
        plot_confidence_distribution(raw_files[0], output_dir / "fig_confidence_distribution.png")
    else:
        print(f"  [SKIP] No raw confidences file ({raw_pattern}) found.")

    print(f"\n[figures] Done! All figures saved to: {output_dir}")
    print(f"[figures] Files generated:")
    for f in sorted(output_dir.iterdir()):
        if f.suffix == ".png":
            print(f"  - {f.name} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

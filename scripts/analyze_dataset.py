"""
scripts/analyze_dataset.py
===========================
Stage 2 — Dataset analysis script.

This is the script equivalent of notebooks/01_dataset_analysis.ipynb.
Both serve the same purpose:
1. Load the generated dataset (and splits).
2. Compute summary statistics.
3. Generate visualizations saved to results/:
   - class_distribution.png
   - text_length_distribution.png
   - split_distribution.png
   - attack_type_per_split.png
4. Print a textual summary to stdout.

WHY THIS EXISTS
---------------
Before training any model, we must inspect the data. Imbalanced classes,
outlier lengths, or split leakage will silently destroy model performance.
Visualization is the cheapest defense against bad data assumptions.

USAGE
-----
    python scripts/analyze_dataset.py
    python scripts/analyze_dataset.py --input data/raw/synthetic_v1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display required
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

# Register fonts for cross-platform consistency (we don't strictly need
# Chinese support for Stage 2 since all labels are ASCII, but registering
# Noto Sans SC keeps later stages working if we add non-ASCII content).
for font_path in [
    "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]:
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)

plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# Project root for default paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def analyze(input_path: Path, splits_dir: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Load full dataset
    # ---------------------------------------------------------------
    print(f"[analyze] Loading {input_path}...")
    df = pd.read_csv(input_path, encoding="utf-8")
    print(f"[analyze]   rows={len(df)}, attack_types={df['attack_type'].nunique()}")

    # ---------------------------------------------------------------
    # 2. Plot class distribution (attack_type × label)
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    counts = df.groupby(["attack_type", "label"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=False, ax=ax,
                color=["#4CAF50", "#E53935"],  # green=benign(0), red=injection(1)
                edgecolor="black", linewidth=0.5)
    ax.set_title("Class Distribution by Attack Type")
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Number of Examples")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Label", labels=["Benign (0)", "Injection (1)"])
    out = results_dir / "class_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[analyze]   saved {out}")

    # ---------------------------------------------------------------
    # 3. Text length distribution
    # ---------------------------------------------------------------
    df["text_length"] = df["text"].str.len()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for label, color, name in [(0, "#4CAF50", "Benign"), (1, "#E53935", "Injection")]:
        subset = df[df["label"] == label]
        ax.hist(subset["text_length"], bins=20, alpha=0.6, color=color, label=name, edgecolor="black")
    ax.set_title("Text Length Distribution by Label")
    ax.set_xlabel("Text Length (characters)")
    ax.set_ylabel("Frequency")
    ax.legend()
    out = results_dir / "text_length_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[analyze]   saved {out}")

    # ---------------------------------------------------------------
    # 4. Splits (if they exist)
    # ---------------------------------------------------------------
    splits: dict[str, pd.DataFrame] = {}
    for name, fname in [("train", "train.csv"), ("validation", "validation.csv"), ("test", "test.csv")]:
        split_path = splits_dir / name / fname
        if split_path.exists():
            splits[name] = pd.read_csv(split_path, encoding="utf-8")
            print(f"[analyze]   loaded split: {name} ({len(splits[name])} rows)")
        else:
            print(f"[analyze]   split not found: {split_path}")

    if splits:
        # 4a. Split sizes
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        names = list(splits.keys())
        sizes = [len(splits[n]) for n in names]
        bars = ax.bar(names, sizes, color=["#2196F3", "#FF9800", "#9C27B0"], edgecolor="black")
        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2, size + 1, str(size),
                    ha="center", va="bottom", fontsize=11)
        ax.set_title("Split Sizes")
        ax.set_xlabel("Split")
        ax.set_ylabel("Number of Examples")
        out = results_dir / "split_sizes.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[analyze]   saved {out}")

        # 4b. Attack type distribution per split (grouped bar chart)
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        attack_types = sorted(df["attack_type"].unique())
        x = list(range(len(attack_types)))
        width = 0.25
        colors = {"train": "#2196F3", "validation": "#FF9800", "test": "#9C27B0"}
        for i, (split_name, split_df) in enumerate(splits.items()):
            counts_by_type = split_df["attack_type"].value_counts().reindex(attack_types, fill_value=0)
            ax.bar([xi + i * width for xi in x], counts_by_type.values, width,
                   label=split_name, color=colors[split_name], edgecolor="black", linewidth=0.5)
        ax.set_title("Attack Type Distribution per Split")
        ax.set_xlabel("Attack Type")
        ax.set_ylabel("Count")
        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels(attack_types, rotation=30, ha="right")
        ax.legend()
        out = results_dir / "split_attack_type_distribution.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[analyze]   saved {out}")

    # ---------------------------------------------------------------
    # 5. Textual summary
    # ---------------------------------------------------------------
    print("\n[analyze] === Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"Unique templates: {df['template_id'].nunique()}")
    print(f"Exact duplicates: {df['text'].duplicated().sum()}")
    print(f"Missing text: {df['text'].isna().sum()}")
    print(f"Empty text: {(df['text'].fillna('').str.len() == 0).sum()}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"Text length — min: {df['text'].str.len().min()}, "
          f"max: {df['text'].str.len().max()}, "
          f"mean: {df['text'].str.len().mean():.1f}, "
          f"median: {df['text'].str.len().median():.1f}")
    print(f"Attack type distribution:")
    for at, n in df["attack_type"].value_counts().items():
        print(f"  {at}: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 dataset analysis.")
    parser.add_argument("--input", type=str, default="data/raw/synthetic_v1.csv",
                        help="Path to the full dataset CSV.")
    parser.add_argument("--splits-dir", type=str, default="data",
                        help="Directory containing train/, validation/, test/ subdirs.")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Where to save the generated PNG plots.")
    args = parser.parse_args()

    analyze(Path(args.input), Path(args.splits_dir), Path(args.results_dir))


if __name__ == "__main__":
    main()

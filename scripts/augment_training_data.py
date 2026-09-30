"""
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

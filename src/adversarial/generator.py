"""
src/adversarial/generator.py
==============================
Stage 6 — Generate adversarial test sets from the original test set.

WHY THIS EXISTS
---------------
We have a held-out test set of 19 prompts. Each transform (Stage 6)
multiplies this by ~10x. With 10 transforms, we get 19 * 10 = 190
adversarial variants. Each variant is a "mutated" version of an
original test prompt.

For each variant, we know:
  - The original prompt text
  - The original label (benign / attack)
  - The original attack_type
  - The transform applied
  - The transformed text

This lets us ask:
  - "Does BERT still catch attacks after leetspeak obfuscation?"
  - "Does rule_based survive synonym substitution?"
  - "Which transform is most damaging to recall?"

USAGE
-----
    from src.adversarial.generator import generate_adversarial_set
    df = generate_adversarial_set("data/test/test.csv", seed=42)
    # df has columns: original_text, original_label, original_attack_type,
    #                 transform, transformed_text, label
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.adversarial.transforms import TRANSFORMS, apply_transform


def generate_adversarial_set(input_csv: str | Path,
                              seed: int = 42,
                              transforms: list[str] | None = None,
                              include_original: bool = True) -> pd.DataFrame:
    """
    Generate an adversarial test set by applying transforms to every prompt.

    Parameters
    ----------
    input_csv       : path to a CSV with columns [text, label, attack_type, ...]
    seed            : reproducibility seed for transforms
    transforms      : list of transform names to apply. If None, applies all.
    include_original: if True, includes the original (untransformed) prompt
                      in the output with transform="original".

    Returns
    -------
    pd.DataFrame with columns:
      - original_text
      - original_label
      - original_attack_type
      - transform
      - transformed_text
      - label  (= original_label, since transforms preserve the label)
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8")

    if transforms is None:
        transforms = list(TRANSFORMS.keys())

    rows: list[dict] = []
    for _, row in df.iterrows():
        original_text = str(row["text"])
        original_label = int(row["label"])
        original_attack_type = str(row["attack_type"])

        if include_original:
            rows.append({
                "original_text": original_text,
                "original_label": original_label,
                "original_attack_type": original_attack_type,
                "transform": "original",
                "transformed_text": original_text,
                "label": original_label,
            })

        for transform_name in transforms:
            if transform_name not in TRANSFORMS:
                continue
            transformed = apply_transform(original_text, transform_name, seed=seed)
            rows.append({
                "original_text": original_text,
                "original_label": original_label,
                "original_attack_type": original_attack_type,
                "transform": transform_name,
                "transformed_text": transformed,
                "label": original_label,
            })

    return pd.DataFrame(rows)


def save_adversarial_set(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the adversarial set to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def summarize_adversarial_set(df: pd.DataFrame) -> dict:
    """Return summary statistics about an adversarial set."""
    n_originals = df["original_text"].nunique()
    n_transforms = df["transform"].nunique()
    n_total = len(df)
    transforms_list = sorted(df["transform"].unique().tolist())

    # Per-transform label distribution
    per_transform = {}
    for transform in transforms_list:
        subset = df[df["transform"] == transform]
        per_transform[transform] = {
            "n": len(subset),
            "n_benign": int((subset["label"] == 0).sum()),
            "n_attack": int((subset["label"] == 1).sum()),
        }

    return {
        "n_original_prompts": n_originals,
        "n_transforms": n_transforms,
        "n_total_rows": n_total,
        "transforms": transforms_list,
        "per_transform": per_transform,
    }


if __name__ == "__main__":
    import sys
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "data/test/test.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "data/test/adversarial_test.csv"

    print(f"[generate] Reading: {input_csv}")
    df = generate_adversarial_set(input_csv, seed=42)
    out = save_adversarial_set(df, output_csv)
    print(f"[generate] Wrote {len(df)} rows to: {out}")

    summary = summarize_adversarial_set(df)
    print(f"\n[generate] Summary:")
    print(f"  Original prompts: {summary['n_original_prompts']}")
    print(f"  Transforms:      {summary['n_transforms']}")
    print(f"  Total rows:      {summary['n_total_rows']}")
    print(f"\n[generate] Per-transform breakdown:")
    for t, info in summary["per_transform"].items():
        print(f"  {t:25s}: n={info['n']:3d}  (benign={info['n_benign']}, attack={info['n_attack']})")

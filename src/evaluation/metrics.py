"""
src/evaluation/metrics.py
===========================
Stage 5 (preliminary) — Binary classification metrics.

WHY THIS EXISTS
---------------
We need to compute the same metrics for every detector (rule-based,
BERT, BERT+adv-trained) so Stage 14 can compare them side-by-side.
Centralizing here means:
  - No metric is computed differently by different scripts.
  - We can extend with new metrics (ROC-AUC, PR-AUC) in one place.
  - Per-attack-type breakdown is reusable across experiments.

METRICS COMPUTED
----------------
- accuracy          : (TP + TN) / N
- precision         : TP / (TP + FP)  — of all flagged, how many real attacks?
- recall            : TP / (TP + FN)  — of all real attacks, how many flagged?
- f1                : 2 * P * R / (P + R)
- false_positive_rate : FP / (FP + TN)  — of all benign, how many wrongly flagged?
- false_negative_rate : FN / (FN + TP)  — of all attacks, how many missed?
- confusion_matrix   : {tn, fp, fn, tp}
- per_attack_type    : same metrics broken down by attack_type column

IMPORTANT — POSITIVE CLASS
--------------------------
In our problem:
  - POSITIVE class (1) = "prompt injection"
  - NEGATIVE class (0) = "benign"

So:
  - TP = attack correctly flagged as attack
  - FP = benign wrongly flagged as attack (false alarm)
  - FN = attack wrongly flagged as benign (missed attack) — WORST
  - TN = benign correctly flagged as benign

For our threat model, FN is much worse than FP (missing an attack is
more dangerous than blocking a legitimate prompt). So recall matters
more than precision in operational settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class BinaryMetrics:
    """Container for binary classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: dict[str, int]  # {tn, fp, fn, tp}
    n_samples: int
    n_positive: int   # actual attacks
    n_negative: int   # actual benign

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "confusion_matrix": self.confusion_matrix,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }

    def __repr__(self) -> str:
        cm = self.confusion_matrix
        return (
            f"BinaryMetrics(\n"
            f"  accuracy={self.accuracy:.4f}  precision={self.precision:.4f}  "
            f"recall={self.recall:.4f}  f1={self.f1:.4f}\n"
            f"  FPR={self.false_positive_rate:.4f}  FNR={self.false_negative_rate:.4f}\n"
            f"  confusion_matrix: TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}  TP={cm['tp']}\n"
            f"  n_samples={self.n_samples}  (pos={self.n_positive}, neg={self.n_negative})\n"
            f")"
        )


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> BinaryMetrics:
    """
    Compute binary classification metrics.

    Parameters
    ----------
    y_true : ground-truth labels (0=benign, 1=injection)
    y_pred : predicted labels (0=benign, 1=injection)

    Returns
    -------
    BinaryMetrics
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}")
    if not y_true:
        raise ValueError("Empty label lists.")

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    n = len(y_true)
    n_pos = tp + fn
    n_neg = tn + fp

    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        confusion_matrix={"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        n_samples=n,
        n_positive=n_pos,
        n_negative=n_neg,
    )


def compute_per_attack_type(df: pd.DataFrame,
                             y_pred: list[int]) -> dict[str, BinaryMetrics]:
    """
    Compute metrics broken down by attack_type.

    Parameters
    ----------
    df     : DataFrame with columns ["text", "label", "attack_type", ...].
             Must have the same rows in the same order as y_pred.
    y_pred : predicted labels, aligned with df rows.

    Returns
    -------
    Dict mapping attack_type -> BinaryMetrics.
    For "benign", positive class is still 1, so recall means "of all benign
    prompts, how many were correctly classified as benign" — i.e. 1 - FPR.
    We surface this as the "benign" entry; it's the false-positive view.
    """
    if len(df) != len(y_pred):
        raise ValueError(f"Length mismatch: df={len(df)} y_pred={len(y_pred)}")

    results: dict[str, BinaryMetrics] = {}
    for attack_type in sorted(df["attack_type"].unique()):
        mask = df["attack_type"] == attack_type
        subset = df[mask].reset_index(drop=True)
        preds = [y_pred[i] for i, m in enumerate(mask) if m]
        # For benign, label=0; for attack types, label=1.
        # We compute metrics with positive class = 1 within this subset.
        # This means: for the "benign" subset, recall = 1 - FPR (good benign recall).
        y_true_subset = subset["label"].tolist()
        results[attack_type] = compute_binary_metrics(y_true_subset, preds)

    return results


def detector_to_predictions(detector, prompts: list[str]) -> tuple[list[int], list[float]]:
    """
    Run a detector over a list of prompts and return (y_pred, confidences).

    y_pred[i] = 1 if detector said "prompt_injection", else 0.
    confidences[i] = detector's confidence score.

    Helper used by scripts/evaluate_detector.py.
    """
    y_pred: list[int] = []
    confs: list[float] = []
    for p in prompts:
        result = detector.detect(p)
        y_pred.append(1 if result.label == "prompt_injection" else 0)
        confs.append(float(result.confidence))
    return y_pred, confs

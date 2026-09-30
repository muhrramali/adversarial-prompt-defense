"""
scripts/run_policy_demo.py
===========================
Stage 8 — Demonstrate the policy engine on the test set.

WHY THIS EXISTS
---------------
We've trained 3 detectors (Model A, B, C). Each produces a confidence
score per prompt. The policy engine translates each into a 3-way
action: ALLOW / REVIEW / BLOCK.

This script:
1. Loads the test set (19 prompts)
2. For each detector (rule_based, bert, bert_adv):
   - Runs every prompt through the detector
   - Runs every DetectionResult through the policy engine
   - Counts: how many ALLOW / REVIEW / BLOCK decisions?
   - For each category, shows example prompts that landed there
3. Saves a JSON report

This is the operational view: not "how accurate is the detector?" but
"if we deployed this in front of an LLM copilot, how many prompts
would reach the LLM vs. be blocked?"

USAGE
-----
    python scripts/run_policy_demo.py
    python scripts/run_policy_demo.py --detectors rule_based bert bert_adv
    python scripts/run_policy_demo.py --allow-below 0.3 --block-at 0.85
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.detection import RuleBasedDetector
from src.defense import PolicyEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_detector(name: str):
    """Factory function."""
    if name == "rule_based":
        return RuleBasedDetector()
    elif name in ("bert", "bert_adv"):
        from src.detection.bert_detector import BertDetector
        path = "models/distilbert_adv_v1" if name == "bert_adv" else "models/distilbert_v1"
        detector = BertDetector(model_path=path)
        detector.name = name  # so reports distinguish Model B from C
        return detector
    else:
        raise ValueError(f"Unknown detector: {name!r}")


def run_policy_for_detector(detector, policy, prompts_with_labels):
    """Run detector + policy on every prompt. Return list of decision dicts."""
    decisions = []
    for prompt, y_true in prompts_with_labels:
        # Detect
        result = detector.detect(prompt)
        # Apply policy
        decision = policy.evaluate(result, prompt=prompt)
        # Annotate with ground truth for analysis
        decisions.append({
            "prompt": prompt,
            "y_true": y_true,
            "detector_label": result.label,
            "detector_confidence": round(result.confidence, 4),
            "policy_action": decision.action,
            "policy_reason": decision.reason,
            "matched_hard_rule": decision.matched_hard_rule,
            "is_correct": (
                (y_true == 1 and decision.action == "block") or
                (y_true == 0 and decision.action == "allow")
            ),
        })
    return decisions


def summarize_decisions(decisions: list[dict]) -> dict:
    """Compute summary statistics on policy decisions."""
    n = len(decisions)
    action_counts = Counter(d["policy_action"] for d in decisions)

    # Safety analysis
    true_attacks_blocked = sum(1 for d in decisions if d["y_true"] == 1 and d["policy_action"] == "block")
    true_attacks_allowed = sum(1 for d in decisions if d["y_true"] == 1 and d["policy_action"] == "allow")
    true_attacks_reviewed = sum(1 for d in decisions if d["y_true"] == 1 and d["policy_action"] == "review")
    benign_blocked = sum(1 for d in decisions if d["y_true"] == 0 and d["policy_action"] == "block")
    benign_allowed = sum(1 for d in decisions if d["y_true"] == 0 and d["policy_action"] == "allow")

    n_attacks = sum(1 for d in decisions if d["y_true"] == 1)
    n_benign = sum(1 for d in decisions if d["y_true"] == 0)

    return {
        "n_total": n,
        "n_attacks": n_attacks,
        "n_benign": n_benign,
        "action_counts": dict(action_counts),
        # SAFETY METRICS
        "attack_block_rate": round(true_attacks_blocked / n_attacks, 4) if n_attacks else 0.0,
        "attack_leak_rate": round(true_attacks_allowed / n_attacks, 4) if n_attacks else 0.0,
        # USABILITY METRICS
        "benign_block_rate": round(benign_blocked / n_benign, 4) if n_benign else 0.0,
        "benign_allow_rate": round(benign_allowed / n_benign, 4) if n_benign else 0.0,
        # COUNTS
        "true_attacks_blocked": true_attacks_blocked,
        "true_attacks_allowed": true_attacks_allowed,
        "true_attacks_reviewed": true_attacks_reviewed,
        "benign_blocked": benign_blocked,
        "benign_allowed": benign_allowed,
    }


def print_demo_report(detector_name: str, decisions: list[dict], summary: dict) -> None:
    """Pretty-print the policy demo for one detector."""
    print("\n" + "=" * 78)
    print(f"  Policy Demo — Detector: {detector_name!r}  |  {summary['n_total']} prompts")
    print("=" * 78)

    print(f"\n  Action distribution:")
    print(f"    ALLOW  : {summary['action_counts'].get('allow', 0):3d} prompts")
    print(f"    REVIEW : {summary['action_counts'].get('review', 0):3d} prompts")
    print(f"    BLOCK  : {summary['action_counts'].get('block', 0):3d} prompts")

    print(f"\n  Safety metrics (against the {summary['n_attacks']} real attacks):")
    print(f"    attack_block_rate = {summary['attack_block_rate']:.4f}  "
          f"({summary['true_attacks_blocked']}/{summary['n_attacks']} blocked)  key safety metric")
    print(f"    attack_leak_rate  = {summary['attack_leak_rate']:.4f}  "
          f"({summary['true_attacks_allowed']}/{summary['n_attacks']} reached LLM)  WORST outcome")
    if summary["true_attacks_reviewed"] > 0:
        print(f"    attack_reviewed  = {summary['true_attacks_reviewed']}/{summary['n_attacks']} "
              f"(held for human review)")

    print(f"\n  Usability metrics (against the {summary['n_benign']} benign prompts):")
    print(f"    benign_allow_rate = {summary['benign_allow_rate']:.4f}  "
          f"({summary['benign_allowed']}/{summary['n_benign']} reached LLM)  key usability metric")
    print(f"    benign_block_rate = {summary['benign_block_rate']:.4f}  "
          f"({summary['benign_blocked']}/{summary['n_benign']} wrongly blocked)")

    # Show some examples of each action
    print(f"\n  Examples of BLOCK decisions:")
    block_examples = [d for d in decisions if d["policy_action"] == "block"][:3]
    for d in block_examples:
        y_tag = "ATTACK" if d["y_true"] == 1 else "BENIGN"
        correct = "OK" if d["is_correct"] else "X WRONG"
        print(f"    [{y_tag}] {correct} conf={d['detector_confidence']:.3f} | {d['prompt'][:60]!r}")

    if summary["true_attacks_allowed"] > 0:
        print(f"\n  Examples of LEAKED attacks (allowed through to LLM):")
        leaked = [d for d in decisions if d["policy_action"] == "allow" and d["y_true"] == 1]
        for d in leaked[:3]:
            print(f"    [ATTACK] conf={d['detector_confidence']:.3f} | {d['prompt'][:60]!r}")

    if summary["benign_blocked"] > 0:
        print(f"\n  Examples of WRONGLY-BLOCKED benign prompts:")
        blocked_benign = [d for d in decisions if d["policy_action"] == "block" and d["y_true"] == 0]
        for d in blocked_benign[:3]:
            print(f"    [BENIGN] conf={d['detector_confidence']:.3f} | {d['prompt'][:60]!r}")

    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate the policy engine on the test set.")
    parser.add_argument("--test-set", type=str, default="data/test/test.csv")
    parser.add_argument("--detectors", type=str, nargs="+",
                        default=["rule_based", "bert", "bert_adv"],
                        help="Detectors to evaluate (default: all three).")
    parser.add_argument("--allow-below", type=float, default=None,
                        help="Override allow threshold (default: from config).")
    parser.add_argument("--block-at", type=float, default=None,
                        help="Override block threshold (default: from config).")
    parser.add_argument("--save", action="store_true",
                        help="Save JSON report to results/.")
    args = parser.parse_args()

    test_csv = Path(args.test_set)
    if not test_csv.is_absolute():
        test_csv = PROJECT_ROOT / test_csv
    if not test_csv.exists():
        raise FileNotFoundError(f"Test set not found: {test_csv}")

    # Load test set
    df = pd.read_csv(test_csv, encoding="utf-8")
    prompts_with_labels = list(zip(df["text"].tolist(), df["label"].astype(int).tolist()))
    print(f"[policy_demo] Loaded {len(prompts_with_labels)} prompts from {test_csv}")

    # Build policy engine (with optional threshold overrides)
    policy = PolicyEngine(
        allow_below=args.allow_below,
        block_at_or_above=args.block_at,
    )
    print(f"[policy_demo] {policy.describe()}")
    print()

    all_reports = []
    for detector_name in args.detectors:
        print(f"\n[policy_demo] Loading detector: {detector_name}")
        try:
            detector = load_detector(detector_name)
        except Exception as e:
            print(f"[policy_demo] Failed to load {detector_name}: {e}")
            continue

        print(f"[policy_demo] Running {detector_name} + policy on {len(prompts_with_labels)} prompts...")
        decisions = run_policy_for_detector(detector, policy, prompts_with_labels)
        summary = summarize_decisions(decisions)
        print_demo_report(detector_name, decisions, summary)
        all_reports.append({
            "detector": detector_name,
            "summary": summary,
            "decisions": decisions,
        })

    # Save combined report
    if args.save and all_reports:
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"policy_demo_{ts}.json"
        combined = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "test_set": str(test_csv),
            "policy_config": {
                "allow_below": policy.allow_below,
                "review_below": policy.review_below,
                "block_at_or_above": policy.block_at_or_above,
            },
            "detectors": all_reports,
        }
        out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[policy_demo] Saved combined report to: {out_path}")


if __name__ == "__main__":
    main()

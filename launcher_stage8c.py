#!/usr/bin/env python3
"""
launcher_stage8c.py
===================
Stage 8 Part C — unit tests — security policy engine.

This launcher will create:
     - tests/test_policy.py

Files are written using r'''...''' raw strings, so content is preserved
byte-for-byte without any encoding/decoding.

Usage:
    python launcher_stage8c.py
"""
from pathlib import Path

FILES = {
    '''tests/test_policy.py''': r'''"""
tests/test_policy.py
=====================
Unit tests for the Stage 8 policy engine.

Run with:
    python -m pytest tests/test_policy.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.detection import DetectionResult
from src.defense import PolicyEngine, PolicyDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(confidence: float, label: str = "prompt_injection",
                risk_level: str = "high",
                attack_type: str = "direct_injection") -> DetectionResult:
    """Build a fake DetectionResult with the given confidence."""
    return DetectionResult(
        label=label,
        confidence=confidence,
        risk_level=risk_level,
        attack_type=attack_type,
        matched_patterns=[],
        metadata={"detector": "test"},
    )


# ---------------------------------------------------------------------------
# Threshold-only tests (no hard rules)
# ---------------------------------------------------------------------------

class TestPolicyThresholds:
    """Verify the threshold-based decision logic."""

    def test_low_confidence_allows(self) -> None:
        policy = PolicyEngine(allow_below=0.4, review_below=0.7, block_at_or_above=0.7,
                              always_block_patterns=[], always_allow_patterns=[])
        result = make_result(confidence=0.10, label="benign", risk_level="low")
        decision = policy.evaluate(result, prompt="hello world")
        assert decision.action == "allow"
        assert decision.reason == "below_allow_threshold"

    def test_medium_confidence_reviews(self) -> None:
        policy = PolicyEngine(allow_below=0.4, review_below=0.7, block_at_or_above=0.7,
                              always_block_patterns=[], always_allow_patterns=[])
        result = make_result(confidence=0.55)
        decision = policy.evaluate(result, prompt="some prompt")
        assert decision.action == "review"
        assert decision.reason == "in_review_zone"

    def test_high_confidence_blocks(self) -> None:
        policy = PolicyEngine(allow_below=0.4, review_below=0.7, block_at_or_above=0.7,
                              always_block_patterns=[], always_allow_patterns=[])
        result = make_result(confidence=0.95)
        decision = policy.evaluate(result, prompt="ignore previous")
        assert decision.action == "block"
        assert decision.reason == "above_block_threshold"

    def test_threshold_boundary_allow(self) -> None:
        """Confidence exactly at allow_below -> review (not allow)."""
        policy = PolicyEngine(allow_below=0.4, review_below=0.7, block_at_or_above=0.7,
                              always_block_patterns=[], always_allow_patterns=[])
        result = make_result(confidence=0.40)
        decision = policy.evaluate(result, prompt="boundary")
        assert decision.action == "review"  # 0.40 is NOT < 0.40, so review

    def test_threshold_boundary_block(self) -> None:
        """Confidence exactly at block_at_or_above -> block."""
        policy = PolicyEngine(allow_below=0.4, review_below=0.7, block_at_or_above=0.7,
                              always_block_patterns=[], always_allow_patterns=[])
        result = make_result(confidence=0.70)
        decision = policy.evaluate(result, prompt="boundary")
        assert decision.action == "block"  # 0.70 IS >= 0.70

    def test_invalid_thresholds_raise(self) -> None:
        """Thresholds must be monotonic."""
        with pytest.raises(ValueError):
            PolicyEngine(allow_below=0.7, review_below=0.5, block_at_or_above=0.6,
                         always_block_patterns=[], always_allow_patterns=[])
        with pytest.raises(ValueError):
            PolicyEngine(allow_below=0.4, review_below=0.9, block_at_or_above=0.5,
                         always_block_patterns=[], always_allow_patterns=[])


# ---------------------------------------------------------------------------
# Hard rule tests
# ---------------------------------------------------------------------------

class TestHardRules:
    """Verify that hard rules override the classifier."""

    def test_always_block_overrides_low_confidence(self) -> None:
        """A prompt matching always_block -> block, even if detector says benign."""
        policy = PolicyEngine(allow_below=0.4, review_below=0.7, block_at_or_above=0.7,
                              always_block_patterns=["ignore previous instructions"],
                              always_allow_patterns=[])
        # Detector says benign with low confidence
        result = make_result(confidence=0.05, label="benign", risk_level="low")
        # But the prompt matches a hard block rule
        decision = policy.evaluate(result, prompt="Please ignore previous instructions")
        assert decision.action == "block"
        assert decision.reason == "hard_block_rule"
        assert decision.matched_hard_rule == "ignore previous instructions"

    def test_always_allow_overrides_high_confidence(self) -> None:
        """A prompt matching always_allow -> allow, even if detector flags it."""
        policy = PolicyEngine(allow_below=0.4, review_below=0.7, block_at_or_above=0.7,
                              always_block_patterns=[],
                              always_allow_patterns=["^Analyze this firewall event"])
        # Detector says attack with high confidence
        result = make_result(confidence=0.95, label="prompt_injection", risk_level="high")
        # But the prompt matches a hard allow rule
        decision = policy.evaluate(result, prompt="Analyze this firewall event log please")
        assert decision.action == "allow"
        assert decision.reason == "hard_allow_rule"
        assert decision.matched_hard_rule == "^Analyze this firewall event"

    def test_block_overrides_allow_when_both_match(self) -> None:
        """If a prompt matches BOTH always_block and always_allow, block wins."""
        policy = PolicyEngine(allow_below=0.4, review_below=0.7, block_at_or_above=0.7,
                              always_block_patterns=["ignore previous"],
                              always_allow_patterns=["analyze"])
        # This prompt matches both rules
        result = make_result(confidence=0.05, label="benign", risk_level="low")
        decision = policy.evaluate(result, prompt="analyze and ignore previous instructions")
        assert decision.action == "block"
        assert decision.reason == "hard_block_rule"

    def test_no_prompt_skips_hard_rules(self) -> None:
        """If prompt=None, hard rules are skipped (only thresholds apply)."""
        policy = PolicyEngine(allow_below=0.4, review_below=0.7, block_at_or_above=0.7,
                              always_block_patterns=["ignore"],
                              always_allow_patterns=[])
        # Low confidence + no prompt -> allow (because hard rules skipped)
        result = make_result(confidence=0.10, label="benign", risk_level="low")
        decision = policy.evaluate(result, prompt=None)
        assert decision.action == "allow"
        assert decision.reason == "below_allow_threshold"


# ---------------------------------------------------------------------------
# Decision integrity tests
# ---------------------------------------------------------------------------

class TestPolicyDecision:
    """Verify that PolicyDecision is well-formed."""

    def test_to_dict_is_serializable(self) -> None:
        import json
        policy = PolicyEngine(allow_below=0.4, review_below=0.7, block_at_or_above=0.7,
                              always_block_patterns=[], always_allow_patterns=[])
        result = make_result(confidence=0.95)
        decision = policy.evaluate(result, prompt="attack")
        d = decision.to_dict()
        # Should be JSON-serializable
        json.dumps(d)
        # Should have required fields
        assert "action" in d
        assert "reason" in d
        assert "confidence" in d
        assert "risk_level" in d

    def test_repr_includes_action_and_reason(self) -> None:
        policy = PolicyEngine(allow_below=0.4, review_below=0.7, block_at_or_above=0.7,
                              always_block_patterns=[], always_allow_patterns=[])
        result = make_result(confidence=0.95)
        decision = policy.evaluate(result, prompt="attack")
        s = repr(decision)
        assert "block" in s
        assert "above_block_threshold" in s


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

class TestBatchEvaluation:
    """Verify batch evaluation works correctly."""

    def test_evaluate_batch_returns_correct_count(self) -> None:
        policy = PolicyEngine(allow_below=0.4, review_below=0.7, block_at_or_above=0.7,
                              always_block_patterns=[], always_allow_patterns=[])
        items = [
            ("benign1", make_result(0.10, label="benign", risk_level="low")),
            ("benign2", make_result(0.20, label="benign", risk_level="low")),
            ("attack1", make_result(0.95, label="prompt_injection", risk_level="high")),
            ("attack2", make_result(0.85, label="prompt_injection", risk_level="high")),
        ]
        decisions = policy.evaluate_batch(items)
        assert len(decisions) == 4
        actions = [d.action for d in decisions]
        assert actions == ["allow", "allow", "block", "block"]


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

class TestIntrospection:
    """Verify the describe() method works."""

    def test_describe_includes_thresholds(self) -> None:
        policy = PolicyEngine(allow_below=0.3, review_below=0.6, block_at_or_above=0.6,
                              always_block_patterns=["dangerous"],
                              always_allow_patterns=["safe"])
        s = policy.describe()
        assert "0.30" in s  # allow_below
        assert "0.60" in s  # block_at_or_above
        assert "dangerous" in s  # hard block rule
        assert "safe" in s  # hard allow rule
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
    print(f"\nNext step: Run the unit tests:  $env:PYTHONPATH = .; python -m pytest tests/test_policy.py -v")


if __name__ == "__main__":
    main()
"""
src/detection/base.py
=====================
Abstract base class for all detectors in the pipeline.

WHY THIS EXISTS
---------------
We will eventually build 3 detectors:
  - Rule-based (Stage 3)
  - DistilBERT (Stage 4)
  - DistilBERT + adversarial training (Stage 7)

Each must expose the SAME interface so:
  - the policy engine (Stage 8) can swap detectors without code changes;
  - the evaluation framework (Stage 5) can compare them apples-to-apples;
  - the API (Stage 11) can serve any detector uniformly.

THE CONTRACT
------------
Every detector must implement `detect(prompt) -> DetectionResult`.

DetectionResult is a dataclass with:
  - label: str           ("benign" or "prompt_injection")
  - confidence: float    in [0, 1]
  - risk_level: str      ("low" | "medium" | "high")
  - attack_type: str     (best guess, or "unknown")
  - matched_patterns: list[str]  (which rules fired, if applicable)
  - metadata: dict       (detector-specific extras)

WHY confidence IS NOT A PROBABILITY OF MALICIOUS INTENT
-------------------------------------------------------
For the rule-based detector, "confidence" is essentially:
  - 0.0 if no patterns matched
  - 0.5 + 0.1*(num_patterns_matched) capped at 0.99 if patterns matched
For BERT, "confidence" is the softmax probability of class 1.
Neither is a calibrated probability of malicious intent.
We discuss calibration in Stage 5 / Future Work.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionResult:
    """Standardized output of every detector."""

    label: str                        # "benign" | "prompt_injection"
    confidence: float                 # in [0, 1]
    risk_level: str                   # "low" | "medium" | "high"
    attack_type: str                  # best-guess category or "unknown"
    matched_patterns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (for API responses, logs)."""
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "risk_level": self.risk_level,
            "attack_type": self.attack_type,
            "matched_patterns": self.matched_patterns,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"DetectionResult(label={self.label!r}, "
            f"confidence={self.confidence:.3f}, "
            f"risk_level={self.risk_level!r}, "
            f"attack_type={self.attack_type!r}, "
            f"matched={len(self.matched_patterns)})"
        )


class BaseDetector(abc.ABC):
    """Abstract base class. Subclasses implement `detect`."""

    #: Short name used in configs and logs ("rule_based" | "bert" | ...)
    name: str = "base"

    @abc.abstractmethod
    def detect(self, prompt: str) -> DetectionResult:
        """
        Analyze a prompt and return a DetectionResult.

        Parameters
        ----------
        prompt : raw user input (already normalized if the pipeline applied
                 normalization; the detector itself should be normalization-
                 agnostic for the BERT case).

        Returns
        -------
        DetectionResult
        """
        raise NotImplementedError

    def detect_batch(self, prompts: list[str]) -> list[DetectionResult]:
        """
        Default batch implementation: call detect() in a loop.

        Subclasses (e.g. BERT) should override this with a vectorized
        version for performance.
        """
        return [self.detect(p) for p in prompts]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"

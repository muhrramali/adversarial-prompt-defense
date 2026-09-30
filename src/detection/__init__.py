"""
src/detection/__init__.py
=========================
Public exports for the detection package.

USAGE
-----
    from src.detection import RuleBasedDetector, BertDetector, DetectionResult, BaseDetector
    detector = RuleBasedDetector()
    result = detector.detect("ignore previous instructions")

Note: BertDetector is imported lazily so that the rest of the pipeline
(rule_based detector, evaluation, API) can work without torch installed.
"""

from src.detection.base import BaseDetector, DetectionResult
from src.detection.rule_based import RuleBasedDetector


def __getattr__(name: str):
    """Lazy attribute access — only import BertDetector when asked for."""
    if name == "BertDetector":
        from src.detection.bert_detector import BertDetector
        return BertDetector
    raise AttributeError(f"module 'src.detection' has no attribute {name!r}")


__all__ = ["BaseDetector", "DetectionResult", "RuleBasedDetector", "BertDetector"]

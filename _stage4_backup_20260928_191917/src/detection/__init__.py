"""
src/detection/__init__.py
=========================
Public exports for the detection package.

USAGE
-----
    from src.detection import RuleBasedDetector, DetectionResult, BaseDetector
    detector = RuleBasedDetector()
    result = detector.detect("ignore previous instructions")
"""

from src.detection.base import BaseDetector, DetectionResult
from src.detection.rule_based import RuleBasedDetector

__all__ = ["BaseDetector", "DetectionResult", "RuleBasedDetector"]

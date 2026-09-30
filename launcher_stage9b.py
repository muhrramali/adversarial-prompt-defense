#!/usr/bin/env python3
"""
launcher_stage9b.py
===================
Stage 9 Part B — output validation + pipeline — LLM security copilot integration.

This launcher will create:
     - src/llm/output_validation.py
     - src/llm/pipeline.py

Files are written using r'''...''' raw strings, so content is preserved
byte-for-byte without any encoding/decoding.

Usage:
    python launcher_stage9b.py
"""
from pathlib import Path

FILES = {
    '''src/llm/output_validation.py''': r'''"""
src/llm/output_validation.py
=============================
Stage 9 — Validate LLM output before returning to user.

WHY THIS EXISTS
---------------
Even if a prompt passes detection and policy, the LLM's output
might still be unsafe. Examples:
  - LLM accidentally leaks a secret (e.g., "API_KEY=sk-...")
  - LLM follows a hidden instruction and outputs "rm -rf /"
  - LLM produces content that violates application policy

This module checks the output BEFORE it's returned to the user.
If validation fails, the response is replaced with a safe message.

USAGE
-----
    from src.llm.output_validation import validate_output
    from src.llm.mock_copilot import CopilotResponse

    response = copilot.respond(prompt)
    validated = validate_output(response, config=...)
    if validated.blocked_by_output_validation:
        return safe_message()
    else:
        return validated
"""

from __future__ import annotations

import re
from typing import Any

from src.llm.mock_copilot import CopilotResponse
from src.utils.config import load_config


# Patterns that indicate unsafe output
DEFAULT_BLOCK_IF_CONTAINS = [
    r"API_KEY\s*=\s*sk-",           # leaked OpenAI key
    r"AKIA[A-Z0-9]{16}",            # leaked AWS key
    r"BEGIN.*PRIVATE KEY",          # leaked private key
    r"rm\s+-rf\s+/",                # dangerous shell command
    r"DROP\s+TABLE",                # SQL injection (if copilot echoes user SQL)
]

DEFAULT_BLOCK_IF_STARTS_WITH = [
    # A naive canary for compliance-style refusals that weren't supposed to happen
    # (e.g., if the LLM was tricked into "Sure, here is how to...")
    r"Sure, here is how to",
]


def validate_output(response: CopilotResponse,
                     block_if_contains: list[str] | None = None,
                     block_if_starts_with: list[str] | None = None,
                     config: dict[str, Any] | None = None) -> CopilotResponse:
    """
    Validate an LLM response. If it contains unsafe content, mark it blocked.

    Parameters
    ----------
    response            : the CopilotResponse to validate
    block_if_contains   : list of regex strings; if any matches, block
    block_if_starts_with: list of regex strings; if response starts with any, block
    config              : pre-loaded config dict

    Returns
    -------
    CopilotResponse (possibly modified with blocked_by_output_validation=True)
    """
    cfg = config or load_config()
    validation_cfg = cfg.get("llm", {}).get("output_validation", {})

    if block_if_contains is None:
        block_if_contains = validation_cfg.get("block_if_contains", DEFAULT_BLOCK_IF_CONTAINS)
    if block_if_starts_with is None:
        block_if_starts_with = validation_cfg.get("block_if_starts_with", DEFAULT_BLOCK_IF_STARTS_WITH)

    text = response.text
    reasons: list[str] = []

    # Check "contains" patterns
    for pat in block_if_contains:
        if re.search(pat, text, re.IGNORECASE):
            reasons.append(f"contains:{pat}")

    # Check "starts_with" patterns
    for pat in block_if_starts_with:
        if re.match(pat, text, re.IGNORECASE):
            reasons.append(f"starts_with:{pat}")

    if reasons:
        # Block the response — replace with safe message
        return CopilotResponse(
            text="[BLOCKED BY OUTPUT VALIDATION] The response was deemed unsafe "
                 "and has been withheld. Please rephrase your request.",
            model=response.model,
            tokens_used=response.tokens_used,
            metadata={**response.metadata, "validation_block_reasons": reasons},
            blocked_by_output_validation=True,
            block_reason=", ".join(reasons),
        )

    # All clear
    return response
''',
    '''src/llm/pipeline.py''': r'''"""
src/llm/pipeline.py
===================
Stage 9 — Full end-to-end pipeline.

WHY THIS EXISTS
---------------
Stages 3-8 built individual components. Stage 9 wires them together:

  user_prompt
      |
      v
  normalize (Stage 13 placeholder — basic Unicode/whitespace cleanup)
      |
      v
  detect (Stage 3/4/7 — rule_based / bert / bert_adv)
      |
      v
  policy.evaluate (Stage 8 — allow / review / block)
      | +----- allow -----+   +-- block/review --+
      |                    |   |                   |
      v                    |   v                   |
  mock_copilot.respond     |   return blocked/review
      |                    |
      v                    |
  output_validation (Stage 16)
      |                    |
      v                    |
  final_response            |
                            v
                      return to user

This module exposes one class: SecurityPipeline.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src.detection import BaseDetector, DetectionResult
from src.defense import PolicyEngine, PolicyDecision
from src.llm.mock_copilot import MockCopilot, CopilotResponse
from src.llm.output_validation import validate_output
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """End-to-end result of processing a single user prompt."""
    original_prompt: str
    normalized_prompt: str
    detection_result: DetectionResult
    policy_decision: PolicyDecision
    copilot_response: CopilotResponse | None = None
    final_action: str = ""  # "allowed", "blocked", "review", "blocked_by_output_validation"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "normalized_prompt": self.normalized_prompt,
            "detection": self.detection_result.to_dict(),
            "policy": self.policy_decision.to_dict(),
            "copilot_response": self.copilot_response.to_dict() if self.copilot_response else None,
            "final_action": self.final_action,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Normalization (Stage 13 placeholder — minimal version)
# ---------------------------------------------------------------------------

def normalize_prompt(prompt: str, config: dict | None = None) -> str:
    """
    Basic input normalization.

    Full Stage 13 will add:
      - control character stripping
      - length limits
      - more sophisticated Unicode handling

    For now we do:
      - NFC Unicode normalization
      - collapse runs of whitespace (spaces AND newlines) to single instances
    """
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")

    # NFC Unicode normalization (combines decomposed characters)
    text = unicodedata.normalize("NFC", prompt)

    # Collapse runs of spaces within each line (preserve newlines)
    text = "\n".join(
        " ".join(line.split())
        for line in text.split("\n")
    )
    # Collapse runs of newlines (3+ -> 1, 2 -> 1)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SecurityPipeline:
    """Full pipeline: normalize -> detect -> policy -> LLM -> validate."""

    def __init__(self,
                 detector: BaseDetector,
                 policy: PolicyEngine | None = None,
                 copilot: MockCopilot | None = None,
                 config: dict[str, Any] | None = None) -> None:
        self.detector = detector
        self.policy = policy or PolicyEngine(config=config)
        self.copilot = copilot or MockCopilot()
        self.config = config or load_config()

    def process(self, user_prompt: str) -> PipelineResult:
        """
        Process a single user prompt through the full pipeline.

        Returns PipelineResult with all intermediate steps.
        """
        # 1. Normalize
        normalized = normalize_prompt(user_prompt, config=self.config)

        # 2. Detect
        detection = self.detector.detect(normalized)

        # 3. Policy decision
        decision = self.policy.evaluate(detection, prompt=normalized)

        # 4. Route based on policy
        if decision.action == "allow":
            # Send to copilot
            response = self.copilot.respond(normalized)

            # 5. Validate output
            validated = validate_output(response, config=self.config)

            if validated.blocked_by_output_validation:
                # Output was unsafe — return blocked message
                return PipelineResult(
                    original_prompt=user_prompt,
                    normalized_prompt=normalized,
                    detection_result=detection,
                    policy_decision=decision,
                    copilot_response=validated,
                    final_action="blocked_by_output_validation",
                    metadata={"output_validation_block": validated.block_reason},
                )

            return PipelineResult(
                original_prompt=user_prompt,
                normalized_prompt=normalized,
                detection_result=detection,
                policy_decision=decision,
                copilot_response=validated,
                final_action="allowed",
            )

        elif decision.action == "block":
            # Don't send to copilot
            return PipelineResult(
                original_prompt=user_prompt,
                normalized_prompt=normalized,
                detection_result=detection,
                policy_decision=decision,
                copilot_response=None,
                final_action="blocked",
                metadata={"block_reason": decision.reason},
            )

        else:  # review
            # Don't send to copilot — hold for human review
            return PipelineResult(
                original_prompt=user_prompt,
                normalized_prompt=normalized,
                detection_result=detection,
                policy_decision=decision,
                copilot_response=None,
                final_action="review",
                metadata={"review_reason": decision.reason},
            )
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
    print(f"\nNext step: Run:  python launcher_stage9c.py")


if __name__ == "__main__":
    main()
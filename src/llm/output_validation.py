"""
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

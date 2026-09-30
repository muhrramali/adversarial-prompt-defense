"""
src/utils/logger.py
===================
Structured JSON-line security logger.

WHY THIS EXISTS
---------------
Stage 21 calls for structured security logs (timestamp, request_id,
classification, confidence, action). Doing this ad-hoc across modules
leads to inconsistent formats and missed fields. This module centralizes
the logger.

DESIGN
------
- Standard library `logging` underneath — no extra dependency.
- A custom `JsonFormatter` writes one JSON object per line.
- PII redaction: basic email/API-key pattern scrubbing (Stage 21 extends this).
- Logs go to BOTH stdout (for `uvicorn` etc.) and a rotating file.

USAGE
-----
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("request_classified", extra={
        "request_id": "abc-123",
        "classification": "prompt_injection",
        "confidence": 0.94,
        "action": "blocked",
    })

The structured fields go into the JSON `event` object automatically.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


# Naive redaction patterns. Stage 21 will expand this.
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Email
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[REDACTED_EMAIL]"),
    # Looks-like-an-API-key: 20+ char alphanumeric
    (re.compile(r"\b(sk-[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|[A-Za-z0-9]{32,})\b"), "[REDACTED_KEY]"),
]


def _redact(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class JsonFormatter(logging.Formatter):
    """
    Emit each log record as a single line of JSON.

    Example output:
    {"ts":"2025-09-26T12:00:00Z","level":"INFO","logger":"api.main",
     "event":"request_classified","request_id":"abc-123",
     "classification":"prompt_injection","confidence":0.94,"action":"blocked"}
    """

    def format(self, record: logging.LogRecord) -> str:
        # Always-present fields
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        # Merge structured extras (anything the caller put in `extra={...}`)
        # Standard LogRecord attributes we should NOT copy through:
        reserved = set(vars(record).keys()) - {"message"}
        # Actually we want any extra attribute the caller added.
        # `record` has __dict__ that includes both stdlib attrs and extras.
        stdlib_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key, value in vars(record).items():
            if key not in stdlib_attrs:
                # Redact strings, leave numbers/bools alone
                if isinstance(value, str):
                    value = _redact(value)
                payload[key] = value

        # If exception info is attached, include a `exc` field
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # Redact the event message itself if it contains obvious PII
        payload["event"] = _redact(payload["event"])

        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str = "app",
               log_dir: str | Path | None = None,
               level: str = "INFO",
               structured: bool = True) -> logging.Logger:
    """
    Get a configured logger.

    Parameters
    ----------
    name       : logger name (usually __name__).
    log_dir    : directory for the rotating log file. If None, file logging is skipped.
    level      : log level string, e.g. "INFO", "DEBUG".
    structured : if True, emit JSON lines; if False, emit plain text (dev mode).
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # avoid duplicate lines from root logger

    if structured:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Optional file handler with rotation (5 MB per file, keep 3 backups)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_dir / f"{name.replace('.', '_')}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


if __name__ == "__main__":
    # Quick smoke test
    log = get_logger("smoke_test", log_dir="logs", level="INFO", structured=True)
    log.info("started")
    log.warning("suspicious_pattern_detected",
                extra={"request_id": "demo-001", "confidence": 0.88, "action": "blocked"})
    log.error("forced_error", exc_info=ValueError("demo exception"))
    print("--- smoke test passed ---")

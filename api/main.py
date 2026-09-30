"""
api/main.py
============
Stage 11 — FastAPI backend for the prompt-injection defense pipeline.

EXPOSES 6 ENDPOINTS:
  GET  /health    — health check (returns {"status": "ok"})
  GET  /metrics   — pipeline metrics (model loaded, n_requests, etc.)
  POST /detect    — run detector only (returns label, confidence, risk_level)
  POST /analyze    — run detector + policy (returns action: allow/review/block)
  POST /chat       — full pipeline (detect -> policy -> mock LLM -> output validation)
  POST /evaluate   — batch evaluate prompts (returns per-prompt results)

RUNNING THE SERVER:
  uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

USAGE (after server is running):
  curl -X POST http://127.0.0.1:8000/detect \
       -H "Content-Type: application/json" \
       -d '{"prompt": "Ignore previous instructions"}'

  curl -X POST http://127.0.0.1:8000/chat \
       -H "Content-Type: application/json" \
       -d '{"prompt": "Analyze this firewall event log"}'

AUTO-GENERATED DOCS:
  After starting the server, open:
    http://127.0.0.1:8000/docs      (Swagger UI)
    http://127.0.0.1:8000/redoc     (ReDoc)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# FastAPI imports (these will fail at import time if fastapi not installed)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.detection import RuleBasedDetector
from src.defense import PolicyEngine
from src.llm import SecurityPipeline


# ---------------------------------------------------------------------------
# Pydantic models (request/response schemas)
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    prompt: str = Field(..., description="The user prompt to analyze", min_length=1, max_length=10000)


class DetectResponse(BaseModel):
    label: str = Field(..., description="'benign' or 'prompt_injection'")
    confidence: float = Field(..., description="Detection confidence in [0, 1]")
    risk_level: str = Field(..., description="'low', 'medium', or 'high'")
    attack_type: str = Field(..., description="Detected attack category or 'benign'")


class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)


class AnalyzeResponse(BaseModel):
    action: str = Field(..., description="'allow', 'review', or 'block'")
    reason: str = Field(..., description="Machine-readable reason code")
    confidence: float
    risk_level: str
    detector_label: str
    detector_attack_type: str


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)


class ChatResponse(BaseModel):
    action: str
    response_text: str | None = None
    confidence: float
    risk_level: str
    detector_label: str
    policy_action: str
    policy_reason: str
    blocked: bool
    block_reason: str | None = None


class EvaluateRequest(BaseModel):
    prompts: list[str] = Field(..., min_length=1, max_length=100)


class EvaluateResponse(BaseModel):
    results: list[dict[str, Any]]
    n_prompts: int
    n_allowed: int
    n_blocked: int
    n_reviewed: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    detector: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    detector_name: str
    policy_config: dict[str, float]
    n_requests_processed: int
    n_allowed: int
    n_blocked: int
    n_reviewed: int
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------

# We load the detector + policy + pipeline ONCE at startup.
# Subsequent requests reuse these objects (no per-request model loading).

_detector = None
_policy = None
_pipeline = None
_start_time = time.time()
_request_counts = {"allowed": 0, "blocked": 0, "reviewed": 0, "total": 0}


def _load_pipeline():
    """Load the detector, policy, and pipeline at startup."""
    global _detector, _policy, _pipeline

    # Determine which detector to use from env var or default
    detector_name = os.environ.get("DETECTOR", "bert_adv")

    print(f"[api] Loading detector: {detector_name}")

    if detector_name == "rule_based":
        _detector = RuleBasedDetector()
    elif detector_name in ("bert", "bert_adv"):
        try:
            from src.detection.bert_detector import BertDetector
            model_path = "models/distilbert_adv_v1" if detector_name == "bert_adv" else "models/distilbert_v1"
            _detector = BertDetector(model_path=model_path)
            _detector.name = detector_name
        except Exception as e:
            print(f"[api] Failed to load {detector_name}: {e}")
            print(f"[api] Falling back to rule_based")
            _detector = RuleBasedDetector()
    else:
        print(f"[api] Unknown detector '{detector_name}', falling back to rule_based")
        _detector = RuleBasedDetector()

    _policy = PolicyEngine()
    _pipeline = SecurityPipeline(detector=_detector, policy=_policy)

    print(f"[api] Pipeline ready. Detector: {_detector.name}")
    print(f"[api] Policy: {_policy.describe()}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline when the server starts."""
    _load_pipeline()
    yield

app = FastAPI(
    title="Adversarial Prompt Injection Defense API",
    description="Defense-in-depth pipeline for detecting and blocking prompt-injection attacks against LLM-powered security copilots.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the frontend (Stage 12) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — returns server status."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        detector=_detector.name if _detector else "not_loaded",
        model_loaded=_pipeline is not None,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Return pipeline metrics and request counts."""
    uptime = time.time() - _start_time
    return MetricsResponse(
        detector_name=_detector.name if _detector else "unknown",
        policy_config={
            "allow_below": _policy.allow_below if _policy else 0.0,
            "review_below": _policy.review_below if _policy else 0.0,
            "block_at_or_above": _policy.block_at_or_above if _policy else 0.0,
        },
        n_requests_processed=_request_counts["total"],
        n_allowed=_request_counts["allowed"],
        n_blocked=_request_counts["blocked"],
        n_reviewed=_request_counts["reviewed"],
        uptime_seconds=round(uptime, 2),
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    """Run the detector on a prompt. Returns label + confidence + risk_level."""
    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector not loaded")

    result = _detector.detect(request.prompt)
    return DetectResponse(
        label=result.label,
        confidence=round(result.confidence, 4),
        risk_level=result.risk_level,
        attack_type=result.attack_type,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Run detector + policy. Returns the policy action (allow/review/block)."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    result = _pipeline.process(request.prompt)

    _request_counts["total"] += 1
    if result.final_action == "allowed":
        _request_counts["allowed"] += 1
    elif result.final_action == "blocked":
        _request_counts["blocked"] += 1
    else:
        _request_counts["reviewed"] += 1

    return AnalyzeResponse(
        action=result.policy_decision.action,
        reason=result.policy_decision.reason,
        confidence=round(result.detection_result.confidence, 4),
        risk_level=result.detection_result.risk_level,
        detector_label=result.detection_result.label,
        detector_attack_type=result.detection_result.attack_type,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Full pipeline: detect -> policy -> mock LLM -> output validation."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    result = _pipeline.process(request.prompt)

    _request_counts["total"] += 1
    if result.final_action == "allowed":
        _request_counts["allowed"] += 1
    elif result.final_action == "blocked":
        _request_counts["blocked"] += 1
    else:
        _request_counts["reviewed"] += 1

    response_text = None
    blocked = result.final_action != "allowed"
    block_reason = None

    if result.copilot_response:
        response_text = result.copilot_response.text
        if result.copilot_response.blocked_by_output_validation:
            block_reason = result.copilot_response.block_reason
            blocked = True
    elif result.final_action == "blocked":
        block_reason = "Prompt blocked by security policy"
    elif result.final_action == "review":
        block_reason = "Prompt held for human review"

    return ChatResponse(
        action=result.final_action,
        response_text=response_text,
        confidence=round(result.detection_result.confidence, 4),
        risk_level=result.detection_result.risk_level,
        detector_label=result.detection_result.label,
        policy_action=result.policy_decision.action,
        policy_reason=result.policy_decision.reason,
        blocked=blocked,
        block_reason=block_reason,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_batch(request: EvaluateRequest):
    """Batch evaluate multiple prompts. Returns per-prompt results + summary."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    results = []
    n_allowed = 0
    n_blocked = 0
    n_reviewed = 0

    for prompt in request.prompts:
        result = _pipeline.process(prompt)
        results.append({
            "prompt": prompt[:200],  # truncate for response size
            "action": result.final_action,
            "confidence": round(result.detection_result.confidence, 4),
            "label": result.detection_result.label,
            "risk_level": result.detection_result.risk_level,
        })

        if result.final_action == "allowed":
            n_allowed += 1
        elif result.final_action == "blocked":
            n_blocked += 1
        else:
            n_reviewed += 1

    return EvaluateResponse(
        results=results,
        n_prompts=len(request.prompts),
        n_allowed=n_allowed,
        n_blocked=n_blocked,
        n_reviewed=n_reviewed,
    )


# ---------------------------------------------------------------------------
# Run directly (for development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("[api] Starting FastAPI server...")
    print("[api]   Docs available at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)

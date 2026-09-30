#!/usr/bin/env python3
"""
launcher_stage12b.py
====================
Stage 12 Part B — JavaScript — frontend dashboard.

This launcher will create:
     - frontend/app.js

Files are written using r'''...''' raw strings, so content is preserved
byte-for-byte without any encoding/decoding.

Usage:
    python launcher_stage12b.py
"""
from pathlib import Path

FILES = {
    '''frontend/app.js''': r'''/* ==========================================================================
   Stage 12 — Dashboard JavaScript
   Calls the FastAPI backend at http://127.0.0.1:8000
   ========================================================================== */

const API_BASE = "http://127.0.0.1:8000";

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

function showTab(tabName) {
    document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".nav-btn").forEach(el => el.classList.remove("active"));
    document.getElementById(tabName).classList.add("active");
    event.target.classList.add("active");
}

// ---------------------------------------------------------------------------
// Helper: fill prompt input
// ---------------------------------------------------------------------------

function fillPrompt(text) {
    document.getElementById("prompt-input").value = text;
}

function clearAll() {
    document.getElementById("prompt-input").value = "";
    document.getElementById("detection-result").classList.add("hidden");
    document.getElementById("policy-result").classList.add("hidden");
    document.getElementById("chat-result").classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Helper: show/hide loading
// ---------------------------------------------------------------------------

function showLoading() {
    document.getElementById("loading").classList.remove("hidden");
}

function hideLoading() {
    document.getElementById("loading").classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Helper: set value with color class
// ---------------------------------------------------------------------------

function setValue(elementId, value, cssClass) {
    const el = document.getElementById(elementId);
    el.textContent = value;
    el.className = "value";
    if (cssClass) {
        el.classList.add(cssClass);
    }
}

// ---------------------------------------------------------------------------
// API call: /detect
// ---------------------------------------------------------------------------

async function callDetect() {
    const prompt = document.getElementById("prompt-input").value.trim();
    if (!prompt) { alert("Please enter a prompt."); return; }

    showLoading();
    document.getElementById("detection-result").classList.add("hidden");

    try {
        const response = await fetch(`${API_BASE}/detect`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: prompt }),
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        setValue("det-label", data.label, data.label);
        setValue("det-confidence", data.confidence.toFixed(4));
        setValue("det-risk", data.risk_level, data.risk_level);
        setValue("det-attack-type", data.attack_type);

        document.getElementById("detection-result").classList.remove("hidden");
    } catch (err) {
        alert("API error: " + err.message + "\n\nIs the API running? Start it with:\nuvicorn api.main:app --reload");
    } finally {
        hideLoading();
    }
}

// ---------------------------------------------------------------------------
// API call: /analyze
// ---------------------------------------------------------------------------

async function callAnalyze() {
    const prompt = document.getElementById("prompt-input").value.trim();
    if (!prompt) { alert("Please enter a prompt."); return; }

    showLoading();
    document.getElementById("policy-result").classList.add("hidden");

    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: prompt }),
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        setValue("pol-action", data.action, data.action);
        setValue("pol-reason", data.reason);

        document.getElementById("policy-result").classList.remove("hidden");
    } catch (err) {
        alert("API error: " + err.message);
    } finally {
        hideLoading();
    }
}

// ---------------------------------------------------------------------------
// API call: /chat (full pipeline)
// ---------------------------------------------------------------------------

async function callChat() {
    const prompt = document.getElementById("prompt-input").value.trim();
    if (!prompt) { alert("Please enter a prompt."); return; }

    showLoading();
    document.getElementById("chat-result").classList.add("hidden");

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: prompt }),
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        setValue("chat-action", data.action, data.action);

        const banner = document.getElementById("chat-blocked-banner");
        const responseText = document.getElementById("chat-response-text");
        const blockReasonRow = document.getElementById("chat-block-reason-row");

        if (data.blocked) {
            banner.classList.remove("hidden");
            responseText.textContent = data.response_text || "(no response — prompt was blocked)";
            responseText.style.opacity = "0.5";
            if (data.block_reason) {
                blockReasonRow.style.display = "flex";
                setValue("chat-block-reason", data.block_reason);
            } else {
                blockReasonRow.style.display = "none";
            }
        } else {
            banner.classList.add("hidden");
            responseText.textContent = data.response_text || "(empty response)";
            responseText.style.opacity = "1";
            blockReasonRow.style.display = "none";
        }

        document.getElementById("chat-result").classList.remove("hidden");
    } catch (err) {
        alert("API error: " + err.message + "\n\nIs the API running?");
    } finally {
        hideLoading();
    }
}

// ---------------------------------------------------------------------------
// Health check on page load
// ---------------------------------------------------------------------------

async function checkApiHealth() {
    const badge = document.getElementById("api-status");
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            const data = await response.json();
            badge.textContent = "API: " + data.detector;
            badge.classList.add("ok");
            badge.classList.remove("error");
        } else {
            badge.textContent = "API: Error";
            badge.classList.add("error");
        }
    } catch (err) {
        badge.textContent = "API: Offline";
        badge.classList.add("error");
    }
}

// Run health check on load
window.addEventListener("DOMContentLoaded", checkApiHealth);
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
    print(f"\nNext step: Start the dashboard: 1) uvicorn api.main:app --reload --port 8000  2) python -m http.server 3000 --directory frontend  3) Open http://localhost:3000")


if __name__ == "__main__":
    main()
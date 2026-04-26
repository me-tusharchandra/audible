# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Audible Env Environment.

This module creates an HTTP server that exposes the AudibleEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
from pathlib import Path

# OpenEnv's web interface looks for the README at /app/README.md or at
# $ENV_README_PATH. Our Dockerfile puts the env at /app/env/, so neither
# default applies — point ENV_README_PATH at the right file before
# create_app reads it. Falls back to the local repo path for `uv run` dev.
for _candidate in (
    Path("/app/env/README.md"),
    Path(__file__).resolve().parent.parent / "README.md",
):
    if _candidate.exists():
        os.environ.setdefault("ENV_README_PATH", str(_candidate))
        break

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import GateAction, GateObservation
    from .audible_env_environment import AudibleEnvironment
except ImportError:
    from models import GateAction, GateObservation
    from server.audible_env_environment import AudibleEnvironment


# Create the app with web interface and README integration
app = create_app(
    AudibleEnvironment,
    GateAction,
    GateObservation,
    env_name="audible_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


# ---------------------------------------------------------------------------
# CORS: the live frontend (Vercel preview/production) must be able to call
# /classify cross-origin. openenv.create_app does not install CORS middleware
# by default, so we add a permissive policy here.
# ---------------------------------------------------------------------------
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /classify: live ambient-gating inference against the trained mobileBERT
# classifier. Lazy-loaded so the Space starts fast.
# ---------------------------------------------------------------------------
import os
from typing import Optional, Dict
from fastapi import HTTPException
from pydantic import BaseModel

try:
    from ..models import PROFILE_DESCRIPTIONS
except ImportError:
    from models import PROFILE_DESCRIPTIONS

# Lazy-load the model so the Space starts fast even before the first /classify call.
_classifier_state: Dict[str, object] = {}


def _load_classifier():
    if "model" in _classifier_state:
        return _classifier_state
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_dir = os.environ.get("AUDIBLE_MODEL_DIR", "/app/env/model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    _classifier_state.update({"tokenizer": tokenizer, "model": model, "torch": torch})
    return _classifier_state


_LABELS = [
    "IGNORE",
    "UPDATE_CONTEXT",
    "ACT_set_timer",
    "ACT_add_calendar_event",
    "ACT_play_music",
    "ACT_web_search",
    "ACT_smart_home_control",
]


class ClassifyRequest(BaseModel):
    utterance: str
    profile: str = "proactive"


class ClassifyResponse(BaseModel):
    decision: str
    tool: Optional[str] = None
    confidence: float
    all_scores: Dict[str, float]


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest) -> ClassifyResponse:
    if req.profile not in PROFILE_DESCRIPTIONS:
        raise HTTPException(400, f"profile must be one of {list(PROFILE_DESCRIPTIONS)}")
    if not req.utterance.strip():
        raise HTTPException(400, "utterance must be non-empty")
    state = _load_classifier()
    tokenizer, model, torch = state["tokenizer"], state["model"], state["torch"]
    inputs = tokenizer(
        PROFILE_DESCRIPTIONS[req.profile],
        req.utterance,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = int(probs.argmax().item())
    label = _LABELS[pred_idx]
    if label.startswith("ACT_"):
        decision, tool = "ACT", label[len("ACT_"):]
    else:
        decision, tool = label, None
    return ClassifyResponse(
        decision=decision,
        tool=tool,
        confidence=float(probs[pred_idx]),
        all_scores={lbl: float(p) for lbl, p in zip(_LABELS, probs.tolist())},
    )


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m audible_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn audible_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)

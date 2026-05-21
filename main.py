from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_NAME = "SVIET OpenRouter Backend"
APP_VERSION = "2.0.0"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Set these on Render / server environment
DEFAULT_API_KEY = os.getenv("OPENROUTER_DEFAULT_API_KEY", "").strip()
DEFAULT_MODEL_ID = os.getenv("OPENROUTER_DEFAULT_MODEL_ID", "google/gemini-2.5-flash:free").strip()

# Keep only chat-capable free models here for fallback
FREE_FALLBACK_MODELS = [
    "openai/gpt-oss-20b:free",
    "baidu/cobuddy:free",
    "openrouter/owl-alpha",
    "inclusionai/ling-2.6-flash",    
]

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()
LAST_REQUEST_AT: Optional[datetime] = None
LAST_SUCCESS_AT: Optional[datetime] = None
LAST_ERROR: Optional[str] = None
LAST_MODEL_USED: Optional[str] = None


class ChatRequest(BaseModel):
    # Optional so frontend can omit one or both
    api_key: Optional[str] = Field(default=None, min_length=0)
    model_id: Optional[str] = Field(default=None, min_length=0)
    prompt: str = Field(..., min_length=1)
    fallback_model_ids: Optional[List[str]] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def _dedupe_models(models: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for m in models:
        m = _clean(m)
        if not m or m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def _resolve_api_key(request_key: Optional[str]) -> str:
    key = _clean(request_key)
    if key:
        return key
    if DEFAULT_API_KEY:
        return DEFAULT_API_KEY
    raise HTTPException(
        status_code=400,
        detail={
            "message": "No API key provided and no backend default API key is configured.",
            "hint": "Set OPENROUTER_DEFAULT_API_KEY on the server or send api_key from the frontend.",
            "backend_status": "live",
        },
    )


def _resolve_model_id(request_model: Optional[str]) -> str:
    model = _clean(request_model)
    if model:
        return model
    if DEFAULT_MODEL_ID:
        return DEFAULT_MODEL_ID
    raise HTTPException(
        status_code=400,
        detail={
            "message": "No model ID provided and no backend default model is configured.",
            "hint": "Set OPENROUTER_DEFAULT_MODEL_ID on the server or send model_id from the frontend.",
            "backend_status": "live",
        },
    )


@app.get("/")
async def root():
    return {
        "ok": True,
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "live",
        "message": "Backend is live",
        "backend_connected": True,
        "server_time_utc": _now_iso(),
    }


@app.get("/health")
async def health():
    return {
        "ok": True,
        "backend_connected": True,
        "status": "live",
    }


@app.get("/status")
async def status():
    return {
        "ok": True,
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "live",
        "backend_connected": True,
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "last_request_at": LAST_REQUEST_AT.isoformat() if LAST_REQUEST_AT else None,
        "last_success_at": LAST_SUCCESS_AT.isoformat() if LAST_SUCCESS_AT else None,
        "last_error": LAST_ERROR,
        "last_model_used": LAST_MODEL_USED,
        "default_api_configured": bool(DEFAULT_API_KEY),
        "default_model_id": DEFAULT_MODEL_ID,
        "server_time_utc": _now_iso(),
    }


async def call_openrouter(api_key: str, model_id: str, prompt: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-OpenRouter-Title": APP_NAME,
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(OPENROUTER_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()

    print("========== OPENROUTER ERROR ==========")
    print("STATUS:", response.status_code)
    print("MODEL:", model_id)
    print("PROMPT_LENGTH:", len(prompt))
    print("BODY:", response.text)
    print("======================================")

    raise HTTPException(
        status_code=response.status_code,
        detail={
            "message": "OpenRouter request failed",
            "status_code": response.status_code,
            "body": response.text,
            "backend_status": "live",
        },
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    global LAST_REQUEST_AT, LAST_SUCCESS_AT, LAST_ERROR, LAST_MODEL_USED

    LAST_REQUEST_AT = datetime.now(timezone.utc)

    resolved_api_key = _resolve_api_key(req.api_key)
    resolved_model_id = _resolve_model_id(req.model_id)

    models_to_try = _dedupe_models(
        [resolved_model_id]
        + (req.fallback_model_ids or [])
        + FREE_FALLBACK_MODELS
    )

    status_trace = [
        "backend connected",
        "input received",
        "credentials resolved",
        "sending data to ai",
        "ai thinking",
    ]

    last_error: Optional[str] = None

    for model_id in models_to_try:
        try:
            data = await call_openrouter(resolved_api_key, model_id, req.prompt)

            reply = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            if not reply:
                raise HTTPException(status_code=502, detail="Empty reply from OpenRouter.")

            LAST_SUCCESS_AT = datetime.now(timezone.utc)
            LAST_ERROR = None
            LAST_MODEL_USED = model_id

            status_trace.append("response received")

            return {
                "ok": True,
                "reply": reply,
                "model_used": model_id,
                "fallback_used": model_id != resolved_model_id,
                "backend_status": "live",
                "status": "response received",
                "status_trace": status_trace,
                "resolved_from_defaults": {
                    "api_key": not bool(_clean(req.api_key)),
                    "model_id": not bool(_clean(req.model_id)),
                },
            }

        except HTTPException as e:
            last_error = str(e.detail)
            LAST_ERROR = last_error

            if e.status_code in [400, 401, 402, 408, 409, 425, 429, 500, 502, 503, 504]:
                continue
            raise

        except Exception as e:
            last_error = str(e)
            LAST_ERROR = last_error
            continue

    raise HTTPException(
        status_code=502,
        detail={
            "message": "All models failed",
            "last_error": last_error,
            "models_tried": models_to_try,
            "backend_status": "live",
            "status_trace": status_trace + ["failed"],
        },
    )

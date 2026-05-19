# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
import time
from datetime import datetime, timezone

APP_NAME = "SVIET OpenRouter Backend"
APP_VERSION = "1.0.0"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Keep only chat-capable free models here
FREE_FALLBACK_MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen3-235b-a22b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain later
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
    api_key: str = Field(..., min_length=10)
    model_id: str = Field(..., min_length=3)
    prompt: str = Field(..., min_length=1)
    fallback_model_ids: Optional[List[str]] = []


def _dedupe_models(models: List[str]) -> List[str]:
    seen = set()
    out = []
    for m in models:
        m = (m or "").strip()
        if not m or m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/")
async def root():
    return {
        "ok": True,
        "service": APP_NAME,
        "version": APP_VERSION,
        "message": "Backend is live",
    }


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/status")
async def status():
    return {
        "ok": True,
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "live",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "last_request_at": LAST_REQUEST_AT.isoformat() if LAST_REQUEST_AT else None,
        "last_success_at": LAST_SUCCESS_AT.isoformat() if LAST_SUCCESS_AT else None,
        "last_error": LAST_ERROR,
        "last_model_used": LAST_MODEL_USED,
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

    # Log useful error info for Render logs
    print("========== OPENROUTER ERROR ==========")
    print("STATUS:", response.status_code)
    print("MODEL:", model_id)
    print("PROMPT_LENGTH:", len(prompt))
    print("BODY:", response.text)
    print("======================================")

    raise HTTPException(
        status_code=response.status_code,
        detail=response.text,
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    global LAST_REQUEST_AT, LAST_SUCCESS_AT, LAST_ERROR, LAST_MODEL_USED

    LAST_REQUEST_AT = datetime.now(timezone.utc)

    models_to_try = _dedupe_models(
        [req.model_id] + (req.fallback_model_ids or []) + FREE_FALLBACK_MODELS
    )

    last_error = None

    for model_id in models_to_try:
        try:
            data = await call_openrouter(req.api_key, model_id, req.prompt)

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

            return {
                "ok": True,
                "reply": reply,
                "model_used": model_id,
                "fallback_used": model_id != req.model_id,
                "backend_status": "live",
            }

        except HTTPException as e:
            last_error = e.detail
            LAST_ERROR = str(last_error)

            # Retry on model/provider/rate-limit/bad-request errors
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
        },
    )

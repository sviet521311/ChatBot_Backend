from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from typing import List, Optional, Literal, Dict, Any

app = FastAPI(title="OpenRouter Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_FALLBACK_MODELS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen3-235b-a22b:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-4-maverick:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    api_key: str = Field(..., min_length=10)
    model_id: str = Field(..., min_length=3)
    prompt: str = Field(..., min_length=1)
    fallback_model_ids: Optional[List[str]] = []


@app.get("/")
async def root():
    return {"message": "Backend is live"}


@app.get("/health")
async def health():
    return {"ok": True}


async def call_openrouter(api_key: str, model_id: str, prompt: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-OpenRouter-Title": "ChatBot Backend",
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(OPENROUTER_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()

    raise HTTPException(
        status_code=response.status_code,
        detail={
            "message": "OpenRouter request failed",
            "model_id": model_id,
            "body": response.text,
        },
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    models_to_try = [req.model_id] + FREE_FALLBACK_MODELS + (req.fallback_model_ids or [])
    models_to_try = list(dict.fromkeys(models_to_try))  # remove duplicates

    last_error = None

    for model_id in models_to_try:
        try:
            data = await call_openrouter(req.api_key, model_id, req.prompt)
            reply = data["choices"][0]["message"]["content"]
            return {
                "ok": True,
                "reply": reply,
                "model_used": model_id,
                "fallback_used": model_id != req.model_id,
            }

        except HTTPException as e:
            last_error = e.detail

            # retry on quota, rate limit, or provider errors
            if e.status_code in [402, 408, 409, 425, 429, 500, 502, 503, 504]:
                continue

            # auth / bad request errors should stop immediately
            raise

        except Exception as e:
            last_error = str(e)
            continue

    raise HTTPException(
        status_code=502,
        detail={
            "message": "All models failed",
            "last_error": last_error,
            "models_tried": models_to_try,
        },
    )

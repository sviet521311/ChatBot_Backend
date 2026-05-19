from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FREE_FALLBACK_MODELS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen3-235b-a22b:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-4-maverick:free",
]

class ChatRequest(BaseModel):
    api_key: str
    model_id: str
    prompt: str

@app.post("/chat")
async def chat(req: ChatRequest):
    models_to_try = [req.model_id] + FREE_FALLBACK_MODELS
    models_to_try = list(dict.fromkeys(models_to_try))

    headers = {
        "Authorization": f"Bearer {req.api_key}",
        "Content-Type": "application/json"
    }

    for model in models_to_try:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": req.prompt
                            }
                        ]
                    }
                )

            if response.status_code == 200:
                data = response.json()
                return {
                    "reply": data["choices"][0]["message"]["content"],
                    "model_used": model
                }

            if response.status_code in [402, 429, 500, 502, 503]:
                continue

        except:
            continue

    raise HTTPException(500, "All models failed")

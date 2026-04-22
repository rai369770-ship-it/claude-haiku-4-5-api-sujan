import json
import os
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="OverChat AI API",
    version="1.0.0",
    description="OverChat AI wrapper service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL = os.environ.get("base_url", "https://api.overchat.ai")
OVERCHAT_URL = f"{BASE_URL}/v1/chat/completions"

OVERCHAT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "*/*",
    "X-Device-Platform": "web",
    "X-Device-Language": "id-ID",
    "X-Device-Uuid": "0084ff72-2faf-4338-ac78-f0e59fad3108",
    "X-Device-Version": "1.0.44",
    "Origin": "https://overchat.ai",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt")
    systemInstructions: Optional[str] = Field(None, description="System instructions")
    model: Optional[str] = Field(None, description="Model name to use")


class ChatResponse(BaseModel):
    success: bool
    response: str


class RootResponse(BaseModel):
    success: bool
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_payload(
    prompt: str,
    system_instructions: Optional[str],
    stream: bool,
    model: Optional[str] = None,
) -> dict:
    return {
        "chatId": str(uuid.uuid4()),
        "model": model or "claude-haiku-4-5-20251001",
        "messages": [
            {
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": prompt,
            },
            {
                "id": str(uuid.uuid4()),
                "role": "system",
                "content": system_instructions or "",
            },
        ],
        "personaId": model or "claude-haiku-4-5-landing",
        "frequency_penalty": 0,
        "max_tokens": 4000,
        "presence_penalty": 0,
        "stream": stream,
        "temperature": 0.5,
        "top_p": 0.95,
    }


def extract_ai_text(raw_text: str) -> str:
    """
    Parse a raw SSE response string and return the concatenated AI content.
    Handles both literal newlines and escaped '\\n' sequences.
    """
    # Normalise escaped newlines that some proxies introduce
    raw_text = raw_text.replace("\\\\n", "\n").replace("\\n", "\n")

    full_text = ""

    for line in raw_text.splitlines():
        line = line.strip()

        if not line.startswith("data: "):
            continue

        data = line[6:].strip()

        if data == "[DONE]":
            break

        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue

        delta = payload.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")

        if content:
            full_text += content

    return full_text


def parse_sse_line(line: str) -> str:
    """Return the delta content from a single SSE line, or '' if none."""
    if not line.startswith("data: "):
        return ""
    data_str = line[6:].strip()
    if data_str == "[DONE]":
        return ""
    try:
        data = json.loads(data_str)
        choices = data.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            return delta.get("content") or ""
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        pass
    return ""


# ---------------------------------------------------------------------------
# Core fetch functions
# ---------------------------------------------------------------------------

async def fetch_full(
    prompt: str,
    system_instructions: Optional[str],
    model: Optional[str] = None,
) -> str:
    """
    Send a non-streaming request and return the fully assembled response text.
    The upstream may still return SSE format even for non-streaming calls,
    so we use extract_ai_text to handle both cases.
    """
    payload = build_payload(prompt, system_instructions, True, model)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(OVERCHAT_URL, json=payload, headers=OVERCHAT_HEADERS)

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        raw = resp.text

    # ---- Try to unwrap {"detail": "<sse-blob>"} envelope first ----
    try:
        outer = json.loads(raw)

        if isinstance(outer, dict) and "detail" in outer:
            result = extract_ai_text(outer["detail"])
            if result:
                return result

        # Direct non-streaming JSON with choices[].message.content
        if isinstance(outer, dict) and "choices" in outer:
            choices = outer.get("choices") or []
            if choices:
                # Non-streaming shape
                message = choices[0].get("message") or {}
                content = message.get("content")
                if content:
                    return content

                # Single SSE chunk shape
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    return content

    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # ---- Fallback: treat the whole body as SSE text ----
    result = extract_ai_text(raw)
    if result:
        return result

    # Last resort: line-by-line parse
    result = ""
    for line in raw.splitlines():
        result += parse_sse_line(line.strip())
    return result


async def fetch_stream(
    prompt: str,
    system_instructions: Optional[str],
    model: Optional[str] = None,
):
    """
    Async generator that yields text chunks from the upstream SSE stream.
    """
    payload = build_payload(prompt, system_instructions, True, model)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        async with client.stream(
            "POST", OVERCHAT_URL, json=payload, headers=OVERCHAT_HEADERS
        ) as resp:
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code, detail="Upstream error"
                )

            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        content = parse_sse_line(line)
                        if content:
                            yield content


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_model=RootResponse)
async def root():
    return RootResponse(
        success=True,
        message=(
            "Use GET or POST /chat for a complete response, "
            "or GET or POST /stream for a streaming response."
        ),
    )


# ── /chat (no streaming) ────────────────────────────────────────────────────

@app.get("/chat", response_model=ChatResponse)
async def chat_get(
    prompt: str = Query(..., description="User prompt"),
    systemInstructions: Optional[str] = Query(None, description="System instructions"),
    model: Optional[str] = Query(None, description="Model name to use"),
):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        result = await fetch_full(prompt.strip(), systemInstructions, model)
        return ChatResponse(success=True, response=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_post(request: ChatRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        result = await fetch_full(
            request.prompt.strip(), request.systemInstructions, request.model
        )
        return ChatResponse(success=True, response=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /stream (always streams) ────────────────────────────────────────────────

@app.get("/stream")
async def stream_get(
    prompt: str = Query(..., description="User prompt"),
    systemInstructions: Optional[str] = Query(None, description="System instructions"),
    model: Optional[str] = Query(None, description="Model name to use"),
):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    return StreamingResponse(
        fetch_stream(prompt.strip(), systemInstructions, model),
        media_type="text/plain; charset=utf-8",
    )


@app.post("/stream")
async def stream_post(request: ChatRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    return StreamingResponse(
        fetch_stream(
            request.prompt.strip(), request.systemInstructions, request.model
        ),
        media_type="text/plain; charset=utf-8",
    )
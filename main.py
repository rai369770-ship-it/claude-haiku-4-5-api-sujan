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

OVERCHAT_URL = "https://api.overchat.ai/v1/chat/completions"

OVERCHAT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "*/*",
    "X-Device-Platform": "web",
    "X-Device-Language": "id-ID",
    "X-Device-Uuid": "0084ff72-2faf-4338-ac78-f0e59fad3108",
    "X-Device-Version": "1.0.44",
    "Origin": "https://overchat.ai",
}


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt")
    systemInstructions: Optional[str] = Field(None, description="System instructions")
    stream: bool = Field(False, description="Stream response")
    model: Optional[str] = Field(None, description="Model name to use")


class ChatResponse(BaseModel):
    success: bool
    response: str


class RootResponse(BaseModel):
    success: bool
    message: str


def build_payload(prompt: str, system_instructions: Optional[str], stream: bool, model: Optional[str] = None) -> dict:
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


def parse_sse_line(line: str) -> str:
    # Handle both direct SSE lines and escaped JSON strings
    if not line.startswith("data: "):
        return ""
    data_str = line[6:].strip()
    if data_str == "[DONE]":
        return ""
    try:
        import json
        data = json.loads(data_str)
        choices = data.get("choices")
        if choices and len(choices) > 0:
            delta = choices[0].get("delta")
            if delta:
                content = delta.get("content")
                if content:
                    return content
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        pass
    return ""


def parse_detail_content(detail_str: str) -> str:
    """Parse SSE content from a detail string that may contain escaped newlines."""
    import json
    
    result = ""
    # Split by double newlines to get individual data blocks
    blocks = detail_str.split("\n\n")
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Remove 'data: ' prefix if present
        if block.startswith("data: "):
            block = block[6:].strip()
        if block == "[DONE]":
            continue
        try:
            data = json.loads(block)
            choices = data.get("choices")
            if choices and len(choices) > 0:
                delta = choices[0].get("delta")
                if delta:
                    content = delta.get("content")
                    if content:
                        result += content
        except (json.JSONDecodeError, KeyError, TypeError, IndexError):
            pass
    
    return result


async def fetch_full(prompt: str, system_instructions: Optional[str], model: Optional[str] = None) -> str:
    payload = build_payload(prompt, system_instructions, False, model)
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(OVERCHAT_URL, json=payload, headers=OVERCHAT_HEADERS)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        
        content = resp.text
        
        # Try to parse the response - it might be wrapped in {"detail": "..."} or direct JSON/SSE
        try:
            import json
            data = json.loads(content)
            
            # Check if response has a "detail" field containing the actual SSE data
            if "detail" in data:
                detail_str = data["detail"]
                # Parse the detail string which contains escaped newlines
                result = parse_detail_content(detail_str)
                if result:
                    return result
            
            # Try parsing as direct non-streaming response (with choices[].message)
            if isinstance(data, dict) and "choices" in data:
                choices = data.get("choices", [])
                if choices and len(choices) > 0:
                    message = choices[0].get("message", {})
                    if message:
                        result = message.get("content", "")
                        if result:
                            return result
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        
        # Fallback: Parse SSE format directly from content
        result = ""
        for line in content.split("\n"):
            line = line.strip()
            if line:
                result += parse_sse_line(line)
        return result


async def fetch_stream(prompt: str, system_instructions: Optional[str], model: Optional[str] = None):
    payload = build_payload(prompt, system_instructions, True, model)
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        async with client.stream(
            "POST", OVERCHAT_URL, json=payload, headers=OVERCHAT_HEADERS
        ) as resp:
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail="Upstream error")
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


@app.get("/", response_model=RootResponse)
async def root():
    return RootResponse(
        success=True,
        message="Use GET or POST /chat to interact with Claude haiku 4.6 AI.",
    )


@app.get("/chat")
async def chat_get(
    prompt: str = Query(..., description="User prompt"),
    systemInstructions: Optional[str] = Query(None, description="System instructions"),
    stream: bool = Query(False, description="Stream response"),
    model: Optional[str] = Query(None, description="Model name to use"),
):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if stream:
        return StreamingResponse(
            fetch_stream(prompt.strip(), systemInstructions, model),
            media_type="text/plain; charset=utf-8",
        )
    try:
        result = await fetch_full(prompt.strip(), systemInstructions, model)
        return ChatResponse(success=True, response=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_post(request: ChatRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if request.stream:
        return StreamingResponse(
            fetch_stream(request.prompt.strip(), request.systemInstructions, request.model),
            media_type="text/plain; charset=utf-8",
        )
    try:
        result = await fetch_full(request.prompt.strip(), request.systemInstructions, request.model)
        return ChatResponse(success=True, response=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
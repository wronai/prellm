"""preLLM API Server — OpenAI-compatible /v1/chat/completions endpoint with small LLM preprocessing.

Usage:
    uvicorn prellm.server:app --host 0.0.0.0 --port 8080
    # or
    prellm serve --port 8080 --small ollama/qwen2.5:3b --large gpt-4o-mini

Curl:
    curl http://localhost:8080/v1/chat/completions -d '{"model":"prellm:qwen→claude","messages":[...]}'
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from prellm.core import PreLLM, preprocess_and_execute
from prellm.env_config import get_env_config, EnvConfig
from prellm.models import (
    DecompositionStrategy,
    DomainRule,
    LLMProviderConfig,
    PreLLMConfig,
    PreLLMResponse,
)

logger = logging.getLogger("prellm.server")

# ============================================================
# Server config from env vars (LiteLLM-compatible)
# ============================================================

_env = get_env_config()

SMALL_MODEL = _env.small_model
LARGE_MODEL = _env.large_model
DEFAULT_STRATEGY = _env.strategy
CONFIG_PATH = _env.config_path
MASTER_KEY = _env.master_key

# ============================================================
# Request / Response models (OpenAI-compatible)
# ============================================================

class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class PreLLMExtras(BaseModel):
    """preLLM-specific extensions in the request body."""
    user_context: str | dict[str, str] | None = None
    strategy: str | None = None
    response_format: str | None = None  # "yaml" | "json" | None
    show_stages: bool = False
    domain_rules: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "prellm:default"
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    prellm: PreLLMExtras | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class PreLLMMeta(BaseModel):
    """preLLM metadata in the response."""
    small_model: str = ""
    large_model: str = ""
    strategy: str = ""
    intent: str | None = None
    confidence: float | None = None
    matched_rule: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    subtasks: int = 0
    retries: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)
    prellm_meta: PreLLMMeta | None = None


class BatchItem(BaseModel):
    query: str
    context: str | dict[str, str] | None = None
    strategy: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    small_model: str = ""
    large_model: str = ""


# ============================================================
# Auth middleware (LITELLM_MASTER_KEY)
# ============================================================

class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer token auth using LITELLM_MASTER_KEY. Skips auth if key is not set."""

    OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        if not MASTER_KEY:
            return await call_next(request)

        if request.url.path in self.OPEN_PATHS:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
        else:
            token = request.headers.get("x-api-key", "")

        if token != MASTER_KEY:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid API key", "type": "authentication_error"}},
            )

        return await call_next(request)


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="preLLM API",
    description="OpenAI-compatible API with small LLM preprocessing before large LLM execution.",
    version="0.3.0",
)

app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_model_pair(model_str: str) -> tuple[str, str]:
    """Parse 'prellm:qwen→claude' or 'prellm:small→large' into (small, large) model strings.

    Special cases:
        'prellm:default' → server defaults
        'prellm:qwen→claude' → ("qwen", "claude")
        'prellm:phi3->gpt-4o' → ("phi3", "gpt-4o")
        'prellm:gpt-4o-mini' → (default_small, "gpt-4o-mini")
        'gpt-4o-mini' → (default_small, "gpt-4o-mini")
    """
    small, large = SMALL_MODEL, LARGE_MODEL

    if ":" in model_str:
        _, pair = model_str.split(":", 1)

        if pair.strip().lower() == "default":
            return small, large

        if "\u2192" in pair:
            parts = pair.split("\u2192", 1)
        elif "->" in pair:
            parts = pair.split("->", 1)
        else:
            parts = [pair]

        if len(parts) == 2:
            small = parts[0].strip() if parts[0].strip() else small
            large = parts[1].strip() if parts[1].strip() else large
        elif len(parts) == 1 and parts[0].strip():
            large = parts[0].strip()

    return small, large


def _build_prellm_meta(result: PreLLMResponse, strategy: str) -> PreLLMMeta:
    """Build preLLM metadata from a PreLLMResponse."""
    meta = PreLLMMeta(
        small_model=result.small_model_used,
        large_model=result.model_used,
        strategy=strategy,
        retries=result.retries,
    )
    if result.decomposition:
        if result.decomposition.classification:
            meta.intent = result.decomposition.classification.intent
            meta.confidence = result.decomposition.classification.confidence
        meta.matched_rule = result.decomposition.matched_rule
        meta.missing_fields = result.decomposition.missing_fields
        meta.subtasks = len(result.decomposition.sub_queries)
    return meta


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
async def health() -> HealthResponse:
    import prellm
    return HealthResponse(
        version=prellm.__version__,
        small_model=SMALL_MODEL,
        large_model=LARGE_MODEL,
    )


@app.get("/v1/models")
async def list_models():
    """List available model pairs."""
    return {
        "object": "list",
        "data": [
            {"id": f"prellm:{SMALL_MODEL}→{LARGE_MODEL}", "object": "model", "owned_by": "prellm"},
            {"id": "prellm:default", "object": "model", "owned_by": "prellm"},
        ],
    }


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions with preLLM preprocessing."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    # Extract user query from last user message
    user_query = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            user_query = msg.content
            break

    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found")

    # Parse model pair
    small, large = _parse_model_pair(req.model)

    # Extract preLLM extras
    extras = req.prellm or PreLLMExtras()
    strategy = extras.strategy or DEFAULT_STRATEGY

    # Build kwargs
    kwargs: dict[str, Any] = {}
    if req.max_tokens:
        kwargs["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature

    # Handle YAML response format hint
    if extras.response_format == "yaml":
        user_query = f"{user_query}\n\nRespond in YAML format."

    # Streaming
    if req.stream:
        return StreamingResponse(
            _stream_response(user_query, small, large, strategy, extras, **kwargs),
            media_type="text/event-stream",
        )

    # Non-streaming
    result = await preprocess_and_execute(
        query=user_query,
        small_llm=small,
        large_llm=large,
        strategy=strategy,
        user_context=extras.user_context,
        domain_rules=extras.domain_rules,
        config_path=CONFIG_PATH,
        **kwargs,
    )

    meta = _build_prellm_meta(result, strategy)

    return ChatCompletionResponse(
        model=req.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=result.content),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=len(user_query.split()),
            completion_tokens=len(result.content.split()),
            total_tokens=len(user_query.split()) + len(result.content.split()),
        ),
        prellm_meta=meta,
    )


async def _stream_response(
    query: str,
    small: str,
    large: str,
    strategy: str,
    extras: PreLLMExtras,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """SSE streaming response with stage progress."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Stage 1: preprocessing
    if extras.show_stages:
        yield f"data: {json.dumps({'id': request_id, 'stage': 'preprocessing', 'model': small, 'progress': 0})}\n\n"

    result = await preprocess_and_execute(
        query=query,
        small_llm=small,
        large_llm=large,
        strategy=strategy,
        user_context=extras.user_context,
        domain_rules=extras.domain_rules,
        config_path=CONFIG_PATH,
        **kwargs,
    )

    if extras.show_stages:
        yield f"data: {json.dumps({'id': request_id, 'stage': 'preprocessing', 'model': small, 'progress': 100})}\n\n"
        yield f"data: {json.dumps({'id': request_id, 'stage': 'execution', 'model': large, 'progress': 100})}\n\n"

    # Stream content in chunks (simulate token-by-token)
    words = result.content.split()
    chunk_size = max(1, len(words) // 5)

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        delta = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "model": f"prellm:{small}→{large}",
            "choices": [{"index": 0, "delta": {"content": chunk + " "}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(delta)}\n\n"

    # Final chunk
    final = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "model": f"prellm:{small}→{large}",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "prellm_meta": _build_prellm_meta(result, strategy).model_dump(),
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/batch")
async def batch_process(items: list[BatchItem]):
    """Process multiple queries in parallel."""
    if not items:
        raise HTTPException(status_code=400, detail="Empty batch")

    async def _process_one(item: BatchItem) -> dict[str, Any]:
        result = await preprocess_and_execute(
            query=item.query,
            small_llm=SMALL_MODEL,
            large_llm=LARGE_MODEL,
            strategy=item.strategy or DEFAULT_STRATEGY,
            user_context=item.context,
            config_path=CONFIG_PATH,
        )
        return {
            "query": item.query,
            "content": result.content,
            "model_used": result.model_used,
            "small_model_used": result.small_model_used,
            "prellm_meta": _build_prellm_meta(result, item.strategy or DEFAULT_STRATEGY).model_dump(),
        }

    results = await asyncio.gather(*[_process_one(item) for item in items])
    return {"object": "batch", "results": list(results)}


def create_app(
    small_model: str | None = None,
    large_model: str | None = None,
    strategy: str | None = None,
    config_path: str | None = None,
    master_key: str | None = None,
    dotenv_path: str | None = None,
) -> FastAPI:
    """Factory function to create a configured preLLM API server.

    Reads .env file first, then overrides with explicit args.
    """
    global SMALL_MODEL, LARGE_MODEL, DEFAULT_STRATEGY, CONFIG_PATH, MASTER_KEY, _env

    if dotenv_path:
        _env = get_env_config(dotenv_path)
        SMALL_MODEL = _env.small_model
        LARGE_MODEL = _env.large_model
        DEFAULT_STRATEGY = _env.strategy
        CONFIG_PATH = _env.config_path
        MASTER_KEY = _env.master_key

    if small_model:
        SMALL_MODEL = small_model
    if large_model:
        LARGE_MODEL = large_model
    if strategy:
        DEFAULT_STRATEGY = strategy
    if config_path:
        CONFIG_PATH = config_path
    if master_key is not None:
        MASTER_KEY = master_key
    return app

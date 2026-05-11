"""FastAPI OpenAI-compatible HTTP API server for a fixed nanobot session.

Provides /v1/chat/completions, /v1/models, and /health endpoints.
All requests route to persistent API sessions managed by AgentLoop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json as _json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from starlette.datastructures import UploadFile

from nanobot.config.paths import get_media_dir
from nanobot.utils.helpers import safe_filename
from nanobot.utils.media_decode import (
    MAX_FILE_SIZE,
)
from nanobot.utils.media_decode import (
    FileSizeExceeded as _FileSizeExceeded,
)
from nanobot.utils.media_decode import (
    save_base64_data_url as _save_base64_data_url,
)
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

__all__ = (
    "MAX_FILE_SIZE",
    "_FileSizeExceeded",
    "_save_base64_data_url",
    "create_app",
    "handle_chat_completions",
)


API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"
_DEFAULT_UPLOAD_TEXT = "请分析上传的文件"


class CompatJSONResponse(JSONResponse):
    """JSONResponse with the old aiohttp-style ``status`` attribute for tests."""

    @property
    def status(self) -> int:
        return self.status_code


def _app_get(request: Request, key: str, default: Any = None) -> Any:
    app = getattr(request, "app", None)
    if isinstance(app, dict):
        return app.get(key, default)
    state = getattr(app, "state", None)
    if state is None:
        return default
    return getattr(state, key, default)


def _request_content_type(request: Request) -> str:
    content_type = getattr(request, "content_type", None)
    if isinstance(content_type, str):
        return content_type
    headers = getattr(request, "headers", {}) or {}
    raw = headers.get("content-type", "") if hasattr(headers, "get") else ""
    if not isinstance(raw, str):
        return ""
    return raw.split(";", 1)[0].strip().lower()


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _error_json(status: int, message: str, err_type: str = "invalid_request_error") -> JSONResponse:
    return CompatJSONResponse(
        {"error": {"message": message, "type": err_type, "code": status}},
        status_code=status,
    )


def _chat_completion_response(content: str, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _response_text(value: Any) -> str:
    """Normalize process_direct output to plain assistant text."""
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    return str(value)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_chunk(delta: str, model: str, chunk_id: str, finish_reason: str | None = None) -> bytes:
    """Format a single OpenAI-compatible SSE chunk."""
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {_json.dumps(payload)}\n\n".encode()


_SSE_DONE = b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def _parse_json_content(body: dict) -> tuple[str, list[str]]:
    """Parse JSON request body. Returns (text, media_paths)."""
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("Only a single user message is supported")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("Only a single user message is supported")

    user_content = message.get("content", "")
    media_dir = get_media_dir("api")
    media_paths: list[str] = []

    if isinstance(user_content, list):
        text_parts: list[str] = []
        for part in user_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    saved = _save_base64_data_url(url, media_dir)
                    if saved:
                        media_paths.append(saved)
                elif url:
                    raise ValueError(
                        "Remote image URLs are not supported. "
                        "Use base64 data URLs or upload files via multipart/form-data."
                    )
        text = " ".join(text_parts)
    elif isinstance(user_content, str):
        text = user_content
    else:
        raise ValueError("Invalid content format")

    return text, media_paths


async def _parse_multipart(request: Request) -> tuple[str, list[str], str | None, str | None]:
    """Parse multipart/form-data. Returns (text, media_paths, session_id, model)."""
    media_dir = get_media_dir("api")
    form = await request.form()
    text = ""
    session_id = None
    model = None
    media_paths: list[str] = []

    for name, part in form.multi_items():
        if name == "message" and not isinstance(part, UploadFile):
            text = str(part)
            continue
        if name == "session_id" and not isinstance(part, UploadFile):
            session_id = str(part).strip()
            continue
        if name == "model" and not isinstance(part, UploadFile):
            model = str(part).strip()
            continue
        if name != "files" or not isinstance(part, UploadFile):
            continue

        raw = await part.read()
        if len(raw) > MAX_FILE_SIZE:
            raise _FileSizeExceeded(
                f"File '{part.filename}' exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit"
            )
        base = safe_filename(part.filename or "upload.bin")
        filename = f"{uuid.uuid4().hex[:12]}_{base}"
        dest = media_dir / filename
        dest.write_bytes(raw)
        media_paths.append(str(dest))

    if not text:
        text = _DEFAULT_UPLOAD_TEXT

    return text, media_paths, session_id, model


def _session_lock_for(request: Request, session_key: str) -> asyncio.Lock:
    session_locks: dict[str, asyncio.Lock] | None = _app_get(request, "session_locks")
    if session_locks is not None:
        return session_locks.setdefault(session_key, asyncio.Lock())

    lock = _app_get(request, "session_lock")
    if lock is not None and hasattr(lock, "__aenter__") and hasattr(lock, "__aexit__"):
        return lock
    return asyncio.Lock()


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def handle_chat_completions(request: Request):
    """POST /v1/chat/completions - supports JSON and multipart/form-data."""
    content_type = _request_content_type(request)

    agent_loop = _app_get(request, "agent_loop")
    timeout_s: float = _app_get(request, "request_timeout", 120.0)
    model_name: str = _app_get(request, "model_name", "nanobot")

    stream = False
    try:
        if content_type.startswith("multipart/"):
            text, media_paths, session_id, requested_model = await _parse_multipart(request)
        else:
            try:
                body = await request.json()
            except Exception:
                return _error_json(400, "Invalid JSON body")
            stream = body.get("stream", False)
            requested_model = body.get("model")
            text, media_paths = _parse_json_content(body)
            session_id = body.get("session_id")
    except ValueError as e:
        return _error_json(400, str(e))
    except _FileSizeExceeded as e:
        return _error_json(413, str(e), err_type="invalid_request_error")
    except Exception:
        logger.exception("Error parsing upload")
        return _error_json(413, "File too large or invalid upload")

    if requested_model and requested_model != model_name:
        return _error_json(400, f"Only configured model '{model_name}' is available")

    session_key = f"api:{session_id}" if session_id else API_SESSION_KEY
    session_lock = _session_lock_for(request, session_key)

    logger.info(
        "API request session_key={} media={} text={} stream={}",
        session_key, len(media_paths), text[:80], stream,
    )

    if stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        async def _events() -> AsyncIterator[bytes]:
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            stream_failed = False
            emitted_content = False

            async def _on_stream(token: str) -> None:
                nonlocal emitted_content
                if token:
                    emitted_content = True
                await queue.put(token)

            async def _on_stream_end(*_a: Any, **_kw: Any) -> None:
                return None

            async def _run() -> None:
                nonlocal stream_failed
                try:
                    async with session_lock:
                        response = await asyncio.wait_for(
                            agent_loop.process_direct(
                                content=text,
                                media=media_paths if media_paths else None,
                                session_key=session_key,
                                channel="api",
                                chat_id=API_CHAT_ID,
                                on_stream=_on_stream,
                                on_stream_end=_on_stream_end,
                            ),
                            timeout=timeout_s,
                        )
                        if not emitted_content:
                            response_text = _response_text(response)
                            if response_text.strip():
                                await queue.put(response_text)
                except Exception:
                    stream_failed = True
                    logger.exception("Streaming error for session {}", session_key)
                finally:
                    await queue.put(None)

            task = asyncio.create_task(_run())
            try:
                while True:
                    token = await queue.get()
                    if token is None:
                        break
                    yield _sse_chunk(token, model_name, chunk_id)
            finally:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

            if not stream_failed:
                yield _sse_chunk("", model_name, chunk_id, finish_reason="stop")
                yield _SSE_DONE

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    fallback = EMPTY_FINAL_RESPONSE_MESSAGE

    try:
        async with session_lock:
            try:
                response = await asyncio.wait_for(
                    agent_loop.process_direct(
                        content=text,
                        media=media_paths if media_paths else None,
                        session_key=session_key,
                        channel="api",
                        chat_id=API_CHAT_ID,
                    ),
                    timeout=timeout_s,
                )
                response_text = _response_text(response)

                if not response_text or not response_text.strip():
                    logger.warning("Empty response for session {}, retrying", session_key)
                    retry_response = await asyncio.wait_for(
                        agent_loop.process_direct(
                            content=text,
                            media=media_paths if media_paths else None,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                        ),
                        timeout=timeout_s,
                    )
                    response_text = _response_text(retry_response)
                    if not response_text or not response_text.strip():
                        logger.warning("Empty response after retry, using fallback")
                        response_text = fallback

            except asyncio.TimeoutError:
                return _error_json(504, f"Request timed out after {timeout_s}s")
            except Exception:
                logger.exception("Error processing request for session {}", session_key)
                return _error_json(500, "Internal server error", err_type="server_error")
    except Exception:
        logger.exception("Unexpected API lock error for session {}", session_key)
        return _error_json(500, "Internal server error", err_type="server_error")

    return CompatJSONResponse(_chat_completion_response(response_text, model_name))


async def handle_models(request: Request) -> JSONResponse:
    """GET /v1/models"""
    model_name = _app_get(request, "model_name", "nanobot")
    return CompatJSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nanobot",
                }
            ],
        }
    )


async def handle_health() -> JSONResponse:
    """GET /health"""
    return CompatJSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    agent_loop, model_name: str = "nanobot", request_timeout: float = 120.0
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        agent_loop: An initialized AgentLoop instance.
        model_name: Model name reported in responses.
        request_timeout: Per-request timeout in seconds.
    """
    app = FastAPI(title="nanobot API")
    app.state.agent_loop = agent_loop
    app.state.model_name = model_name
    app.state.request_timeout = request_timeout
    app.state.session_locks = {}

    app.add_api_route("/v1/chat/completions", handle_chat_completions, methods=["POST"])
    app.add_api_route("/v1/models", handle_models, methods=["GET"])
    app.add_api_route("/health", handle_health, methods=["GET"])
    return app

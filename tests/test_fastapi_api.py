"""FastAPI coverage for the OpenAI-compatible API surface."""

from __future__ import annotations

import asyncio
import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from nanobot.api.server import API_CHAT_ID, API_SESSION_KEY, create_app


def _make_mock_agent(response_text: str = "mock response") -> MagicMock:
    agent = MagicMock()
    agent.process_direct = AsyncMock(return_value=response_text)
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()
    return agent


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_fastapi_successful_request_uses_fixed_api_session() -> None:
    agent = _make_mock_agent()
    app = create_app(agent, model_name="test-model")

    async with _client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "mock response"
    assert body["model"] == "test-model"
    agent.process_direct.assert_called_once_with(
        content="hello",
        media=None,
        session_key=API_SESSION_KEY,
        channel="api",
        chat_id=API_CHAT_ID,
    )


@pytest.mark.asyncio
async def test_fastapi_models_and_health_endpoints() -> None:
    app = create_app(_make_mock_agent(), model_name="m")

    async with _client(app) as client:
        models = await client.get("/v1/models")
        health = await client.get("/health")

    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "m"
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_fastapi_streaming_sse_returns_chunks() -> None:
    agent = MagicMock()

    async def fake_process_direct(*, on_stream=None, on_stream_end=None, **kwargs):
        assert on_stream is not None
        assert on_stream_end is not None
        await on_stream("Hello")
        await on_stream(" world")
        await on_stream_end(resuming=False)
        return "Hello world"

    agent.process_direct = fake_process_direct
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()
    app = create_app(agent, model_name="m")

    async with _client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    data_lines = [
        line[len("data: "):]
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert data_lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in data_lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[1]["choices"][0]["delta"]["content"] == " world"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_fastapi_multipart_upload_saves_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    agent = _make_mock_agent()
    app = create_app(agent, model_name="m")

    async with _client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            data={"message": "analyze this", "session_id": "s1"},
            files={"files": ("report.txt", BytesIO(b"hello"), "text/plain")},
        )

    assert resp.status_code == 200
    call_kwargs = agent.process_direct.call_args.kwargs
    assert call_kwargs["content"] == "analyze this"
    assert call_kwargs["session_key"] == "api:s1"
    media = call_kwargs.get("media") or []
    assert len(media) == 1
    assert media[0].endswith("report.txt")


@pytest.mark.asyncio
async def test_fastapi_fixed_session_requests_are_serialized() -> None:
    order: list[str] = []

    async def slow_process(content, **kwargs):
        order.append(f"start:{content}")
        await asyncio.sleep(0.05)
        order.append(f"end:{content}")
        return content

    agent = MagicMock()
    agent.process_direct = slow_process
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()
    app = create_app(agent, model_name="m")

    async with _client(app) as client:
        async def send(msg: str) -> httpx.Response:
            return await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": msg}]},
            )

        r1, r2 = await asyncio.gather(send("first"), send("second"))

    assert r1.status_code == 200
    assert r2.status_code == 200
    if order[0] == "start:first":
        assert order.index("end:first") < order.index("start:second")
    else:
        assert order.index("end:second") < order.index("start:first")

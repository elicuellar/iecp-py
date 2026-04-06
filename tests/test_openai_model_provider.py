"""OpenAI Model Provider Tests -- SS11 of the IECP specification.

Mocks httpx to simulate SSE streaming from an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from iecp_core.artificer.openai_model_provider import OpenAIModelProvider
from iecp_core.artificer.types import ArtificerModelConfig, ModelMessage

CONFIG = ArtificerModelConfig(
    base_url="https://api.example.com",
    api_key="test-key",
    model="test-model",
    max_tokens=1024,
    temperature=0.5,
    timeout_ms=5000,
)

MESSAGES = [
    ModelMessage(role="system", content="You are a helpful assistant."),
    ModelMessage(role="user", content="Hello!"),
]


def _make_sse_data(chunks: list[str], includes_done: bool = True) -> bytes:
    """Build SSE byte stream from content chunks."""
    data = ""
    for chunk in chunks:
        data += f'data: {json.dumps({"choices": [{"delta": {"content": chunk}, "finish_reason": None}]})}\n\n'
    if includes_done:
        data += "data: [DONE]\n\n"
    return data.encode("utf-8")


def _make_finish_reason_data(chunks: list[str]) -> bytes:
    """Build SSE byte stream that uses finish_reason instead of [DONE]."""
    data = ""
    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        data += f'data: {json.dumps({"choices": [{"delta": {"content": chunk}, "finish_reason": "stop" if is_last else None}]})}\n\n'
    return data.encode("utf-8")


class _FakeStreamResponse:
    """Mimics httpx async streaming response context manager."""

    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self._body = body
        self._chunks = [body]  # deliver all at once

    async def aread(self) -> bytes:
        return self._body

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeAsyncClient:
    """Mimics httpx.AsyncClient for testing."""

    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response
        self.last_method: str | None = None
        self.last_url: str | None = None
        self.last_kwargs: dict = {}

    def stream(self, method: str, url: str, **kwargs):
        self.last_method = method
        self.last_url = url
        self.last_kwargs = kwargs
        return self._response

    async def aclose(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.mark.asyncio
class TestOpenAIModelProvider:
    async def test_streams_chunks_from_sse(self) -> None:
        body = _make_sse_data(["Hello", " world", "!"])
        fake_response = _FakeStreamResponse(200, body)
        fake_client = _FakeAsyncClient(fake_response)

        with patch("iecp_core.artificer.openai_model_provider.httpx.AsyncClient", return_value=fake_client):
            provider = OpenAIModelProvider()
            chunks: list[str] = []
            got_done = False

            async for chunk in provider.stream(MESSAGES, CONFIG):
                if chunk.text:
                    chunks.append(chunk.text)
                if chunk.done:
                    got_done = True

            assert chunks == ["Hello", " world", "!"]
            assert got_done is True

            # Verify URL
            assert fake_client.last_url == "https://api.example.com/v1/chat/completions"
            # Verify auth header
            assert fake_client.last_kwargs["headers"]["Authorization"] == "Bearer test-key"

    async def test_handles_abort(self) -> None:
        """Test that abort closes the client."""
        provider = OpenAIModelProvider()
        mock_client = MagicMock()
        provider._client = mock_client

        provider.abort()

        mock_client.close.assert_called_once()
        assert provider._client is None

    async def test_handles_http_error(self) -> None:
        body = b"Rate limit exceeded"
        fake_response = _FakeStreamResponse(429, body)
        fake_client = _FakeAsyncClient(fake_response)

        with patch("iecp_core.artificer.openai_model_provider.httpx.AsyncClient", return_value=fake_client):
            provider = OpenAIModelProvider()

            with pytest.raises(RuntimeError, match="429"):
                async for _chunk in provider.stream(MESSAGES, CONFIG):
                    pass

    async def test_handles_malformed_sse(self) -> None:
        data = (
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
            "data: {INVALID_JSON}\n\n"
            'data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n'
        )
        fake_response = _FakeStreamResponse(200, data.encode("utf-8"))
        fake_client = _FakeAsyncClient(fake_response)

        with patch("iecp_core.artificer.openai_model_provider.httpx.AsyncClient", return_value=fake_client):
            provider = OpenAIModelProvider()
            chunks: list[str] = []

            async for chunk in provider.stream(MESSAGES, CONFIG):
                if chunk.text:
                    chunks.append(chunk.text)

            assert chunks == ["ok", "!"]

    async def test_handles_finish_reason_without_done(self) -> None:
        body = _make_finish_reason_data(["Hi", " there"])
        fake_response = _FakeStreamResponse(200, body)
        fake_client = _FakeAsyncClient(fake_response)

        with patch("iecp_core.artificer.openai_model_provider.httpx.AsyncClient", return_value=fake_client):
            provider = OpenAIModelProvider()
            chunks: list[str] = []

            async for chunk in provider.stream(MESSAGES, CONFIG):
                if chunk.text:
                    chunks.append(chunk.text)

            assert "Hi" in chunks
            assert " there" in chunks

    async def test_sends_correct_request_body(self) -> None:
        body = _make_sse_data(["ok"])
        fake_response = _FakeStreamResponse(200, body)
        fake_client = _FakeAsyncClient(fake_response)

        with patch("iecp_core.artificer.openai_model_provider.httpx.AsyncClient", return_value=fake_client):
            provider = OpenAIModelProvider()
            async for _chunk in provider.stream(MESSAGES, CONFIG):
                pass

            req_body = fake_client.last_kwargs["json"]
            assert req_body["model"] == "test-model"
            assert req_body["max_tokens"] == 1024
            assert req_body["temperature"] == 0.5
            assert req_body["stream"] is True
            assert req_body["messages"] == [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ]

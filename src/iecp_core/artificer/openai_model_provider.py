"""OpenAI-Compatible Model Provider -- SS11 of the IECP specification.

Implements the ModelProvider interface using httpx against any
OpenAI-compatible /v1/chat/completions endpoint. Works with
OpenAI, Gemini AI Studio, Anthropic via proxy, etc.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .types import ArtificerModelConfig, ModelMessage, StreamChunk


class OpenAIModelProvider:
    """OpenAI-compatible streaming model provider using httpx."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def stream(
        self,
        messages: list[ModelMessage],
        config: ArtificerModelConfig,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from an OpenAI-compatible endpoint."""
        base_url = config.base_url.rstrip("/")
        url = f"{base_url}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }

        body = {
            "model": config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": True,
        }

        timeout = config.timeout_ms / 1000.0

        self._client = httpx.AsyncClient(timeout=timeout)
        try:
            async with self._client.stream(
                "POST", url, headers=headers, json=body
            ) as response:
                if response.status_code != 200:
                    body_text = await response.aread()
                    raise RuntimeError(
                        f"Model API returned {response.status_code}: {body_text.decode()}"
                    )

                buffer = ""
                async for raw_bytes in response.aiter_bytes():
                    buffer += raw_bytes.decode("utf-8")
                    lines = buffer.split("\n")
                    buffer = lines.pop()  # keep partial line

                    for line in lines:
                        trimmed = line.strip()
                        if not trimmed or trimmed.startswith(":"):
                            continue

                        if trimmed == "data: [DONE]":
                            yield StreamChunk(text="", done=True)
                            return

                        if trimmed.startswith("data: "):
                            json_str = trimmed[6:]
                            try:
                                parsed = json.loads(json_str)
                            except json.JSONDecodeError:
                                continue

                            choices = parsed.get("choices", [])
                            if not choices:
                                continue

                            choice = choices[0]
                            text = (choice.get("delta") or {}).get("content", "")
                            is_done = choice.get("finish_reason") is not None

                            if text or is_done:
                                yield StreamChunk(text=text, done=is_done)
                                if is_done:
                                    return

                # Stream ended without [DONE]
                yield StreamChunk(text="", done=True)
        finally:
            client = self._client
            self._client = None
            await client.aclose()

    def abort(self) -> None:
        """Abort in-flight generation."""
        if self._client is not None:
            client = self._client
            self._client = None
            # Close synchronously -- httpx handles this gracefully
            try:
                client.close()
            except Exception:
                pass

"""Artificer Runtime Types -- §11 of the IECP specification.

Defines the model interface, persona config, streaming abstractions,
and runtime configuration for server-side AI agents (Artificers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from pydantic import BaseModel

from ..types.entity import EntityId
from ..types.event import ConversationId, Event


# -- Model Messages -----------------------------------------------------------


class ModelMessage(BaseModel):
    """A single message in the model's conversation format."""

    role: str  # 'system' | 'user' | 'assistant'
    content: str


# -- Model Configuration -------------------------------------------------------


@dataclass
class ArtificerModelConfig:
    """Configuration for an Artificer's model."""

    base_url: str
    api_key: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_ms: int = 120_000


# -- Persona ------------------------------------------------------------------


@dataclass
class ArtificerPersona:
    """Persona configuration for an Artificer."""

    name: str
    role: str
    phase: str
    system_prompt: str


# -- Streaming ----------------------------------------------------------------


@dataclass
class StreamChunk:
    """A chunk yielded during streaming."""

    text: str
    done: bool


# -- Model Provider -----------------------------------------------------------


class ModelProvider(Protocol):
    """Abstraction over any OpenAI-compatible streaming API."""

    def stream(
        self,
        messages: list[ModelMessage],
        config: ArtificerModelConfig,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion. Returns an async iterable of chunks."""
        ...

    def abort(self) -> None:
        """Abort in-flight generation."""
        ...


# -- Runtime Configuration ----------------------------------------------------


@dataclass(frozen=True)
class ArtificerRuntimeConfig:
    """Runtime configuration for the ArtificerRuntime."""

    stream_flush_interval_ms: int = 100
    """How often to emit stream chunks to gateway (ms)."""

    max_retries: int = 3
    """Max retries on model error before posting system event."""

    retry_base_delay_ms: int = 1000
    """Base delay for retry backoff (ms)."""

    max_concurrent_invocations: int = 5
    """Maximum concurrent artificer invocations across all conversations."""


DEFAULT_ARTIFICER_RUNTIME_CONFIG = ArtificerRuntimeConfig()


# -- Runtime Events -----------------------------------------------------------


class StreamChunkEvent(BaseModel):
    """Emitted when a stream chunk is ready for the gateway."""

    conversation_id: ConversationId
    entity_id: EntityId
    text: str
    chunk_index: int


class MessageCommittedEvent(BaseModel):
    """Emitted when a message is committed to the event log."""

    conversation_id: ConversationId
    entity_id: EntityId
    event: Event


class InterruptedEvent(BaseModel):
    """Emitted when generation is interrupted."""

    conversation_id: ConversationId
    entity_id: EntityId
    partial_text: str


class RuntimeErrorEvent(BaseModel):
    """Emitted on error during generation."""

    conversation_id: ConversationId
    entity_id: EntityId
    error: str
    retry_count: int

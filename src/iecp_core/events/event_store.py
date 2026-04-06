from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from ..types import (
    ConversationId,
    Event,
    EventId,
    EventStatus,
)


class ReadEventsOptions(BaseModel):
    after: EventId | None = None
    before: EventId | None = None
    limit: int = 50
    event_types: list[str] | None = None
    author_id: str | None = None


class ReadEventsResult(BaseModel):
    events: list[Event]
    has_more: bool = False
    cursor: EventId | None = None


class EventStore(Protocol):
    async def append(self, event: Event) -> Event: ...

    async def read_events(
        self,
        conversation_id: ConversationId,
        options: ReadEventsOptions | None = None,
    ) -> ReadEventsResult: ...

    async def read_by_batch(self, batch_id: str) -> list[Event]: ...

    async def get_by_id(self, event_id: EventId) -> Event | None: ...

    async def update_status(
        self, event_id: EventId, status: EventStatus
    ) -> None: ...

from __future__ import annotations

from typing import Protocol

from ..types import ConversationId, EntityCursor, EntityId, EventId
from ..utils import compare_ulids


class CursorRepository(Protocol):
    async def find(self, entity_id: EntityId, conversation_id: ConversationId) -> EntityCursor | None: ...
    async def save(self, cursor: EntityCursor) -> EntityCursor: ...


class CursorManager:
    def __init__(self, repository: CursorRepository) -> None:
        self._repo = repository

    async def get_cursor(
        self, entity_id: EntityId, conversation_id: ConversationId
    ) -> EntityCursor:
        cursor = await self._repo.find(entity_id, conversation_id)
        if cursor is not None:
            return cursor
        new_cursor = EntityCursor(
            entity_id=entity_id,
            conversation_id=conversation_id,
        )
        return await self._repo.save(new_cursor)

    async def advance_received(
        self,
        entity_id: EntityId,
        conversation_id: ConversationId,
        event_id: EventId,
    ) -> EntityCursor:
        cursor = await self.get_cursor(entity_id, conversation_id)

        # Only advance forward
        if cursor.cursor_received is not None:
            if compare_ulids(event_id, cursor.cursor_received) <= 0:
                return cursor

        updated = cursor.model_copy(update={"cursor_received": event_id})
        return await self._repo.save(updated)

    async def advance_processed(
        self,
        entity_id: EntityId,
        conversation_id: ConversationId,
        event_id: EventId,
    ) -> EntityCursor:
        cursor = await self.get_cursor(entity_id, conversation_id)

        # Only advance forward
        if cursor.cursor_processed is not None:
            if compare_ulids(event_id, cursor.cursor_processed) <= 0:
                return cursor

        # Can't exceed received
        if cursor.cursor_received is not None:
            if compare_ulids(event_id, cursor.cursor_received) > 0:
                raise ValueError(
                    "Cannot advance processed cursor beyond received cursor"
                )
        else:
            raise ValueError(
                "Cannot advance processed cursor when no events have been received"
            )

        updated = cursor.model_copy(update={"cursor_processed": event_id})
        return await self._repo.save(updated)

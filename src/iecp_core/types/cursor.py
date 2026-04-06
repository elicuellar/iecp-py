from __future__ import annotations

from pydantic import BaseModel

from .entity import EntityId
from .event import ConversationId, EventId


class EntityCursor(BaseModel):
    entity_id: EntityId
    conversation_id: ConversationId
    cursor_received: EventId | None = None
    cursor_processed: EventId | None = None


def is_cursor_order_valid(cursor: EntityCursor) -> bool:
    if cursor.cursor_received is None and cursor.cursor_processed is None:
        return True
    if cursor.cursor_received is not None and cursor.cursor_processed is None:
        return True
    if cursor.cursor_received is None and cursor.cursor_processed is not None:
        return False
    # Both are set — processed must be <= received
    return cursor.cursor_processed <= cursor.cursor_received  # type: ignore[operator]


def has_unprocessed_events(cursor: EntityCursor) -> bool:
    if cursor.cursor_received is None:
        return False
    if cursor.cursor_processed is None:
        return True
    return cursor.cursor_processed < cursor.cursor_received

from __future__ import annotations

from typing import NewType

from pydantic import BaseModel

from .event import EventId

FileId = NewType("FileId", str)


class FileAttachment(BaseModel):
    id: FileId
    event_id: EventId
    filename: str
    mime_type: str
    size_bytes: int
    url: str
    created_at: str

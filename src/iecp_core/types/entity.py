from __future__ import annotations

from typing import Literal, NewType

from pydantic import BaseModel

EntityId = NewType("EntityId", str)
EntityType = Literal["human", "artificer", "daemon"]
EntityLifecycleStatus = Literal[
    "joined", "active", "idle", "processing", "disconnected", "left"
]
ResponseLatency = Literal["low", "medium", "high"]
AdapterType = Literal["mcp_cli"]


class EntityCapabilities(BaseModel):
    can_initiate: bool = True
    can_respond: bool = True
    can_moderate: bool = False
    response_latency: ResponseLatency = "medium"
    supported_adapters: list[AdapterType] = []


class Entity(BaseModel):
    id: EntityId
    name: str
    type: EntityType
    capabilities: EntityCapabilities
    created_at: str
    updated_at: str
    metadata: dict = {}


VALID_LIFECYCLE_TRANSITIONS: dict[EntityLifecycleStatus, list[EntityLifecycleStatus]] = {
    "joined": ["active", "left"],
    "active": ["idle", "processing", "disconnected", "left"],
    "idle": ["active", "processing", "disconnected", "left"],
    "processing": ["active", "idle", "disconnected", "left"],
    "disconnected": ["active", "idle", "left"],
    "left": [],
}


def is_valid_lifecycle_transition(
    from_: EntityLifecycleStatus, to: EntityLifecycleStatus
) -> bool:
    valid_targets = VALID_LIFECYCLE_TRANSITIONS.get(from_, [])
    return to in valid_targets

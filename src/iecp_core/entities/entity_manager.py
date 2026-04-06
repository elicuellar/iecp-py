from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from ..types import Entity, EntityCapabilities, EntityId, EntityLifecycleStatus, EntityType, is_valid_lifecycle_transition
from ..utils import generate_id


class EntityRepository(Protocol):
    async def save(self, entity: Entity) -> Entity: ...
    async def find_by_id(self, entity_id: EntityId) -> Entity | None: ...
    async def update(self, entity_id: EntityId, updates: dict[str, Any]) -> Entity: ...
    async def delete(self, entity_id: EntityId) -> None: ...
    async def list(self) -> list[Entity]: ...


class EntityManager:
    def __init__(self, repository: EntityRepository) -> None:
        self._repo = repository

    async def create_entity(
        self,
        name: str,
        type: EntityType,
        capabilities: EntityCapabilities | None = None,
        metadata: dict | None = None,
    ) -> Entity:
        now = datetime.now(timezone.utc).isoformat()
        entity = Entity(
            id=EntityId(generate_id()),
            name=name,
            type=type,
            capabilities=capabilities or EntityCapabilities(),
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        return await self._repo.save(entity)

    async def get_entity(self, entity_id: EntityId) -> Entity | None:
        return await self._repo.find_by_id(entity_id)

    async def update_entity(self, entity_id: EntityId, updates: dict[str, Any]) -> Entity:
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        return await self._repo.update(entity_id, updates)

    async def delete_entity(self, entity_id: EntityId) -> None:
        await self._repo.delete(entity_id)

    def validate_lifecycle_transition(
        self, from_: EntityLifecycleStatus, to: EntityLifecycleStatus
    ) -> bool:
        return is_valid_lifecycle_transition(from_, to)

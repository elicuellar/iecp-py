from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from ..types import (
    Conversation,
    ConversationConfig,
    ConversationId,
    ConversationStatus,
    EntityId,
    EntityLifecycleStatus,
    Participant,
    ParticipantRole,
    validate_conversation_config,
)
from ..utils import generate_id


class ConversationRepository(Protocol):
    async def save(self, conversation: Conversation) -> Conversation: ...
    async def find_by_id(self, conversation_id: ConversationId) -> Conversation | None: ...
    async def update(self, conversation_id: ConversationId, updates: dict[str, Any]) -> Conversation: ...


class ConversationManager:
    def __init__(self, repository: ConversationRepository) -> None:
        self._repo = repository

    async def create_conversation(
        self,
        title: str,
        created_by: EntityId,
        config: ConversationConfig | None = None,
    ) -> Conversation:
        now = datetime.now(timezone.utc).isoformat()
        conv_config = config or ConversationConfig()

        errors = validate_conversation_config(conv_config)
        if errors:
            raise ValueError(f"Invalid conversation config: {'; '.join(errors)}")

        owner_participant = Participant(
            entity_id=created_by,
            conversation_id=ConversationId(generate_id()),
            role="owner",
            lifecycle_status="joined",
            joined_at=now,
        )

        conversation = Conversation(
            id=ConversationId(generate_id()),
            title=title,
            config=conv_config,
            status="active",
            created_by=created_by,
            created_at=now,
            updated_at=now,
            participants=[owner_participant],
        )
        # Fix the participant's conversation_id to match
        conversation.participants[0].conversation_id = conversation.id

        return await self._repo.save(conversation)

    async def get_conversation(self, conversation_id: ConversationId) -> Conversation | None:
        return await self._repo.find_by_id(conversation_id)

    async def update_config(
        self, conversation_id: ConversationId, config_updates: dict[str, Any]
    ) -> Conversation:
        conversation = await self._repo.find_by_id(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        merged = conversation.config.model_copy(update=config_updates)
        errors = validate_conversation_config(merged)
        if errors:
            raise ValueError(f"Invalid conversation config: {'; '.join(errors)}")

        return await self._repo.update(
            conversation_id,
            {
                "config": merged,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def archive_conversation(self, conversation_id: ConversationId) -> Conversation:
        return await self._repo.update(
            conversation_id,
            {
                "status": "archived",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def add_participant(
        self,
        conversation_id: ConversationId,
        entity_id: EntityId,
        role: ParticipantRole = "member",
    ) -> Conversation:
        conversation = await self._repo.find_by_id(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Check duplicate
        for p in conversation.participants:
            if p.entity_id == entity_id:
                raise ValueError(f"Entity {entity_id} is already a participant")

        # Check max_participants
        if len(conversation.participants) >= conversation.config.max_participants:
            raise ValueError(
                f"Conversation has reached maximum participants ({conversation.config.max_participants})"
            )

        now = datetime.now(timezone.utc).isoformat()
        participant = Participant(
            entity_id=entity_id,
            conversation_id=conversation_id,
            role=role,
            lifecycle_status="joined",
            joined_at=now,
        )

        new_participants = [*conversation.participants, participant]
        return await self._repo.update(
            conversation_id,
            {
                "participants": new_participants,
                "updated_at": now,
            },
        )

    async def remove_participant(
        self, conversation_id: ConversationId, entity_id: EntityId
    ) -> Conversation:
        conversation = await self._repo.find_by_id(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        new_participants = [p for p in conversation.participants if p.entity_id != entity_id]
        if len(new_participants) == len(conversation.participants):
            raise ValueError(f"Entity {entity_id} is not a participant")

        return await self._repo.update(
            conversation_id,
            {
                "participants": new_participants,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def get_participants(self, conversation_id: ConversationId) -> list[Participant]:
        conversation = await self._repo.find_by_id(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        return conversation.participants

    async def update_participant_lifecycle(
        self,
        conversation_id: ConversationId,
        entity_id: EntityId,
        lifecycle_status: EntityLifecycleStatus,
    ) -> Conversation:
        conversation = await self._repo.find_by_id(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        now = datetime.now(timezone.utc).isoformat()
        new_participants = []
        found = False
        for p in conversation.participants:
            if p.entity_id == entity_id:
                found = True
                new_participants.append(
                    p.model_copy(update={"lifecycle_status": lifecycle_status, "last_active_at": now})
                )
            else:
                new_participants.append(p)

        if not found:
            raise ValueError(f"Entity {entity_id} is not a participant")

        return await self._repo.update(
            conversation_id,
            {
                "participants": new_participants,
                "updated_at": now,
            },
        )

    async def update_participant_role(
        self,
        conversation_id: ConversationId,
        entity_id: EntityId,
        role: ParticipantRole,
    ) -> Conversation:
        conversation = await self._repo.find_by_id(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        new_participants = []
        found = False
        for p in conversation.participants:
            if p.entity_id == entity_id:
                found = True
                new_participants.append(p.model_copy(update={"role": role}))
            else:
                new_participants.append(p)

        if not found:
            raise ValueError(f"Entity {entity_id} is not a participant")

        return await self._repo.update(
            conversation_id,
            {
                "participants": new_participants,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

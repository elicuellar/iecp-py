from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .entity import EntityId, EntityLifecycleStatus
from .event import ConversationId

ParticipantRole = Literal["owner", "admin", "member"]
RespondentMode = Literal["auto", "mentioned_only"]
ConversationStatus = Literal["active", "archived"]


class ConversationConfig(BaseModel):
    debounce_ms: int = 3000
    debounce_adaptive: bool = True
    lock_ttl_max_ms: int = 60000
    allow_human_interrupt: bool = True
    max_cascade_depth: int = 3
    default_respondent_mode: RespondentMode = "auto"
    allow_unsolicited_ai: bool = False
    context_history_depth: int = 50
    context_summary_enabled: bool = True
    max_participants: int = 20
    max_ai_invocations_per_hour: int = 100
    max_concurrent_ai_processing: int = 3
    decision_capture_enabled: bool = True
    decision_requires_human_affirmation: bool = True


DEFAULT_CONVERSATION_CONFIG = ConversationConfig()


def validate_conversation_config(config: ConversationConfig) -> list[str]:
    errors: list[str] = []
    if config.debounce_ms < 0:
        errors.append("debounce_ms must be >= 0")
    if config.lock_ttl_max_ms < 1000:
        errors.append("lock_ttl_max_ms must be >= 1000")
    if config.max_cascade_depth < 0:
        errors.append("max_cascade_depth must be >= 0")
    if config.context_history_depth < 1:
        errors.append("context_history_depth must be >= 1")
    if config.max_participants < 2:
        errors.append("max_participants must be >= 2")
    if config.max_ai_invocations_per_hour < 1:
        errors.append("max_ai_invocations_per_hour must be >= 1")
    if config.max_concurrent_ai_processing < 1:
        errors.append("max_concurrent_ai_processing must be >= 1")
    return errors


class Participant(BaseModel):
    entity_id: EntityId
    conversation_id: ConversationId
    role: ParticipantRole = "member"
    lifecycle_status: EntityLifecycleStatus = "joined"
    respondent_mode: RespondentMode = "auto"
    joined_at: str
    last_active_at: str | None = None


class Conversation(BaseModel):
    id: ConversationId
    title: str
    config: ConversationConfig = ConversationConfig()
    status: ConversationStatus = "active"
    created_by: EntityId
    created_at: str
    updated_at: str
    participants: list[Participant] = []

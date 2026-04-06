"""Routing Engine -- Pure function for AI entity selection.

Determines which AI entity should respond to a sealed batch
based on mentions, respondent mode, domain matching, and
round-robin tie-breaking. No side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..debounce.types import SealedBatch
from ..types.conversation import Conversation, Participant, RespondentMode
from ..types.entity import Entity, EntityId
from ..types.event import Event, HandoffContent, MessageContent
from .types import OrchestratorConfig, RoutingDecision, RoutingRule


@dataclass
class RoutingParams:
    """Input parameters for the routing function."""

    batch: SealedBatch
    events: list[Event]
    conversation: Conversation
    participants: list[Participant]
    entities: dict[EntityId, Entity]
    config: OrchestratorConfig
    active_handoff: HandoffContent | None = None
    last_served: dict[EntityId, float] | None = None


# -- Helpers -----------------------------------------------------------------


def _is_ai(entity: Entity) -> bool:
    """Check if an entity is an AI (artificer or daemon)."""
    return entity.type in ("artificer", "daemon")


def _extract_mentions(events: list[Event]) -> list[EntityId]:
    """Extract all @mentions from the batch's events, preserving order."""
    seen: set[EntityId] = set()
    mentions: list[EntityId] = []
    for event in events:
        if event.type == "message":
            content = event.content
            if isinstance(content, MessageContent):
                for m in content.mentions:
                    if m not in seen:
                        seen.add(m)
                        mentions.append(m)
    return mentions


def _extract_batch_text(events: list[Event]) -> str:
    """Extract all text from the batch's message events."""
    parts: list[str] = []
    for event in events:
        if event.type == "message":
            content = event.content
            if isinstance(content, MessageContent):
                parts.append(content.text)
    return " ".join(parts).lower()


def _domain_match_score(domains: list[str], text: str) -> int:
    """Simple keyword-based domain match score."""
    score = 0
    for domain in domains:
        if domain.lower() in text:
            score += 1
    return score


# -- Routing Function --------------------------------------------------------


def resolve_routing(params: RoutingParams) -> RoutingDecision:
    """Resolve which AI entity should respond to a sealed batch.

    Priority order:
    1. Handoff override
    2. Explicit mention
    3. Respondent mode (mentioned_only / single_respondent / auto)
    4. Filter non-AI entities
    5. Self-reply prohibition
    """
    mode: RespondentMode = params.config.default_respondent_mode

    # Build the set of AI participants who are active/idle in this conversation
    ai_participants: list[EntityId] = []
    for p in params.participants:
        entity = params.entities.get(p.entity_id)
        if entity is None:
            continue
        if not _is_ai(entity):
            continue
        # Filter out the batch author (self-reply prohibition)
        if p.entity_id == params.batch.author_id:
            continue
        ai_participants.append(p.entity_id)

    # No AI participants at all
    if len(ai_participants) == 0:
        return RoutingDecision(
            eligible_entities=[],
            selected_entity=None,
            reason="No AI participants in conversation",
            rule_applied="no_eligible",
        )

    # 1. Handoff override
    if params.active_handoff is not None:
        target = params.active_handoff.to_entity
        if target in ai_participants:
            return RoutingDecision(
                eligible_entities=[target],
                selected_entity=target,
                reason=f"Handoff override: routed to {target} (from {params.active_handoff.from_entity})",
                rule_applied="handoff_override",
            )

    # 2. Extract mentions
    mentions = _extract_mentions(params.events)
    mentioned_ais = [m for m in mentions if m in ai_participants]

    # 3. Explicit mention -- always wins (after handoff)
    if len(mentioned_ais) > 0:
        return RoutingDecision(
            eligible_entities=mentioned_ais,
            selected_entity=mentioned_ais[0],
            reason=(
                f"Explicitly mentioned: {mentioned_ais[0]}"
                if len(mentioned_ais) == 1
                else f"Multiple AIs mentioned: {', '.join(mentioned_ais)}. First mentioned gets priority."
            ),
            rule_applied="explicit_mention",
        )

    # 4. Respondent mode
    if mode == "mentioned_only":
        return RoutingDecision(
            eligible_entities=[],
            selected_entity=None,
            reason="Mode is mentioned_only and no AI entity was mentioned",
            rule_applied="mentioned_only_mode",
        )

    if mode == "auto":
        return _resolve_auto_mode(
            ai_participants, params.events, params.entities, params.last_served
        )

    # Fallback: no dispatch
    return RoutingDecision(
        eligible_entities=ai_participants,
        selected_entity=None,
        reason=f"Unknown respondent mode: {mode}",
        rule_applied="no_eligible",
    )


# -- Auto Mode ---------------------------------------------------------------


def _resolve_auto_mode(
    ai_participants: list[EntityId],
    events: list[Event],
    entities: dict[EntityId, Entity],
    last_served: dict[EntityId, float] | None,
) -> RoutingDecision:
    """Resolve auto mode: select the best AI from all eligible participants.

    Selection criteria:
    a. Domain match (keyword)
    b. Type priority (artificer > daemon)
    c. Round-robin tie-break (least recently served)
    """
    if len(ai_participants) == 1:
        return RoutingDecision(
            eligible_entities=ai_participants,
            selected_entity=ai_participants[0],
            reason=f"Auto mode: only one AI participant ({ai_participants[0]})",
            rule_applied="auto_round_robin",
        )

    batch_text = _extract_batch_text(events)

    # Score each AI participant
    scored: list[dict[str, Any]] = []
    for entity_id in ai_participants:
        entity = entities[entity_id]
        domains = entity.capabilities.domains if entity.capabilities else []
        domain_score = _domain_match_score(domains, batch_text)
        type_priority = 1 if entity.type == "artificer" else 0
        last_served_at = (last_served or {}).get(entity_id, 0.0)

        scored.append(
            {
                "id": entity_id,
                "domain_score": domain_score,
                "type_priority": type_priority,
                "last_served_at": last_served_at,
            }
        )

    # Sort: domain score DESC, type priority DESC, last served ASC (round-robin)
    scored.sort(
        key=lambda x: (-x["domain_score"], -x["type_priority"], x["last_served_at"])
    )

    best = scored[0]
    rule: RoutingRule = (
        "auto_domain_match" if best["domain_score"] > 0 else "auto_round_robin"
    )
    reason = (
        f"Auto mode: domain match for {best['id']} (score: {best['domain_score']})"
        if best["domain_score"] > 0
        else f"Auto mode: round-robin selected {best['id']}"
    )

    return RoutingDecision(
        eligible_entities=ai_participants,
        selected_entity=best["id"],
        reason=reason,
        rule_applied=rule,
    )

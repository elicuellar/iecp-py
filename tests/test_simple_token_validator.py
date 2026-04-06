"""SimpleTokenValidator tests -- Phase 7."""

from __future__ import annotations

import pytest

from iecp_core.gateway import SimpleTokenValidator
from iecp_core.gateway.types import AuthToken
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId

# -- Fixtures ----------------------------------------------------------------

TOKEN_1 = AuthToken(
    entity_id=EntityId("entity-1"),
    type="human",
    conversation_ids=[ConversationId("conv-1")],
)


# -- Tests -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validates_a_known_token() -> None:
    validator = SimpleTokenValidator()
    validator.add_token("sk_valid", TOKEN_1)

    result = await validator.validate("sk_valid")
    assert result == TOKEN_1


@pytest.mark.asyncio
async def test_returns_none_for_unknown_token() -> None:
    validator = SimpleTokenValidator()
    result = await validator.validate("sk_unknown")
    assert result is None


@pytest.mark.asyncio
async def test_supports_initial_tokens_via_constructor() -> None:
    initial = {"sk_init": TOKEN_1}
    validator = SimpleTokenValidator(initial)

    result = await validator.validate("sk_init")
    assert result == TOKEN_1


@pytest.mark.asyncio
async def test_add_token_registers_new_token() -> None:
    validator = SimpleTokenValidator()
    assert await validator.validate("sk_new") is None

    validator.add_token("sk_new", TOKEN_1)
    assert await validator.validate("sk_new") == TOKEN_1


@pytest.mark.asyncio
async def test_remove_token_revokes_a_token() -> None:
    validator = SimpleTokenValidator()
    validator.add_token("sk_revoke", TOKEN_1)
    assert await validator.validate("sk_revoke") == TOKEN_1

    validator.remove_token("sk_revoke")
    assert await validator.validate("sk_revoke") is None

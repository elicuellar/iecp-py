"""FastAPI Application -- Phase 10 of the IECP protocol.

Creates the FastAPI app and registers all routes + middleware.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..artificer.artificer_runtime import ArtificerRuntime
from ..conversations.conversation_manager import ConversationManager
from ..cursors.cursor_manager import CursorManager
from ..decisions.decision_manager import DecisionManager
from ..entities.entity_manager import EntityManager
from ..events.event_store import EventStore
from ..gateway.simple_token_validator import SimpleTokenValidator
from ..handoff.handoff_manager import HandoffManager
from ..lock.floor_lock import FloorLock
from ..signals.attention_signal_manager import AttentionSignalManager
from ..types.entity import EntityId
from .errors import ConflictError, NotFoundError, ValidationError
from .routes.auth import create_auth_router
from .routes.artificers import ArtificerRegistration, create_artificer_router
from .routes.conversations import create_conversation_router
from .routes.cursors import create_cursor_router
from .routes.decisions import create_decision_router
from .routes.entities import create_entity_router
from .routes.events import create_event_router
from .routes.handoffs import create_handoff_router
from .routes.health import create_health_router, create_status_router
from .routes.lock import create_lock_router
from .routes.signals import create_signal_router


# --- Services ---------------------------------------------------------------


class AppServices:
    def __init__(
        self,
        *,
        event_store: EventStore,
        entity_manager: EntityManager,
        entity_repo: Any,
        conversation_manager: ConversationManager,
        cursor_manager: CursorManager,
        orchestrator: Any | None,
        floor_lock: FloorLock,
        signal_manager: AttentionSignalManager,
        decision_manager: DecisionManager,
        handoff_manager: HandoffManager,
        gateway: Any | None,
        token_validator: SimpleTokenValidator,
        artificer_registry: dict[EntityId, ArtificerRegistration],
        metrics_collector: Any | None = None,
        trace_logger: Any | None = None,
        artificer_runtime: ArtificerRuntime | None = None,
    ) -> None:
        self.event_store = event_store
        self.entity_manager = entity_manager
        self.entity_repo = entity_repo
        self.conversation_manager = conversation_manager
        self.cursor_manager = cursor_manager
        self.orchestrator = orchestrator
        self.floor_lock = floor_lock
        self.signal_manager = signal_manager
        self.decision_manager = decision_manager
        self.handoff_manager = handoff_manager
        self.gateway = gateway
        self.token_validator = token_validator
        self.artificer_registry = artificer_registry
        self.metrics_collector = metrics_collector
        self.trace_logger = trace_logger
        self.artificer_runtime = artificer_runtime


# --- API Key Store ----------------------------------------------------------


class ApiKeyStore:
    def __init__(self, admin_api_key: str | None = None) -> None:
        self._keys: dict[str, dict[str, Any]] = {}
        if admin_api_key:
            self._keys[admin_api_key] = {
                "entity_id": None,
                "permissions": {"admin"},
            }

    def add_key(self, key: str, entity_id: Any, permissions: set[str]) -> None:
        self._keys[key] = {"entity_id": entity_id, "permissions": permissions}

    def remove_key(self, key: str) -> None:
        self._keys.pop(key, None)

    def get_entry(self, key: str) -> dict[str, Any] | None:
        return self._keys.get(key)


# --- Auth Middleware Helper --------------------------------------------------


def _require_auth(request: Request, key_store: ApiKeyStore) -> dict[str, Any]:
    """Validate Bearer token and return auth entry."""
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Missing Authorization header"},
        )

    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer":
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid Authorization format. Expected: Bearer <token>",
            },
        )

    token = parts[1]
    entry = key_store.get_entry(token)
    if not entry:
        raise HTTPException(
            status_code=403,
            detail={"code": "FORBIDDEN", "message": "Invalid API key"},
        )

    return entry


# --- App Factory ------------------------------------------------------------


def create_app(services: AppServices, admin_api_key: str) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="IECP Server", version="1.0.0-rc1")

    key_store = ApiKeyStore(admin_api_key)
    start_time = time.time() * 1000

    # --- Global exception handlers ------------------------------------------

    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "NOT_FOUND", "message": exc.message}},
        )

    @app.exception_handler(ValidationError)
    async def validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
        body: dict[str, Any] = {
            "error": {"code": "VALIDATION_ERROR", "message": exc.message}
        }
        if exc.details is not None:
            body["error"]["details"] = exc.details  # type: ignore[index]
        return JSONResponse(status_code=400, content=body)

    @app.exception_handler(ConflictError)
    async def conflict_handler(request: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"error": {"code": "CONFLICT", "message": exc.message}},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "ERROR", "message": str(exc.detail)}},
        )

    # --- Health (no auth required) ------------------------------------------
    health_router = create_health_router(services, start_time)
    app.include_router(health_router)

    # --- Auth-protected routes ----------------------------------------------
    status_router = create_status_router(services, key_store)
    app.include_router(status_router)

    entity_router = create_entity_router(services, key_store)
    app.include_router(entity_router, prefix="/api/v1/entities")

    conversation_router = create_conversation_router(services, key_store)
    app.include_router(conversation_router, prefix="/api/v1/conversations")

    event_router = create_event_router(services, key_store)
    app.include_router(event_router, prefix="/api/v1/conversations/{conv_id}/events")

    cursor_router = create_cursor_router(services, key_store)
    app.include_router(cursor_router, prefix="/api/v1/conversations/{conv_id}/cursors")

    lock_router = create_lock_router(services, key_store)
    app.include_router(lock_router, prefix="/api/v1/conversations/{conv_id}/lock")

    signal_router = create_signal_router(services, key_store)
    app.include_router(signal_router, prefix="/api/v1/conversations/{conv_id}/signals")

    decision_router = create_decision_router(services, key_store)
    app.include_router(decision_router, prefix="/api/v1/conversations/{conv_id}/decisions")

    handoff_router = create_handoff_router(services, key_store)
    app.include_router(handoff_router, prefix="/api/v1/conversations/{conv_id}/handoffs")

    artificer_router = create_artificer_router(services, key_store)
    app.include_router(artificer_router, prefix="/api/v1/artificers")

    auth_router = create_auth_router(services, key_store)
    app.include_router(auth_router, prefix="/api/v1/auth")

    # Metrics routes (Phase 10 observability)
    if services.metrics_collector is not None and services.trace_logger is not None:
        from .routes.metrics import create_metrics_router
        metrics_router = create_metrics_router(services, key_store)
        app.include_router(metrics_router, prefix="/api/v1/metrics")

    return app

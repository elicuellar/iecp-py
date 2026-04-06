"""Metrics API Routes -- Phase 10 of the IECP protocol (§17).

Exposes observability data via REST endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...types.entity import EntityId
from ...types.event import ConversationId
from ..errors import NotFoundError


def create_metrics_router(services: Any, key_store: Any) -> APIRouter:
    router = APIRouter()

    @router.get("")
    async def get_system_metrics(request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        metrics = services.metrics_collector.get_system_metrics()
        return JSONResponse(content=metrics.model_dump())

    @router.get("/conversations/{conv_id}")
    async def get_conversation_metrics(conv_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        metrics = services.metrics_collector.get_conversation_metrics(ConversationId(conv_id))
        return JSONResponse(content=metrics.model_dump())

    @router.get("/entities/{entity_id}")
    async def get_entity_metrics(entity_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        metrics = services.metrics_collector.get_entity_metrics(EntityId(entity_id))
        return JSONResponse(content=metrics.model_dump())

    @router.get("/traces")
    async def get_traces(request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        params = request.query_params

        filter_kwargs: dict[str, Any] = {}
        if params.get("conversationId"):
            filter_kwargs["conversation_id"] = ConversationId(params["conversationId"])
        if params.get("entityId"):
            filter_kwargs["entity_id"] = EntityId(params["entityId"])
        if params.get("outcome"):
            filter_kwargs["outcome"] = params["outcome"]
        if params.get("since"):
            filter_kwargs["since"] = float(params["since"])
        if params.get("limit"):
            filter_kwargs["limit"] = int(params["limit"])

        from ...observability.trace_logger import TraceQuery
        query = TraceQuery(**filter_kwargs)
        traces = services.trace_logger.query(query)
        stats = services.trace_logger.get_stats()

        return JSONResponse(content={
            "traces": [t.model_dump() for t in traces],
            "stats": stats.model_dump(),
        })

    @router.get("/traces/{trace_id}")
    async def get_trace(trace_id: str, request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        trace = services.trace_logger.get(trace_id)
        if not trace:
            raise NotFoundError(f"Trace not found: {trace_id}")

        return JSONResponse(content=trace.model_dump())

    return router

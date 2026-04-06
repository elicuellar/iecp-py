"""Health & Status Routes -- Phase 10 of the IECP protocol."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


def create_health_router(services: Any, start_time: float) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health(request: Request) -> JSONResponse:
        uptime = time.time() * 1000 - start_time

        gateway_status = "ok"
        database_status = "ok"

        queue_active = 0
        queue_queued = 0
        artificer_queue_status = "ok"

        if services.artificer_runtime is not None:
            stats = services.artificer_runtime.get_queue_stats()
            queue_active = stats.get("active", 0)
            queue_queued = stats.get("queued", 0)
            if queue_queued > 10:
                artificer_queue_status = "saturated"

        overall_status = "degraded" if artificer_queue_status == "saturated" else "ok"

        return JSONResponse(
            content={
                "status": overall_status,
                "uptime": uptime,
                "checks": {
                    "database": database_status,
                    "gateway": gateway_status,
                    "artificerQueue": {
                        "active": queue_active,
                        "queued": queue_queued,
                        "status": artificer_queue_status,
                    },
                },
                "version": "1.0.0-rc1",
            }
        )

    return router


def create_status_router(services: Any, key_store: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/api/v1/status")
    async def status(request: Request) -> JSONResponse:
        from ..app import _require_auth
        _require_auth(request, key_store)

        return JSONResponse(
            content={
                "connections": 0,
                "conversations": 0,
                "artificers": len(services.artificer_registry),
                "queue": {"active": 0, "queued": 0},
            }
        )

    return router

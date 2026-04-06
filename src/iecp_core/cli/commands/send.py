"""'iecp send' -- one-shot message send.

Connects, authenticates, fetches unread batch, acquires lock,
sends message, commits, and disconnects.

Mirrors packages/cli/src/commands/send.ts exactly.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable

from ..ws.websocket_client import WebSocketClient, WebSocketClientConfig


@dataclass
class SendOptions:
    """Options for the send command."""

    server: str
    token: str
    room: str
    stdin: bool = False


def execute_send(
    text: str,
    opts: SendOptions,
    ws_factory: Callable[[str], Any] | None = None,
) -> None:
    """Execute the send logic.

    Exported for testing. The ws_factory parameter allows test injection.
    """
    config = WebSocketClientConfig(
        server_url=opts.server,
        token=opts.token,
        reconnect=False,
    )
    client = WebSocketClient(config, ws_factory=ws_factory)

    conversation_id = opts.room

    # ── Helper: set up a one-shot listener and return its event ──────────────

    def make_waiter(msg_type: str) -> tuple[threading.Event, list[dict[str, Any]]]:
        """Register a listener before sending, return (event, result_list)."""
        ev = threading.Event()
        result: list[dict[str, Any]] = []

        def handler(msg: dict[str, Any]) -> None:
            mt = msg.get("type")
            if mt == msg_type or (mt == "error" and "request_id" in msg):
                client.remove_listener("message", handler)
                result.append(msg)
                ev.set()

        client.on("message", handler)
        return ev, result

    # ── Connect ───────────────────────────────────────────────────────────────

    connected_ev = threading.Event()
    client.on("connected", lambda: connected_ev.set())
    client.connect()
    connected_ev.wait()

    # ── Authenticate ──────────────────────────────────────────────────────────

    auth_ev, auth_result = make_waiter("authenticated")
    client.send({
        "type": "authenticate",
        "token": opts.token,
        "conversation_id": conversation_id,
    })
    auth_ev.wait()
    auth_msg = auth_result[0]
    if auth_msg.get("type") != "authenticated":
        raise RuntimeError("Authentication failed")
    eid = str(auth_msg["entity_id"])

    # ── Fetch unread (required before lock) ───────────────────────────────────

    fetch_ev, _ = make_waiter("unread_batch")
    client.send({
        "type": "fetch_unread_batch",
        "request_id": "send_fetch",
        "conversation_id": conversation_id,
        "entity_id": eid,
    })
    fetch_ev.wait()

    # ── Acquire lock ──────────────────────────────────────────────────────────

    lock_ev, lock_result = make_waiter("lock_acquired")
    client.send({
        "type": "acquire_speaking_lock",
        "request_id": "send_lock",
        "conversation_id": conversation_id,
        "entity_id": eid,
        "estimated_ms": 5000,
    })
    lock_ev.wait()
    lock_msg = lock_result[0]
    if lock_msg.get("type") == "error":
        raise RuntimeError("Failed to acquire lock")

    # ── Send text as stream chunk ─────────────────────────────────────────────

    chunk_ev, _ = make_waiter("chunk_ack")
    client.send({
        "type": "append_stream_chunk",
        "request_id": "send_chunk",
        "conversation_id": conversation_id,
        "entity_id": eid,
        "text": text,
    })
    chunk_ev.wait()

    # ── Commit ────────────────────────────────────────────────────────────────

    commit_ev, commit_result = make_waiter("commit_response")
    client.send({
        "type": "commit_message",
        "request_id": "send_commit",
        "conversation_id": conversation_id,
        "entity_id": eid,
    })
    commit_ev.wait()
    commit_msg = commit_result[0]
    if commit_msg.get("type") == "commit_response":
        sys.stdout.write(f"Sent: {commit_msg['event_id']}\n")

    # ── Disconnect ────────────────────────────────────────────────────────────

    client.disconnect()

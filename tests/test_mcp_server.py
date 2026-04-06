"""McpServer tests -- Phase 8.

Uses mock WS client and piped stdin/stdout streams.
Mirrors packages/cli/tests/mcp/mcp-server.test.ts exactly.
"""

from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from iecp_core.cli.mcp.mcp_server import McpServer, McpServerDeps
from iecp_core.cli.mcp.tools import IECP_TOOLS
from iecp_core.cli.ws.websocket_client import WebSocketClient, WebSocketClientConfig
from iecp_core.types.entity import EntityId
from iecp_core.types.event import ConversationId


# ─── Mock WS Client ────────────────────────────────────────────────────────────


class MockWsClient:
    """Mock WebSocket client that records sent messages and allows response injection."""

    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []
        self._listeners: dict[str, list[Callable[..., Any]]] = {}

    def on(self, event: str, listener: Callable[..., Any]) -> None:
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)

    def off(self, event: str, listener: Callable[..., Any]) -> None:
        listeners = self._listeners.get(event)
        if listeners and listener in listeners:
            listeners.remove(listener)

    def emit(self, event: str, *args: Any) -> None:
        for listener in list(self._listeners.get(event, [])):
            listener(*args)

    def send(self, msg: Any) -> None:
        if isinstance(msg, dict):
            self.sent_messages.append(msg)
        else:
            self.sent_messages.append(msg)

    def is_connected(self) -> bool:
        return True

    def simulate_response(self, msg: dict[str, Any]) -> None:
        """Emit a server response to all message listeners."""
        self.emit("message", msg)


# ─── Piped I/O helpers ─────────────────────────────────────────────────────────


class PipeIO:
    """A simple in-memory pipe: writer pushes lines, reader blocks until data arrives."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data = io.StringIO()
        self._has_data = threading.Event()
        self._lines: list[str] = []
        self._closed = False

    # -- Writer interface (like PassThrough.write) ─────────────────────────────

    def write(self, data: str) -> None:
        with self._lock:
            self._lines.append(data)
        self._has_data.set()

    def flush(self) -> None:
        pass

    # -- Reader interface (iterable, line-by-line) ─────────────────────────────

    def __iter__(self) -> "PipeIO":
        return self

    def __next__(self) -> str:
        while True:
            with self._lock:
                if self._lines:
                    return self._lines.pop(0)
                if self._closed:
                    raise StopIteration
            self._has_data.wait(timeout=0.1)
            self._has_data.clear()

    def close(self) -> None:
        self._closed = True
        self._has_data.set()

    def read_line(self, timeout: float = 2.0) -> str | None:
        """Read one written line, waiting up to timeout seconds."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._lines:
                    return self._lines.pop(0)
            time.sleep(0.005)
        return None


# ─── Helper functions ──────────────────────────────────────────────────────────


def create_test_server() -> tuple[McpServer, MockWsClient, PipeIO, PipeIO]:
    """Create a McpServer with mock deps."""
    ws_client = MockWsClient()
    input_pipe = PipeIO()
    output_pipe = PipeIO()

    deps = McpServerDeps(
        ws_client=ws_client,  # type: ignore[arg-type]
        entity_id=EntityId("ent_test"),
        conversation_id=ConversationId("conv_test"),
        input=input_pipe,  # type: ignore[arg-type]
        output=output_pipe,  # type: ignore[arg-type]
        request_timeout=5000,
    )
    server = McpServer(deps)
    return server, ws_client, input_pipe, output_pipe


def send_request(
    input_pipe: PipeIO,
    req_id: int | str,
    method: str,
    params: dict[str, Any] | None = None,
) -> None:
    """Write a JSON-RPC request to the input pipe."""
    req: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        req["params"] = params
    input_pipe.write(json.dumps(req) + "\n")


def read_response(output_pipe: PipeIO, timeout: float = 2.0) -> dict[str, Any]:
    """Read one JSON-RPC response line from the output pipe."""
    line = output_pipe.read_line(timeout=timeout)
    assert line is not None, "Timed out waiting for response"
    return json.loads(line.strip())


# ─── Tests ─────────────────────────────────────────────────────────────────────


class TestMcpServer:
    def setup_method(self) -> None:
        server, ws_client, input_pipe, output_pipe = create_test_server()
        self.server = server
        self.ws_client = ws_client
        self.input_pipe = input_pipe
        self.output_pipe = output_pipe
        self.server.start()

    def teardown_method(self) -> None:
        self.server.stop()
        self.input_pipe.close()

    def test_initialize_returns_server_info(self) -> None:
        send_request(self.input_pipe, 1, "initialize")
        resp = read_response(self.output_pipe)

        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        result = resp["result"]
        assert result["protocolVersion"] == "2024-11-05"
        info = result["serverInfo"]
        assert info["name"] == "iecp-mcp-server"

    def test_tools_list_returns_all_11_tools(self) -> None:
        send_request(self.input_pipe, 2, "tools/list")
        resp = read_response(self.output_pipe)

        result = resp["result"]
        tools = result["tools"]
        assert len(tools) == len(IECP_TOOLS)
        assert len(tools) == 11

        tool_names = [t["name"] for t in tools]
        assert "get_room_status" in tool_names
        assert "fetch_unread_batch" in tool_names
        assert "acquire_speaking_lock" in tool_names
        assert "append_stream_chunk" in tool_names
        assert "commit_message" in tool_names
        assert "yield_floor" in tool_names
        assert "report_action" in tool_names
        assert "signal_attention" in tool_names
        assert "propose_decision" in tool_names
        assert "handoff_to" in tool_names
        assert "fetch_history" in tool_names

    def test_tools_call_get_room_status_sends_ws_request_and_returns_response(self) -> None:
        send_request(self.input_pipe, 3, "tools/call", {
            "name": "get_room_status",
            "arguments": {},
        })

        # Wait for WS message to be sent
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self.ws_client.sent_messages:
                break
            time.sleep(0.005)

        assert len(self.ws_client.sent_messages) == 1
        ws_sent = self.ws_client.sent_messages[0]
        assert ws_sent["type"] == "get_room_status"

        # Simulate server response
        self.ws_client.simulate_response({
            "type": "room_status",
            "request_id": ws_sent["request_id"],
            "conversation_id": "conv_test",
            "lock_holder": None,
            "participants": [],
            "ai_depth_counter": 0,
            "your_status": "active",
        })

        resp = read_response(self.output_pipe)
        result = resp["result"]
        content = result["content"][0]
        assert content["type"] == "text"
        parsed = json.loads(content["text"])
        assert parsed["conversation_id"] == "conv_test"
        assert parsed["lock_holder"] is None

    def test_tools_call_acquire_speaking_lock_enforces_read_before_write(self) -> None:
        send_request(self.input_pipe, 4, "tools/call", {
            "name": "acquire_speaking_lock",
            "arguments": {"estimated_ms": 5000},
        })
        resp = read_response(self.output_pipe)
        result = resp["result"]
        content = result["content"][0]
        assert result["isError"] is True
        assert "fetch_unread_batch" in content["text"]

    def test_tools_call_append_stream_chunk_enforces_lock_precondition(self) -> None:
        send_request(self.input_pipe, 5, "tools/call", {
            "name": "append_stream_chunk",
            "arguments": {"text": "hello"},
        })
        resp = read_response(self.output_pipe)
        result = resp["result"]
        assert result["isError"] is True
        content = result["content"][0]
        assert "Floor Lock" in content["text"]

    def test_tools_call_commit_message_enforces_lock_precondition(self) -> None:
        send_request(self.input_pipe, 6, "tools/call", {
            "name": "commit_message",
            "arguments": {},
        })
        resp = read_response(self.output_pipe)
        result = resp["result"]
        assert result["isError"] is True
        content = result["content"][0]
        assert "Floor Lock" in content["text"]

    def test_tools_call_signal_attention_sends_signal_without_lock(self) -> None:
        send_request(self.input_pipe, 7, "tools/call", {
            "name": "signal_attention",
            "arguments": {"signal": "thinking"},
        })

        # Wait for WS message
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self.ws_client.sent_messages:
                break
            time.sleep(0.005)

        assert len(self.ws_client.sent_messages) == 1
        ws_sent = self.ws_client.sent_messages[0]
        assert ws_sent["type"] == "signal_attention"
        assert ws_sent["signal"] == "thinking"

        # Simulate response
        self.ws_client.simulate_response({
            "type": "signal_response",
            "request_id": ws_sent["request_id"],
            "ok": True,
        })

        resp = read_response(self.output_pipe)
        result = resp["result"]
        assert result.get("isError") is None

    def test_tools_call_yield_floor_sends_yield_message(self) -> None:
        send_request(self.input_pipe, 8, "tools/call", {
            "name": "yield_floor",
            "arguments": {},
        })

        # Wait for WS message
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self.ws_client.sent_messages:
                break
            time.sleep(0.005)

        assert len(self.ws_client.sent_messages) == 1
        ws_sent = self.ws_client.sent_messages[0]
        assert ws_sent["type"] == "yield_floor"

        # Simulate response
        self.ws_client.simulate_response({
            "type": "yield_response",
            "request_id": ws_sent["request_id"],
            "ok": True,
        })

        resp = read_response(self.output_pipe)
        result = resp["result"]
        assert result.get("isError") is None

    def test_tools_call_propose_decision_sends_decision_without_lock(self) -> None:
        send_request(self.input_pipe, 9, "tools/call", {
            "name": "propose_decision",
            "arguments": {"summary": "Use WebGL for rendering"},
        })

        # Wait for WS message
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self.ws_client.sent_messages:
                break
            time.sleep(0.005)

        assert len(self.ws_client.sent_messages) == 1
        ws_sent = self.ws_client.sent_messages[0]
        assert ws_sent["type"] == "propose_decision"
        assert ws_sent["summary"] == "Use WebGL for rendering"

        # Simulate response
        self.ws_client.simulate_response({
            "type": "decision_response",
            "request_id": ws_sent["request_id"],
            "event_id": "evt_123",
            "decision_status": "proposed",
        })

        resp = read_response(self.output_pipe)
        result = resp["result"]
        assert result.get("isError") is None

    def test_tools_call_handoff_to_sends_handoff_without_lock(self) -> None:
        send_request(self.input_pipe, 10, "tools/call", {
            "name": "handoff_to",
            "arguments": {
                "to_entity": "ent_other",
                "reason": "Architecture question",
                "context_summary": "Discussing rendering pipeline",
            },
        })

        # Wait for WS message
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self.ws_client.sent_messages:
                break
            time.sleep(0.005)

        assert len(self.ws_client.sent_messages) == 1
        ws_sent = self.ws_client.sent_messages[0]
        assert ws_sent["type"] == "handoff_to"
        assert ws_sent["to_entity"] == "ent_other"

        # Simulate response
        self.ws_client.simulate_response({
            "type": "handoff_response",
            "request_id": ws_sent["request_id"],
            "event_id": "evt_456",
        })

        resp = read_response(self.output_pipe)
        result = resp["result"]
        assert result.get("isError") is None

    def test_returns_error_for_unknown_method(self) -> None:
        send_request(self.input_pipe, 11, "unknown/method")
        resp = read_response(self.output_pipe)
        error = resp["error"]
        assert error["code"] == -32603
        assert "Unknown method" in error["message"]

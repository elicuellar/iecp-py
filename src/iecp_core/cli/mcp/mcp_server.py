"""MCP Server -- minimal JSON-RPC over stdio that exposes IECP tools.

Implements only the subset needed for V1:
- initialize -> server info + capabilities
- tools/list -> IECP tool definitions
- tools/call -> execute tool via WebSocket

Mirrors packages/cli/src/mcp/McpServer.ts exactly.
"""

from __future__ import annotations

import dataclasses
import io
import json
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, IO

from ...types.entity import EntityId
from ...types.event import ConversationId, EventId
from ..ws.websocket_client import WebSocketClient
from .stream_accumulator import StreamAccumulator
from .tools import IECP_TOOLS
from .types import (
    JsonRpcErrorResponse,
    JsonRpcRequest,
    JsonRpcSuccessResponse,
    McpCapabilities,
    McpInitializeResult,
    McpServerInfo,
    McpToolCallParams,
    McpToolResult,
)


@dataclass
class _PendingRequest:
    """An in-flight WS request waiting for a server response."""

    resolve: Callable[[dict[str, Any]], None]
    reject: Callable[[Exception], None]
    timer: threading.Timer


@dataclass
class McpServerDeps:
    """Dependencies for McpServer."""

    ws_client: WebSocketClient
    entity_id: EntityId
    conversation_id: ConversationId
    input: IO[str] | None = None
    """Readable text stream (default: sys.stdin)."""
    output: IO[str] | None = None
    """Writable text stream (default: sys.stdout)."""
    request_timeout: float = 30_000
    """Request timeout in ms (default: 30000)."""


class McpServer:
    """Minimal JSON-RPC MCP server over stdio that exposes IECP tools."""

    def __init__(self, deps: McpServerDeps) -> None:
        self._ws_client = deps.ws_client
        self._entity_id = deps.entity_id
        self._conversation_id = deps.conversation_id
        self._input: IO[str] = deps.input or sys.stdin
        self._output: IO[str] = deps.output or sys.stdout
        self._request_timeout = deps.request_timeout

        self._accumulator = StreamAccumulator()
        self._holds_lock = False
        self._has_read_since_last_commit = False
        self._pending_requests: dict[str, _PendingRequest] = {}
        self._request_counter = 0

        self._reader_thread: threading.Thread | None = None
        self._running = False

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin reading JSON-RPC requests from input."""
        self._running = True

        # Wire WS responses to our handler
        self._ws_client.on("message", self._handle_ws_response)

        # Start reader thread (reads lines from input)
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def stop(self) -> None:
        """Stop reading from input."""
        self._running = False

        # Cancel all pending requests
        for pending in list(self._pending_requests.values()):
            pending.timer.cancel()
            pending.reject(Exception("MCP server stopped"))
        self._pending_requests.clear()

    # ─── Reader Loop ──────────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """Read lines from input and dispatch them."""
        try:
            for line in self._input:
                if not self._running:
                    break
                self._handle_line(line)
        except Exception:
            pass  # Input closed

    def _handle_line(self, line: str) -> None:
        """Process a single JSON-RPC line."""
        trimmed = line.strip()
        if not trimmed:
            return

        try:
            raw = json.loads(trimmed)
            request = JsonRpcRequest(
                jsonrpc=raw.get("jsonrpc", "2.0"),
                id=raw.get("id", 0),
                method=raw.get("method", ""),
                params=raw.get("params"),
            )
        except (json.JSONDecodeError, Exception):
            self._write_response(
                JsonRpcErrorResponse(
                    jsonrpc="2.0",
                    id=0,
                    error={"code": -32700, "message": "Parse error"},
                )
            )
            return

        try:
            result = self._handle_request(request)
            self._write_response(
                JsonRpcSuccessResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    result=result,
                )
            )
        except Exception as err:
            self._write_response(
                JsonRpcErrorResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error={
                        "code": -32603,
                        "message": str(err),
                    },
                )
            )

    # ─── Request Routing ─────────────────────────────────────────────────────

    def _handle_request(self, request: JsonRpcRequest) -> Any:
        """Route a JSON-RPC request to the appropriate handler."""
        if request.method == "initialize":
            return self._handle_initialize()
        elif request.method == "tools/list":
            return self._handle_tools_list()
        elif request.method == "tools/call":
            params_raw = request.params or {}
            params = McpToolCallParams(
                name=str(params_raw.get("name", "")),
                arguments=dict(params_raw.get("arguments", {})),
            )
            return self._handle_tool_call(params)
        else:
            raise Exception(f"Unknown method: {request.method}")

    def _handle_initialize(self) -> dict[str, Any]:
        """Handle initialize request."""
        result = McpInitializeResult(
            protocolVersion="2024-11-05",
            serverInfo=McpServerInfo(name="iecp-mcp-server", version="0.1.0"),
            capabilities=McpCapabilities(tools={}),
        )
        return dataclasses.asdict(result)

    def _handle_tools_list(self) -> dict[str, Any]:
        """Handle tools/list request."""
        return {
            "tools": [dataclasses.asdict(tool) for tool in IECP_TOOLS]
        }

    def _handle_tool_call(self, params: McpToolCallParams) -> dict[str, Any]:
        """Handle tools/call request."""
        try:
            result = self._execute_tool(params.name, params.arguments)
            return dataclasses.asdict(
                McpToolResult(content=[{"type": "text", "text": json.dumps(result)}])
            )
        except Exception as err:
            return dataclasses.asdict(
                McpToolResult(
                    content=[{"type": "text", "text": str(err)}],
                    isError=True,
                )
            )

    # ─── Tool Dispatch ────────────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Execute a single IECP tool synchronously."""
        if name == "get_room_status":
            return self._tool_get_room_status()
        elif name == "fetch_unread_batch":
            return self._tool_fetch_unread_batch()
        elif name == "fetch_history":
            return self._tool_fetch_history(args)
        elif name == "acquire_speaking_lock":
            return self._tool_acquire_speaking_lock(args)
        elif name == "append_stream_chunk":
            return self._tool_append_stream_chunk(args)
        elif name == "commit_message":
            return self._tool_commit_message(args)
        elif name == "yield_floor":
            return self._tool_yield_floor()
        elif name == "report_action":
            return self._tool_report_action(args)
        elif name == "signal_attention":
            return self._tool_signal_attention(args)
        elif name == "propose_decision":
            return self._tool_propose_decision(args)
        elif name == "handoff_to":
            return self._tool_handoff_to(args)
        else:
            raise Exception(f"Unknown tool: {name}")

    # ─── Tool Implementations ─────────────────────────────────────────────────

    def _tool_get_room_status(self) -> Any:
        req_id = self._next_request_id()
        self._ws_client.send({
            "type": "get_room_status",
            "request_id": req_id,
            "conversation_id": self._conversation_id,
        })
        resp = self._wait_for_response(req_id)
        return {
            "conversation_id": resp.get("conversation_id"),
            "lock_holder": resp.get("lock_holder"),
            "participants": resp.get("participants"),
            "ai_depth_counter": resp.get("ai_depth_counter"),
            "your_status": resp.get("your_status"),
        }

    def _tool_fetch_unread_batch(self) -> Any:
        req_id = self._next_request_id()
        self._ws_client.send({
            "type": "fetch_unread_batch",
            "request_id": req_id,
            "conversation_id": self._conversation_id,
            "entity_id": self._entity_id,
        })
        resp = self._wait_for_response(req_id)
        self._has_read_since_last_commit = True
        return resp.get("payload")

    def _tool_fetch_history(self, args: dict[str, Any]) -> Any:
        req_id = self._next_request_id()
        msg: dict[str, Any] = {
            "type": "fetch_history",
            "request_id": req_id,
            "conversation_id": self._conversation_id,
            "limit": args.get("limit", 20),
        }
        if "before_id" in args:
            msg["before_id"] = args["before_id"]
        self._ws_client.send(msg)
        resp = self._wait_for_response(req_id)
        return {"messages": resp.get("messages"), "has_more": resp.get("has_more")}

    def _tool_acquire_speaking_lock(self, args: dict[str, Any]) -> Any:
        if not self._has_read_since_last_commit:
            raise Exception(
                "Pre-condition failed: must call fetch_unread_batch before acquiring lock"
            )
        req_id = self._next_request_id()
        msg: dict[str, Any] = {
            "type": "acquire_speaking_lock",
            "request_id": req_id,
            "conversation_id": self._conversation_id,
            "entity_id": self._entity_id,
            "estimated_ms": args["estimated_ms"],
        }
        if "intent_summary" in args:
            msg["intent_summary"] = args["intent_summary"]
        self._ws_client.send(msg)
        resp = self._wait_for_response(req_id)
        if resp.get("granted"):
            self._holds_lock = True
            self._accumulator.clear()
        return {
            "granted": resp.get("granted"),
            "reason": resp.get("reason"),
            "ttl_ms": resp.get("ttl_ms"),
        }

    def _tool_append_stream_chunk(self, args: dict[str, Any]) -> Any:
        if not self._holds_lock:
            raise Exception("Pre-condition failed: must hold Floor Lock to stream chunks")
        text = args["text"]
        self._accumulator.append(text)

        req_id = self._next_request_id()
        self._ws_client.send({
            "type": "append_stream_chunk",
            "request_id": req_id,
            "conversation_id": self._conversation_id,
            "entity_id": self._entity_id,
            "text": text,
        })
        resp = self._wait_for_response(req_id)
        return {"ok": resp.get("ok")}

    def _tool_commit_message(self, args: dict[str, Any]) -> Any:
        if not self._holds_lock:
            raise Exception("Pre-condition failed: must hold Floor Lock to commit")
        req_id = self._next_request_id()
        msg: dict[str, Any] = {
            "type": "commit_message",
            "request_id": req_id,
            "conversation_id": self._conversation_id,
            "entity_id": self._entity_id,
        }
        if "mentions" in args:
            msg["mentions"] = args["mentions"]
        if "metadata" in args:
            msg["metadata"] = args["metadata"]
        self._ws_client.send(msg)
        resp = self._wait_for_response(req_id)
        self._holds_lock = False
        self._has_read_since_last_commit = False
        self._accumulator.clear()
        return {"event_id": resp.get("event_id"), "created_at": resp.get("created_at")}

    def _tool_yield_floor(self) -> Any:
        req_id = self._next_request_id()
        self._ws_client.send({
            "type": "yield_floor",
            "request_id": req_id,
            "conversation_id": self._conversation_id,
            "entity_id": self._entity_id,
        })
        resp = self._wait_for_response(req_id)
        self._holds_lock = False
        self._has_read_since_last_commit = False
        self._accumulator.clear()
        return {"ok": resp.get("ok")}

    def _tool_report_action(self, args: dict[str, Any]) -> Any:
        req_id = self._next_request_id()
        msg: dict[str, Any] = {
            "type": "report_action",
            "request_id": req_id,
            "conversation_id": self._conversation_id,
            "entity_id": self._entity_id,
            "action_type": args["action_type"],
            "description": args["description"],
            "status": args["status"],
        }
        if "result" in args:
            msg["result"] = args["result"]
        self._ws_client.send(msg)
        resp = self._wait_for_response(req_id)
        return {"event_id": resp.get("event_id")}

    def _tool_signal_attention(self, args: dict[str, Any]) -> Any:
        req_id = self._next_request_id()
        msg: dict[str, Any] = {
            "type": "signal_attention",
            "request_id": req_id,
            "conversation_id": self._conversation_id,
            "entity_id": self._entity_id,
            "signal": args["signal"],
        }
        if "utterance_ref" in args:
            msg["utterance_ref"] = args["utterance_ref"]
        if "note" in args:
            msg["note"] = args["note"]
        self._ws_client.send(msg)
        resp = self._wait_for_response(req_id)
        return {"ok": resp.get("ok")}

    def _tool_propose_decision(self, args: dict[str, Any]) -> Any:
        req_id = self._next_request_id()
        msg: dict[str, Any] = {
            "type": "propose_decision",
            "request_id": req_id,
            "conversation_id": self._conversation_id,
            "entity_id": self._entity_id,
            "summary": args["summary"],
        }
        if "context_events" in args:
            msg["context_events"] = args["context_events"]
        self._ws_client.send(msg)
        resp = self._wait_for_response(req_id)
        return {
            "event_id": resp.get("event_id"),
            "decision_status": resp.get("decision_status"),
        }

    def _tool_handoff_to(self, args: dict[str, Any]) -> Any:
        req_id = self._next_request_id()
        msg: dict[str, Any] = {
            "type": "handoff_to",
            "request_id": req_id,
            "conversation_id": self._conversation_id,
            "entity_id": self._entity_id,
            "to_entity": args["to_entity"],
            "reason": args["reason"],
            "context_summary": args["context_summary"],
        }
        if "source_event" in args:
            msg["source_event"] = args["source_event"]
        self._ws_client.send(msg)
        resp = self._wait_for_response(req_id)
        return {"event_id": resp.get("event_id")}

    # ─── Request/Response Plumbing ────────────────────────────────────────────

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"mcp_{self._request_counter}"

    def _wait_for_response(self, request_id: str) -> dict[str, Any]:
        """Wait synchronously for a WS response matching request_id.

        Uses a threading.Event to block until the WS message handler resolves
        or the timeout fires.
        """
        event = threading.Event()
        result: dict[str, Any] = {}
        error: list[Exception] = []

        def resolve(msg: dict[str, Any]) -> None:
            result.update(msg)
            event.set()

        def reject(err: Exception) -> None:
            error.append(err)
            event.set()

        timeout_s = self._request_timeout / 1000.0

        def _timeout() -> None:
            self._pending_requests.pop(request_id, None)
            reject(Exception(f"Request {request_id} timed out"))

        timer = threading.Timer(timeout_s, _timeout)
        self._pending_requests[request_id] = _PendingRequest(
            resolve=resolve,
            reject=reject,
            timer=timer,
        )
        timer.start()

        event.wait()

        if error:
            raise error[0]
        return result

    def _handle_ws_response(self, msg: dict[str, Any]) -> None:
        """Handle a server response from the WebSocket."""
        request_id = msg.get("request_id")
        if isinstance(request_id, str):
            pending = self._pending_requests.pop(request_id, None)
            if pending is not None:
                pending.timer.cancel()
                if msg.get("type") == "error":
                    pending.reject(Exception(msg.get("message", "Server error")))
                else:
                    pending.resolve(msg)

    def _write_response(self, response: JsonRpcSuccessResponse | JsonRpcErrorResponse) -> None:
        """Write a JSON-RPC response to output."""
        if isinstance(response, JsonRpcSuccessResponse):
            obj: dict[str, Any] = {
                "jsonrpc": response.jsonrpc,
                "id": response.id,
                "result": response.result,
            }
        else:
            obj = {
                "jsonrpc": response.jsonrpc,
                "id": response.id,
                "error": response.error,
            }
        self._output.write(json.dumps(obj) + "\n")
        self._output.flush()

    # ─── Test Helpers ─────────────────────────────────────────────────────────

    @property
    def holds_lock(self) -> bool:
        return self._holds_lock

    @property
    def has_read_since_last_commit(self) -> bool:
        return self._has_read_since_last_commit

    @property
    def accumulator(self) -> StreamAccumulator:
        return self._accumulator

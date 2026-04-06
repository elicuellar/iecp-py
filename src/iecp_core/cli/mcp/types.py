"""MCP Protocol Types -- minimal JSON-RPC subset for IECP.

Mirrors packages/cli/src/mcp/types.ts exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ─── JSON-RPC ─────────────────────────────────────────────────────────────────


@dataclass
class JsonRpcRequest:
    jsonrpc: str
    id: str | int
    method: str
    params: dict[str, Any] | None = None


@dataclass
class JsonRpcSuccessResponse:
    jsonrpc: str
    id: str | int
    result: Any


@dataclass
class JsonRpcErrorResponse:
    jsonrpc: str
    id: str | int
    error: dict[str, Any]


JsonRpcResponse = JsonRpcSuccessResponse | JsonRpcErrorResponse


# ─── MCP Tool Definition ──────────────────────────────────────────────────────


@dataclass
class McpToolDefinition:
    name: str
    description: str
    inputSchema: dict[str, Any]


# ─── MCP Initialize ───────────────────────────────────────────────────────────


@dataclass
class McpServerInfo:
    name: str
    version: str


@dataclass
class McpCapabilities:
    tools: dict[str, Any] = field(default_factory=dict)


@dataclass
class McpInitializeResult:
    protocolVersion: str
    serverInfo: McpServerInfo
    capabilities: McpCapabilities


# ─── MCP Tool Call ────────────────────────────────────────────────────────────


@dataclass
class McpToolCallParams:
    name: str
    arguments: dict[str, Any]


@dataclass
class McpToolResult:
    content: list[dict[str, str]]
    isError: bool | None = None

"""MCP sub-package for the IECP CLI."""

from .mcp_server import McpServer, McpServerDeps
from .stream_accumulator import StreamAccumulator
from .tools import IECP_TOOLS
from .types import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcSuccessResponse,
    JsonRpcErrorResponse,
    McpToolDefinition,
    McpToolCallParams,
    McpToolResult,
    McpInitializeResult,
    McpServerInfo,
    McpCapabilities,
)

__all__ = [
    "McpServer",
    "McpServerDeps",
    "StreamAccumulator",
    "IECP_TOOLS",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "JsonRpcSuccessResponse",
    "JsonRpcErrorResponse",
    "McpToolDefinition",
    "McpToolCallParams",
    "McpToolResult",
    "McpInitializeResult",
    "McpServerInfo",
    "McpCapabilities",
]

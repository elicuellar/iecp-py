"""MCP Tool Definitions -- all 11 IECP tools per §10.2.

Mirrors packages/cli/src/mcp/tools.ts exactly.
"""

from __future__ import annotations

from .types import McpToolDefinition

IECP_TOOLS: list[McpToolDefinition] = [
    McpToolDefinition(
        name="get_room_status",
        description="Get current room state: participants, lock holder, AI depth counter",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    McpToolDefinition(
        name="fetch_unread_batch",
        description="Fetch unread messages and context payload",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    McpToolDefinition(
        name="fetch_history",
        description="Retrieve older messages for deep context",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "number", "description": "Number of messages to retrieve"},
                "before_id": {"type": "string", "description": "Fetch messages before this event ID"},
            },
            "required": ["limit"],
        },
    ),
    McpToolDefinition(
        name="acquire_speaking_lock",
        description=(
            "Request the Floor Lock to speak. "
            "Pre-condition: must have called fetch_unread_batch since last commit/yield."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "estimated_ms": {"type": "number", "description": "Estimated response time in milliseconds"},
                "intent_summary": {"type": "string", "description": "Brief description of what you plan to say"},
            },
            "required": ["estimated_ms"],
        },
    ),
    McpToolDefinition(
        name="append_stream_chunk",
        description="Push a partial response for real-time display. Pre-condition: must hold the Floor Lock.",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Partial response text to stream"},
            },
            "required": ["text"],
        },
    ),
    McpToolDefinition(
        name="commit_message",
        description="Finalize the streamed response and release the Floor Lock.",
        inputSchema={
            "type": "object",
            "properties": {
                "mentions": {
                    "type": "array",
                    "description": "Entity IDs to mention",
                    "items": {"type": "string"},
                },
                "metadata": {"type": "object", "description": "Additional metadata for the message"},
            },
            "required": [],
        },
    ),
    McpToolDefinition(
        name="yield_floor",
        description="Release the lock without committing -- nothing to say.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    McpToolDefinition(
        name="report_action",
        description="Report an external action performed by the daemon. No lock needed.",
        inputSchema={
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "description": "Type of action performed"},
                "description": {"type": "string", "description": "Human-readable description of the action"},
                "result": {"type": "object", "description": "Result data from the action"},
                "status": {
                    "type": "string",
                    "description": "Current status of the action",
                    "enum": ["initiated", "in_progress", "completed", "failed"],
                },
            },
            "required": ["action_type", "description", "status"],
        },
    ),
    McpToolDefinition(
        name="signal_attention",
        description=(
            "Emit an attention signal (thinking, acknowledged, deferred, listening). "
            "No lock needed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "signal": {
                    "type": "string",
                    "description": "Type of attention signal",
                    "enum": ["listening", "thinking", "deferred", "acknowledged"],
                },
                "utterance_ref": {"type": "string", "description": "Event ID this signal responds to"},
                "note": {"type": "string", "description": "Optional note (e.g. reason for deferral)"},
            },
            "required": ["signal"],
        },
    ),
    McpToolDefinition(
        name="propose_decision",
        description="Propose a decision for the conversation. No lock needed.",
        inputSchema={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Decision summary text"},
                "context_events": {
                    "type": "array",
                    "description": "Related event IDs for context",
                    "items": {"type": "string"},
                },
            },
            "required": ["summary"],
        },
    ),
    McpToolDefinition(
        name="handoff_to",
        description="Transfer responsibility to another entity. No lock needed.",
        inputSchema={
            "type": "object",
            "properties": {
                "to_entity": {"type": "string", "description": "Entity ID to hand off to"},
                "reason": {"type": "string", "description": "Why you are handing off"},
                "context_summary": {"type": "string", "description": "Summary of relevant context"},
                "source_event": {"type": "string", "description": "Event that triggered the handoff"},
            },
            "required": ["to_entity", "reason", "context_summary"],
        },
    ),
]

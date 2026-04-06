"""Gateway module -- Phase 7 of the IECP protocol.

WebSocket gateway for real-time event delivery to humans and daemon CLIs.
"""

from .connection_manager import ConnectionManager
from .daemon_buffer import DaemonBuffer
from .simple_token_validator import SimpleTokenValidator
from .types import (
    DEFAULT_GATEWAY_CONFIG,
    ActiveSignal,
    AttentionSignalType,
    AuthToken,
    ClientType,
    GatewayClient,
    GatewayConfig,
    TokenValidator,
)
from .websocket_gateway import WebSocketGateway

__all__ = [
    "ConnectionManager",
    "DaemonBuffer",
    "SimpleTokenValidator",
    "WebSocketGateway",
    "DEFAULT_GATEWAY_CONFIG",
    "ActiveSignal",
    "AttentionSignalType",
    "AuthToken",
    "ClientType",
    "GatewayClient",
    "GatewayConfig",
    "TokenValidator",
]

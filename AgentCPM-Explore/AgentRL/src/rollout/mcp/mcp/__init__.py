"""
Official MCP (Model Context Protocol) SDK support module.

This module uses the official MCP Python SDK to connect to MCP servers.

Based on: https://modelcontextprotocol.io/docs/develop/build-client

Main components:
- MCPHandlerSDK: Client handler using official MCP SDK
- MCPManagerSDK: Manager using official MCP SDK

Supported transports:
- SSE (Server-Sent Events) - for HTTP servers
- stdio - for local process servers

Installation:
    pip install mcp

Documentation:
    - SDK Usage: SDK_USAGE.md
    - Official Docs Compliance: OFFICIAL_DOCS_COMPLIANCE.md
    - Implementation Details: TRUE_SDK_IMPLEMENTATION.md
"""

try:
    from .mcp_handler_sdk import MCPHandlerSDK
    from .mcp_manager_sdk import MCPManagerSDK
    
    # Backward compatibility aliases
    MCPHandler = MCPHandlerSDK
    MCPManager = MCPManagerSDK
    
    __all__ = ["MCPHandlerSDK", "MCPManagerSDK", "MCPHandler", "MCPManager"]
    
except ImportError as e:
    import logging
    logging.warning(f"MCP SDK not available: {e}. Install with: pip install mcp")
    raise ImportError(
        "MCP SDK is required. Install with: pip install mcp\n"
        "See SDK_USAGE.md for details."
    ) from e


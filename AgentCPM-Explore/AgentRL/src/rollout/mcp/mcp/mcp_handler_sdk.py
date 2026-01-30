#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Official MCP SDK-based Handler

Provides MCPHandler interface using official MCP SDK.
Compatible with MCPAPIHandler for drop-in replacement.
"""

import uuid
from log import logger
from typing import Dict, List, Any, Optional

from .mcp_manager_sdk import MCPManagerSDK, MCP_SDK_AVAILABLE


class MCPHandlerSDK:
    """
    MCP Handler using official SDK
    
    Provides the same interface as MCPHandler and MCPAPIHandler
    but uses official MCP SDK internally.
    """
    
    def __init__(
        self,
        server_url: str,
        config_file: str = "config.toml",  # For compatibility, not used
        auth_token: Optional[str] = None,
    ):
        """
        Initialize MCP handler with SDK
        
        Args:
            server_url: MCP server URL (e.g., http://localhost:8000/mcp)
            config_file: Config file path (for compatibility, not used)
        """
        if not MCP_SDK_AVAILABLE:
            raise ImportError(
                "MCP SDK is not installed. Install with: pip install mcp\n"
                "Or use MCPHandler for manual implementation."
            )
        
        self.server_url = server_url
        self.config_file = config_file
        self.tools = []
        self.openai_tools = []
        self.tool_to_server_map = {}
        self.auth_token = auth_token
        
        # Tool name mapping
        self.tool_name_mapping = {
            "fetch_url": "fetch_url"  # display_name -> actual_tool_name
        }
        self.reverse_tool_name_mapping = {v: k for k, v in self.tool_name_mapping.items()}
        
        # Blocked tools list
        self.blocked_tools = set()
        
        # Manager instance (using SDK)
        self.manager = None
        self.session_id = str(uuid.uuid4())
        
    async def initialize(self) -> bool:
        """
        Initialize connection and tools using SDK
        
        Returns:
            Whether initialization was successful
        """
        try:
            # Ensure blocked_tools is an iterable set
            if self.blocked_tools is None:
                self.blocked_tools = set()

            # Create and initialize manager with SDK
            self.manager = MCPManagerSDK(server_url_or_command=self.server_url, auth_token=self.auth_token)
            
            if not await self.manager.initialize():
                logger.error("MCP Manager (SDK) initialization failed")
                return False
            
            # Get tool list from manager
            self.openai_tools = self.manager.openai_tools
            
            # Filter blocked tools
            self.openai_tools = [
                tool for tool in self.openai_tools
                if tool.get("function", {}).get("name") not in self.blocked_tools
            ]
            
            # Build tool to server map (for compatibility)
            # Since official MCP has a single server endpoint, we just use the URL
            for tool in self.openai_tools:
                if "function" in tool:
                    tool_name = tool["function"].get("name", "unknown")
                    self.tool_to_server_map[tool_name] = self.server_url
            
            logger.info(f"MCP handler (SDK) initialized with {len(self.openai_tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"MCP handler (SDK) initialization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call a tool using SDK
        
        Args:
            tool_name: Tool name (may be display name)
            arguments: Arguments dictionary
            
        Returns:
            Tool call result with "status" and "content" fields
        """
        try:
            # Check if tool is blocked
            if tool_name in self.blocked_tools:
                logger.warning(f"Attempted to call blocked tool: {tool_name}")
                return {
                    "status": "error",
                    "content": {
                        "error": f"Tool {tool_name} is blocked and cannot be used"
                    }
                }
            
            # Map display name to actual tool name
            actual_tool_name = self.tool_name_mapping.get(tool_name, tool_name)
            
            # Check if actual tool name is blocked
            if actual_tool_name in self.blocked_tools:
                logger.warning(f"Attempted to call blocked tool: {tool_name} (actual: {actual_tool_name})")
                return {
                    "status": "error",
                    "content": {
                        "error": f"Tool {actual_tool_name} is blocked and cannot be used"
                    }
                }
            
            # Call tool through SDK manager
            result = await self.manager.call_tool(actual_tool_name, arguments)
            
            # Result already in correct format from manager
            return result
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Error details: {tb}")
            
            return {
                "status": "error",
                "content": {
                    "error": str(e)
                }
            }
             
    async def close(self) -> None:
        """
        Close connection and resources
        """
        if self.manager:
            try:
                await self.manager.close()
            except Exception as e:
                logger.error(f"Error closing MCP Manager (SDK): {str(e)}")


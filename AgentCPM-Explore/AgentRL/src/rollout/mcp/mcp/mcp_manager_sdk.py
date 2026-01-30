#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Official MCP SDK-based Manager

This module implements the client using the official MCP Python SDK.
Uses official ClientSession and transport components.

Installation:
    pip install mcp

Documentation:
    https://github.com/modelcontextprotocol/python-sdk
"""

import json
import asyncio
from contextlib import AsyncExitStack
from log import logger
from typing import Dict, List, Any, Optional
from functools import wraps

try:
    # Import official MCP SDK components (following official docs)
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.session import ClientSession as MCPClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp import types
    MCP_SDK_AVAILABLE = True
except ImportError:
    MCP_SDK_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    logger.warning("MCP SDK not available. Install with: pip install mcp")


def retry_on_error(max_retries: int = 3, backoff_factor: float = 1.5, base_delay: float = 1.0):
    """
    Decorator for retrying async functions on transient errors
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_count = 0
            last_exception = None
            
            while retry_count <= max_retries:
                try:
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    # Retry on network/connection errors
                    # Don't retry on protocol/validation errors
                    if "validation" in str(e).lower() or "protocol" in str(e).lower():
                        raise
                    
                    last_exception = e
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        delay = base_delay * (backoff_factor ** (retry_count - 1))
                        logger.warning(f"{func.__name__} error: {e}, retrying after {delay:.2f}s ({retry_count}/{max_retries})")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
            
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


class MCPManagerSDK:
    """
    MCP Manager using official SDK (following official best practices)
    
    Based on: https://modelcontextprotocol.io/docs/develop/build-client
    
    Uses official MCP Python SDK's ClientSession and transport components.
    Provides the same interface as MCPManager for compatibility.
    """
    
    def __init__(
        self, 
        server_url_or_command: str, 
        timeout: int = 150, 
        use_sse: bool = True,
        server_args: Optional[List[str]] = None,
        auth_token: Optional[str] = None,
    ):
        """
        Initialize MCP Manager with official SDK
        
        Args:
            server_url_or_command: 
                - For SSE: HTTP URL (e.g., "http://localhost:8000")
                - For stdio: Command to run server (e.g., "python" or "node")
            timeout: Request timeout in seconds
            use_sse: Whether to use SSE transport (True) or stdio (False)
            server_args: Arguments for stdio server (e.g., ["server.py"])
        """
        if not MCP_SDK_AVAILABLE:
            raise ImportError(
                "MCP SDK is not installed. Install with: pip install mcp\n"
                "Or use mcp_manager.MCPManager for manual implementation."
            )
        
        self.server_url_or_command = server_url_or_command
        self.timeout = timeout
        self.use_sse = use_sse
        self.server_args = server_args or []
        self.auth_token = auth_token
        
        # Official SDK components (following official docs)
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()  # ✅ As per official docs
        
        # State
        self._initialized = False
        self._closed = False
        
        # Server info
        self.server_info = {}
        self.tools = []
        self.openai_tools = []
        
        # Blocked tools
        self.blocked_tools = {"read_file"}
        
        # Tool name mapping
        self.tool_name_mapping = {
            "fetch_url": "fetch_url"
        }
        self.reverse_tool_name_mapping = {v: k for k, v in self.tool_name_mapping.items()}
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self._initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        return False
    
    @retry_on_error(max_retries=3)
    async def initialize(self) -> bool:
        """
        Initialize MCP connection using official SDK
        
        Follows official best practices from:
        https://modelcontextprotocol.io/docs/develop/build-client
        """
        if self._closed:
            logger.error("Cannot initialize: manager has been closed")
            return False
        
        if self._initialized:
            logger.warning("Manager already initialized")
            return True
        
        try:
            headers = None
            if self.auth_token:
                headers = {"Authorization": f"Bearer {self.auth_token}"}

            if self.use_sse:
                # ✅ Use SSE transport (for HTTP servers)
                logger.info(f"Initializing MCP with SSE transport: {self.server_url_or_command}")
                
                # Following official pattern with AsyncExitStack
                try:
                    stdio_transport = await self.exit_stack.enter_async_context(
                        sse_client(self.server_url_or_command, headers=headers) if headers else sse_client(self.server_url_or_command)
                    )
                except TypeError:
                    # Fallback if SDK sse_client does not accept headers kwarg
                    logger.warning("sse_client does not support headers kwarg; retrying without Authorization header")
                    stdio_transport = await self.exit_stack.enter_async_context(
                        sse_client(self.server_url_or_command)
                    )
                read_stream, write_stream = stdio_transport
                
            else:
                # ✅ Use stdio transport (for local processes) - as per official docs
                logger.info(f"Initializing MCP with stdio transport: {self.server_url_or_command}")
                
                # Following official pattern
                server_params = StdioServerParameters(
                    command=self.server_url_or_command,
                    args=self.server_args,
                    env=None
                )
                
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                read_stream, write_stream = stdio_transport
            
            # ✅ Create ClientSession using AsyncExitStack (official pattern)
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            # ✅ Initialize the session (official SDK handles the handshake)
            init_result = await self.session.initialize()
            
            # Store server info from SDK result
            self.server_info = {
                "serverInfo": init_result.server_info.__dict__ if hasattr(init_result, 'server_info') else {},
                "protocolVersion": init_result.protocol_version if hasattr(init_result, 'protocol_version') else "unknown",
                "capabilities": init_result.capabilities.__dict__ if hasattr(init_result, 'capabilities') else {}
            }
            logger.info(f"MCP initialized via SDK: {self.server_info.get('serverInfo', {})}")
            
            # ✅ List tools using SDK (as per official docs)
            if not await self._list_tools():
                logger.warning("Failed to list tools, but initialization succeeded")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing MCP with SDK: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Clean up on failure
            await self._cleanup_session()
            return False
    
    @retry_on_error(max_retries=3)
    async def _list_tools(self) -> bool:
        """List available tools using official SDK"""
        if not self.session:
            logger.error("Session not initialized")
            return False
        
        try:
            # Use official SDK to list tools
            tools_result = await self.session.list_tools()
            
            # Convert SDK tool objects to our format
            self.tools = []
            for tool in tools_result.tools:
                tool_dict = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                }
                self.tools.append(tool_dict)
            
            # Convert to OpenAI format
            for tool_dict in self.tools:
                tool_name = tool_dict.get("name", "")
                
                if tool_name in self.blocked_tools:
                    continue
                
                display_name = self.reverse_tool_name_mapping.get(tool_name, tool_name)
                description = tool_dict.get("description", "")
                input_schema = tool_dict.get("inputSchema", {})
                
                # Process execute_code tool
                display_name, description, parameters = self._process_execute_code_tool(
                    tool_name, display_name, description, input_schema
                )
                
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": display_name,
                        "description": description,
                        "parameters": parameters
                    }
                }
                self.openai_tools.append(openai_tool)
            
            logger.info(f"Listed {len(self.openai_tools)} tools from MCP server via SDK")
            return True
            
        except Exception as e:
            logger.error(f"Error listing tools via SDK: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Call an MCP tool using official SDK
        
        Returns:
            Dict with "status" and "content" fields
        """
        if self._closed:
            logger.error(f"Manager has been closed: {self._closed}")
            return {
                "status": "error",
                "content": {
                    "error": "Manager has been closed",
                    "detail": "Cannot call tools on a closed manager"
                }
            }
        
        if not self._initialized or not self.session:
            logger.error(f"Manager not initialized or session not available: {self._initialized} {self.session}")
            return {
                "status": "error",
                "content": {
                    "error": "Manager not initialized",
                    "detail": "Call initialize() before calling tools"
                }
            }
            
        
        try:
            actual_tool_name = self.tool_name_mapping.get(tool_name, tool_name)
            
            # Use retry decorator for non-execute_code tools
            if "execute_code" in actual_tool_name.lower():
                result = await self._call_tool_once_sdk(actual_tool_name, arguments)
            else:
                # Wrap with retry
                @retry_on_error(max_retries=3)
                async def _retry_call():
                    return await self._call_tool_once_sdk(actual_tool_name, arguments)
                
                result = await _retry_call()
            
            return result
            
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "content": {
                    "error": "Tool call timeout",
                    "detail": f"Tool {tool_name} timed out after {timeout or self.timeout}s"
                }
            }
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} via SDK: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "content": {
                    "error": str(e),
                    "detail": "Internal error"
                }
            }
    
    async def _call_tool_once_sdk(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Single tool call attempt using official SDK"""
        if not self.session:
            return {
                "status": "error",
                "content": {
                    "error": "Session not available",
                    "detail": "SDK session is not initialized"
                }
            }
        
        try:
            # Use official SDK to call tool
            result = await self.session.call_tool(tool_name, arguments=arguments)
            
            # Check if it's an error
            if result.isError:
                error_message = ""
                for content in result.content:
                    if hasattr(content, 'text'):
                        error_message = content.text
                        break
                
                return {
                    "status": "error",
                    "content": {
                        "error": error_message or "Tool execution failed",
                        "detail": f"Tool {tool_name} returned error"
                    }
                }
            
            # Extract content from SDK result
            combined_content = ""
            for content in result.content:
                if hasattr(content, 'text'):
                    combined_content += content.text
                elif hasattr(content, 'type'):
                    # Handle other content types
                    if content.type == "image":
                        combined_content += f"[Image: {getattr(content, 'mimeType', 'unknown')}]"
                    elif content.type == "resource":
                        combined_content += getattr(content, 'text', getattr(content, 'blob', ''))
            
            if not combined_content.strip():
                return {
                    "status": "error",
                    "content": {
                        "error": "Tool returned empty content",
                        "detail": f"Tool {tool_name} with arguments {arguments}"
                    }
                }
            
            return {
                "status": "success",
                "content": combined_content
            }
            
        except Exception as e:
            logger.error(f"SDK call_tool error: {type(e).__name__}: {str(e)}")
            return {
                "status": "error",
                "content": {
                    "error": f"SDK error: {type(e).__name__}",
                    "detail": str(e)
                }
            }
    
    def _process_execute_code_tool(
        self,
        actual_tool_name: str,
        display_tool_name: str,
        description: str,
        parameters: Dict[str, Any]
    ) -> tuple[str, str, Dict[str, Any]]:
        """Process execute_code tool"""
        if actual_tool_name == "execute_code":
            display_tool_name = "PythonInterpreter"
            description = QWEN_CODE_TOOL_DESCRIPTION
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
        
        return display_tool_name, description, parameters
    
    async def _cleanup_session(self):
        """
        Internal cleanup helper
        
        Uses AsyncExitStack for proper resource cleanup (official pattern)
        """
        try:
            # ✅ Use AsyncExitStack.aclose() (official pattern)
            await self.exit_stack.aclose()
            logger.debug("AsyncExitStack closed successfully")
        except Exception as e:
            logger.warning(f"Error during AsyncExitStack cleanup: {e}")
        finally:
            self.session = None
    
    async def close(self):
        """
        Close SDK session and cleanup resources
        
        Follows official cleanup pattern using AsyncExitStack
        """
        if self._closed:
            logger.debug("Manager already closed")
            return
        
        try:
            await self._cleanup_session()
            logger.debug("MCP SDK session closed successfully")
        except Exception as e:
            logger.error(f"Error closing MCP SDK session: {type(e).__name__}: {str(e)}")
        finally:
            self._closed = True
            self._initialized = False


QWEN_CODE_TOOL_DESCRIPTION = """Executes Python code in a sandboxed environment. To use this tool, you must follow this format:
1. The 'arguments' JSON object must be empty: {}.
2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.

IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function like : print(f"The result is: {np.mean([1,2,3])}").

Example of a correct call:
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
import numpy as np
# Your code here
print(f"The result is: {np.mean([1,2,3])}")
</code>
</tool_call>"""


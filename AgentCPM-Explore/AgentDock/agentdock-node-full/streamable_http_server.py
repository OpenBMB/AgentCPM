#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AgentDock Node Full - Streamable HTTP MCP Server
Full-feature MCP server exposing comprehensive tools via streamable-http protocol.
"""

import os
import sys
import json
import uuid
import asyncio
import logging
import traceback
from typing import Dict, Set, Optional, Any

from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from client import MCPClient
import toml
from mcp.types import TextContent, ImageContent, EmbeddedResource
from openai.types.chat import ChatCompletionMessageToolCall
from auth import auth_middleware

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='[%(asctime)s] %(levelname)s: %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("streamable-http-mcp-server")

# Create FastAPI app
app = FastAPI(
    title="AgentDock MCP Streamable HTTP Server",
    description="Server exposing MCP tools via streamable-http protocol",
    version="0.0.1"
)
app.middleware("http")(auth_middleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store session information
sessions: Dict[str, Dict[str, Any]] = {}
# MCP client
mcp_client: Optional[MCPClient] = None


@app.on_event("startup")
async def startup_event():
    """Event executed on server startup"""
    global mcp_client
    
    # Load config file
    config_path = os.environ.get("CONFIG_FILE_PATH", os.path.join(os.path.dirname(__file__), "config.toml"))
    
    # Auto-select loading method based on file extension
    if config_path.lower().endswith('.toml'):
        config = toml.load(config_path)
    elif config_path.lower().endswith('.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")
    
    # Ensure mcpServers section exists
    if 'mcpServers' not in config:
        raise ValueError(f"mcpServers section not found in config file: {config_path}")
    
    # Initialize MCP client
    mcp_client = MCPClient(config=config["mcpServers"])
    await mcp_client.init_all_sessions()
    logger.info(f"Initialized MCP client, connected to {len(mcp_client.sessions)} servers")


@app.on_event("shutdown")
async def shutdown_event():
    """Event executed on server shutdown"""
    global mcp_client
    if mcp_client:
        await mcp_client.cleanup()
        logger.info("MCP client resources cleaned up")


@app.get("/")
async def root():
    """Root path handler, returns simple info page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AgentDock MCP Service</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                line-height: 1.6;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }
            .info {
                margin-top: 20px;
            }
            .endpoints {
                margin-top: 20px;
            }
            .endpoint {
                background-color: #fff;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 3px;
                border-left: 3px solid #0d6efd;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to AgentDock MCP Service</h1>
            
            <div class="info">
                <p>This server exposes MCP tools via streamable-http protocol.</p>
                <p>Status: <strong>Running</strong></p>
            </div>
            
            <div class="endpoints">
                <h2>Available Endpoints:</h2>
                
                <div class="endpoint">
                    <strong>GET /health</strong>
                    <p>Health check</p>
                </div>
                
                <div class="endpoint">
                    <strong>POST /mcp</strong>
                    <p>Streamable HTTP connection and message handling</p>
                </div>
                
                <div class="endpoint">
                    <strong>GET /sse</strong>
                    <p>Compatibility mode: Establish SSE connection</p>
                </div>
                
                <div class="endpoint">
                    <strong>POST /messages</strong>
                    <p>Compatibility mode: Send message processing request</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "0.0.1",
        "uptime": 0,
        "timestamp": "",
        "connections": len(sessions)
    }


async def get_session_id(request: Request) -> str:
    """Get or create session ID from request
    
    Args:
        request: Request object
        
    Returns:
        str: Session ID
    """
    # Try to get session ID from headers
    session_id = request.headers.get("mcp-session-id") or request.headers.get("X-MCP-Session-ID")
    
    if not session_id:
        # Create new session ID
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session ID: {session_id}")
    
    # Create session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": asyncio.get_event_loop().time(),
            "last_active": asyncio.get_event_loop().time()
        }
        logger.info(f"Created new session: {session_id}")
    else:
        # Update session activity time
        sessions[session_id]["last_active"] = asyncio.get_event_loop().time()
    
    return session_id


def process_tool_arguments(tool_args):
    """Process tool arguments, compatible with string and dict formats
    
    Args:
        tool_args: Tool arguments, can be string or dict
        
    Returns:
        str: Arguments formatted as JSON string
    """
    try:
        if isinstance(tool_args, str):
            try:
                json_obj = json.loads(tool_args)
                return tool_args
            except json.JSONDecodeError:
                return json.dumps({"code": tool_args})
        else:
            return json.dumps(tool_args)
    except Exception as e:
        logger.error(f"Error processing tool arguments: {e}")
        raise ValueError(f"Error processing tool arguments: {str(e)}")


@app.post("/mcp")
async def streamable_http_endpoint(request: Request):
    """Streamable HTTP connection and message handling endpoint"""
    # Get session ID
    session_id = await get_session_id(request)
    
    # Log request
    logger.info(f"Received request: {request.method} {request.url}")
    logger.info(
        "Request headers received: %s",
        {key: "<redacted>" for key in request.headers if key.lower() in {"authorization", "x-api-key", "cookie"}},
    )
    body_bytes = await request.body()
    
    # Update session activity time
    if session_id in sessions:
        sessions[session_id]["last_active"] = asyncio.get_event_loop().time()
    
    # If request body is empty, this is a connection establishment request
    content_length = request.headers.get("content-length")
    if not content_length or int(content_length) == 0:
        logger.info(f"Establishing new streamable-http connection: {session_id}")
        return JSONResponse({
            "sessionId": session_id,
            "message": "Connection successful"
        }, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
    
    # Process request body
    try:
        body = await request.json()
        logger.debug("Parsed request body: %s", body)
        
        # Extract JSON-RPC request
        jsonrpc = body.get("jsonrpc")
        method = body.get("method")
        params = body.get("params", {})
        id = body.get("id")
        
        # Handle Inspector CLI requests
        if method == "tools/list":
            method = "listTools"
            logger.info("Converting tools/list method to listTools")
        elif method == "tools/call":
            method = "callTool"
            logger.info("Converting tools/call method to callTool")
        elif method == "prompts/list":
            method = "listPrompts"
            logger.info("Converting prompts/list method to listPrompts")
        elif method == "resources/list":
            method = "listResources"
            logger.info("Converting resources/list method to listResources")
        elif method and method.startswith("notifications/"):
            logger.info(f"Received {method} notification")
            return JSONResponse({
                "jsonrpc": "2.0",
                "result": {},
                "id": id if id is not None else str(uuid.uuid4())
            }, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
        
        # Validate request
        if not method:
            logger.error("Invalid JSON-RPC request: missing method field")
            return JSONResponse({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32600,
                    "message": "Invalid JSON-RPC request: missing method field"
                },
                "id": id if id is not None else None
            }, status_code=400, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
            
        # Add default values if missing
        if not jsonrpc:
            jsonrpc = "2.0"
            logger.warning(f"Request missing jsonrpc field, using default: {jsonrpc}")
            
        if id is None:
            id = str(uuid.uuid4())
            logger.warning(f"Request missing id field, using default: {id}")
        
        # Process request based on method
        result = None
        
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "serverInfo": {
                    "name": "AgentDock MCP Server",
                    "version": "0.0.1"
                },
                "capabilities": {
                    "tools": {},
                    "prompts": {},
                    "resources": {}
                }
            }
            
        elif method == "listTools" or method == "tools/list":
            if mcp_client:
                mcp_tools = mcp_client.list_tools()
                standard_tools = []
                for tool in mcp_tools:
                    if tool.get("type") == "function" and "function" in tool:
                        function_info = tool["function"]
                        standard_tool = {
                            "name": function_info.get("name", ""),
                            "description": function_info.get("description", ""),
                            "inputSchema": function_info.get("parameters", {})
                        }
                        standard_tools.append(standard_tool)
                
                result = {"tools": standard_tools}
            else:
                result = {"tools": []}
                
        elif method == "listPrompts":
            if mcp_client:
                result = {"prompts": mcp_client.list_prompts()}
            else:
                result = {"prompts": []}
                
        elif method == "listResources":
            if mcp_client:
                result = {"resources": mcp_client.list_resources()}
            else:
                result = {"resources": []}
                
        elif method == "callTool" or method == "tools/call":
            if not mcp_client:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "MCP client not initialized"
                    },
                    "id": id
                }, status_code=500, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
            
            # Parse tool name and arguments
            tool_name = None
            tool_args = {}
            
            if "name" in params:
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
            elif "tool" in params:
                tool_name = params.get("tool")
                tool_args = params.get("params", {})
            
            if not tool_name:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32602,
                        "message": "Missing tool name"
                    },
                    "id": id
                }, status_code=400, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
            
            try:
                args_json = process_tool_arguments(tool_args)
                logger.info(f"Calling tool {tool_name}, arguments: {args_json}")
                
                response = await asyncio.wait_for(
                    mcp_client.call_tool(tool_name=tool_name, tool_args=args_json),
                    timeout=60.0
                )
                
                result_content = []
                for content in response.content:
                    if isinstance(content, TextContent):
                        result_content.append({
                            "type": "text",
                            "text": content.text
                        })
                    elif isinstance(content, ImageContent):
                        result_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + content.data,
                            }
                        })
                    elif isinstance(content, EmbeddedResource):
                        pass
                
                result = {"content": result_content}
                logger.info(f"Tool {tool_name} call successful, returned {len(result_content)} items")
                
            except asyncio.TimeoutError:
                logger.error(f"Tool {tool_name} call timeout (60s)")
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Tool call timeout: {tool_name}",
                        "data": {
                            "details": "Tool call timed out, please try again later"
                        }
                    },
                    "id": id
                }, status_code=500, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
                
            except Exception as e:
                error_details = traceback.format_exc()
                error_msg = str(e)
                logger.error(f"Failed to call tool {tool_name}: {error_msg}\n{error_details}")
                
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Failed to call tool: {error_msg}",
                        "data": {
                            "details": error_msg,
                            "traceback": error_details,
                            "tool_name": tool_name
                        }
                    },
                    "id": id
                }, status_code=500, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
                
        elif method == "getPrompt":
            if not mcp_client:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "MCP client not initialized"
                    },
                    "id": id
                }, status_code=500, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
            
            prompt_name = params.get("name")
            prompt_args = params.get("arguments")
            
            if not prompt_name:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32602,
                        "message": "Missing prompt name"
                    },
                    "id": id
                }, status_code=400, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
            
            try:
                prompt = await mcp_client.get_prompt(prompt_name, prompt_args)
                result = {"content": prompt.content}
            except Exception as e:
                error_details = traceback.format_exc()
                error_msg = str(e)
                logger.error(f"Failed to get prompt: {error_msg}\n{error_details}")
                
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Failed to get prompt: {error_msg}",
                        "data": {
                            "details": error_msg,
                            "traceback": error_details
                        }
                    },
                    "id": id
                }, status_code=500, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
                
        elif method == "readResource":
            if not mcp_client:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "MCP client not initialized"
                    },
                    "id": id
                }, status_code=500, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
            
            resource_name = params.get("uri")
            
            if not resource_name:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32602,
                        "message": "Missing resource URI"
                    },
                    "id": id
                }, status_code=400, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
            
            try:
                resource = await mcp_client.read_resource(resource_name)
                result = {"content": resource.content}
            except Exception as e:
                error_details = traceback.format_exc()
                error_msg = str(e)
                logger.error(f"Failed to read resource: {error_msg}\n{error_details}")
                
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Failed to read resource: {error_msg}",
                        "data": {
                            "details": error_msg,
                            "traceback": error_details
                        }
                    },
                    "id": id
                }, status_code=500, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
        
        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method not supported: {method}"
                },
                "id": id
            }, status_code=400, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
                                    
        # Construct response
        response = {
            "jsonrpc": "2.0",
            "result": result,
            "id": id
        }
        
        return JSONResponse(response, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
        
    except json.JSONDecodeError:
        return JSONResponse({
            "jsonrpc": "2.0",
            "error": {
                "code": -32700,
                "message": "Invalid JSON data"
            },
            "id": None
        }, status_code=400, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})
    except Exception as e:
        error_details = traceback.format_exc()
        error_msg = str(e)
        logger.error(f"Error processing message: {error_msg}\n{error_details}")
        
        return JSONResponse({
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": f"Error processing message: {error_msg}",
                "data": {
                    "details": error_msg,
                    "traceback": error_details
                }
            },
            "id": None
        }, status_code=500, headers={"mcp-session-id": session_id, "X-MCP-Session-ID": session_id})


# SSE compatibility mode code
from sse_starlette.sse import EventSourceResponse

# Store active connections
connections: Dict[str, EventSourceResponse] = {}


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE connection establishment endpoint - backward compatibility mode"""
    session_id = str(uuid.uuid4())
    
    async def event_generator():
        """Generate SSE events"""
        try:
            yield {
                "event": "connection_established",
                "data": json.dumps({"session_id": session_id})
            }
            
            heartbeat_count = 0
            max_heartbeats = 1440  # Auto-disconnect after 12 hours
            
            while heartbeat_count < max_heartbeats:
                try:
                    if session_id not in sessions:
                        logger.info(f"Session {session_id} cleaned up, disconnecting SSE")
                        break
                        
                    if mcp_client is None or not mcp_client.sessions:
                        logger.warning(f"MCP client connection lost, disconnecting SSE {session_id}")
                        break
                    
                    await asyncio.sleep(30)
                    heartbeat_count += 1
                    
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({
                            "timestamp": asyncio.get_event_loop().time(),
                            "heartbeat_count": heartbeat_count,
                            "session_id": session_id
                        })
                    }
                    
                except asyncio.CancelledError:
                    logger.info(f"SSE connection {session_id} cancelled")
                    break
                except Exception as e:
                    logger.error(f"SSE heartbeat failed: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"SSE event generator error: {e}")
        finally:
            if session_id in connections:
                connections.pop(session_id, None)
            logger.info(f"SSE connection {session_id} disconnected and cleaned up")
    
    response = EventSourceResponse(event_generator())
    connections[session_id] = response
    
    @response.background
    async def remove_connection():
        await request.is_disconnected()
        logger.info(f"SSE connection closed: {session_id}")
        connections.pop(session_id, None)
        sessions.pop(session_id, None)
    
    logger.info(f"New SSE connection established: {session_id}")
    return response


@app.post("/messages")
async def handle_messages(request: Request):
    """Handle messages from client - backward compatibility mode"""
    session_id = request.query_params.get("sessionId")
    if not session_id:
        return {"error": "Missing sessionId parameter"}
    
    if session_id not in connections:
        return {"error": "Invalid session ID"}
    
    try:
        body = await request.json()
        logger.debug(f"Received message: {body}")
        
        jsonrpc = body.get("jsonrpc")
        method = body.get("method")
        params = body.get("params", {})
        id = body.get("id")
        
        if not jsonrpc or not method or not isinstance(id, (int, str)):
            return {"error": "Invalid JSON-RPC request"}
        
        result = None
        
        if method == "initialize":
            result = {"capabilities": {}}
            
        elif method == "listTools":
            if mcp_client:
                result = {"tools": mcp_client.list_tools()}
            else:
                result = {"tools": []}
                
        elif method == "listPrompts":
            if mcp_client:
                result = {"prompts": mcp_client.list_prompts()}
            else:
                result = {"prompts": []}
                
        elif method == "listResources":
            if mcp_client:
                result = {"resources": mcp_client.list_resources()}
            else:
                result = {"resources": []}
                
        elif method == "callTool":
            if not mcp_client:
                return {"error": "MCP client not initialized"}
            
            tool_name = params.get("name")
            args = params.get("arguments", {})
            
            if not tool_name:
                return {"error": "Missing tool name"}
            
            try:
                args_json = process_tool_arguments(args)
                response = await mcp_client.call_tool(tool_name=tool_name, tool_args=args_json)
                
                result_content = []
                for content in response.content:
                    if isinstance(content, TextContent):
                        try:
                            text = json.loads(content.text)
                        except json.JSONDecodeError:
                            text = content.text
                        result_content.append({
                            "type": "text",
                            "text": text
                        })
                    elif isinstance(content, ImageContent):
                        result_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + content.data,
                            }
                        })
                    elif isinstance(content, EmbeddedResource):
                        pass
                
                result = {"content": result_content}
                
            except Exception as e:
                error_details = traceback.format_exc()
                error_msg = str(e)
                logger.error(f"Failed to call tool: {error_msg}\n{error_details}")
                return {
                    "error": f"Failed to call tool: {error_msg}", 
                    "details": error_details
                }
                
        elif method == "getPrompt":
            if not mcp_client:
                return {"error": "MCP client not initialized"}
            
            prompt_name = params.get("name")
            prompt_args = params.get("arguments")
            
            if not prompt_name:
                return {"error": "Missing prompt name"}
            
            try:
                prompt = await mcp_client.get_prompt(prompt_name, prompt_args)
                result = {"content": prompt.content}
            except Exception as e:
                error_details = traceback.format_exc()
                error_msg = str(e)
                logger.error(f"Failed to get prompt: {error_msg}\n{error_details}")
                return {
                    "error": f"Failed to get prompt: {error_msg}", 
                    "details": error_details
                }
                
        elif method == "readResource":
            if not mcp_client:
                return {"error": "MCP client not initialized"}
            
            resource_name = params.get("uri")
            
            if not resource_name:
                return {"error": "Missing resource URI"}
            
            try:
                resource = await mcp_client.read_resource(resource_name)
                result = {"content": resource.content}
            except Exception as e:
                error_details = traceback.format_exc()
                error_msg = str(e)
                logger.error(f"Failed to read resource: {error_msg}\n{error_details}")
                return {
                    "error": f"Failed to read resource: {error_msg}", 
                    "details": error_details
                }
        
        else:
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method not supported: {method}"}, "id": id}
                                    
        response = {
            "jsonrpc": "2.0",
            "result": result,
            "id": id
        }
        
        return response
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON data"}
    except Exception as e:
        error_details = traceback.format_exc()
        error_msg = str(e)
        logger.error(f"Error processing message: {error_msg}\n{error_details}")
        return {
            "error": f"Error processing message: {error_msg}", 
            "details": error_details
        }


# Session cleanup task
@app.on_event("startup")
async def start_session_cleanup():
    """Start session cleanup task"""
    asyncio.create_task(session_cleanup_task())


async def session_cleanup_task():
    """Background task to clean up expired sessions"""
    session_timeout = 3600  # 1 hour timeout
    cleanup_interval = 300  # Check every 5 minutes
    error_count = 0
    max_errors = 10
    
    while error_count < max_errors:
        try:
            current_time = asyncio.get_event_loop().time()
            expired_sessions = []
            expired_connections = []
            
            # Find expired sessions
            for session_id, session in list(sessions.items()):
                if current_time - session["last_active"] > session_timeout:
                    expired_sessions.append(session_id)
            
            # Find expired connections
            for session_id in list(connections.keys()):
                if session_id not in sessions:
                    expired_connections.append(session_id)
            
            # Execute cleanup
            for session_id in expired_sessions:
                logger.info(f"Cleaning up expired session: {session_id}")
                sessions.pop(session_id, None)
                
            for session_id in expired_connections:
                logger.info(f"Cleaning up invalid connection: {session_id}")
                connections.pop(session_id, None)
                
            # Log cleanup stats
            if expired_sessions or expired_connections:
                logger.info(f"Session cleanup complete: cleaned {len(expired_sessions)} sessions, {len(expired_connections)} connections")
                
            error_count = 0
            await asyncio.sleep(cleanup_interval)
            
        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled")
            break
        except Exception as e:
            error_count += 1
            logger.error(f"Session cleanup task error (attempt {error_count}): {e}")
            
            wait_time = min(60 * error_count, 600)
            await asyncio.sleep(wait_time)
            
    if error_count >= max_errors:
        logger.error(f"Session cleanup task stopped due to too many errors (count: {error_count})")
    else:
        logger.info("Session cleanup task ended normally")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 9088))
    uvicorn.run(app, host="0.0.0.0", port=port)

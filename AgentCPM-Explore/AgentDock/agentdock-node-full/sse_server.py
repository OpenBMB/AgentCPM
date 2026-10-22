#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AgentDock Node Full - SSE MCP Server
Full-feature MCP server exposing comprehensive tools via SSE protocol.
"""

import os
import sys
import json
import uuid
import asyncio
import logging
from typing import Dict, Set, Optional, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import HTMLResponse

from client import MCPClient
import toml
from mcp.types import TextContent, ImageContent, EmbeddedResource
from openai.types.chat import ChatCompletionMessageToolCall
from auth import auth_middleware

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='[%(asctime)s] %(levelname)s: %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("sse-mcp-server")

# Create FastAPI application
app = FastAPI(
    title="AgentDock MCP SSE Server",
    description="MCP server exposing tools via SSE protocol",
    version="1.0.0"
)
app.middleware("http")(auth_middleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should be more restrictive in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active connections
connections: Dict[str, EventSourceResponse] = {}
# Store session messages
session_messages: Dict[str, Dict[str, Any]] = {}
# MCP client
mcp_client: Optional[MCPClient] = None

@app.on_event("startup")
async def startup_event():
    """Event executed on service startup"""
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
    
    # Ensure mcpServers section exists in config
    if 'mcpServers' not in config:
        raise ValueError(f"mcpServers section not found in config: {config_path}")
    
    # Initialize MCP client
    mcp_client = MCPClient(config=config["mcpServers"])
    await mcp_client.init_all_sessions()
    logger.info(f"Initialized MCP client, connected to {len(mcp_client.sessions)} servers")

@app.on_event("shutdown")
async def shutdown_event():
    """Event executed on service shutdown"""
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
                <p>This is a server exposing MCP tools via SSE protocol.</p>
                <p>Status: <strong>Running</strong></p>
            </div>
            
            <div class="endpoints">
                <h2>Available Endpoints:</h2>
                
                <div class="endpoint">
                    <strong>GET /health</strong>
                    <p>Health check</p>
                </div>
                
                <div class="endpoint">
                    <strong>GET /sse</strong>
                    <p>Establish SSE connection</p>
                </div>
                
                <div class="endpoint">
                    <strong>POST /messages</strong>
                    <p>Send message processing request</p>
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
        "version": "1.0.0",
        "uptime": 0,
        "timestamp": "",
        "connections": len(connections)
    }

@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE connection establishment endpoint"""
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    async def event_generator():
        """Generate SSE events"""
        # Send connection established event
        yield {
            "event": "connection_established",
            "data": json.dumps({"session_id": session_id})
        }
        
        # Keep connection until client disconnects
        while True:
            # Heartbeat logic
            await asyncio.sleep(30)
            yield {
                "event": "heartbeat",
                "data": json.dumps({"timestamp": ""})
            }
    
    # Create SSE response
    response = EventSourceResponse(event_generator())
    # Register connection
    connections[session_id] = response
    
    # Remove connection when closed
    @response.background
    async def remove_connection():
        await request.is_disconnected()
        logger.info(f"SSE connection closed: {session_id}")
        connections.pop(session_id, None)
        session_messages.pop(session_id, None)
    
    logger.info(f"New SSE connection established: {session_id}")
    return response

@app.post("/messages")
async def handle_messages(request: Request):
    """Handle messages from client"""
    # Get session ID
    session_id = request.query_params.get("sessionId")
    if not session_id:
        return {"error": "Missing sessionId parameter"}
    
    # Ensure client has established SSE connection
    if session_id not in connections:
        return {"error": "Invalid session ID"}
    
    # Parse request body
    try:
        body = await request.json()
        logger.debug(f"Received message: {body}")
        
        # Extract JSON-RPC request
        jsonrpc = body.get("jsonrpc")
        method = body.get("method")
        params = body.get("params", {})
        id = body.get("id")
        
        if not jsonrpc or not method or not isinstance(id, (int, str)):
            return {"error": "Invalid JSON-RPC request"}
        
        # Process request based on method
        result = None
        
        if method == "initialize":
            # Initialize request
            result = {"capabilities": {}}
            
        elif method == "listTools":
            # List tools
            if mcp_client:
                result = {"tools": mcp_client.list_tools()}
            else:
                result = {"tools": []}
                
        elif method == "listPrompts":
            # List prompts
            if mcp_client:
                result = {"prompts": mcp_client.list_prompts()}
            else:
                result = {"prompts": []}
                
        elif method == "listResources":
            # List resources
            if mcp_client:
                result = {"resources": mcp_client.list_resources()}
            else:
                result = {"resources": []}
                
        elif method == "callTool":
            # Call tool
            if not mcp_client:
                return {"error": "MCP client not initialized"}
            
            tool_name = params.get("name")
            args = params.get("arguments", {})
            
            if not tool_name:
                return {"error": "Missing tool name"}
            
            try:
                # Convert arguments to JSON string
                args_json = json.dumps(args)
                # Call tool
                response = await mcp_client.call_tool(tool_name=tool_name, tool_args=args_json)
                
                # Process tool call result
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
                                "url": "data:image/jpeg;base64,"+content.data,
                            }
                        })
                    elif isinstance(content, EmbeddedResource):
                        # EmbeddedResource not supported yet
                        pass
                
                result = {"content": result_content}
                
            except Exception as e:
                logger.error(f"Tool call failed: {e}")
                return {"error": f"Tool call failed: {str(e)}"}
                
        elif method == "getPrompt":
            # Get prompt
            if not mcp_client:
                return {"error": "MCP client not initialized"}
            
            prompt_name = params.get("name")
            prompt_args = params.get("arguments")
            
            if not prompt_name:
                return {"error": "Missing prompt name"}
            
            try:
                # Get prompt
                prompt = await mcp_client.get_prompt(prompt_name, prompt_args)
                result = {"content": prompt.content}
            except Exception as e:
                logger.error(f"Get prompt failed: {e}")
                return {"error": f"Get prompt failed: {str(e)}"}
                
        elif method == "readResource":
            # Read resource
            if not mcp_client:
                return {"error": "MCP client not initialized"}
            
            resource_name = params.get("uri")
            
            if not resource_name:
                return {"error": "Missing resource URI"}
            
            try:
                # Read resource
                resource = await mcp_client.read_resource(resource_name)
                result = {"content": resource.content}
            except Exception as e:
                logger.error(f"Read resource failed: {e}")
                return {"error": f"Read resource failed: {str(e)}"}
        
        else:
            # Unsupported method
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method not supported: {method}"}, "id": id}
        
        # Construct response
        response = {
            "jsonrpc": "2.0",
            "result": result,
            "id": id
        }
        
        return response
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON data"}
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return {"error": f"Error processing message: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # Start server
    port = int(os.environ.get("PORT", 8088))
    uvicorn.run(app, host="0.0.0.0", port=port)

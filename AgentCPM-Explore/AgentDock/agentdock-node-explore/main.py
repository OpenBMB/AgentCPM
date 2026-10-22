"""
AgentDock Node Explore - Main API
Advanced MCP server for search, web exploration and analysis.
"""
import os
import toml
import json

from typing import Optional
from fastapi import FastAPI, HTTPException
from client import MCPClient
from openai.types.chat import ChatCompletionMessageToolCall
from contextlib import asynccontextmanager
from mcp.types import TextContent, ImageContent, EmbeddedResource
from auth import auth_middleware

client: MCPClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI"""
    global client
    
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
        raise ValueError(f"mcpServers section not found in config file: {config_path}")
    
    client = MCPClient(config=config["mcpServers"])
    await client.init_all_sessions()
    yield
    await client.cleanup()


app = FastAPI(lifespan=lifespan)
app.middleware("http")(auth_middleware)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hello, World!"
    }
    

@app.get("/session_cfg")
async def get_cfg():
    """Get all servers"""
    return {
        "servers": client.config
    }
    

@app.get("/list_tools")
async def list_tools():
    """List all tools and their descriptions"""
    return client.list_tools()


@app.get("/list_prompts")
async def list_prompts():
    """List all prompts and their descriptions"""
    return client.list_prompts()


@app.get("/list_resources")
async def list_resources():
    """List all resources and their descriptions"""
    return client.list_resources()


@app.get("/get_prompt")
async def get_prompt(prompt_name: str, prompt_args: Optional[str] = None):
    """Get a prompt by ID"""
    try:
        prompt = await client.get_prompt(prompt_name)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))
    
    return prompt


@app.get("/read_resource")
async def read_resource(resource_name: str):
    """Read a resource by name"""
    try:
        resource = await client.read_resource(resource_name)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))

    return resource


@app.post("/call_tool")
async def call_tool(tool_call: ChatCompletionMessageToolCall):
    """Call a tool with parameters"""
    try:
        res = await client.call_tool(tool_name=tool_call.function.name, tool_args=tool_call.function.arguments)
    except Exception as e:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(e)
        }
    
    ret = {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": []
    }
    for content in res.content:
        if isinstance(content, TextContent):
            try:
                text = json.loads(content.text)
            except json.JSONDecodeError:
                text = content.text
            ret["content"].append({
                "type": "text",
                "text": text
            })
        elif isinstance(content, ImageContent):
            ret["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + content.data,
                }
            })
        elif isinstance(content, EmbeddedResource):
            raise NotImplementedError("EmbeddedResource is not implemented yet")
        else:
            raise NotImplementedError(f"Unknown content type: {type(content)}")
    
    if len(ret["content"]) == 1 and ret["content"][0]["type"] == "text":
        ret["content"] = ret["content"][0]["text"]
    return ret


@app.get("/mcpapi/servers")
async def list_servers():
    """List all available MCP servers"""
    return {
        "servers": list(client.config.keys())
    }


@app.get("/mcpapi/tools")
async def list_all_tools():
    """List tools from all servers"""
    return {
        "tools": client.list_tools()
    }


@app.get("/mcpapi/server/{server_id}/tools")
async def list_server_tools(server_id: str):
    """List tools from a specific server"""
    if server_id not in client.session_tools:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")
    
    return {
        "server": server_id,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in client.session_tools[server_id]
        ]
    }


@app.post("/mcpapi/tool/{tool_id}")
async def call_tool_by_id(tool_id: str, args: dict):
    """Call a tool by ID"""
    try:
        # Find which server the tool belongs to
        server_id = None
        tool_name = tool_id
        
        # Check if tool name contains server separator
        if "." in tool_id:
            parts = tool_id.split(".", 1)
            if len(parts) == 2:
                server_id, tool_name = parts
        
        # If no explicit server ID, search for the tool across all servers
        if server_id is None:
            for srv_id, tools in client.session_tools.items():
                for tool in tools:
                    if tool.name == tool_id:
                        server_id = srv_id
                        break
                if server_id:
                    break
            
            if not server_id:
                raise HTTPException(status_code=404, detail=f"Tool not found: {tool_id}")
        
        # Construct full tool name
        full_tool_name = f"{server_id}.{tool_name}"
        
        # Convert args to JSON string
        tool_args = json.dumps(args)
        
        # Call the tool
        res = await client.call_tool(tool_name=full_tool_name, tool_args=tool_args)
        
        # Process response
        result = []
        for content in res.content:
            if isinstance(content, TextContent):
                try:
                    text = json.loads(content.text)
                except json.JSONDecodeError:
                    text = content.text
                result.append({
                    "type": "text",
                    "text": text
                })
            elif isinstance(content, ImageContent):
                result.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + content.data,
                    }
                })
            elif isinstance(content, EmbeddedResource):
                raise NotImplementedError("EmbeddedResource is not implemented yet")
            else:
                raise NotImplementedError(f"Unknown content type: {type(content)}")
        
        # Simplify single text response
        if len(result) == 1 and result[0]["type"] == "text":
            return {
                "content": result[0]["text"]
            }
        else:
            return {
                "content": result
            }
            
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

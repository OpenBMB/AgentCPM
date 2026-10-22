"""
AgentDock Node Full - Full Feature MCP Server
Provides comprehensive MCP tool capabilities for AI agents.
"""
import os
import toml
import json
import asyncio
import signal

from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from client import MCPClient
from openai.types.chat import ChatCompletionMessageToolCall
from contextlib import asynccontextmanager
from mcp.types import TextContent, ImageContent, EmbeddedResource
from auth import auth_middleware

client: MCPClient

# Track active request count
active_requests = 0


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
    
    # Ensure mcpServers section exists
    if 'mcpServers' not in config:
        raise ValueError(f"mcpServers section not found in config file: {config_path}")
    
    client = MCPClient(config=config["mcpServers"])
    await client.init_all_sessions()
    
    # Start scheduled restart task
    restart_task = asyncio.create_task(restart_client(6))
    
    yield
    
    # Cancel restart task on cleanup
    restart_task.cancel()
    try:
        await restart_task
    except asyncio.CancelledError:
        pass
    
    # Clean up MCP client
    try:
        await client.cleanup()
    except Exception as e:
        print(f"Client cleanup error: {e}")


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
        
        # If no explicit server ID, search across all servers
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


# Middleware to track active requests
@app.middleware("http")
async def count_requests(request: Request, call_next):
    global active_requests
    active_requests += 1
    try:
        response = await call_next(request)
        return response
    finally:
        active_requests -= 1


# Scheduled restart when idle
async def restart_if_idle(interval_hours=12):
    while True:
        await asyncio.sleep(20)
        if active_requests == 0:
            print("No active requests, restarting process...")
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                await asyncio.sleep(5)
            except:
                os._exit(0)
        else:
            print(f"Still busy ({active_requests} active), postpone restart.")


async def restart_client(interval_hours=6):
    """Periodically restart the MCP client to refresh connections"""
    global client
    restart_count = 0
    max_restarts = 24  # Limit restarts to avoid infinite loop
    
    while restart_count < max_restarts:
        try:
            await asyncio.sleep(interval_hours * 3600)
            
            # Use efficient waiting mechanism
            wait_timeout = 300  # Max wait 5 minutes
            wait_start = asyncio.get_event_loop().time()
            
            while active_requests > 0:
                current_time = asyncio.get_event_loop().time()
                if current_time - wait_start > wait_timeout:
                    print(f"Wait timeout for active requests ({active_requests} still running), forcing restart")
                    break
                await asyncio.sleep(5)

            print(f"Starting MCPClient restart #{restart_count + 1}...")
            
            # Clean up old client
            if client:
                try:
                    await asyncio.wait_for(client.cleanup(), timeout=30.0)
                    print("Old client cleanup complete")
                except asyncio.TimeoutError:
                    print("Client cleanup timeout, continuing...")
                except Exception as e:
                    print(f"Client cleanup error: {e}")
            
            # Recreate and initialize client
            config_path = os.environ.get("CONFIG_FILE_PATH", os.path.join(os.path.dirname(__file__), "config.toml"))
            if config_path.lower().endswith('.toml'):
                config = toml.load(config_path)
            elif config_path.lower().endswith('.json'):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
                
            client = MCPClient(config=config["mcpServers"])
            
            # Initialize with timeout
            await asyncio.wait_for(client.init_all_sessions(), timeout=60.0)
            
            restart_count += 1
            print(f"MCPClient restart #{restart_count} complete")
            
        except asyncio.CancelledError:
            print("Client restart task cancelled")
            break
        except Exception as e:
            restart_count += 1
            print(f"MCPClient restart #{restart_count} failed: {e}")
            # Wait longer after failure
            await asyncio.sleep(300)
            
    if restart_count >= max_restarts:
        print(f"Reached max restart limit ({max_restarts}), stopping auto-restart")
    else:
        print("Client restart task ended normally")

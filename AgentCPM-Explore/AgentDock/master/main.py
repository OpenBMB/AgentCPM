"""
AgentDock - Docker Container Node Manager for MCP Services
"""
import os
import psutil
import uvicorn
import random
import threading
import asyncio
import datetime
import docker
import json
import subprocess
import shutil
import hmac

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from beanie.exceptions import RevisionIdWasChanged

from config import CONFIG, NODE_API_TOKEN, logger, MANAGER_NAME, db, docker_client
from config import DockerContainerNode, DCNodeChecker

from node import node_router, check_nodes_status_loop
from auth import (
    AUTH_COOKIE,
    auth_middleware,
    configured_token,
    is_authorized,
    node_proxy_headers,
    security_configured,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    AgentDock lifecycle management.
    Sets up necessary configurations like checking and creating 
    table nodes if not exists in database, creating subprocess
    to update node status, and registering path to node.
    """
    from beanie import init_beanie
    await init_beanie(database=db,
                      document_models=[DockerContainerNode, DCNodeChecker],
                      multiprocessing_mode=True)
    
    # Create subprocess to update node status
    await asyncio.sleep(3*random.random()*CONFIG.node.health_check_interval)

    # 1. Check if there is a NodeChecker already running for this manager instance
    exist = False
    async for checker in DCNodeChecker.find(DCNodeChecker.creator_id == MANAGER_NAME):
        if checker.pid is None:
            await checker.delete()
        elif not psutil.pid_exists(checker.pid):
            await checker.delete()
        else:
            exist = True

    # 2. If not, create a NodeChecker and start the subprocess
    if not exist:
        success = False
        try:
            checker = DCNodeChecker(
                creator_id=MANAGER_NAME,
                pid=os.getpid(),
                interval=float(CONFIG.node.health_check_interval),
            )
            await checker.save()
            success = True
        except RevisionIdWasChanged:
            logger.warning("NodeChecker already created by another workers!")
        
        if success:
            # 3. Remove test nodes
            for node in docker_client.containers.list(all=True, filters={
                "label": CONFIG.testingNodeLabel
            }):
                node.remove(force=True)
                logger.info("Test node: " + node.id + " removed!")
            
            # 4. Setup a thread for a checker's event loop
            def monitor_loop():
                asyncio.run(check_nodes_status_loop(checker))
            
            monitor = threading.Thread(target=monitor_loop, daemon=True)
            monitor.start()
            app.monitor = monitor
            
            logger.info(f"NodeChecker created with ident {monitor.ident}!")

    yield

    # Clean up
    
    # 1. Delete checker for this manager instance
    async for checker in DCNodeChecker.find(DCNodeChecker.creator_id == MANAGER_NAME):
        await checker.delete()
    
    # 2. Stop all created nodes first
    if CONFIG.node.stop_after_exit:
        async for node in DockerContainerNode.find(DockerContainerNode.creator_id == MANAGER_NAME):
            container = docker_client.containers.get(node.id)
            if container is not None and container.attrs["State"]["Status"] != "exited":
                container.stop()
                logger.info("Stopping Node: " + node.id)
                node.status = "exited"
                node.health = "Unhealthy"
                await node.replace()

    if hasattr(app, "monitor"):
        app.monitor.join()

    # 3. Close db connection
    db.client.close()
    

app = FastAPI(
    title="AgentDock",
    description="Docker Container Node Manager for MCP Services",
    version="1.0.0",
    lifespan=lifespan
)
app.middleware("http")(auth_middleware)

app.include_router(node_router, prefix="/node")

templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """AgentDock main dashboard page."""
    if not is_authorized(request):
        return HTMLResponse(
            """
            <!doctype html>
            <html lang="en">
            <head><meta charset="utf-8"><title>AgentDock Login</title></head>
            <body>
              <form id="login">
                <label>API token <input id="token" type="password" autocomplete="current-password"></label>
                <button type="submit">Sign in</button>
              </form>
              <p id="error"></p>
              <script>
                document.getElementById("login").addEventListener("submit", async (event) => {
                  event.preventDefault();
                  const response = await fetch("/auth/login", {
                    method: "POST",
                    headers: {"content-type": "application/json"},
                    body: JSON.stringify({token: document.getElementById("token").value})
                  });
                  if (response.ok) {
                    location.reload();
                  } else {
                    document.getElementById("error").textContent = "Invalid token";
                  }
                });
              </script>
            </body>
            </html>
            """,
            status_code=200,
        )
    nodes = DockerContainerNode.find_all()
    
    manager_info = {
        "name": MANAGER_NAME,
        "status": "Running",
        "instance_count": await nodes.count(),
        "node_checker": (await DCNodeChecker.find_one(DCNodeChecker.creator_id == MANAGER_NAME)).model_dump(mode="json"),
    }
    nodes = await nodes.to_list()
    nodes_data = [
        {
            "id": node.id,
            "status": node.status,
            "health": node.health,
            "creator_id": node.creator_id,
            "creator_name": node.creator_name,
            "image_name": node.image_name,
        } for node in nodes if nodes
    ]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "manager_info": manager_info,
        "nodes_data": nodes_data
    })


@app.post("/auth/login")
async def login(payload: dict, response: Response):
    token = configured_token()
    if not security_configured():
        raise HTTPException(
            status_code=503,
            detail="AgentDock API tokens are not configured or are not distinct",
        )
    supplied = str(payload.get("token", ""))
    if not supplied or not hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid token")
    response.set_cookie(
        key=AUTH_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=os.environ.get("AGENTDOCK_COOKIE_SECURE", "").lower() in {"1", "true", "yes"},
    )
    return {"status": "ok"}


@app.get('/api/resources/system')
async def get_system_resources():
    """Get system resource usage information."""
    try:
        load_avg = os.getloadavg()
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return JSONResponse({
            "status": "success",
            "data": {
                "load_average": {
                    "1min": round(load_avg[0], 2),
                    "5min": round(load_avg[1], 2),
                    "15min": round(load_avg[2], 2)
                },
                "cpu": {
                    "count": cpu_count,
                    "percent": round(cpu_percent, 1)
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": round(memory.percent, 1),
                    "total_gb": round(memory.total / (1024**3), 1),
                    "available_gb": round(memory.available / (1024**3), 1),
                    "used_gb": round(memory.used / (1024**3), 1)
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": round((disk.used / disk.total) * 100, 1),
                    "total_gb": round(disk.total / (1024**3), 1),
                    "used_gb": round(disk.used / (1024**3), 1),
                    "free_gb": round(disk.free / (1024**3), 1)
                }
            }
        })
    except Exception as e:
        logger.error(f"Failed to get system resources: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@app.get('/api/resources/docker')
async def get_docker_resources():
    """Get Docker container resource usage information."""
    try:
        containers_data = []
        containers = docker_client.containers.list()
        
        for container in containers:
            try:
                stats = container.stats(stream=False)
                
                # Calculate CPU usage
                cpu_percent = 0.0
                try:
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                    if system_delta > 0 and 'percpu_usage' in stats['cpu_stats']['cpu_usage']:
                        cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
                    elif system_delta > 0:
                        cpu_count = psutil.cpu_count()
                        cpu_percent = (cpu_delta / system_delta) * cpu_count * 100
                except (KeyError, TypeError, ZeroDivisionError):
                    cpu_percent = 0.0
                
                # Memory usage
                memory_usage = 0
                memory_limit = 0
                memory_percent = 0
                try:
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)
                    memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
                except (KeyError, TypeError, ZeroDivisionError):
                    pass
                
                # Network I/O
                network_rx = 0
                network_tx = 0
                try:
                    if 'networks' in stats:
                        for interface in stats['networks'].values():
                            network_rx += interface.get('rx_bytes', 0)
                            network_tx += interface.get('tx_bytes', 0)
                except (KeyError, TypeError):
                    pass
                
                # Disk I/O
                disk_read = 0
                disk_write = 0
                try:
                    if 'blkio_stats' in stats and 'io_service_bytes_recursive' in stats['blkio_stats']:
                        for item in stats['blkio_stats']['io_service_bytes_recursive']:
                            if item.get('op') == 'Read':
                                disk_read += item.get('value', 0)
                            elif item.get('op') == 'Write':
                                disk_write += item.get('value', 0)
                except (KeyError, TypeError):
                    pass
                
                # Container config
                container_info = container.attrs
                host_config = container_info['HostConfig']
                
                memory_limit_config = host_config.get('Memory', 0)
                cpu_limit = host_config.get('NanoCpus', 0)
                cpu_quota = host_config.get('CpuQuota', 0)
                cpu_period = host_config.get('CpuPeriod', 0)
                
                cpu_limit_cores = 0
                if cpu_limit > 0:
                    cpu_limit_cores = cpu_limit / 1000000000
                elif cpu_quota > 0 and cpu_period > 0:
                    cpu_limit_cores = cpu_quota / cpu_period
                
                has_cpu_limit = cpu_limit > 0 or (cpu_quota > 0 and cpu_period > 0)
                has_memory_limit = memory_limit_config > 0
                
                containers_data.append({
                    "id": container.id[:12],
                    "name": container.name,
                    "image": container.image.tags[0] if container.image.tags else "unknown",
                    "status": container.status,
                    "created": container.attrs['Created'],
                    "cpu": {
                        "percent": round(cpu_percent, 2),
                        "limit_cores": cpu_limit_cores
                    },
                    "memory": {
                        "usage": memory_usage,
                        "limit": memory_limit,
                        "percent": round(memory_percent, 1),
                        "usage_mb": round(memory_usage / (1024**2), 1),
                        "limit_mb": round(memory_limit / (1024**2), 1) if memory_limit > 0 else 0,
                        "limit_config": memory_limit_config,
                        "limit_config_gb": round(memory_limit_config / (1024**3), 1) if memory_limit_config > 0 else 0
                    },
                    "network": {
                        "rx_bytes": network_rx,
                        "tx_bytes": network_tx,
                        "rx_mb": round(network_rx / (1024**2), 2),
                        "tx_mb": round(network_tx / (1024**2), 2)
                    },
                    "disk": {
                        "read_bytes": disk_read,
                        "write_bytes": disk_write,
                        "read_mb": round(disk_read / (1024**2), 2),
                        "write_mb": round(disk_write / (1024**2), 2)
                    },
                    "limits": {
                        "has_memory_limit": has_memory_limit,
                        "has_cpu_limit": has_cpu_limit,
                        "memory_limit_gb": round(memory_limit_config / (1024**3), 1) if memory_limit_config > 0 else None,
                        "cpu_limit_cores": cpu_limit_cores if cpu_limit_cores > 0 else None
                    }
                })
                
            except Exception as e:
                logger.warning(f"Failed to get stats for container {container.name}: {e}")
                containers_data.append({
                    "id": container.id[:12],
                    "name": container.name,
                    "image": container.image.tags[0] if container.image.tags else "unknown",
                    "status": container.status,
                    "error": str(e)
                })
        
        total_containers = len(containers)
        unlimited_containers = sum(1 for c in containers_data if 'limits' in c and not (c['limits']['has_memory_limit'] or c['limits']['has_cpu_limit']))
        
        return JSONResponse({
            "status": "success",
            "data": {
                "containers": containers_data,
                "summary": {
                    "total_containers": total_containers,
                    "unlimited_containers": unlimited_containers,
                    "limited_containers": total_containers - unlimited_containers
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get Docker resources: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@app.get('/api/resources/alerts')
async def get_resource_alerts():
    """Get resource alert information."""
    try:
        alerts = []
        
        # Check system load
        load_avg = os.getloadavg()
        if load_avg[0] > 6.0:
            alerts.append({
                "type": "critical",
                "category": "system",
                "message": f"System load too high: {load_avg[0]:.2f}",
                "recommendation": "Consider stopping some Docker containers"
            })
        elif load_avg[0] > 4.0:
            alerts.append({
                "type": "warning",
                "category": "system", 
                "message": f"System load is high: {load_avg[0]:.2f}",
                "recommendation": "Monitor system performance"
            })
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.available < 1024 * 1024 * 1024:
            alerts.append({
                "type": "critical",
                "category": "memory",
                "message": f"Available memory low: {memory.available / (1024**3):.1f}GB",
                "recommendation": "Free up memory or stop containers immediately"
            })
        elif memory.percent > 85:
            alerts.append({
                "type": "warning",
                "category": "memory",
                "message": f"Memory usage high: {memory.percent:.1f}%",
                "recommendation": "Consider cleaning memory or stopping some containers"
            })
        
        # Check unlimited containers
        containers = docker_client.containers.list()
        unlimited_containers = []
        for container in containers:
            host_config = container.attrs['HostConfig']
            memory_limit = host_config.get('Memory', 0)
            cpu_limit = host_config.get('NanoCpus', 0)
            cpu_quota = host_config.get('CpuQuota', 0)
            
            if memory_limit == 0 and cpu_limit == 0 and cpu_quota == 0:
                # Skip critical system containers
                if not any(keyword in container.name.lower() for keyword in ['agentdock', 'mongodb', 'nginx', 'redis']):
                    unlimited_containers.append(container.name)
        
        if unlimited_containers:
            alerts.append({
                "type": "warning",
                "category": "docker",
                "message": f"Found {len(unlimited_containers)} containers without resource limits",
                "recommendation": "Consider setting CPU and memory limits for containers",
                "details": unlimited_containers[:5]
            })
        
        return JSONResponse({
            "status": "success", 
            "data": {
                "alerts": alerts,
                "alert_count": len(alerts),
                "has_critical": any(alert['type'] == 'critical' for alert in alerts)
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get resource alerts: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@app.post('/api/resources/emergency-stop')
async def emergency_stop_containers():
    """Emergency stop high-resource containers."""
    try:
        stopped_containers = []
        containers = docker_client.containers.list()
        container_stats = []
        
        for container in containers:
            try:
                # Skip critical system containers
                if any(keyword in container.name.lower() for keyword in ['agentdock', 'mongodb', 'nginx', 'redis']):
                    continue
                
                stats = container.stats(stream=False)
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                cpu_percent = 0.0
                if system_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
                
                container_stats.append({
                    'container': container,
                    'cpu_percent': cpu_percent,
                    'memory_usage': stats['memory_stats']['usage']
                })
            except:
                continue
        
        container_stats.sort(key=lambda x: x['cpu_percent'], reverse=True)
        
        for i, item in enumerate(container_stats[:3]):
            if i >= 3:
                break
                
            container = item['container']
            try:
                container.stop(timeout=10)
                stopped_containers.append({
                    "name": container.name,
                    "id": container.id[:12],
                    "cpu_percent": round(item['cpu_percent'], 2)
                })
                logger.info(f"Emergency stopped container: {container.name}")
            except Exception as e:
                logger.error(f"Failed to stop container {container.name}: {e}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Stopped {len(stopped_containers)} high-resource containers",
            "data": {
                "stopped_containers": stopped_containers
            }
        })
        
    except Exception as e:
        logger.error(f"Emergency stop failed: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@app.get('/alive')
async def alive():
    """Health check endpoint."""
    try:
        db_info = await db.command("serverStatus")
        docker_info = docker_client.info()
        checker = await DCNodeChecker.find_one(DCNodeChecker.creator_id == MANAGER_NAME)
        
        return {
            "status": "healthy",
            "service": "AgentDock",
            "manager": MANAGER_NAME,
            "db_connected": True,
            "docker_connected": True,
            "node_checker_running": checker is not None,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }


@app.api_route("/container/{container_id}/mcp", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def route_to_container_8088_direct(request: Request, container_id: str):
    """
    Route request to container's port 8088 (MCP port) - direct access.
    URL format: /container/{container_id}/mcp
    """
    return await route_to_container_port(request, container_id, "", 8088)


@app.api_route("/container/{container_id}/mcp/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def route_to_container_8088_with_path(request: Request, container_id: str, path: str):
    """
    Route request to container's port 8088 (MCP port) - with path.
    URL format: /container/{container_id}/mcp/{path}
    """
    return await route_to_container_port(request, container_id, path, 8088)


@app.api_route("/container/{container_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def route_to_container_8000(request: Request, container_id: str, path: str):
    """
    Route request to container's port 8000 (main API port).
    URL format: /container/{container_id}/{path}
    """
    return await route_to_container_port(request, container_id, path, 8000)


async def route_to_container_port(request: Request, container_id: str, path: str, target_port: int):
    """Route request to specified container port."""
    import httpx
    import urllib.parse
    
    try:
        node = None
        
        node = await DockerContainerNode.find_one(DockerContainerNode.id == container_id)
        
        if not node:
            node = await DockerContainerNode.find_one(DockerContainerNode.short_id == container_id)
        
        if not node:
            async for n in DockerContainerNode.find_all():
                if n.id.startswith(container_id) or n.short_id.startswith(container_id):
                    node = n
                    break
        
        if not node:
            raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
        
        if node.status != "running":
            try:
                container = docker_client.containers.get(node.id)
                if container is not None:
                    container.restart()
                    await asyncio.sleep(5)
                logger.info(f"Restarting container: {node.id}")
                
                from node import update_node_status
                await update_node_status(node)
            except Exception as e:
                logger.error(f"Failed to restart container {node.id}: {str(e)}")
            
            raise HTTPException(
                status_code=503, detail=f"Container {container_id} is not running, restart attempted")
        
        if target_port == 8000:
            if node.port == 0:
                raise HTTPException(status_code=503, detail=f"Container {container_id} port 8000 not mapped")
            mapped_port = node.port
        elif target_port == 8088:
            if node.sse_port == 0:
                raise HTTPException(status_code=503, detail=f"Container {container_id} port 8088 not mapped")
            mapped_port = node.sse_port
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported target port: {target_port}")
        
        node.last_req_time = datetime.datetime.now()
        await node.replace()
        
        if target_port == 8088:
            url = f"http://{node.ip}:{mapped_port}/mcp"
        else:
            url = f"http://{node.ip}:{mapped_port}/{path}"
        
        logger.info(f"Routing {request.method} request to container {container_id}:{target_port} -> {url}")
        try:
            headers = node_proxy_headers(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        timeout = httpx.Timeout(120.0, connect=10.0)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                if request.method in ["POST", "PUT", "PATCH"]:
                    data = await request.body()
                    response = await client.stream(
                        request.method, url, data=data, headers=headers
                    ).__aenter__()
                else:
                    params = request.query_params
                    if params:
                        url += '?' + urllib.parse.urlencode(params)
                    response = await client.stream(
                        request.method, url, headers=headers
                    ).__aenter__()
                
                status_code = response.status_code
                response_headers = dict(response.headers)
                
                content_type = response_headers.get('content-type', '').lower()
                if 'application/json' in content_type:
                    try:
                        content = await response.aread()
                        await response.aclose()
                        return Response(
                            content=content,
                            status_code=status_code,
                            headers=response_headers,
                            media_type=content_type
                        )
                    except Exception as e:
                        logger.error(f"Error reading JSON response from container {container_id}: {str(e)}")
                        await response.aclose()
                        raise HTTPException(status_code=502, detail=f"Error reading response: {str(e)}")
                else:
                    async def stream_response():
                        try:
                            async for chunk in response.aiter_bytes():
                                yield chunk
                        except Exception as e:
                            logger.error(f"Error streaming response from container {container_id}: {str(e)}")
                            yield f"Error communicating with container: {str(e)}".encode()
                        finally:
                            await response.aclose()
                    
                    return StreamingResponse(
                        stream_response(), 
                        status_code=status_code, 
                        headers=response_headers
                    )
                
            except httpx.RequestError as e:
                logger.error(f"Request error when connecting to container {container_id}: {str(e)}")
                raise HTTPException(status_code=503, detail=f"Error connecting to container: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error when routing to container {container_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in route_to_container_port: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get('/api/container/{container_id}/urls')
async def get_container_urls(request: Request, container_id: str):
    """Get container access URL information."""
    try:
        node = None
        
        node = await DockerContainerNode.find_one(DockerContainerNode.id == container_id)
        
        if not node:
            node = await DockerContainerNode.find_one(DockerContainerNode.short_id == container_id)
        
        if not node:
            async for n in DockerContainerNode.find_all():
                if n.id.startswith(container_id) or n.short_id.startswith(container_id):
                    node = n
                    break
        
        if not node:
            raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
        
        base_url = str(request.base_url).rstrip('/')
        
        urls = {
            "container_id": node.id,
            "short_id": node.short_id,
            "status": node.status,
            "direct_access": {
                "port_8000": f"http://{node.ip}:{node.port}" if node.port > 0 else None,
                "port_8088": f"http://{node.ip}:{node.sse_port}/mcp" if node.sse_port > 0 else None
            },
            "proxy_access": {
                "api_url": f"{base_url}/container/{node.short_id}" if node.port > 0 else None,
                "mcpapi_url": f"{base_url}/container/{node.short_id}/mcpapi" if node.port > 0 else None,
                "mcp_url": f"{base_url}/container/{node.short_id}/mcp" if node.sse_port > 0 else None
            }
        }
        
        return JSONResponse(content=urls)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting container URLs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

"""
AgentDock Node Management Module
Docker Container Node Manager for MCP Services
"""
import os
import asyncio
import time
import httpx
import datetime
import docker.types
import docker.errors
import tenacity
import io
import json
import urllib.parse

from typing import Optional, Dict, List, Any
from copy import deepcopy
from functools import partial
from fastapi import APIRouter, Request, HTTPException, Response, Cookie, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from config import CONFIG, MANAGER_NAME, NODE_API_TOKEN, logger, docker_client
from config import DockerContainerNode, DCNodeChecker
from auth import node_proxy_headers, security_configured

node_router = APIRouter()


async def shell_exec(command: str):
    """
    Execute a shell command.

    Args:
        command (str): the shell command to execute.

    Returns:
        str: the output of the command.
    """
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=os.path.dirname(__file__),
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise Exception(stderr.decode("utf-8"))
    return stdout


def find_available_port(start_port=48000, max_port=49000):
    """
    Find an available port.
    
    Args:
        start_port: Starting port number
        max_port: Maximum port number
        
    Returns:
        int: Available port number
    """
    import socket
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available port in range {start_port}-{max_port}")


async def check_nodes_status():
    """
    Check the status of all existing nodes from the database.
    If a node doesn't exist in Docker, it will be deleted from the database. 

    Raises:
        docker.errors.NotFound: Raised when a Node is not found in Docker
        docker.errors.APIError: Raised when it fails to get Node info from Docker
    """
    tasks = []
    async for node in DockerContainerNode.find_all():
        tasks.append(update_node_status(node))
    await asyncio.gather(*tasks)


@tenacity.retry(stop=tenacity.stop_after_attempt(3))
async def check_nodes_status_loop(checker: DCNodeChecker):
    """
    An infinite loop that checks the status of the nodes and waits before each iteration.
    """
    checker_id = checker.id
    logger.info("Nodes status checker started.")
    while True:
        try:
            await check_nodes_status()
        except:
            import traceback
            traceback.print_exc()
            
        checker = await DCNodeChecker.find_one(DCNodeChecker.id == checker_id)
        if checker is None:
            logger.info("Nodes status checker stopped.")
            return
        await asyncio.sleep(CONFIG.node.health_check_interval)


def create_container(image_name: Optional[str] = None, port: Optional[int] = None, sse_port: Optional[int] = None):
    """
    Create a Docker container with the specified configuration.
    
    Args:
        image_name: Docker image name (optional)
        port: Main port mapping (optional, auto-assign if None)
        sse_port: SSE port mapping (optional, auto-assign if None)
        
    Returns:
        Docker container object
    """
    cfg = deepcopy(CONFIG.node.creation_kwargs)
    selected_image = image_name or cfg.get("image")
    if not security_configured():
        raise HTTPException(
            status_code=503,
            detail="AgentDock API tokens are not configured or are not distinct",
        )
    if selected_image not in CONFIG.node.allowed_images:
        raise HTTPException(status_code=400, detail=f"Image is not allowed: {selected_image}")
    cfg["image"] = selected_image
    cfg["privileged"] = False
    cfg["cap_drop"] = ["ALL"]
    cfg.pop("cap_add", None)
    cfg.pop("devices", None)
    cfg["security_opt"] = ["no-new-privileges:true"]
    environment = cfg.setdefault("environment", {})
    if isinstance(environment, dict):
        environment.pop("AGENTDOCK_API_TOKEN", None)
        environment["AGENTDOCK_NODE_API_TOKEN"] = NODE_API_TOKEN
    else:
        cfg["environment"] = [
            item
            for item in environment
            if not item.startswith(("AGENTDOCK_API_TOKEN=", "AGENTDOCK_NODE_API_TOKEN="))
        ]
        cfg["environment"].append(f"AGENTDOCK_NODE_API_TOKEN={NODE_API_TOKEN}")
    
    if 'ports' not in cfg:
        cfg['ports'] = {}
        
    # Configure port mapping for MCP server images
    if image_name and ("sse-mcpserver" in image_name or "mcpserver" in image_name):
        if 'ports' not in cfg:
            cfg['ports'] = {}
        cfg['ports'] = {
            8000: port,
            8088: sse_port
        }
        if port or sse_port:
            logger.info(f"Configured port mapping for {image_name}: 8000->{port}, 8088->{sse_port}")
        else:
            logger.info(f"Configured auto port mapping for {image_name}")
        
        # Configure volumes for MinerU images (v0.4 and v0.5)
        if "v0.4" in image_name or "v0.5" in image_name or "v0-4" in image_name or "v0-5" in image_name:
            import os
            if 'volumes' not in cfg:
                cfg['volumes'] = {}
            
            # Volume paths from environment variables
            mineru_volumes = {}
            
            mineru_source = os.environ.get('MINERU_SOURCE_PATH')
            if mineru_source:
                mineru_volumes[mineru_source] = {'bind': '/app/mineru_source', 'mode': 'rw'}
            
            models_cache = os.environ.get('MODELS_CACHE_PATH')
            if models_cache:
                mineru_volumes[models_cache] = {'bind': '/app/external_models_cache', 'mode': 'ro'}
            
            mineru_config = os.environ.get('MINERU_CONFIG_PATH')
            if mineru_config:
                mineru_volumes[mineru_config] = {'bind': '/root/mineru.json', 'mode': 'ro'}
            
            dataset_path = os.environ.get('DATASET_PATH')
            if dataset_path:
                mineru_volumes[dataset_path] = {'bind': '/app/dataset', 'mode': 'rw'}
            
            # Only mount paths that exist
            for host_path, mount_config in mineru_volumes.items():
                if os.path.exists(host_path):
                    cfg['volumes'][host_path] = mount_config
                    logger.info(f"Added mount: {host_path} -> {mount_config['bind']}")
                else:
                    logger.warning(f"Mount path does not exist, skipping: {host_path}")
            
            # GPU environment variables
            if 'environment' not in cfg:
                cfg['environment'] = {}
            
            mineru_env = {
                'MODELSCOPE_CACHE': '/app/models_cache',
                'HF_HOME': '/app/.cache/huggingface',
                'MINERU_MODEL_SOURCE': 'local',
                'CUDA_VISIBLE_DEVICES': '0',
                'NVIDIA_VISIBLE_DEVICES': 'all',
                'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility'
            }
            cfg['environment'].update(mineru_env)
            logger.info(f"Configured volumes and environment for MinerU image {image_name}")
    else:
        # For other images, configure dual port mapping
        cfg['ports'] = {
            8000: port,
            8088: sse_port
        }
        if port and sse_port:
            logger.info(f"Configured fixed ports for {image_name or cfg.get('image', 'default')}: 8000->{port}, 8088->{sse_port}")
        elif port:
            logger.info(f"Configured fixed main port for {image_name or cfg.get('image', 'default')}: 8000->{port}")
    
    # Add GPU device request for MinerU images
    device_requests = None
    if image_name and ("v0.4" in image_name or "v0.5" in image_name or "v0-4" in image_name or "v0-5" in image_name):
        device_requests = [docker.types.DeviceRequest(device_ids=["0"], capabilities=[["gpu"]])]
        logger.info(f"Added GPU device request for MinerU image {image_name}")
    elif CONFIG.node.device_requests:
        device_requests = [docker.types.DeviceRequest(**req) for req in CONFIG.node.device_requests]
    
    return docker_client.containers.run(
        device_requests=device_requests,
        **cfg,
    )


async def wait_for_node_startup(node_id: str):
    """
    Wait for the startup of node with id node_id.

    Args:
        node_id (str): The unique identifier of the node.

    Returns:
        DockerContainerNode: Node object if started successfully, None if timeout.

    Raises:
        HTTPException: If node is not found in the database.
    """
    MAX_PROBE_TIMES = CONFIG.node.creation_wait_seconds
    probe_times = 0

    # Initial wait time for container startup
    await asyncio.sleep(5)
    
    t = time.time()
    error = None
    
    while probe_times < MAX_PROBE_TIMES:
        try:
            node = await DockerContainerNode.find_one(
                DockerContainerNode.id == node_id, 
                DockerContainerNode.creator_id == MANAGER_NAME
            )

            if node is None:
                raise HTTPException(
                    status_code=503, detail="Failed to detect node status! Node not found in db!")
            
            # Check if container is running
            container = docker_client.containers.get(node_id)
            if container.status != "running":
                logger.info(f"Node {node_id} not yet running, waiting...")
                await asyncio.sleep(2)
                continue
                
            # Return node if healthy or health check disabled
            if node.health == 'healthy' or not CONFIG.node.health_check:
                return node
            
            # HTTP health check for MCP server images
            if "sse-mcpserver" in node.image_name or "mcpserver" in node.image_name:
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        url = f"http://{node.ip}:{node.port}/alive"
                        logger.info(f"Testing node availability: {url}")
                        resp = await client.get(url)
                        if resp.status_code == 200:
                            logger.info(f"Node {node_id} is responsive")
                            node.health = 'healthy'
                            await node.replace()
                            return node
                except Exception as e:
                    logger.info(f"Node not yet ready: {str(e)}")
        except Exception as e:
            logger.warning(f"Error during node startup check: {str(e)}")
        
        probe_times += 1
        await asyncio.sleep(1)
    
    logger.error(f"Node startup timeout: {node_id}, error: {error}")
    return None


async def update_node_status(node: DockerContainerNode):
    """
    Update the status of a node in the database.

    Args:
        node (DockerContainerNode): The node to be updated.
    """
    try:
        container = await asyncio.to_thread(docker_client.containers.get, node.id)
    except docker.errors.NotFound:
        await node.delete()
        logger.info("Node deleted from db: " + node.id + '(not in docker)')
        return
    except docker.errors.APIError:
        logger.warning("Failed to get node info from docker: " + node['node_id'])
        return
    
    node_status = container.attrs["State"]["Status"]
    current_time = datetime.datetime.now()
    
    # Record start time when node transitions to running state
    if node_status == "running" and node.status != "running":
        node.start_time = current_time
        logger.info(f"Node {node.short_id} start time recorded: {current_time.isoformat()}")

    if node_status != node.status:
        logger.info(f"Node {node.short_id} status updated: " + node.status + " -> " + node_status)
    node.status = node_status
    
    # Update image name
    image_name = container.attrs["Config"]["Image"]
    if image_name != node.image_name:
        node.image_name = image_name
        logger.info(f"Node {node.short_id} image updated: {image_name}")
    
    if node.status == "running":
        node.ip = os.environ.get("DOCKER_HOST_IP", "localhost")
        
        # Get main port mapping
        if f'{CONFIG.node.port}/tcp' in container.attrs['NetworkSettings']['Ports']:
            current_port = int(container.attrs['NetworkSettings']['Ports'][f'{CONFIG.node.port}/tcp'][0]["HostPort"])
            node.port = current_port
            
            # Persist port on first assignment
            if node.stored_port is None:
                node.stored_port = current_port
                logger.info(f"Node {node.short_id} first assigned main port: {current_port}")
            elif node.stored_port != current_port:
                logger.warning(f"Node {node.short_id} port changed! stored: {node.stored_port}, current: {current_port}")
                logger.info(f"Node {node.short_id} detected port drift, use /node/reconnect to restore fixed port {node.stored_port}")
        
        # Get SSE port mapping (port 8088)
        if '8088/tcp' in container.attrs['NetworkSettings']['Ports']:
            current_sse_port = int(container.attrs['NetworkSettings']['Ports']['8088/tcp'][0]["HostPort"])
            node.sse_port = current_sse_port
            
            if node.stored_sse_port is None:
                node.stored_sse_port = current_sse_port
                logger.info(f"Node {node.short_id} first assigned SSE port: {current_sse_port}")
        else:
            logger.info(f"Node {node.short_id} ports: {container.attrs['NetworkSettings']['Ports']}")
            node.sse_port = 0
        
        if CONFIG.node.health_check:
            health = container.attrs['State']['Health']['Status']
            if health != node.health:
                logger.info(f"Node {node.short_id} health updated: " + node.health + " -> " + health)
            node.health = health
        else:
            node.health = 'healthy' if node.status else 'unhealthy'
        
    node = await node.replace()
    
    # Idle timeout check
    if node_status == "running":
        min_runtime_minutes = 5
        is_new_node = False
        
        if node.start_time is None:
            node.start_time = current_time
            await node.replace()
            is_new_node = True
        else:
            is_new_node = (current_time - node.start_time).total_seconds() < min_runtime_minutes * 60
        
        # Stop node if idle timeout exceeded
        if not is_new_node and (current_time - node.last_req_time >= datetime.timedelta(minutes=CONFIG.node.idling_close_minutes)):
            logger.info(f"Node {node.short_id} has been idle for {CONFIG.node.idling_close_minutes} minutes")
            container.stop()
            logger.info("Stopping node: " + node.id + " due to idle timeout")


@node_router.get("/details")
async def node_detail(node_id: str = Cookie(None), node_id_q: str = Query(None)):
    """
    Fetch node details from the database.

    Args:
        node_id (str, optional): Node identifier from cookie.
        node_id_q (str, optional): Node identifier from query parameter.

    Returns:
        JSONResponse: Node details.
    """
    if node_id_q is not None:
        node_id = node_id_q
    node = await DockerContainerNode.find_one(DockerContainerNode.id == node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return JSONResponse(content={
        "id": node.id,
        "short_id": node.short_id,
        "ip": node.ip,
        "port": node.port,
        "sse_port": node.sse_port,
        "stored_port": node.stored_port,
        "stored_sse_port": node.stored_sse_port,
        "image_name": node.image_name,
        "creator_name": node.creator_name,
        "status": node.status,
        "health": node.health,
        "last_req_time": node.last_req_time.isoformat(),
        "start_time": node.start_time.isoformat() if node.start_time else None,
        "uptime_minutes": round((datetime.datetime.now() - node.start_time).total_seconds() / 60) if node.start_time else 0
    })


@node_router.post("/create")
async def create_node(image_name: Optional[str] = None, creator_name: Optional[str] = None):
    """
    Create a new Docker container node.

    Args:
        image_name: Docker image name (optional)
        creator_name: Creator name (optional)

    Returns:
        JSONResponse: Response with node creation details.

    Raises:
        HTTPException: If node creation timeout or user exceeds container limit.
    """
    # Check container limit per user
    if creator_name:
        existing_nodes = await DockerContainerNode.find(
            DockerContainerNode.creator_name == creator_name,
            DockerContainerNode.status != "exited"
        ).count()
        
        MAX_CONTAINERS_PER_USER = 3
        if existing_nodes >= MAX_CONTAINERS_PER_USER:
            logger.warning(f"User {creator_name} has reached maximum container limit ({MAX_CONTAINERS_PER_USER})")
            raise HTTPException(
                status_code=400,
                detail=f"User {creator_name} has reached maximum container limit ({MAX_CONTAINERS_PER_USER}). Please release existing containers first."
            )
        
        logger.info(f"User {creator_name} has {existing_nodes} containers, creating new one...")
    
    container = None
    cost_times = {}
    t = time.time()
    
    # Pre-allocate fixed ports
    allocated_port = find_available_port()
    allocated_sse_port = find_available_port(start_port=allocated_port + 1)
    logger.info(f"Pre-allocated fixed ports for new container: {allocated_port}, {allocated_sse_port}")
    
    # Create docker container with fixed ports
    container = await asyncio.to_thread(
        partial(create_container, image_name=image_name, port=allocated_port, sse_port=allocated_sse_port),
    )
    
    cost_times["docker_run"] = time.time()-t
    
    t = time.time()
    logger.info("Node created: " + container.id)
    container.reload()
    cost_times["reload_container"] = time.time()-t
    
    t = time.time()    
    node = DockerContainerNode(
        id=container.id,
        short_id=container.short_id,
        creator_id=MANAGER_NAME,
        creator_name=creator_name,
        last_req_time=datetime.datetime.now(),
        start_time=datetime.datetime.now(),
        image_name=container.attrs["Config"]["Image"] if image_name is None else image_name
    )
    node = await node.insert()
    cost_times["insert_node"] = time.time()-t

    t = time.time()
    
    # Wait for node startup
    created_node = await wait_for_node_startup(node.id)
    if created_node is not None:
        logger.info("Node startup success: " + created_node.id)
        cost_times["wait_for_node_startup"] = time.time()-t
        logger.info(cost_times)
        
        image_info = image_name if image_name else CONFIG.node.creation_kwargs["image"]
        content = {
            "message": f"Node {created_node.short_id} created!", 
            "Image": image_info, 
            "Creator": creator_name if creator_name else MANAGER_NAME
        }
        response = JSONResponse(content=content)
        response.headers["Server"] = "AgentDock/" + CONFIG.version
        response.set_cookie(key="node_id", value=container.id)
        response.set_cookie(key="node_ip", value=created_node.ip)
        response.set_cookie(key="node_port", value=created_node.port)
        return response
    else:
        logger.warning("Node status detection timeout: " + node.id)
        raise HTTPException(status_code=503, detail="Node creation timeout!")


@node_router.post("/reconnect")
async def reconnect(node_id: str = Cookie(None), node_id_q: str = Query(None)):
    """
    Reconnect session of a node. Restarts the node if it exists.
    If the container has been deleted, recreates it using stored ports.

    Args:
        node_id (str, optional): Node identifier from cookie.
        node_id_q (str, optional): Node identifier from query parameter.

    Returns:
        str: Success message if node restarts successfully.

    Raises:
        HTTPException: If node restart timeout occurs.
    """
    if node_id_q is not None:
        node_id = node_id_q
    node = await DockerContainerNode.find_one(DockerContainerNode.id == node_id)
    if node is None:
        return "invalid node_id: " + str(node_id)
    
    try:
        container = docker_client.containers.get(node_id)
        container.restart()
        logger.info("Node restarted: " + node_id)
    except docker.errors.NotFound:
        logger.info(f"Container {node_id} not found, recreating with stored ports: port={node.stored_port}, sse_port={node.stored_sse_port}")
        
        container = await asyncio.to_thread(
            partial(create_container, 
                   image_name=node.image_name, 
                   port=node.stored_port, 
                   sse_port=node.stored_sse_port),
        )
        
        old_id = node.id
        node.id = container.id
        node.short_id = container.short_id
        await node.replace()
        logger.info(f"Node recreated with new ID: {container.id}, using stored ports")

    if await wait_for_node_startup(node.id):
        return "Reconnect session: " + str(node.id)
    else:
        logger.warning("Node restart timeout: " + node.id)
        raise HTTPException(status_code=503, detail="Node restart timeout!")


@node_router.post("/close")
async def close(node_id: str = Cookie(None), node_id_q: str = Query(None)):
    """
    Close session of a node. Stops the node if running.

    Args:
        node_id (str, optional): Node identifier from cookie.
        node_id_q (str, optional): Node identifier from query parameter.

    Returns:
        str: Success message if node stops successfully.
    """
    if node_id_q is not None:
        node_id = node_id_q
        
    node = await DockerContainerNode.find_one(DockerContainerNode.id == node_id)
    if node is None:
        return "invalid node_id: " + str(node_id)
    
    container = docker_client.containers.get(node_id)
    if container is not None and container.attrs["State"]["Status"] != "exit":
        container.stop()
        logger.info("Node stopped: " + node_id)
        await update_node_status(node)
    return "Close session: " + str(node_id)


@node_router.post("/release")
async def release(node_id: str = Cookie(None), node_id_q: str = Query(None)):
    """
    Release session of a node. Kills and removes the container.

    Args:
        node_id (str, optional): Node identifier from cookie.
        node_id_q (str, optional): Node identifier from query parameter.

    Returns:
        str: Success message if node is released.
    """
    if node_id_q is not None:
        node_id = node_id_q
    node = await DockerContainerNode.find_one(DockerContainerNode.id == node_id)
    if node is None:
        return "invalid node_id: " + str(node_id)

    container = docker_client.containers.get(node_id)
    if container is not None:
        if container.attrs["State"]["Status"] != "exited":
            container.kill()
            logger.info("Node killed: " + node_id)
        container.remove()
        await update_node_status(node)
    return "Release session: " + str(node_id)


@node_router.post("/delete_all")
async def delete_all():
    """
    Delete all nodes.

    Returns:
        str: Success message with count of deleted nodes.
    """
    tasks = []
    async for node in DockerContainerNode.find_all():
        tasks.append(release(node.id, node.id))
    await asyncio.gather(*tasks)
    return f"Release all {len(tasks)} sessions!"


@node_router.api_route("/{node_ip}/{node_port}/{path:path}", methods=["GET", "POST"])
async def route_to_node(request: Request, node_ip: str, node_port: int, path: str):
    """
    Routes a request to a specific node.

    Args:
        request (Request): The request object.
        node_ip (str): The IP address of the node.
        node_port (int): The port of the node.
        path (str): The path to route the request to.

    Returns:
        Response: The response from the node.

    Raises:
        HTTPException: If node is not found or not responding.
    """
    logger.info(f"Route to {node_ip}:{node_port}, path: {path}")
    
    node = None
    try:
        async for n in DockerContainerNode.find(DockerContainerNode.port == node_port, DockerContainerNode.ip == node_ip):
            node = n
            break
            
        if not node:
            raise HTTPException(status_code=404, detail=f"No node found with IP {node_ip} and port {node_port}")
                
        if node.status != "running":
            try:
                container = docker_client.containers.get(node.id)
                if container is not None:
                    container.restart()
                    await asyncio.sleep(5)
                logger.info("Restarting node: " + node.id)
                await update_node_status(node)
            except Exception as e:
                logger.error(f"Failed to restart node {node.id}: {str(e)}")
            
            raise HTTPException(
                status_code=503, detail="Node is not running and restart attempt was made")

        node.last_req_time = datetime.datetime.now()
        node = await node.replace()

        url = f"http://{node.ip}:{node.port}/{path}"
        logger.info("Request url: " + url)
        try:
            headers = node_proxy_headers(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        
        timeout = httpx.Timeout(10.0, connect=5.0)
        client = httpx.AsyncClient(timeout=timeout)
        
        try:
            if request.method == "POST":
                data = await request.body()
                response = await client.stream("POST", url, data=data, headers=headers).__aenter__()
            else:
                params = request.query_params
                if params:
                    url += '?' + urllib.parse.urlencode(params)
                response = await client.stream("GET", url, headers=headers).__aenter__()
            
            status_code = response.status_code
            headers = dict(response.headers)
            
            async def stream_response():
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                except Exception as e:
                    logger.error(f"Error streaming response from node {node.id}: {str(e)}")
                    yield f"Error communicating with node: {str(e)}".encode()
                finally:
                    await client.aclose()
            
            return StreamingResponse(stream_response(), status_code=status_code, headers=headers)
            
        except httpx.RequestError as e:
            logger.error(f"Request error when connecting to node {node.id}: {str(e)}")
            await client.aclose()
            raise HTTPException(status_code=503, detail=f"Error connecting to node: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error when routing to node {node.id}: {str(e)}")
            await client.aclose()
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in route_to_node: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

    raise HTTPException(status_code=404, detail="node_ip: " + str(node_ip) + ' and node_port: ' + str(node_port) + ' is not found')


@node_router.get("/images")
async def list_images():
    """
    List available Docker images for node creation.
    Excludes agentdock-manager and mongodb images.

    Returns:
        JSONResponse: List of available images.
    """
    images = []
    for image in docker_client.images.list():
        if image.tags:
            for tag in image.tags:
                if not tag.startswith("agentdock-manager:") and not tag.startswith("mongo:"):
                    images.append(tag)
    
    return JSONResponse(content={"images": images})


templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


def node_root(request: Request, node: DockerContainerNode):
    """
    Serve the node root page.

    Args:
        node (DockerContainerNode): the node to serve the root page for.
    """
    return templates.TemplateResponse("node.html", {
        "request": request,
        "BASE_URL": '/node/{}/{}'.format(node.ip, node.port),
    })

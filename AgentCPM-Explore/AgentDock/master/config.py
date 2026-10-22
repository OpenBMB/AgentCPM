"""
AgentDock Configuration Module
Docker Container Node Manager for MCP Services
"""
import os
import docker
import pydantic
import logging
from typing import Dict, Any, Union, Optional
import datetime

from motor.motor_asyncio import AsyncIOMotorClient
from motor.core import AgnosticDatabase
from beanie import Document, Indexed


class AgentDockConfig(pydantic.BaseModel, extra='forbid'):
    """
    AgentDock configuration class.
    Configuration settings are initially loaded from a yaml file. 
    However, if an environment variable exists with the same name as a configuration setting, 
    the value from the environment variable will be used instead.
    """
    version: str = '1.0.0'
    
    class LoggingConfig(pydantic.BaseModel):
        logger: str = 'gunicorn.error'
        logger_level: int = logging.INFO
        
    loggingcfg: LoggingConfig = LoggingConfig()
    
    label: str = 'role=AgentDockManager'
    testingNodeLabel: str = 'role=AgentDockTester'
    
    class MongoDBConfig(pydantic.BaseModel):
        host: str = os.environ.get("MONGO_HOST", 'localhost')
        port: int = int(os.environ.get("MONGO_PORT", 27017))
        username: str = os.environ.get("MONGO_USER", '')
        password: str = os.environ.get("MONGO_PASS", '')
        collection: str = os.environ.get("MONGO_COLLECTION", 'AgentDock')
        
    mongodb: MongoDBConfig = MongoDBConfig()
    
    class NodeConfig(pydantic.BaseModel):
        label: str = 'role=AgentDockNode'
        port: int = 8000
        yadb_file: str = os.path.join(os.path.dirname(__file__), 'yadb')
        stop_after_exit: bool = False
        creation_wait_seconds: int = 300
        idling_close_minutes: int = 14400  # Idle timeout in minutes
        health_check_interval: int = 1
        health_check: bool = True
        creation_kwargs: Dict[str, Any] = {}
        allowed_images: list[str] = []
        device_requests: Any | None = None
    
    node: NodeConfig = NodeConfig()

    @classmethod
    def from_yaml(cls, yaml_stream: str | os.PathLike) -> "AgentDockConfig":
        import yaml
        if os.path.exists(yaml_stream):
            with open(yaml_stream, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        elif isinstance(yaml_stream, str):
            config = yaml.load(yaml_stream, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError(f"File {yaml_stream} not found.")
        return AgentDockConfig(**config)


# Backward compatibility alias
ManagerConfig = AgentDockConfig

CONFIG = AgentDockConfig.from_yaml(os.environ.get('CONFIG_FILE', '/app/config.yml'))
API_TOKEN = os.environ.get("AGENTDOCK_API_TOKEN", "").strip()
NODE_API_TOKEN = os.environ.get("AGENTDOCK_NODE_API_TOKEN", "").strip()
logger = logging.getLogger(CONFIG.loggingcfg.logger)
logger.setLevel(CONFIG.loggingcfg.logger_level)


# Obtain manager container id
MANAGER_NAME = os.uname().nodename

if MANAGER_NAME is None or MANAGER_NAME == "":
    logger.error("Failed to obtain AgentDock container id!")
    os._exit(1)
else:
    logger.info("AgentDock container id: " + MANAGER_NAME)


def setup_mongodb():
    """Initialize MongoDB connection."""
    db_url = f"mongodb://{CONFIG.mongodb.username}:{CONFIG.mongodb.password}@{CONFIG.mongodb.host}:{CONFIG.mongodb.port}/"
    mongo_client = AsyncIOMotorClient(db_url)
    db: AgnosticDatabase = mongo_client[CONFIG.mongodb.collection]
    return db


db = setup_mongodb()
logger.info("Database connected")


# Docker client for container management
docker_client = docker.from_env()
logger.info("Docker client connected")


class DockerContainerNode(Document):
    """Represents a container node in the database."""
    id: str
    short_id: str
    creator_id: str
    creator_name: Optional[str] = None
    image_name: Optional[str] = None
    ip: str = "localhost"
    port: int = 8000
    sse_port: int = 0  # Streamable-HTTP MCP service port
    stored_port: Optional[int] = None  # Persisted main port
    stored_sse_port: Optional[int] = None  # Persisted SSE port
    status: str = "unknown"
    health: str = "unknown"
    last_req_time: datetime.datetime = datetime.datetime.now()
    start_time: Optional[datetime.datetime] = None
    width: int = 0
    height: int = 0
    video: bool = False


class DCNodeChecker(Document):
    """Node health checker document."""
    creator_id: Indexed(str, unique=True)
    pid: Optional[int] = None
    interval: float

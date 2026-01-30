"""
Agent configuration loading module.

Loads agent configurations from YAML file and provides access to browse_agent and scorer_agent configs.
"""
import yaml
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Global cache for agent config
_agent_config_cache: Optional[Dict[str, Any]] = None
_agent_config_path: Optional[str] = None


def load_agent_config(config_path: str) -> Dict[str, Any]:
    """
    Load agent configuration from YAML file.
    
    Args:
        config_path: Path to the agent configuration YAML file
        
    Returns:
        Dictionary containing agent configurations (browse_agent, scorer_agent)
    """
    global _agent_config_cache, _agent_config_path
    
    # Return cached config if path hasn't changed
    if _agent_config_cache is not None and _agent_config_path == config_path:
        return _agent_config_cache
    
    # Resolve path
    config_path_str = str(config_path)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        # Try relative to current working directory first
        cwd_config = Path.cwd() / config_path
        if cwd_config.exists():
            config_path = cwd_config
        else:
            # Try relative to project root (assuming agent_config.py is in src/rollout/mcp/context/)
            # From src/rollout/mcp/context/agent_config.py to project root is 5 levels up
            project_root = Path(__file__).parent.parent.parent.parent.parent
            project_config = project_root / config_path_str
            if project_config.exists():
                config_path = project_config
            else:
                # Last resort: try as-is (might be relative to cwd)
                config_path = Path(config_path_str)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Agent config file not found: {config_path}. Tried: {config_path_str}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise ValueError(f"Invalid agent config file format: expected dict, got {type(config)}")
        
        # Validate required sections
        if "browse_agent" not in config:
            raise ValueError("Missing 'browse_agent' section in agent config")
        if "scorer_agent" not in config:
            raise ValueError("Missing 'scorer_agent' section in agent config")
        
        # Validate browse_agent
        browse_agent = config["browse_agent"]
        if "models" not in browse_agent or not isinstance(browse_agent["models"], list):
            raise ValueError("browse_agent must have a 'models' list")
        if len(browse_agent["models"]) == 0:
            raise ValueError("browse_agent.models list cannot be empty")
        
        # Validate optional extra_body if present
        if "extra_body" in browse_agent and not isinstance(browse_agent["extra_body"], dict):
            raise ValueError("browse_agent.extra_body must be a dictionary if provided")
        
        # Validate scorer_agent
        scorer_agent = config["scorer_agent"]
        if "models" not in scorer_agent or not isinstance(scorer_agent["models"], list):
            raise ValueError("scorer_agent must have a 'models' list")
        if len(scorer_agent["models"]) == 0:
            raise ValueError("scorer_agent.models list cannot be empty")
        
        # Cache the config
        _agent_config_cache = config
        _agent_config_path = str(config_path)
        
        logger.info(f"Loaded agent config from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load agent config from {config_path}: {e}")


def get_browse_agent_config(config_path: str) -> Dict[str, Any]:
    """
    Get browse agent configuration.
    
    Args:
        config_path: Path to the agent configuration YAML file
        
    Returns:
        Dictionary containing browse_agent configuration
    """
    config = load_agent_config(config_path)
    return config["browse_agent"]


def get_scorer_agent_config(config_path: str) -> Dict[str, Any]:
    """
    Get scorer agent configuration.
    
    Args:
        config_path: Path to the agent configuration YAML file
        
    Returns:
        Dictionary containing scorer_agent configuration
    """
    config = load_agent_config(config_path)
    return config["scorer_agent"]


def reset_config_cache():
    """Reset the global config cache. Useful for testing or reloading config."""
    global _agent_config_cache, _agent_config_path
    _agent_config_cache = None
    _agent_config_path = None


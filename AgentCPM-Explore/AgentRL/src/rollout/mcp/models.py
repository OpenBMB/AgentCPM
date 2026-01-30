"""
MCP-related data models for rollout.
"""
from typing import Optional, Dict, Any
from databases import Task, Record, register_new_model


@register_new_model
class MCPTask(Task):
    """Task model for MCP tool-using tasks."""
    system_prompt: str = ""
    query_prompt: str = ""
    answer_schema: str = "answer"
    task_answer: Dict[str, Any] = {}
    scorer: str = "agentcpm"
    task_id: Optional[str] = None
    turns: Optional[list[int]] = []


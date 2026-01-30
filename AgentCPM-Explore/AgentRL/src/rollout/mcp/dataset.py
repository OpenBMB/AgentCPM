"""
Dataset for MCP tool-using tasks.
"""
import os
import json
import random
from typing import List, Optional
from torch.utils.data import Dataset
from .models import MCPTask
import uuid


MIXED_PROMPT = """You are a deep research assistant. Your core function is to think step by step and conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final  within <answer></answer> tags.
Date today is 2026-01-26.
"""

# MIXED_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. 
# You must handle both broad, open-domain inquiries and queries within specialized academic fields, as well as short, single-step puzzle, logic, or math questions.

# For questions that can be solved purely by internal reasoning (e.g., closed-form math problems, self-contained logic or puzzle questions that do not depend on external facts), you should:
# - Rely on your own step-by-step reasoning.
# - Carefully check arithmetic (compute digit by digit when needed).

# For all other knowledge-intensive questions, synthesize information from credible, diverse sources to deliver a 
# comprehensive, accurate, and objective response.

# When you have gathered sufficient information and are ready to provide the definitive response, you must form a clear report and enclose it within <answer></answer> tags."""


MCP_USER_PROMPT = """Your task is to answer the user's question: {query}.
"""


class MCPDataset(Dataset):
    """Dataset for MCP tool-using tasks."""
    
    def __init__(self, train_file: str, split: str = "train"):
        """
        Initialize the MCP dataset.
        
        Args:
            train_file: Path to the training file (JSON or JSONL)
            split: Dataset split ("train", "valid", "test")
        """
        self.split = split
        self.data = []
        
        if os.path.exists(train_file):
            if train_file.endswith('.jsonl'):
                with open(train_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self.data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            else:
                with open(train_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                    if not isinstance(self.data, list):
                        self.data = [self.data]
        else:
            # Try to load from HuggingFace datasets
            try:
                import datasets
                ds = datasets.load_dataset(train_file, split=split)
                self.data = [item for item in ds]
            except Exception as e:
                raise ValueError(f"Cannot load dataset from {train_file}: {e}")
        
        # Shuffle data with seed
        random.seed(33)
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        # Extract query from item (similar to extract_sample_params)
        query = (
            item.get("q") or item.get("Question") or item.get("question") or item.get("query") or 
            item.get("content") or item.get("instruction") or item.get("traj_question") or ""
        )
        # Try to extract from messages in item
        if not query and "messages" in item:
            for msg in item["messages"]:
                if msg.get("role") == "user":
                    query = msg["content"]
                    break
        # Try to extract from raw.messages if query still not found
        if not query and "raw" in item and isinstance(item.get("raw"), dict):
            raw = item["raw"]
            if "messages" in raw:
                for msg in raw["messages"]:
                    if msg.get("role") == "user":
                        query = msg["content"]
                        break
        if not query:
            query = "query field not found"
        
        # Extract answer and build task_answer dict
        answer_value = (
            item.get("a") or item.get("Final answer") or item.get("Final Answer") or 
            item.get("Final sample answer") or item.get("answer") or 
            item.get("expected_answer") or None
        )
        # Try to extract from raw.expected_answer if answer still not found
        if answer_value is None and "raw" in item and isinstance(item.get("raw"), dict):
            raw = item["raw"]
            answer_value = raw.get("expected_answer") or raw.get("answer") or None
        if answer_value is not None:
            answer_value = str(answer_value).strip()
        
        task_answer = {
            "answer": answer_value,
            "aug_answer": item.get("aug_answers", None),
            "query": item.get("Question") or item.get("question") or item.get("traj_question") or query
        }
        
        # Use MIXED_PROMPT as system_prompt and MCP_USER_PROMPT for query_prompt
        system_prompt = item.get("system_prompt", MIXED_PROMPT)
        query_prompt = item.get("query_prompt", MCP_USER_PROMPT.format(query=query))
        
        # Get other fields
        answer_schema = item.get("answer_schema", "answer")
        scorer = item.get("scorer", "agentcpm")
        
        # Convert task_id to string if it exists
        task_id = item.get("id", item.get("task_id"))
        if task_id is not None:
            task_id = str(task_id)
        
        return MCPTask(
            split=self.split,
            system_prompt=system_prompt,
            query_prompt=query_prompt,
            answer_schema=answer_schema,
            task_answer=task_answer,
            scorer=scorer,
            description=query,
            task_id=task_id
        )


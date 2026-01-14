#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GAIA API Test (AgentCPM-MCP Version)

This version uses MCPManager to interact with MCP servers
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import random
import time
import copy
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from concurrent.futures import ThreadPoolExecutor
import re
import tiktoken
from transformers import AutoTokenizer
import copy
import traceback
import hashlib
import uuid
from datetime import timezone

from pymongo import MongoClient, UpdateOne
from bson.json_util import loads as bson_loads 


file_path = Path(__file__).resolve()
script_dir = file_path.parent
src_dir = script_dir.parent
project_root = src_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(script_dir))  


from src.mcp_manager import MCPManager

from src.extended_openai_client import LLMClientManager 

from browser_processor import process_browser_tool_return
from context.historyx import HistoryX
from context.models import Procedure
from context.prompts import PROCEDURAL_MEMORY_SYSTEM_PROMPT
from context_processor import generate_procedure_summary


from utils.gaia_dataset import GaiaDataset
from utils.gaia_api_test_utils import (
    visualize_conversation_history, 
    format_gaia_dialog, 
    save_conversation_text,
    save_meta_config,
    save_tool_calls,
    save_llm_interactions,
    save_batch_summary
)

MANAGE_CONTEXT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "manage_context",
        "description": "Execute memory management, process summarization, task planning, and dynamic context management.\n   Should be called at the start of tasks and after completion of each major step, and when needed.",
        "parameters": {
            "properties": {
                "plan_steps": {
                    "description": "Complete execution plan, please first plan a complete, clear, and minimal-step plan, then fill each step of the plan into plan_steps, and mark each step's completion status with [completed], [partially completed], or [failed]",
                    "items": {
                    "type": "string"
                    },
                    "title": "Plan Steps",
                    "type": "array"
                },
                "next_step_goal": {
                    "description": "Goal to be achieved in the next step",
                    "title": "Next Step Goal",
                    "type": "string"
                }
            },
            "required": [
            "plan_steps"
            ],
            "title": "manage_contextArguments",
            "type": "object"
            }
        }
    }


SYSTEM_PROMPT_QWEN_record = """
You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective resKEEponse. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# General Objective

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.

## Task Strategy

1. **Analyze the user's request** to clarify the task objective, break it down into clear sub-goals, and arrange them in logical order.
2. **If the task does not require tool use, think step by step and answer the user directly.**
3. **If the task requires tool use, develop a concise step-by-step plan** (e.g., 1., 2., 3.), with each step corresponding to a specific sub-goal, obey tool-use guidelines to solve the task.

## Tool-Use Guidelines
4. **Call only one tool per step**, prioritizing the tool that best advances the current sub-goal.
5. **After each tool call, stop responding immediately** and wait for user feedback or tool results. Do not assume results or continue analysis.
6. **Extract and summarize key information from tool results** to inform the next step.
7. **Adjust your plan promptly when new information or challenges arise**, ensuring all sub-goals are covered and nothing is missed.
8. **For key conclusions, you must cross-validate using multiple tools or methods** to ensure the accuracy and consistency of the answer.
9. **After you have verified the answer, output the final answer in the specified format**.

## Answer Format
- **Answers should be direct and concise**, preferably using single words, numbers with commas and unit, or brief phrases.
- **Strictly follow the format requirements**, wrapping the final answer in `<answer></answer>` tags.

**Your goal: Minimize unnecessary thinking, act decisively, continuously use tools to gather information, and cross-validate with multiple tools until you can confidently provide the most concise and accurate answer.**


# Tools

You may call one or more functions to assist with the user query. You are provided with functions:

{tools_description}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""

SYSTEM_PROMPT_QWEN = """
You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective resKEEponse. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query. You are provided with functions:

{tools_description}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""


logger = logging.getLogger("gaia_api_test")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

import re

def simplify_playwright_snapshot(snapshot: str) -> str:
    """
    Analyze the original Playwright accessibility snapshot and extract a concise, clean list of interactive elements for the LLM.
    Args:
        snapshot: Original multi-line string obtained from Playwright snapshot tool.
    Returns:
        A simplified string listing key interactive elements and their references.
    """
    if not isinstance(snapshot, str):
        return str(snapshot)

   
    INTERACTIVE_KEYWORDS = {
        "link", "button", "textbox", "combobox", "option",
        "checkbox", "radio", "search", "textarea", "menuitem"
    }

    simplified_lines = []
    
    
    title_match = re.search(r"Page Title:\s*(.*)", snapshot)
    url_match = re.search(r"Page URL:\s*(.*)", snapshot)
    
    if title_match and title_match.group(1).strip():
        simplified_lines.append(f"Page Title: {title_match.group(1).strip()}")
    if url_match and url_match.group(1).strip():
        simplified_lines.append(f"Page URL: {url_match.group(1).strip()}\n")

    simplified_lines.append("Interactive Elements:")
    
    found_elements = False
    for line in snapshot.splitlines():
        
        if "[ref=" not in line:
            continue

        
        ref_match = re.search(r'\[ref=([a-zA-Z0-9_]+)\]', line)
        if not ref_match:
            continue
        ref_id = ref_match.group(1)

        
        element_type = ""
        
        for keyword in INTERACTIVE_KEYWORDS:
            if re.search(rf'\b{keyword}\b', line, re.IGNORECASE):
                element_type = keyword.capitalize()
                break
        
       
        if not element_type:
            continue

        
        label_match = re.search(r'"(.*?)"', line)
        label = label_match.group(1).strip() if label_match else ""

        
        indent_match = re.match(r'^\s*', line)
        indent = indent_match.group(0) if indent_match else ""

        simplified_lines.append(f'{indent}- {element_type} [ref={ref_id}] "{label}"')
        found_elements = True

    if not found_elements:
        simplified_lines.append("  (No interactive elements found on this page)")

    return "\n".join(simplified_lines)


TOOL_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)

def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _stable_oid(seed: str) -> str:

    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return h[:24]

def _gen_oid() -> str:

    return uuid.uuid4().hex[:24]

def _strip_tool_response(text: str) -> str:
    if not text:
        return ""
    m = TOOL_RE.search(text)
    if m:
        return (m.group(1) or "").strip()
    return text.strip()

def _safe_json_loads(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not s:
        return x
    try:
        return json.loads(s)
    except Exception:
        return x

def _is_tool_response_user(msg: Dict[str, Any]) -> bool:
    if msg.get("role") != "user":
        return False
    c = msg.get("content") or ""
    return isinstance(c, str) and "<tool_response>" in c and "</tool_response>" in c

def _tool_call_block(name: str, args_obj: Any) -> str:
    payload = {"name": name, "arguments": args_obj}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>\n\n"

def _strip_token_outputs(obj: Any) -> Any:

    DROP_KEYS = {
        "logprobs", "tokens", "bytes", "top_logprobs", "token_logprobs", "text_offset",
        "completion_tokens_details", "prompt_tokens_details",
    }
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in DROP_KEYS:
                continue
            out[k] = _strip_token_outputs(v)
        return out
    if isinstance(obj, list):
        return [_strip_token_outputs(x) for x in obj]
    return obj

def _maybe_strip_token_outputs_from_json_string(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    t = s.strip()
    if not t:
        return s
    if not (t.startswith("{") or t.startswith("[")):
        return s
    try:
        parsed = json.loads(t)
    except Exception:
        return s
    cleaned = _strip_token_outputs(parsed)
    return json.dumps(cleaned, ensure_ascii=False)

def _find_last_assistant_index(dialog: List[Dict[str, Any]]) -> Optional[int]:
    for i in range(len(dialog) - 1, -1, -1):
        if dialog[i].get("role") == "assistant":
            return i
    return None

def _base_template(task_oid: str) -> Dict[str, Any]:

    now = _utc_now_iso_z()
    return {
        "_id": {"$oid": _gen_oid()},
        "_class_id": "DispatchedSamplingTask",
        "advantage": 0,
        "creat_time": {"$date": now},          
        "created_at_step": 1,
        "finish_time": {"$date": now},
        "is_minio_managed": False,
        "priority": 0,
        "req_type": "chatcompletions",
        "request": {
            "messages": [],
            "model": "train-model",
            "tools": []
        },
        "response": {
            "id": "chatcmpl-" + _gen_oid(),
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "",
                        "refusal": None,
                        "role": "assistant",
                        "annotations": None,
                        "audio": None,
                        "function_call": None,
                        "tool_calls": None,
                        "reasoning_content": None,
                    },
                    "matched_stop": None,
                }
            ],
            "created": int(time.time()),
            "model": "train-model",
            "object": "chat.completion",
            "service_tier": None,
            "system_fingerprint": None,
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0
            },
            "metadata": {"weight_version": "default"},
        },
        "sampled_from": "",
        "score": 0,
        "status": "completed",
        "task": {"$ref": "Task", "$id": {"$oid": task_oid}},
        "traj_id": 0,
    }

def convert_dialog_step_to_doc(
    dialog: List[Dict[str, Any]],
    *,
    step: int,
    traj_id: int,
    task_oid: str,
    model_name: str,
    tools: List[Dict[str, Any]],
    usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Core: Convert "dialog.json format message list so far" into a mongo-ready doc.
    - response = last assistant
    - request.messages = timeline except last assistant (including tool role conversion)
    - assistant tool_calls => insert <tool_call> blocks in content, and queue tool_response to match tool_call_id/name
    """
    out = _base_template(task_oid)
    now = _utc_now_iso_z()

    out["_id"] = {"$oid": _gen_oid()}
    out["creat_time"] = {"$date": now}
    out["finish_time"] = {"$date": now}
    out["created_at_step"] = step
    out["traj_id"] = traj_id
    out["request"]["model"] = model_name
    out["response"]["model"] = model_name
    out["request"]["tools"] = tools

    last_ai = _find_last_assistant_index(dialog)
    final_ai = dialog[last_ai] if last_ai is not None else {}

    # Fill response
    msg0 = out["response"]["choices"][0]["message"]
    msg0["role"] = "assistant"
    msg0["content"] = final_ai.get("content", "") or ""
    msg0["reasoning_content"] = final_ai.get("thought", None)

 
    if final_ai.get("tool_calls"):
      
        msg0["tool_calls"] = final_ai.get("tool_calls")

    out["response"]["created"] = int(time.time())

    # Build request.messages
    messages: List[Dict[str, Any]] = []
    pending_calls: List[Dict[str, Optional[str]]] = []  # queue: {"id":..., "name":...}

    for idx, msg in enumerate(dialog):
        role = msg.get("role")

        # skip final assistant from request.messages (goes into response)
        if last_ai is not None and idx == last_ai and role == "assistant":
            continue

        if role == "user":
            if _is_tool_response_user(msg):
                call = pending_calls.pop(0) if pending_calls else {"id": None, "name": None}
                tool_content = _strip_tool_response(msg.get("content", "") or "")
                tool_content = _maybe_strip_token_outputs_from_json_string(tool_content)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": call.get("name"),
                    "content": tool_content,
                })
            else:
                user_content = msg.get("content", "") or ""
                user_content = _maybe_strip_token_outputs_from_json_string(user_content)
                messages.append({"role": "user", "content": user_content})

        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                thought = (msg.get("thought") or "").rstrip()
                parts: List[str] = []
                if thought:

                    parts.append(thought + "\n</think>\n\n")

                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name") or ""
                    args_obj = _safe_json_loads(fn.get("arguments", ""))
                    args_obj = _strip_token_outputs(args_obj)
                    parts.append(_tool_call_block(name, args_obj))
                    pending_calls.append({"id": tc.get("id"), "name": name})

                messages.append({"role": "assistant", "content": "".join(parts)})
            else:
                # normal assistant: prefer content, fallback to thought
                c = msg.get("content")
                if not c and msg.get("thought"):
                    c = msg.get("thought")
                messages.append({"role": "assistant", "content": c or ""})



    out["request"]["messages"] = messages


    if isinstance(usage, dict):
        u = out["response"]["usage"]
        u["prompt_tokens"] = int(usage.get("prompt_tokens", u["prompt_tokens"]))
        u["completion_tokens"] = int(usage.get("completion_tokens", u["completion_tokens"]))
        u["total_tokens"] = int(usage.get("total_tokens", u["total_tokens"]))


    # Final sweep: remove token outputs anywhere
    out = _strip_token_outputs(out)
    return out

class MongoTraceWriter:
    """
    Convert Extended JSON (containing $oid/$date) to BSON and upsert to Mongo.
    Suitable for "real-time update of the same trace".
    """
    def __init__(self, uri: str, db: str = "gaia", collection: str = "trace_latest"):
        self.client = MongoClient(uri)
        self.col = self.client[db][collection]

        if os.getenv("MONGO_CREATE_INDEX", "0") == "1":
            try:
                self.col.create_index([("task.$id", 1)])
                self.col.create_index([("creat_time", -1)])
            except Exception as e:
                logger.warning(f"create_index failed: {e}")

    def emit(self, doc: dict) -> None:
        try:
            s = json.dumps(doc, ensure_ascii=False, default=str)
            bson_doc = bson_loads(s)
            # Use _id to locate and replace the entire document => always maintain the "latest snapshot"
            self.col.replace_one({"_id": bson_doc["_id"]}, bson_doc, upsert=True)
        except Exception as e:
            logger.warning(f"[MongoTraceWriter] emit failed: {e}")
            raise

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


class TeeTraceWriter:
    """
    Write to multiple writers simultaneously: write locally first (fallback), Mongo later (failure doesn't affect main process).
    """
    def __init__(self, *writers, name: str = "tee"):
        self.writers = [w for w in writers if w is not None]
        self.name = name

    def emit(self, doc: dict) -> None:
        errors = []
        for w in self.writers:
            try:
                w.emit(doc)
            except Exception as e:
                errors.append((w.__class__.__name__, str(e)))
        if errors:
            logger.warning(f"[{self.name}] emit partial failure: {errors}")

    def close(self) -> None:
        for w in self.writers:
            try:
                w.close()
            except Exception:
                pass


def filter_huggingface_results(content: str, tool_name: str) -> str:
    """
    Filter entries containing huggingface from search results

    Args:
        content: Content returned by tool
        tool_name: Tool name (currently only handles search)

    Returns:
        Filtered content

    Note:
        fetch_url filtering is performed before tool calls to avoid wasting resources
    """
    if not isinstance(content, str) or not content.strip():
        return content
    
    if tool_name == "search":

        lines = content.split('\n')
        filtered_lines = []
        skip_until_next_item = False
        
        for i, line in enumerate(lines):

            item_match = re.match(r'^(\d+)\.\s+\[.*?\]\((.*?)\)', line)
            
            if item_match:
              
                item_num = item_match.group(1)
                url = item_match.group(2)
                
                # Check if URL contains huggingface
                if 'huggingface.co' in url.lower():
                    skip_until_next_item = True
                    continue 
                else:
                    skip_until_next_item = False
                    filtered_lines.append(line)
            else:
               
                if not skip_until_next_item:
                    filtered_lines.append(line)
        
        # Renumber
        result_lines = []
        current_number = 1
        
        for line in filtered_lines:
            # Detect if this is the beginning of a result item
            item_match = re.match(r'^(\d+)(\.\s+\[.*?\]\(.*?\))', line)
            if item_match:
                # Replace the number
                new_line = f"{current_number}{item_match.group(2)}"
                result_lines.append(new_line)
                current_number += 1
            else:
                result_lines.append(line)
        
        # Update the total result count
        filtered_content = '\n'.join(result_lines)
        
        # Update the result count description in the first line
        # Format: "A Google search for '...' found X total results (showing top Y):"
        first_line_match = re.match(r"(A Google search for .* found )\d+( total results \(showing top )\d+(\):)", filtered_content)
        if first_line_match:
            new_count = current_number - 1
            filtered_content = re.sub(
                r"(A Google search for .* found )\d+( total results \(showing top )\d+(\):)",
                f"\\g<1>{new_count}\\g<2>{new_count}\\g<3>",
                filtered_content,
                count=1
            )
        
        return filtered_content
    

    return content


class StreamTraceWriter:
    """
    Single file trace.json: each emit() overwrites with latest complete doc.
    Use os.replace for atomic replacement to avoid reading half JSON.
    """
    def __init__(self, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.out_path = out_path

    def emit(self, doc: dict) -> None:
        tmp = self.out_path + f".tmp.{uuid.uuid4().hex}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, self.out_path) 

    def close(self) -> None:

        pass


class GaiaApiTest:
    """
    GAIA API Test (AgentCPM-MCP Version)
    Used to test LLM interaction with MCP tools based on GAIA dataset
    """


    def __init__(
        self,
        # 1. Main model parameters (e.g., DeepSeek)
        main_provider: str,
        main_model: str,
        main_api_key: Optional[str],
        main_base_url: Optional[str],
        # # 2. Processor model parameters 
        processor_provider: Optional[str] = None,
        processor_model: Optional[str] = None,
        processor_api_key: Optional[str] = None,
        processor_base_url: Optional[str] = None,
        # --- Receive switch parameters ---
        use_browser_processor: bool = False,
        return_thought: bool = False,
        use_context_manager: bool = False,
        # --- Common parameters remain unchanged ---
        manager_url: str = "http://localhost:8000/mcpapi",
        max_interactions: int = 10,
        output_dir: str = None,
        tool_start_tag: Optional[str] = None,
        tool_end_tag: Optional[str] = None,
        files_dir: Optional[str] = None,
        main_temperature: float = 1.0,
        main_top_p: float = 1.0,
        main_presence_penalty: float = 1.0,
        main_max_tokens: int = 16384,
        hf_tokenizer_path: Optional[str] = None
    ):
        """
        Initialize test program (V2: supports dual main/processor models)
        """
  
        self.mcp_manager = MCPManager(manager_url=manager_url)
        self.tool_start_tag = tool_start_tag
        self.tool_end_tag = tool_end_tag
        self.files_dir = files_dir
        self.use_browser_processor = use_browser_processor
        self.return_thought=return_thought
        self.use_context_manager = use_context_manager
        self.historyx: Optional[HistoryX] = None
        self.context_tokens: int = 0

        # ====== Model context budget ======
        self.model_context_len: int = 128000
        self.reserve_gen_tokens: int = main_max_tokens   # Reserve for final output/thinking chain
        self.safety_margin_tokens: int = 1024  # Prevent errors from template/system fields
        self.force_answer_prompt_budget: int = (
            self.model_context_len - self.reserve_gen_tokens - self.safety_margin_tokens
        )


        self.max_context_tokens: int = 15000000

        self.main_temperature = main_temperature
        self.main_top_p = main_top_p
        self.main_presence_penalty = main_presence_penalty
        self.main_max_tokens = main_max_tokens

        self.hf_tokenizer = None
        # Use provided path, then environment variable, finally default to a public Qwen model
        self.hf_tokenizer_path = hf_tokenizer_path or os.getenv("HF_TOKENIZER_PATH", "Qwen/Qwen3-4B-Thinking-2507")
        try:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                self.hf_tokenizer_path,
                trust_remote_code=True,
                use_fast=True,
            )
            logger.info(f"HF tokenizer initialized successfully: {self.hf_tokenizer_path}")
        except Exception as e:
            logger.warning(f"HF tokenizer initialization failed, falling back to tiktoken estimation. Reason: {e}")

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("p50k_base")
        logger.info(f"Fallback tiktoken tokenizer '{self.tokenizer.name}' initialized successfully.")


      
        client_manager = LLMClientManager()

        # 2. Create and hold main model client
        self.main_llm_client = client_manager.create_client(
            client_name="main_model",
            provider=main_provider,
            model=main_model,
            api_key=main_api_key,
            base_url=main_base_url,
            tool_start_tag=self.tool_start_tag,
            tool_end_tag=self.tool_end_tag
        )

        # 3. Conditionally create processor model client
        self.processor_llm_client = None # Defaults to None
        if self.use_browser_processor or self.use_context_manager:
            logger.info("Browser processor/Context manager enabled, initializing processor model client...")
            if not all([processor_provider, processor_model, processor_base_url]):
                 raise ValueError("When enabling browser processor, must provide processor_provider, processor_model, and processor_base_url.")
            self.processor_llm_client = client_manager.create_client(
                client_name="processor_model",
                provider=processor_provider,
                model=processor_model,
                api_key=processor_api_key,
                base_url=processor_base_url
            )
        else:
            logger.info("Browser processor disabled, skipping processor model client initialization.")

        # --- Other settings remain unchanged ---
        self.max_interactions = max_interactions
        # Use the main model's name for reporting and logging
        self.model = main_model
        self.provider = main_provider
        self.api_key = main_api_key
        self.base_url = main_base_url

        self.all_tools = []
        self.tools_by_server = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir:
            self.result_dir = Path(output_dir)
        else:
            self.result_dir = project_root / "gaia_results" / timestamp

        os.makedirs(self.result_dir, exist_ok=True)
        logger.info(f"Will save results to: {self.result_dir}")

    def _get_response_signature(self, thought: str, content: str, tool_calls: List[Dict]) -> str:
        """
        Generate unique fingerprint of response for repetition detection.
        Fingerprint consists of serialization of thought, content and tool_calls.
        """
        # Sort and serialize tool_calls to prevent misjudgments due to different dictionary orders
        tc_str = ""
        if tool_calls:
            # Simplify tool_calls, only take key fields, and sort
            simplified_tc = []
            for tc in tool_calls:
                # Extract function name and arguments
                func = tc.get("function", {})
                simplified_tc.append({
                    "name": func.get("name"),
                    "arguments": func.get("arguments")
                })
            # Sort by name
            simplified_tc.sort(key=lambda x: x.get("name", ""))
            tc_str = json.dumps(simplified_tc, sort_keys=True)
            
        return f"THOUGHT:{thought or ''}|CONTENT:{content or ''}|TOOLS:{tc_str}"


    def _count_text_tokens(self, text: str) -> int:
        """Count single text segment (prioritize Qwen tokenizer, fallback to tiktoken)"""
        if not text:
            return 0
        if self.hf_tokenizer is not None:
            return len(self.hf_tokenizer.encode(text, add_special_tokens=False))
        return len(self.tokenizer.encode(text))

    def _count_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count messages: try to be close to vLLM/sglang's chat_template behavior.
        """
        if self.hf_tokenizer is not None:
            # add_generation_prompt=True is usually closer to the real prompt of server-side chat.completions
            prompt = self.hf_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return len(self.hf_tokenizer.encode(prompt, add_special_tokens=False))

        # fallback: your original rough estimation algorithm
        tokens = 0
        for msg in messages:
            tokens += 4
            if msg.get("content"):
                tokens += len(self.tokenizer.encode(str(msg["content"])))
            if msg.get("name"):
                tokens += len(self.tokenizer.encode(str(msg["name"])))
        return tokens

    def _shrink_forced_answer(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        import copy as _copy

        force_prompt = (
            "You have now reached the maximum context length you can handle. "
            "You MUST stop making tool calls. "
            "Based on all the information above, provide the most likely final answer "
            "in the required format: <answer>your answer</answer>"
        )

        new_msgs = _copy.deepcopy(messages)

    
        idx = None
        for i in range(len(new_msgs) - 1, -1, -1):
            if new_msgs[i].get("role") == "tool":
                idx = i
                break
       
        if idx is None:
            for i in range(len(new_msgs) - 1, -1, -1):
                if new_msgs[i].get("role") == "user":
                    idx = i
                    break
    
        if idx is None:
            idx = len(new_msgs) - 1

        new_msgs[idx]["content"] = force_prompt

     
        while self._count_prompt_tokens(new_msgs) > self.force_answer_prompt_budget and len(new_msgs) > 2:
            
            new_msgs.pop(1)

        return new_msgs


    def _format_tools_for_prompt(self) -> str:

        if not self.all_tools:
            return "<tools>\nNo tools available.\n</tools>"

        TOOLS_TO_KEEP = {
            'search',
            'fetch_url'
        }

        PYTHON_INTERPRETER_TOOL = {
            "type": "function",
            "function": {
                "name": "PythonInterpreter",
                "description": (
                    "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:\n"
                    "1. The 'arguments' JSON object must be empty: {}.\n"
                    "2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.\n\n"
                    "CRITICAL: The environment is STATELESS. Variables, functions, and imports from previous calls are LOST. You MUST re-import all libraries and re-define all variables in every code block.\n\n"
                    "IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.\n\n"
                    "Example of a correct call:\n"
                    "<tool_call>\n"
                    "{\"name\": \"PythonInterpreter\", \"arguments\": {}}\n"
                    "<code>\n"
                    "import numpy as np\n"
                    "# Your code here\n"
                    "print(f\"The result is: {np.mean([1,2,3])}\")\n"
                    "</code>\n"
                    "</tool_call>"
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

        tool_json_strings = []
        
        for tool_original in self.all_tools:
            try:

                tool_name = tool_original.get('function', {}).get('name', '')
                if tool_name not in TOOLS_TO_KEEP:
                    continue  # If tool is not in whitelist, skip it
                tool = json.loads(json.dumps(tool_original))
                if "type" not in tool or "function" not in tool:
                    logger.warning(f"Skipping tool with missing 'type' or 'function': {tool}")
                    continue


                tool_str = json.dumps(tool, ensure_ascii=False)
                tool_json_strings.append(tool_str)
                
            except Exception as e:
                logger.warning(f"Error formatting tool {tool_original.get('function', {}).get('name')}: {e}")

        tool_json_strings.append(json.dumps(PYTHON_INTERPRETER_TOOL, ensure_ascii=False))
        
        return f"<tools>\n" + "\n".join(tool_json_strings) + "\n</tools>"
        
    

    async def initialize(self) -> bool:

        try:
            logger.info("Initializing MCPManager client...")
            if not await self.mcp_manager.initialize():
                logger.error("MCPManager client initialization failed")
                return False
            
            # Get all tools
            self.all_tools = self.mcp_manager.openai_tools
            
            # Save the tool information to a Markdown file for review
            tools_md_path = self.result_dir / "available_tools.md"
            try:
                import json
                with open(tools_md_path, "w", encoding="utf-8") as f:
                    f.write("# Available tools provided by MCPManager\n\n")
                    f.write(f"**Total number of tools**: {len(self.all_tools)}\n\n")
                    f.write("---\n\n")
                    
                    for i, tool in enumerate(self.all_tools):
                        function_spec = tool.get("function", {})
                        tool_name = function_spec.get("name", "N/A")
                        description = function_spec.get("description", "No description information.")
                        parameters = function_spec.get("parameters", {})
                        
                        f.write(f"### {i+1}. Tool Name: `{tool_name}`\n\n")
                        f.write(f"**Function Description:**\n")
                        f.write(f"> {description}\n\n")
                        f.write(f"**Parameter Structure:**\n")
                        f.write("```json\n")
                        f.write(json.dumps(parameters, indent=2, ensure_ascii=False))
                        f.write("\n```\n\n")
                        f.write("---\n\n")

                logger.info(f"Detailed information of all tools saved to file: {tools_md_path}")
            except Exception as e:
                logger.error(f"Error saving tool information to Markdown file: {e}")
            logger.info(f"Retrieved {len(self.all_tools)} tools from MCPManager")
            
            # Organize tools by server
            servers = await self.mcp_manager.list_servers()
            self.tool_to_server_map = {}  # Add tool to server mapping
            
            for server in servers:
                server_tools = await self.mcp_manager.get_server_tools(server)
                self.tools_by_server[server] = server_tools
                logger.info(f"Server '{server}' has {len(self.tools_by_server[server])} tools")

                # Create mapping from each tool to server
                for tool in server_tools:
                    if "function" in tool:
                        tool_name = tool["function"].get("name", "unknown")
                        self.tool_to_server_map[tool_name] = server
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            traceback.print_exc()
            return False

    async def close(self):
        """Close resources"""
        if hasattr(self, 'mcp_manager') and self.mcp_manager:
            await self.mcp_manager.close()
            logger.info("MCPManager client closed")

    async def _manage_context_tool(self) -> Dict[str, Any]:
      
        logger.info("Executing local 'manage_context' tool...")

        if not self.historyx or len(self.historyx) <= 3: #
            logger.info("History record too short, skipping compression.")
            return {
                "status": "success",
                "content": {"messages": self.historyx.to_raw_history() if self.historyx else []},
            }

        compress_status = await self._compress_history()

        return {
            "status": "success" if compress_status else "failed",
            "content": {"messages": self.historyx.to_raw_history() if compress_status else ""},
        }

    async def _compress_history(self) -> bool:
        
        if not self.historyx:
            logger.error("History compression failed: self.historyx not initialized.")
            return False
            
        logger.debug(f"Compressing history, current message count: {len(self.historyx)}")

        # 1. Get history with IDs for generating user_prompt
        try:
            history_with_id = self.historyx.to_raw_history(include_id=True, include_system_prompt=False)

        except Exception as e:
            logger.error(f"Error serializing history records: {e}")
            return False

        procedure_obj: Optional[Procedure] = None
        try:

            procedure_obj = generate_procedure_summary(
                processor_llm_client=self.processor_llm_client,
                history_to_summarize=history_with_id 
            )
            
            if not procedure_obj:
                raise ValueError("generate_procedure_summary (secondary model) returned None or failed.")

        except Exception as e:
            logger.error(f"History summary generation failed: {e}")
            return False


        
        try:
            update_history_with_procedure(self.historyx, procedure_obj)
        except Exception as e:
            logger.error(f"Failed to update historyx with Procedure summary: {e}")
            return False

        logger.info(f"History compression successful, message count after compression: {len(self.historyx)}")
        

        self.context_tokens = 0
        return True

    async def _discard_and_summarize_history(self) -> bool:
        """
        Discard & Summarize

        """
        if not self.historyx or len(self.historyx) <= 2:
            return False

        logger.info("Executing Discard & Summarize (new version) to clean context...")

        # 1) Get all messages (excluding system), and collect msg_ids to remove
        raw_msgs = self.historyx.to_raw_history(include_id=True, include_system_prompt=False)

        ids_to_remove: List[int] = []
        assistant_nodes: List[Dict[str, str]] = []

        for m in raw_msgs:
            msg_id = m.get("id")
            if msg_id is None:
                continue

            # Keep id=1 (usually "Your task is ..." or original user question), remove all others
            if msg_id > 1:
                ids_to_remove.append(msg_id)

            # Summary material: only keep assistant's thought + content (tools/user pretending tool_response are not included)
            if m.get("role") == "assistant":
                assistant_nodes.append({
                    "thought": m.get("thought", "") or "",
                    "content": m.get("content", "") or ""
                })

        # 2) Define serialization and token counting
        def material_text(nodes: List[Dict[str, str]]) -> str:
            parts: List[str] = []
            for n in nodes:
                parts.append(
                    f"[Assistant Thought]: {n.get('thought','')}\n"
                    f"[Assistant Content]: {n.get('content','')}\n"
                )
            return "\n".join(parts)

        def material_tokens(nodes: List[Dict[str, str]]) -> int:
            return self._count_text_tokens(material_text(nodes))

        # 3) Pruning (only for assistant thought/content, tools have been excluded)
        TOKEN_SOFT_LIMIT = 58000 

        def truncate_text_by_tokens(text: str, keep_tokens: int) -> str:
            """Truncate text to keep_tokens (by tokenizer tokens), and add truncation prompt at the end."""
            if not text:
                return ""
            if keep_tokens <= 0:
                return ""
            if self.hf_tokenizer is not None:
                ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
                if len(ids) <= keep_tokens:
                    return text
                cut = self.hf_tokenizer.decode(ids[:keep_tokens], skip_special_tokens=True)
                return cut + "\n...(truncated by system due to context limit)..."
            # fallback: when tiktoken cannot reliably decode, truncate by characters (conservative)
            return text[: max(0, min(len(text), keep_tokens * 4))] + "\n...(truncated by system due to context limit)..."

        # A) If exceeded: prioritize truncating "longest thought", try to truncate only a small amount until it fits
        #    Here use "sort by token length" to repeatedly truncate, avoid truncating too much at once
        safety_iters = 20
        while assistant_nodes and material_tokens(assistant_nodes) > TOKEN_SOFT_LIMIT and safety_iters > 0:
            safety_iters -= 1

            thought_lens = [
                self._count_text_tokens(n.get("thought", "")) for n in assistant_nodes
            ]
            max_i = max(range(len(thought_lens)), key=lambda i: thought_lens[i])
            max_len = thought_lens[max_i]

            if max_len <= 0:
                break

            over = material_tokens(assistant_nodes) - TOKEN_SOFT_LIMIT
            buffer = 256
            keep = max(256, max_len - (over + buffer))
            if keep >= max_len:
                break

            assistant_nodes[max_i]["thought"] = truncate_text_by_tokens(
                assistant_nodes[max_i].get("thought", ""), keep
            )

        # B) Still exceeded: remove thought from the 2nd assistant message onwards (the 1st one is usually the plan, try to preserve)
        if material_tokens(assistant_nodes) > TOKEN_SOFT_LIMIT and len(assistant_nodes) > 1:
            for i in range(1, len(assistant_nodes)):
                if material_tokens(assistant_nodes) <= TOKEN_SOFT_LIMIT:
                    break
                assistant_nodes[i]["thought"] = "(Thought removed to save context tokens)"

        # C) Still exceeded: remove entire assistant records from the 2nd one onwards (preserve the 1st as task/plan anchor)
        while material_tokens(assistant_nodes) > TOKEN_SOFT_LIMIT and len(assistant_nodes) > 2:
            assistant_nodes.pop(1)

        # 4) Construct summary input
        initial_raw = self.historyx.to_raw_history()
        initial_query = "Unknown Task"
        if isinstance(initial_raw, list) and len(initial_raw) > 1:
            initial_query = initial_raw[1].get("content", "Unknown Task")

        history_for_summary = material_text(assistant_nodes) if assistant_nodes else (
            "(No assistant thought/content available; context was reset due to token limit.)"
        )

        summary_system_prompt = """You are an intelligent summarization assistant. Your task is to summarize all the findings and actions from the conversation history.

    Please provide a concise summary in Markdown format that includes:
    1. **Actions**: Actions taken by the agent
    2. **Results**: Results of the actions
    3. **Current Status**: What has been accomplished and what remains

    Format your response as clean Markdown without code blocks."""

        summary_user_prompt = (
            f"Please summarize the following conversation history for the task: '{initial_query}'\n\n"
            f"## History (Assistant thought+content only; tools removed)\n"
            f"{history_for_summary}\n\n"
            "Please provide a concise summary in Markdown format covering all key findings, actions, and results."
        )

        try:
            resp = self.processor_llm_client.create_completion(
                messages=[
                    {"role": "system", "content": summary_system_prompt},
                    {"role": "user", "content": summary_user_prompt},
                ]
            )
            summary_text = resp.get("response", "Failed to generate summary.")
        except Exception as e:
            logger.error(f"Secondary model summary generation failed: {e}")
            summary_text = "Context cleared due to token limit. Summary failed to generate properly."

        # 5) Execute context reset: replace all messages with msg_id>1 with one summary assistant message
        summary_msg_param = {
            "role": "assistant",
            "content": (
                "## Execution Summary (Context Reset)\n\n"
                f"{summary_text}\n\n"
                "I will now continue the task based on this summary."
            )
        }

        try:
            self.historyx.replace_ids_with_message(ids_to_remove, summary_msg_param)
            self.context_tokens = 0
            logger.info(f"Successfully cleaned {len(ids_to_remove)} messages (tools discarded; summary based on thought+content).")
            return True
        except Exception as e:
            logger.error(f"HistoryX replacement failed: {e}")
            return False

   

    def _try_parse_tool_call_arguments(self, arguments_input: Union[str, Dict]) -> Dict[str, Any]:
        
        
        if isinstance(arguments_input, dict):
            return arguments_input
        
       
        if isinstance(arguments_input, str):
            arguments_str = arguments_input.strip()
            if not arguments_str:
                return {}
            
            
            try:
                return json.loads(arguments_str)
            except json.JSONDecodeError:
                
                # If standard parsing fails, check if it's because the ending '}' is missing
                if arguments_str.startswith('{') and not arguments_str.endswith('}'):
                    logger.info("Detected JSON may be missing right brace, attempting automatic repair...")
                    healed_str = arguments_str + '}'
                    try:
                        # Try to parse the repaired string again
                        return json.loads(healed_str)
                    except json.JSONDecodeError:
                        # If still fails after repair, give up repair and let subsequent logic handle it
                        logger.warning("Parsing still failed after automatic repair, continuing with other parsing methods.")
                

               
                try:
                    import json5
                    return json5.loads(arguments_str)
                except Exception as e:
                    
                    logger.warning(f"Unable to parse parameter '{arguments_str}' as JSON, will process as single query. Error: {e}")
                    return {"query": arguments_str}
        
        return {}

    async def _call_tool_with_retry(
        self,
        full_tool_name: str,
        arguments: Dict[str, Any],
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Tool call retry wrapper:
        - When tool_result["status"] == "error" or call throws exception, retry up to max_attempts times
        - Retry only happens internally, intermediate failures won't be written to historyx / full_conversation_log
        """
        last_result: Optional[Dict[str, Any]] = None

        for attempt in range(1, max_attempts + 1):
            try:
                result = await self.mcp_manager.call_tool(full_tool_name, arguments)
            except Exception as e:
   
                result = {
                    "status": "error",
                    "content": {
                        "error": str(e),
                        "detail": "Exception raised during mcp_manager.call_tool",
                        "traceback": traceback.format_exc(),
                    },
                }

    
            if not isinstance(result, dict):
                result = {
                    "status": "error",
                    "content": {
                        "error": "Tool returned non-dict result",
                        "detail": f"type={type(result)}",
                        "traceback": "",
                    },
                }

            status = result.get("status", "error")
            if status != "error":
                if attempt > 1:
                    logger.info(f"Tool {full_tool_name} succeeded after {attempt}/{max_attempts} attempts.")
                return result

            # status == "error"
            err_msg = ""
            try:
                err_msg = result.get("content", {}).get("error", "")  
            except Exception:
                err_msg = ""
            logger.warning(f"Tool {full_tool_name} attempt {attempt}/{max_attempts} failed: {err_msg or 'unknown error'}")

            last_result = result


            if attempt < max_attempts:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay += random.random() * 0.1
                await asyncio.sleep(delay)

        logger.error(f"Tool {full_tool_name} failed {max_attempts} times consecutively, returning last error.")
        return last_result or {"status": "error", "content": {"error": "Unknown tool failure", "detail": "", "traceback": ""}}


    async def run_test_conversation(
        self, 
        query: str, 
        context: Optional[str] = None,
        reference_code: Optional[str] = None,
        reference_files: Optional[Dict[str, str]] = None,
        file_name: Optional[str] = None,
        task_id: Optional[str] = None,
        run_output_dir: Optional[str] = None,
        traj_id: int = 0, 
    ) -> Dict[str, Any]:
        """
        Run test conversation 
       
        """
        if task_id is None:
            task_id = f"gaia_{int(time.time())}_{random.randint(1000, 9999)}"
        
        start_time = datetime.now()
        full_conversation_log = []
        tool_calls_made = []
        interaction_count = 0

        base_dir = Path(run_output_dir).resolve() if run_output_dir else (self.result_dir / str(task_id))
        run_dir = base_dir / f"traj_{traj_id}"
        os.makedirs(run_dir, exist_ok=True)


        # ===== [NEW] streaming trace (single output file) =====
        task_oid = _stable_oid(str(task_id))   # Use task_id to stably generate a fake Task $oid
        trace_path = str((run_dir / "trace.json").resolve())
        local_writer = StreamTraceWriter(trace_path)

        mongo_writer = None
        mongo_client = None

        mongo_uri = os.getenv("MONGO_URI")
        mongo_db  = os.getenv("MONGO_DB", "gaia")
        mongo_col = os.getenv("MONGO_COL", "trace_latest")

        print(f"[MONGO DEBUG] uri={mongo_uri!r} db={mongo_db!r} col={mongo_col!r}", flush=True)

        mongo_enabled = bool(mongo_uri and mongo_db and mongo_col)

        if mongo_uri:
            try:
                mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
                mongo_client.admin.command("ping")  
                mongo_writer = MongoTraceWriter(mongo_uri, db=mongo_db, collection=mongo_col)
                logger.info(f"Mongo trace enabled: {mongo_db}.{mongo_col}")
            except Exception as e:
                print(f"[mongo] disabled: connect failed. db={mongo_db} col={mongo_col} err={e}")
                mongo_enabled = False
                mongo_client = None
                mongo_writer = None
        else:
            logger.info("[mongo] disabled: missing MONGO_URI/MONGO_DB/MONGO_COL")

        trace_writer = TeeTraceWriter(local_writer, mongo_writer, name=f"trace:{task_id}:{traj_id}")

        
        dialog_like_so_far: List[Dict[str, Any]] = []
        trace_oid = _stable_oid(f"trace:{task_id}:{traj_id}")
   

        self.MAX_CONSECUTIVE_NO_OP = 3 
        self.NO_OP_PROMPT = "You didn't provide a tool call or a final answer. Please reassess your plan, take the next step, or provide the final answer in `<answer></answer>` tags."

        self.NO_OP_FORCE_SYSTEM_PROMPT = (
            "You have repeatedly failed to produce a tool call or a final answer. "
            "Do NOT make any tool calls. "
            "Provide your best-guess final answer NOW, strictly wrapped in <answer></answer> tags."
        )
        

        consecutive_no_op_count = 0
        
        self.context_tokens = 0 
        
        try:
             # (Tool removal logic remains unchanged)
            TOOLS_TO_REMOVE = {
                # 'read_file',
                # 'execute_code'
            }
            available_tools = self.all_tools # By default, all tools are available
            logger.info(f"File task activated, removing blacklist tools from tool list: {TOOLS_TO_REMOVE}")
            available_tools = [
                tool for tool in self.all_tools
                if tool.get('function', {}).get('name', '') not in TOOLS_TO_REMOVE
            ]
            logger.info(f"After filtering, number of tools available for this model call: {len(available_tools)}")
           
            original_all_tools = self.all_tools
            self.all_tools = available_tools
            tools_description_str = self._format_tools_for_prompt()
            self.all_tools = original_all_tools 
            
            system_message = SYSTEM_PROMPT_QWEN.format(tools_description=tools_description_str)

            if context:
                system_message += f"\n\nContext information:\n{context}"
            if reference_code:
                system_message += f"\n\nReference code:\n```\n{reference_code}\n```"
            if reference_files:
                system_message += "\n\nReference files:"
                for filename, content in reference_files.items():
                    system_message += f"\n\nFile: {filename}\n```\n{content}\n```"
            if file_name and self.files_dir:
                full_path = os.path.join(self.files_dir, file_name).replace('\\', '/')
                system_message += "\n\n--- Associated File ---\n"
                system_message += f"A file is associated with this task.\n"
                system_message += f"The absolute path to this file for your tools is: {full_path}\n"
           
            
            
            self.historyx = HistoryX.from_raw_history([
                {"role": "system", "content": system_message}
            ])
            
            self.historyx.add_message({"role": "user", "content": f"Your task is to answer the user's question: {query}"})
            dialog_like_so_far.append({"role": "user", "content": f"Your task is to answer the user's question: {query}"})
            full_conversation_log.extend(self.historyx.to_raw_history())
            
           
            

            
            consecutive_repetition_count = 0
            last_response_signature = None
            REPETITION_THRESHOLD = 4  
            # --- Main loop starts ---
            while interaction_count < self.max_interactions:
                interaction_count += 1
                logger.info(f"Conversation round {interaction_count}/{self.max_interactions}")

                if self.use_context_manager and self.context_tokens > self.max_context_tokens:
                            logger.info(f"Passive trigger: accumulated tokens ({self.context_tokens}) exceed threshold ({self.max_context_tokens}), executing history compression...")

                            
                            compress_success = await self._compress_history()
                            
                            if compress_success:
                                logger.info(f"Passive compression successful. Token count has been reset.")

                       
                                
                                
                                
                                raw_history_for_next_turn = self.historyx.to_raw_history(include_system_prompt=True)
                                next_pruned_history = []
                                
                             
                                for msg in raw_history_for_next_turn:
                                    clean_msg = msg.copy()
                                    if clean_msg.get("role") == "assistant":
                                        if getattr(self, 'return_thought', False):
                                            model_thought = clean_msg.get("thought")
                                            model_content = clean_msg.get("content")
                                            new_content_parts = []
                                            if model_thought: new_content_parts.append(f"<think>{model_thought}</think>")
                                            if model_content: new_content_parts.append(model_content)
                                            if new_content_parts: clean_msg['content'] = "\n".join(new_content_parts)
                                        
                                        if 'thought' in clean_msg: del clean_msg['thought']
                                    next_pruned_history.append(clean_msg)

                               
                                split_marker = {
                                    "role": "__CONTEXT_SPLIT__",
                                    "next_history_segment": next_pruned_history
                                }
                                full_conversation_log.append(split_marker)
                                logger.info("Split marker has been inserted.")
                                
                            else:
                                logger.error(f"Passive compression failed.")

                # (Available tool logic remains unchanged)
                available_tools = self.all_tools
                if file_name and file_name.endswith('.json11'):
                    logger.info(f"JSON file task activated, removing blacklist tools from tool list: {TOOLS_TO_REMOVE}")
                    available_tools = [
                        tool for tool in self.all_tools
                        if tool.get('function', {}).get('name', '') not in TOOLS_TO_REMOVE
                    ]
                    logger.info(f"After filtering, number of tools available for this model call: {len(available_tools)}")

                if self.use_context_manager:
                    extended_tools = list(available_tools)
                    extended_tools.append(MANAGE_CONTEXT_TOOL_SCHEMA)
                    available_tools = extended_tools
                    logger.info(f"Context manager enabled, 'manage_context' tool has been added to the model.")

                # [Modified] Get raw history records from self.historyx for processing
                # We use to_raw_history() to get a processable list of dicts
                raw_history_for_pruning = self.historyx.to_raw_history(include_system_prompt=True)
                pruned_history = []
                for i, msg in enumerate(raw_history_for_pruning):
                    clean_msg = msg.copy()
                    
                    if clean_msg.get("role") == "assistant":
                        if getattr(self, 'return_thought', False):
                            model_thought = clean_msg.get("thought")
                            model_content = clean_msg.get("content")
                            
                            new_content_parts = []
                            if model_thought:
                                new_content_parts.append(f"<think>{model_thought}</think>")
                            if model_content:
                                new_content_parts.append(model_content)
                            
                            if new_content_parts:
                                clean_msg['content'] = "\n".join(new_content_parts)

                        if "tool_calls" in clean_msg:
                            del clean_msg["tool_calls"]

                        
                        if 'thought' in clean_msg:
                            del clean_msg['thought']

                    pruned_history.append(clean_msg)
                
                
                
                


                # ====== [NEW] Precise prompt token counting (prioritize Qwen tokenizer) ======
                current_prompt_tokens = self._count_prompt_tokens(pruned_history)
                self.context_tokens = current_prompt_tokens
                logger.info(f"Token count: {int(current_prompt_tokens)} (Prompt) (budget: {self.force_answer_prompt_budget})")

                # ====== [NEW] Token trigger: force convergence before context explosion ======
                forced_by_token_limit = False

                if current_prompt_tokens > self.force_answer_prompt_budget:
                    logger.warning(f"Prompt tokens ({int(current_prompt_tokens)}) exceed budget, triggering Discard & Summarize...")

                    # 1. Execute summary surgery (internally modifies self.historyx)
                    success = await self._discard_and_summarize_history()

                    if success:
                        # 2. Surgery successful, re-acquire simplified history
                        raw_history_after_reset = self.historyx.to_raw_history(include_system_prompt=True)
                        

                        pruned_history = []
                        for msg in raw_history_after_reset:
                            clean_msg = msg.copy()
                            if clean_msg.get("role") == "assistant":
                                if getattr(self, 'return_thought', False):
                                    model_thought = clean_msg.get("thought")
                                    model_content = clean_msg.get("content")
                                    new_content_parts = []
                                    if model_thought: new_content_parts.append(f"<think>{model_thought}</think>")
                                    if model_content: new_content_parts.append(model_content)
                                    if new_content_parts: clean_msg['content'] = "\n".join(new_content_parts)
                                if "tool_calls" in clean_msg:
                                    del clean_msg["tool_calls"]

                                if 'thought' in clean_msg: del clean_msg['thought']
                            pruned_history.append(clean_msg)
                        
                        logger.info("Discard Summary completed, preparing to continue task...")
                    else:
                        # 4. If summary fails, fall back to forced truncation and answer
                        logger.error("Discard Summary failed, executing forced truncation...")
                        pruned_history = self._shrink_forced_answer(pruned_history)

                    # Update counter to show correct logs
                    current_prompt_tokens = self._count_prompt_tokens(pruned_history)
                    self.context_tokens = current_prompt_tokens
                    logger.info(f"Current prompt tokens after processing: {int(current_prompt_tokens)}")
           


                llm_response = self.main_llm_client.create_completion(
                    messages=pruned_history,
                    temperature=self.main_temperature,          
                    top_p=self.main_top_p,               
                    presence_penalty=self.main_presence_penalty,
                    max_tokens=self.main_max_tokens
                )


                model_thought = llm_response.get("thought", "")
                model_response = llm_response.get("response", "")
                tool_calls = llm_response.get("tool_calls", [])

                
                current_signature = self._get_response_signature(model_thought, model_response, tool_calls)
                
                if current_signature == last_response_signature:
                    consecutive_repetition_count += 1
                else:
                    consecutive_repetition_count = 1
                    last_response_signature = current_signature

                if consecutive_repetition_count >= REPETITION_THRESHOLD:
                    logger.warning(f"⚠️ Detected model repeating consecutively {consecutive_repetition_count} times! Triggering forced Discard & Summarize...")

                    # 1. Execute history cutoff and summarization
                    success = await self._discard_and_summarize_history()

                    if success:
                        consecutive_repetition_count = 0
                        last_response_signature = None

                        logger.info("Repetition handling completed, skipping current round, entering next round generation...")
                        continue
                    else:
                        logger.error("Repetition detection triggered summary failure, will continue execution...")
                

                current_completion_tokens = 0
                if model_thought:
                    current_completion_tokens += self._count_text_tokens(model_thought)
                if model_response:
                    current_completion_tokens += self._count_text_tokens(model_response)
                if tool_calls:
                    current_completion_tokens += self._count_text_tokens(json.dumps(tool_calls, ensure_ascii=False))

                self.context_tokens += current_completion_tokens
                logger.info(f"Token count: {int(current_completion_tokens)} (Completion) (Total: {int(self.context_tokens)})")



                
                
               
                if not tool_calls and model_response:
                    try:
                        import re
                        json_str_to_parse = ""
                        cleaned_model_response = model_response.strip()
                        match = re.search(r'```(?:\w+)?\s*\n(.*?)\n\s*```', model_response, re.DOTALL)
                        if match:
                            logger.info("Parse strategy 1: Successfully matched Markdown code block.")
                            json_str_to_parse = match.group(1).strip()
                            model_response = model_response[:match.start()].strip()
                        elif '<tool_call>' in model_response:
                            logger.info("Parse strategy 2: Successfully matched <tool_call> tag.")
                            tool_start_tag = '<tool_call>'
                            tool_end_tag = '</tool_call>'
                            start_index = model_response.find(tool_start_tag)
                            end_index = model_response.rfind(tool_end_tag)
                            if end_index > start_index:
                                json_str_to_parse = model_response[start_index + len(tool_start_tag):end_index].strip()
                                model_response = model_response[:start_index].strip()
                        elif (cleaned_model_response.startswith('{') and cleaned_model_response.endswith('}')) or \
                             (cleaned_model_response.startswith('[') and cleaned_model_response.endswith(']')):
                            logger.info("Parse strategy 3: Successfully matched raw JSON string.")
                            json_str_to_parse = cleaned_model_response
                            model_response = ""
                        if json_str_to_parse:
                        
                            parsed_json = self._try_parse_tool_call_arguments(json_str_to_parse) 
                            if isinstance(parsed_json, dict) and parsed_json.get("name"):
                                 tool_calls = [{"id": f"call_{random.randint(10000, 99999)}", "type": "function", "function": {"name": parsed_json.get("name"), "arguments": json.dumps(parsed_json.get("arguments", {}), ensure_ascii=False)}}]
                            elif isinstance(parsed_json, list):
                                tool_calls = parsed_json 
                    except Exception as e:
                        logger.warning(f"Unknown error occurred when trying to manually parse tool calls from content: {e}", exc_info=True)
                
              
                if not tool_calls:
                    logger.info("LLM did not call any tools this time, check if it's the final answer...")
                    
                    
                    # Relaxed check for answer in response
                    has_answer_in_response = bool(extract_last_answer(model_response))
                    
                    
                    if not has_answer_in_response and model_thought:
                        # Fallback regex for extraction from thought (complex replacement logic)
                        answer_pattern_relaxed = re.compile(r"(?is)(?:<answer>|<answer|answer>)(.*?)(?:</answer>|</answer|/answer>)")
                        match = answer_pattern_relaxed.search(model_thought.strip())

                        if match:
                            logger.info("Final answer detected in 'thought', executing 'transfer' operation...")
                            answer_block = match.group(0)
                            model_response = answer_block
                            model_thought = model_thought.replace(answer_block, "").strip()

                    # [Modified] Extract clean answer content (strip tags) before saving to history
                    extracted_clean_answer = extract_last_answer(model_response)
                    if extracted_clean_answer:
                         model_response = extracted_clean_answer

                    final_assistant_message = {"role": "assistant", "content": model_response}
                    if model_thought:
                        final_assistant_message["thought"] = model_thought
                    
                    full_conversation_log.append(final_assistant_message)
                    self.historyx.add_message(final_assistant_message)

                    
                    dialog_like_so_far.append({
                        "role": "assistant",
                        "content": final_assistant_message.get("content", ""),
                        **({"thought": final_assistant_message.get("thought")} if final_assistant_message.get("thought") else {})
                    })
                    doc = convert_dialog_step_to_doc(
                        dialog_like_so_far,
                        step=interaction_count,
                        traj_id=traj_id,
                        task_oid=task_oid,
                        model_name=self.model,                 
                        tools=available_tools,                 
                        usage=llm_response.get("usage") if isinstance(llm_response, dict) else None,
                    )
                    doc["_id"] = {"$oid": trace_oid} 
                    trace_writer.emit(doc)
                   

                    

                    # Relaxed check for complete answer block
                    # We check extracted_clean_answer because model_response is now stripped
                    if has_answer_in_response or extracted_clean_answer:
                        logger.info("Complete <answer>...</answer>tag pair detected (and stripped), conversation completed.")
                        break
                    else:
                        consecutive_no_op_count += 1
                        logger.warning(f"LLM did not provide tool or answer. Consecutive invalid responses: {consecutive_no_op_count}/{self.MAX_CONSECUTIVE_NO_OP}")

                        if consecutive_no_op_count < self.MAX_CONSECUTIVE_NO_OP:
                            logger.info(
                                f"NO-OP {consecutive_no_op_count}/{self.MAX_CONSECUTIVE_NO_OP} time: not inserting prompt, continuing directly to next conversation round."
                            )
                            continue

                        # 3rd time: insert system forced answer once (only once, don't stuff system every time)
                        logger.warning(
                            f"NO-OP has reached threshold ({self.MAX_CONSECUTIVE_NO_OP}): inserting forced answer system prompt and continuing."
                        )
                        force_msg = {"role": "system", "content": self.NO_OP_FORCE_SYSTEM_PROMPT}
                        full_conversation_log.append(force_msg)
                        self.historyx.add_message(force_msg)

                        # Reset count to avoid immediate re-injection of system if still no-op in next round
                        consecutive_no_op_count = 0
                        continue
                
                # --- Tool calling logic ---
                else:
                    consecutive_no_op_count = 0
                    assistant_message = {"role": "assistant", "content": model_response, "tool_calls": tool_calls}
                    if model_thought:
                        assistant_message["thought"] = model_thought
                    
                    full_conversation_log.append(assistant_message)
                    # [Modified] Use HistoryX to add assistant's tool call message
                    self.historyx.add_message(assistant_message)

                    # ===== [NEW] stream emit after each assistant generation (tool-calling turn) =====
                    dialog_like_so_far.append({
                        "role": "assistant",
                        "content": assistant_message.get("content", ""),
                        **({"thought": assistant_message.get("thought")} if assistant_message.get("thought") else {}),
                        "tool_calls": assistant_message.get("tool_calls", [])
                    })
                    doc = convert_dialog_step_to_doc(
                        dialog_like_so_far,
                        step=interaction_count,
                        traj_id=traj_id,
                        task_oid=task_oid,
                        model_name=self.model,
                        tools=available_tools,
                        usage=llm_response.get("usage") if isinstance(llm_response, dict) else None,
                    )
                    doc["_id"] = {"$oid": trace_oid}
                    trace_writer.emit(doc)
                    # =====================================================================

                    

                    for tool_call_original in tool_calls:
                        # [Modified] Directly modify tool_call_original to ensure "history records" store corrected tool names and parameters
                        # This way the model in the next round reviewing history will think it called fetch_url + purpose, not visit + goal


                        tool_name = tool_call_original.get("function", {}).get("name", "")
                        arguments_input = tool_call_original.get("function", {}).get("arguments", "{}")
                        arguments = self._try_parse_tool_call_arguments(arguments_input)

                        
                        if tool_name in ("fetch_url", "visit") and "url" in arguments:
                            if isinstance(arguments["url"], str):
                                arguments["url"] = [arguments["url"]]

                        if tool_name in ("search") and "query" in arguments:
                            if isinstance(arguments["query"], str):
                                arguments["query"] = [arguments["query"]]

                        
                        if tool_name == "visit":
                            logger.info("Adapter: Intercepted 'visit', correcting to 'fetch_url' in history.")
                            tool_call_original["function"]["name"] = "fetch_url"
                            tool_name = "fetch_url" 

                        
                        if "goal" in arguments:
                            if "purpose" not in arguments: 
                                arguments["purpose"] = arguments.pop("goal")
                                logger.info(f"Adapter: Intercepted 'goal' (in {tool_name} call), correcting to 'purpose' in history.")
                            else:

                                arguments.pop("goal")
                                logger.warning(f"Adapter: Both 'goal' and 'purpose' exist in {tool_name} call, 'goal' has been discarded.")

                        
                        if tool_name == "fetch_url":
                            if "purpose" not in arguments or not arguments["purpose"]:
                                logger.info(f"Adapter: 'fetch_url' call missing 'purpose', adding default value.")
                                arguments["purpose"] = "summary the main content of the url"
                    
                        mcp_tool_name = tool_name
                        mcp_arguments = arguments

                        if tool_name == "PythonInterpreter":

                            code_str = tool_call_original.get("code", "")
                            if not isinstance(code_str, str) or not code_str.strip():
                                logger.warning(f"⚠️ Model called PythonInterpreter but provided no code block. Constructing error feedback...")
                                
  
                                error_msg = (
                                    "Error: Invalid tool call. You called 'PythonInterpreter' but failed to provide the Python code inside <code>...</code> tags.\n"
                                    "Correct format:\n"
                                    "<tool_call>\n{\"name\": \"PythonInterpreter\", \"arguments\": {}}\n"
                                    "<code>\n# your code here\nprint(result)\n</code>\n</tool_call>"
                                )
                                
                                # 2. Construct tool response message
                                
                                tool_message_to_log = {
                                    "role": "user",
                                    "content": f"<tool_response>\n{error_msg}\n</tool_response>"
                                }
                                
                                # 3. Add to history
                                full_conversation_log.append(tool_message_to_log)
                                self.historyx.add_message(tool_message_to_log)
                                dialog_like_so_far.append({"role": "user", "content": tool_message_to_log["content"]})
                                # 4. [Important] Update token count (simulated errors also count as tokens)
                                self.context_tokens += self._count_text_tokens(str(tool_message_to_log))
                                
                                # 5. Skip subsequent MCP calls, directly process next tool (if any) or enter next round
                                continue

                            
                            mcp_tool_name = "execute_code"
                            mcp_arguments = {"code": code_str}

                            # To comply with the specification given to the model: PythonInterpreter's arguments must be {}
                            arguments = {}


                        
                        try:
                            tool_call_original["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
                        except Exception as e:
                            logger.error(f"Failed to write back corrected parameters to tool_call_original: {e}")

                        # 6. Now generate copy for MCP call (at this point the copy already contains all the above corrections)
                        tool_call_for_mcp = copy.deepcopy(tool_call_original)

                        logger.info(f"Processing tool call: {tool_name}")
                        logger.debug(f"Tool parameters: {arguments}")
                        
                        server_id = self.tool_to_server_map.get(mcp_tool_name)
                        if not server_id:
                            server_id = self.tool_to_server_map.get(tool_name)


                        if not server_id:
                            if "." in mcp_tool_name:
                                server_id, _ = mcp_tool_name.split(".", 1)
                            else:
                                server_id = "default"
                                logger.warning(f"Cannot find server mapping for tool {tool_name}, using default server")
                        
                        full_tool_name = f"{server_id}.{mcp_tool_name}" if "." not in mcp_tool_name and server_id else mcp_tool_name
                        
                        
                        if self.use_context_manager and tool_name == "manage_context":
                            logger.info("Intercepted 'manage_context' local tool call, preparing to execute log segmentation...")
                            
                            # 1. Execute compression, this will directly modify self.historyx
                            tool_result = await self._manage_context_tool()
                            
                            # 2. Only add simple success/failure feedback to the model's "working memory"
                            if tool_result["status"] in ["success", "skipped"]:
                                feedback_content = {"status": tool_result["status"], "detail": "Context management processed."}
                                self.historyx.add_message({
                                    "role": "tool", "tool_call_id": tool_call_original["id"], "name": tool_name,
                                    "content": json.dumps(feedback_content)
                                })
                            else: # Failure case
                                error_content = {"status": "error", "detail": "Context compression failed."}
                                self.historyx.add_message({
                                    "role": "tool", "tool_call_id": tool_call_original["id"], "name": tool_name,
                                    "content": json.dumps(error_content)
                                })
                            
                            # 3. Immediately generate the compressed pruned_history to be sent to the model in the next round
                            
                            raw_history_for_next_turn = self.historyx.to_raw_history(include_system_prompt=True)
                            next_pruned_history = []
                            for msg in raw_history_for_next_turn:
                                clean_msg = msg.copy()
                                if clean_msg.get("role") == "assistant":
                                    if getattr(self, 'return_thought', False):
                                        model_thought = clean_msg.get("thought")
                                        model_content = clean_msg.get("content")
                                        new_content_parts = []
                                        if model_thought: new_content_parts.append(f"<think>{model_thought}</think>")
                                        if model_content: new_content_parts.append(model_content)
                                        if new_content_parts: clean_msg['content'] = "\n".join(new_content_parts)
                                    else:
                                        if clean_msg.get("tool_calls"): clean_msg['content'] = None
                                    if 'thought' in clean_msg: del clean_msg['thought']
                                next_pruned_history.append(clean_msg)

                            
                            split_marker = {
                                "role": "__CONTEXT_SPLIT__",
                                "next_history_segment": next_pruned_history
                            }
                            full_conversation_log.append(split_marker)


                            continue
                        
                      
                        logger.info(f"Calling MCP Manager tool: {full_tool_name}")
                        if tool_name == "search":
                            # default 10 results
                            if "num_results" not in arguments:
                                arguments["num_results"] = 10
                        
                        # [New] For fetch_url, check if URL contains huggingface before calling
 
                        skip_tool_call = False
                        if tool_name == "fetch_url":
                            url_param = arguments.get("url", [])
                            
                            if isinstance(url_param, str):
                                url_param = [url_param]
                            elif not isinstance(url_param, list):
                                url_param = []
                            
                            
                            filtered_urls = []
                            blocked_urls = []
                            for url in url_param:
                                if isinstance(url, str) and 'huggingface.co' in url.lower():
                                    blocked_urls.append(url)
                                    logger.info(f"Access to huggingface URL blocked: {url}")
                                else:
                                    filtered_urls.append(url)


                            if blocked_urls and not filtered_urls:
                                skip_tool_call = True
                                tool_result = {
                                    "status": "success",
                                    "content": f"This URL has been filtered out (huggingface domain): {', '.join(blocked_urls)}"
                                }
                                logger.info(f"All URLs for fetch_url have been filtered, skipping tool call.")
                            elif blocked_urls and filtered_urls:

                                arguments["url"] = filtered_urls
                                mcp_arguments["url"] = filtered_urls
                                logger.info(f"Partial URLs filtered for fetch_url: {', '.join(blocked_urls)}, will access remaining URLs: {', '.join(filtered_urls)}")
                        
                        # Only actually call the tool if it wasn't skipped
                        if not skip_tool_call:
                            tool_result = await self._call_tool_with_retry(full_tool_name, mcp_arguments, max_attempts=3)
                            
                            
                            if tool_name == "search" and tool_result.get("status") == "success":
                                content = tool_result.get("content", "")
                                if content:
                                    filtered_content = filter_huggingface_results(content, tool_name)
                                    tool_result["content"] = filtered_content
                                    

                       
                        is_browser_tool = "browser_" in tool_name or (tool_name == "fetch_url" or tool_name == "visit")
                        if self.use_browser_processor and is_browser_tool:
                            logger.info(f"Browser processor enabled, processing return result of tool '{tool_name}'...")

                            if tool_name == "fetch_url":
                                raw_content = tool_result.get("content", "")
                                if not isinstance(raw_content, str) or not raw_content.strip():
                                    logger.warning("Content returned by fetch_url tool is empty or incorrectly formatted, skipping processing.")
                                else:
                                    # 1. Extract specific purpose for this visit
                                    purpose = ""
                                    try:

                                        args_dict = json.loads(tool_call_original.get("function", {}).get("arguments", "{}"))
  
                                        purpose = args_dict.get("purpose", args_dict.get("goal", ""))
                                    except Exception:
                                        logger.warning("Unable to parse 'purpose' from fetch_url parameters.")

                                    # 2. Use separator to split content returned by multiple pages
                                    pages_content = raw_content.split('\n=======\n')
                                    logger.info(f"Detected fetch_url returned content for {len(pages_content)} pages, will process each one.")

                                    all_summaries = []
                                    # 3. Independently process the content of each page in a loop
                                    for i, page_block in enumerate(pages_content):
                                        if not page_block.strip():
                                            continue

                                        # Extract URL and specific page content
                                        import re
                                        url_match = re.match(r"The content from (https?://[^\s]+):", page_block)
                                        current_url = url_match.group(1) if url_match else f"URL_{i+1}_unknown"

                                        # Build a temporary tool_result containing only single page content
                                        single_page_tool_result = {"status": "success", "content": page_block}

                                        logger.info(f"Processing page {i+1}/{len(pages_content)}: {current_url}")

                                        # 4. Call secondary processor and pass the new purpose parameter
                                        processed_page = process_browser_tool_return(
                                            self.processor_llm_client,
                                            tool_name,
                                            single_page_tool_result,
                                            question=query,
                                            purpose=purpose,
                                            tokenizer_path=self.hf_tokenizer_path
                                        )

                                        summary_text = processed_page.get("content", "Processing failed, no summary returned.")
                                        all_summaries.append(f"URL: {current_url}\n\nSummary:\n{summary_text}")

                                    # 5. Merge all independent summaries into a clear report
                                    final_report = "\n\n---\n\n".join(all_summaries)
                                    tool_result = {"status": "success", "content": final_report}
                                    logger.info("Summaries for all fetch_url pages have been merged.")

                            # --- New addition ends ---
                            else:
                                # For other browser tools like fetch_url, maintain original logic
                                logger.info(f"Executing standard single-page processing for tool '{tool_name}'...")
                                processed_result = process_browser_tool_return(
                                    self.processor_llm_client,
                                    tool_name,
                                    tool_result,
                                    question=query,
                                    tokenizer_path=self.hf_tokenizer_path
                                )
                                tool_result = processed_result

                        elif is_browser_tool:
                            logger.info(f"Browser tool '{tool_name}' detected, but processor is disabled, will use raw output.")

                        tool_call_original["server_id"] = server_id
                        
                        # --- Handle MCP tool response ---
                        if tool_result["status"] == "error":
                            error_msg = tool_result.get("content", {}).get("error", "Unknown error")
                            error_detail = tool_result.get("content", {}).get("detail", "")
                            error_traceback = tool_result.get("content", {}).get("traceback", "")
                            if error_detail: logger.error(f"Tool {full_tool_name} call failed: {error_msg} - Details: {error_detail}")
                            else: logger.error(f"Tool {full_tool_name} call failed: {error_msg}")
                            if error_traceback: logger.debug(f"Error stack: {error_traceback}")
                            error_content = {"error": error_msg, "server_id": server_id}
                            if error_detail: error_content["detail"] = error_detail
                            if error_traceback: error_content["traceback"] = error_traceback
                            tool_call_original["error"] = error_msg
                            tool_call_original["error_detail"] = error_detail
                            tool_call_original["error_traceback"] = error_traceback
                            tool_call_original["response"] = {"status_code": 500, "detail": error_detail or error_msg or error_traceback, "error": error_msg}
                            
                            error_content_str = json.dumps(error_content) 
                            
                            tool_message_to_log = {
                                "role": "user", 
                                "content": f"<tool_response>\n{error_content_str}\n</tool_response>"
                            }
                            
                            
                            full_conversation_log.append(tool_message_to_log)
                            
                            
                            self.historyx.add_message(tool_message_to_log)
                            dialog_like_so_far.append({"role": "user", "content": tool_message_to_log["content"]})
                            # ===============================================================


                            error_tokens = len(self.tokenizer.encode(error_content_str))
                            self.context_tokens += error_tokens
                            logger.info(f"Tool error token count: {int(error_tokens)} (Total: {int(self.context_tokens)})")
                        
                        else: # tool_result["status"] is "success"
                            tool_content = tool_result.get("content", {})
                            logger.info(f"Tool {full_tool_name} call successful")
                            content_to_add = tool_content
                            if isinstance(tool_content, str) and "Page Snapshot" in tool_content:
                                content_to_add = simplify_playwright_snapshot(tool_content)
                                logger.info("Playwright snapshot purification successful.")
                            tool_call_original["response"] = {"status_code": 200, "detail": "", "content": tool_content}
                            if isinstance(content_to_add, (dict, list)):
                                content_to_add = json.dumps(content_to_add, ensure_ascii=False, indent=2)

                            tool_content_str = str(content_to_add) 
                            tool_message_to_log = {
                                "role": "user",
                                "content": f"<tool_response>\n{tool_content_str}\n</tool_response>"
                            }
                            full_conversation_log.append(tool_message_to_log)

                     
                            self.historyx.add_message(tool_message_to_log)
                            dialog_like_so_far.append({"role": "user", "content": tool_message_to_log["content"]})
                            # ===============================================================

                            tool_tokens = self._count_text_tokens(tool_content_str)
                            self.context_tokens += tool_tokens
                            logger.info(f"Tool response token count: {int(tool_tokens)} (total: {int(self.context_tokens)})")
                        tool_calls_made.append(tool_call_original)

                    
            
          
                

            # [New] Check if loop exited due to "reaching maximum rounds"
            max_interactions_reached = interaction_count >= self.max_interactions

            # [New] Check if last message contains answer (relaxed)
            last_message_content = full_conversation_log[-1].get("content", "") if full_conversation_log else ""
            answer_pattern_relaxed = re.compile(r"(?is)(?:<answer>|<answer|answer>)(.*?)(?:</answer>|</answer|/answer>)")
            has_answer = bool(answer_pattern_relaxed.search(last_message_content))

            final_llm_response = llm_response # Default is the last response in the loop

            # [New] Forced summarization logic
            if max_interactions_reached and not has_answer:
                logger.warning(f"Maximum interaction rounds reached ({self.max_interactions}), but no answer found. Forcing model summarization.")
                

                force_answer_prompt = (
                    "You have now reached the maximum interaction limit. "
                    "You MUST stop making tool calls. "
                    "Based on all the information you have gathered so far, "
                    "you must synthesize and provide what you consider the most likely answer "
                    "in the required format: <answer>your answer</answer>"
                )
                

                force_user_message = {"role": "system", "content": force_answer_prompt}
                full_conversation_log.append(force_user_message)
                self.historyx.add_message(force_user_message)


                raw_history_for_final_call = self.historyx.to_raw_history(include_system_prompt=True)
                final_pruned_history = []
                for msg in raw_history_for_final_call:
                    clean_msg = msg.copy()
                    if clean_msg.get("role") == "assistant":
                        if getattr(self, 'return_thought', False):
                            model_thought = clean_msg.get("thought")
                            model_content = clean_msg.get("content")
                            new_content_parts = []
                            if model_thought: new_content_parts.append(f"<think>{model_thought}</think>")
                            if model_content: new_content_parts.append(model_content)
                            if new_content_parts: clean_msg['content'] = "\n".join(new_content_parts)
                        else:
                            if clean_msg.get("tool_calls"): clean_msg['content'] = None
                        if 'thought' in clean_msg: del clean_msg['thought']
                    final_pruned_history.append(clean_msg)

            
                final_llm_response = self.main_llm_client.create_completion(
                    messages=final_pruned_history,
                    temperature=self.main_temperature,          
                    top_p=self.main_top_p,               
                    presence_penalty=self.main_presence_penalty,    
                    max_tokens=self.main_max_tokens
                )
                
           
                final_assistant_message = {
                    "role": "assistant", 
                    "content": final_llm_response.get("response", "No final answer generated."),
                    "thought": final_llm_response.get("thought", "Final synthesis thought.")
                }
                full_conversation_log.append(final_assistant_message)
                self.historyx.add_message(final_assistant_message)

                # ===== [NEW] stream emit after forced final synthesis =====
                dialog_like_so_far.append({
                    "role": "assistant",
                    "content": final_assistant_message.get("content", ""),
                    **({"thought": final_assistant_message.get("thought")} if final_assistant_message.get("thought") else {})

                })
                doc = convert_dialog_step_to_doc(
                    dialog_like_so_far,
                    step=interaction_count,
                    traj_id=traj_id,
                    task_oid=task_oid,
                    model_name=self.model,
                    tools=available_tools,
                    usage=final_llm_response.get("usage") if isinstance(final_llm_response, dict) else None,
                )
                doc["_id"] = {"$oid": trace_oid}
                trace_writer.emit(doc)
                # =====================================================================


                logger.info("Forced summarization call completed.")


            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Use the last response to get logprobs
            llm_logprobs = final_llm_response.get("logprobs", [])
            
            result = {
                "task_id": task_id, "query": query, "success": True,
                "conversation_history": full_conversation_log, 
                "tool_calls": tool_calls_made,
                "execution_time": execution_time, "interactions": interaction_count,
                "max_interactions_reached": max_interactions_reached,
                "logprobs": llm_logprobs
            }
            
            await self.save_result(task_id, result, run_output_dir=str(run_dir), enable_context_split=False)

            try:
                trace_writer.close()
            except Exception:
                pass

            return result
        
        except Exception as e:
            logger.error(f"Error occurred while running test conversation: {str(e)}")
            traceback.print_exc()
            
            # [Modified] Export history from self.historyx even if error occurs
            error_result = {
                "task_id": task_id, "query": query, "success": False, "error": str(e),
                "conversation_history": full_conversation_log,
                "tool_calls": tool_calls_made,
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "interactions": interaction_count
            }

            try:
                await self.save_result(task_id, error_result, run_output_dir=str(run_dir), enable_context_split=False)
            except Exception as save_error:
                logger.error(f"Error saving error results: {str(save_error)}")
            
            return error_result

    async def run_test_from_gaia_data(self, gaia_sample: Dict[str, Any], traj_id: int = 0) -> Dict[str, Any]:
        """
        Run test from GAIA data sample

        Args:
            gaia_sample: GAIA data sample

        Returns:
            Dict[str, Any]: Test result
        """
        # Extract information from GAIA sample
        # Extract task_id, prioritize task_id or id in GAIA sample
        sample_id = gaia_sample.get("task_id", gaia_sample.get("id", "unknown_id"))
        
        # Get query content according to GAIA data format
        # Support multiple possible data formats
        query = None
        

        if "Question" in gaia_sample:
            query = gaia_sample.get("Question", "")
        elif "query" in gaia_sample:
            query = gaia_sample.get("query", "")
        elif "content" in gaia_sample:
            query = gaia_sample.get("content", "")
        elif "instruction" in gaia_sample:
            query = gaia_sample.get("instruction", "")
        elif "messages" in gaia_sample and isinstance(gaia_sample["messages"], list):

            for msg in gaia_sample["messages"]:
                if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
                    query = msg["content"]
                    break

        if not query and isinstance(gaia_sample.get("data", {}), dict):
            data = gaia_sample["data"]
            if "question" in data:
                query = data["question"]
            elif "query" in data:
                query = data["query"]
        

        if not query:
            query = "No query content found, please check GAIA data format"
            logger.warning(f"Query content not found in GAIA sample {sample_id}")

        # Extract context information
        context = gaia_sample.get("context", "")

        # Extract reference code
        reference_code = gaia_sample.get("reference_code", None)

        # Extract reference files
        reference_files = gaia_sample.get("reference_files", None)
        # [Core modification] New addition of file_name field extraction
        file_name = gaia_sample.get("file_name", None)
        logger.info(f"Starting GAIA sample test ID: {sample_id}")
        logger.info(f"Query content: {query[:100]}...")
        

        result = await self.run_test_conversation(
            query=query,
            context=context,
            reference_code=reference_code,
            reference_files=reference_files,
            file_name=file_name,

            task_id=sample_id,  
            traj_id=traj_id, 
            
        )
        

        result["gaia_sample"] = gaia_sample
        result["gaia_id"] = sample_id
        result["task_id"] = sample_id
        
        return result

    async def save_result(self, task_id: str, result: dict, run_output_dir: Optional[str] = None, enable_context_split: bool = False) -> None:
        """
        [Core modification] Save task results.
        - result.json and other log files save [complete] history.
        - dialog.json is [split] and saved to different folders according to __CONTEXT_SPLIT__ markers.
        """
        main_task_dir = Path(run_output_dir).resolve() if run_output_dir else (self.result_dir / str(task_id))
        os.makedirs(main_task_dir, exist_ok=True)

        try:
            # --- 1. Preparation ---
            # Original complete history containing split markers
            full_conversation_log = result.get("conversation_history", [])

            # Create a [clean, no split markers] version of history for saving complete logs like result.json
            clean_history_for_full_log = [msg for msg in full_conversation_log if msg.get("role") != "__CONTEXT_SPLIT__"]
            
            # Create main task directory (e.g., /.../task_id_123)
            os.makedirs(main_task_dir, exist_ok=True)

            # --- 2. Save all log files that need [complete history] ---
            # Create a result copy for saving and replace with clean history
            result_for_saving = result.copy()
            result_for_saving["conversation_history"] = clean_history_for_full_log

            # Save result.json
            result_path = main_task_dir / "result.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result_for_saving, f, ensure_ascii=False, indent=2)
            logger.info(f"Complete result.json saved to {result_path}")
            
            # Save conversation.txt
            conversation_path = main_task_dir / "conversation.txt"
            save_conversation_text(clean_history_for_full_log, conversation_path)
            
            # Save visualized conversation history
            visualized_history = visualize_conversation_history(clean_history_for_full_log)
            visualized_path = main_task_dir / "visual_conversation.txt"
            with open(visualized_path, "w", encoding="utf-8") as f: f.write(visualized_history)
            
            # Save all other logs including meta config, tool calls, LLM interactions, etc.
            system_prompt = next((msg.get("content") for msg in clean_history_for_full_log if msg.get("role") == "system"), None)
            save_meta_config(main_task_dir, self.model, self.provider, len(self.all_tools), system_prompt, self.all_tools)
            if result.get("tool_calls"): save_tool_calls(main_task_dir, result["tool_calls"])
            # save_llm_interactions(main_task_dir, clean_history_for_full_log, self.model, self.provider)
            if result.get("logprobs"):
                logprobs_path = main_task_dir / "logprobs.json"
                with open(logprobs_path, "w", encoding="utf-8") as f: json.dump(result["logprobs"], f, ensure_ascii=False, indent=2)
            
            logger.info(f"All log files depending on complete history saved to main directory: {main_task_dir}")

            # --- 3. [Core logic] Split and save dialog.json ---
            context_split_count = 0
            current_segment = []

            for msg in full_conversation_log:
                if msg.get("role") == "__CONTEXT_SPLIT__":
                    # When encountering split marker, save current segment
                    # Determine the save directory for current segment
                    if context_split_count == 0:
                        current_dir = main_task_dir
                    else:
                        current_dir = self.result_dir / f"{task_id}-context-{context_split_count}"
                    os.makedirs(current_dir, exist_ok=True)
                    
                    # Call _save_dialog_format to save current segment
                    self._save_dialog_format(current_dir, current_segment)
                    
                    # Prepare next segment
                    context_split_count += 1
                    current_segment = msg["next_history_segment"]
                else:
                    # Regular message, add to current segment
                    current_segment.append(msg)
            
            # After loop ends, save the last segment
            if current_segment:
                if context_split_count == 0:
                    last_dir = main_task_dir
                else:
                    last_dir = self.result_dir / f"{task_id}-context-{context_split_count}"
                os.makedirs(last_dir, exist_ok=True)
                self._save_dialog_format(last_dir, current_segment)

        except Exception as e:
            logger.error(f"Serious error occurred while saving results: {str(e)}")
            traceback.print_exc()

    

    def _save_dialog_format(self, task_dir: Path, conversation_segment: List[Dict]) -> None:
        """
        [Modified] Save a [conversation segment] as standard format dialog.json.

        Args:
            task_dir: Target directory to save to
            conversation_segment: List containing only messages from this segment
        """
        try:
            # [Modified] Directly use the passed conversation_segment
            dialog = []
            
            # This internal formatting logic remains basically unchanged
            for msg in conversation_segment:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "user":
                    dialog.append({"role": "user", "content": content})
                elif role == "assistant":
                    assistant_msg = {"role": "assistant", "content": content}
                    model_thought = msg.get("thought")
                    if model_thought:
                        assistant_msg["thought"] = model_thought
                    
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        processed_tool_calls = []
                        for tool_call in tool_calls:
                            tool_name = tool_call.get("function", {}).get("name", "")
                            server_id = tool_call.get("server_id") or self.tool_to_server_map.get(tool_name) or "default"
                            processed_call = {
                                "id": tool_call.get("id", ""),
                                "function": tool_call.get("function", {}),
                                "server_id": server_id
                            }
                            processed_tool_calls.append(processed_call)
                        assistant_msg["tool_calls"] = processed_tool_calls
                    dialog.append(assistant_msg)

                elif role == "tool":
                    tool_call_id = msg.get("tool_call_id", "")
                    tool_name = msg.get("name", "")
                    server_id = self.tool_to_server_map.get(tool_name) or "default"
                    try:
                        tool_content = json.loads(content) if isinstance(content, str) and content.strip().startswith('{') else content
                    except:
                        tool_content = content
                    tool_response = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": tool_content,
                        "server_id": server_id
                    }
                    dialog.append(tool_response)
            
            dialog_path = task_dir / "dialog.json"
            with open(dialog_path, "w", encoding="utf-8") as f:
                json.dump(dialog, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Conversation segment saved to {dialog_path}")

        except Exception as e:
            logger.error(f"Error saving conversation format segment: {str(e)}")
            traceback.print_exc()

    async def run_batch_tests(self, gaia_data_file: str, max_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Run batch tests

        Args:
            gaia_data_file: GAIA data file path
            max_samples: Maximum number of samples

        Returns:
            List[Dict[str, Any]]: List of test results
        """
        # Load data using GaiaDataset
        try:

            if gaia_data_file.endswith('.jsonl'):
  
                gaia_data = []
                with open(gaia_data_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:  
                            gaia_data.append(json.loads(line))
            else:

                path_parts = Path(gaia_data_file).parts
                year = "2023"  
                split = "validation"  
                

                for i, part in enumerate(path_parts):
                    if part in ["2022", "2023", "2024", "2025"]:
                        year = part
                        if i + 1 < len(path_parts):
                            if path_parts[i + 1] in ["validation", "test", "train"]:
                                split = path_parts[i + 1]
                
                # Create GaiaDataset instance
                dataset_dir = str(Path(gaia_data_file).parent.parent.parent)
                gaia_dataset = GaiaDataset(dataset_dir)
                
                # Load dataset
                gaia_data = gaia_dataset.get_dataset(year, split)
                
                # If dataset is empty, try to load file directly
                if not gaia_data:
                    logger.warning(f"Failed to load dataset using GaiaDataset, trying to load file directly")
                    with open(gaia_data_file, "r", encoding="utf-8") as f:
                        gaia_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading GAIA data file: {str(e)}")
            traceback.print_exc()
            return []
        
        # Limit sample count
        if max_samples > 0:
            gaia_data = gaia_data[:max_samples]
        
        # Run tests
        results = []
        for i, sample in enumerate(gaia_data):
            logger.info(f"Running test {i+1}/{len(gaia_data)}")
            
            # Run tests
            result = await self.run_test_from_gaia_data(sample)
            results.append(result)
            
            # Save batch test results
            try:
                save_batch_summary(
                    result_dir=self.result_dir,
                    results=results,
                    model=self.model,
                    provider=self.provider
                )
            except Exception as e:
                logger.error(f"Error saving batch test summary: {str(e)}")
                # Backup method: save results directly
                batch_result_path = self.result_dir / "batch_results.json"
                with open(batch_result_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results

    async def run_multi_worker_tests(
        self, 
        gaia_data_file: str, 
        max_samples: int = 10,
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:

        # Load data using GaiaDataset
        try:
            # Check if it is a JSONL file
            if gaia_data_file.endswith('.jsonl'):
                # JSONL file needs special processing
                gaia_data = []
                with open(gaia_data_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:  
                            gaia_data.append(json.loads(line))
            else:

                path_parts = Path(gaia_data_file).parts
                year = "2023"  
                split = "validation"  
                
                # Try to extract year and split information from file path
                for i, part in enumerate(path_parts):
                    if part in ["2022", "2023", "2024", "2025"]:
                        year = part
                        if i + 1 < len(path_parts):
                            if path_parts[i + 1] in ["validation", "test", "train"]:
                                split = path_parts[i + 1]
                
                
                dataset_dir = str(Path(gaia_data_file).parent.parent.parent)
                gaia_dataset = GaiaDataset(dataset_dir)
                
                
                gaia_data = gaia_dataset.get_dataset(year, split)
                
                
                if not gaia_data:
                    logger.warning(f"Failed to load dataset using GaiaDataset, trying to load file directly")
                    with open(gaia_data_file, "r", encoding="utf-8") as f:
                        gaia_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading GAIA data file: {str(e)}")
            traceback.print_exc()
            return []
        
        
        if max_samples > 0:
            gaia_data = gaia_data[:max_samples]
        
        
        worker_samples = []
        for i in range(max_workers):
            worker_samples.append([])
        
        for i, sample in enumerate(gaia_data):
            worker_idx = i % max_workers
            worker_samples[worker_idx].append(sample)
        
       
        async def worker_task(worker_id: int, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            
            worker_test = GaiaApiTest(
                manager_url=self.mcp_manager.manager_url,
                main_provider=self.provider,
                main_model=self.model,
                main_api_key=self.api_key,
                main_base_url=self.base_url,
                max_interactions=self.max_interactions,
                output_dir=str(self.result_dir / f"worker_{worker_id}"),
                tool_start_tag=self.tool_start_tag,
                tool_end_tag=self.tool_end_tag,
            )
            
            
            if not await worker_test.initialize():
                logger.error(f"Worker thread {worker_id} initialization failed")
                return []
            
            
            worker_results = []
            for i, sample in enumerate(samples):
                logger.info(f"Worker thread {worker_id} running test {i+1}/{len(samples)}")
                
                
                result = await worker_test.run_test_from_gaia_data(sample)
                worker_results.append(result)
            
            
            await worker_test.close()
            
            return worker_results
        
        
        tasks = []
        for i, samples in enumerate(worker_samples):
            if samples: 
                tasks.append(worker_task(i, samples))
        
        
        worker_results = await asyncio.gather(*tasks)
        
        
        all_results = []
        for results in worker_results:
            all_results.extend(results)
        
        
        try:
            save_batch_summary(
                result_dir=self.result_dir,
                results=all_results,
                model=self.model,
                provider=self.provider
            )
        except Exception as e:
            logger.error(f"Error saving batch test summary: {str(e)}")
          
            batch_result_path = self.result_dir / "all_results.json"
            with open(batch_result_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        return all_results


def extract_json(text: str) -> Any:

    text = text.strip()
 
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
   
        json_str = text

        if not (json_str.startswith('{') or json_str.startswith('[')):
             raise ValueError(f"Cannot find JSON. Original content: {text[:200]}...")


    for _ in range(3):
        try:

            obj = json.loads(json_str) 
            if isinstance(obj, str):
                json_str = obj
                continue
            return obj
        except Exception as e:

            last_brace = json_str.rfind('}')
            last_bracket = json_str.rfind(']')
            end_index = max(last_brace, last_bracket)
            
            if end_index > -1:
                truncated_json_str = json_str[:end_index + 1]
                try:
                    obj = json.loads(truncated_json_str)
                    if isinstance(obj, str):
                         json_str = obj
                         continue
                    logger.warning(f"Fixed truncated JSON. Original ending: {json_str[end_index+1:]}")
                    return obj
                except Exception:
                    pass 
            
            raise ValueError(f"Cannot parse as JSON: {e}\nOriginal content: {json_str[:500]}...")
            
    raise ValueError(f"Still not parsed as dict/list after multiple recursions, original content: {json_str[:500]}...")

def extract_last_answer(content: str, answer_schema: str="answer") -> str:
    """
    Extract content using robust split logic.
    Features:
    1. Supports relaxed tags (answer>, <answer, etc.)
    2. Handles missing closing tags (returns content until end of string) - Critical for truncated outputs.
    3. Fixed: Avoids matching start tags inside end tags (e.g., 'answer>' in '</answer>')
    """
    if not content: return ""
    
    if answer_schema == "answer":
        # Try different tag combinations in priority order
        tag_patterns = [
            ("<answer>", "</answer>"),   # Standard format
            ("<answer>", "</answer"),    # Missing >
            ("<answer>", "/answer>"),    # Missing <
            ("<answer", "</answer>"),    # Start tag missing >
            ("<answer", "</answer"),     # Both missing
            ("answer>", "</answer>"),    # Start tag missing <
            ("answer>", "</answer"),     # Start missing <, end missing >
            ("answer>", "/answer>"),     # Both missing <
        ]
    else:
        tag_patterns = [(f"<{answer_schema}>", f"</{answer_schema}>")]
    
    # Try each tag pattern
    for start_tag, end_tag in tag_patterns:
        # Search for start tag from end to beginning
        search_pos = len(content)
        while search_pos > 0:
            start_idx = content.rfind(start_tag, 0, search_pos)
            if start_idx == -1:
                break
            
            # Check if this start tag is part of an end tag
            # e.g., '</answer>' contains 'answer>', need to exclude
            is_inside_end_tag = False
            for et in ["</answer>", "</answer", "/answer>"]:
                et_idx = content.rfind(et, 0, start_idx + len(start_tag))
                if et_idx != -1 and et_idx <= start_idx < et_idx + len(et):
                    is_inside_end_tag = True
                    break
            
            if is_inside_end_tag:
                # Continue searching forward
                search_pos = start_idx
                continue
            
            # Found a valid start tag
            content_start = start_idx + len(start_tag)
            remaining_content = content[content_start:]
            
            # Search for end tag
            end_idx = remaining_content.find(end_tag)
            
            if end_idx != -1:
                # Found complete answer block
                return remaining_content[:end_idx].strip()
            else:
                # No end tag, return content until end of string (handle truncation)
                return remaining_content.strip()
        
    return ""


def update_history_with_procedure(historyx: HistoryX, procedure_obj: Procedure) -> HistoryX:
    """
    Update HistoryX instance with procedure summary.
    (Logic ported from manage_context.py)
    """
    
    replace_range = procedure_obj.replace_history_index
    if not replace_range:
        logger.error("Procedure object missing 'replace_history_index' field, cannot replace history.")
        return historyx
        
    try:
        if "-" in replace_range:
            start_idx, end_idx = map(int, replace_range.split("-"))
            ids_to_replace = list(range(start_idx, end_idx + 1))
        else:
            
            idx = int(replace_range)
            ids_to_replace = [idx]
    except ValueError as e:
        logger.error(f"Failed to parse 'replace_history_index' ({replace_range}): {e}")
        return historyx


    raw_procedures_text = procedure_obj.procedures
    if raw_procedures_text.startswith("replace_history_index:"):
        raw_procedures_text = re.sub(r"^\s*replace_history_index\s*:.*?\n", "", raw_procedures_text, count=1).lstrip()

    content = f"<procedure_summary>\n{raw_procedures_text}\n" 
    

    if procedure_obj.step_goal:
        content += f"\n**Last Step Goal**: {procedure_obj.step_goal}\n"
    if procedure_obj.step_outcome:
        content += f"\n**Last Step Outcome**: {procedure_obj.step_outcome}\n"
    if procedure_obj.step_status:
        content += f"\n**Last Step Status**: {procedure_obj.step_status}\n"
    
    content += """
    **Usage Guidelines**:
    - Review this procedure to understand the context and what has been tried.
    - If a step succeeded, you may build upon it.
    - If a step failed, avoid repeating it unless you have identified the issue and have a different strategy.
    - **Important**: If you find summarized information here useful, retrieve the complete/original information again to avoid working with incomplete summaries.
    </procedure_summary>"""
    

    summary_message = HistoryX.build_message_param({
        "role": "user",
        "content": content
    })
    
 
    try:
        historyx.replace_ids_with_message(ids_to_replace, summary_message)
        logger.info(f"Successfully replaced history record IDs {ids_to_replace} with one summary.")
    except ValueError as e:
         logger.error(f"Error replacing history records using 'replace_ids_with_message': {e}")
    
    return historyx


async def main():

    parser = argparse.ArgumentParser(description="GAIA API Test (AgentCPM-MCP Version)")
    parser.add_argument("--manager-url", default="http://localhost:8000/mcpapi", help="MCPManager API URL")
    parser.add_argument("--gaia-file", help="GAIA data file path")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum number of samples")
    parser.add_argument("--max-interactions", type=int, default=10, help="Maximum conversation rounds")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of worker threads")
    parser.add_argument("--output-dir", help="Result output directory")
    parser.add_argument("--query", help="Single query test")
    

    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"], help="LLM provider")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--base-url", help="API base URL")
    parser.add_argument("--tool-start-tag", type=str, help="Tool call start tag")
    parser.add_argument("--tool-end-tag", type=str, help="Tool call end tag")
    parser.add_argument("--tokenizer-path", help="HuggingFace tokenizer path or model name")
    
    args = parser.parse_args()
    

    test = GaiaApiTest(
        manager_url=args.manager_url,
        main_provider=args.provider,
        main_model=args.model,
        main_api_key=args.api_key,
        main_base_url=args.base_url,
        max_interactions=args.max_interactions,
        output_dir=args.output_dir,
        tool_start_tag=args.tool_start_tag,
        tool_end_tag=args.tool_end_tag,
        hf_tokenizer_path=args.tokenizer_path,
    )

    try:
    
        logger.info("Initializing test...")
        if not await test.initialize():
            logger.error("Test initialization failed")
            return 1
        
        if args.query:
       
            logger.info(f"Running single query test: {args.query}")
            result = await test.run_test_conversation(args.query)
            logger.info(f"Test completed, results saved to: {test.result_dir}")
        elif args.gaia_file:
           
            logger.info(f"Running batch test, data file: {args.gaia_file}")
            
            if args.max_workers > 1:
             
                logger.info(f"Using {args.max_workers} worker threads")
                results = await test.run_multi_worker_tests(
                    args.gaia_file,
                    args.max_samples,
                    args.max_workers
                )
            else:
               
                results = await test.run_batch_tests(
                    args.gaia_file,
                    args.max_samples
                )
                
            logger.info(f"Test completed, {len(results)} results saved to: {test.result_dir}")
        else:
            logger.error("Please provide --query or --gaia-file parameter")
            return 1
        
        return 0
    except KeyboardInterrupt:
        logger.info("User interrupted test")
        return 130
    except Exception as e:
        logger.error(f"Error occurred during testing: {str(e)}")
        traceback.print_exc()
        return 1
    finally:
        
        await test.close()




if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 
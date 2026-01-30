"""
MCP tool-using sampler for rollout.
"""
import asyncio
import json
import uuid
import re
import traceback
from typing import List, Dict, Any, Optional, Literal, Tuple
import openai.types.chat

from log import logger
from configs import AgentTrainingConfig
from beanie import UpdateResponse
from beanie.operators import In
from databases import DispatchedSamplingTask, Record, NoTaskAvailableError, DistributedCounter, DistributedLock
from transformers import AutoTokenizer

from ..sampler import AsyncSampler
from .models import MCPTask
from .mcpapi import MCPAPIHandler, parse_tool, to_dict
from .mcp import MCPHandler
from .context.summary_url import summary_url_content
from .context.discard_all_tools import reset_messages
from ._rewarding import apply_mcp_rewarding_logic


def extract_last_answer(content: str, answer_schema: str = "answer") -> str:
    """
    Extract the last answer from content
    Can be modified to extract various answer schemas.
    
    Supported schemas:
    - "answer" or any XML tag format: <answer>...</answer>
    - "boxed": \\boxed{...} format (for math problems)
    """
    # Support boxed format for math problems
    if answer_schema == "boxed":
        idx = content.rfind("\\boxed{")
        if idx < 0:
            idx = content.rfind("\\fbox{")
            if idx < 0:
                return ""
            start_tag = "\\fbox{"
        else:
            start_tag = "\\boxed{"
        
        start_idx = idx + len(start_tag)
        # Find matching closing brace
        brace_count = 1
        end_idx = start_idx
        while end_idx < len(content) and brace_count > 0:
            if content[end_idx] == "{":
                brace_count += 1
            elif content[end_idx] == "}":
                brace_count -= 1
            end_idx += 1
        
        if brace_count == 0:
            return content[start_idx:end_idx-1].strip()
        return ""
    
    # Default XML tag format
    start_tag = f"<{answer_schema}>"
    end_tag = f"</{answer_schema}>"
    end_idx = content.rfind(end_tag)
    if end_idx == -1:
        return ""
    start_idx = content.rfind(start_tag, 0, end_idx)
    if start_idx == -1:
        return ""
    return content[start_idx + len(start_tag):end_idx]


def filter_huggingface_results(content: str, tool_name: str) -> str:
    """
    过滤搜索结果中包含 huggingface 的条目

    Args:
        content: 工具返回的内容
        tool_name: 工具名称 (目前只处理 search)

    Returns:
        过滤后的内容

    Note:
        fetch_url 的过滤在工具调用之前进行，避免浪费资源
    """
    if not isinstance(content, str) or not content.strip():
        return content

    if tool_name == "search":
        # 使用正则表达式匹配搜索结果的每一项
        # 格式: 数字. [标题](url)
        # 需要匹配从 "数字." 开始到下一个 "数字." 之前的内容

        lines = content.split('\n')
        filtered_lines = []
        skip_until_next_item = False

        for i, line in enumerate(lines):
            # 检测是否是新的结果项（以数字. 开头）
            item_match = re.match(r'^(\d+)\.\s+\[.*?\]\((.*?)\)', line)

            if item_match:
                # 这是一个新的结果项
                item_num = item_match.group(1)
                url = item_match.group(2)

                # 检查 URL 是否包含 huggingface
                if 'huggingface' in url.lower():
                    skip_until_next_item = True
                    continue  # 跳过这一行
                else:
                    skip_until_next_item = False
                    filtered_lines.append(line)
            else:
                # 不是新的结果项
                if not skip_until_next_item:
                    filtered_lines.append(line)

        # 重新编号
        result_lines = []
        current_number = 1

        for line in filtered_lines:
            # 检测是否是结果项的开头
            item_match = re.match(r'^(\d+)(\.\s+\[.*?\]\(.*?\))', line)
            if item_match:
                # 替换编号
                new_line = f"{current_number}{item_match.group(2)}"
                result_lines.append(new_line)
                current_number += 1
            else:
                result_lines.append(line)

        # 更新总结果数量
        filtered_content = '\n'.join(result_lines)

        # 更新第一行的结果数量说明
        # 格式: "A Google search for '...' found X total results (showing top Y):"
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

class MCPSampler(AsyncSampler):
    """
    Sampler for MCP tool-using tasks with context management support.
    """
    
    def __init__(
        self,
        config: AgentTrainingConfig,
        *args,
        task_class: type = MCPTask,
        record_class: type = Record,
        split: Literal["train", "valid", "test"] = "train",
        **kwargs
    ):
        super().__init__(
            config,
            *args,
            task_class=task_class,
            record_class=record_class,
            split=split,
            **kwargs
        )
        # Initialize MCP handler based on the manager URL
        # If URL ends with "/mcpapi", use our custom MCPAPI implementation
        # Otherwise, use official MCP Streamable HTTP protocol
        if config.mcp_manager_url.endswith("/mcpapi"):
            self.mcp_handler = MCPAPIHandler(
                server_name=config.mcp_server_name,
                manager_url=config.mcp_manager_url
            )
        else:
            # Official MCP Streamable HTTP
            self.mcp_handler = MCPHandler(
                server_url=config.mcp_manager_url,
                auth_token=getattr(config, "mcp_auth_token", None)
            )
        
        self.context = {
            "context_tokens": 0,
            "context_turns": 0, # The number of turns in the context （after reset_messages）
            "turns": 0, # The number of turns all
        }
        
        # Tool selection state: reserved for agent's autonomous tool selection capability
        self.chosen_tools = []
        self.available_tools = []
        self.chosen_servers = []
        self.available_servers = []

        # Load agent configs first to check for tokenizer_path
        from .context.agent_config import get_browse_agent_config, get_scorer_agent_config
        agent_config_path = config.agent_config_path
        self.agent_configs = {
            "browse": get_browse_agent_config(agent_config_path),
            "scorer": get_scorer_agent_config(agent_config_path)
        }
        
        # Initialize model tokenizer (for model-related operations)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        # logger.info(f"Using model tokenizer from: {config.model_name_or_path}")
        
        # Initialize browse tokenizer (for browse agent operations)
        # Use tokenizer_path from browse_agent_config if provided, otherwise use model_name_or_path
        tokenizer_path = self.agent_configs["browse"].get("tokenizer_path")
        if tokenizer_path:
            self.browse_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Using browse tokenizer from browse_agent_config: {tokenizer_path}")
        else:
            self.browse_tokenizer = self.tokenizer  # Reuse model tokenizer if not specified
            # logger.info(f"Using model tokenizer for browse operations: {config.model_name_or_path}")
    
    async def __aenter__(self):

        """Enter the asynchronous context manager."""
        # find a task to sample from
        self.task = await self.task_class.find_one(
            self.task_class.num_samples < self.config.num_generations if self.split == "train" else self.task_class.num_samples < 1,
            self.task_class.split == self.split,
            In(self.task_class.status, [self.task_class.Status.WAITING, self.task_class.Status.RUNNING]),
            with_children=True
        ).update(
            {"$inc": {self.task_class.num_samples: 1}, "$set": {self.task_class.status: self.task_class.Status.RUNNING}},
            response_type = UpdateResponse.NEW_DOCUMENT
        )
        
        if self.task is None:
            self.task = await self.task_class.find_one(
                self.task_class.num_samples < self.config.num_generations if self.split == "train" else self.task_class.num_samples < 1,
                self.task_class.split == self.split,
                self.task_class.status == self.task_class.Status.CREATED,
                with_children=True
            ).update(
                {"$inc": {self.task_class.num_samples: 1}, "$set": {self.task_class.status: self.task_class.Status.RUNNING}},
                response_type = UpdateResponse.NEW_DOCUMENT
            )
        
        if self.task is None:
            raise NoTaskAvailableError("No task available for sampling. Please create a task first.")


        self.record = self.record_class(
            task=self.task,
            traj_id=self.task.num_samples - 1,
            split=self.split,
            status=self.record_class.Status.RUNNING,
        )
        # save the record
        await self.record.save()
        
        self.infer_count = await DistributedCounter.create(name=f"infer")
        self.split_lock = await DistributedLock.create(name=self.split)
        self.running_sampler = await DistributedCounter.create(name=f"running-sampler-{self.split}")
        await self.running_sampler.inc()
        self.global_step_counter = await DistributedCounter.create(name="global_step")

        
        # Initialize MCP handler
        # Reserve the training while server errors
        while True:
            status = await self.mcp_handler.initialize()
            if status:
                break
            else:
                await asyncio.sleep(30)
        
        self.available_tools = self.mcp_handler.openai_tools
        self.chosen_tools = self.mcp_handler.openai_tools
        self.available_servers = list(self.mcp_handler.tool_to_server_map.values())
        self.chosen_servers = []
        
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit the async context manager with MCP-specific rewarding logic."""
        if self.mcp_handler:
            await self.mcp_handler.close()
        
        # Handle the basic status logic from parent class
        if self.record is not None:
            # Handle CREATED status (shouldn't happen, but handle it for safety)
            if self.record.status == self.record_class.Status.CREATED:
                if exc_type is None:
                    # If no exception but still CREATED, mark as FAILED
                    self.record.status = self.record_class.Status.FAILED
                    logger.warning(f"Record {self.record.traj_id} for task {self.record.task.id} was CREATED but never started, marking as FAILED")
                else:
                    self.record.status = self.record_class.Status.FAILED
                    logger.warning(f"Record {self.record.traj_id} for task {self.record.task.id} failed with error: {exc_value}")
                # retrying
                await self.task_class.find_one(
                    self.task_class.id == self.task.id, with_children=True
                ).update({
                    "$inc": {"num_samples": -1},
                })
                await self.record.save()
                
            elif self.record.status == self.record_class.Status.RUNNING:
                if exc_type is None:
                    self.record.status = self.record_class.Status.COMPLETED
                else:
                    self.record.status = self.record_class.Status.FAILED
                    logger.warning(f"Record {self.record.traj_id} for task {self.record.task.id} failed with error: {exc_value}")
                    # retrying
                    await self.task_class.find_one(
                        self.task_class.id == self.task.id, with_children=True
                    ).update({
                        "$inc": {"num_samples": -1},
                    })
                await self.record.save()
            
            if self.record.status == self.record_class.Status.COMPLETED:
                # scoring it
                self.record.status = self.record_class.Status.SCORING
                await self.record.save()
                try:
                    score = await self.evaluate_record(self.record)
                except Exception as e:
                    logger.error("Error when calculate score for record "+ str(self.record.id)+"\n"+traceback.format_exc())
                    score = 0.0
                self.record.score = score
                self.record.status = self.record_class.Status.SCORED
                await self.record.save()

                task: MCPTask = await self.task_class.find_one(
                    self.task_class.id == self.task.id, with_children=True
                ).update({
                    "$push": {"scores": score,"turns": self.context["context_turns"]}
                }, response_type=UpdateResponse.NEW_DOCUMENT)
                
                # MCP-specific rewarding logic (similar to tool_rewarding.py)
                if task.split in ["valid", "test"]:
                    # For valid/test, just mark as ready
                    self.record.status = self.record_class.Status.READY
                    await self.record.save()
                else:
                    # For train split, check if we have enough records
                    if len(task.scores) < self.config.num_generations:
                        # Not enough records yet, just mark as scored
                        self.record.status = self.record_class.Status.SCORED
                        await self.record.save()
                    else:
                        # All records collected, apply MCP-specific rewarding logic
                        await apply_mcp_rewarding_logic(
                            task=task,
                            record_class=self.record_class,
                            task_class=self.task_class,
                            config=self.config
                        )
                
                await task.save()
                self.record = None
        
        if self.task is not None:
            self.task = None
        
        # Decrease running sampler count
        if hasattr(self, 'running_sampler'):
            await self.running_sampler.dec()
    
    async def create_chat_completions(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        tools: Optional[List[dict]] = None,
        priority: int = 0,
        timeout: Optional[int] = None,
        repeat_penalty: Optional[float] = 1.0,
        finish_status: Optional[DispatchedSamplingTask.Status] = DispatchedSamplingTask.Status.COMPLETED,
        task_type: Optional[str] = None,
        **kwargs
    ) -> tuple[openai.types.chat.ChatCompletion, List]:
        """
        Create chat completions with tool support and post-processing.
        """
        # Use chosen_tools if tools not provided
        if tools is None:
            tools = self.chosen_tools
        
        # Call parent method
        response = await super().create_chat_completions(
            messages=messages,
            model=model,
            tools=tools,
            priority=priority,
            timeout=timeout,
            finish_status=finish_status,
            **kwargs
        )
        
        # Get the dispatched task
        dispatched_task = await self.record.traj[-1].fetch(True)
        
        # Post-process response for tool calls
        message_content = response.choices[0].message.content
        # Ensure message_content is a string (not None) before processing
        # This is critical because parse_tool may call rfind on content
        if message_content is None:
            message_content = ""
            # Also set it in the response to avoid None issues in parse_tool
            response.choices[0].message.content = ""
        
        if isinstance(message_content, str):
            # Extract reasoning content if present
            idx = message_content.rfind("</think>")
            if idx != -1:
                start_tag = "<think>"
                if message_content.startswith(start_tag):
                    start_idx = len(start_tag)
                else:
                    start_idx = 0
                # Store reasoning content
                reasoning_content = message_content[start_idx:idx].strip()
                if reasoning_content:
                    if not hasattr(response.choices[0].message, 'reasoning_content'):
                        response.choices[0].message.reasoning_content = reasoning_content
                message_content = message_content[idx + len("</think>"):].strip()
        
        # Parse tool calls (content is now guaranteed to be a string, not None)
        tool_calls = parse_tool(response.choices[0].message)
        if len(tool_calls) > 0:
            if not hasattr(response.choices[0].message, 'tool_calls') or response.choices[0].message.tool_calls is None:
                response.choices[0].message.tool_calls = tool_calls
        
        # Extract tool_calls before removing them
        extracted_tool_calls = tool_calls if len(tool_calls) > 0 else []
        
        # Remove tool_call tags from content
        if isinstance(message_content, str):
            idx = message_content.rfind("<tool_call>")
            if idx != -1:
                message_content = message_content[:idx].strip()
        
        # Update message content if changed
        if isinstance(response.choices[0].message.content, str) and message_content != response.choices[0].message.content:
            response.choices[0].message.content = message_content
        
        # Update context
        self.context["context_tokens"] = response.usage.total_tokens
        self.context["context_turns"] += 1
        self.context["turns"] += 1
        
        # Before saving, modify response: remove tool_calls and reasoning_content, use logprobs for content
        # We recommend using logprobs to reconstruct content, because it is always accurate and stable for training.
        if response and hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            
            if choice and hasattr(choice, 'message'):
                # Remove tool_calls and reasoning_content
                if hasattr(choice.message, 'tool_calls'):
                    choice.message.tool_calls = None
                if hasattr(choice.message, 'reasoning_content'):
                    choice.message.reasoning_content = None
                
                # Use logprobs to reconstruct content
                if hasattr(choice, "logprobs") and choice.logprobs and hasattr(choice.logprobs, "content") and choice.logprobs.content:
                    try:
                        token_ids = []
                        token_strings = []
                        
                        for item in choice.logprobs.content:
                            # Try to get token ID first (if available)
                            if hasattr(item, "token") and item.token:
                                token_str = item.token
                                # Try to convert token string to ID using tokenizer
                                try:
                                    # If tokenizer has this token, use it
                                    token_id = self.tokenizer.convert_tokens_to_ids([token_str])[0]
                                    unk_token_id = getattr(self.tokenizer, 'unk_token_id', None)
                                    if unk_token_id is None or token_id != unk_token_id:
                                        token_ids.append(token_id)
                                        token_strings.append(token_str)
                                    else:
                                        # If unknown, try decoding bytes
                                        if hasattr(item, "bytes") and item.bytes:
                                            try:
                                                decoded = bytes(item.bytes).decode('utf-8')
                                                token_strings.append(decoded)
                                            except (UnicodeDecodeError, TypeError):
                                                token_strings.append(token_str)
                                        else:
                                            token_strings.append(token_str)
                                except (KeyError, IndexError, AttributeError):
                                    # Fallback: use bytes or token string directly
                                    if hasattr(item, "bytes") and item.bytes:
                                        try:
                                            decoded = bytes(item.bytes).decode('utf-8')
                                            token_strings.append(decoded)
                                        except (UnicodeDecodeError, TypeError):
                                            token_strings.append(token_str)
                                    else:
                                        token_strings.append(token_str)
                            elif hasattr(item, "bytes") and item.bytes:
                                # Decode bytes to string
                                try:
                                    decoded = bytes(item.bytes).decode('utf-8')
                                    token_strings.append(decoded)
                                except (UnicodeDecodeError, TypeError):
                                    pass
                        
                        # Remove the last token
                        if token_ids:
                            token_ids = token_ids[:-1]
                        if token_strings:
                            token_strings = token_strings[:-1]
                        
                        # Use tokenizer to decode if we have token IDs, otherwise join token strings
                        if token_ids:
                            content = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                        elif token_strings:
                            # Join token strings directly
                            content = "".join(token_strings)
                        else:
                            content = choice.message.content if choice.message.content else ""
                        
                        choice.message.content = content
                    except Exception as e:
                        # Keep original content if logprobs processing fails
                        logger.warning(f"Logprobs processing failed: {e}")
        
        # Save the modified response
        # This field is reserved for multi-agent training.
        dispatched_task.response = response
        if task_type:
            # Store task_type in meta_infos if needed
            if not hasattr(dispatched_task, 'task_type'):
                # Store in request metadata if possible
                pass
        await dispatched_task.save()
        
        # Return response and tool_calls separately
        return response, extracted_tool_calls
    
    async def _reset_messages(self, messages: List[dict], task: MCPTask) -> List[dict]:
        """
        Reset messages based on new_context mode.
        Option 1: discard_all_tools - Clear all tool messages, keep system, user, and assistant messages.
        Option 2: discard_all - Clear all tool messages, generate summary, and replace messages after the 3rd one.
        """
        if self.config.new_context == "disabled":
            return messages
        
        # Reset context tracking for both modes
        self.context["context_tokens"] = 0
        self.context["context_turns"] = 0
        
        # Call unified reset function
        filtered_messages = await reset_messages(
            messages=messages,
            mode=self.config.new_context,
            task_system_prompt=task.system_prompt,
            task_query_prompt=task.query_prompt,
            browse_agent_config=self.agent_configs["browse"] if self.config.new_context == "discard_all" else None,
            tokenizer=self.browse_tokenizer if self.config.new_context == "discard_all" else None
        )
        
        return filtered_messages
    
    def _create_error_tool_result(self, error_msg: str) -> Dict[str, Any]:
        """
        Create a standardized error tool result.
        
        Args:
            error_msg: The error message to include
            
        Returns:
            A dictionary with error status and content
        """
        return {
            "status": "error",
            "content": {
                "error": error_msg
            }
        }
    
    async def _process_mcp_tool_call(
        self,
        tool_call: Dict[str, Any],
        response: Any,
        task: MCPTask,
        messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]], Optional[List[str]]]:
        """
        Process a single MCP tool call.
        
        Args:
            tool_call: The tool call dictionary
            response: The chat completion response
            task: The MCPTask instance
            messages: Current conversation messages
            
        Returns:
            Tuple of (tool_message, updated_messages, updated_chosen_tools, updated_chosen_servers)
            where updated_* can be None if not changed
        """
        # Validate and process tool call
        call_id = tool_call.get("id", "")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"call_{uuid.uuid4().hex}"
        
        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            function = {}
        
        tool_name = function.get("name", "")
        if not isinstance(tool_name, str) or not tool_name.strip():
            # Try to extract error message from arguments if it's a format error from parser
            error_message = "Invalid tool name, skipping"
            error_detail = ""
            try:
                arguments_str = function.get("arguments", "{}")
                arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                if isinstance(arguments, dict) and arguments.get("_format_error"):
                    error_message = arguments.get("error_message", error_message)
            except Exception:
                pass  # If parsing fails, use default error message
            
            logger.error(f"Invalid tool name: {error_message}")
            # Set status to FORMATERROR for this sample when tool call format is invalid
            try:
                # Get the current sample (last one in trajectory)
                current_sample = await self.record.traj[-1].fetch(True)
                current_sample.status = DispatchedSamplingTask.Status.FORMATERROR
                current_sample.advantage = 0
                logger.info(f"Invalid tool call format: {error_message}, set status to FORMATERROR and advantage = 0 for current sample.")
                await current_sample.save()
            except Exception as e:
                logger.error(f"Failed to set status for format error sample: {e}")
            
            # Create error tool_message similar to other tool errors (error/error_detail format)
            error_response = {"error": error_message}
            if error_detail:
                error_response["detail"] = error_detail
            
            tool_message = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "format_error",  # Special name to indicate format error
                "content": json.dumps(error_response)
            }
            return tool_message, None, None, None
        
        # Parse arguments
        try:
            arguments = json.loads(function.get("arguments", "{}"))
        except Exception:
            arguments = {}
        
        # Initialize tool_result to None
        tool_result = None
        
        # Check execute_code for missing or empty code parameter
        # Note: Code extraction from <code> tags is already handled in parse_qwen.py
        # Here we only validate that code exists and is not empty
        if tool_name == "execute_code" or tool_name == "PythonInterpreter":
            # Check if code exists in arguments and is not empty
            if "code" not in arguments or not arguments.get("code") or (isinstance(arguments.get("code"), str) and not arguments.get("code").strip()):
                # Create error tool_result for missing or empty code
                error_msg = (
                    "Error: Invalid tool call. You called 'PythonInterpreter' but failed to provide the Python code inside <code>...</code> tags.\n"
                    "Correct format:\n"
                    "<tool_call>\n{\"name\": \"PythonInterpreter\", \"arguments\": {}}\n"
                    "<code>\n# your code here\nprint(result)\n</code>\n</tool_call>"
                )
                logger.error(f"Tool {tool_name} called without code or with empty code")
                tool_result = self._create_error_tool_result(error_msg)
        
        # Check browse tools for missing purpose parameter
        browse_tools = self.agent_configs["browse"].get("browse_tools", ["fetch_url", "visit"])
        if tool_name in browse_tools and "purpose" not in arguments:
            # First, try to convert goal to purpose if goal exists
            if "url" in arguments:
                if "goal" in arguments:
                    arguments["purpose"] = arguments.pop("goal")
                else:
                    # If no goal, use default purpose from config
                    default_purpose = self.agent_configs["browse"].get("default_purpose", {}).get(
                        tool_name, "Summarize the main content of this page."
                    )
                    arguments["purpose"] = default_purpose
                # Update function arguments
                function["arguments"] = json.dumps(arguments)
                tool_call["function"] = function
        
        if tool_name in browse_tools and "url" in arguments:
            if isinstance(arguments["url"], str):
                arguments["url"] = [arguments["url"]]

        if tool_name in ("search") and "query" in arguments:
            if isinstance(arguments["query"], str):
                arguments["query"] = [arguments["query"]]
        
        # Inject context into tool arguments if needed
        if "messages" in arguments:
            arguments["messages"] = messages
        if "available_tools" in arguments:
            arguments["available_tools"] = self.available_tools
        if "chosen_tools" in arguments:
            arguments["chosen_tools"] = self.chosen_tools
        if "available_servers" in arguments:
            arguments["available_servers"] = self.available_servers
        
        # Map PythonInterpreter to execute_code for actual tool call
        actual_tool_name = "execute_code" if tool_name == "PythonInterpreter" else tool_name
        
        # [新增] 对于 fetch_url，在调用之前检查 URL 是否包含 huggingface
        # 避免浪费资源进行 summary 处理
        skip_tool_call = False
        if tool_name == "fetch_url":
            url_param = arguments.get("url", [])
            # url 可能是字符串或列表
            if isinstance(url_param, str):
                url_param = [url_param]
            elif not isinstance(url_param, list):
                url_param = []

            # 检查是否有任何 URL 包含 huggingface
            filtered_urls = []
            blocked_urls = []
            for url in url_param:
                if isinstance(url, str) and 'huggingface' in url.lower():
                    blocked_urls.append(url)
                    logger.info(f"已阻止访问 huggingface URL: {url}")
                else:
                    filtered_urls.append(url)

            # 如果所有 URL 都被过滤，跳过工具调用
            if blocked_urls and not filtered_urls:
                skip_tool_call = True
                tool_result = {
                    "status": "success",
                    "content": f"This URL has been filtered out (huggingface domain): {', '.join(blocked_urls)}"
                }
                logger.info(f"fetch_url 的所有 URL 均被过滤，跳过工具调用。")
            elif blocked_urls and filtered_urls:
                # 部分 URL 被过滤，更新参数为过滤后的 URL
                arguments["url"] = filtered_urls
                function["arguments"] = json.dumps(arguments)
                tool_call["function"] = function
                logger.info(f"fetch_url 的部分 URL 被过滤: {', '.join(blocked_urls)}，将访问剩余 URL: {', '.join(filtered_urls)}")

        # Only call the tool if tool_result is not already set (e.g., from empty code check)
        if tool_result is None and not skip_tool_call:
            # Call the tool with the actual tool name
            # 空字典重试逻辑已在 mcp_manager.py 中统一处理（包括等待时间和重建连接）
            tool_result = await self.mcp_handler.call_tool(actual_tool_name, arguments)
            
        
        # [新增] 对于 search 工具，在返回后过滤 huggingface 结果
        if tool_name == "search" and tool_result and tool_result.get("status") == "success":
            content = tool_result.get("content", "")
            if content:
                filtered_content = filter_huggingface_results(content, tool_name)
                tool_result["content"] = filtered_content
                # logger.info(f"已对 search 工具的返回结果进行 huggingface 过滤。")
        
        # Special handling for browse tools with summarization
        # Skip summarization if purpose indicates full content is needed
        purpose = arguments.get("purpose", "").strip().lower()
        skip_summary_keywords = ["full content", "complete content", "raw content", "entire content"]
        should_skip_summary = any(keyword in purpose for keyword in skip_summary_keywords)
        summary_enabled = self.agent_configs["browse"].get("status", True)
        if tool_name in browse_tools and summary_enabled and not should_skip_summary:
            try:
                content = tool_result.get("content", str(tool_result))
                content_str = str(content)
                # Use browse_tokenizer to count tokens (for browse agent operations)
                encoded = self.browse_tokenizer.encode(content_str, add_special_tokens=False)
                token_count = len(encoded)
                
                # Clip to max_prompt_length tokens if exceeds (from browse_agent_config)
                max_prompt_length = self.agent_configs["browse"].get("max_prompt_length", 60000)
                if token_count > max_prompt_length:
                    encoded = encoded[:max_prompt_length]
                    content_str = self.browse_tokenizer.decode(encoded, skip_special_tokens=True)
                    content = content_str
                    tool_result["content"] = content
                    token_count = max_prompt_length
                
                # Only summarize if content exceeds min_prompt_length (from browse_agent_config)
                min_prompt_length = self.agent_configs["browse"].get("min_prompt_length", 2000)
                if token_count > min_prompt_length:
                    question = task.query_prompt.split("Your task is to answer the user's question:")[1].split("The filepath")[0].strip() if "Your task is to answer the user's question:" in task.query_prompt else task.query_prompt
                    tool_result = await summary_url_content(
                        content,
                        question,
                        self.agent_configs["browse"],
                        arguments.get("purpose", None),
                        str(self.record.id)
                    )
            except Exception as e:
                logger.error(f"Summary URL content failed: {e}")
        
        
        # Prepare return values
        updated_messages = None
        updated_chosen_tools = None
        updated_chosen_servers = None
        
        if isinstance(tool_result, dict) and tool_result.get("status") == "error":
            error_content = tool_result.get("content", {})
            error_msg = error_content.get("error", "Unknown error") if isinstance(error_content, dict) else "Unknown error"
            error_detail = error_content.get("detail", "") if isinstance(error_content, dict) else ""
            
            # Log error with detail if available
            if error_detail:
                logger.error(f"Tool {tool_name} failed: {error_msg} - {error_detail}")
            else:
                logger.error(f"Tool {tool_name} failed: {error_msg}")
            
            # Check if error is a schema validation error (format error)
            # Simple rule: if error message contains "Failed validating 'required' in schema"
            error_text = f"{error_msg} {error_detail}".lower()
            is_format_error = "failed validating 'required' in schema" in error_text or "failed validating" in error_text and "required" in error_text
            
            try:
                # Get the current sample (last one in trajectory)
                current_sample = await self.record.traj[-1].fetch(True)
                # Only set status if it hasn't been set to FORMATERROR already (e.g., from parser format error)
                # This prevents overwriting a format error detected earlier in the pipeline
                if current_sample.status != DispatchedSamplingTask.Status.FORMATERROR:
                    if is_format_error:
                        # Set status to FORMATERROR for schema validation errors
                        current_sample.status = DispatchedSamplingTask.Status.FORMATERROR
                        current_sample.advantage = 0
                        logger.info(f"Tool {tool_name} schema validation error detected, set status to FORMATERROR and advantage = 0 for current sample.")
                    else:
                        # Set status to TOOLFAILED for other tool errors
                        current_sample.status = DispatchedSamplingTask.Status.TOOLFAILED
                        logger.info(f"Tool {tool_name} failed, set status to TOOLFAILED for current sample.")
                    await current_sample.save()
                else:
                    logger.debug(f"Sample status already set to FORMATERROR (likely from parser), skipping status update for tool error.")
            except Exception as e:
                logger.error(f"Failed to set status for tool error sample: {e}")
            
            # Include both error and detail in the message
            error_response = {"error": error_msg}
            if error_detail:
                error_response["detail"] = error_detail
            
            tool_message = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": json.dumps(error_response)
            }
        else:
            tool_content = tool_result.get("content", {})
            
            # Handle special tool responses
            if isinstance(tool_content, dict):
                if len(tool_content.get("chosen_tools", [])) > 0:
                    updated_chosen_tools = tool_content["chosen_tools"]
                    tool_content.pop("chosen_tools")
                if len(tool_content.get("chosen_servers", [])) > 0:
                    updated_chosen_servers = tool_content["chosen_servers"]
                    tool_content.pop("chosen_servers")
                if len(tool_content.get("messages", [])) > 0:
                    updated_messages = tool_content["messages"]
                    tool_content.pop("messages")
            
            tool_message = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": json.dumps(tool_content) if isinstance(tool_content, (dict, list)) else str(tool_content)
            }
        
        return tool_message, updated_messages, updated_chosen_tools, updated_chosen_servers
    
    async def run(self, task: Optional[MCPTask] = None):
        """
        Run the MCP tool-using sampler.
        """
        if task is None:
            task = self.task
        
        assert isinstance(task, MCPTask), f"task must be a MCPTask, got {type(task)}"
        
        # Initialize context tracking in meta_infos
        self.record.meta_infos["context_tokens"] = 0
        self.record.meta_infos["context_turns"] = 0
        self.record.meta_infos["turns"] = 0
        
        messages = [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": task.query_prompt}
        ]
        
        while self.context["turns"] < self.config.max_turns:
            try:
                response, tool_calls = await self.create_chat_completions(
                    messages=messages,
                    tools=self.chosen_tools,
                    task_type="task"
                )
            except Exception as e:
                logger.error(f"Create chat completions failed: {e}\n{traceback.format_exc()}")
                self.record.meta_infos["final_answer"] = "failed_error"
                self.record.meta_infos["error_detail"] = str(e)
                break
            
            # Always try to extract answer regardless of tool calls
            # If answer is found, break; otherwise continue (with or without tool calls)
            if response.choices and response.choices[0].message.content:
                if isinstance(task.answer_schema, str) and task.answer_schema:
                    final_answer = extract_last_answer(
                        response.choices[0].message.content,
                        task.answer_schema
                    )
                    if final_answer:
                        self.record.meta_infos["final_answer"] = final_answer.strip()
                        break
            
            # Process tool calls (already extracted above)
            if not isinstance(tool_calls, list):
                tool_calls = []
            
            # Convert Pydantic objects to dicts if needed
            tool_calls = [to_dict(tc) if not isinstance(tc, dict) else tc for tc in tool_calls]
            
            assistant_msg = {
                "role": "assistant",
                "content": response.choices[0].message.content
            }
            
            # Update the saved response: remove tool_calls and reasoning_content, use new content
            messages_assistant = [assistant_msg]
            
            # Process tool calls if any
            if len(tool_calls) == 0:
                # No tool calls, just add assistant message and continue
                messages.extend(messages_assistant)
                
                # TODO:(This is a trick for evaluation) If toolcall is empty and no answer extracted, add a user message 
                # if "final_answer" not in self.record.meta_infos or not self.record.meta_infos.get("final_answer"):
                #     messages.append({
                #         "role": "user",
                #         "content": "You didn't provide a tool call or a final answer. Please reassess your plan, think step by step to solve the problem, take the next tool call, or provide the final answer in `<answer></answer>` tags."
                #     })
                
                continue
            
            for tool_call in tool_calls:
                
                # Process the tool call using the class method
                try:
                    tool_message, updated_messages, updated_chosen_tools, updated_chosen_servers = await self._process_mcp_tool_call(
                        tool_call=tool_call,
                        response=response,
                        task=task,
                        messages=messages
                    )
                except Exception as e:
                    logger.error(f"Failed to process tool call: {type(tool_call)}, error: {e}")
                    continue
                
                # Skip if tool_message is None (invalid tool call)
                if tool_message is None:
                    continue
                
                # Update context state if tool returned special responses
                if updated_messages is not None:
                    messages = updated_messages
                if updated_chosen_tools is not None:
                    self.chosen_tools = updated_chosen_tools
                if updated_chosen_servers is not None:
                    self.chosen_servers = updated_chosen_servers
                
                # Add tool message to assistant messages
                messages_assistant.append(tool_message)
                
                # Update context tokens using tokenizer
                # Handle both JSON string and plain string content
                if isinstance(tool_message["content"], str):
                    try:
                        tool_content = json.loads(tool_message["content"])
                    except (json.JSONDecodeError, ValueError):
                        # If content is not valid JSON, use it as plain string
                        tool_content = tool_message["content"]
                else:
                    tool_content = tool_message["content"]
                
                tool_content_str = json.dumps(tool_content) if isinstance(tool_content, (dict, list)) else str(tool_content)
                tool_content_tokens = len(self.tokenizer.encode(tool_content_str, add_special_tokens=False))
                self.context["context_tokens"] += tool_content_tokens
            
            # Add assistant and tool messages first
            messages.extend(messages_assistant)
            
            # Check for repetition: if current response content matches any previous assistant message exactly
            # This check happens AFTER messages.extend, so we check against all messages including the current one
            # For all repetitions (content_count >= 2), set advantage = 0
            # But only trigger early stop when content_count >= 4 
            repetition_detected = False
            content_count = 0
            if response.choices and response.choices[0].message.content:
                current_content = response.choices[0].message.content.strip()
                # Count how many times this content has appeared (including current)
                # Check against all assistant messages (including the one we just added)
                for prev_msg in messages:
                    if prev_msg.get("role") == "assistant":
                        prev_content = prev_msg.get("content", "").strip()
                        if prev_content and current_content == prev_content:
                            content_count += 1
                
                # Set advantage = 0 and status = REPEATERROR for all repeated samples (content_count >= 2)
                # If the calsulated advantage is below 0, it will be overrided by the rewarder.
                if content_count >= 2:
                    try:
                        # Get the current sample (last one in trajectory)
                        current_sample = await self.record.traj[-1].fetch(True)
                        current_sample.advantage = 0
                        current_sample.status = DispatchedSamplingTask.Status.REPEATERROR
                        await current_sample.save()
                        logger.info(f"Detected repetition: response content has appeared {content_count} times, set advantage = 0 and status = REPEATERROR for current sample.")
                    except Exception as e:
                        logger.error(f"Failed to set advantage and status for repeated sample: {e}")
                
                # Trigger early stop based on repetition_early_stop_times config
                # -1 and 0 mean disabled, other values mean stop when content_count >= that value
                if self.config.repetition_early_stop_times > 0 and content_count >= self.config.repetition_early_stop_times:
                    logger.warning(f"Detected repetition: response content has appeared {content_count} times (triggering early stop at {self.config.repetition_early_stop_times}).")
                    repetition_detected = True
            
            # Handle repetition: trigger reset or early stop based on config
            if repetition_detected:
                if self.config.enable_repetition_compress and self.config.new_context != "disabled":
                    # Reset messages using new_context mode instead of exiting
                    logger.info(f"Repetition detected, resetting messages using mode '{self.config.new_context}'.")
                    messages = await self._reset_messages(messages, task)
                    self.record.meta_infos["reset"] = True
                elif self.config.repetition_early_stop_times > 0:
                    # Early stop when repetition is detected
                    logger.warning("Repetition detected, stopping early.")
                    self.record.meta_infos["final_answer"] = "repetition_error"
                    break
            
            # Handle context limit
            if self.context["context_tokens"] > self.config.max_prompt_tokens - 1024:
                if self.config.new_context != "disabled":
                    # Reset messages using new_context mode instead of exiting
                    logger.info(f"Context limit reached, resetting messages using mode '{self.config.new_context}'.")
                    messages = await self._reset_messages(messages, task)
                    self.record.meta_infos["reset"] = True
                else:
                    self.record.meta_infos["final_answer"] = "length_limit_error"
                    break

        
        # Set final answer if not set
        if "final_answer" not in self.record.meta_infos:
            if self.context["turns"] == self.config.max_turns:
                self.record.meta_infos["final_answer"] = "turns_limit_error"
            else:
                self.record.meta_infos["final_answer"] = "failed_error"
        
        # Store right answer and context info in meta_infos
        self.record.meta_infos["right_answer"] = task.task_answer
        self.record.meta_infos["context_tokens"] = self.context["context_tokens"]
        self.record.meta_infos["context_turns"] = self.context["context_turns"]
        self.record.meta_infos["turns"] = self.context["turns"]
        
        return self.record
    
    async def evaluate_record(self, record: Record) -> float:
        """
        Calculate the score for the given record using the MCP-specific evaluation logic.
        Similar to tool_calculate_score in old_codes/tool/tool_rewarding.py,
        this method calculates the score and assigns it to each DispatchedSamplingTask in the trajectory.
        
        Args:
            record: The Record instance to evaluate
            
        Returns:
            The calculated score as a float
        """
        from .scorer import MCPScorerFactory, agentcpm_scorer
        from beanie import Link
        
        scorer_agent_config = self.agent_configs["scorer"]
        
        # Handle both Link and already-fetched task
        if isinstance(record.task, Link):
            task: MCPTask = await record.task.fetch(True)
        else:
            task: MCPTask = record.task
        
        final_answer = record.meta_infos.get("final_answer", "")
        
        # Skip scoring if final_answer is "failed_error"
        if final_answer == "failed_error":
            score = 0.0
        else:
            # Get right_answer from meta_infos or task.task_answer
            right_answer = record.meta_infos.get("right_answer", task.task_answer)
            
            # Ensure right_answer is a valid dict
            if not isinstance(right_answer, dict):
                if isinstance(right_answer, str):
                    right_answer = {"query": "", "answer": right_answer}
                else:
                    right_answer = {"query": "", "answer": ""}
            
            # Ensure right_answer has required fields
            if "query" not in right_answer:
                right_answer["query"] = ""
            if "answer" not in right_answer:
                right_answer["answer"] = ""
            
            # Get scorer name, default to "agentcpm"
            scorer_name = getattr(task, "scorer", None) or "agentcpm"
            if not scorer_name or not isinstance(scorer_name, str):
                scorer_name = "agentcpm"
            
            try:
                # For other scorers, try to get from factory and call with config if supported
                scorer = MCPScorerFactory.get_scorer(scorer_name)
                # Try to call with config if it accepts 3 args, otherwise fallback to agentcpm
                import inspect
                sig = inspect.signature(scorer)
                if len(sig.parameters) >= 3:
                    score = await scorer(final_answer, right_answer, scorer_agent_config)
                else:
                    # Fallback to agentcpm_scorer if the scorer doesn't support config
                    logger.warning(f"Scorer '{scorer_name}' doesn't support config parameter, using 'agentcpm' scorer instead")
                    score = await agentcpm_scorer(final_answer, right_answer, scorer_agent_config)
            except Exception as e:
                logger.error(f"[evaluate_record] Error calculating score for record {record.id}: {e}")
                logger.error(traceback.format_exc())
                score = 0.0
        
        # Assign score to each DispatchedSamplingTask in the trajectory
        for i in range(len(record.traj)):
            sample = await record.traj[i].fetch(True)
            sample.score = score
            await sample.save()
        
        return float(score)


# ============================================================================
# Error Types for records (Record Level)
# ============================================================================
# The sampler can set final_answer to one of the following error types:
#
# 1. "repetition_error": Set when repetition is detected and early stop is enabled
#    (repetition_early_stop_times > 0, content_count >= repetition_early_stop_times)
#
# 2. "context_limit_error": Set when context token limit is reached and new_context
#    is disabled (context_tokens > max_prompt_tokens - 1024, new_context="disabled")
#
# 3. "turns_limit_error": Set when max_turns is reached without finding an answer
#    (turns == max_turns, no final_answer extracted)
#
# 4. "failed_error": Default error type set when no answer is found and no other
#    error conditions are met (fallback case), mostly due to environment errors.
#
# Note: If create_chat_completions raises an exception, final_answer is set to
#       "failed_error" and the exception message is stored in meta_infos["error_detail"].
# ============================================================================

# ============================================================================
# Error Types for DispatchedSamplingTask (Sample Level)
# ============================================================================
# The sampler can set sample status to one of the following error types:
#
# 1. FORMATERROR: Set when tool call format is invalid (tool_name is empty or invalid)
#    - Sets advantage = 0
#
# 2. REPEATERROR: Set when repetition is detected (content_count >= 2)
#    - Sets advantage = 0
#    - Note: Early stop is triggered when content_count >= 4
#
# 3. TOOLFAILED: Set when tool execution returns an error status
#    - Does NOT set advantage = 0 (advantage may be overridden by rewarder)
#    - Location: _process_mcp_tool_call() when tool_result.get("status") == "error"
#
# Note: These error types are set at the DispatchedSamplingTask level and affect
#       individual samples in the trajectory, not the entire record.
#       When advantage = 0, the sample's advantage is capped at 0, meaning it
#       will not participate in positive training (advantage > 0) but will still
#       participate in negative training (advantage <= 0).
# ============================================================================


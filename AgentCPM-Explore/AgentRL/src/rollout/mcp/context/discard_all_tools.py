"""
Handle context reset and summarization functionality for discard_all_tools and discard_all modes.
"""
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Literal
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


async def reset_messages_discard_tools(
    messages: List[dict],
    task_system_prompt: str,
    task_query_prompt: str
) -> List[dict]:
    """
    Reset messages (discard_all_tools mode): Remove all tool messages, keep system, user and assistant messages.
    
    Processing flow:
    1. Filter out all messages with role="tool"
    2. Delete last min(15, length-10) messages (keep at least 10)
    3. Ensure system and user prompts exist
    4. Return filtered message list
    
    Args:
        messages: Original message list
        task_system_prompt: Task system prompt
        task_query_prompt: Task query prompt
        
    Returns:
        Reset message list
    """
    # Count messages
    tool_count = sum(1 for msg in messages if msg.get("role") == "tool")
    assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
    system_count = sum(1 for msg in messages if msg.get("role") == "system")
    user_count = sum(1 for msg in messages if msg.get("role") == "user")
    
    # Filter out all tool messages
    filtered_messages = [
        msg for msg in messages 
        if msg.get("role") != "tool"
    ]
    
    # Calculate number of removed tool messages
    removed_count = len(messages) - len(filtered_messages)
    kept_before_add = len(filtered_messages)
    
    # Delete last min(15, length-10) messages (keep at least 10)
    delete_count = min(15, max(0, len(filtered_messages) - 10))
    deleted_recent = 0
    if delete_count > 0:
        deleted_recent = delete_count
        filtered_messages = filtered_messages[:-delete_count]
        logger.info(f"Deleted last {deleted_recent} messages after tool filtering (kept {len(filtered_messages)} messages).")
    
    # Ensure system and user prompts exist
    has_system = any(msg.get("role") == "system" for msg in filtered_messages)
    has_user = any(msg.get("role") == "user" for msg in filtered_messages)
    
    added_count = 0
    if not has_system:
        filtered_messages.insert(0, {"role": "system", "content": task_system_prompt})
        added_count += 1
    if not has_user:
        # Find last user message or use query_prompt
        last_user_msg = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        if last_user_msg:
            filtered_messages.append(last_user_msg)
        else:
            filtered_messages.append({"role": "user", "content": task_query_prompt})
        added_count += 1
    
    # Log statistics
    ratio_info = f"ratio assistant:tool = {assistant_count}:{tool_count}"
    if assistant_count > 0:
        ratio_info += f" ({tool_count/assistant_count:.2f} tools per assistant)"
    delete_info = f", deleted {deleted_recent} recent messages" if deleted_recent > 0 else ""
    logger.info(
        f"Reset messages using mode 'discard_all_tools': removed {removed_count} tool messages (was {tool_count}), "
        f"kept {kept_before_add} messages (system:{system_count}, user:{user_count}, assistant:{assistant_count}), "
        f"{ratio_info}{delete_info}"
        + (f", added {added_count} missing prompt(s)" if added_count > 0 else "")
        + f", total {len(filtered_messages)} messages after reset."
    )
    
    return filtered_messages


async def reset_messages_with_summary(
    messages: List[dict],
    task_system_prompt: str,
    task_query_prompt: str,
    tokenizer: Any,
    browse_agent_config: Dict[str, Any]
) -> List[dict]:
    """
    Reset messages and generate summary.
    
    Processing flow:
    1. Filter out all tool messages
    2. Call summary model to summarize all findings
    3. Return message list replacing 3rd message and onwards (keep first 2, replace 3rd and onwards with summary)
    
    Args:
        messages: Original message list
        task_system_prompt: Task system prompt
        task_query_prompt: Task query prompt
        tokenizer: tokenizer for calculating token count
        
    Returns:
        Reset message list
    """
    # Count messages
    tool_count = sum(1 for msg in messages if msg.get("role") == "tool")
    assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
    system_count = sum(1 for msg in messages if msg.get("role") == "system")
    user_count = sum(1 for msg in messages if msg.get("role") == "user")
    
    # Filter out all tool messages
    filtered_messages = [
        msg for msg in messages 
        if msg.get("role") != "tool"
    ]
    
    removed_count = len(messages) - len(filtered_messages)
    kept_before_add = len(filtered_messages)
    
    # Ensure system and user prompts exist
    has_system = any(msg.get("role") == "system" for msg in filtered_messages)
    has_user = any(msg.get("role") == "user" for msg in filtered_messages)
    
    added_count = 0
    if not has_system:
        filtered_messages.insert(0, {"role": "system", "content": task_system_prompt})
        added_count += 1
    if not has_user:
        # Find last user message or use query_prompt
        last_user_msg = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        if last_user_msg:
            filtered_messages.append(last_user_msg)
        else:
            filtered_messages.append({"role": "user", "content": task_query_prompt})
        added_count += 1
    
    # Prepare summary content: collect all filtered message content
    # Extract content from all assistant and tool messages in original messages for summarization
    summary_content_parts = []
    for msg in messages:
        role = msg.get("role", "")
        if role in ["assistant", "tool"]:
            content = msg.get("content", "")
            if content:
                if role == "assistant":
                    summary_content_parts.append(f"[Assistant]: {content}")
                elif role == "tool":
                    # Try to parse JSON content of tool message
                    try:
                        if isinstance(content, str):
                            tool_data = json.loads(content)
                            summary_content_parts.append(f"[Tool {msg.get('name', 'unknown')}]: {json.dumps(tool_data, ensure_ascii=False)}")
                        else:
                            summary_content_parts.append(f"[Tool {msg.get('name', 'unknown')}]: {str(content)}")
                    except:
                        summary_content_parts.append(f"[Tool {msg.get('name', 'unknown')}]: {str(content)}")
    
    # Call summary model to generate summary
    summary_text = await generate_summary(summary_content_parts, task_query_prompt, browse_agent_config)
    
    # Build final message list: keep first 2, then replace 3rd and onwards with summary
    final_messages = []
    
    # Keep first 2 messages (usually system, user)
    keep_count = min(2, len(filtered_messages))
    final_messages = filtered_messages[:keep_count].copy()
    
    # Add summary as an assistant message (replacing 3rd message and onwards)
    final_messages.append({
        "role": "assistant",
        "content": summary_text
    })
    
    # Log statistics
    ratio_info = f"ratio assistant:tool = {assistant_count}:{tool_count}"
    if assistant_count > 0:
        ratio_info += f" ({tool_count/assistant_count:.2f} tools per assistant)"
    logger.info(
        f"Reset messages using mode 'discard_all': removed {removed_count} tool messages (was {tool_count}), "
        f"kept {kept_before_add} messages (system:{system_count}, user:{user_count}, assistant:{assistant_count}), "
        f"{ratio_info}"
        + (f", added {added_count} missing prompt(s)" if added_count > 0 else "")
        + f", total {len(final_messages)} messages after reset (kept first {keep_count}, replaced from 3rd message onwards with summary)."
    )
    
    return final_messages


async def generate_summary(content_parts: List[str], question: str, browse_agent_config: Dict[str, Any]) -> str:
    """
    Use summary model to generate Markdown format summary.
    
    Args:
        content_parts: List of content parts to summarize
        question: Task question, for guiding summary
        browse_agent_config: browse_agent config dict, contains models list and other config
        
    Returns:
        Markdown format summary text
    """
    if not content_parts:
        return "## Summary\n\nNo content to summarize."
    
    # Merge all content
    full_content = "\n\n".join(content_parts)
    
    # Build prompt
    system_prompt = """You are an intelligent summarization assistant. Your task is to summarize all the findings and actions from the conversation history.

Please provide a concise summary in Markdown format that includes:
1. **Actions**: Actions taken by the agent
2. **Results**: Results of the actions
3. **Current Status**: What has been accomplished and what remains

Format your response as clean Markdown without code blocks."""
    
    user_prompt = f"""Please summarize all the findings from the following conversation history. The original task question is: "{question}"

## Conversation History
{full_content}

Please provide a concise summary in Markdown format covering all key findings, actions, and results."""
    
    models = browse_agent_config.get("models", [])
    max_retries = browse_agent_config.get("max_retries", 3)
    retry_delay = browse_agent_config.get("retry_delay", 2)
    timeout = browse_agent_config.get("timeout", 100)
    
    if not models:
        error_msg = "No models configured in browse_agent config"
        logger.error(error_msg)
        return f"## Summary\n\nError: {error_msg}"
    
    # Try each model in config, in order
    for i, model_config in enumerate(models):
        api_key = model_config.get("api_key")
        base_url = model_config.get("base_url")
        model_name = model_config.get("model")
        
        if not api_key or not base_url or not model_name:
            logger.warning(f"Model {i+1} in browse_agent config is missing required fields (api_key, base_url, or model), skipping")
            continue
        
        try:
            llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"Trying browse_agent model {i+1} to generate summary: {model_name}")
            
            current_retry_delay = retry_delay
            for attempt in range(max_retries):
                try:
                    response = await llm_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=model_name,
                        timeout=timeout
                    )
                    
                    summary_text = response.choices[0].message.content
                    
                    # Clean possible Markdown code block markers
                    if summary_text.strip().startswith("```markdown"):
                        summary_text = summary_text.strip()[len("```markdown"):]
                    if summary_text.strip().startswith("```"):
                        summary_text = summary_text.strip()[3:]
                    if summary_text.strip().endswith("```"):
                        summary_text = summary_text.strip()[:-3]
                    
                    return summary_text.strip()
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Summary generation failed (attempt {attempt + 1}/{max_retries}) using model {i+1} ({model_name}): {e}, retrying in {current_retry_delay}s...")
                        await asyncio.sleep(current_retry_delay)
                        current_retry_delay *= 2
                    else:
                        logger.warning(f"Summary generation failed after {max_retries} attempts using model {i+1} ({model_name}): {e}")
                        break
        except Exception as e:
            logger.warning(f"Browse agent model {i+1} ({model_name}) call exception: {e}")
            continue
    
    # All models failed
    error_msg = f"Failed to generate summary after trying all {len(models)} models in browse_agent config"
    logger.error(error_msg)
    return f"## Summary\n\nError: {error_msg}"


async def reset_messages(
    messages: List[dict],
    mode: Literal["discard_all_tools", "discard_all"],
    task_system_prompt: str,
    task_query_prompt: str,
    browse_agent_config: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[Any] = None
) -> List[dict]:
    """
    Unified message reset function, selects different processing based on mode.
    
    Args:
        messages: Original message list
        mode: Reset mode, "discard_all_tools" or "discard_all"
        task_system_prompt: Task system prompt
        task_query_prompt: Task query prompt
        browse_agent_config: browse_agent config dict (only needed for discard_all mode)
        tokenizer: tokenizer (only needed for discard_all mode, but currently unused)
        
    Returns:
        Reset message list
    """
    if mode == "discard_all_tools":
        return await reset_messages_discard_tools(
            messages=messages,
            task_system_prompt=task_system_prompt,
            task_query_prompt=task_query_prompt
        )
    elif mode == "discard_all":
        if browse_agent_config is None:
            raise ValueError("browse_agent_config is required for 'discard_all' mode")
        return await reset_messages_with_summary(
            messages=messages,
            task_system_prompt=task_system_prompt,
            task_query_prompt=task_query_prompt,
            tokenizer=tokenizer,
            browse_agent_config=browse_agent_config
        )
    else:
        raise ValueError(f"Unknown reset mode: {mode}. Must be 'discard_all_tools' or 'discard_all'")


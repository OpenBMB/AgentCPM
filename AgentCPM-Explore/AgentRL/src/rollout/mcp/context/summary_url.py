import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncOpenAI
import re
import asyncio
import uuid
from pathlib import Path


def create_specialist_prompt(raw_content: str, question: Optional[str] = None, purpose: Optional[str] = None) -> List[Dict]:
    """Create context-aware, goal-oriented prompts for the 'webpage analysis expert' model."""
    
    user_goal_section = ""
    if purpose:
        user_goal_section = f"The purpose for visiting this page is: {purpose}"# \n The global task is: {question}" 
    else:
        user_goal_section = "No specific user goal provided."

    user_prompt = f"""
        Please process the following webpage or local file content and user goal to extract relevant information:

        ## **Webpage/Local file Content** {raw_content}

        ## **User Goal**
        {user_goal_section}

        ## **Task Guidelines**
        1. **Rational & Step-by-Step Analysis**: 
           - Scan the content to locate specific sections directly related to the user's goal.
           - Perform a **step-by-step analysis** to evaluate *why* this information is relevant and how it addresses the user's specific request.
        2. **Key Extraction for Evidence**: Identify and extract the **most relevant information**. Output the **full original context** (can be more than three paragraphs).
        3. **Strict No-Calculation & Literal Extraction Policy**:
            - **Rule**: You are strictly prohibited from performing any math (sum, average, etc.). You must act as a copy-paste tool for numbers.
            - **Positive Example (Do This)**:
              * *Source Text*: "Base Salary: $50,000. Annual Bonus: $10,000."
              * *Your Output Must Be*: **"Base Salary: $50,000; Annual Bonus: $10,000"** (List them separately).
            - **Negative Example (Don't Do This)**:
              * *Wrong Output*: "Total Compensation: $60,000" (Do NOT sum them up).
            - **Instruction**: Even if the user asks for a "Total" or "Result", providing the separate raw numbers found in the text is the ONLY correct answer.
        4. **Report Output**: Based on the analysis above, organize the findings into a formal **Report**. Create a logical narrative that answers the user's goal using the extracted evidence.

        **Final Output Format: You MUST use Markdown with the following headings:**
        ## Rational
        (Your step-by-step analysis of the content and its relevance to the user goal)

        ## Evidence
        (Your extracted evidence here. *Note: List only the raw, separated data exactly as it appears in the text.*)

        ## Report
        (Your final organized report here)
        """

    return [{"role": "user", "content": user_prompt}]


async def _try_llm_call(
    llm_client: AsyncOpenAI,
    messages: List[Dict],
    model: str,
    max_retries: int = 3,
    retry_delay: int = 2,
    task_id: Optional[str] = None,
    enable_logging: bool = False,
    extra_body: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], Optional[Exception]]:
    """
    Attempt to call LLM, returns (response_text, error)
    If successful returns (text, None), if failed returns (None, exception)
    
    Args:
        llm_client: OpenAI client
        messages: Message list
        model: Model name
        max_retries: Maximum retry count
        retry_delay: Retry delay (seconds)
        task_id: Task ID (optional), for logging
        enable_logging: Whether to log requests and responses to file
    """
    current_retry_delay = retry_delay
    
    # Use extra_body from config if provided, otherwise use default values
    if extra_body is None:
        extra_body = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    
    for attempt in range(max_retries):
        try:
            response = await llm_client.chat.completions.create(
                messages=messages, 
                model=model,
                timeout=180,
                extra_body=extra_body,
            )
            full_response_text = response.choices[0].message.content
            
            # Log request and response to file (if enabled)
            if enable_logging:
                try:
                    request_id = str(uuid.uuid4())
                    folder_name = task_id if task_id else request_id
                    log_dir = Path(f"./fetch_url/{model}") / folder_name
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_file = log_dir / f"{request_id}.json"
                    log_data = {
                        "request": {
                            "messages": messages,
                            "model": model,
                        },
                        "response": response.model_dump()
                    }
                    with open(log_file, "w", encoding="utf-8") as f:
                        json.dump(log_data, f, ensure_ascii=False, indent=2)
                except Exception as log_e:
                    logging.warning(f"Failed to log request/response: {log_e}")
            
            # Clean possible Markdown code block markers
            if full_response_text.strip().startswith("```markdown"):
                full_response_text = full_response_text.strip()[len("```markdown"):]
            if full_response_text.strip().endswith("```"):
                full_response_text = full_response_text.strip()[:-len("```")]

            return full_response_text.strip(), None
                
        except Exception as e:
            if attempt < max_retries - 1:  # Not the last attempt
                logging.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {current_retry_delay}s...")
                await asyncio.sleep(current_retry_delay)
                current_retry_delay *= 2  # Exponential backoff: 2s, 4s
            else:  # Last attempt also failed
                logging.error(f"Error: Expert model call failed after {max_retries} retries: {e}", exc_info=True)
                return None, e
    
    return None, Exception(f"All {max_retries} attempts failed")


async def process_with_llm(
    raw_content: str, 
    question: str, 
    browse_agent_config: Dict[str, Any],
    purpose: Optional[str] = None, 
    task_id: Optional[str] = None
) -> str:
    """Process content with LLM and return pure Markdown report. Uses multiple models from config, tries in order, falls back to next on failure."""
    
    messages = create_specialist_prompt(raw_content, question, purpose)
    models = browse_agent_config.get("models", [])
    max_retries = browse_agent_config.get("max_retries", 3)
    retry_delay = browse_agent_config.get("retry_delay", 2)
    enable_logging = browse_agent_config.get("enable_logging", False)
    extra_body = browse_agent_config.get("extra_body", None)
    
    if not models:
        error_msg = "No models configured in browse_agent config"
        logging.error(error_msg)
        return f"## Error\n{error_msg}"
    
    # Try each model in config, in order
    for i, model_config in enumerate(models):
        api_key = model_config.get("api_key")
        base_url = model_config.get("base_url")
        model_name = model_config.get("model")
        
        if not api_key or not base_url or not model_name:
            logging.warning(f"Model {i+1} in browse_agent config is missing required fields (api_key, base_url, or model), skipping")
            continue
        
        try:
            llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            logging.info(f"Trying browse_agent model {i+1}: {model_name}")
            response_text, error = await _try_llm_call(
                llm_client, messages, model_name, max_retries, retry_delay, task_id, enable_logging, extra_body
            )
            if response_text is not None:
                return response_text
            logging.warning(f"Browse agent model {i+1} ({model_name}) call failed: {error}")
        except Exception as e:
            logging.warning(f"Browse agent model {i+1} ({model_name}) call exception: {e}")
            continue
    
    # All models failed
    error_msg = f"LLM call failed: all {len(models)} models in browse_agent config failed after {max_retries} attempts each"
    logging.error(error_msg)
    return f"## Error\n{error_msg}"


async def summary_url_content(
    tool_content: Any, 
    question: str, 
    browse_agent_config: Dict[str, Any],
    purpose: Optional[str] = None, 
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Summarize fetch_url return content, generate summaries concurrently by page and merge into one report.
    - tool_content: Can be str or dict; if dict, prefer 'content' or longer string value.
    - question: Main task question, for guiding summary focus.
    - browse_agent_config: browse_agent config dict, contains models list and other config.
    - purpose: Immediate intent for this visit (optional), further focus summary.
    - task_id: Task ID (optional), for logging.
    Returns: {"status": "success", "content": final_report}
    """
    
    # 1) Normalize to string raw_content
    raw_content = ""
    if isinstance(tool_content, dict):
        raw_content = tool_content.get("content", "")
        if not raw_content:
            # Fallback strategy: pick the longest string value from dict
            raw_content = max((v for v in tool_content.values() if isinstance(v, str)), key=len, default="")
    elif isinstance(tool_content, str):
        raw_content = tool_content

    if not raw_content.strip():
        return {"status": "success", "content": "No textual content to summarize."}

    # 2) Split by fetch_url multi-page separator
    pages_content = raw_content.split("\n=======\n")

    async def process_single_page(i: int, page_block: str) -> Optional[str]:
        block = page_block.strip()
        if not block:
            return None

        url_match = re.match(r"The content from (https?://[^\s]+):", block)
        current_url = url_match.group(1) if url_match else f"URL_{i+1}_unknown"

        try:
            summary_text = await process_with_llm(block, question, browse_agent_config, purpose, task_id)
        except Exception as e:
            summary_text = f"## Error\nSummarization failed: {e}"

        return f"URL: {current_url}\n\nSummary:\n{summary_text}"

    # 3) Process all pages concurrently
    tasks = [process_single_page(i, page_block) for i, page_block in enumerate(pages_content)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 4) Filter and merge results
    summaries = [r for r in results if r and not isinstance(r, Exception)]
    final_report = "\n\n---\n\n".join(summaries) if summaries else "No page content detected."
    
    return {"status": "success", "content": final_report}
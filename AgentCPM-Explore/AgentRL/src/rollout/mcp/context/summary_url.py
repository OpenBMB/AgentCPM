import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncOpenAI
import re
import asyncio
import uuid
from pathlib import Path


def create_specialist_prompt(raw_content: str, question: Optional[str] = None, purpose: Optional[str] = None) -> List[Dict]:
    """为"网页分析专家"模型创建带有上下文的、目标导向的提示词。"""
    
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
    尝试调用LLM，返回 (response_text, error)
    如果成功返回 (text, None)，如果失败返回 (None, exception)
    
    Args:
        llm_client: OpenAI 客户端
        messages: 消息列表
        model: 模型名称
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        task_id: 任务ID（可选），用于日志记录
        enable_logging: 是否记录请求和响应到文件
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
            
            # 记录请求和响应到文件（如果启用）
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
            
            # 清理可能的 Markdown 代码块标记
            if full_response_text.strip().startswith("```markdown"):
                full_response_text = full_response_text.strip()[len("```markdown"):]
            if full_response_text.strip().endswith("```"):
                full_response_text = full_response_text.strip()[:-len("```")]

            return full_response_text.strip(), None
                
        except Exception as e:
            if attempt < max_retries - 1:  # 不是最后一次尝试
                logging.warning(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {e}，{current_retry_delay}秒后重试...")
                await asyncio.sleep(current_retry_delay)
                current_retry_delay *= 2  # 指数退避：2s, 4s
            else:  # 最后一次尝试也失败了
                logging.error(f"错误：专家模型调用失败，已重试{max_retries}次: {e}", exc_info=True)
                return None, e
    
    return None, Exception(f"All {max_retries} attempts failed")


async def process_with_llm(
    raw_content: str, 
    question: str, 
    browse_agent_config: Dict[str, Any],
    purpose: Optional[str] = None, 
    task_id: Optional[str] = None
) -> str:
    """使用LLM处理内容，并返回纯粹的 Markdown 报告。使用配置中的多个模型列表，按顺序尝试，失败时使用下一个保底。"""
    
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
    
    # 尝试配置中的每个模型，按顺序
    for i, model_config in enumerate(models):
        api_key = model_config.get("api_key")
        base_url = model_config.get("base_url")
        model_name = model_config.get("model")
        
        if not api_key or not base_url or not model_name:
            logging.warning(f"Model {i+1} in browse_agent config is missing required fields (api_key, base_url, or model), skipping")
            continue
        
        try:
            llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            logging.info(f"尝试使用 browse_agent model {i+1}: {model_name}")
            response_text, error = await _try_llm_call(
                llm_client, messages, model_name, max_retries, retry_delay, task_id, enable_logging, extra_body
            )
            if response_text is not None:
                return response_text
            logging.warning(f"Browse agent model {i+1} ({model_name}) 调用失败: {error}")
        except Exception as e:
            logging.warning(f"Browse agent model {i+1} ({model_name}) 调用异常: {e}")
            continue
    
    # 所有模型都失败了
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
    汇总 fetch_url 的返回内容，按页面并发生成摘要并合并为一个报告。
    - tool_content: 可能是 str 或 dict；若为 dict，优先取其中的 'content' 或较长的字符串值。
    - question: 主任务问题，用于指引摘要焦点。
    - browse_agent_config: browse_agent 配置字典，包含 models 列表和其他配置。
    - purpose: 本次访问的即时意图（可选），进一步聚焦摘要。
    - task_id: 任务ID（可选），用于日志记录。
    返回: {"status": "success", "content": final_report}
    """
    
    # 1) 归一化为字符串 raw_content
    raw_content = ""
    if isinstance(tool_content, dict):
        raw_content = tool_content.get("content", "")
        if not raw_content:
            # 退化策略：从 dict 里挑最长的字符串值
            raw_content = max((v for v in tool_content.values() if isinstance(v, str)), key=len, default="")
    elif isinstance(tool_content, str):
        raw_content = tool_content

    if not raw_content.strip():
        return {"status": "success", "content": "No textual content to summarize."}

    # 2) 按 fetch_url 的多页分隔符拆分
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

    # 3) 并发处理所有页面
    tasks = [process_single_page(i, page_block) for i, page_block in enumerate(pages_content)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 4) 过滤并合并结果
    summaries = [r for r in results if r and not isinstance(r, Exception)]
    final_report = "\n\n---\n\n".join(summaries) if summaries else "No page content detected."
    
    return {"status": "success", "content": final_report}
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Browser Return Value Processing Agent (Browser Processor Agent)
"""

import logging
import json
import os
from typing import Any, Dict, List, Optional
import tiktoken

def truncate_to_tokens(
    text: str,
    max_tokens: int = 120000,
    tokenizer_path: Optional[str] = None,
) -> str:
    """
    Truncate to max_tokens using specified tokenizer.
    - If tokenizer_path is provided, use HuggingFace tokenizer (adapted for Qwen).
    - Otherwise fall back to tiktoken(cl100k_base).
    Note: Must use the same tokenizer as the target model for truncation
    """
    if not text:
        return text

    if tokenizer_path:
       
        if not hasattr(truncate_to_tokens, "_hf_tokenizer_cache"):
            truncate_to_tokens._hf_tokenizer_cache = {}

        tok = truncate_to_tokens._hf_tokenizer_cache.get(tokenizer_path)
        if tok is None:
            from transformers import AutoTokenizer  
            tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            truncate_to_tokens._hf_tokenizer_cache[tokenizer_path] = tok

        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        ids = ids[:max_tokens]
        return tok.decode(ids, skip_special_tokens=True)

    # fallback：tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def create_specialist_prompt(raw_content: str, tool_name: str, question: Optional[str] = None, purpose: Optional[str] = None, tokenizer_path: Optional[str] = None) -> List[Dict]:
    """
    Create context-aware, goal-oriented prompts for the "webpage analysis expert" model.
    """
    # Use provided path, then environment variable. If neither, final_tokenizer_path is None (triggers tiktoken fallback).
    final_tokenizer_path = tokenizer_path or os.getenv("HF_TOKENIZER_PATH")
    raw_content = truncate_to_tokens(raw_content, max_tokens=58000, tokenizer_path=final_tokenizer_path) 

    user_goal_section = ""
    if purpose:
        user_goal_section = f"The immediate purpose for visiting this page is: {purpose}"
    else:
        user_goal_section = "No specific user goal provided."

    
    user_prompt = f"""
Please process the following webpage or local file content and user goal to extract relevant information:

## **Webpage/Local file Content** {raw_content}

## **User Goal**
{user_goal_section}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format: You MUST use Markdown with the following headings:**
## Rational
(Your analysis of relevance here)

## Evidence
(Your extracted evidence here)

## Summary
(Your final summary here)
"""

    
    return [
        {"role": "user", "content": user_prompt}
    ]

def process_with_llm(llm_client: Any, raw_content: str, tool_name: str, question: str, purpose: Optional[str] = None, tokenizer_path: Optional[str] = None) -> str:
    """
    Process content using LLM and return pure Markdown report.
    """
    if not hasattr(llm_client, 'create_completion'):
        raise ValueError("The passed llm_client object does not have 'create_completion' method.")

    messages = create_specialist_prompt(raw_content, tool_name, question, purpose, tokenizer_path=tokenizer_path)
    
    try:
        logging.info(f"Calling expert model (Markdown mode) to process return content of '{tool_name}'...")
        response = llm_client.create_completion(messages=messages, tools=[], temperature=0.7)


        full_response_text = response.get("response", "Error: Unable to get summary from expert model.")
        
    
        if full_response_text.strip().startswith("```markdown"):
            full_response_text = full_response_text.strip()[len("```markdown"):]
        if full_response_text.strip().endswith("```"):
            full_response_text = full_response_text.strip()[:-len("```")]

        return full_response_text.strip() 
            
    except Exception as e:
        logging.error(f"Error: Expert model call failed: {e}", exc_info=True)
        return f"## Error\nLLM call failed during content processing: {e}"
       


def process_browser_tool_return(llm_client: Any, tool_name: str, tool_result: Dict[str, Any], question: str, purpose: Optional[str] = None, tokenizer_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point of the processor, receives question parameter and passes it layer by layer.
    """
    content = tool_result.get("content", {})
    
    tools_to_process_with_llm = ["browser_get_page_info", "browser_get_markdown", "fetch_url", "visit"]

    if tool_name in tools_to_process_with_llm:
        logging.info(f"Detected tool '{tool_name}' that needs processing, starting processor...")

        raw_content = ""
        try:
            if isinstance(content, dict):
                if "html" in content and isinstance(content["html"], str): raw_content = content["html"]
                elif "content" in content and isinstance(content["content"], str):
                    try:
                        nested_data = json.loads(content["content"])
                        if "data" in nested_data and "content" in nested_data["data"]: raw_content = nested_data["data"]["content"]
                    except (json.JSONDecodeError, TypeError): pass
                if not raw_content:
                    longest_str_value = ""
                    for value in content.values():
                        if isinstance(value, str) and len(value) > len(longest_str_value): longest_str_value = value
                    if longest_str_value: raw_content = longest_str_value
            elif isinstance(content, str): raw_content = content
        except Exception: pass
        
        if raw_content:

            processed_content = process_with_llm(llm_client, raw_content, tool_name, question, purpose, tokenizer_path=tokenizer_path)
            logging.info(f"Content of tool '{tool_name}' has been successfully processed by expert model.")
            return {"status": "success", "content": processed_content}
        else:
            logging.warning(f"No processable text found in return value of tool '{tool_name}', skipping processing.")
            
    return tool_result                                                                    
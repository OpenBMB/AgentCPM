import re
import json5
import json
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def keep_after_think(response: str) -> str:
    """Keep content after </think>"""
    idx = response.find("</think>")
    if idx != -1:
        return response[idx + len("</think>") :]
    return response

def keep_after_last_think(response: str) -> str:
    """Keep content after the last </think>"""
    idx = response.rfind("</think>")
    if idx != -1:
        return response[idx + len("</think>") :]
    return response

def _extract_code_block(text: str) -> Optional[str]:
    """Extract content from <code> tags in text"""
    match = re.search(r'<code>(.*?)</code>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def _clean_and_parse_json(json_candidate: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Attempt to clean and parse JSON string.
    Returns: (parsed_json_dict, error_message)
    """
    # 1. Try direct parsing
    try:
        tool_call = json5.loads(json_candidate)
        return tool_call, None
    except Exception:
        pass

    # 2. Try to extract valid JSON by counting braces
    json_start_idx = json_candidate.find('{')
    if json_start_idx == -1:
        return None, "No JSON start brace '{' found"
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(json_start_idx, len(json_candidate)):
        char = json_candidate[i]
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete JSON object
                    try:
                        potential_json = json_candidate[json_start_idx:i+1]
                        return json5.loads(potential_json), None
                    except Exception as e:
                        return None, f"Brace matching found candidate but parse failed: {str(e)}"
    
    return None, "Failed to find matching closing brace"

def parse_tool_for_qwen(content: str) -> List[Dict[str, Any]]:
    response = keep_after_last_think(content)
    
    # Match <tool_call> blocks
    tool_call_regex = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    matches = tool_call_regex.findall(response)
    
    raw_function_calls = []
    
    for match_content in matches:
        try:
            # 1. Detect code blocks
            code_match = re.search(r'<code>(.*?)</code>', match_content, re.DOTALL)
            extracted_code = None
            json_candidate = match_content

            if code_match:
                extracted_code = code_match.group(1).strip()
                # Key: Extract the part before <code> as JSON candidate to avoid parse error
                json_candidate = match_content[:code_match.start()].strip()
            
            # 2. Parse JSON
            parsed_json, error = _clean_and_parse_json(json_candidate)
            
            if not parsed_json:
                # Parse failed, create a tool_call marked with format error
                error_message = f"Failed to parse JSON part, JSON parse error: {error}"
                logger.debug(f"Failed to parse JSON part: {match_content[:500] if len(match_content) > 500 else match_content}, JSON parse error: {error}")
                
                # Create a special tool_call to mark format error
                # Empty function.name will be detected by sampler.py and set to FORMATERROR
                format_error_tool_call = {
                    "id": f"call__{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": "",  # Empty name will be detected by sampler.py as format error
                        "arguments": json.dumps({
                            "_format_error": True,
                            "error_message": error_message
                        }, ensure_ascii=False)
                    }
                }
                raw_function_calls.append(format_error_tool_call)
                continue

            # 3. Handle code injection (for execute_code / PythonInterpreter)
            tool_name = parsed_json.get("name", "")
            if tool_name in ("execute_code", "PythonInterpreter"):
                # Ensure arguments is a dict
                if "arguments" not in parsed_json or not isinstance(parsed_json["arguments"], dict):
                    parsed_json["arguments"] = {}
                
                args = parsed_json["arguments"]
                existing_code = args.get("code")

                # Priority logic:
                # 1. If arguments has code and is non-empty, use it
                # 2. Otherwise, use code extracted from <code> tag
                # 3. Otherwise, try to extract again from original match (fallback)
                if not existing_code or (isinstance(existing_code, str) and not existing_code.strip()):
                    if extracted_code:
                        args["code"] = extracted_code
                    else:
                        # Fallback: Try to extract from original text again in case splitting logic missed it
                        fallback_code = _extract_code_block(match_content)
                        if fallback_code:
                            args["code"] = fallback_code

            # 4. Construct OpenAI format Tool Call object
            # Note: Use standard json.dumps to serialize arguments string to ensure downstream compatibility
            if "name" in parsed_json and "arguments" in parsed_json:
                arguments_str = json.dumps(parsed_json["arguments"], ensure_ascii=False)
                tool_call = {
                    "id": f"call__{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": parsed_json["name"], 
                        "arguments": arguments_str
                    }
                }
                raw_function_calls.append(tool_call)
            
            elif "content" in parsed_json and isinstance(parsed_json["content"], dict):
                # Handle some variant formats
                inner_content = parsed_json["content"]
                if "name" in inner_content and "arguments" in inner_content:
                    arguments_str = json.dumps(inner_content["arguments"], ensure_ascii=False)
                    tool_call = {
                        "id": f"call__{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": inner_content["name"], 
                            "arguments": arguments_str
                        }
                    }
                    raw_function_calls.append(tool_call)
                else:
                    raw_function_calls.append(parsed_json)
            else:
                raw_function_calls.append(parsed_json)

        except Exception as e:
            logger.error(f"Unexpected error parsing tool call: {e}", exc_info=True)
            raw_function_calls.append({"content": match_content})

    return raw_function_calls
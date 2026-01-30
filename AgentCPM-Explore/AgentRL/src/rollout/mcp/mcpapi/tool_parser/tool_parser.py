from .parse_minicpm import parse_tool_for_minicpm3
from .parse_qwen import parse_tool_for_qwen
from .parse_openai import parse_tool_for_openai
from typing import List, Dict, Any
import json5 as json
import re

# class ToolParser :
#     def __init__(self,model_name:str,is_openai_format:bool = False):
#         self.model_name = model_name
#         self.is_openai_format = is_openai_format

#     def parse_tool_from_response(self,completion):
#         if self.is_openai_format:
#             tool_calls = parse_tool_for_openai(completion=completion)
#         else:
#             if "qwen" in self.model_name:
#                 tool_calls = parse_tool_for_qwen(completion=completion)
#             elif "minicpm" in self.model_name:
#                 tool_calls = parse_tool_for_minicpm3(sequence=completion["response"]) 
#             else:
#                 tool_calls = []
#         # Defensive: ensure return is a list
#         if tool_calls is None:
#             tool_calls = []
#         return tool_calls
            
def parse_tool(completion: dict) -> list:
    tool_calls = []
    openai_tool_calls = parse_tool_for_openai(completion=completion) or []
    tool_calls.extend(openai_tool_calls)
    if len(tool_calls) == 0:
        content = completion.content if hasattr(completion, "content") else completion.get("content", "")
        qwen_tool_calls = parse_tool_for_qwen(content=content) or []
        tool_calls.extend(qwen_tool_calls)
    return tool_calls

# #TODO: Simplify

# def parse_tool(completion) -> list:
#     """
#     Minimal version. Does not distinguish sources, both openai tool_calls and qwen <tool_call> are put into the same list.
#     Failed parsing content is kept as string directly.
#     """
#     results = []

#     # OpenAI format
#     if isinstance(completion, dict) and "tool_calls" in completion:
#         tool_calls = parse_tool_for_openai(completion=completion)
#         results.extend(tool_calls)

#     # Qwen format
#     # Can pass str, or dict may have nested "content" field storing Qwen output
#     if isinstance(completion, str):
#         text = completion
#     elif isinstance(completion, dict):
#         # When dict, also try to extract "content" from it
#         text = completion.get("content", "")
#     else:
#         text = ""
#     if text:
#         tool_call_regex = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
#         matches = tool_call_regex.findall(text)
#         for m in matches:
#             item = m.strip()
#             try:
#                 item = json.loads(item)
#             except Exception:
#                 pass  # On parse failure, use original string
#             results.append(item)

#     return results
    




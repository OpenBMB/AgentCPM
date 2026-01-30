def to_dict(obj):
    # pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # pydantic v1
    if hasattr(obj, "dict"):
        return obj.dict()
    # Regular object
    if hasattr(obj, "__dict__"):
        return vars(obj)
    # Already a dict
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot convert {type(obj)} to dict")

def parse_tool_for_openai(completion):
    if isinstance(completion, dict):
        tool_calls = completion.get("tool_calls", [])
    else:
        tool_calls = getattr(completion, "tool_calls", [])
    # Defensive: ensure tool_calls is a list
    if tool_calls is None:
        tool_calls = []
    # Convert all to dict
    return [to_dict(tc) for tc in tool_calls] 
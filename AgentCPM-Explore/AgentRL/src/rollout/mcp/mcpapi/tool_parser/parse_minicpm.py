import re
import json
import logging
import keyword
import ast
import uuid
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("minicpm parser")

# from ShishirPatil/gorilla
def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}

def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        output = "..." if value.value is Ellipsis else value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value  # type: ignore
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
            value,
            ast.NameConstant):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
            value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = ast.literal_eval(ast.unparse(value))  # type: ignore
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)  # type: ignore
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = ast.literal_eval(
            ast.unparse(  # type: ignore
                value.body[0].value))  # type: ignore
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)  # type: ignore
        except Exception as e:
            logger.error("Error parsing tool call: %s", str(e))
            output = (
                ast.unparse(value.value) + "[" +  # type: ignore
                ast.unparse(value.slice) + "]")  # type: ignore
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output

def parse_tool_for_minicpm3(
    sequence: str,
    tool_call_start="<|tool_call_start|>",
    tool_call_end="<|tool_call_end|>",
):
    try:
        if tool_call_start in sequence and tool_call_end in sequence:
            tool_call_string, content = sequence.rsplit(tool_call_end, 1)
            tool_call_string = tool_call_string.split(tool_call_start, 1)[1]
            tool_calls = []
            tool_call_string = tool_call_string.strip()
            if tool_call_string.startswith("```"):
                tool_call_string = tool_call_string[3:].strip()
                if tool_call_string.startswith("python"):
                    tool_call_string = tool_call_string.lstrip(
                        "python").strip()
            if tool_call_string.endswith("```"):
                tool_call_string = tool_call_string[:-3].strip()
            for kw in keyword.kwlist:
                tool_call_string = tool_call_string.replace(
                    "," + kw + "=", "," + kw + "_=")
                tool_call_string = tool_call_string.replace(
                    " " + kw + "=", " " + kw + "_=")
                tool_call_string = tool_call_string.replace(
                    "(" + kw + "=", "(" + kw + "_=")
            need_replace = False
            replaced_tool_call_string = tool_call_string.replace("-","_")
            if replaced_tool_call_string != tool_call_string:
                need_replace = True
                tool_call_string = replaced_tool_call_string
            parsed: ast.Module = ast.parse(tool_call_string)

            for elem in parsed.body:
                assert isinstance(elem.value, ast.Call)  # type: ignore
                calls = resolve_ast_call(elem.value)  # type: ignore

                for func_name, func_args in calls.items():
                    new_args = {}
                    for k, v in func_args.items():
                        for kw in keyword.kwlist:
                            if k == kw + "_":
                                k = kw
                        new_args[k] = v

                    this_one = {"name": func_name, "arguments": new_args}
                    tool_calls.append({ "id":str(uuid.uuid4()),"function":this_one,"type":"function"})
            if need_replace:
                for tool_call in tool_calls:
                    tool_call["function"]["name"] = tool_call["function"]["name"].replace("_","-")
            return tool_calls
        else:
            return []
    except Exception:
        return []

if __name__ == "__main__":
    yjr = " <|thought_start|>\nI will now search for \"Amazon product scraper\" with a limit of 20 to find the most popular scrapers for the user.\n<|thought_end|>\n<|tool_call_start|>\n```python\nsearch-actors(limit=20,search=\"Amazon product scraper\")\n```\n<|tool_call_end|>\n"
    parse_tool_for_minicpm3(yjr) 
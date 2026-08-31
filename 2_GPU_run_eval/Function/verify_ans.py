# 从模型响应里抽出函数调用并与 ground truth 对比
#
# GPU 主路径：OpenAI message.tool_calls（function.name + function.arguments JSON 字符串）
# 文本兜底（OMC / 部分 serving 把 <tool_call> 写进 content）：
#   Qwen <tool_call>...</tool_call>；再退到 JSON 数组 / 单对象
# 不再把 JSON Schema（name + parameters.properties）误当成一次调用
#
# verify_answer(eval_type, expect, response, tool_calls=None):
#   eval_type:
#     - "ast"         : expect 为 ground_truth JSON 字符串，做 AST 匹配
#     - "relevance"   : 至少调用一次 -> True
#     - "irrelevance" : 没解析到任何调用 -> True
#   tool_calls 非 None 时优先用结构化 tool_calls，不再从文本抽
#   返回 (correct: bool, predicted_calls_json: str)

import json
import re

import ast_checker


# testCaseName 解析正则；eval_results 用它分组，verify 用它推 eval_type。
# BFCL 用例形如 project-1-bfcl-test-simple-0 / project-1-bfcl-test-live_irrelevance-12。
TEST_CASE_NAME_PATTERN = re.compile(
    r"^(?P<project_name>[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*-[0-9]+)-"
    r"(?P<flavor>[a-zA-Z0-9_]+)-test-"
    r"(?P<vertical>[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*)-[0-9]+$"
)


# 13 个 BFCL 类别 -> 评分路径。与 1_Data_gen/Function/bfcl.py 的 DEFAULT_CATEGORIES 必须保持一致。
CATEGORY_TO_EVAL_TYPE = {
    "simple": "ast",
    "multiple": "ast",
    "parallel": "ast",
    "parallel_multiple": "ast",
    "java": "ast",
    "javascript": "ast",
    "live_simple": "ast",
    "live_multiple": "ast",
    "live_parallel": "ast",
    "live_parallel_multiple": "ast",
    "irrelevance": "irrelevance",
    "live_irrelevance": "irrelevance",
    "live_relevance": "relevance",
}


def get_eval_type(category):
    """category 不在表里就回退到 'ast'（最严格的路径，宁可误判否定也别给假阳）。"""
    return CATEGORY_TO_EVAL_TYPE.get(category, "ast")


_CODE_FENCE_PATTERN = re.compile(
    r"```(?:json|tool|tool_code|python)?\s*(.+?)```",
    re.DOTALL | re.IGNORECASE,
)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED_PATTERN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_NO_CALL_PATTERN = re.compile(r"\bNO[_\s-]?CALL\b", re.IGNORECASE)
_TOOL_CALL_XML_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_thinking(text):
    text = _THINK_PATTERN.sub("", text or "")
    text = _THINK_UNCLOSED_PATTERN.sub("", text)
    return text


def _iter_balanced_blocks(text, open_ch, close_ch):
    depth = 0
    start = -1
    in_str = False
    str_ch = ""
    esc = False
    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == str_ch:
                in_str = False
            continue
        if c in ('"', "'"):
            in_str = True
            str_ch = c
            continue
        if c == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif c == close_ch:
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    yield text[start : i + 1]
                    start = -1


def _try_parse_json(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _looks_like_json_schema(obj):
    """{name, parameters: {type, properties}} 是工具 schema，不是一次调用。"""
    if not isinstance(obj, dict):
        return False
    params = obj.get("parameters")
    if not isinstance(params, dict):
        return False
    if "properties" in params:
        return True
    return params.get("type") in ("object", "dict")


def _parse_arguments(args):
    if args is None:
        return {}
    if isinstance(args, str):
        parsed = _try_parse_json(args)
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(args, dict):
        return args
    return {}


def _item_to_call(item):
    """把一个 dict 收成 {name, arguments}；schema / 非调用返回 None。"""
    if not isinstance(item, dict):
        return None
    if "function" in item and isinstance(item["function"], dict):
        fn = item["function"]
        name = fn.get("name")
        if not name:
            return None
        return {"name": name, "arguments": _parse_arguments(fn.get("arguments"))}
    if _looks_like_json_schema(item):
        return None
    if "name" in item:
        args = item.get("arguments")
        if args is None:
            args = item.get("args")
        return {"name": item["name"], "arguments": _parse_arguments(args)}
    if len(item) == 1:
        (k, v), = item.items()
        if isinstance(v, dict) and not _looks_like_json_schema({"name": k, "parameters": v}):
            return {"name": k, "arguments": v}
    return None


def _coerce_to_call_list(parsed):
    if parsed is None:
        return None
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return None

    calls = []
    for item in parsed:
        call = _item_to_call(item)
        if call is None:
            return None
        calls.append(call)
    return calls


def normalize_openai_tool_calls(tool_calls):
    """OpenAI tool_calls（dict 或 SDK 对象）-> [{name, arguments}]。"""
    if not tool_calls:
        return []
    calls = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            name = fn.get("name")
            args = fn.get("arguments")
        else:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None) if fn is not None else None
            args = getattr(fn, "arguments", None) if fn is not None else None
        if not name:
            continue
        calls.append({"name": str(name), "arguments": _parse_arguments(args)})
    return calls


def extract_calls(response_text):
    """从文本抽调用。优先 Qwen <tool_call> 块（可多段并行），再 JSON。
    返回 (calls, raw_tag)。
    calls=[] 表示明确无调用；calls=None 表示没解析到调用结构。"""
    if not response_text:
        return None, ""

    text = _strip_thinking(response_text)

    xml_blocks = _TOOL_CALL_XML_PATTERN.findall(text)
    if xml_blocks:
        calls = []
        for block in xml_blocks:
            parsed = _try_parse_json(block.strip())
            coerced = _coerce_to_call_list(parsed)
            if coerced:
                calls.extend(coerced)
        if calls:
            return calls, "tool_call_xml"

    candidates = []
    for m in _CODE_FENCE_PATTERN.findall(text):
        candidates.append(m.strip())
    for block in _iter_balanced_blocks(text, "[", "]"):
        candidates.append(block.strip())
    for block in _iter_balanced_blocks(text, "{", "}"):
        candidates.append(block.strip())

    for cand in candidates:
        parsed = _try_parse_json(cand)
        calls = _coerce_to_call_list(parsed)
        if calls is not None:
            return calls, cand

    if _NO_CALL_PATTERN.search(text):
        return [], ""
    return None, ""


def verify_answer(eval_type, expect, response_text, tool_calls=None):
    """返回 (correct: bool, prediction_str: str)。

    tool_calls is not None：用结构化 OpenAI tool_calls（GPU 主路径，空列表=明确没调）。
    否则从 response_text 抽（OMC / API 文本兜底）。
    """
    if tool_calls is not None:
        calls = normalize_openai_tool_calls(tool_calls)
    else:
        calls, _ = extract_calls(response_text)
    prediction = json.dumps(calls if calls is not None else [], ensure_ascii=False)

    if eval_type == "irrelevance":
        is_call = ast_checker.is_call_list(calls)
        return (not is_call), prediction

    if eval_type == "relevance":
        is_call = ast_checker.is_call_list(calls)
        return is_call, prediction

    if calls is None or not expect:
        return False, prediction
    gt = _try_parse_json(expect)
    if not isinstance(gt, list):
        return False, prediction
    return ast_checker.match_ast(gt, calls), prediction

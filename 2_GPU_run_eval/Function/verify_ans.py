# 从模型响应里抽出函数调用 JSON 并与 ground truth 对比
#
# 兼容：```json``` 代码块 / 单对象 / OpenAI tool_calls 嵌套 / <think> 标签
# verify_answer(eval_type, expect, response):
#   eval_type:
#     - "ast"         : expect 为 ground_truth JSON 字符串，做 AST 匹配
#     - "relevance"   : 至少调用一次 -> True
#     - "irrelevance" : 没解析到任何调用 -> True
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


# 13 个 BFCL 类别 -> 评分路径。OMC/GPU/API 三种 mode 都不在 test case 里塞 eval_type，
# evaluator 从 testCaseName 解析 category 后用这张表查路径，三种 mode 完全对等。
# 与 1_Data_gen/Function/bfcl.py 的 DEFAULT_CATEGORIES 必须保持一致。
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
_NO_CALL_PATTERN = re.compile(r"\bNO[_\s-]?CALL\b", re.IGNORECASE)


def _strip_thinking(text):
    return _THINK_PATTERN.sub("", text or "")


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


def _coerce_to_call_list(parsed):
    if parsed is None:
        return None
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return None

    calls = []
    for item in parsed:
        if not isinstance(item, dict):
            return None
        # OpenAI tool_call 风格
        if "function" in item and isinstance(item["function"], dict):
            fn = item["function"]
            name = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                args = _try_parse_json(args) or {}
            calls.append({"name": name, "arguments": args or {}})
            continue
        if "name" in item:
            args = item.get("arguments")
            if args is None:
                args = item.get("parameters") or item.get("args") or {}
            if isinstance(args, str):
                args = _try_parse_json(args) or {}
            calls.append({"name": item["name"], "arguments": args or {}})
            continue
        # gorilla 旧格式: {"func_name": {arg1: v1, ...}}
        if len(item) == 1:
            (k, v), = item.items()
            if isinstance(v, dict):
                calls.append({"name": k, "arguments": v})
                continue
        return None
    return calls


def extract_calls(response_text):
    """返回 (calls, raw_block)。
    calls=[] 表示明确 NO_CALL 或没解析到调用；calls=None 表示连 JSON 块都没找到。"""
    if not response_text:
        return None, ""

    text = _strip_thinking(response_text)

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


def verify_answer(eval_type, expect, response_text):
    """返回 (correct: bool, prediction_str: str)。"""
    calls, _ = extract_calls(response_text)
    prediction = json.dumps(calls if calls is not None else [], ensure_ascii=False)

    if eval_type == "irrelevance":
        is_call = ast_checker.is_call_list(calls)
        return (not is_call), prediction

    if eval_type == "relevance":
        is_call = ast_checker.is_call_list(calls)
        return is_call, prediction

    # eval_type == "ast"（默认）
    if calls is None or not expect:
        return False, prediction
    gt = _try_parse_json(expect)
    if not isinstance(gt, list):
        return False, prediction
    return ast_checker.match_ast(gt, calls), prediction

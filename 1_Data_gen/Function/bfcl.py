# 从 BFCL v3 数据集生成测试用例
#
# 数据集目录约定（路径由调用方传入 dataset_root）：
#   {dataset_root}/BFCL_v3_{cat}.json + possible_answer/BFCL_v3_{cat}.json
#
# 协议对齐真实 agent / OpenAI chat.completions：
#   messages    = 固定 system prompt + 数据集原始 question[0]（保留 role）
#   tools       = BFCL function[] 转成 OpenAI tools（schema 不塞进 user 文本）
#   tool_choice = "auto"
#   expect      = 只给评分器用，不进模型请求；写在 JSON 字段末尾
#
# 三种 mode：
#   - "gpu" : sentences[0].messages + tools + tool_choice，evaluator 直接送 vLLM
#   - "api" : OpenAI chat.completions 请求体（messages / tools / temperature / max_tokens / ...）
#   - "omc" : apply_chat_template(messages, tools=tools) 渲成单串 prompt
#             （vLLM 接到 tools 后内部也是这么渲染的）

import json
import os

import test_utils

# 13 个单轮类别 -> eval_type
# 评分逻辑见 verify_ans / ast_checker（exec / rest / sql / multi_turn / chatable 第一版不接）
DEFAULT_CATEGORIES = {
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


# 仍走 native tools，所以不再要求模型在 content 里吐 JSON 数组 / NO_CALL。
SYSTEM_PROMPT = (
    "You are a function-calling agent. Read the user request and the list of "
    "available functions, then decide which (if any) function(s) to call.\n\n"
    "Rules:\n"
    "1. If one or more functions should be called, invoke them through the "
    "provided tools. If parallel calls are needed, issue all of them together.\n"
    "2. If NONE of the available functions match the user request, do not call "
    "any tool.\n"
    "3. Use the exact function names and parameter names as declared. Do NOT "
    "invent parameters that are not in the schema.\n"
    "4. Respect the parameter types: integers as numbers, booleans as true/false, "
    "lists as JSON arrays, strings as JSON strings.\n"
)


_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

# BFCL schema 类型 -> JSON Schema / OpenAI tools 类型
_TYPE_MAP = {
    "dict": "object",
    "object": "object",
    "hashmap": "object",
    "float": "number",
    "double": "number",
    "number": "number",
    "integer": "integer",
    "int": "integer",
    "long": "integer",
    "string": "string",
    "boolean": "boolean",
    "array": "array",
    "arraylist": "array",
    "tuple": "array",
    "any": None,
}


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _conversation_messages(question):
    """官方 question 是 List[List[message]]；本仓库 13 类都是单 turn，取 question[0]。"""
    if isinstance(question, list) and question and isinstance(question[0], list):
        msgs = question[0]
    elif isinstance(question, list):
        msgs = question
    else:
        return _with_system_prompt(
            [{"role": "user", "content": "" if question is None else str(question)}]
        )

    out = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        if role not in _ALLOWED_ROLES:
            role = "user"
        content = m.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        out.append({"role": role, "content": content})
    if not out:
        out = [{"role": "user", "content": ""}]
    return _with_system_prompt(out)


def _with_system_prompt(messages):
    """保证 messages[0] 是我们的 agent system prompt。

    数据集少数 live 条本身带 system（澄清规则等）：接到我们的 prompt 后面，
    避免出现两条 system（Qwen tools 模板只吃 messages[0]）。
    """
    messages = [dict(m) for m in messages]
    if messages and messages[0].get("role") == "system":
        existing = messages[0].get("content") or ""
        if existing.strip():
            messages[0]["content"] = SYSTEM_PROMPT.rstrip() + "\n\n" + existing
        else:
            messages[0]["content"] = SYSTEM_PROMPT
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages


def _map_json_type(raw):
    if raw is None:
        return None
    return _TYPE_MAP.get(str(raw).strip().lower(), str(raw).strip().lower())


def _convert_schema_node(node):
    """把 BFCL 参数节点（可能含 type/properties/items/description）收成 JSON Schema。"""
    if not isinstance(node, dict):
        return node

    out = {}
    mapped = _map_json_type(node.get("type"))
    if mapped:
        out["type"] = mapped

    for key in ("description", "default", "enum", "format", "minimum", "maximum"):
        if key in node:
            out[key] = node[key]

    if "properties" in node and isinstance(node["properties"], dict):
        out["properties"] = {
            k: _convert_schema_node(v) if isinstance(v, dict) else v
            for k, v in node["properties"].items()
        }
    if "required" in node:
        out["required"] = node["required"]
    if "items" in node:
        items = node["items"]
        out["items"] = _convert_schema_node(items) if isinstance(items, dict) else items
    return out


def functions_to_openai_tools(functions):
    """BFCL function[] -> OpenAI tools[]。空列表表示这条没有可调用工具。"""
    tools = []
    if not functions:
        return tools
    for fn in functions:
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters")
        if isinstance(params, dict):
            parameters = _convert_schema_node(params)
            if "type" not in parameters:
                parameters["type"] = "object"
        else:
            parameters = {"type": "object", "properties": {}}
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": fn.get("description") or "",
                "parameters": parameters,
            },
        }
        tools.append(tool)
    return tools


def _sanitize_category_for_name(cat):
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in cat)


def _build_one_case(mode, project_name, cat, idx, messages, tools, expect, max_output_token_length):
    name = f"{project_name}-bfcl-test-{_sanitize_category_for_name(cat)}-{idx}"

    if mode == "omc":
        prompt = test_utils.render_chat_template_messages(messages, tools=tools or None)
        return test_utils.create_omc_test(name, prompt, expect, max_output_token_length)

    if mode == "gpu":
        return test_utils.create_gpu_test(
            name, messages, expect, max_output_token_length, tools=tools
        )
    if mode == "api":
        return test_utils.create_api_test(
            name, messages, expect, max_output_token_length, tools=tools
        )
    raise ValueError(f"unknown mode: {mode}")


def generate_bfcl_tests(
    project_name,
    dataset_root,
    max_cases_per_category,
    mode,
    max_output_token_length=5000,
    categories=None,
):
    """mode in {'omc', 'gpu', 'api'}。"""
    if mode not in ("omc", "gpu", "api"):
        raise ValueError(f"mode must be one of omc/gpu/api, got {mode}")
    if categories is None:
        categories = DEFAULT_CATEGORIES

    possible_dir = os.path.join(dataset_root, "possible_answer")

    test_cases = []
    for cat, eval_type in categories.items():
        data_path = os.path.join(dataset_root, f"BFCL_v3_{cat}.json")
        if not os.path.isfile(data_path):
            print(f"[WARN] missing data file: {data_path}, skip")
            continue

        records = _read_jsonl(data_path)

        gt_map = {}
        gt_path = os.path.join(possible_dir, f"BFCL_v3_{cat}.json")
        if eval_type == "ast":
            if not os.path.isfile(gt_path):
                print(f"[WARN] AST category {cat} has no possible_answer file, skip")
                continue
            for r in _read_jsonl(gt_path):
                gt_map[r["id"]] = r.get("ground_truth")

        if max_cases_per_category is not None:
            records = records[:max_cases_per_category]

        for i, r in enumerate(records):
            messages = _conversation_messages(r.get("question", []))
            tools = functions_to_openai_tools(r.get("function") or [])
            gt = gt_map.get(r.get("id"))
            try:
                expect = json.dumps(gt, ensure_ascii=False) if gt is not None else ""
            except (TypeError, ValueError):
                expect = str(gt) if gt is not None else ""

            test_cases.append(
                _build_one_case(
                    mode,
                    project_name,
                    cat,
                    i,
                    messages,
                    tools,
                    expect,
                    max_output_token_length,
                )
            )

    return test_cases

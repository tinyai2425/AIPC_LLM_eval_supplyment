# 从 BFCL v3 数据集生成测试用例
#
# 数据集目录约定（路径由调用方传入 dataset_root，默认 AIPC_LLM_eval_supplyment/Data_set/Berkeley-Function-Calling-Leaderboard/）：
#   {dataset_root}/
#       BFCL_v3_simple.json
#       ...
#       possible_answer/
#           BFCL_v3_simple.json
#           ...
#
# 数据集本身没有提交到 git，需自行下载：
#   HuggingFace: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
#   Gorilla 项目: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
# 详细下载步骤见 README 第 "二.3 数据集" 节。
#
# 每个数据文件是一行一个 JSON 对象（jsonl 风格）：
#   {id, question:[[{role,content}, ...]], function:[{name, description, parameters}, ...]}
# possible_answer：{id, ground_truth:[{func_name:{param:[allowed_values]}}, ...]}
#
# 关联好之后我们做 prompt-based function calling：
#   system_prompt：要求模型只输出 JSON 数组形式的函数调用
#   user_prompt  ：把函数 schema 序列化进去 + 拼回原始 user/system 对话
#
# 三种 mode 出三种产物结构（详见 test_utils.py）：
#   - "omc" : 严格 Ceval，system+user 已用 chat_template 渲染成单串 prompt
#   - "gpu" : Ceval 外壳 + sentences[0].messages（直接送 vLLM）
#   - "api" : OpenAI 风格顶层 messages

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


SYSTEM_PROMPT = (
    "You are a function-calling agent. Read the user request and the list of "
    "available functions, then decide which (if any) function(s) to call.\n\n"
    "Output rules (STRICT):\n"
    "1. If one or more functions should be called, respond with ONLY a JSON "
    "array of function calls, no prose, no markdown fences. Each element must "
    "be of the form: {\"name\": \"<function_name>\", \"arguments\": {<param>: <value>, ...}}\n"
    "2. If parallel calls are needed, include all of them in the same JSON array.\n"
    "3. If NONE of the available functions match the user request, respond with "
    "exactly: NO_CALL\n"
    "4. Use the exact function names and parameter names as declared. Do NOT "
    "invent parameters that are not in the schema.\n"
    "5. Respect the parameter types: integers as numbers, booleans as true/false, "
    "lists as JSON arrays, strings as JSON strings.\n"
)


USER_PROMPT_TEMPLATE = (
    "Available functions (JSON):\n{functions_json}\n\n"
    "Conversation so far:\n{conversation}\n\n"
    "Respond now following the output rules."
)


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _flatten_conversation(question):
    """官方 question 字段是 List[List[message]]，第一版只看第一个 turn 的 user/system。"""
    if isinstance(question, list) and question and isinstance(question[0], list):
        msgs = question[0]
    elif isinstance(question, list):
        msgs = question
    else:
        return str(question)

    parts = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "system":
            parts.append(f"[system context] {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def _serialize_functions(functions):
    if not functions:
        return "[]"
    try:
        return json.dumps(list(functions), ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(functions)


def build_user_prompt(question, functions):
    return USER_PROMPT_TEMPLATE.format(
        functions_json=_serialize_functions(functions),
        conversation=_flatten_conversation(question),
    )


def _sanitize_category_for_name(cat):
    # testCaseName 的 vertical 段只允许 [a-zA-Z0-9_-]；BFCL 类别都合法，这里保险走一遍。
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in cat)


def _build_one_case(mode, project_name, cat, idx, user_prompt, expect, max_output_token_length):
    name = f"{project_name}-bfcl-test-{_sanitize_category_for_name(cat)}-{idx}"

    if mode == "omc":
        prompt = test_utils.render_chat_template(SYSTEM_PROMPT, user_prompt)
        return test_utils.create_omc_test(name, prompt, expect, max_output_token_length)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if mode == "gpu":
        return test_utils.create_gpu_test(name, messages, expect, max_output_token_length)
    if mode == "api":
        return test_utils.create_api_test(name, messages, expect, max_output_token_length)
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
            user_prompt = build_user_prompt(r.get("question", []), r.get("function", []))
            gt = gt_map.get(r.get("id"))
            try:
                expect = json.dumps(gt, ensure_ascii=False) if gt is not None else ""
            except (TypeError, ValueError):
                expect = str(gt) if gt is not None else ""

            test_cases.append(
                _build_one_case(
                    mode, project_name, cat, i, user_prompt, expect, max_output_token_length
                )
            )

    return test_cases

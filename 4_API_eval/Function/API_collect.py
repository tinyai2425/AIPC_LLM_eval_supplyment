# 解析 API 端跑出来的日志文件 -> DataFrame
#
# 日志格式：每条用例三行
#   1) 请求 JSON：{"model", "testCaseName", "messages":[{role,content}, ...], "expect", "options":{...}}
#   2) 响应 JSON：{"created_at", "model", "message":{"role","content"},
#                  "prompt_eval_count", "prompt_eval_duration"(ns),
#                  "eval_count", "eval_duration"(ns), "total_duration"(ns), ...}
#   3) API_total_time: <float>   # 客户端 wall-clock，秒
#
# 时长口径：
#   first_token_time = prompt_eval_duration / 1e9   (即 prefill 时间，约等于首 token 延迟)
#   decode_time      = eval_duration / 1e9
#   total_time       = API_total_time              (含网络 RTT)
#
# 评分：与 OMC / GPU 链路一致，从 testCaseName 解析 category 后查 eval_type。

import json
from typing import Dict, List, Union

import pandas as pd
from transformers import AutoTokenizer


TOKENIZER_CONFIG_PATH = None
tokenizer = None

# 由主脚本注入
GET_answer = None
calculate_repetition_rate = None
calculate_token_entropy = None
verify_answer = None
get_eval_type = None
TEST_CASE_NAME_PATTERN = None
iter_lines_safely = None


def init_tokenizer(path: str):
    global TOKENIZER_CONFIG_PATH, tokenizer
    TOKENIZER_CONFIG_PATH = path
    tokenizer = AutoTokenizer.from_pretrained(path)


def _messages_to_str(messages):
    """把 messages 拼成单串，仅用于 tokenize 算 prompt 长度。"""
    parts = []
    for m in messages or []:
        role = m.get("role", "")
        content = m.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n".join(parts)


def parse_llm_api_results(input_path: str) -> pd.DataFrame:
    results: List[Dict[str, Union[str, float, int]]] = []
    lines = [line.strip() for line in iter_lines_safely(input_path) if line.strip()]

    if len(lines) % 3 != 0:
        raise ValueError(
            f"输入文件 {input_path} 行数 ({len(lines)}) 不是 3 的倍数，请检查格式。"
        )

    for i in range(0, len(lines), 3):
        req_line = json.loads(lines[i])
        resp_line = json.loads(lines[i + 1])
        api_time_line = lines[i + 2]

        case: Dict[str, Union[str, float, int]] = {}

        case_name = req_line.get("testCaseName", "")
        case["testCaseName"] = case_name
        matched = TEST_CASE_NAME_PATTERN.match(case_name) if case_name else None
        if matched:
            case["project_name"] = matched.group("project_name")
            case["flavor"] = matched.group("flavor")
            case["vertical"] = matched.group("vertical")
            case["category"] = matched.group("vertical")
        else:
            case["project_name"] = case["flavor"] = case["vertical"] = case["category"] = ""

        messages = req_line.get("messages", [])
        prompt_str = _messages_to_str(messages)
        try:
            question_str = json.dumps(messages, ensure_ascii=False)
        except (TypeError, ValueError):
            question_str = str(messages)
        case["question"] = question_str

        case["expect"] = (req_line.get("expect", "") or "").strip()

        response_text = resp_line.get("message", {}).get("content", "")
        case["response"] = response_text

        # === 采样参数 ===
        options = req_line.get("options", {}) or {}
        case["temperature"] = options.get("temperature")
        case["top_k"] = options.get("top_k")
        case["top_p"] = options.get("top_p")
        case["repetitionPenalty"] = options.get("repeat_penalty")
        case["max gen tokens"] = options.get("num_predict")

        # === Token 长度 ===
        case["prompt_token_len"] = len(tokenizer(prompt_str)["input_ids"])
        case["response_token_len"] = len(tokenizer(response_text)["input_ids"])

        # === 时长 ===
        prefill_ns = float(resp_line.get("prompt_eval_duration", 0) or 0)
        decode_ns = float(resp_line.get("eval_duration", 0) or 0)
        api_total_time = float(api_time_line.replace("API_total_time:", "").strip())

        case["first_token_time"] = prefill_ns / 1e9
        case["decode_time"] = decode_ns / 1e9
        case["total_time"] = api_total_time

        # === 内容质量指标 ===
        case["get_ans"] = GET_answer(response_text)
        case["repeat"] = calculate_repetition_rate(response_text)
        case["entropy"] = calculate_token_entropy(response_text)

        # === 评分 ===
        eval_type = get_eval_type(case["category"]) if case["category"] else "ast"
        case["eval_type"] = eval_type
        correct, prediction = verify_answer(eval_type, case["expect"], response_text)
        case["correct"] = bool(correct)
        case["prediction"] = prediction

        results.append(case)

    print(f"{len(results)} tests processed in results {input_path}")
    return pd.DataFrame(results)

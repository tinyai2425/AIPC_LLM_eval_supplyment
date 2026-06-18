# 解析 API 端跑出来的日志文件 -> DataFrame
#
# 每条用例 3 行：请求 JSON / 响应 JSON / "API_total_time: <s>"
# 响应 JSON 支持两种格式（运行时自动嗅探）：
#
# A) Ollama 风格（旧）：
#    {"created_at": "...", "model": "...", "message": {"role": "...", "content": "..."},
#     "prompt_eval_count": N, "prompt_eval_duration": N_ns,
#     "eval_count": N, "eval_duration": N_ns, "total_duration": N_ns, ...}
#    时长口径：
#      first_token_time = prompt_eval_duration / 1e9   (prefill 时间≈首 token 延迟)
#      decode_time      = eval_duration / 1e9
#      total_time       = API_total_time              (含网络 RTT)
#
# B) OpenAI chat.completion 风格（新）：
#    {"created": ..., "model": "...", "object": "chat.completion",
#     "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."},
#                  "finish_reason": "stop"}],
#     "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}}
#    时长口径：服务端没回 prefill / decode 时长，只能写 None；total_time 仍用 API_total_time。
#
# 评分：与 OMC / GPU 链路一致——从 testCaseName 解析 category 后查 eval_type，再交给 verify_answer。
#       BFCL 注入 verify_ans，HumanEval 注入 verify_ans_humaneval，互不干扰。

import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union

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


def _detect_format(resp: dict) -> str:
    """返回 'chat_completion' / 'ollama' / 'unknown'。
    嗅探规则：
      - 顶层有 choices[].message.content     -> OpenAI chat.completion
      - 顶层有 message.content               -> Ollama
      - 都不像                                -> unknown
    """
    if not isinstance(resp, dict):
        return "unknown"
    if resp.get("object") == "chat.completion":
        return "chat_completion"
    choices = resp.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict) and "content" in msg:
            return "chat_completion"
    msg = resp.get("message")
    if isinstance(msg, dict) and "content" in msg:
        return "ollama"
    return "unknown"


def _extract_response_fields(resp: dict) -> Tuple[str, Optional[float], Optional[float], Optional[int], Optional[int]]:
    """从响应 JSON 抽出统一字段：
        (response_text, first_token_time_s, decode_time_s, prompt_tokens, completion_tokens)
    缺失字段返回 None，由 DataFrame 转成 NaN。
    """
    fmt = _detect_format(resp)

    if fmt == "chat_completion":
        choices = resp.get("choices") or []
        response_text = ""
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            response_text = msg.get("content", "") or ""
        usage = resp.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        # OpenAI chat.completion 不分段返回时长，无法准确切 prefill / decode
        return response_text, None, None, prompt_tokens, completion_tokens

    if fmt == "ollama":
        msg = resp.get("message") or {}
        response_text = msg.get("content", "") or ""
        prefill_ns = resp.get("prompt_eval_duration")
        decode_ns = resp.get("eval_duration")
        prompt_tokens = resp.get("prompt_eval_count")
        completion_tokens = resp.get("eval_count")
        ftt = float(prefill_ns) / 1e9 if prefill_ns is not None else None
        dec = float(decode_ns) / 1e9 if decode_ns is not None else None
        return response_text, ftt, dec, prompt_tokens, completion_tokens

    # unknown：尽力提取 message.content
    msg = resp.get("message") if isinstance(resp, dict) else None
    response_text = (msg or {}).get("content", "") if isinstance(msg, dict) else ""
    return response_text, None, None, None, None


def parse_llm_api_results(input_path: str) -> pd.DataFrame:
    results: List[Dict[str, Union[str, float, int]]] = []
    lines = [line.strip() for line in iter_lines_safely(input_path) if line.strip()]

    if len(lines) % 3 != 0:
        raise ValueError(
            f"输入文件 {input_path} 行数 ({len(lines)}) 不是 3 的倍数，请检查格式。"
        )

    fmt_counts: Dict[str, int] = {}

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

        # === 响应字段（按响应格式自适应）===
        fmt = _detect_format(resp_line)
        fmt_counts[fmt] = fmt_counts.get(fmt, 0) + 1
        response_text, ftt_s, decode_s, prompt_tok, completion_tok = _extract_response_fields(resp_line)
        case["resp_format"] = fmt
        case["response"] = response_text

        # === 采样参数 ===
        options = req_line.get("options", {}) or {}
        case["temperature"] = options.get("temperature")
        case["top_k"] = options.get("top_k")
        case["top_p"] = options.get("top_p")
        case["repetitionPenalty"] = options.get("repeat_penalty")
        case["max gen tokens"] = options.get("num_predict")

        # === Token 长度（优先用响应 usage / count，缺了再本地 tokenize 兜底）===
        if prompt_tok is not None:
            case["prompt_token_len"] = int(prompt_tok)
        else:
            case["prompt_token_len"] = len(tokenizer(prompt_str)["input_ids"])
        if completion_tok is not None:
            case["response_token_len"] = int(completion_tok)
        else:
            case["response_token_len"] = len(tokenizer(response_text)["input_ids"])

        # === 时长 ===
        api_total_time = float(api_time_line.replace("API_total_time:", "").strip())
        case["first_token_time"] = ftt_s if ftt_s is not None else math.nan
        case["decode_time"] = decode_s if decode_s is not None else math.nan
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
    if fmt_counts:
        fmt_summary = ", ".join(f"{k}={v}" for k, v in fmt_counts.items())
        print(f"[INFO] response formats detected: {fmt_summary}")
    return pd.DataFrame(results)

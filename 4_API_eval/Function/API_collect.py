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
#    响应 message 可能是 content 文本，或 content=null + tool_calls（BFCL GPU/agent 协议）。
#    时长口径：服务端没回 prefill / decode 时长，只能写 None；total_time 仍用 API_total_time。
#
#   评分：与 OMC / GPU 链路一致——从 testCaseName 解析 category 后查 eval_type，再交给 verify_answer。
#       BFCL 注入 verify_ans（优先 message.tool_calls），HumanEval 注入 verify_ans_humaneval。

import inspect
import json
import math
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


def _detect_format(resp: dict) -> str:
    """返回 'chat_completion' / 'ollama' / 'unknown'。
    嗅探规则：
      - 顶层 object==chat.completion，或 choices[].message 含 content / tool_calls
      - 顶层有 message.content / message.tool_calls -> Ollama
    """
    if not isinstance(resp, dict):
        return "unknown"
    if resp.get("object") == "chat.completion":
        return "chat_completion"
    choices = resp.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict) and ("content" in msg or "tool_calls" in msg):
            return "chat_completion"
    msg = resp.get("message")
    if isinstance(msg, dict) and ("content" in msg or "tool_calls" in msg):
        return "ollama"
    return "unknown"


def _extract_response_fields(resp: dict):
    """从响应 JSON 抽出：
        (response_text, tool_calls, first_token_time_s, decode_time_s, prompt_tokens, completion_tokens)
    """
    fmt = _detect_format(resp)

    if fmt == "chat_completion":
        choices = resp.get("choices") or []
        response_text = ""
        tool_calls = None
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            response_text = msg.get("content", "") or ""
            raw_tc = msg.get("tool_calls")
            if raw_tc:
                tool_calls = raw_tc
        usage = resp.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return response_text, tool_calls, None, None, prompt_tokens, completion_tokens

    if fmt == "ollama":
        msg = resp.get("message") or {}
        response_text = msg.get("content", "") or ""
        raw_tc = msg.get("tool_calls")
        tool_calls = raw_tc if raw_tc else None
        prefill_ns = resp.get("prompt_eval_duration")
        decode_ns = resp.get("eval_duration")
        prompt_tokens = resp.get("prompt_eval_count")
        completion_tokens = resp.get("eval_count")
        ftt = float(prefill_ns) / 1e9 if prefill_ns is not None else None
        dec = float(decode_ns) / 1e9 if decode_ns is not None else None
        return response_text, tool_calls, ftt, dec, prompt_tokens, completion_tokens

    msg = resp.get("message") if isinstance(resp, dict) else None
    response_text = (msg or {}).get("content", "") if isinstance(msg, dict) else ""
    raw_tc = (msg or {}).get("tool_calls") if isinstance(msg, dict) else None
    return response_text, raw_tc if raw_tc else None, None, None, None, None


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
        response_text, tool_calls, ftt_s, decode_s, prompt_tok, completion_tok = _extract_response_fields(resp_line)
        case["resp_format"] = fmt
        case["response"] = response_text
        try:
            case["tool_calls"] = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else "[]"
        except (TypeError, ValueError):
            case["tool_calls"] = "[]"

        # === 采样参数 ===
        options = req_line.get("options") or {}
        case["temperature"] = req_line.get("temperature", options.get("temperature"))
        case["top_k"] = req_line.get("top_k", options.get("top_k"))
        case["top_p"] = req_line.get("top_p", options.get("top_p"))
        case["repetitionPenalty"] = req_line.get(
            "repetition_penalty", options.get("repeat_penalty")
        )
        case["max gen tokens"] = req_line.get(
            "max_tokens", options.get("num_predict")
        )

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

        quality_text = response_text
        if tool_calls and not quality_text:
            try:
                quality_text = json.dumps(tool_calls, ensure_ascii=False)
            except (TypeError, ValueError):
                quality_text = str(tool_calls)
        case["get_ans"] = bool(tool_calls) or GET_answer(response_text)
        case["repeat"] = calculate_repetition_rate(quality_text)
        case["entropy"] = calculate_token_entropy(quality_text)

        eval_type = get_eval_type(case["category"]) if case["category"] else "ast"
        case["eval_type"] = eval_type
        verify_kwargs = {}
        try:
            if "tool_calls" in inspect.signature(verify_answer).parameters:
                verify_kwargs["tool_calls"] = tool_calls
        except (TypeError, ValueError):
            pass
        correct, prediction = verify_answer(
            eval_type, case["expect"], response_text, **verify_kwargs
        )
        case["correct"] = bool(correct)
        case["prediction"] = prediction

        results.append(case)

    print(f"{len(results)} tests processed in results {input_path}")
    if fmt_counts:
        fmt_summary = ", ".join(f"{k}={v}" for k, v in fmt_counts.items())
        print(f"[INFO] response formats detected: {fmt_summary}")
    return pd.DataFrame(results)

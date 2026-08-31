# 单条样本的推理 + 评分（BFCL / HumanEval 共用，靠注入的 verify_ans 区分）
#
# BFCL GPU：
#   sentence 含 messages / tools / tool_choice=auto / expect + 采样参数
#   vLLM chat.completions stream=True，传 tools；从 delta.tool_calls 组装调用再评分
#   content 仅作兜底（部分 serving 把 <tool_call> 写进正文）
# HumanEval GPU：sentence 无 tools，行为与原来一致，只读 delta.content

import inspect
import json
import re
import time

from transformers import AutoTokenizer

# 主脚本注入
verify_ans = None
extend_metrics = None


TOKENIZER_CONFIG_PATH = None
tokenizer = None


def init_tokenizer(path):
    global TOKENIZER_CONFIG_PATH, tokenizer
    TOKENIZER_CONFIG_PATH = path
    tokenizer = AutoTokenizer.from_pretrained(path)


def _messages_to_str(messages):
    """把 messages 拼成单串文本，仅用于本地 tokenize 算 prompt token 长度。"""
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n".join(parts)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return default


def _accumulate_tool_call_delta(buckets, delta_tool_calls):
    """按 index 拼 stream 下来的 tool_calls 碎片。"""
    if not delta_tool_calls:
        return
    for tc in delta_tool_calls:
        idx = getattr(tc, "index", None)
        if idx is None and isinstance(tc, dict):
            idx = tc.get("index", 0)
        if idx is None:
            idx = 0
        if idx not in buckets:
            buckets[idx] = {
                "id": None,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        slot = buckets[idx]

        tc_id = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
        tc_type = getattr(tc, "type", None) if not isinstance(tc, dict) else tc.get("type")
        if tc_id:
            slot["id"] = tc_id
        if tc_type:
            slot["type"] = tc_type

        fn = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function")
        if fn is None:
            continue
        if isinstance(fn, dict):
            name = fn.get("name")
            arguments = fn.get("arguments")
        else:
            name = getattr(fn, "name", None)
            arguments = getattr(fn, "arguments", None)
        if name:
            slot["function"]["name"] += name
        if arguments:
            slot["function"]["arguments"] += arguments


def _prompt_token_len(messages, tools):
    global tokenizer
    try:
        kwargs = {"tokenize": True, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        ids = tokenizer.apply_chat_template(messages, **kwargs)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return len(ids)
    except (TypeError, ValueError, Exception):  # noqa: BLE001
        prompt_for_count = _messages_to_str(messages)
        if tools:
            try:
                prompt_for_count += "\n" + json.dumps(tools, ensure_ascii=False)
            except (TypeError, ValueError):
                pass
        return len(tokenizer(prompt_for_count)["input_ids"])


def _call_verify(eval_type, expect, response_text, tool_calls):
    """HumanEval 的 verify_answer 没有 tool_calls 参数，按签名兼容。"""
    fn = verify_ans.verify_answer
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    if "tool_calls" in params:
        return fn(eval_type, expect, response_text, tool_calls=tool_calls)
    return fn(eval_type, expect, response_text)


def measure_performance(client, model_id, testCaseName, sentence):
    global tokenizer
    if tokenizer is None:
        raise RuntimeError(
            "Tokenizer not initialized. Call init_tokenizer(path) before measure_performance."
        )

    messages = sentence["messages"]
    tools = sentence.get("tools") or None
    tool_choice = sentence.get("tool_choice") or "auto"

    enable_thinking = _as_bool(sentence.get("enableThinking"), default=False)
    create_kwargs = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "seed": sentence.get("seed"),
        "temperature": sentence.get("temperature"),
        "top_p": sentence.get("topP"),
        "max_tokens": sentence["maxGenTokens"],
        "extra_body": {
            "top_k": sentence.get("topK"),
            "repetition_penalty": sentence.get("repetitionPenalty"),
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    }
    if tools:
        create_kwargs["tools"] = tools
        create_kwargs["tool_choice"] = tool_choice
    stop = sentence.get("stopSeq")
    if stop:
        create_kwargs["stop"] = stop

    start_time = time.time()
    first_token_time = None
    full_response = ""
    reasoning_content = ""
    tool_call_buckets = {}

    response = client.chat.completions.create(**create_kwargs)

    for chunk in response:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta is None:
            continue

        piece = getattr(delta, "content", None)
        if piece:
            if first_token_time is None:
                first_token_time = time.time() - start_time
            full_response += piece

        # vLLM --reasoning-parser 把思考写到 reasoning_content，不在 content 里
        reason_piece = getattr(delta, "reasoning_content", None) or getattr(
            delta, "reasoning", None
        )
        if reason_piece:
            if first_token_time is None:
                first_token_time = time.time() - start_time
            reasoning_content += reason_piece

        delta_tcs = getattr(delta, "tool_calls", None)
        if delta_tcs:
            if first_token_time is None:
                first_token_time = time.time() - start_time
            _accumulate_tool_call_delta(tool_call_buckets, delta_tcs)

    total_time = time.time() - start_time
    decode_time = (
        total_time - first_token_time if first_token_time is not None else None
    )

    assembled_tool_calls = [
        tool_call_buckets[i] for i in sorted(tool_call_buckets)
        if tool_call_buckets[i].get("function", {}).get("name")
    ]

    matched = re.match(verify_ans.TEST_CASE_NAME_PATTERN, testCaseName)
    if matched is None:
        raise ValueError(f"testCaseName `{testCaseName}` does not match pattern.")

    category = matched.group("vertical")
    eval_type = verify_ans.get_eval_type(category)

    # 结构化 tool_calls 优先；没有则从 content + reasoning 文本抽 <tool_call>
    verify_tool_calls = assembled_tool_calls if assembled_tool_calls else None
    fallback_text = full_response
    if not verify_tool_calls:
        fallback_text = (reasoning_content or "") + (full_response or "")
    correct, prediction = _call_verify(
        eval_type, sentence.get("expect", ""), fallback_text, verify_tool_calls
    )

    quality_text = full_response or reasoning_content
    if assembled_tool_calls and not quality_text:
        try:
            quality_text = json.dumps(assembled_tool_calls, ensure_ascii=False)
        except (TypeError, ValueError):
            quality_text = str(assembled_tool_calls)

    try:
        question_str = json.dumps(messages, ensure_ascii=False)
    except (TypeError, ValueError):
        question_str = str(messages)

    try:
        tool_calls_str = json.dumps(assembled_tool_calls, ensure_ascii=False)
    except (TypeError, ValueError):
        tool_calls_str = str(assembled_tool_calls)

    result = {
        "testCaseName": testCaseName,
        "project_name": matched.group("project_name"),
        "flavor": matched.group("flavor"),
        "vertical": category,
        "category": category,
        "eval_type": eval_type,
        "question": question_str,
        "expect": sentence.get("expect", ""),
        "prompt_token_len": _prompt_token_len(messages, tools),
        "response": full_response,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls_str,
        "response_token_len": len(tokenizer(quality_text or "")["input_ids"]),
        "first_token_time": first_token_time,
        "total_time": total_time,
        "decode_time": decode_time,
        "correct": bool(correct),
        "prediction": prediction,
        "get_ans": bool(assembled_tool_calls) or extend_metrics.GET_answer(fallback_text),
        "repeat": extend_metrics.calculate_repetition_rate(quality_text or ""),
        "entropy": extend_metrics.calculate_token_entropy(quality_text or ""),
    }

    return result

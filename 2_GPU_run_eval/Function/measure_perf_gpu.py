# 单条 BFCL 样本的推理 + 评分
# - sentence 由 Gen_bfcl_cases.py 以 mode="gpu" 产出，含 messages / expect + 采样参数
# - category / eval_type 不在 sentence 字段里，从 testCaseName 解析 vertical 后查
#   verify_ans.CATEGORY_TO_EVAL_TYPE 得到（OMC/GPU/API 共用同一套推导路径）
# - vLLM 走 chat.completions，stream=True，记录 TTFT / 总耗时 / decode 时间
# - 评分调用 verify_ans.verify_answer(eval_type, expect, response)

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


def measure_performance(client, model_id, testCaseName, sentence):
    global tokenizer
    if tokenizer is None:
        raise RuntimeError(
            "Tokenizer not initialized. Call init_tokenizer(path) before measure_performance."
        )

    messages = sentence["messages"]
    start_time = time.time()
    first_token_time = None
    full_response = ""

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        stream=True,
        seed=sentence.get("seed"),
        temperature=sentence.get("temperature"),
        top_p=sentence.get("topP") if "temperature" not in sentence else None,
        max_tokens=sentence["maxGenTokens"],
        extra_body={
            "top_k": sentence.get("topK"),
            "repetition_penalty": sentence.get("repetitionPenalty"),
        },
    )

    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            if first_token_time is None:
                first_token_time = time.time() - start_time
            full_response += piece

    total_time = time.time() - start_time
    decode_time = (
        total_time - first_token_time if first_token_time is not None else None
    )

    matched = re.match(verify_ans.TEST_CASE_NAME_PATTERN, testCaseName)
    if matched is None:
        raise ValueError(f"testCaseName `{testCaseName}` does not match pattern.")

    category = matched.group("vertical")
    eval_type = verify_ans.get_eval_type(category)
    correct, prediction = verify_ans.verify_answer(
        eval_type, sentence.get("expect", ""), full_response
    )

    prompt_for_count = _messages_to_str(messages)

    # 把 messages 序列化进结果（parquet 不太擅长存嵌套 dict 列；string 化更稳）
    try:
        question_str = json.dumps(messages, ensure_ascii=False)
    except (TypeError, ValueError):
        question_str = str(messages)

    # 不要把 messages 这个嵌套字段直接铺进 result（parquet 不友好），
    # 用 question_str 代替；其它采样参数从 sentence 取。
    result = {
        "testCaseName": testCaseName,
        "project_name": matched.group("project_name"),
        "flavor": matched.group("flavor"),
        "vertical": category,
        "category": category,
        "eval_type": eval_type,
        "question": question_str,
        "expect": sentence.get("expect", ""),
        "prompt_token_len": len(tokenizer(prompt_for_count)["input_ids"]),
        "response": full_response,
        "response_token_len": len(tokenizer(full_response)["input_ids"]),
        "first_token_time": first_token_time,
        "total_time": total_time,
        "decode_time": decode_time,
        "correct": bool(correct),
        "prediction": prediction,
        "get_ans": extend_metrics.GET_answer(full_response),
        "repeat": extend_metrics.calculate_repetition_rate(full_response),
        "entropy": extend_metrics.calculate_token_entropy(full_response),
    }

    return result

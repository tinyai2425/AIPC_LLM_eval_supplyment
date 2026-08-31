# 解析 OMC 端跑出来的日志文件 -> DataFrame
#
# 日志结构（每个 test case 之间用 [INFO] ## testCaseName: <name>, start... / ..., Done 分隔）：
#   [INFO] ## testCaseName: project-1-bfcl-test-simple-0, start...
#   [INFO] prompt: <chat-template 应用后的多行 prompt>
#   ...
#   [INFO] curr generation: <think>...               # 流式打印，可忽略
#   [INFO] all generation: <think>...                # 完整响应，多行
#   [INFO] inputTokenCount: 385.
#   [INFO] outputTokenCount: 1766.
#   [INFO] decodeTimeMs: 63319.07700 ms.
#   [INFO] prefillTimeMs: 532.90600 ms.
#   ...
#   [INFO] expect: [{"calculate_triangle_area": {...}}]
#   [INFO] ## testCaseName: project-1-bfcl-test-simple-0, Done
#
# 解析完每条用例后立即跑 verify_ans.verify_answer(eval_type, expect, response) 出 correct + prediction。
# OMC 没有结构化 tool_calls，从 all generation 文本抽 Qwen <tool_call> / JSON。
# eval_type 从 testCaseName 解析出 category 再查 CATEGORY_TO_EVAL_TYPE，跟 GPU 链路一致。

import re
from typing import Dict, List, Optional, Union

import pandas as pd
from transformers import AutoTokenizer


TOKENIZER_CONFIG_PATH = None
tokenizer = None

# 由主脚本注入（避免循环依赖）
GET_answer = None
calculate_repetition_rate = None
calculate_token_entropy = None
verify_answer = None
get_eval_type = None
TEST_CASE_NAME_PATTERN = None
iter_lines_safely = None


def init_tokenizer(path):
    global TOKENIZER_CONFIG_PATH, tokenizer
    TOKENIZER_CONFIG_PATH = path
    tokenizer = AutoTokenizer.from_pretrained(path)


def convert_to_ms(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "us" or unit == "µs":
        return value / 1000
    if unit == "ns":
        return value / 1_000_000
    if unit == "s":
        return value * 1000
    return value  # ms


def parse_llm_test_results(output_path: str) -> pd.DataFrame:
    case_start_pattern = re.compile(
        r"\[(?:INFO|WARNING|ERROR|EXCEPTION)\] ## testCaseName: ([a-zA-Z0-9_\-]+), start.*"
    )
    case_end_pattern = re.compile(
        r"\[(?:INFO|WARNING|ERROR|EXCEPTION)\] ## testCaseName: ([a-zA-Z0-9_\-]+), Done"
    )
    log_prefix_pattern = re.compile(
        r"^\[(?:INFO|WARNING|ERROR|EXCEPTION|ERR|EXCEPT)\]"
    )

    numeric_pattern = re.compile(
        r"\[(?:INFO|WARNING|ERROR|EXCEPTION)\] *([a-zA-Z0-9_ \-]+) *(?::|, time =|=) *(\d+(?:\.\d*)?)\s*$"
    )
    time_pattern = re.compile(
        r"\[(?:INFO|WARNING|ERROR|EXCEPTION)\] *([a-z0-9A-Z_ \-]+) *: *(\d+(?:\.\d*)) *([mun]?s)\.?\s*$"
    )
    text_field_start_pattern = re.compile(
        r"\[(?:INFO|WARNING|ERROR|EXCEPTION)\] *([a-zA-Z0-9_ \-]+) *: *(.*\s)$"
    )

    results: List[Dict[str, Union[str, int, float]]] = []
    case_data: Optional[dict] = None
    current_text_field: Optional[str] = None
    text_buffer: List[str] = []

    for line in iter_lines_safely(output_path):
        if case_data:
            if log_prefix_pattern.match(line):
                # finalize multi-line text field
                if current_text_field:
                    assert text_buffer
                    case_data[current_text_field] = "".join(text_buffer)
                    current_text_field = None
                    text_buffer = []

                if end_match := case_end_pattern.match(line):
                    case_name = end_match.group(1)
                    matched = re.match(TEST_CASE_NAME_PATTERN, case_name)
                    if matched:
                        case_data["project_name"] = matched.group("project_name")
                        case_data["flavor"] = matched.group("flavor")
                        case_data["vertical"] = matched.group("vertical")
                        case_data["category"] = matched.group("vertical")

                    assert case_data and case_data["testCaseName"] == case_name, (
                        f"case_name={case_name}\ncase_data={case_data}, \nline: {line}\n"
                    )

                    # 时间字段 (ms -> s)
                    case_data["decode_time"] = case_data["decodeTimeMs"] / 1000
                    case_data["first_token_time"] = case_data["prefillTimeMs"] / 1000
                    case_data["total_time"] = (
                        case_data["decode_time"] + case_data["first_token_time"]
                    )

                    # 重新 tokenize 算长度（与 GPU 链路口径一致）
                    case_data["prompt_token_len"] = len(
                        tokenizer(case_data.get("prompt", ""))["input_ids"]
                    )
                    response = case_data.get("all generation", "")
                    case_data["response"] = response
                    case_data["response_token_len"] = len(
                        tokenizer(response)["input_ids"]
                    )

                    # 内容质量指标
                    case_data["get_ans"] = GET_answer(response)
                    case_data["repeat"] = calculate_repetition_rate(response)
                    case_data["entropy"] = calculate_token_entropy(response)

                    # 评分：eval_type 从 category 查
                    category = case_data.get("category", "")
                    eval_type = get_eval_type(category)
                    case_data["eval_type"] = eval_type
                    expect_str = (case_data.get("expect", "") or "").strip()
                    case_data["expect"] = expect_str
                    correct, prediction = verify_answer(eval_type, expect_str, response)
                    case_data["correct"] = bool(correct)
                    case_data["prediction"] = prediction

                    results.append(case_data)
                    case_data = None

                elif time_match := time_pattern.match(line):
                    field, value, unit = time_match.groups()
                    case_data[field] = convert_to_ms(float(value), unit)
                elif num_match := numeric_pattern.match(line):
                    field, value = num_match.groups()
                    case_data[field] = float(value)
                elif text_match := text_field_start_pattern.match(line):
                    field, initial_content = text_match.groups()
                    assert not current_text_field and not text_buffer
                    current_text_field = field
                    text_buffer = [initial_content]

            elif current_text_field:
                # multi-line text body
                assert text_buffer
                text_buffer.append(line)

        elif start_match := case_start_pattern.match(line):
            case_name = start_match.group(1)
            assert not case_data
            case_data = {"testCaseName": case_name}

    print(f"{len(results)} tests processed in results {output_path}")
    return pd.DataFrame(results)

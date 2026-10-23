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
# 解析完每条用例后立即评分。
# BFCL：若注入了 vllm_parser，用 vLLM 的 reasoning / tool-call parser 把原文拆成
# reasoning_content + content + tool_calls，再按结构化 tool_calls 评分（与 GPU 同一条 verify）。
# 没注入时（HumanEval，或 BFCL 未指定 parser）仍把 all generation 整段交给 verify_answer。
# eval_type 从 testCaseName 解析出 category 再查 CATEGORY_TO_EVAL_TYPE，跟 GPU 链路一致。

import inspect
import json
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
# BFCL 主脚本注入。None 时不走 vLLM parser。
vllm_parser = None
_parser_error_logs = 0


def init_tokenizer(path):
    global TOKENIZER_CONFIG_PATH, tokenizer
    TOKENIZER_CONFIG_PATH = path
    tokenizer = AutoTokenizer.from_pretrained(path)


def init_vllm_parser(parser):
    global vllm_parser, _parser_error_logs
    vllm_parser = parser
    _parser_error_logs = 0


def _score_case(eval_type, expect_str, response_text, tool_calls):
    fn = verify_answer
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    if tool_calls is not None and "tool_calls" in params:
        return fn(eval_type, expect_str, response_text, tool_calls=tool_calls)
    return fn(eval_type, expect_str, response_text)


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
    global _parser_error_logs
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

                    # 内容质量指标按原文算（含思考），和 token 长度同一口径
                    case_data["get_ans"] = GET_answer(response)
                    case_data["repeat"] = calculate_repetition_rate(response)
                    case_data["entropy"] = calculate_token_entropy(response)

                    category = case_data.get("category", "")
                    eval_type = get_eval_type(category)
                    case_data["eval_type"] = eval_type
                    expect_str = (case_data.get("expect", "") or "").strip()
                    case_data["expect"] = expect_str

                    score_text = response
                    tool_calls_for_verify = None
                    if vllm_parser is not None:
                        try:
                            parsed = vllm_parser.parse(
                                response, prompt=case_data.get("prompt", "")
                            )
                        except Exception as exc:  # noqa: BLE001
                            _parser_error_logs += 1
                            if _parser_error_logs <= 5:
                                print(
                                    f"[WARN] vLLM parser failed on {case_name}: {exc}"
                                )
                            case_data["reasoning_content"] = ""
                            case_data["parsed_content"] = ""
                            case_data["tool_calls"] = "[]"
                            case_data["correct"] = False
                            case_data["prediction"] = json.dumps(
                                {"parser_error": str(exc)}, ensure_ascii=False
                            )
                            results.append(case_data)
                            case_data = None
                            continue

                        case_data["reasoning_content"] = parsed.reasoning_content
                        case_data["parsed_content"] = parsed.content
                        case_data["tool_calls"] = json.dumps(
                            parsed.tool_calls, ensure_ascii=False
                        )
                        # 空列表也要传进去：表示 parser 明确没调工具，不再走内置文本抽取
                        tool_calls_for_verify = parsed.tool_calls
                        score_text = parsed.content or ""
                        case_data["get_ans"] = bool(parsed.tool_calls) or GET_answer(
                            score_text
                        ) or GET_answer(response)

                    correct, prediction = _score_case(
                        eval_type, expect_str, score_text, tool_calls_for_verify
                    )
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

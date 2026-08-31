# 多线程拉远端 vLLM
# 与 Ceval 版接口对齐：load test_config_path -> 每条 sentences[0] 串成任务 -> 收集结果到 DataFrame。
# 单条失败不再从分母消失：记 correct=False + prediction.error，继续跑完。

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import pandas as pd

# 主脚本注入
verify_ans = None
measure_perf_gpu = None


def _error_result(testCaseName, exc):
    matched = re.match(verify_ans.TEST_CASE_NAME_PATTERN, testCaseName)
    category = matched.group("vertical") if matched else ""
    return {
        "testCaseName": testCaseName,
        "project_name": matched.group("project_name") if matched else "",
        "flavor": matched.group("flavor") if matched else "",
        "vertical": category,
        "category": category,
        "eval_type": verify_ans.get_eval_type(category) if category else "",
        "question": "",
        "expect": "",
        "prompt_token_len": 0,
        "response": "",
        "reasoning_content": "",
        "tool_calls": "[]",
        "response_token_len": 0,
        "first_token_time": None,
        "total_time": None,
        "decode_time": None,
        "correct": False,
        "prediction": json.dumps({"error": str(exc)}, ensure_ascii=False),
        "get_ans": False,
        "repeat": 0.0,
        "entropy": 0.0,
    }


def parallel_fetch_reference_model(test_config_path, server_ip, server_port, model_id):
    client = openai.OpenAI(
        base_url=f"http://{server_ip}:{server_port}/v1",
        api_key="no-api-key-needed",
    )

    with open(test_config_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    assert all(len(t["sentences"]) == 1 for t in test_cases)
    assert all(
        re.match(verify_ans.TEST_CASE_NAME_PATTERN, t["testCaseName"]) for t in test_cases
    )
    assert len({t["testCaseName"] for t in test_cases}) == len(test_cases)

    start_time = time.time()
    results = []
    max_workers = max(1, (os.cpu_count() or 4) * 5)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {}
        for test in test_cases:
            future = executor.submit(
                measure_perf_gpu.measure_performance,
                client,
                model_id,
                test["testCaseName"],
                test["sentences"][0],
            )
            future_to_name[future] = test["testCaseName"]

        for i, future in enumerate(as_completed(future_to_name)):
            name = future_to_name[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"\n[ERROR] {name} failed: {exc}")
                result = _error_result(name, exc)
            results.append(result)
            ttft = result.get("first_token_time") or 0.0
            ttc = result.get("total_time") or 0.0
            remaining_time = (time.time() - start_time) / (i + 1) * (len(test_cases) - i - 1)
            print(
                f"\r{i + 1}/{len(test_cases)} test cases processed "
                f"(TTFT:{ttft:.2f}s/TTC:{ttc:.2f}s), {remaining_time:.2f}s remaining ...",
                end="",
                flush=True,
            )

    result_map = {r["testCaseName"]: r for r in results}
    sorted_results = [result_map[t["testCaseName"]] for t in test_cases if t["testCaseName"] in result_map]
    df_results = pd.DataFrame(sorted_results)
    print(
        f"\r{len(sorted_results)} test cases processed "
        f"({time.time() - start_time:.2f}s spent) in {test_config_path}"
    )
    return df_results


def save_answers_jsonl(df, path):
    """把 GPU 原始回答落成 jsonl：每行 testCaseName / messages / response。"""
    with open(path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            messages = rec.get("question") or []
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages) if messages else []
                except (TypeError, ValueError):
                    messages = [{"role": "user", "content": messages}]
            tool_calls = rec.get("tool_calls", [])
            if isinstance(tool_calls, str):
                try:
                    tool_calls = json.loads(tool_calls) if tool_calls else []
                except (TypeError, ValueError):
                    tool_calls = []
            elif tool_calls is None or (isinstance(tool_calls, float) and pd.isna(tool_calls)):
                tool_calls = []
            content = rec.get("response")
            if content is None or (isinstance(content, float) and pd.isna(content)):
                content = ""
            reasoning = rec.get("reasoning_content")
            if reasoning is None or (isinstance(reasoning, float) and pd.isna(reasoning)):
                reasoning = ""
            line = {
                "testCaseName": rec.get("testCaseName", ""),
                "messages": messages,
                "response": {
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": tool_calls,
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

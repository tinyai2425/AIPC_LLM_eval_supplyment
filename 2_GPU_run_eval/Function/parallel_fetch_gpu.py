# 多线程拉远端 vLLM
# 与 Ceval 版接口对齐：load test_config_path -> 每条 sentences[0] 串成任务 -> 收集结果到 DataFrame。

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
        futures = []
        for test in test_cases:
            future = executor.submit(
                measure_perf_gpu.measure_performance,
                client,
                model_id,
                test["testCaseName"],
                test["sentences"][0],
            )
            futures.append(future)

        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
            except Exception as exc:
                print(f"\n[ERROR] one sample failed: {exc}")
                continue
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
    # 按测试用例文件里的原顺序对齐输出
    sorted_results = [result_map[t["testCaseName"]] for t in test_cases if t["testCaseName"] in result_map]
    df_results = pd.DataFrame(sorted_results)
    print(
        f"\r{len(sorted_results)} test cases processed "
        f"({time.time() - start_time:.2f}s spent) in {test_config_path}"
    )
    return df_results

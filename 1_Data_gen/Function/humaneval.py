# 从 OpenAI HumanEval 数据集生成测试用例
#
# 数据集目录约定（路径由调用方传入 dataset_root，默认 AIPC_LLM_eval_supplyment/Data_set/openai_humaneval/）：
#   {dataset_root}/openai_humaneval/test-*.parquet      ← HuggingFace 默认布局
#   或 {dataset_root}/test-*.parquet                     ← 直接放 parquet 也支持
#
# 数据集本身没有提交到 git，需自行下载：
#   HuggingFace: https://huggingface.co/datasets/openai_humaneval
#   建议用 `huggingface-cli download openai_humaneval --repo-type dataset --local-dir <path>`
#
# parquet 字段：task_id / prompt / canonical_solution / test / entry_point
#   - prompt          函数 header + docstring（要求模型续写函数体）
#   - canonical_solution  官方参考实现（不喂模型，只是参考）
#   - test            一段 Python 代码，定义 `def check(candidate)` 做单元测试断言
#   - entry_point     函数名，verify 时 exec `check(<entry_point>)`
#
# 三种 mode 出三种产物结构（详见 test_utils.py）：
#   - "omc" : 严格 Ceval，system+user 已用 chat_template 渲染成单串 prompt
#   - "gpu" : Ceval 外壳 + sentences[0].messages（直接送 vLLM）
#   - "api" : OpenAI 风格顶层 messages
#
# expect 字段塞 JSON-encoded {prompt, test, entry_point}，evaluator 那侧解出来 exec 跑测试。

import glob
import json
import os

import pyarrow.parquet as pq

import test_utils


SYSTEM_PROMPT = (
    "You are an expert Python programmer. The user will give you a Python function "
    "header (signature + docstring). Your job is to write the complete function "
    "implementation.\n\n"
    "Output rules (STRICT):\n"
    "1. Output ONLY Python code wrapped in a single ```python ... ``` fenced block.\n"
    "2. The code block MUST contain the complete function: re-emit the exact signature "
    "the user gave you, then write the body.\n"
    "3. Do not include any prose / explanation / test code / example calls outside the "
    "code block.\n"
    "4. Do not redefine, import, or modify anything outside the function body unless "
    "the function header itself relies on additional imports.\n"
)


USER_PROMPT_TEMPLATE = (
    "Complete this Python function. Output the full function (signature + body) in a "
    "single ```python ... ``` code block.\n\n"
    "{prompt}"
)


# HumanEval 是代码续写任务，除模型默认 stopSeq 外再加一组"刹车"，
# 避免模型继续往下写多余的 def / 测试 / main 入口。
EXTRA_STOP_SEQ = [
    "\nclass ",
    "\nif __name__",
    "\ndef test_",
    "\nprint(",
]


def _find_parquet_files(dataset_root):
    """兼容 HuggingFace 下载布局（带 openai_humaneval 子目录）和直接放 parquet 两种情况。"""
    candidates = []
    for sub in (".", "openai_humaneval"):
        pattern = os.path.join(dataset_root, sub, "test-*.parquet")
        candidates.extend(sorted(glob.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(
            f"未在 {dataset_root} 下找到 test-*.parquet（试过 ./ 和 ./openai_humaneval/）"
        )
    return candidates


def load_humaneval(dataset_root, limit=None):
    """读取所有 parquet 拼成 list[dict]，每条含 task_id / prompt / test / entry_point / canonical_solution。"""
    rows = []
    for f in _find_parquet_files(dataset_root):
        table = pq.read_table(f)
        rows.extend(table.to_pylist())
    if limit is not None:
        rows = rows[:limit]
    return rows


def _task_index_from_id(task_id):
    """HumanEval/123 -> 123。"""
    if "/" in task_id:
        tail = task_id.rsplit("/", 1)[-1]
    else:
        tail = task_id
    try:
        return int(tail)
    except ValueError:
        raise ValueError(f"task_id `{task_id}` 不是 HumanEval/N 形式")


def build_user_prompt(prompt):
    return USER_PROMPT_TEMPLATE.format(prompt=prompt)


def _build_one_case(mode, project_name, idx, record, max_output_token_length):
    name = f"{project_name}-humaneval-test-python-{idx}"

    user_prompt = build_user_prompt(record["prompt"])

    expect = json.dumps(
        {
            "prompt": record["prompt"],
            "test": record["test"],
            "entry_point": record["entry_point"],
            "task_id": record["task_id"],
        },
        ensure_ascii=False,
    )

    if mode == "omc":
        rendered = test_utils.render_chat_template(SYSTEM_PROMPT, user_prompt)
        return test_utils.create_omc_test(
            name, rendered, expect, max_output_token_length, extra_stop_seq=EXTRA_STOP_SEQ
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if mode == "gpu":
        return test_utils.create_gpu_test(
            name, messages, expect, max_output_token_length, extra_stop_seq=EXTRA_STOP_SEQ
        )
    if mode == "api":
        return test_utils.create_api_test(
            name, messages, expect, max_output_token_length, extra_stop_seq=EXTRA_STOP_SEQ
        )
    raise ValueError(f"unknown mode: {mode}")


def generate_humaneval_tests(
    project_name,
    dataset_root,
    mode,
    max_output_token_length=5000,
    limit=None,
):
    """mode in {'omc', 'gpu', 'api'}；limit=None 跑全量 164 条，整数表示前 N 条。"""
    if mode not in ("omc", "gpu", "api"):
        raise ValueError(f"mode must be one of omc/gpu/api, got {mode}")

    records = load_humaneval(dataset_root, limit=limit)

    test_cases = []
    for r in records:
        idx = _task_index_from_id(r["task_id"])
        test_cases.append(
            _build_one_case(mode, project_name, idx, r, max_output_token_length)
        )

    return test_cases

# HumanEval 评分：抽 Python 代码块 -> 组装完整程序 -> subprocess 沙箱执行 -> 看 check() 是否通过
#
# 抽取规则（尽量宽容）：
#   1) 优先 ```python``` / ```py``` / 无标记 ``` 代码块
#   2) 退化：去掉 <think>...</think> 后用整段
#   3) 如果抽出来的代码已经含 `def {entry_point}(`，直接当完整程序；
#      否则当作"只写了函数体"，把原 HumanEval prompt（含 def 行和 docstring）拼在前面。
#
# 执行规则：
#   - 用 `python -c <program>` 起子进程跑
#   - 超时（默认 10s）按失败计
#   - 程序结尾追加 `check({entry_point})`；只要 returncode == 0 就算 PASS
#
# expect 字段由 1_Data_gen/Function/humaneval.py 写入，是 JSON：
#   {"prompt": "<def header + docstring>", "test": "<check(candidate) 函数定义>",
#    "entry_point": "<函数名>", "task_id": "HumanEval/N"}

import json
import os
import re
import subprocess
import sys


# 与 BFCL 共用同一条 testCaseName 解析正则。直接 import 复用，避免重复定义。
from verify_ans import TEST_CASE_NAME_PATTERN  # noqa: E402,F401


# HumanEval 只有一种评分路径；保留接口以兼容 measure_perf_gpu 注入式调用
def get_eval_type(category):
    return "humaneval"


_FENCED_PATTERN = re.compile(
    r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE
)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text):
    return _THINK_PATTERN.sub("", text or "")


def extract_code(response):
    """从模型响应里抽 Python 代码。失败也不抛异常，返回原文兜底。

    注意：返回值**不能 .strip()**——HumanEval 的 body-only 模式下，
    模型只输出缩进的函数体（首行就是 4 个空格），strip 会把这层缩进吃掉
    导致拼回 prompt 后变成顶格 SyntaxError/IndentationError。
    """
    if not response:
        return ""
    text = _strip_thinking(response)
    fenced = _FENCED_PATTERN.findall(text)
    if fenced:
        # 取最长的一段（避免抽到示例小段）；只剪两端的换行，保留首行可能的缩进
        return max(fenced, key=len).strip("\n")
    # 无 fence：剪掉前置空白行 + 末尾空白，但保留每行的缩进
    return text.lstrip("\n").rstrip()


def assemble_program(prompt_header, extracted_code, test_code, entry_point):
    """把模型抽出的代码组装成完整可执行 Python 程序。

    - 若抽出的代码含 `def {entry_point}(`，认为模型直接写了完整函数（包含可能的 import），直接拿来跑；
    - 否则视为只写了函数体（缩进的若干行），把 HumanEval 原 prompt（函数 header + docstring）拼到前面。
    """
    has_signature = re.search(
        rf"def\s+{re.escape(entry_point)}\s*\(", extracted_code or ""
    ) is not None

    if has_signature:
        body_section = extracted_code
    else:
        body_section = prompt_header + (extracted_code or "")

    return (
        body_section
        + "\n\n"
        + test_code
        + "\n\n"
        + f"check({entry_point})\n"
    )


def _run_check(program, timeout):
    """returns (returncode, stderr_tail). returncode == 0 即通过。"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            timeout=timeout,
            text=True,
            # 隔离一下：不继承父进程的关键环境变量（防止意外打到外部 API 等）
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        return proc.returncode, proc.stderr[-800:] if proc.stderr else ""
    except subprocess.TimeoutExpired:
        return -1, f"[TIMEOUT after {timeout}s]"
    except Exception as exc:  # noqa: BLE001
        return -2, f"[EXEC ERROR] {exc!r}"


def verify_answer(eval_type, expect, response_text, timeout=10):
    """返回 (correct: bool, prediction_str: str)。
    prediction_str 是个 JSON：{"code": <抽出的代码>, "error": <若 fail 时的 stderr 尾巴>}。"""
    code = extract_code(response_text)

    try:
        meta = json.loads(expect) if expect else {}
    except (TypeError, ValueError):
        meta = {}

    prompt_header = meta.get("prompt", "")
    test_code = meta.get("test", "")
    entry_point = meta.get("entry_point", "")

    if not entry_point or not test_code:
        # expect 不完整，没法判分
        return False, json.dumps({"code": code, "error": "missing entry_point/test in expect"})

    program = assemble_program(prompt_header, code, test_code, entry_point)
    rc, stderr_tail = _run_check(program, timeout)
    correct = (rc == 0)

    return correct, json.dumps(
        {"code": code, "error": "" if correct else stderr_tail},
        ensure_ascii=False,
    )

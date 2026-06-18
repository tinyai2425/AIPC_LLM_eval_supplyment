# 用法:
#   python Gen_humaneval_cases.py <model_config.json> <DATASET_ROOT> [<LIMIT>]
#
# 产出（在 ../../model-eval-storage/{model_name}/project-N/ 下）：
#   project-N-HUMANEVAL-OMC-{N}.json   OMC：严格原版 Ceval 结构（chat template 已应用）
#   project-N-HUMANEVAL-GPU-{N}.json   GPU：Ceval 外壳 + sentences[0].messages
#   project-N-HUMANEVAL-API-{N}.jsonl  API：OpenAI 风格顶层 messages
#   api_config.json                    推理 runtime 参数
#
# 注：HumanEval 是代码续写评估，sentence.stopSeq 在模型默认基础上额外加了 \nclass / \nif __name__ / 等代码"刹车"。
# 评分见 2_GPU_run_eval/Function/verify_ans_humaneval.py：抽 ```python``` 块 -> 拼 test -> subprocess 执行。

import json
import os
import re
import sys
from types import SimpleNamespace

from transformers import AutoTokenizer

sys.path.append(os.path.abspath("Function"))
import humaneval as he
import test_utils


USAGE = "Usage: python Gen_humaneval_cases.py <model_config.json> <DATASET_ROOT> [<LIMIT>]"


if len(sys.argv) < 3 or len(sys.argv) > 4:
    print(USAGE)
    sys.exit(1)

config_path = sys.argv[1]
dataset_root = sys.argv[2]
limit = None
if len(sys.argv) == 4:
    try:
        limit = int(sys.argv[3])
    except ValueError:
        raise ValueError(f"LIMIT 必须是整数，got: {sys.argv[3]}")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"找不到配置文件: {config_path}")
if not os.path.isdir(dataset_root):
    raise FileNotFoundError(f"找不到数据集目录: {dataset_root}")

with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)
mp = SimpleNamespace(**config_dict)

# 跟 BFCL 对齐：5000 token 上限。
# 注：HumanEval 函数本体一般 64~512 token，但 Qwen3 之类的 thinking 模型常常写 1500+ 思考 token，
# 给到 5000 才能容下"完整 think + 完整代码块"——之前给 2000 导致大量样例被截断，造成假阴性。
MAX_OUTPUT_TOKEN_LENGTH = 5000

#if not re.fullmatch(r"[A-Za-z0-9_\-]+", mp.model_name):
#    raise ValueError(f"非法的 MODEL_NAME: {mp.model_name}，只能包含字母、数字、- 和 _")

PROJECT_BASE = f"../../model-eval-storage/{mp.model_name}"
PROJECT_PREFIX = "project"


def get_unique_subdir(base_dir, prefix):
    counter = 1
    while os.path.exists(f"{base_dir}/{prefix}-{counter}"):
        counter += 1
    return f"{prefix}-{counter}"


PROJECT_NAME = get_unique_subdir(PROJECT_BASE, PROJECT_PREFIX)
PROJECT_PATH = f"{PROJECT_BASE}/{PROJECT_NAME}"
os.makedirs(PROJECT_PATH, exist_ok=True)

print(f"PROJECT_NAME {PROJECT_NAME}")
print(f"Full path to project dir: {PROJECT_PATH}")

test_utils.set_model_config(mp)


def _gen(mode):
    return he.generate_humaneval_tests(
        PROJECT_NAME,
        dataset_root,
        mode=mode,
        max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH,
        limit=limit,
    )


# ===== OMC =====
omc_cases = _gen("omc")
N = len(omc_cases)
test_path_omc = f"{PROJECT_PATH}/{PROJECT_NAME}-HUMANEVAL-OMC-{N}.json"
with open(test_path_omc, "w", encoding="utf-8") as f:
    json.dump(omc_cases, f, indent=4, ensure_ascii=False)
    print(f"{N} OMC test cases saved to {test_path_omc}")

# ===== GPU =====
gpu_cases = _gen("gpu")
test_path_gpu = f"{PROJECT_PATH}/{PROJECT_NAME}-HUMANEVAL-GPU-{N}.json"
with open(test_path_gpu, "w", encoding="utf-8") as f:
    json.dump(gpu_cases, f, indent=4, ensure_ascii=False)
    print(f"{len(gpu_cases)} GPU test cases saved to {test_path_gpu}")

# ===== API =====
api_cases = _gen("api")
test_path_api = f"{PROJECT_PATH}/{PROJECT_NAME}-HUMANEVAL-API-{N}.jsonl"
with open(test_path_api, "w", encoding="utf-8") as f:
    for case in api_cases:
        json.dump(case, f, ensure_ascii=False)
        f.write("\n")
    print(f"{len(api_cases)} API test cases saved to {test_path_api}")


# ===== api_config.json =====
# 与 BFCL 共用：把模型 runtime 参数 + chat template + license 一并写出
tokenizer = AutoTokenizer.from_pretrained(mp.TOKENIZER_CONFIG_PATH)
CHAT_TEMPLATE = tokenizer.chat_template

license_text = ""
license_file = f"{mp.TOKENIZER_CONFIG_PATH}/LICENSE"
if os.path.isfile(license_file):
    with open(license_file, "r", encoding="utf-8") as f:
        license_text = f.read()

api_config_path = f"{PROJECT_PATH}/api_config.json"
api_config = {
    "inferType": mp.INFER_TYPE,
    "tokenizerType": mp.TOKENIZER_TYPE,
    "tokenizerPath": mp.TOKENIZER_PATH,
    "modelType": mp.MODEL_TYPE,
    "modelPath": mp.MODEL_PATH,
    "weightDir": mp.WEIGHT_DIR,
    "prefixPrompt": mp.PREFIX_PROMPT,
    "pmtCacheOperation": mp.PMT_CACHE_OP,
    "pfxInitTokenLen": mp.PFX_INIT_TOKEN_LEN,
    "loraCfgPath": mp.LORA_CFG_PATH,
    "expect": "",
    "callbackFreq": mp.CALLBACK_FREQ,
    "sampleFlag": mp.SAMPLE_FLAG,
    "seed": mp.SEED,
    "topK": mp.TOPK,
    "topP": mp.TOPP,
    "temperature": mp.TEMPERATURE,
    "maxGenTokens": MAX_OUTPUT_TOKEN_LENGTH,
    "repetitionPenalty": mp.REPETITIONPENALTY,
    "initTokenLen": mp.INIT_TOKEN_LEN,
    "stopSeq": list(mp.STOP_SEQ) + he.EXTRA_STOP_SEQ,
    "isAsync": mp.IS_ASYNC,
    "chatTemplate": CHAT_TEMPLATE,
    "modelInfo": {"license": license_text},
}
with open(api_config_path, "w", encoding="utf-8") as f:
    json.dump(api_config, f, indent=4, ensure_ascii=False)
    print(f"api_config.json saved to {api_config_path}")

print("[ALL DONE]")

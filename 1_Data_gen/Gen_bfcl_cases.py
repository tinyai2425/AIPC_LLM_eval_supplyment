# 用法:
#   python Gen_bfcl_cases.py <model_config.json> <DATASET_ROOT> <MAX_CASES_PER_CATEGORY> [--categories cat1,cat2,...]
#
# 产出（在 ../../model-eval-storage/{model_name}/project-N/ 下）：
#   project-N-BFCL-OMC-{P}.json   OMC：严格原版 Ceval 结构（chat template 已应用，单串 prompt）
#   project-N-BFCL-GPU-{P}.json   GPU：Ceval 外壳 + sentences[0].messages（直接送 vLLM）
#   project-N-BFCL-API-{P}.jsonl  API：OpenAI 风格顶层 messages
#   api_config.json               推理 runtime 参数
#
# 注：性能测试用例在 Ceval_ref 那一套里做，本目录只生成工具调用评估用例。

import json
import os
import re
import sys
from types import SimpleNamespace

from transformers import AutoTokenizer

sys.path.append(os.path.abspath("Function"))
import bfcl
import test_utils


USAGE = (
    "Usage: python Gen_bfcl_cases.py <model_config.json> <DATASET_ROOT> "
    "<MAX_CASES_PER_CATEGORY> [--categories cat1,cat2,...]"
)


def _flag_val(flag):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv):
        return None
    return sys.argv[i + 1]


if len(sys.argv) < 4:
    print(USAGE)
    sys.exit(1)

config_path = sys.argv[1]
dataset_root = sys.argv[2]
try:
    MAX_CASES_PER_CATEGORY = int(sys.argv[3])
except ValueError:
    raise ValueError("MAX_CASES_PER_CATEGORY 必须是整数")

categories_str = _flag_val("--categories")
categories = None
if categories_str:
    wanted = [c.strip() for c in categories_str.split(",") if c.strip()]
    missing = [c for c in wanted if c not in bfcl.DEFAULT_CATEGORIES]
    if missing:
        print(f"[WARN] unknown categories ignored: {missing}")
    categories = {c: bfcl.DEFAULT_CATEGORIES[c] for c in wanted if c in bfcl.DEFAULT_CATEGORIES}
    if not categories:
        raise ValueError(f"No valid categories from --categories {categories_str}")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"找不到配置文件: {config_path}")
if not os.path.isdir(dataset_root):
    raise FileNotFoundError(f"找不到数据集目录: {dataset_root}")

with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)
mp = SimpleNamespace(**config_dict)

MAX_OUTPUT_TOKEN_LENGTH = 5000

if not re.fullmatch(r"[A-Za-z0-9_\-]+", mp.model_name):
    raise ValueError(f"非法的 MODEL_NAME: {mp.model_name}，只能包含字母、数字、- 和 _")

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
    return bfcl.generate_bfcl_tests(
        PROJECT_NAME,
        dataset_root,
        MAX_CASES_PER_CATEGORY,
        mode=mode,
        max_output_token_length=MAX_OUTPUT_TOKEN_LENGTH,
        categories=categories,
    )


# ===== OMC =====
test_path_omc = f"{PROJECT_PATH}/{PROJECT_NAME}-BFCL-OMC-{MAX_CASES_PER_CATEGORY}.json"
with open(test_path_omc, "w", encoding="utf-8") as f:
    cases = _gen("omc")
    json.dump(cases, f, indent=4, ensure_ascii=False)
    print(f"{len(cases)} OMC test cases saved to {test_path_omc}")

# ===== GPU =====
test_path_gpu = f"{PROJECT_PATH}/{PROJECT_NAME}-BFCL-GPU-{MAX_CASES_PER_CATEGORY}.json"
with open(test_path_gpu, "w", encoding="utf-8") as f:
    cases = _gen("gpu")
    json.dump(cases, f, indent=4, ensure_ascii=False)
    print(f"{len(cases)} GPU test cases saved to {test_path_gpu}")

# ===== API =====
test_path_api = f"{PROJECT_PATH}/{PROJECT_NAME}-BFCL-API-{MAX_CASES_PER_CATEGORY}.jsonl"
with open(test_path_api, "w", encoding="utf-8") as f:
    cases = _gen("api")
    for case in cases:
        json.dump(case, f, ensure_ascii=False)
        f.write("\n")
    print(f"{len(cases)} API test cases saved to {test_path_api}")


# ===== api_config.json =====
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
    "stopSeq": mp.STOP_SEQ,
    "isAsync": mp.IS_ASYNC,
    "chatTemplate": CHAT_TEMPLATE,
    "modelInfo": {"license": license_text},
}
with open(api_config_path, "w", encoding="utf-8") as f:
    json.dump(api_config, f, indent=4, ensure_ascii=False)
    print(f"api_config.json saved to {api_config_path}")

print("[ALL DONE]")

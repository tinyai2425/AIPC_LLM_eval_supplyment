# AIPC LLM Eval Supplement — BFCL / HumanEval 评估

参照 `Ceval_ref` 的流程，把数据集换成 **两套互相独立**的评估目标，可分别单独跑：

| 数据集 | 任务类型 | 评分方式 | 主入口 |
| --- | --- | --- | --- |
| **BFCL v3**（Berkeley Function Calling Leaderboard） | 工具调用 / prompt-based function calling | AST 调用无序匹配 + relevance/irrelevance 布尔判定 | `Gen_bfcl_cases.py` / `Eval_BFCL_GPU_results.py` |
| **HumanEval**（OpenAI HumanEval） | 代码生成（Python 函数补全） | 抽出代码 → 拼 test → subprocess 执行 `check(entry_point)` → pass/fail | `Gen_humaneval_cases.py` / `Eval_HumanEval_GPU_results.py` |

两套都不生成性能测试用例（prefill/decode 吞吐的随机长 prompt 测试在 `Ceval_ref` 那一侧做）。

### BFCL 覆盖

第一版覆盖 BFCL v3 的"单轮 AST + 相关性"共 **13 个类别**（约 3000 条），不包含 exec / rest / sql / multi_turn / chatable。

| 大组         | 细类                                                                                          | 评分方式          |
| ------------ | --------------------------------------------------------------------------------------------- | ----------------- |
| Non-live AST | `simple` / `multiple` / `parallel` / `parallel_multiple` / `java` / `javascript`              | AST 调用匹配      |
| Live AST     | `live_simple` / `live_multiple` / `live_parallel` / `live_parallel_multiple`                  | AST 调用匹配      |
| Relevance    | `irrelevance` / `live_irrelevance`（应拒绝调用）/ `live_relevance`（应至少调用一次）          | 调用与否布尔判定  |

### HumanEval 覆盖

全部 164 条 Python 函数补全。模型输入是函数 header + docstring，要求输出完整函数；评分时把抽出来的代码与官方 `check(candidate)` 测试组装成完整程序，在子进程里跑（默认 10s 超时）。

prompt 形式跟 BFCL 不同：HumanEval 是代码续写任务，所以 system 明确要求模型把完整函数（signature+body）放在一个 ```` ```python ... ``` ```` 代码块里输出。模型默认的 stopSeq 之外又加了一组代码续写"刹车"（` \nclass ` / ` \nif __name__` / ` \ndef test_` / ` \nprint(`），避免越界续写其他函数 / 测试 / main 入口。

## 一、目录结构

```
AIPC_LLM_eval_supplyment/
├── README.md
├── requirements.txt                # Python 依赖
├── model_config/                   # 各模型采样/路径配置 JSON（与 Ceval_ref 同款）
├── Model_file/                     # 本地 tokenizer / 模型权重目录
├── Data_set/
│   ├── Berkeley-Function-Calling-Leaderboard/   # BFCL v3 数据集（自行下载，见 二.3 节）
│   └── openai_humaneval/                        # HumanEval 数据集（自行下载，见 二.3 节）
├── 1_Data_gen/                     # 阶段一：生成 OMC / GPU / API 三套测试用例
│   ├── Gen_bfcl_cases.py           # BFCL 入口
│   ├── run_gen_bfcl.sh
│   ├── Gen_humaneval_cases.py      # HumanEval 入口
│   ├── run_gen_humaneval.sh
│   └── Function/
│       ├── bfcl.py                 # BFCL：13 类 jsonl + possible_answer 合并加载，拼 prompt
│       ├── humaneval.py            # HumanEval：parquet 加载 + 代码续写 prompt + extra_stop_seq
│       └── test_utils.py           # create_omc_test / create_gpu_test / create_api_test（含 extra_stop_seq 参数）
├── 2_GPU_run_eval/                 # 阶段二 a：打远端 vLLM，落 parquet + Excel + 趋势图
│   ├── Eval_BFCL_GPU_results.py    # BFCL GPU 评估入口
│   ├── run_eval_bfcl_gpu_results.sh
│   ├── Eval_HumanEval_GPU_results.py    # HumanEval GPU 评估入口
│   ├── run_eval_humaneval_gpu_results.sh
│   └── Function/
│       ├── verify_ans.py                # BFCL：从响应抽 JSON 调用 + 路由到 AST / relevance 评分
│       ├── ast_checker.py               # BFCL AST 调用匹配核心
│       ├── eval_results.py              # BFCL：summary / 3 大组 / 13 细类 三层 sheet
│       ├── verify_ans_humaneval.py      # HumanEval：抽 ```python``` 代码 + subprocess 跑 check()
│       ├── eval_results_humaneval.py    # HumanEval：summary（Pass@1） + failed cases 列表
│       ├── measure_perf_gpu.py          # 通用：单条推理 + 评分（注入式 verify_ans）
│       ├── parallel_fetch_gpu.py        # 通用：多线程并发
│       ├── extend_metrics.py            # 通用：get_ans / repeat / token entropy
│       ├── anlz_cont_quality.py         # 通用：entropy / repeat 趋势图
│       ├── extract_name.py              # 通用：解析路径 + 生成带日期的输出目录
│       └── write_average_sheet.py       # 通用：多 version 合并求均值 -> markdown
├── 3_OMC_eval/                     # 阶段二 b：解析 OMC 日志 .txt 出精度（复用 2_GPU_run_eval 的评分/聚合模块）
│   ├── Eval_BFCL_OMC_results.py
│   ├── run_eval_bfcl_omc_results.sh
│   └── Function/
│       ├── OMC_collect.py          # 把 OMC 日志拆成 per-case 字段 + 跑 BFCL 评分
│       └── file_utils.py           # 自动检测编码逐行读
└── 4_API_eval/                     # 阶段二 c：解析 API（Ollama 风格）日志 .txt 出精度
    ├── Eval_BFCL_API_results.py
    ├── run_eval_bfcl_api_results.sh
    └── Function/
        └── API_collect.py          # 把 API 日志（每 3 行为一组）拆成 per-case 字段 + 跑 BFCL 评分
```

测试用例生成产物落在 `../../model-eval-storage/{model_name}/project-N/`，跟 Ceval_ref 一致。

## 二、前置准备

### 1. Python 环境

要求 **Python >= 3.10**（开发环境为 3.10.20）。建议用 conda / venv 隔离：

```bash
# conda 方式
conda create -n bfcl_eval python=3.10 -y
conda activate bfcl_eval

# 或者 venv 方式
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
cd AIPC_LLM_eval_supplyment
pip install -r requirements.txt
```

`requirements.txt` 包含：

| 库             | 用途                                                            |
| -------------- | --------------------------------------------------------------- |
| `transformers`   | 数据生成阶段用 `AutoTokenizer.apply_chat_template` 拼 OMC 用的 prompt；评估阶段用 tokenize 算 prompt/response token 长度 |
| `jinja2`         | `apply_chat_template` 背后的模板引擎，**transformers 当软依赖必须显式装**，少了会报 `ImportError: apply_chat_template requires jinja2` |
| `sentencepiece`  | SentencePiece 系 tokenizer（Llama/Mistral 等）必装；Qwen 走 BPE 可以不装，留着无害 |
| `pandas`         | 结果聚合，落 parquet / Excel                                    |
| `pyarrow`        | pandas 的 parquet 后端（`GPU-Results-*.parquet`）               |
| `openpyxl`       | pandas 的 Excel 后端（`GPU_Summary.xlsx`）                      |
| `openai`         | 远端 vLLM OpenAI 兼容接口（stream + `include_usage` 需 >=1.30） |
| `matplotlib`     | entropy / repeat 趋势图                                         |
| `numpy`          | matplotlib 配套                                                 |
| `tabulate`       | 多 version 合并均值后导出 markdown 表                           |

> ⚠️ `transformers` 默认会把 `torch` 拉进来。如果只是跑本目录的 tokenizer（不做训练/本地推理），可以用 CPU 版 torch 节省空间：`pip install torch --index-url https://download.pytorch.org/whl/cpu`，或在装 transformers 前先装好对应平台的 torch wheel。

> ⚠️ 装完后如果跑数据生成报 `ImportError: apply_chat_template requires jinja2`，说明 `jinja2` 没装上：`pip install jinja2` 即可。同理 SentencePiece 系模型缺包会报 `requires the SentencePiece library but it was not found`，装 `sentencepiece` 即可。

### 3. 数据集

BFCL v3 数据集由于体积较大，**没有提交到 git 仓库**（被 `.gitignore` 忽略），需要从 HuggingFace 自行下载：

- 数据集主页：<https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard>
- 上游 Gorilla 项目（含数据集说明 / 评分规则原始定义）：<https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard>

下载方式任选其一：

```bash
# 方式一：huggingface-cli
pip install -U "huggingface_hub[cli]"
huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard \
    --repo-type dataset \
    --local-dir AIPC_LLM_eval_supplyment/Data_set/Berkeley-Function-Calling-Leaderboard

# 方式二：git lfs clone
cd AIPC_LLM_eval_supplyment/Data_set
git lfs install
git clone https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
```

下载完后目录结构应该是：

```
AIPC_LLM_eval_supplyment/Data_set/Berkeley-Function-Calling-Leaderboard/
├── BFCL_v3_simple.json
├── BFCL_v3_multiple.json
├── BFCL_v3_parallel.json
├── BFCL_v3_parallel_multiple.json
├── BFCL_v3_java.json
├── BFCL_v3_javascript.json
├── BFCL_v3_live_simple.json
├── BFCL_v3_live_multiple.json
├── BFCL_v3_live_parallel.json
├── BFCL_v3_live_parallel_multiple.json
├── BFCL_v3_irrelevance.json
├── BFCL_v3_live_irrelevance.json
├── BFCL_v3_live_relevance.json
└── possible_answer/
    ├── BFCL_v3_simple.json
    └── ... （只对 AST 类别提供 ground_truth）
```

从 `1_Data_gen/` 出发的相对路径是 `../Data_set/Berkeley-Function-Calling-Leaderboard`，跟 `run_gen_bfcl.sh` 里默认的 `DATASET_ROOT` 一致。本仓库第一版只跑 13 个单轮 + 相关性类别（见顶部表），exec / rest / sql / multi_turn / chatable 这些 jsonl 即便下载下来本仓库也不会用到。

#### HumanEval 数据集

也不进 git，需自行从 HuggingFace 下载：

- 数据集主页：<https://huggingface.co/datasets/openai_humaneval>
- 上游 OpenAI human-eval：<https://github.com/openai/human-eval>

```bash
# 推荐：huggingface-cli
huggingface-cli download openai_humaneval \
    --repo-type dataset \
    --local-dir AIPC_LLM_eval_supplyment/Data_set/openai_humaneval
```

下载完后目录结构（HuggingFace 默认布局，HumanEval loader 兼容）：

```
AIPC_LLM_eval_supplyment/Data_set/openai_humaneval/
├── README.md
└── openai_humaneval/
    └── test-00000-of-00001.parquet     # 164 条样本：task_id / prompt / canonical_solution / test / entry_point
```

从 `1_Data_gen/` 出发的相对路径是 `../Data_set/openai_humaneval`，跟 `run_gen_humaneval.sh` 里默认的 `DATASET_ROOT` 一致。loader 同时兼容把 parquet 直接放在 `dataset_root/` 下的扁平布局。

### 4. 本地 tokenizer / 模型目录

放到 `Model_file/{model_name}/`，用于数据生成时 apply chat template + 评估时算 prompt/response token 长度。这里**只需要 tokenizer 相关文件**（`tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json` / `vocab` 等），权重不必下载。

### 5. model_config

参考 `model_config/Qwen3-4B-test-config.json`，里头要正确指向 `TOKENIZER_CONFIG_PATH` 等字段。

### 6. vLLM 服务

先把目标模型用 vLLM 起好（OpenAI 兼容接口），记下 `ip:port` 和 `--served-model-name`，后面 GPU 评估的 `VLLM_IP / VLLM_PORT / VLLM_MODEL_ID` 都对应这里。

## 三、阶段一：生成测试用例

### BFCL

```bash
cd 1_Data_gen
bash run_gen_bfcl.sh
```

或直接命令行：

```bash
python Gen_bfcl_cases.py \
    ../model_config/Qwen3-4B-test-config.json \
    ../Data_set/Berkeley-Function-Calling-Leaderboard \
    100                     # MAX_CASES_PER_CATEGORY
    # 可选：--categories "simple,multiple,live_simple"
```

参数说明：
- `MAX_CASES_PER_CATEGORY`：每类最多取多少条（全量约 3000；调试常用 5 或 10）
- `--categories`：只跑指定子类（逗号分隔，名字须在 13 类内）

### HumanEval

```bash
cd 1_Data_gen
bash run_gen_humaneval.sh
```

或直接命令行：

```bash
python Gen_humaneval_cases.py \
    ../model_config/Qwen3-4B-test-config.json \
    ../Data_set/openai_humaneval \
    [LIMIT]                 # 可选：前 N 条（默认 164 条全量）
```

HumanEval 的产物文件名 `project-N-HUMANEVAL-{OMC|GPU|API}-{N}.{json|jsonl}`，`testCaseName` 形如 `project-1-humaneval-test-python-{i}`（i 是 task_id 后面的数字，0..163）。`sentence.stopSeq` 在模型默认基础上额外加了 `\nclass ` / `\nif __name__` / `\ndef test_` / `\nprint(`。每条 `expect` 是 JSON 字符串，里头塞了 `{prompt, test, entry_point, task_id}` 四个字段供 GPU 评估器解出来 exec 测试。

---

产出（每跑一次自动递增 `project-N` 子目录）：

```
model-eval-storage/{model_name}/project-N/
├── project-N-BFCL-OMC-{P}.json   # OMC：严格原版 Ceval 结构，sentences[0].prompt 是 chat_template 已渲染好的单串
├── project-N-BFCL-GPU-{P}.json   # GPU：Ceval 外壳 + sentences[0].messages（OpenAI 风格 role/content 列表，直接送 vLLM）
├── project-N-BFCL-API-{P}.jsonl  # API：OpenAI 风格顶层 messages
└── api_config.json               # 推理 runtime 参数（含 chat_template）
```

三种 mode 的 sentence/顶层结构对比：

| Mode | 顶层字段                                                | sentences[0] 字段                                                                                       |
| ---- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| OMC  | `testCaseName` + Ceval 既有字段（inferType/tokenizerPath/…） | `prompt`（system+user 经 chat_template 渲染后的单串）/ `expect` + Ceval 既有采样参数。**严格遵循原版 Ceval，不允许任何额外字段**。 |
| GPU  | 同 OMC                                                  | `messages`（role/content 列表）/ `expect` + 采样参数。evaluator 直接读 `sentences[0]['messages']` 喂 vLLM。      |
| API  | OpenAI 风格：`model` / `testCaseName` / `messages` / `expect` / `stream` / `options` | —（没有 sentences 包装） |

`testCaseName` 规约统一为 `{project}-bfcl-test-{category}-{i}`，三种 mode 完全对等。**`category` / `eval_type` 不进任何字段**——evaluator 用 `verify_ans.TEST_CASE_NAME_PATTERN` 解析出 `category` 段，再用 `verify_ans.CATEGORY_TO_EVAL_TYPE` 表查到 `eval_type`（ast / relevance / irrelevance）。这套推导对 OMC/GPU/API 都成立，将来给 OMC 写评估器可以复用同一套解析。

OMC 的 `expect` 字段塞了 BFCL ground truth JSON 字符串而非 Ceval 那种正则模板——OMC 自带的 last-line regex 校验对这个 expect 不会命中，因此本目录的 OMC 用例只用作"过推理"的输入，评估走 GPU 那条路径（如果以后要做 OMC 端评估需要另写一套）。

## 四、阶段二：GPU 评估（BFCL）

编辑 `2_GPU_run_eval/run_eval_bfcl_gpu_results.sh` 顶部：

```bash
TOKENIZER_PATH="../Model_file/Qwen3-4B"
INPUT_JSON_PATH="../../model-eval-storage/Qwen3-4B/project-1/project-1-BFCL-GPU-100.json"
VLLM_IP="10.93.64.30"
VLLM_PORT=8895
VLLM_MODEL_ID="qwen3_4b"
VERSION_FLAG=1          # 跑 N 次，落 GPU-Results-1..N.parquet，并求均值
SHOW_DETAIL="true"      # "true" 出 category/breakdown sheet + 趋势图；否则只 summary
```

然后：

```bash
cd 2_GPU_run_eval
bash run_eval_bfcl_gpu_results.sh
```

或直接命令行：

```bash
python Eval_BFCL_GPU_results.py \
    ../Model_file/Qwen3-4B \
    ../../model-eval-storage/Qwen3-4B/project-1/project-1-BFCL-GPU-100.json \
    10.93.64.30 8895 qwen3_4b 1 \
    --show_detail
```

产出在 `{project_base}/GPU-BFCL-<YYYYMMDD>[-N]/` 下（同一项目同日多次跑会自动追加 `-2` / `-3`）：

```
project-N/GPU-BFCL-20260529/
├── GPU-Results-{V}.parquet     # 第 V 次的逐条原始结果
├── GPU_Summary.xlsx            # 三种 sheet：summary_v* / category_v* / breakdown_v*
├── analysis_v{V}.png           # entropy / repeat 趋势图（show_detail）
├── summary_GPU_avg.md          # 多 version 合并均值（show_detail）
├── category_GPU_avg.md
└── breakdown_GPU_avg.md
```

逐条结果字段：
`testCaseName, project_name, flavor, vertical, category, eval_type, prompt, expect, response, prediction, prompt_token_len, response_token_len, first_token_time, total_time, decode_time, correct, get_ans, repeat, entropy`

## 四.5、阶段二：GPU 评估（HumanEval）

跟 BFCL 完全独立的入口：`Eval_HumanEval_GPU_results.py` / `run_eval_humaneval_gpu_results.sh`。

编辑 `run_eval_humaneval_gpu_results.sh` 顶部：

```bash
TOKENIZER_PATH="../Model_file/Qwen3-4B"
INPUT_JSON_PATH="../../model-eval-storage/Qwen3-4B/project-1/project-1-HUMANEVAL-GPU-164.json"
VLLM_IP="10.93.64.30"
VLLM_PORT=8895
VLLM_MODEL_ID="qwen3_4b"
VERSION_FLAG=1
SHOW_DETAIL="true"          # "true" 输出 failed sheet + 趋势图
```

```bash
cd 2_GPU_run_eval
bash run_eval_humaneval_gpu_results.sh
```

或直接命令行：

```bash
python Eval_HumanEval_GPU_results.py \
    ../Model_file/Qwen3-4B \
    ../../model-eval-storage/Qwen3-4B/project-1/project-1-HUMANEVAL-GPU-164.json \
    10.93.64.30 8895 qwen3_4b 1 \
    --show_detail
```

产出在 `{project_base}/GPU-HUMANEVAL-<YYYYMMDD>[-N]/` 下（同日重跑追加 `-2` / `-3`）：

```
project-N/GPU-HUMANEVAL-20260529/
├── GPU-Results-{V}.parquet     # 逐条原始结果
├── GPU_Summary.xlsx            # summary_v*（含 Pass@1）/ failed_v*（失败用例 + stderr 尾巴）
├── analysis_v{V}.png           # entropy / repeat 趋势图（show_detail）
└── summary_GPU_avg.md          # 多 version 求均值（show_detail）
```

逐条结果的 `prediction` 字段是 JSON：`{"code": "<抽出的代码>", "error": "<失败时的 stderr 尾巴，pass 时为空>"}`，方便排查为啥某条 fail。

⚠️ **代码执行隔离**：`verify_ans_humaneval` 用 `subprocess.run([sys.executable, "-c", program], timeout=10)` 跑模型生成的代码。**这意味着代码在跟你 evaluator 同一台机器上运行**——只在可信环境 / 沙箱里跑，不要拿来评估恶意模型输出。如果担心，把 GPU 评估 host 跑在 docker / firejail / unshare 里面。

## 四'、阶段二：OMC 评估

OMC 工具跑完 `*-{BFCL|HUMANEVAL}-OMC-*.json` 后会落一份 `.txt` 日志（按 `[INFO] ## testCaseName: <name>, start/Done` 分隔每条用例）。`3_OMC_eval` 把日志解析成 DataFrame、跑对应数据集的评分，落 parquet + Excel + 趋势图。两个数据集复用同一份 `OMC_collect.py`（日志格式完全一致），只是注入的 `verify_answer` 不同。

### BFCL OMC

```bash
cd 3_OMC_eval
bash run_eval_bfcl_omc_results.sh
```

或直接命令行：

```bash
python Eval_BFCL_OMC_results.py \
    ../Model_file/Qwen3-4B \
    ../../model-eval-storage/Qwen3-4B/project-1/project-1-BFCL-OMC-100.txt \
    --show_detail
```

产出在 `{project_base}/OMC-BFCL-<YYYYMMDD>[-N]/`。

### HumanEval OMC

```bash
cd 3_OMC_eval
bash run_eval_humaneval_omc_results.sh
```

或直接命令行：

```bash
python Eval_HumanEval_OMC_results.py \
    ../Model_file/Qwen2.5-Coder-7B-Instruct \
    ../../model-eval-storage/Qwen2.5-Coder-7B-Instruct/project-1/project-1-HUMANEVAL-OMC-164.txt \
    --show_detail
```

产出在 `{project_base}/OMC-HUMANEVAL-<YYYYMMDD>[-N]/`。**注意**：HumanEval 评分会在 evaluator host 上跑模型生成的代码（subprocess + 10s 超时），不是真沙箱——只在可信环境跑。

---

OMC 端逐条结果字段：与 GPU 端基本同款，外加 OMC 日志自带的原始指标列（`inputTokenCount` / `outputTokenCount` / `decodeTimeMs` / `prefillTimeMs` / `decodeTimeMs per token` 等，便于排查）。时长口径：
- `first_token_time` = `prefillTimeMs / 1000`
- `decode_time` = `decodeTimeMs / 1000`
- `total_time` = `first_token_time + decode_time`

## 四''、阶段二：API 评估

API 客户端跑完 `*-{BFCL|HUMANEVAL}-API-*.jsonl` 会落一份每 3 行一组的 `.txt`（请求 JSON / 响应 JSON / `API_total_time: <s>`）。`4_API_eval` 按 3 行一组解析。

**响应格式：自动嗅探两种接口**

| 格式 | 嗅探规则 | 响应字段位置 | 时长字段 |
| --- | --- | --- | --- |
| **Ollama 风格**（旧） | 顶层有 `message.content`，含 `prompt_eval_duration` / `eval_duration` | `resp.message.content` | `prompt_eval_duration` (ns) / `eval_duration` (ns) |
| **OpenAI chat.completion 风格**（新） | 顶层 `object == "chat.completion"` 或有 `choices[].message.content` | `resp.choices[0].message.content` | 无 prefill / decode 分段，TTFT / decode 写 NaN，total 只有 `API_total_time` |

每条 result 多一列 `resp_format`（取值 `ollama` / `chat_completion` / `unknown`）方便事后定位，启动时也会在日志里打 `[INFO] response formats detected: ...` 汇总。token 长度优先用响应自带的 `usage.prompt_tokens` / `usage.completion_tokens`（chat.completion）或 `prompt_eval_count` / `eval_count`（Ollama），缺了再本地 tokenize 兜底。

### BFCL API

```bash
cd 4_API_eval
bash run_eval_bfcl_api_results.sh
```

```bash
python Eval_BFCL_API_results.py \
    ../Model_file/Qwen3-4B \
    ../../model-eval-storage/Qwen3-4B/project-1/project-1-BFCL-API-100.txt \
    --show_detail
```

产出在 `{project_base}/API-BFCL-<YYYYMMDD>[-N]/`。

### HumanEval API

```bash
cd 4_API_eval
bash run_eval_humaneval_api_results.sh
```

```bash
python Eval_HumanEval_API_results.py \
    ../Model_file/Qwen2.5-Coder-7B-Instruct \
    ../../model-eval-storage/Qwen2.5-Coder-7B-Instruct/project-1/project-1-HUMANEVAL-API-164.txt \
    --show_detail
```

产出在 `{project_base}/API-HUMANEVAL-<YYYYMMDD>[-N]/`。同样在 evaluator host 上跑 subprocess，注意沙箱事项。

---

时长口径（视响应格式）：
- Ollama：`first_token_time = prompt_eval_duration / 1e9`、`decode_time = eval_duration / 1e9`、`total_time = API_total_time`
- chat.completion：`first_token_time = NaN`、`decode_time = NaN`、`total_time = API_total_time`（接口没回分段时长，`eval_results._safe_tps` 会自动 fall back 到 total_time 估 TPS）

## 五、评分细则

### AST 类别
1. 模型按约定输出 `[{"name": "func", "arguments": {...}}, ...]`；
2. `verify_ans.extract_calls` 抽 JSON，兼容 ```json``` 代码块、单对象、OpenAI tool_calls 嵌套、`<think>...</think>` 标签包裹；
3. `ast_checker.match_ast` 与 ground truth 做无顺序双向匹配：
   - 函数名严格相等；
   - 每个 ground truth 声明的参数必须命中 allowed_values 之一；
   - allowed_values 里 `""` 表示该参数可以"不出现"；
   - 数值容忍 int/float 等价；字符串归一空白/大小写；list 按多重集合比对；
   - 不允许模型给 ground truth 没声明的多余参数。

### Relevance 类别
- `irrelevance` / `live_irrelevance`：模型应当 NOT 给出调用 → 没解析到调用即得分；
- `live_relevance`：模型应当至少给出一个调用 → 解析到非空调用列表即得分。

### HumanEval（Pass@1）

1. `verify_ans_humaneval.extract_code` 从响应抽 Python 代码：
   - 优先 ```` ```python ``` ```` / ```` ```py ``` ```` / 无标记 ```` ``` ``` ```` 代码块（取最长的一段）；
   - 去掉 `<think>...</think>` 思考标签；
   - **不调用 `.strip()`**——为了保留 body-only 模式下的首行缩进，否则拼回 prompt 后会变 IndentationError。
2. `assemble_program` 组装完整程序：
   - 如果抽出的代码含 `def {entry_point}(`，认为模型直接写了完整函数（可能含 import），直接拿来跑；
   - 否则视作只写了函数体，把 HumanEval 原 prompt（函数 header + docstring）拼到前面；
   - 末尾追加 `test` 字段定义的 `def check(candidate):` + 一行 `check({entry_point})`。
3. `subprocess.run([sys.executable, "-c", program], timeout=10)` 执行：`returncode == 0` 即 Pass；超时或非零返回都算 Fail，`stderr` 尾巴 800 字符塞进 prediction 字段便于排查。

## 六、常见问题

**Q: 模型不肯吐 JSON，全输出"我建议调用 ..."文字怎么办？**
A: 把 `enable_thinking` 关掉、`temperature` 调小（如 0.2），或在 `1_Data_gen/Function/bfcl.py` 的 `SYSTEM_PROMPT` 里追加 few-shot 示例。

**Q: `prediction` 全是 `[]`，AST 类别 ACC 接近 0？**
A: 大概率没遵守 JSON 格式。先 `pd.read_parquet(...)` 抓几条 `response` 字段看真实输出，再调 prompt 或采样参数。

**Q: 想跑 exec / rest / sql / multi_turn / chatable？**
A: 第一版没接。后续要加得另起一轮：`exec_*` 需要 Python 沙箱跑 ground truth 做 `exact_match`；`rest` 要真打 RapidAPI；`sql` 要 SQL AST 比对；`multi_turn` 要把官方 8 个 backend class 在本地实例化并按 path 顺序调用、检查最终状态。

**Q: 我只想跑两三个类别验证管线？**
A: 数据生成时用 `--categories "simple,live_relevance"`，或者直接把 `MAX_CASES_PER_CATEGORY` 设成 5。

**Q: HumanEval 评分时模型生成的代码会不会在 evaluator 机器上乱搞？**
A: 会。`verify_ans_humaneval` 直接 `subprocess.run([python, '-c', program])`，跑的是模型输出的真实代码。只在可信环境跑；若评估不可信模型，把 evaluator host 放进 docker / firejail / unshare 容器隔离。超时默认 10s，避免死循环卡死，但**不防恶意写文件 / 联网 / 删数据**。

**Q: HumanEval 模型不肯用 markdown fence，直接吐了个函数体？**
A: 也能 PASS。`extract_code` 兜底成原始响应；`assemble_program` 检测到不含 `def {entry_point}` 时会自动把 HumanEval 原 prompt（含 def 行 + docstring）拼到前面，再加 test 跑。

**Q: HumanEval 跑出来 Pass@1 接近 0？**
A: 先 `pd.read_parquet(...)` 抓几条 `prediction` 看 `error` 字段：常见原因是模型在代码块外又写了一堆解释 / 多包了一层 fence / 漏写 import。确认后回 `1_Data_gen/Function/humaneval.py` 的 `SYSTEM_PROMPT` 加 few-shot 或调 sampling。

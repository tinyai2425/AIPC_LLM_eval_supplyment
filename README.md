# AIPC LLM Eval

BFCL v3 工具调用（13 个单轮类）和 HumanEval（164 条）。流程：生成用例 → GPU / OMC / API 评分。采样参数在 `model_config/*.json`，`ENABLE_THINKING` 会进 GPU 请求和 OMC prompt。

依赖：`pip install -r requirements.txt`。OMC 的 BFCL 评分机还要装 **vLLM**（parser 名必须是该版本里有的）。

```
1_Data_gen/          生成 OMC / GPU / API 用例，落到 ../../model-eval-storage/{model}/project-N/
2_GPU_run_eval/      打 vLLM，出 parquet + 汇总
3_OMC_eval/          解析 OMC 日志 .txt
4_API_eval/          解析 API 日志 .txt
Data_set/            BFCL、HumanEval 原始数据
Model_file/          本地 tokenizer
```

BFCL 请求是 OpenAI chat：`messages` + `tools` + `tool_choice=auto`。GPU 用非流式返回的 `tool_calls` 评分（vLLM Hermes 流式解析会丢 arguments）。OMC 日志是原文，评分时调用本机 vLLM 的同名 parser。HumanEval 无 tools，只评生成代码。

## 启动 vLLM

`--served-model-name` 要和 GPU 脚本里的 `VLLM_MODEL_ID` 一致。`--tool-call-parser` 按模型选，OMC 评分用同一个名字。

Qwen3 Instruct（4B / 8B）：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3-4B \
  --served-model-name Qwen3-4B \
  --host 0.0.0.0 --port 8895 \
  --max-model-len 30240 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Qwen3.5：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3.5-4B \
  --served-model-name qwen3.5-4b \
  --host 0.0.0.0 --port 8895 \
  --max-model-len 30240 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

| 模型 | `--tool-call-parser` |
| --- | --- |
| Qwen3 Instruct | `hermes` |
| Qwen3.5 / Qwen3-Coder | `qwen3_xml` |

`qwen3_coder` 不要用于 Qwen3 Instruct，也不要用于 Qwen3.5 的流式评测。

## BFCL

改 `1_Data_gen/run_gen_bfcl.sh` 里的 config 和数据路径，然后：

```bash
cd 1_Data_gen && bash run_gen_bfcl.sh
```

产物：`project-N-BFCL-{OMC,GPU,API}-{N}.json`。OMC 的 prompt 由 `apply_chat_template(messages, tools=...)` 渲染。旧的「正文里写 JSON 数组 / NO_CALL」用例不能拿来配 `hermes` 评分，需要重新生成再跑推理。

GPU：改 `2_GPU_run_eval/run_eval_bfcl_gpu_results.sh` 的 tokenizer、json、IP、端口、`VLLM_MODEL_ID`。

```bash
cd 2_GPU_run_eval && bash run_eval_bfcl_gpu_results.sh
```

OMC：先在设备上跑 `*-BFCL-OMC-*.json`，得到 `.txt`。评分机安装 vLLM 后，改 `3_OMC_eval/run_eval_bfcl_omc_results.sh`：

```bash
REASONING_PARSER="qwen3"
TOOL_CALL_PARSER="hermes"    # Qwen3.5 改为 qwen3_xml
ENABLE_THINKING=true          # 与 model_config 一致；关掉则 false
```

```bash
cd 3_OMC_eval && bash run_eval_bfcl_omc_results.sh
```

API：`cd 4_API_eval && bash run_eval_bfcl_api_results.sh`，日志路径在脚本里改。

## HumanEval

```bash
cd 1_Data_gen && bash run_gen_humaneval.sh
cd 2_GPU_run_eval && bash run_eval_humaneval_gpu_results.sh
cd 3_OMC_eval && bash run_eval_humaneval_omc_results.sh
cd 4_API_eval && bash run_eval_humaneval_api_results.sh
```

HumanEval 不走 tool parser。评分会在本机执行模型生成的代码（子进程，超时 10s）。

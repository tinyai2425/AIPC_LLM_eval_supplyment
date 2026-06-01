#!/bin/bash
# GPU 侧 BFCL 评估
# INPUT_JSON_PATH 指向 1_Data_gen 生成的 *-BFCL-GPU-*.json

TOKENIZER_PATH="../Model_file/Qwen3-4B"
INPUT_JSON_PATH="../../model-eval-storage/Qwen3-4B/project-1/project-1-BFCL-GPU-100.json"
VLLM_IP="127.0.0.1"
VLLM_PORT=8895
VLLM_MODEL_ID="qwen3_4b"
VERSION_FLAG=1           # 跑 N 次：会出 GPU-Results-1.parquet ... GPU-Results-N.parquet，并自动求均值 markdown
SHOW_DETAIL="true"       # "true" 输出 category/breakdown sheet + 趋势图；其它值则只打印 summary

CMD="python Eval_BFCL_GPU_results.py \"$TOKENIZER_PATH\" \"$INPUT_JSON_PATH\" \"$VLLM_IP\" \"$VLLM_PORT\" \"$VLLM_MODEL_ID\" \"$VERSION_FLAG\""

if [ "$SHOW_DETAIL" = "true" ]; then
    CMD="$CMD --show_detail"
fi

eval $CMD

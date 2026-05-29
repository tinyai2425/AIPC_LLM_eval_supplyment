#!/bin/bash
# API 端 BFCL 评估
# TEST_RESULT_PATH 指向 API 客户端跑 *-BFCL-API-*.jsonl 后产生的 .txt 日志

TOKENIZER_PATH="../Model_file/Qwen3-4B"
TEST_RESULT_PATH="../../model-eval-storage/Qwen3-4B/project-1/project-1-BFCL-API-100.txt"
SHOW_DETAIL=true   # true / false

if [ "$SHOW_DETAIL" = true ]; then
    python Eval_BFCL_API_results.py "$TOKENIZER_PATH" "$TEST_RESULT_PATH" --show_detail
else
    python Eval_BFCL_API_results.py "$TOKENIZER_PATH" "$TEST_RESULT_PATH"
fi

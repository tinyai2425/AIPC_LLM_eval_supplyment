#!/bin/bash
# OMC 端 HumanEval 评估
# TEST_RESULT_PATH 指向 OMC 跑 *-BFCL-OMC-*.json -> *-HUMANEVAL-OMC-*.json 后产生的 .txt 日志

TOKENIZER_PATH="../Model_file/Qwen2.5-Coder-7B-Instruct"
TEST_RESULT_PATH="../../model-eval-storage/Qwen2.5-Coder-7B-Instruct/project-1/project-1-HUMANEVAL-OMC-example.txt"
SHOW_DETAIL=true   # true / false

if [ "$SHOW_DETAIL" = true ]; then
    python Eval_HumanEval_OMC_results.py "$TOKENIZER_PATH" "$TEST_RESULT_PATH" --show_detail
else
    python Eval_HumanEval_OMC_results.py "$TOKENIZER_PATH" "$TEST_RESULT_PATH"
fi

#!/bin/bash
# API 端 HumanEval 评估
# TEST_RESULT_PATH 指向 API 客户端跑 *-HUMANEVAL-API-*.jsonl 后产生的 .txt 日志
# 支持 Ollama 风格 + OpenAI chat.completion 风格响应（运行时自动嗅探）

TOKENIZER_PATH="../Model_file/Qwen2.5-Coder-7B-Instruct"
TEST_RESULT_PATH="../../model-eval-storage/Qwen2.5-Coder-7B-Instruct/project-1/project-1-HUMANEVAL-API-164.txt"
SHOW_DETAIL=true   # true / false

if [ "$SHOW_DETAIL" = true ]; then
    python Eval_HumanEval_API_results.py "$TOKENIZER_PATH" "$TEST_RESULT_PATH" --show_detail
else
    python Eval_HumanEval_API_results.py "$TOKENIZER_PATH" "$TEST_RESULT_PATH"
fi

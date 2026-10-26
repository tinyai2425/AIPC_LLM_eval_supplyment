#!/bin/bash
# OMC 端 BFCL 评估
# TEST_RESULT_PATH 指向 OMC 跑 *-BFCL-OMC-*.json 后产生的 .txt 日志
#
# REASONING_PARSER / TOOL_CALL_PARSER 与 vllm serve 的同名参数一致，
# 评分直接调用本机 vLLM 里注册的 parser，不按模型改评测代码。
#   Qwen3 Instruct : TOOL_CALL_PARSER=hermes
#   Qwen3.5        : TOOL_CALL_PARSER=qwen3_xml
# 思考关掉时把 ENABLE_THINKING 设为 false（会传 --disable-thinking）。

TOKENIZER_PATH="../Model_file/Qwen3-4B"
TEST_RESULT_PATH="../../model-eval-storage/Qwen3-4B/project-7/project-3-BFCL-OMC-100.txt"
SHOW_DETAIL=true   # true / false
REASONING_PARSER="qwen3"
TOOL_CALL_PARSER="hermes"
ENABLE_THINKING=true   # true / false

ARGS=(
    "$TOKENIZER_PATH"
    "$TEST_RESULT_PATH"
    --reasoning-parser "$REASONING_PARSER"
    --tool-call-parser "$TOOL_CALL_PARSER"
)
if [ "$ENABLE_THINKING" = true ]; then
    ARGS+=(--enable-thinking)
else
    ARGS+=(--disable-thinking)
fi
if [ "$SHOW_DETAIL" = true ]; then
    ARGS+=(--show_detail)
fi

python Eval_BFCL_OMC_results.py "${ARGS[@]}"

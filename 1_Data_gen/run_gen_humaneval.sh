#!/bin/bash
# 生成 HumanEval 代码生成测试用例（OMC/GPU/API 三套）+ api_config.json
# 全量 164 条；LIMIT 留空跑全量，调试可设小一点（如 5）

CONFIG_PATH="../model_config/Qwen3-4B-test-config.json"
DATASET_ROOT="../Data_set/openai_humaneval"
LIMIT=""   # 留空 = 164 条全量；填整数 = 前 N 条

CMD="python Gen_humaneval_cases.py \"$CONFIG_PATH\" \"$DATASET_ROOT\""
if [ -n "$LIMIT" ]; then
    CMD="$CMD $LIMIT"
fi

eval $CMD

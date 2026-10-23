#!/bin/bash
# 生成 BFCL 工具调用测试用例（OMC/GPU/API 三套）+ api_config.json
# 性能测试在 Ceval_ref 那一侧做，此处不生成 perf 用例。

CONFIG_PATH="../model_config/Qwen3-4B-test-config.json"
DATASET_ROOT="../Data_set/Berkeley-Function-Calling-Leaderboard"

# 每个类别最多取 N 条；BFCL 13 类全量约 3000，调试可设小一点（如 5）
MAX_CASES_PER_CATEGORY=100
# 不传 --categories 默认跑全部 13 类；想筛选可填 "simple,multiple,live_simple"
CATEGORIES=""

CMD="python Gen_bfcl_cases.py \"$CONFIG_PATH\" \"$DATASET_ROOT\" \"$MAX_CASES_PER_CATEGORY\""
if [ -n "$CATEGORIES" ]; then
    CMD="$CMD --categories \"$CATEGORIES\""
fi

eval $CMD

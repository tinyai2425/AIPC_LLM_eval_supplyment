# API 端 BFCL 评估
#   读取 1_Data_gen 产出的 *-BFCL-API-{P}.jsonl 在 API 上跑出来的日志（.txt），
#   解析每条用例（每 3 行：请求 / 响应 / API_total_time）+ 跑 BFCL AST 评分。
#
# 用法：
#   python Eval_BFCL_API_results.py <tokenizer_path> <test_result_path> [--show_detail]
#
# 输出目录命名：{project_base}/API-BFCL-<YYYYMMDD>[-N]/，同日重跑追加 -2 / -3。

import os
import sys

# 借用 GPU eval 的共享模块 + 复用 OMC 那侧的 file_utils
sys.path.append(os.path.abspath("Function"))
sys.path.append(os.path.abspath("../2_GPU_run_eval/Function"))
sys.path.append(os.path.abspath("../3_OMC_eval/Function"))

import API_collect
import file_utils

import verify_ans
import extend_metrics
import ast_checker  # noqa: F401  verify_ans 内部用
import extract_name
import eval_results
import anlz_cont_quality


USAGE = "Usage: python Eval_BFCL_API_results.py <tokenizer_path> <test_result_path> [--show_detail]"


def parse_args():
    if len(sys.argv) < 3:
        print(USAGE)
        sys.exit(1)
    tokenizer_path = sys.argv[1]
    test_result_path = sys.argv[2]
    show_detail = "--show_detail" in sys.argv
    return tokenizer_path, test_result_path, show_detail


if __name__ == "__main__":
    tokenizer_path, test_result_path, show_detail = parse_args()
    project_base, file_name = extract_name.parse_project_base_and_filename(test_result_path)

    api_output_dir = extract_name.make_dated_output_dir(project_base, "API", dataset="BFCL")

    API_collect.init_tokenizer(tokenizer_path)
    extend_metrics.init_tokenizer(tokenizer_path)

    API_collect.GET_answer = extend_metrics.GET_answer
    API_collect.calculate_repetition_rate = extend_metrics.calculate_repetition_rate
    API_collect.calculate_token_entropy = extend_metrics.calculate_token_entropy
    API_collect.verify_answer = verify_ans.verify_answer
    API_collect.get_eval_type = verify_ans.get_eval_type
    API_collect.TEST_CASE_NAME_PATTERN = verify_ans.TEST_CASE_NAME_PATTERN
    API_collect.iter_lines_safely = file_utils.iter_lines_safely

    print(f"[INFO] input txt    : {test_result_path}")
    print(f"[INFO] tokenizer    : {tokenizer_path}")
    print(f"[INFO] output dir   : {api_output_dir}")

    df_results = API_collect.parse_llm_api_results(test_result_path)

    if len(df_results) == 0:
        print("[ERROR] No test cases parsed; check the log format.")
        sys.exit(1)

    parquet_path = os.path.join(api_output_dir, "API-Results.parquet")
    df_results.to_parquet(parquet_path)
    print(f"[SAVE] Results saved to {parquet_path}")

    save_summary_path = (
        os.path.join(api_output_dir, "API_Summary.xlsx") if show_detail else None
    )

    df_ref_results = eval_results.evaluate_reference_results(
        df_results,
        verbal=show_detail,
        save_summary_path=save_summary_path,
        sheet_prefix="api",
    )

    if show_detail:
        fig_save_path = os.path.join(api_output_dir, "analysis.png")
        anlz_cont_quality.analyze_entropy_and_repeat(
            df_ref_results, save_fig_path=fig_save_path
        )

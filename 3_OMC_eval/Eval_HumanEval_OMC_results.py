# OMC 端 HumanEval 评估
#   读取 1_Data_gen 产出的 *-HUMANEVAL-OMC-{N}.json 在 OMC 上跑出来的日志（.txt），
#   解析每条用例的 prompt / all generation / expect / 时长指标 + 跑代码执行评分（subprocess），
#   落 parquet + Excel。
#
# 用法：
#   python Eval_HumanEval_OMC_results.py <tokenizer_path> <test_result_path> [--show_detail]
#
# 关键点：复用 BFCL 同款 OMC_collect.py（日志格式完全一致），只在驱动层注入
# verify_ans_humaneval 的 verify_answer / get_eval_type / TEST_CASE_NAME_PATTERN。
# 输出目录命名：{project_base}/OMC-HUMANEVAL-<YYYYMMDD>[-N]/，同日重跑追加 -2 / -3。

import os
import sys

# 借用 2_GPU_run_eval/Function 下的共享模块（verify_ans_humaneval / extend_metrics / extract_name / anlz_cont_quality）
sys.path.append(os.path.abspath("Function"))
sys.path.append(os.path.abspath("../2_GPU_run_eval/Function"))

import OMC_collect
import file_utils

import verify_ans_humaneval as verify_ans
import eval_results_humaneval as eval_results
import extend_metrics
import ast_checker  # noqa: F401  verify_ans_humaneval 用不到，但保留 import 防止 sys.path 漏装
import extract_name
import anlz_cont_quality


USAGE = "Usage: python Eval_HumanEval_OMC_results.py <tokenizer_path> <test_result_path> [--show_detail]"


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

    omc_output_dir = extract_name.make_dated_output_dir(project_base, "OMC", dataset="HUMANEVAL")

    OMC_collect.init_tokenizer(tokenizer_path)
    extend_metrics.init_tokenizer(tokenizer_path)

    # 把 HumanEval 的 verify_answer / get_eval_type / 正则 注入到 OMC_collect
    OMC_collect.GET_answer = extend_metrics.GET_answer
    OMC_collect.calculate_repetition_rate = extend_metrics.calculate_repetition_rate
    OMC_collect.calculate_token_entropy = extend_metrics.calculate_token_entropy
    OMC_collect.verify_answer = verify_ans.verify_answer
    OMC_collect.get_eval_type = verify_ans.get_eval_type
    OMC_collect.TEST_CASE_NAME_PATTERN = verify_ans.TEST_CASE_NAME_PATTERN
    OMC_collect.iter_lines_safely = file_utils.iter_lines_safely

    print(f"[INFO] input txt    : {test_result_path}")
    print(f"[INFO] tokenizer    : {tokenizer_path}")
    print(f"[INFO] output dir   : {omc_output_dir}")

    df_results = OMC_collect.parse_llm_test_results(test_result_path)

    if len(df_results) == 0:
        print("[ERROR] No test cases parsed; check the log format.")
        sys.exit(1)

    parquet_path = os.path.join(omc_output_dir, "OMC-Results.parquet")
    df_results.to_parquet(parquet_path)
    print(f"[SAVE] Results saved to {parquet_path}")

    save_summary_path = (
        os.path.join(omc_output_dir, "OMC_Summary.xlsx") if show_detail else None
    )

    df_ref_results = eval_results.evaluate_reference_results(
        df_results,
        verbal=show_detail,
        save_summary_path=save_summary_path,
        sheet_prefix="omc",
    )

    if show_detail:
        fig_save_path = os.path.join(omc_output_dir, "analysis.png")
        anlz_cont_quality.analyze_entropy_and_repeat(
            df_ref_results, save_fig_path=fig_save_path
        )

# OMC 端 BFCL 评估
#   读取 1_Data_gen 产出的 *-BFCL-OMC-{P}.json 在 OMC 上跑出来的日志（.txt），
#   解析每条用例的 prompt / all generation / 时长指标 + 跑 BFCL AST 评分，
#   落 parquet + Excel + 趋势图。
#
# 用法：
#   python Eval_BFCL_OMC_results.py <tokenizer_path> <test_result_path> \
#       --reasoning-parser qwen3 --tool-call-parser hermes [--show_detail]
#
# --reasoning-parser / --tool-call-parser 与 vllm serve 的同名参数一致，
# 评分时调用本机 vLLM 注册的 parser，不在本仓库里写死某一种格式。
# Qwen3 Instruct 用 hermes；Qwen3.5 / Qwen3-Coder 用 qwen3_xml。
#
# 输出目录命名：{project_base}/OMC-BFCL-<YYYYMMDD>[-N]/，同日重跑追加 -2 / -3。

import argparse
import os
import sys

# 借用 2_GPU_run_eval/Function 下的共享模块（verify_ans / extend_metrics / eval_results / ...）
sys.path.append(os.path.abspath("Function"))
sys.path.append(os.path.abspath("../2_GPU_run_eval/Function"))

import OMC_collect
import file_utils
import vllm_output_parser

import verify_ans
import extend_metrics
import ast_checker  # noqa: F401  verify_ans 内部用
import extract_name
import eval_results
import anlz_cont_quality


def parse_args():
    parser = argparse.ArgumentParser(
        description="用 vLLM 的 reasoning / tool-call parser 给 OMC BFCL 日志评分。"
    )
    parser.add_argument("tokenizer_path")
    parser.add_argument("test_result_path")
    parser.add_argument("--show_detail", action="store_true")
    parser.add_argument(
        "--reasoning-parser",
        default="",
        help="vLLM reasoning parser 名，例如 qwen3。留空则不拆思考。",
    )
    parser.add_argument(
        "--tool-call-parser",
        default="",
        help="vLLM tool-call parser 名，例如 hermes 或 qwen3_xml。",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="传给 vLLM parser 的 chat_template_kwargs.enable_thinking（默认开）。",
    )
    thinking.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="模型没开思考时用这个，避免把正文当成 reasoning。",
    )
    parser.set_defaults(enable_thinking=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer_path = args.tokenizer_path
    test_result_path = args.test_result_path
    show_detail = args.show_detail
    project_base, file_name = extract_name.parse_project_base_and_filename(test_result_path)

    omc_output_dir = extract_name.make_dated_output_dir(project_base, "OMC", dataset="BFCL")

    OMC_collect.init_tokenizer(tokenizer_path)
    extend_metrics.init_tokenizer(tokenizer_path)

    OMC_collect.GET_answer = extend_metrics.GET_answer
    OMC_collect.calculate_repetition_rate = extend_metrics.calculate_repetition_rate
    OMC_collect.calculate_token_entropy = extend_metrics.calculate_token_entropy
    OMC_collect.verify_answer = verify_ans.verify_answer
    OMC_collect.get_eval_type = verify_ans.get_eval_type
    OMC_collect.TEST_CASE_NAME_PATTERN = verify_ans.TEST_CASE_NAME_PATTERN
    OMC_collect.iter_lines_safely = file_utils.iter_lines_safely

    if args.tool_call_parser:
        parser = vllm_output_parser.VLLMGenerationParser(
            OMC_collect.tokenizer,
            reasoning_parser=args.reasoning_parser,
            tool_call_parser=args.tool_call_parser,
            enable_thinking=args.enable_thinking,
        )
        OMC_collect.init_vllm_parser(parser)
        try:
            import vllm
            vllm_version = getattr(vllm, "__version__", "?")
        except Exception:  # noqa: BLE001
            vllm_version = "?"
        print(
            "[INFO] vLLM parser : "
            f"version={vllm_version} "
            f"reasoning={args.reasoning_parser or '(none)'} "
            f"tool_call={args.tool_call_parser} "
            f"enable_thinking={args.enable_thinking}"
        )
    else:
        OMC_collect.init_vllm_parser(None)
        print(
            "[WARN] 未指定 --tool-call-parser，BFCL 会退回内置文本抽取。"
            " Qwen3 用 hermes，Qwen3.5 用 qwen3_xml，并加上 --reasoning-parser qwen3。"
        )

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

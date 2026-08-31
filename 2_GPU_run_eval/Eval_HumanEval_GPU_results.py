# GPU 侧 HumanEval 评估
#   读取 1_Data_gen 产出的 {project}-HUMANEVAL-GPU-*.json
#   起多线程打远端 vLLM，落 parquet + Excel + 趋势图 + 多 version 求均值 markdown
#
# 用法：
#   python Eval_HumanEval_GPU_results.py <tokenizer_path> <input_json_path> <ip> <port> <model_id> <version_flag> [--show_detail]
#
# 输出目录命名：{project_base}/GPU-HUMANEVAL-<YYYYMMDD>[-N]/，同日重跑追加 -2 / -3。

import os
import sys

import pandas as pd

sys.path.append(os.path.abspath("Function"))
# 注意：HumanEval 走自己的 verify_ans_humaneval / eval_results_humaneval，
# 跟 BFCL 那套（verify_ans / eval_results）完全独立。
import verify_ans_humaneval as verify_ans
import eval_results_humaneval as eval_results
import extend_metrics

import parallel_fetch_gpu
import measure_perf_gpu

import anlz_cont_quality
import write_average_sheet
import extract_name


USAGE = (
    "Usage: python Eval_HumanEval_GPU_results.py <tokenizer_path> <input_json_path> "
    "<ip> <port> <model_id> <version_flag> [--show_detail]"
)


def parse_args():
    if len(sys.argv) < 7:
        print(USAGE)
        sys.exit(1)
    return (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
        sys.argv[5],
        int(sys.argv[6]),
        "--show_detail" in sys.argv,
    )


def get_version_nums(version_flag):
    if version_flag < 1:
        raise ValueError("version_flag must be >= 1")
    return list(range(1, version_flag + 1))


if __name__ == "__main__":
    (
        tokenizer_path,
        input_json_path,
        ip,
        port,
        model_id,
        version_flag,
        show_detail,
    ) = parse_args()

    project_base, file_name = extract_name.parse_project_base_and_filename(input_json_path)
    version_nums = get_version_nums(version_flag)

    gpu_output_dir = extract_name.make_dated_output_dir(project_base, "GPU", dataset="HUMANEVAL")
    save_summary_path = os.path.join(gpu_output_dir, "GPU_Summary.xlsx")

    measure_perf_gpu.init_tokenizer(tokenizer_path)
    extend_metrics.init_tokenizer(tokenizer_path)

    # 把 HumanEval 的 verify_ans 注入到 measure_perf_gpu / parallel_fetch_gpu
    measure_perf_gpu.verify_ans = verify_ans
    measure_perf_gpu.extend_metrics = extend_metrics

    parallel_fetch_gpu.verify_ans = verify_ans
    parallel_fetch_gpu.measure_perf_gpu = measure_perf_gpu

    print("Notice: Proxy settings should be disabled before proceeding.")
    print(f"[INFO] input json    : {input_json_path}")
    print(f"[INFO] vLLM endpoint : http://{ip}:{port}/v1, model={model_id}")
    print(f"[INFO] tokenizer     : {tokenizer_path}")
    print(f"[INFO] output dir    : {gpu_output_dir}")

    df_ref_results = {}
    evaluated_results = {}

    with pd.ExcelWriter(save_summary_path) as writer:
        for version_num in version_nums:
            print(f"\n[INFO] Processing version {version_num}...")

            ref_result_parquet_path = os.path.join(
                gpu_output_dir, f"GPU-Results-{version_num}.parquet"
            )

            df_ref_results[version_num] = parallel_fetch_gpu.parallel_fetch_reference_model(
                os.path.join(project_base, file_name),
                ip,
                port,
                model_id,
            )

            df_ref_results[version_num].to_parquet(ref_result_parquet_path)
            print(f"[SAVE] Results saved to {ref_result_parquet_path}")

            answers_jsonl_path = os.path.join(
                gpu_output_dir, f"GPU-Answers-{version_num}.jsonl"
            )
            parallel_fetch_gpu.save_answers_jsonl(
                df_ref_results[version_num], answers_jsonl_path
            )
            print(f"[SAVE] Answers jsonl saved to {answers_jsonl_path}")

            ref_results = pd.read_parquet(ref_result_parquet_path)
            evaluated_results[version_num] = eval_results.evaluate_reference_results(
                ref_results,
                verbal=show_detail,
                save_summary_path=save_summary_path if show_detail else None,
                save_writer=writer if show_detail else None,
                sheet_prefix=f"v{version_num}",
            )
            print(f"[DONE] Evaluation for version {version_num}")

            if show_detail:
                fig_save_path = os.path.join(
                    gpu_output_dir, f"analysis_v{version_num}.png"
                )
                anlz_cont_quality.analyze_entropy_and_repeat(
                    evaluated_results[version_num],
                    save_fig_path=fig_save_path,
                )

    if show_detail:
        write_average_sheet.save_overall_summary(save_summary_path, "summary")

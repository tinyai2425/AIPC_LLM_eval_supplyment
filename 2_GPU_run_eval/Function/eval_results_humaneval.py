# HumanEval 没有分类聚合（164 条都是 Python 函数完成）；产出一行 summary 即可。
# 接口跟 BFCL 版 eval_results.evaluate_reference_results 同名同签，方便复用上层驱动。

import pandas as pd


def _safe_tps(df):
    decode_sum = df["decode_time"].dropna().sum()
    if decode_sum <= 0:
        total = df["total_time"].sum()
        ftt = df["first_token_time"].dropna().sum()
        decode_sum = max(1e-5, total - ftt)
    return df["response_token_len"].sum() / decode_sum


def _overall_row(df, project_name):
    correct_cnt = int(df["correct"].sum())
    return {
        "Project": project_name,
        "Dataset": "HumanEval",
        "Test No.": len(df),
        "Passed": correct_cnt,
        "Pass@1": round(df["correct"].mean() * 100, 2),
        "prompt len": round(df["prompt_token_len"].mean(), 2),
        "response len": round(df["response_token_len"].mean(), 2),
        "get_ans": round(df["get_ans"].mean(), 2),
        "rp>0.1": round((df["repeat"] > 0.1).mean(), 4),
        "rp@99%": round(df["repeat"].quantile(0.99), 5),
        "ent<14.5": round((df["entropy"] < 14.5).mean(), 4),
        "ent@1%": round(df["entropy"].quantile(0.01), 2),
        "TTFT": round(df["first_token_time"].mean(), 2),
        "TPS": round(_safe_tps(df), 2),
    }


def evaluate_reference_results(
    result_get,
    verbal=False,
    save_summary_path=None,
    save_writer=None,
    sheet_prefix="",
):
    df_results = result_get.copy()
    project_name = (
        df_results["project_name"].iloc[0]
        if "project_name" in df_results.columns and len(df_results)
        else "HumanEval"
    )

    df_summary = pd.DataFrame([_overall_row(df_results, project_name)])
    print(f"\n[Summary] {sheet_prefix}")
    print(df_summary.to_string(index=False))

    # === verbal：把所有 fail 用例的 task_id 列出来方便排查 ===
    df_failed = None
    if verbal:
        df_failed = df_results[~df_results["correct"]].copy()
        if len(df_failed):
            print(f"\n[Failed cases] {sheet_prefix}  ({len(df_failed)} 条)")
            cols = [c for c in ["testCaseName", "response_token_len", "first_token_time"] if c in df_failed.columns]
            print(df_failed[cols].to_string(index=False))

    internal_writer = None
    if save_summary_path and save_writer is None:
        internal_writer = pd.ExcelWriter(save_summary_path)
        save_writer = internal_writer

    if save_writer:
        suffix = f"_{sheet_prefix}" if sheet_prefix else ""
        summary_sheet = ("summary" + suffix)[:31]
        df_summary.to_excel(save_writer, sheet_name=summary_sheet, index=False)
        print(f"[Excel] Summary written to sheet: {summary_sheet}")

        if verbal and df_failed is not None and len(df_failed):
            failed_sheet = ("failed" + suffix)[:31]
            # 仅留几个有用列，避免 Excel 单元格太大（prediction 里塞了 stderr）
            keep_cols = [c for c in [
                "testCaseName", "response_token_len", "first_token_time",
                "total_time", "decode_time", "prediction"
            ] if c in df_failed.columns]
            df_failed[keep_cols].to_excel(save_writer, sheet_name=failed_sheet, index=False)
            print(f"[Excel] Failed cases written to sheet: {failed_sheet}")

    if internal_writer:
        internal_writer.close()

    return df_results

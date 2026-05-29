# 三层聚合：
#   - summary_{prefix}   : 整体一行
#   - category_{prefix}  : BFCL 3 大组（Non-live AST / Live AST / Relevance）
#   - breakdown_{prefix} : 13 细类

import pandas as pd


# 13 细类 -> 3 大组（与 BFCL 官方分组口径一致）
CATEGORY_TO_GROUP = {
    "simple": "Non-live AST",
    "multiple": "Non-live AST",
    "parallel": "Non-live AST",
    "parallel_multiple": "Non-live AST",
    "java": "Non-live AST",
    "javascript": "Non-live AST",
    "live_simple": "Live AST",
    "live_multiple": "Live AST",
    "live_parallel": "Live AST",
    "live_parallel_multiple": "Live AST",
    "irrelevance": "Relevance",
    "live_irrelevance": "Relevance",
    "live_relevance": "Relevance",
}


def _safe_tps(df):
    decode_sum = df["decode_time"].dropna().sum()
    if decode_sum <= 0:
        total = df["total_time"].sum()
        ftt = df["first_token_time"].dropna().sum()
        decode_sum = max(1e-5, total - ftt)
    return df["response_token_len"].sum() / decode_sum


def _overall_row(df, project_name, flavor):
    correct_cnt = int(df["correct"].sum())
    return {
        "Project": project_name,
        "Flavor": flavor,
        "Test No.": len(df),
        "Correct": correct_cnt,
        "prompt len": round(df["prompt_token_len"].mean(), 2),
        "response len": round(df["response_token_len"].mean(), 2),
        "get_ans": round(df["get_ans"].mean(), 2),
        "rp>0.1": round((df["repeat"] > 0.1).mean(), 4),
        "rp@99%": round(df["repeat"].quantile(0.99), 5),
        "ent<14.5": round((df["entropy"] < 14.5).mean(), 4),
        "ent@1%": round(df["entropy"].quantile(0.01), 2),
        "TTFT": round(df["first_token_time"].mean(), 2),
        "TPS": round(_safe_tps(df), 2),
        "ACC": round(df["correct"].mean() * 100, 2),
    }


def evaluate_reference_results(
    result_get,
    verbal=False,
    save_summary_path=None,
    save_writer=None,
    sheet_prefix="",
):
    df_results = result_get.copy()

    # 兜底：如果 measure_perf_gpu 没塞 category，就用 testCaseName 解析出的 vertical
    if "category" not in df_results.columns:
        df_results["category"] = df_results.get("vertical", "")
    df_results["group"] = df_results["category"].map(CATEGORY_TO_GROUP).fillna("Other")

    # === summary：整体一行 ===
    project_name = (
        df_results["project_name"].iloc[0]
        if "project_name" in df_results.columns and len(df_results)
        else "BFCL"
    )
    df_summary = pd.DataFrame([_overall_row(df_results, project_name, "bfcl")])
    print(f"\n[Summary] {sheet_prefix}")
    print(df_summary.to_string(index=False))

    df_category = None
    df_breakdown = None
    if verbal:
        # === 4 大组 ===
        print(f"\n[Group breakdown] {sheet_prefix}")
        cat_rows = []
        for grp, df_g in df_results.groupby("group", dropna=False):
            cat_rows.append({
                "Project": project_name,
                "Flavor": grp,
                "样本数": len(df_g),
                "Correct": int(df_g["correct"].sum()),
                "prompt len": round(df_g["prompt_token_len"].mean(), 2),
                "response_token_len": round(df_g["response_token_len"].mean(), 2),
                "TTFT": round(df_g["first_token_time"].mean(), 2),
                "TPS": round(_safe_tps(df_g), 2),
                "ACC": round(df_g["correct"].mean() * 100, 2),
            })
        df_category = pd.DataFrame(cat_rows)
        print(df_category.to_string(index=False))

        # === 13 细类 ===
        print(f"\n[Breakdown by category] {sheet_prefix}")
        details = []
        for (grp, cat), df_c in df_results.groupby(["group", "category"], dropna=False):
            details.append({
                "Project": project_name,
                "Flavor": grp,
                "Vertical": cat,
                "样本数": len(df_c),
                "Correct": int(df_c["correct"].sum()),
                "prompt len": round(df_c["prompt_token_len"].mean(), 2),
                "response_token_len": round(df_c["response_token_len"].mean(), 2),
                "TTFT": round(df_c["first_token_time"].mean(), 2),
                "TPS": round(_safe_tps(df_c), 2),
                "ACC": round(df_c["correct"].mean() * 100, 2),
            })
        df_breakdown = pd.DataFrame(details)
        print(df_breakdown.to_string(index=False))

    # 写 Excel：复用 save_writer（多 version 合并到同一文件）或自管 writer
    internal_writer = None
    if save_summary_path and save_writer is None:
        internal_writer = pd.ExcelWriter(save_summary_path)
        save_writer = internal_writer

    if save_writer:
        suffix = f"_{sheet_prefix}" if sheet_prefix else ""
        summary_sheet = ("summary" + suffix)[:31]
        category_sheet = ("category" + suffix)[:31]
        breakdown_sheet = ("breakdown" + suffix)[:31]

        df_summary.to_excel(save_writer, sheet_name=summary_sheet, index=False)
        print(f"[Excel] Summary written to sheet: {summary_sheet}")

        if verbal:
            if df_category is not None:
                df_category.to_excel(save_writer, sheet_name=category_sheet, index=False)
                print(f"[Excel] Category written to sheet: {category_sheet}")
            if df_breakdown is not None:
                df_breakdown.to_excel(save_writer, sheet_name=breakdown_sheet, index=False)
                print(f"[Excel] Breakdown written to sheet: {breakdown_sheet}")

    if internal_writer:
        internal_writer.close()

    return df_results

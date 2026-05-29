# 把以 prefix_ 开头的所有 sheet 合并求平均 -> 输出 markdown
# 与 Ceval 版同款逻辑

import os

import pandas as pd
from tabulate import tabulate


def save_overall_summary(save_summary_path, prefix):
    xls = pd.ExcelFile(save_summary_path)
    target_sheets = [s for s in xls.sheet_names if s.startswith(f"{prefix}_")]

    if not target_sheets:
        print(f"No sheet starts with '{prefix}_' found in {save_summary_path}")
        return

    print(f"Aggregating sheets: {target_sheets}")

    df_list = [pd.read_excel(xls, sheet_name=s) for s in target_sheets]
    combined = pd.concat(df_list, ignore_index=True)

    group_keys = [c for c in ["Project", "Flavor", "Vertical"] if c in combined.columns]
    numeric_cols = (
        combined.select_dtypes(include="number")
        .columns.difference(group_keys)
        .tolist()
    )

    df_avg = combined.groupby(group_keys)[numeric_cols].mean().reset_index()

    df_first = pd.read_excel(xls, sheet_name=target_sheets[0])
    for col in combined.columns:
        if col not in numeric_cols + group_keys and col in df_first.columns:
            values = df_first[[*group_keys, col]]
            df_avg = pd.merge(df_avg, values, on=group_keys, how="left")

    ordered_cols = [c for c in df_first.columns if c in df_avg.columns]
    df_avg = df_avg[ordered_cols]

    md_path = os.path.join(
        os.path.dirname(save_summary_path), f"{prefix}_GPU_avg.md"
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(tabulate(df_avg, headers="keys", tablefmt="github"))

    print(f"Markdown summary saved: {md_path}")

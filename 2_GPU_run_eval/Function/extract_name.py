import datetime
import os


def parse_project_base_and_filename(input_path):
    abs_path = os.path.abspath(input_path)
    return os.path.dirname(abs_path), os.path.basename(abs_path)


def make_dated_output_dir(project_base, mode, dataset="BFCL", date_str=None):
    """生成 <project_base>/<MODE>-<DATASET>-<YYYYMMDD>[-N] 形式的目录路径。

    同名目录已存在则追加 -2 / -3 / ...，确保跨多次评估不冲突。
    会自动 makedirs 后返回最终路径。
    """
    if date_str is None:
        date_str = datetime.date.today().strftime("%Y%m%d")

    base = f"{mode}-{dataset}-{date_str}"
    candidate = base
    n = 1
    while os.path.exists(os.path.join(project_base, candidate)):
        n += 1
        candidate = f"{base}-{n}"

    full_path = os.path.join(project_base, candidate)
    os.makedirs(full_path, exist_ok=True)
    return full_path

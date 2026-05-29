# 安全逐行读取文件（自动处理 GBK / UTF-8 等中文编码）

from typing import Iterator, TextIO

import chardet


def read_file_with_fallback(file_path: str) -> TextIO:
    with open(file_path, "rb") as f:
        raw_data = f.read(10000)
        encoding = chardet.detect(raw_data)["encoding"] or "utf-8"

    try:
        return open(file_path, "r", encoding=encoding, errors="replace")
    except UnicodeDecodeError:
        for enc in ["gb18030", "utf-8", "latin1"]:
            try:
                return open(file_path, "r", encoding=enc, errors="replace")
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法解码文件 {file_path}")


def iter_lines_safely(file_path: str) -> Iterator[str]:
    with read_file_with_fallback(file_path) as f:
        yield from f

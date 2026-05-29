# 跟 Ceval 同名同接口；保留 init_tokenizer 以便复用 measure_perf 的初始化路径。
# GET_answer 改成检查"是否给出了显式调用 JSON / NO_CALL"（BFCL 没有"答案：X"这种结尾）。

import math
import re
from collections import defaultdict
from difflib import SequenceMatcher

from transformers import AutoTokenizer


TOKENIZER_CONFIG_PATH = None
tokenizer = None


def init_tokenizer(path):
    global TOKENIZER_CONFIG_PATH, tokenizer
    TOKENIZER_CONFIG_PATH = path
    tokenizer = AutoTokenizer.from_pretrained(path)


def GET_answer(answer):
    """模型是否产出了显式的 function call JSON / NO_CALL。"""
    if not answer:
        return False
    if "```" in answer:
        return True
    if re.search(r"\[\s*\{", answer):
        return True
    if re.search(r"\bno[_\s-]?call\b", answer, re.IGNORECASE):
        return True
    return False


def split_sentences(text):
    delimiters = r"[。！？；;.!?\n\r]+"
    return [s.strip() for s in re.split(delimiters, text) if s.strip()]


def calculate_repetition_rate(text, similarity_threshold=0.85):
    sentences = split_sentences(text)
    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        return 0.0

    groups = defaultdict(list)
    for sent in sentences:
        matched = False
        for key in groups:
            if SequenceMatcher(None, key, sent).ratio() >= similarity_threshold:
                groups[key].append(sent)
                matched = True
                break
        if not matched:
            groups[sent].append(sent)

    repeated_chars = sum(
        sum(len(s) for s in group[1:])
        for group in groups.values()
        if len(group) > 1
    )
    return repeated_chars / total_chars


def calculate_token_entropy(text):
    """与 Ceval 一致：用 tokenizer 估算 token 级累积熵均值。"""
    global tokenizer
    if tokenizer is None:
        raise RuntimeError(
            "Tokenizer not initialized. Please call init_tokenizer(path) before using calculate_token_entropy."
        )

    vocab_size = tokenizer.vocab_size
    tokens = tokenizer.tokenize(text)
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0

    counts = defaultdict(int)
    entropy_sum = 0.0
    for i, token in enumerate(tokens):
        count = counts[token]
        probability = (count + 1) / (i + vocab_size)
        entropy_sum += -math.log2(probability)
        counts[token] += 1

    return entropy_sum / total_tokens

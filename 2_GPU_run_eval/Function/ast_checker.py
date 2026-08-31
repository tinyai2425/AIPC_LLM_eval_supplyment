# BFCL 风格的 AST 函数调用匹配（移植自官方 checker_simple_ast 的核心规则）
#
# ground_truth (例: simple):
#   [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
#   - allowed_values 列表里 "" 表示该参数允许"不出现"
#   - 数值容忍 int/float 等价；字符串归一空白 + 大小写；list 按多重集合比对
#   - 函数名严格相等；模型不能传 ground truth 没声明的多余参数

import math
import re


def _to_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _normalize_string(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _is_allowed_value_spec(obj):
    """GT 嵌套 object：值全是 allowed-values 列表，例如
    {'department': ['Science'], 'school': ['Bluebird High School', '']}。"""
    return (
        isinstance(obj, dict)
        and len(obj) > 0
        and all(isinstance(v, list) for v in obj.values())
    )


def _match_nested_object(allowed_spec, actual):
    if not isinstance(actual, dict):
        return False
    extra = set(actual.keys()) - set(allowed_spec.keys())
    if extra:
        return False
    for param, allowed_values in allowed_spec.items():
        present = param in actual
        if not _match_param_value(allowed_values, actual.get(param), present):
            return False
    return True


def _values_equal(allowed, actual):
    if _is_allowed_value_spec(allowed) and isinstance(actual, dict):
        return _match_nested_object(allowed, actual)

    if isinstance(allowed, bool) or isinstance(actual, bool):
        # Python 里 True == 1，所以必须先把 bool 隔离开
        return (
            isinstance(allowed, bool)
            and isinstance(actual, bool)
            and allowed == actual
        )

    if isinstance(allowed, (int, float)) or isinstance(actual, (int, float)):
        a = _to_number(allowed)
        b = _to_number(actual)
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)

    if isinstance(allowed, list) and isinstance(actual, list):
        return _list_equal(allowed, actual)

    if isinstance(allowed, dict) and isinstance(actual, dict):
        if set(allowed.keys()) != set(actual.keys()):
            return False
        return all(_values_equal(allowed[k], actual[k]) for k in allowed)

    if allowed is None and actual is None:
        return True

    if isinstance(allowed, str) and isinstance(actual, str):
        return _normalize_string(allowed) == _normalize_string(actual)

    return allowed == actual


def _list_equal(allowed_list, actual_list):
    if len(allowed_list) != len(actual_list):
        return False
    used = [False] * len(actual_list)
    for a in allowed_list:
        matched = False
        for i, b in enumerate(actual_list):
            if used[i]:
                continue
            if _values_equal(a, b):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return True


def _match_param_value(allowed_values, actual_value, present):
    for allowed in allowed_values:
        if allowed == "" and not present:
            return True
        if not present:
            continue
        if _values_equal(allowed, actual_value):
            return True
    return False


def _match_single_call(gt_call, pred_call):
    if not isinstance(pred_call, dict) or "name" not in pred_call:
        return False

    gt_func_name = next(iter(gt_call.keys()))
    pred_name = str(pred_call.get("name", "")).strip()
    if pred_name != gt_func_name:
        return False

    gt_params = gt_call[gt_func_name] or {}
    pred_args = pred_call.get("arguments", {}) or {}
    if not isinstance(pred_args, dict):
        return False

    extra = set(pred_args.keys()) - set(gt_params.keys())
    if extra:
        return False

    for param, allowed_values in gt_params.items():
        if not isinstance(allowed_values, list):
            allowed_values = [allowed_values]
        present = param in pred_args
        actual = pred_args.get(param)
        if not _match_param_value(allowed_values, actual, present):
            return False
    return True


def match_ast(ground_truth, predicted):
    """对应一组 ground truth 调用与一组预测调用做无顺序匹配。
    每条 ground truth 必须有且只有一条预测命中。"""
    if not isinstance(ground_truth, list) or not isinstance(predicted, list):
        return False
    if len(ground_truth) != len(predicted):
        return False

    used = [False] * len(predicted)
    for gt in ground_truth:
        matched = False
        for i, pred in enumerate(predicted):
            if used[i]:
                continue
            if _match_single_call(gt, pred):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return True


def is_call_list(parsed):
    """判断模型确实给出了至少一次合法调用。"""
    if not isinstance(parsed, list) or len(parsed) == 0:
        return False
    return all(
        isinstance(c, dict) and "name" in c and str(c["name"]).strip()
        for c in parsed
    )

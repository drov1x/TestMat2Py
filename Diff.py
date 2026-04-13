import os
import time
import math
from collections.abc import Iterable
import numpy as np

# ---------- 配置常量 ----------
MAX_DISPLAY_ELEMENTS = 10
LARGE_ARRAY_THRESHOLD = 100
INDENT = "  "

# ---------- 全局统计变量（在每次比较开始时重置） ----------
_stats = {
    'num_arrays': 0,
    'total_array_elements': 0,
    'num_scalars': 0,
    'num_dicts': 0,
    'num_strings': 0,
    'num_other': 0,
    'passed_items': [],      # 每个元素为 (path, type_desc, summary)
    'diff_items': [],        # 每个元素为 (path, diff_detail_string)
}

def _reset_stats():
    global _stats
    _stats = {
        'num_arrays': 0,
        'total_array_elements': 0,
        'num_scalars': 0,
        'num_dicts': 0,
        'num_strings': 0,
        'num_other': 0,
        'passed_items': [],
        'diff_items': [],
    }

# ---------- 辅助函数 ----------

def _format_value(value, max_len=80):
    """格式化单个值用于报告输出（对于大数组只显示摘要）。"""
    if isinstance(value, np.ndarray):
        shape_str = "×".join(str(d) for d in value.shape)
        dtype_str = str(value.dtype)
        if value.size <= LARGE_ARRAY_THRESHOLD:
            content = np.array2string(value, threshold=LARGE_ARRAY_THRESHOLD, edgeitems=3)
            return f"ndarray(shape=({shape_str}), dtype={dtype_str})\n{content}"
        else:
            flat = value.flatten()
            head = flat[:MAX_DISPLAY_ELEMENTS]
            tail = flat[-MAX_DISPLAY_ELEMENTS:] if MAX_DISPLAY_ELEMENTS > 0 else []
            head_str = np.array2string(head, separator=', ')
            tail_str = np.array2string(tail, separator=', ')
            return (f"ndarray(shape=({shape_str}), dtype={dtype_str}, size={value.size})\n"
                    f"  first {MAX_DISPLAY_ELEMENTS}: {head_str}\n"
                    f"  last {MAX_DISPLAY_ELEMENTS}: {tail_str}")
    elif isinstance(value, (list, tuple)):
        if len(value) <= MAX_DISPLAY_ELEMENTS * 2:
            return repr(value)
        else:
            head = value[:MAX_DISPLAY_ELEMENTS]
            tail = value[-MAX_DISPLAY_ELEMENTS:]
            return f"{type(value).__name__}[{len(value)}]: {head} ... {tail}"
    elif isinstance(value, dict):
        if len(value) <= MAX_DISPLAY_ELEMENTS * 2:
            return repr(value)
        else:
            keys = list(value.keys())
            head_keys = keys[:MAX_DISPLAY_ELEMENTS]
            tail_keys = keys[-MAX_DISPLAY_ELEMENTS:]
            return f"dict[{len(value)}] keys: {head_keys} ... {tail_keys}"
    else:
        return repr(value)

def _almost_equal(a, b, tol):
    """比较两个标量是否在容差内相等（支持复数、数字、字符串等）。"""
    if a is None and b is None:
        return True, None
    if a is None or b is None:
        return False, None

    type_a, type_b = type(a), type(b)

    if isinstance(a, str) and isinstance(b, str):
        return a == b, None

    if isinstance(a, complex) and isinstance(b, complex):
        diff_real = abs(a.real - b.real)
        diff_imag = abs(a.imag - b.imag)
        if diff_real <= tol and diff_imag <= tol:
            return True, None
        else:
            return False, f"diff_real={diff_real:.6g}, diff_imag={diff_imag:.6g}"

    try:
        a_num = float(a)
        b_num = float(b)
        diff = abs(a_num - b_num)
        if diff <= tol:
            return True, None
        else:
            return False, f"diff={diff:.6g}"
    except (TypeError, ValueError):
        pass

    return a == b, None

def _get_type_desc(obj):
    """返回对象的类型描述字符串，用于报告中的类型列。"""
    if isinstance(obj, np.ndarray):
        return "Numeric Array" if np.issubdtype(obj.dtype, np.number) else "Array"
    elif isinstance(obj, (list, tuple)):
        return "List/Tuple"
    elif isinstance(obj, dict):
        return "Dict"
    elif isinstance(obj, str):
        return "String"
    elif isinstance(obj, (int, float, complex)):
        return "Scalar"
    elif obj is None:
        return "None"
    else:
        return type(obj).__name__

def _get_data_summary(obj):
    """生成对象的数据概要字符串。"""
    if isinstance(obj, np.ndarray):
        shape_str = "×".join(str(d) for d in obj.shape)
        return f"shape: ({shape_str}), size: {obj.size}"
    elif isinstance(obj, (list, tuple)):
        return f"length: {len(obj)}"
    elif isinstance(obj, dict):
        return f"keys: {len(obj)}"
    elif isinstance(obj, (int, float, complex, str)):
        return repr(obj)
    elif obj is None:
        return "None"
    else:
        return repr(obj)[:50]

def _compare_arrays(arr1, arr2, tol, path, report_equal):
    """
    比较两个 numpy 数组，处理维度不匹配、复数、大数组统计等。
    返回一个字符串报告片段（差异详情），同时更新全局统计和通过项列表。
    """
    global _stats
    _stats['num_arrays'] += 1
    _stats['total_array_elements'] += arr1.size + arr2.size  # 粗略统计

    report_lines = []
    shape1 = arr1.shape
    shape2 = arr2.shape

    arr1_sq = np.squeeze(arr1)
    arr2_sq = np.squeeze(arr2)
    shape1_sq = arr1_sq.shape
    shape2_sq = arr2_sq.shape

    dimension_mismatch = (shape1 != shape2)
    squeezable_match = (shape1_sq == shape2_sq) and (shape1 != shape2)

    if dimension_mismatch:
        if squeezable_match:
            report_lines.append(f"{path}: 维度不匹配 (原始 {shape1} vs {shape2})，但 squeeze 后形状一致 ({shape1_sq})，降维比较。")
            arr1 = arr1_sq
            arr2 = arr2_sq
        else:
            diff_detail = f"{path}: 维度不匹配且无法降维匹配 (形状 {shape1} vs {shape2})，跳过详细比较。"
            _stats['diff_items'].append((path, diff_detail))
            return diff_detail

    is_complex = np.iscomplexobj(arr1) or np.iscomplexobj(arr2)

    if not np.issubdtype(arr1.dtype, np.number) or not np.issubdtype(arr2.dtype, np.number):
        if np.array_equal(arr1, arr2):
            if report_equal:
                type_desc = _get_type_desc(arr1)
                summary = _get_data_summary(arr1)
                _stats['passed_items'].append((path, type_desc, summary))
            return ""
        else:
            diff_detail = f"{path}: 数组不相等（包含非数值元素，无法进行容差比较）。"
            _stats['diff_items'].append((path, diff_detail))
            return diff_detail

    total_elements = arr1.size
    passed = False
    diff_stats = ""

    if is_complex:
        real1 = arr1.real
        real2 = arr2.real
        imag1 = arr1.imag
        imag2 = arr2.imag

        diff_real = np.abs(real1 - real2)
        diff_imag = np.abs(imag1 - imag2)

        out_of_tol = (diff_real > tol) | (diff_imag > tol)
        num_diff_elements = np.sum(out_of_tol)

        if num_diff_elements == 0:
            passed = True
            if report_equal:
                type_desc = _get_type_desc(arr1)
                summary = _get_data_summary(arr1)
                _stats['passed_items'].append((path, type_desc, summary))
        else:
            max_diff_real = np.max(diff_real)
            max_diff_imag = np.max(diff_imag)
            mean_diff_real = np.mean(diff_real)
            mean_diff_imag = np.mean(diff_imag)

            diff_lines = []
            diff_lines.append(f"{path}: 复数数组形状 {arr1.shape}，共 {total_elements} 个元素，{num_diff_elements} 个超出容差 ({tol})。")
            diff_lines.append(f"{path}:   实部最大差异: {max_diff_real:.6g}，平均: {mean_diff_real:.6g}")
            diff_lines.append(f"{path}:   虚部最大差异: {max_diff_imag:.6g}，平均: {mean_diff_imag:.6g}")

            diff_magnitude = np.sqrt(diff_real**2 + diff_imag**2)
            flat_indices = np.argsort(diff_magnitude.flatten())[-5:][::-1]
            unravel_indices = np.unravel_index(flat_indices, diff_magnitude.shape)
            diff_lines.append(f"{path}:   差异最大的位置示例：")
            for idx_tuple in zip(*unravel_indices):
                pos = tuple(i.item() if hasattr(i, 'item') else i for i in idx_tuple)
                val1 = arr1[pos]
                val2 = arr2[pos]
                d_mag = diff_magnitude[pos]
                diff_lines.append(f"{path}:     位置 {pos}: {val1} vs {val2} (|diff|={d_mag:.6g})")
            diff_stats = "\n".join(diff_lines)
            _stats['diff_items'].append((path, diff_stats))
    else:
        try:
            arr1_f = arr1.astype(float, copy=False)
            arr2_f = arr2.astype(float, copy=False)
        except (ValueError, TypeError):
            if np.array_equal(arr1, arr2):
                passed = True
                if report_equal:
                    type_desc = _get_type_desc(arr1)
                    summary = _get_data_summary(arr1)
                    _stats['passed_items'].append((path, type_desc, summary))
            else:
                diff_detail = f"{path}: 数组不相等（包含无法转换为浮点数的元素）。"
                _stats['diff_items'].append((path, diff_detail))
                return diff_detail
        else:
            diff = np.abs(arr1_f - arr2_f)
            num_diff_elements = np.sum(diff > tol)

            if num_diff_elements == 0:
                passed = True
                if report_equal:
                    type_desc = _get_type_desc(arr1)
                    summary = _get_data_summary(arr1)
                    _stats['passed_items'].append((path, type_desc, summary))
            else:
                max_diff = np.max(diff)
                min_diff = np.min(diff)
                mean_diff = np.mean(diff)

                diff_lines = []
                diff_lines.append(f"{path}: 数组形状 {arr1.shape}，共 {total_elements} 个元素，其中 {num_diff_elements} 个超出容差 ({tol})。")
                diff_lines.append(f"{path}:   最大差异: {max_diff:.6g}，最小差异: {min_diff:.6g}，平均差异: {mean_diff:.6g}")

                flat_diff = diff.flatten()
                flat_indices = np.argsort(flat_diff)[-5:][::-1]
                unravel_indices = np.unravel_index(flat_indices, diff.shape)
                diff_lines.append(f"{path}:   差异最大的位置示例：")
                for idx_tuple in zip(*unravel_indices):
                    pos = tuple(i.item() if hasattr(i, 'item') else i for i in idx_tuple)
                    val1 = arr1_f[pos]
                    val2 = arr2_f[pos]
                    d = diff[pos]
                    diff_lines.append(f"{path}:     位置 {pos}: {val1:.6g} vs {val2:.6g} (diff={d:.6g})")
                diff_stats = "\n".join(diff_lines)
                _stats['diff_items'].append((path, diff_stats))

    return diff_stats

def _compare_recursive(orig, new, tol, path, report_equal):
    """
    递归比较两个对象，更新全局统计和通过/差异列表。
    """
    global _stats

    # 处理 None
    if orig is None and new is None:
        if report_equal:
            _stats['passed_items'].append((path, "None", "None"))
        return
    if orig is None or new is None:
        diff_detail = f"{path}: 一方为 None (orig={orig}, new={new})"
        _stats['diff_items'].append((path, diff_detail))
        return

    type_orig = type(orig)
    type_new = type(new)

    # 更新类型统计（粗略）
    if isinstance(orig, dict):
        _stats['num_dicts'] += 1
    elif isinstance(orig, str):
        _stats['num_strings'] += 1
    elif isinstance(orig, (int, float, complex)):
        _stats['num_scalars'] += 1
    elif not isinstance(orig, np.ndarray):
        _stats['num_other'] += 1

    # 处理 numpy 数组
    if isinstance(orig, np.ndarray) and isinstance(new, np.ndarray):
        diff_report = _compare_arrays(orig, new, tol, path, report_equal)
        return

    # 类型不一致
    if isinstance(orig, np.ndarray) or isinstance(new, np.ndarray):
        diff_detail = f"{path}: 类型不匹配 (orig: {type_orig.__name__}, new: {type_new.__name__})"
        _stats['diff_items'].append((path, diff_detail))
        return

    # 处理字典
    if isinstance(orig, dict) and isinstance(new, dict):
        keys_orig = set(orig.keys())
        keys_new = set(new.keys())
        only_orig = keys_orig - keys_new
        only_new = keys_new - keys_orig
        common_keys = keys_orig & keys_new

        for k in only_orig:
            diff_detail = f"{path}['{k}']: 仅在原始数据中存在"
            _stats['diff_items'].append((f"{path}['{k}']", diff_detail))
        for k in only_new:
            diff_detail = f"{path}['{k}']: 仅在新数据中存在"
            _stats['diff_items'].append((f"{path}['{k}']", diff_detail))

        for k in sorted(common_keys):
            _compare_recursive(orig[k], new[k], tol, f"{path}['{k}']", report_equal)
        return

    # 处理列表、元组等序列
    if isinstance(orig, (list, tuple)) and isinstance(new, (list, tuple)):
        len_orig = len(orig)
        len_new = len(new)
        min_len = min(len_orig, len_new)

        if len_orig != len_new:
            diff_detail = f"{path}: 长度不一致 (orig={len_orig}, new={len_new})"
            _stats['diff_items'].append((path, diff_detail))

        for i in range(min_len):
            _compare_recursive(orig[i], new[i], tol, f"{path}[{i}]", report_equal)

        if len_orig > len_new:
            for i in range(len_new, len_orig):
                diff_detail = f"{path}[{i}]: 仅在原始数据中存在: {_format_value(orig[i])}"
                _stats['diff_items'].append((f"{path}[{i}]", diff_detail))
        elif len_new > len_orig:
            for i in range(len_orig, len_new):
                diff_detail = f"{path}[{i}]: 仅在新数据中存在: {_format_value(new[i])}"
                _stats['diff_items'].append((f"{path}[{i}]", diff_detail))
        return

    # 处理其他可迭代对象（排除字符串）
    if isinstance(orig, Iterable) and isinstance(new, Iterable) and not isinstance(orig, (str, bytes)):
        try:
            list_orig = list(orig)
            list_new = list(new)
            _compare_recursive(list_orig, list_new, tol, f"{path}(converted_to_list)", report_equal)
        except:
            diff_detail = f"{path}: 无法比较的可迭代类型 ({type_orig.__name__} vs {type_new.__name__})"
            _stats['diff_items'].append((path, diff_detail))
        return

    # 标量比较
    if type_orig != type_new:
        try:
            a_num = float(orig)
            b_num = float(new)
            diff = abs(a_num - b_num)
            if diff > tol:
                diff_detail = f"{path}: 类型不同但数值不等 ({type_orig.__name__} vs {type_new.__name__}) diff={diff:.6g}"
                _stats['diff_items'].append((path, diff_detail))
            elif report_equal:
                type_desc = f"{type_orig.__name__}/{type_new.__name__}"
                summary = f"{orig} ≈ {new}"
                _stats['passed_items'].append((path, type_desc, summary))
        except (TypeError, ValueError):
            diff_detail = f"{path}: 类型不同且无法转换为数值比较 ({type_orig.__name__} vs {type_new.__name__})"
            _stats['diff_items'].append((path, diff_detail))
        return

    # 同类型标量
    equal, extra_info = _almost_equal(orig, new, tol)
    if not equal:
        info_str = f" ({extra_info})" if extra_info else ""
        diff_detail = f"{path}: 值不相等: {_format_value(orig)} vs {_format_value(new)}{info_str}"
        _stats['diff_items'].append((path, diff_detail))
    elif report_equal:
        type_desc = _get_type_desc(orig)
        summary = _get_data_summary(orig)
        _stats['passed_items'].append((path, type_desc, summary))

def _generate_formatted_report(tolerance, report_equal):
    """根据全局统计生成类似用户示例的结构化报告。"""
    lines = []
    sep = "=" * 70
    subsep = "-" * 70

    lines.append(sep)
    lines.append(f"🔍 深度差异对齐报告  [容差设置: {tolerance:g}]")
    lines.append(sep)

    # 概述区
    lines.append("📊 已完成深度遍历与匹配 (在容差内安全通过) :")
    lines.append(f"   ▶ 数值矩阵/数组: {_stats['num_arrays']} 个 (包含元素总数: {_stats['total_array_elements']})")
    lines.append(f"   ▶ 数值标量节点: {_stats['num_scalars']} 个")
    lines.append(f"   ▶ 字典嵌套对象: {_stats['num_dicts']} 个")
    lines.append(f"   ▶ 文本/字符串: {_stats['num_strings']} 个")
    lines.append(subsep)

    num_passed = len(_stats['passed_items'])
    num_diffs = len(_stats['diff_items'])

    if num_diffs == 0:
        lines.append("✅ 测试通过：核心数值、结构和实质计算结果完全匹配合规！")
        if num_passed > 0:
            lines.append(f"   (注: 共确认 {num_passed} 处关键节点一致)")
    else:
        lines.append(f"⚠️ 测试存在差异：共发现 {num_diffs} 处不匹配项。")
        if num_passed > 0:
            lines.append(f"   (另有 {num_passed} 处节点在容差内通过)")

    lines.append(sep)

    # 通过项列表（当 report_equal=True 或用户总是希望显示通过项时）
    if report_equal and num_passed > 0:
        lines.append("✅ 下列核心维度/数据节点已确认一致 (容差内):")
        lines.append(subsep)
        for idx, (path, type_desc, summary) in enumerate(_stats['passed_items'], 1):
            lines.append(f"   [{idx}] 成功路径：{path} ")
            lines.append(f"       ▶ 类型: {type_desc} | 数据概要: {summary}")
        lines.append(sep)

    # 差异项详情
    if num_diffs > 0:
        lines.append("❌ 差异详情列表:")
        lines.append(subsep)
        for idx, (path, detail) in enumerate(_stats['diff_items'], 1):
            lines.append(f"[差异 {idx}] 路径: {path}")
            lines.append(detail)
            lines.append(subsep)
        lines.append(sep)

    return "\n".join(lines)

def getComparisonReport(originalOutput, newOutput, tolerance, report_equal):
    """生成比较报告字符串（新版格式）。"""
    _reset_stats()
    if originalOutput is None and newOutput is None:
        return "两者均为 None，无差异。"
    _compare_recursive(originalOutput, newOutput, tolerance, "root", report_equal)
    return _generate_formatted_report(tolerance, report_equal)

def saveComparisonResult(originalOutput, newOutput, filepath, tolerance, report_equal):
    """保存比较报告到文件。"""
    report = getComparisonReport(originalOutput, newOutput, tolerance, report_equal)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
        f.write("\n")

def TestDiff(outputName='default', originalOutput=None, newOutput=None, tolerance=0.0, report_equal=True):
    """
    主接口函数：比较两个数据并生成报告文件。

    参数：
        outputName (str): 报告文件前缀名。
        originalOutput: 待比较的第一个数据。
        newOutput: 待比较的第二个数据。
        tolerance (float): 数值比较的绝对容差。
        report_equal (bool): 是否在报告中列出所有通过项。默认为 True（生成详细对齐报告）。
    """
    if originalOutput is None:
        originalOutput = [1]
    if newOutput is None:
        newOutput = [1]

    reportText = getComparisonReport(originalOutput, newOutput, tolerance, report_equal)
    print(reportText)

    savedir = "reports"
    os.makedirs(savedir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{outputName}_{timestamp}_reports.txt"
    filepath = os.path.join(savedir, filename)

    saveComparisonResult(originalOutput, newOutput, filepath, tolerance, report_equal)

    print(f"\n报告已保存至: {filepath}")
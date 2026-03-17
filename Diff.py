import difflib
import json
import numpy as np  # 新增导入
from typing import Any, Dict, List, Tuple, Optional

def compareData(originalData: Any, newData: Any, tolerance: float = 0.0) -> Dict[str, Any]:
    """
    比较两个数据，返回比对结果
    
    Args:
        originalData: 原始数据
        newData: 新数据
        tolerance: 数值容差（用于浮点数比较）
    
    Returns:
        比对结果字典
    """
    result = {
        'isEqual': False,
        'differences': [],
        'summary': {
            'totalItems': 0,
            'matchingItems': 0,
            'differentItems': 0
        }
    }
    
    def _compare(value1: Any, value2: Any, path: str = '') -> List[Dict]:
        """递归比较函数"""
        differences = []
        
        # 类型检查
        if type(value1) != type(value2):
            differences.append({
                'path': path,
                'type': 'type_mismatch',
                'original': f"type: {type(value1).__name__}",
                'new': f"type: {type(value2).__name__}",
                'message': f"类型不匹配: {type(value1).__name__} vs {type(value2).__name__}"
            })
            return differences
        
        # ---------- 新增对 numpy.ndarray 的处理 ----------
        if isinstance(value1, np.ndarray):
            # 将 numpy 数组转换为列表后再递归比较，这样可以利用已有的列表比较逻辑
            return _compare(value1.tolist(), value2.tolist(), path)
        # ------------------------------------------------
        
        # 根据类型比较
        if isinstance(value1, dict):
            allKeys = set(value1.keys()) | set(value2.keys())
            for key in allKeys:
                newPath = f"{path}.{key}" if path else str(key)
                if key in value1 and key in value2:
                    differences.extend(_compare(value1[key], value2[key], newPath))
                elif key in value1:
                    differences.append({
                        'path': newPath,
                        'type': 'missing_key',
                        'original': value1[key],
                        'new': 'KEY_MISSING',
                        'message': f"新数据中缺少键: {key}"
                    })
                else:
                    differences.append({
                        'path': newPath,
                        'type': 'extra_key',
                        'original': 'KEY_MISSING',
                        'new': value2[key],
                        'message': f"原始数据中缺少键: {key}"
                    })
        
        elif isinstance(value1, list):
            len1, len2 = len(value1), len(value2)
            # 记录长度不匹配
            if len1 != len2:
                differences.append({
                    'path': path,
                    'type': 'length_mismatch',
                    'original': f"长度: {len1}",
                    'new': f"长度: {len2}",
                    'message': f"列表长度不匹配: {len1} vs {len2}"
                })
            
            # 比较公共索引部分
            for i in range(min(len1, len2)):
                newPath = f"{path}[{i}]"
                differences.extend(_compare(value1[i], value2[i], newPath))
            
            # 处理多余/缺失的元素
            if len1 < len2:
                for i in range(len1, len2):
                    newPath = f"{path}[{i}]"
                    differences.append({
                        'path': newPath,
                        'type': 'extra_element',
                        'original': 'INDEX_MISSING',
                        'new': value2[i],
                        'message': f"新数据中有额外元素，索引 {i}"
                    })
            elif len1 > len2:
                for i in range(len2, len1):
                    newPath = f"{path}[{i}]"
                    differences.append({
                        'path': newPath,
                        'type': 'missing_element',
                        'original': value1[i],
                        'new': 'INDEX_MISSING',
                        'message': f"新数据中缺少元素，索引 {i}"
                    })
        
        elif isinstance(value1, (int, float)):
            if abs(value1 - value2) > tolerance:
                differences.append({
                    'path': path,
                    'type': 'int_float_mismatch',
                    'original': value1,
                    'new': value2,
                    'message': f"数值不匹配: {value1} vs {value2} : {abs(value1 - value2)} > 容差 {tolerance}"
                })
        
        elif isinstance(value1, str):
            if value1 != value2:
                differences.append({
                    'path': path,
                    'type': 'string_mismatch',
                    'original': value1,
                    'new': value2,
                    'message': f"字符串不匹配"
                })
                
                # 可选：添加字符串差异详情
                if len(value1) > 0 and len(value2) > 0:
                    diffResult = getStringDifference(value1, value2)
                    differences[-1]['diff'] = diffResult
        
        else:  # 其他类型（bool, None等）
            if value1 != value2:
                differences.append({
                    'path': path,
                    'type': 'bool_none_mismatch',
                    'original': value1,
                    'new': value2,
                    'message': f"值不匹配: {value1} vs {value2}"
                })
        
        return differences
    
    # 执行比较
    differences = _compare(originalData, newData)
    
    # 统计结果
    result['isEqual'] = len(differences) == 0
    result['differences'] = differences
    
    # 如果数据是列表或字典，计算总数（此处简化，可按需调整）
    if isinstance(originalData, (dict, list)):
        if isinstance(originalData, dict):
            result['summary']['totalItems'] = len(originalData)
        elif isinstance(originalData, list):
            result['summary']['totalItems'] = len(originalData)
        
        # 这里简单统计差异项数量，实际可更精细
        result['summary']['matchingItems'] = result['summary']['totalItems'] - len(differences)
        result['summary']['differentItems'] = len(differences)
    
    return result

def getStringDifference(str1: str, str2: str) -> Dict[str, Any]:
    """
    获取两个字符串的差异详情
    """
    matcher = difflib.SequenceMatcher(None, str1, str2)
    
    # 找到差异块
    diffBlocks = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            diffBlocks.append({
                'type': tag,  # replace, delete, insert
                'original': str1[i1:i2] if tag in ('replace', 'delete') else '',
                'new': str2[j1:j2] if tag in ('replace', 'insert') else '',
                'position': {'start': i1, 'end': i2} if tag in ('replace', 'delete') else None
            })
    
    return {
        'similarity': matcher.ratio(),
        'diffBlocks': diffBlocks,
        'totalDiff': abs(len(str1) - len(str2))
    }

def getComparisonReport(originalData: Any, newData: Any, tolerance: float = 0.0) -> str:
    """
    生成详细的比对报告
    """
    comparisonResult = compareData(originalData, newData, tolerance)
    
    reportLines = [
        "=" * 60,
        "数据比对报告",
        "=" * 60
    ]
    
    if comparisonResult['isEqual']:
        reportLines.append("✅ 数据完全一致")
    else:
        reportLines.append("❌ 数据存在差异")
        
        # 添加统计信息
        if comparisonResult['summary']['totalItems'] > 0:
            reportLines.append(f"\n📊 统计信息:")
            reportLines.append(f"   总项数: {comparisonResult['summary']['totalItems']}")
            reportLines.append(f"   一致项: {comparisonResult['summary']['matchingItems']}")
            reportLines.append(f"   差异项: {comparisonResult['summary']['differentItems']}")
        
        # 添加差异详情
        if comparisonResult['differences']:
            reportLines.append(f"\n🔍 差异详情 ({len(comparisonResult['differences'])} 处):")
            for i, diff in enumerate(comparisonResult['differences'], 1):
                reportLines.append(f"\n{i}. 路径: {diff['path']}")
                reportLines.append(f"   类型: {diff['type']}")
                # 显示值，注意处理特殊标记
                original_val = diff['original'] if diff['original'] not in ('INDEX_MISSING', 'KEY_MISSING') else '（缺失）'
                new_val = diff['new'] if diff['new'] not in ('INDEX_MISSING', 'KEY_MISSING') else '（缺失）'
                reportLines.append(f"   原始: {original_val}")
                reportLines.append(f"   新的: {new_val}")
                reportLines.append(f"   信息: {diff['message']}")
                
                # 如果是字符串差异，添加更多详情
                if 'diff' in diff:
                    diffInfo = diff['diff']
                    reportLines.append(f"   相似度: {diffInfo['similarity']:.2%}")
                    if diffInfo['diffBlocks']:
                        for block in diffInfo['diffBlocks']:
                            if block['type'] == 'replace':
                                reportLines.append(f"   ↪ 替换: '{block['original']}' -> '{block['new']}'")
                            elif block['type'] == 'delete':
                                reportLines.append(f"   ✂ 删除: '{block['original']}'")
                            elif block['type'] == 'insert':
                                reportLines.append(f"   ➕ 插入: '{block['new']}'")
    
    reportLines.append("\n" + "=" * 60)
    
    return "\n".join(reportLines)

# ---------- 自定义 JSON 编码器，处理 NumPy 数据类型 ----------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
# ---------------------------------------------------------

def saveComparisonResult(originalData: Any, newData: Any, outputPath: str, tolerance: float = 0.0):
    """
    保存比对结果到文件
    """
    comparisonResult = compareData(originalData, newData, tolerance)
    report = getComparisonReport(originalData, newData, tolerance)
    
    result = {
        'comparison': comparisonResult,
        'report': report,
        'originalData': originalData,
        'newData': newData
    }
    
    with open(outputPath, 'w', encoding='utf-8') as f:
        if outputPath.endswith('.json'):
            # 使用自定义编码器以支持 NumPy 类型
            json.dump(result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        else:
            f.write(report)
    
    print(f"比对结果已保存到: {outputPath}")
    return result

def TestDiff(outputName = 'comparison_result.json', originalOutput = [1], newOutput = [1], tolerance=0.0):
    """
    外部调用入口：比较两个列表（或其他数据）并输出报告
    """
    result = compareData(originalOutput, newOutput, tolerance)
    print(f"数据是否一致: {result['isEqual']}")
    
    report = getComparisonReport(originalOutput, newOutput, tolerance)
    print(report)

    # 处理文件名
    outputpath = outputName + "_comparison.json"
    saveComparisonResult(originalOutput, newOutput, outputpath, tolerance)

# 使用示例（略，保持原样）...
if __name__ == "__main__":
    # ... 原有示例代码不变
    pass
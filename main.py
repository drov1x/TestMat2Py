"""
TestMat2Py/main.py
测试脚本 TestMat2Py 的主程序

函数组成：
    IsNumericScalar & To* :     将scipy.io.loadmat获取到的原始数据的数值类型逐一转换成所需的数值类型
                                其中：
                                    数值统一转换成int or float
                                    int和float类型的数组 统一转换成np.ndarray
                                    str类型的数组统一转换成字符串
                                    结构体统一转换成字典，保留键名
    ConvertValue :              供 ToDict 递归用
    DataLoader :                数据处理的主要函数，功能如下：
                                    1. 去除 scipy.io.loadmat 元数据中的 "__header__"、"__version__"、"__globals__" 字段
                                    2. 对每个数据自动判断类型，调用对应的函数
                                    3. 整合成字典返回
    Test :                      自动填入测试参数，调用待测试程序，捕获输出

外部调用：
    scipy.io.loadmat :          读取MATLAB .mat文件，获取输入输出数据
    Diff :                      位于 TestMat2Py/Diff.py ，对比数据用

主程序流程 (line 325) ：

    1. 获取测试参数
        第一个 if 如果是 1 则表示希望从 CLI 输入数据，如果为 0 则代表用户自己在 py 文件内填入数据

        参数详细：
            FilePath : 待测试代码文件路径 (py)
            InputFilePath : 输入数据路径 (mat)
            OutputFilePath : 输出数据路径 (mat)
                注：以上路径支持绝对路径和相对路径，支持反斜杠
            tolerance : 数据对比时支持的容差
            names : 参数具体名字，包括输入和输出
            id :
                当 id[0] = True 时表示待用户手动确定每个矩阵是否降维， False 则使用默认值
                后面按顺序决定每个矩阵是否降维

    2. 读取原始数据并对参数重命名，如果报错会输出 scipy.io.loadmat 获得的原始数据
    3. 对数据进行转化
    4. 调用待测试程序
        如果目标程序报错，会输出完整的错误信息
    5. 调用 Diff.py 对比，输出对比报告 （如果上述步骤出现错误导致程序没有输出则不对比）
    6. 测试结束，提供选项：
        y : 重新测试
        n : 退出程序
        i : 输出输入数据
        o : 输出程序输出数据
        s : 输出原始输出数据
        b : 重置降维选项，重新测试

参数获取：
    在 CLI 获取参数的时候，直接敲回车会返回默认值
"""

from scipy.io import loadmat
from Diff import TestDiff
import importlib.util
import traceback
import inspect
import numpy as np
import sys
import os

from twisted.python.filepath import FilePath


def IsNumericScalar(value):
    """
        判断值是否为NumPy数值标量类型。
    """
    return isinstance(value, np.generic) and np.issubdtype(type(value), np.number)


def ToNumber(value):
    """
        将NumPy标量转换为Python原生数值。
    """
    return value.item()


def ToNdArray(value, id = True):
    """
        将输入值转换为NumPy数组，向量转一维，矩阵保留二维。
    """
    arr = np.asarray(value)
    # loadmat通常将MATLAB向量读成(1, n)或(n, 1)，这里还原为一维数组。
    if id and arr.ndim == 2 and 1 in arr.shape:
        return arr.reshape(-1)
    return arr


def ToString(value):
    """
        将字符串相关输入统一转换为Python字符串。
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", errors="ignore")
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            if value.ndim == 0:
                return str(value.item())
            if value.ndim == 1:
                return "".join(map(str, value.tolist()))
            # MATLAB字符矩阵按行组成字符串，多行使用换行拼接。
            rows = ["".join(map(str, row.tolist())) for row in value]
            return "\n".join(rows)
        if value.dtype.kind in {"u", "i"} and value.ndim <= 2:
            flat = value.reshape(-1)
            if flat.size and np.all((flat >= 0) & (flat <= 0x10FFFF)):
                return "".join(chr(int(c)) for c in flat)
    return value


def IsBoolScalar(value):
    """
        判断值是否为布尔标量。
    """
    return isinstance(value, (bool, np.bool_))


def ToBool(value):
    """
        将布尔相关输入转换为Python布尔值。
    """
    return bool(value)


def ToDict(value):
    """
        将MATLAB结构体转换为Python字典结构。
    """
    if isinstance(value, np.ndarray) and value.dtype.names is not None:
        if value.size == 1:
            return {
                field: ConvertValue(value[field].item())
                for field in value.dtype.names
            }
        return np.array([ToDict(item) for item in value.reshape(-1)], dtype=object).reshape(value.shape)
    if hasattr(value, "_fieldnames"):
        return {
            field: ConvertValue(getattr(value, field))
            for field in value._fieldnames
        }
    return value


def ToCellArray(value):
    """
        将MATLAB元胞数组逐元素递归转换为对象数组。
    """
    converted = np.empty(value.shape, dtype=object)
    for idx, item in np.ndenumerate(value):
        converted[idx] = ConvertValue(item)
    return converted


def ConvertValue(value):
    """
        识别数据类型，选择转换方式，并整合成字典返回
    """
    if IsBoolScalar(value):
        return ToBool(value)

    if IsNumericScalar(value):
        return ToNumber(value)

    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_ and value.size == 1:
            return ToBool(value.item())
        if value.dtype.names is not None:
            return ToDict(value)
        if value.dtype == object:
            return ToCellArray(value)
        if value.dtype.kind in {"U", "S"}:
            return ToString(value)
        if np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_:
            if value.size == 1:
                scalar = value.item()
                if isinstance(scalar, (bool, np.bool_)):
                    return ToBool(scalar)
                if isinstance(scalar, np.generic):
                    return ToNumber(scalar)
                return scalar
            return ToNdArray(value)

    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        return ToString(value)

    if hasattr(value, "_fieldnames"):
        return ToDict(value)

    return value


def DataLoader(DataFile, names = [], id = [False]):
    """
        预处理数据，去除元数据
    """
    Data = {}
    i = 0
    j = -1
    for key in DataFile.keys():
        if (key != '__header__' and key != '__version__' and key != '__globals__'):
            j += 1
            if IsBoolScalar(DataFile[key]):
                Data[key] = ToBool(DataFile[key])
                continue

            if IsNumericScalar(DataFile[key]):
                Data[key] = ToNumber(DataFile[key])
                continue

            if isinstance(DataFile[key], np.ndarray):
                if DataFile[key].dtype == np.bool_ and DataFile[key].size == 1:
                    Data[key] = ToBool(DataFile[key].item())
                    continue
                if DataFile[key].dtype.names is not None:
                    Data[key] = ToDict(DataFile[key])
                    continue
                if DataFile[key].dtype == object:
                    Data[key] = ToCellArray(DataFile[key])
                    continue
                if DataFile[key].dtype.kind in {"U", "S"}:
                    Data[key] = ToString(DataFile[key])
                    continue
                if np.issubdtype(DataFile[key].dtype, np.number) or DataFile[key].dtype == np.bool_:
                    if DataFile[key].size == 1:
                        scalar = DataFile[key].item()
                        if isinstance(scalar, (bool, np.bool_)):
                            Data[key] = ToBool(scalar)
                            continue
                        if isinstance(scalar, np.generic):
                            Data[key] = ToNumber(scalar)
                            continue
                        Data[key] = scalar
                        continue

                    i += 1
                    if id[0]:
                        if len(id) > i:
                            id[i] = (bool(str(input(f"对于数组{names[j]}, 是否需要降维？(y/n)：") or ('y' if id[i] else 'n')).lower() == 'y'))
                        else:
                            id.append(bool(str(input(f"对于数组{names[j]}, 是否需要降维？(y/n)：")).lower() == 'y'))
                        Data[key] = ToNdArray(DataFile[key], id[i])
                    else:
                        try:
                            Data[key] = ToNdArray(DataFile[key], id[i])
                        except:
                            Data[key] = ToNdArray(DataFile[key])
                    continue

            if isinstance(DataFile[key], (str, bytes, np.str_, np.bytes_)):
                Data[key] = ToString(DataFile[key])
                continue

            if hasattr(DataFile[key], "_fieldnames"):
                Data[key] = ToDict(DataFile[key])
                continue

            Data[key] = DataFile[key]
    id[0] = False
    return Data


def Test(filePath, funcName=None, dataPool=None):
    """
        自动填充数据并调用，捕获输出
    """
    # try:
    fileName = os.path.basename(filePath)
    moduleName = os.path.splitext(fileName)[0]

    spec = importlib.util.spec_from_file_location(moduleName, filePath)

    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {filePath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[moduleName] = module
    spec.loader.exec_module(module)

    if funcName is None:
        funcName = moduleName

    if not hasattr(module, funcName):
        for commonName in ['main', 'run', 'execute']:
            if hasattr(module, commonName):
                funcName = commonName
                break
        else:
            raise AttributeError(f"模块中未找到函数: {funcName}")

    func = getattr(module, funcName)

    sig = inspect.signature(func)
    params = sig.parameters

    kwargs = {}

    for paramName, param in params.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        if dataPool is not None and paramName in dataPool:
            kwargs[paramName] = dataPool[paramName]
        elif param.default != inspect.Parameter.empty:
            kwargs[paramName] = param.default
        elif param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            kwargs[paramName] = None
        elif param.kind == param.POSITIONAL_ONLY:
            raise TypeError(f"缺少必需的位置参数: {paramName}")

    return func(**kwargs)


# except Exception as e:
#    print(f"调用文件出错: {e}")
#    return None


# 获取路径
if (0):#根据实际情况修改
    id = [False]
    if __name__ == "__main__":
        FilePath = str(input("请输入要测试的Python文件路径 (默认0) : ") or "SampleFunction.py")
        InputFilePath = str(input("请输入输入数据文件路径 (默认0) : ") or "input.mat")
        OutputFilePath = str(input("请输入输出数据文件路径 (默认0) : ") or "output.mat")
        tolerance = float(input("请输入数值比较的容差 (默认0.01) : ") or "0.01")
        names = list(str(input("请输入输入输出变量名称列表，逗号分隔: ") or "a,b,c,d").split(","))
        id.append(bool(str(input("需要为每个输入参数指定是否降维吗？(y/n)：")).lower() == 'y'))
    else:
        FilePath = "SampleFunction.py"
        InputFilePath = "input.mat"
        OutputFilePath = "output.mat"
        tolerance = 0.01
        names = ["a", "b", "c", "d"]
else:#如果不希望频繁改变路径
    FilePath = "SampleFunction.py"
    InputFilePath = "input.mat"
    OutputFilePath = "output.mat"
    tolerance = 0.01
    names = ["a", "b", "c", "d"]
    id = [False]

for s in FilePath:
    if s == '\\':
        s = '/'
for s in InputFilePath:
    if s == '\\':
        s = '/'
for s in OutputFilePath:
    if s == '\\':
        s = '/'

while 1:
    # 加载输入输出
    InputFile = loadmat(InputFilePath)
    OutputFile = loadmat(OutputFilePath)

    try:
        k = {}
        i = 0
        for key, value in InputFile.items():
            if not(key in ["__header__", "__version__", "__globals__"]):
                k[names[i]] = value
                i += 1
        InputFile = k

        k = {}
        for key, value in OutputFile.items():
            if not(key in ["__header__", "__version__", "__globals__"]):
                k[names[i]] = value
                i += 1
        OutputFile = k
    except:
        print(InputFile)
        print("---------------------------------")
        print(OutputFile)
        break
    # 处理输入输出
    Inputs = DataLoader(InputFile, names, id)
    Outputs = DataLoader(OutputFile)
    disOutputs = []
    for key in Outputs:
        disOutputs.append(Outputs[key])
    Outputs = disOutputs

    # 获取待测试代码路径
    fileName = os.path.basename(FilePath)
    ModuleName = os.path.splitext(fileName)[0]
    NewOutput = []

    # 测试
    try:
        OutPut = Test(FilePath, ModuleName, Inputs)

        # 处理多变量
        if isinstance(OutPut, tuple):
            NewOutput = list(OutPut)
        else:
            NewOutput.append(OutPut)

        print(NewOutput)
        # print(Inputs)
        print(Outputs)

        # 调用Diff.py比对
        TestDiff(ModuleName, Outputs, NewOutput, tolerance)
        b = str(input(
            "测试完成!\n输入 y 重试\n输入 n 退出\n输入 i 获得输入\n输入 o 获得输出\n输入 s 获得原始输出\n输入 b 重置降维选项\n"))
    except Exception as e:
        err = traceback.format_exc()
        b = str(input(
            f"调用文件出错: {sys.exc_info()[0]}\n完整如下：{err}\n输入 y 重试\n输入 n 退出\n输入 i 获得输入\n输入 o 获得输出\n输入 s 获得原始输出\n输入 b 重置降维选项\n"))
    b = b.lower()
    if b == 'n':
        break
    elif b == 'y':
        continue
    elif b == 's' or b == 'i' or b == 'o':
        while (1):
            if b == 's':
                print(Outputs)
            elif b == 'i':
                print(Inputs)
            elif b == 'o':
                print(NewOutputs)
            b = str(input("继续吗？(y/n/i/o/n)")).lower()
            if (b == 'y' or b == 'n'):
                break
        if (b == 'n'):
            break
    elif b == 'b':
        id[0] = True
        continue
    else:
        print("重试")
print("测试结束！")
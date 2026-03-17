from scipy.io import loadmat
from Diff import TestDiff
import importlib.util
import inspect
import numpy as np
import sys
import os


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


def ToNdArray(value):
    """
        将输入值转换为NumPy数组，向量转一维，矩阵保留二维。
    """
    arr = np.asarray(value)
    # loadmat通常将MATLAB向量读成(1, n)或(n, 1)，这里还原为一维数组。
    if arr.ndim == 2 and 1 in arr.shape:
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

def DataLoader(DataFile):
    """
        预处理数据，去除元数据
    """
    Data = {}
    for key in DataFile.keys():
        if (key != '__header__' and key != '__version__' and key != '__globals__'):
            Data[key] = ConvertValue(DataFile[key])
    return Data

def Test(filePath, funcName=None, dataPool=None):
    """
        自动填充数据并调用，捕获输出
    """
#try:
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
    
#except Exception as e:
#    print(f"调用文件出错: {e}")
#    return None


# 获取路径
if __name__ == "__main__":
    FilePath = str(input("请输入要测试的Python文件路径 (默认0) : ") or "SampleFunction.py")
    InputFilePath = str(input("请输入输入数据文件路径 (默认0) : ") or "input.mat")
    OutputFilePath = str(input("请输入输出数据文件路径 (默认0) : ") or "output.mat")
    tolerance = float(input("请输入数值比较的容差 (默认0.01) : ") or "0.01")
else:   
    FilePath = "SampleFunction.py"
    InputFilePath = "input.mat"
    OutputFilePath = "output.mat"
    tolerance = 0.01

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

    # 处理输入输出
    Inputs = DataLoader(InputFile)
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
    OutPut = Test(FilePath, ModuleName, Inputs)

    # 处理多变量
    if isinstance(OutPut, tuple):
        NewOutput = list(OutPut)
    else:
        NewOutput.append(OutPut)

    print(NewOutput)
    #print(Inputs)
    print(Outputs)

    # 调用Diff.py比对
    TestDiff(ModuleName, Outputs, NewOutput, tolerance)
    b = str(input("比对完成，需要重新比对吗？(y/n)"))
    if b.lower() != 'y':
        break
print("测试结束！")
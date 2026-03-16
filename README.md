# TestMat2Py
---
### 基于mat数据包的自动化python调试脚本
使用方法：
- 获得需要测试的python代码路径、输入数据包路径、输出数据包路径（支持绝对路径或者相对路径，绝对路径要求用正斜杠`/`）
- 启动test.py，根据提示输入测试路径
- 自动运行捕获输出并进行比对，比对报告生成在当前文件夹
  
默认值（输入-1）：
```python
    FilePath = "SampleFunction.py"
    InputFilePath = "input.mat"
    O=tputFilePath = "output.mat"
```
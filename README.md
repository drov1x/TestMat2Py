# TestMat2Py
### 基于mat数据包的自动化python调试脚本
使用方法：
- 获得需要测试的文件路径
- 启动test.py，根据提示输入测试路径
- 自动运行捕获输出并进行比对，比对报告生成在当前文件夹
  
提示：
- 路径支持绝对路径或者相对路径

默认值（输入-1）：
```python
    FilePath = "SampleFunction.py"
    InputFilePath = "input.mat"
    OutputFilePath = "output.mat"
```

---
##### 更新日志
20260317 优化元组，支持反斜杠
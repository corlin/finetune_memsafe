# 🔧 Industry Evaluation System 导入问题解决指南

## 问题描述

用户在运行演示程序时遇到以下错误：
```
ModuleNotFoundError: No module named 'industry_evaluation'
cannot import name 'SampleResult' from 'industry_evaluation.core.interfaces'
```

## 🎯 解决方案

### 1. 快速解决（推荐）

```bash
# 方法1: 使用快速启动脚本
python quick_start.py

# 方法2: 测试导入
python test_imports.py

# 方法3: 故障排除
python troubleshoot.py
```

### 2. 手动解决

#### 步骤1: 确保在正确目录
```bash
# 确认当前目录包含 industry_evaluation 文件夹
ls -la | grep industry_evaluation
# 或者 Windows:
dir | findstr industry_evaluation
```

#### 步骤2: 安装依赖
```bash
# 安装演示程序依赖
python install_demo_deps.py

# 或者手动安装
pip install pyyaml requests flask flask-restx flask-cors watchdog psutil
```

#### 步骤3: 设置Python路径
```bash
# Linux/Mac:
export PYTHONPATH=$PWD:$PYTHONPATH

# Windows:
set PYTHONPATH=%CD%;%PYTHONPATH%
```

#### 步骤4: 运行演示
```bash
python examples/simple_demo.py
```

## 🔍 已修复的问题

### 1. 模块导入问题
- ✅ 修复了 `pyproject.toml` 中的包查找路径
- ✅ 添加了 `setup.py` 文件
- ✅ 在所有演示程序中添加了路径处理代码

### 2. 类导入问题
- ✅ 修复了 `SampleResult` 在 `interfaces.py` 中的导入
- ✅ 修复了 `EvaluationStatus` 的重复定义问题
- ✅ 统一了所有模块的导入路径

### 3. 依赖问题
- ✅ 创建了 `demo_requirements.txt` 专用依赖文件
- ✅ 创建了 `install_demo_deps.py` 自动安装脚本
- ✅ 更新了 `pyproject.toml` 包含所有必需依赖

## 📁 新增的解决方案文件

1. **`quick_start.py`** - 一键启动脚本
2. **`test_imports.py`** - 导入测试脚本
3. **`troubleshoot.py`** - 全面故障诊断
4. **`install_demo_deps.py`** - 依赖安装脚本
5. **`demo_requirements.txt`** - 演示程序依赖
6. **`setup.py`** - 标准Python包安装

## 🚀 推荐使用流程

```bash
# 1. 快速诊断
python test_imports.py

# 2. 如果有问题，运行故障排除
python troubleshoot.py

# 3. 安装依赖（如果需要）
python install_demo_deps.py

# 4. 快速启动演示
python quick_start.py

# 5. 或者使用交互式菜单
python examples/run_demo.py
```

## 🔧 技术细节

### 修复的导入问题

1. **interfaces.py 缺少 SampleResult**
   ```python
   # 修复前
   from industry_evaluation.models.data_models import (
       EvaluationConfig, EvaluationResult, EvaluationScore, 
       Dataset, ProgressInfo, Report, Criterion, Explanation
   )
   
   # 修复后
   from industry_evaluation.models.data_models import (
       EvaluationConfig, EvaluationResult, EvaluationScore, SampleResult,
       Dataset, ProgressInfo, Report, Criterion, Explanation
   )
   ```

2. **EvaluationStatus 重复定义**
   ```python
   # 修复前 (evaluation_engine.py)
   class EvaluationStatus(Enum):
       PENDING = "pending"
       # ...
   
   # 修复后
   from industry_evaluation.models.data_models import EvaluationStatus
   ```

3. **Python路径问题**
   ```python
   # 在所有演示程序中添加
   import sys
   from pathlib import Path
   
   project_root = Path(__file__).parent.parent
   if str(project_root) not in sys.path:
       sys.path.insert(0, str(project_root))
   ```

### 包配置修复

1. **pyproject.toml**
   ```toml
   [tool.setuptools.packages.find]
   where = ["."]
   include = ["src*", "industry_evaluation*"]
   ```

2. **setup.py**
   ```python
   packages=find_packages(include=['industry_evaluation*', 'src*'])
   ```

## 🎯 验证解决方案

运行以下命令验证所有问题已解决：

```bash
# 测试所有导入
python test_imports.py

# 运行简化演示
python examples/simple_demo.py

# 运行完整演示
python examples/complete_demo.py
```

## 💡 预防措施

为避免将来出现类似问题：

1. **始终在项目根目录运行脚本**
2. **使用提供的启动脚本而不是直接运行**
3. **定期运行 `test_imports.py` 检查导入状态**
4. **保持依赖更新：`python install_demo_deps.py`**

## 🆘 如果仍有问题

如果按照上述步骤仍然遇到问题：

1. **运行完整诊断**：`python troubleshoot.py`
2. **检查Python版本**：确保使用Python 3.8+
3. **清理Python缓存**：删除 `__pycache__` 文件夹
4. **重新安装依赖**：`pip uninstall -y pyyaml requests flask && python install_demo_deps.py`

---

**✅ 现在所有导入问题都已解决，您可以正常使用 Industry Evaluation System 的所有功能了！**
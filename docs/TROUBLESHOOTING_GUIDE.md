# 故障排除指南

本指南帮助您解决使用数据拆分和评估系统时可能遇到的常见问题。

## 目录

1. [安装问题](#安装问题)
2. [数据拆分问题](#数据拆分问题)
3. [模型评估问题](#模型评估问题)
4. [内存和性能问题](#内存和性能问题)
5. [配置问题](#配置问题)
6. [基准测试问题](#基准测试问题)
7. [报告生成问题](#报告生成问题)
8. [常见错误信息](#常见错误信息)
9. [调试技巧](#调试技巧)
10. [获取帮助](#获取帮助)

## 安装问题

### 问题：uv安装失败

**错误信息:**
```
curl: command not found
```

**解决方案:**
1. 在Windows上使用PowerShell：
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. 手动下载安装：
   - 访问 https://github.com/astral-sh/uv/releases
   - 下载对应平台的二进制文件
   - 添加到PATH环境变量

### 问题：依赖包安装失败

**错误信息:**
```
ERROR: Could not find a version that satisfies the requirement
```

**解决方案:**
1. 更新uv到最新版本：
   ```bash
   uv self update
   ```

2. 清理缓存：
   ```bash
   uv cache clean
   ```

3. 使用特定Python版本：
   ```bash
   uv sync --python 3.8
   ```

### 问题：GPU相关包安装失败

**错误信息:**
```
RuntimeError: CUDA out of memory
```

**解决方案:**
1. 安装CPU版本的PyTorch：
   ```bash
   uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. 检查CUDA版本兼容性：
   ```bash
   nvidia-smi
   ```

## 数据拆分问题

### 问题：数据集太小无法拆分

**错误信息:**
```
ValueError: 数据集太小，无法按指定比例拆分
```

**解决方案:**
1. 调整最小样本数要求：
   ```python
   splitter = DataSplitter(min_samples_per_split=1)
   ```

2. 修改拆分比例：
   ```python
   splitter = DataSplitter(
       train_ratio=0.8,
       val_ratio=0.1,
       test_ratio=0.1
   )
   ```

3. 增加数据集大小或使用数据增强

### 问题：分层抽样失败

**错误信息:**
```
ValueError: 某些类别样本数不足
```

**解决方案:**
1. 检查类别分布：
   ```python
   from collections import Counter
   label_counts = Counter(dataset['label'])
   print(label_counts)
   ```

2. 禁用分层抽样：
   ```python
   splitter = DataSplitter(stratify_by=None)
   ```

3. 合并小类别或移除样本数过少的类别

### 问题：数据泄露检测误报

**错误信息:**
```
Warning: 检测到数据泄露
```

**解决方案:**
1. 检查数据预处理是否正确
2. 确认重复样本是否合理
3. 调整泄露检测阈值：
   ```python
   splitter = DataSplitter(leakage_threshold=0.1)
   ```

## 模型评估问题

### 问题：模型加载失败

**错误信息:**
```
OSError: Can't load tokenizer for 'model_path'
```

**解决方案:**
1. 检查模型路径是否正确
2. 确认模型文件完整性
3. 使用正确的模型加载方式：
   ```python
   from transformers import AutoModel, AutoTokenizer
   
   model = AutoModel.from_pretrained("model_path", trust_remote_code=True)
   tokenizer = AutoTokenizer.from_pretrained("model_path", trust_remote_code=True)
   ```

### 问题：评估指标计算错误

**错误信息:**
```
ValueError: 预测文本和参考文本数量不匹配
```

**解决方案:**
1. 检查数据格式：
   ```python
   print(f"预测数量: {len(predictions)}")
   print(f"参考数量: {len(references)}")
   ```

2. 处理缺失数据：
   ```python
   # 填充缺失的预测
   predictions = [pred if pred else "" for pred in predictions]
   
   # 或者过滤掉缺失数据
   valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
   predictions, references = zip(*valid_pairs)
   ```

### 问题：BLEU分数计算异常

**错误信息:**
```
ZeroDivisionError: division by zero
```

**解决方案:**
1. 检查文本是否为空：
   ```python
   predictions = [p.strip() for p in predictions if p.strip()]
   references = [r.strip() for r in references if r.strip()]
   ```

2. 使用平滑处理：
   ```python
   calculator = MetricsCalculator(smoothing=True)
   ```

### 问题：BERTScore计算缓慢

**解决方案:**
1. 使用较小的模型：
   ```python
   calculator = MetricsCalculator(bertscore_model="bert-base-chinese")
   ```

2. 启用缓存：
   ```python
   calculator = MetricsCalculator(cache_dir=".bertscore_cache")
   ```

3. 减少样本数量：
   ```python
   config = EvaluationConfig(num_samples=100)
   ```

## 内存和性能问题

### 问题：CUDA内存不足

**错误信息:**
```
RuntimeError: CUDA out of memory
```

**解决方案:**
1. 减少批次大小：
   ```python
   config = EvaluationConfig(batch_size=2)
   ```

2. 启用内存优化：
   ```python
   config = EvaluationConfig(
       memory_optimization=True,
       gradient_checkpointing=True
   )
   ```

3. 清理GPU缓存：
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. 使用CPU评估：
   ```python
   config = EvaluationConfig(device="cpu")
   ```

### 问题：评估速度过慢

**解决方案:**
1. 启用并行处理：
   ```python
   engine = EvaluationEngine(config, max_workers=4)
   ```

2. 使用GPU加速：
   ```python
   config = EvaluationConfig(device="cuda")
   ```

3. 减少评估样本数：
   ```python
   config = EvaluationConfig(num_samples=500)
   ```

4. 优化批次大小：
   ```python
   # 找到最优批次大小
   for batch_size in [4, 8, 16, 32]:
       config = EvaluationConfig(batch_size=batch_size)
       # 测试性能
   ```

### 问题：内存使用过高

**解决方案:**
1. 启用内存监控：
   ```python
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

2. 使用数据流处理：
   ```python
   # 分批处理大数据集
   for batch in dataset.iter(batch_size=1000):
       result = engine.evaluate_model(model, tokenizer, {"task": batch})
   ```

3. 定期清理缓存：
   ```python
   import gc
   gc.collect()
   ```

## 配置问题

### 问题：配置文件格式错误

**错误信息:**
```
yaml.YAMLError: 配置文件格式错误
```

**解决方案:**
1. 检查YAML语法：
   ```bash
   # 使用在线YAML验证器检查语法
   ```

2. 检查缩进：
   ```yaml
   # 正确的缩进
   evaluation:
     tasks:
       - "classification"
     metrics:
       - "accuracy"
   ```

3. 检查特殊字符：
   ```yaml
   # 使用引号包围包含特殊字符的值
   description: "这是一个包含:特殊字符的描述"
   ```

### 问题：环境变量替换失败

**错误信息:**
```
KeyError: 'UNDEFINED_VAR'
```

**解决方案:**
1. 设置默认值：
   ```yaml
   model_path: ${MODEL_PATH:./default_model}
   ```

2. 检查环境变量：
   ```bash
   echo $MODEL_PATH
   ```

3. 在代码中设置环境变量：
   ```python
   import os
   os.environ['MODEL_PATH'] = '/path/to/model'
   ```

### 问题：配置验证失败

**错误信息:**
```
ConfigurationError: 配置验证失败
```

**解决方案:**
1. 检查必需字段：
   ```python
   config_manager = ConfigManager()
   errors = config_manager.get_validation_errors(config)
   for error in errors:
       print(f"配置错误: {error}")
   ```

2. 使用默认配置：
   ```python
   default_config = config_manager.get_default_config()
   merged_config = config_manager.merge_configs(default_config, user_config)
   ```

## 基准测试问题

### 问题：基准数据集下载失败

**错误信息:**
```
ConnectionError: 无法下载基准数据集
```

**解决方案:**
1. 检查网络连接
2. 使用代理：
   ```python
   import os
   os.environ['HTTP_PROXY'] = 'http://proxy.example.com:8080'
   os.environ['HTTPS_PROXY'] = 'https://proxy.example.com:8080'
   ```

3. 手动下载数据集：
   ```python
   # 将数据集放在指定目录
   benchmark_manager = BenchmarkManager(benchmark_dir="./local_benchmarks")
   ```

### 问题：基准测试结果异常

**解决方案:**
1. 检查数据格式：
   ```python
   # 确认数据格式符合基准要求
   print(dataset[0])
   ```

2. 验证评估协议：
   ```python
   # 使用官方评估协议
   result = benchmark_manager.run_clue_evaluation(
       model, tokenizer, "model_name",
       strict_protocol=True
   )
   ```

## 报告生成问题

### 问题：HTML报告生成失败

**错误信息:**
```
TemplateNotFound: 找不到模板文件
```

**解决方案:**
1. 检查模板目录：
   ```python
   generator = ReportGenerator(template_dir="./templates")
   ```

2. 使用内置模板：
   ```python
   generator = ReportGenerator(use_builtin_templates=True)
   ```

### 问题：图表生成失败

**错误信息:**
```
ImportError: No module named 'matplotlib'
```

**解决方案:**
1. 安装可视化依赖：
   ```bash
   uv add matplotlib seaborn
   ```

2. 禁用图表生成：
   ```python
   generator = ReportGenerator(include_plots=False)
   ```

### 问题：PDF报告生成失败

**解决方案:**
1. 安装PDF依赖：
   ```bash
   uv add weasyprint  # 或 wkhtmltopdf
   ```

2. 使用HTML报告替代：
   ```python
   report = generator.generate_evaluation_report(result, format="html")
   ```

## 常见错误信息

### ImportError相关

```
ImportError: No module named 'evaluation'
```

**解决方案:**
- 检查Python路径
- 重新安装包：`uv sync`
- 确认在正确的虚拟环境中

### FileNotFoundError相关

```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方案:**
- 检查文件路径
- 创建必要的目录
- 使用绝对路径

### PermissionError相关

```
PermissionError: [Errno 13] Permission denied
```

**解决方案:**
- 检查文件权限
- 使用管理员权限运行
- 更改输出目录

### TimeoutError相关

```
TimeoutError: 操作超时
```

**解决方案:**
- 增加超时时间
- 检查网络连接
- 使用更小的数据集测试

## 调试技巧

### 1. 启用详细日志

```python
import logging
from evaluation.logging_system import setup_logging

# 设置DEBUG级别日志
setup_logging(level="DEBUG", file="debug.log")
```

### 2. 使用小数据集测试

```python
# 创建小测试数据集
test_dataset = dataset.select(range(10))
config = EvaluationConfig(num_samples=5)
```

### 3. 分步调试

```python
try:
    # 步骤1：数据拆分
    print("开始数据拆分...")
    split_result = splitter.split_data(dataset, "debug_splits")
    print("数据拆分完成")
    
    # 步骤2：模型评估
    print("开始模型评估...")
    result = engine.evaluate_model(model, tokenizer, datasets, "debug_model")
    print("模型评估完成")
    
except Exception as e:
    print(f"错误发生在: {e}")
    import traceback
    traceback.print_exc()
```

### 4. 内存监控

```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"当前内存使用: {memory_mb:.2f} MB")

# 在关键点监控内存
monitor_memory()
result = engine.evaluate_model(...)
monitor_memory()
```

### 5. 性能分析

```python
import time
import cProfile

def profile_function(func, *args, **kwargs):
    start_time = time.time()
    
    # 使用cProfile分析
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    profiler.dump_stats('profile_output.prof')
    
    end_time = time.time()
    print(f"执行时间: {end_time - start_time:.2f} 秒")
    
    return result

# 分析评估函数性能
result = profile_function(
    engine.evaluate_model,
    model, tokenizer, datasets, "profile_model"
)
```

### 6. 错误恢复

```python
from evaluation.exceptions import EvaluationError

def robust_evaluation(engine, model, tokenizer, datasets, model_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            return engine.evaluate_model(model, tokenizer, datasets, model_name)
        except EvaluationError as e:
            print(f"尝试 {attempt + 1} 失败: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # 指数退避
```

## 获取帮助

### 1. 检查文档

- [用户指南](EVALUATION_USER_GUIDE.md)
- [API参考](API_REFERENCE.md)
- [配置指南](CONFIGURATION_GUIDE.md)

### 2. 查看示例代码

- [基本使用示例](../examples/basic_usage.py)
- [高级评估示例](../examples/advanced_evaluation.py)

### 3. 运行测试

```bash
# 运行测试以验证安装
uv run python tests/test_runner.py --type unit
```

### 4. 社区支持

- 搜索已知问题
- 查看FAQ
- 提交问题报告

### 5. 创建最小复现示例

当报告问题时，请提供：

```python
# 最小复现示例
from evaluation import DataSplitter
from datasets import Dataset

# 创建最小数据集
data = [{"text": "test", "label": "A"}] * 5
dataset = Dataset.from_list(data)

# 复现问题的代码
splitter = DataSplitter()
try:
    result = splitter.split_data(dataset, "test_output")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
```

### 6. 系统信息收集

```python
import sys
import platform
import torch

def collect_system_info():
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__ if torch else "Not installed",
        "cuda_available": torch.cuda.is_available() if torch else False,
    }
    
    try:
        import evaluation
        info["evaluation_version"] = evaluation.__version__
    except:
        info["evaluation_version"] = "Unknown"
    
    return info

print(collect_system_info())
```

## 预防措施

### 1. 定期备份

```python
# 备份实验数据
tracker.backup_experiments("backup.json")

# 备份配置文件
config_manager.backup_config("config.yaml")
```

### 2. 版本控制

- 使用Git管理配置文件
- 记录依赖版本
- 标记重要的实验版本

### 3. 监控资源使用

```python
# 设置资源监控
from evaluation.monitoring import ResourceMonitor

monitor = ResourceMonitor()
monitor.start()

# 执行评估
result = engine.evaluate_model(...)

# 查看资源使用报告
report = monitor.get_report()
```

### 4. 渐进式测试

1. 先用小数据集测试
2. 逐步增加数据量
3. 监控性能变化
4. 调整配置参数

---

*本故障排除指南会持续更新，如遇到新问题请及时反馈。*
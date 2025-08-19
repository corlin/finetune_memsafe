# 评估系统故障排除指南

## 概述

本指南帮助您诊断和解决评估系统中常见的数据处理问题，特别是批次数据处理相关的问题。

## 快速诊断工具

### 1. 自动诊断

```python
from src.evaluation import create_enhanced_evaluation_engine

# 创建增强评估引擎
engine = create_enhanced_evaluation_engine()

# 诊断数据集
diagnosis = engine.diagnose_dataset(your_dataset, "your_task_name")

print("=== 诊断结果 ===")
print(f"批次信息: {diagnosis['batch_info']}")
print(f"验证结果: {diagnosis['validation_result']}")
print(f"字段检测: {diagnosis['field_detection_result']}")
print(f"建议: {diagnosis['recommendations']}")
```

### 2. 配置验证

```python
from src.evaluation import validate_config_file

# 验证配置文件
result = validate_config_file("your_config.yaml")

if not result["is_valid"]:
    print("配置问题:")
    for error in result["errors"]:
        print(f"  - {error}")
    
    print("建议:")
    for suggestion in result["suggestions"]:
        print(f"  - {suggestion}")
```

## 常见问题分类

### A. 数据格式问题

#### A1. 批次数据为空

**症状**:
```
WARNING - 批次数据为空，任务: text_generation，批次键: ['input_ids', 'attention_mask', 'labels']
WARNING - 跳过空批次，索引: 0-4
```

**原因**: 数据字段名称不匹配系统预期

**诊断步骤**:
```python
# 1. 检查数据集结构
print(f"数据集字段: {list(dataset.column_names)}")
print(f"数据集大小: {len(dataset)}")
print(f"样本示例: {dataset[0]}")

# 2. 使用诊断工具
diagnosis = engine.diagnose_dataset(dataset, task_name)
print(f"可用字段: {diagnosis['batch_info']['available_fields']}")
print(f"推荐字段: {diagnosis['field_mapping_info']['recommended_input_field']}")
```

**解决方案**:

**方案1: 使用自定义字段映射**
```python
custom_config = {
    "data_processing": {
        "field_mapping": {
            "text_generation": {
                "input_fields": ["your_input_field_name"],
                "target_fields": ["your_target_field_name"]
            }
        }
    }
}

engine = create_enhanced_evaluation_engine(config_data=custom_config)
```

**方案2: 重命名数据集字段**
```python
# 重命名字段以匹配标准名称
dataset = dataset.rename_column("your_input_field", "text")
dataset = dataset.rename_column("your_target_field", "target")
```

**方案3: 启用降级处理**
```python
config = {
    "data_processing": {
        "validation": {
            "enable_fallback": True,
            "enable_data_cleaning": True
        }
    }
}
```

#### A2. 数据类型错误

**症状**:
```
ERROR - 字段 'text' 数据类型错误，期望list，实际str
```

**原因**: 字段包含非列表类型的数据

**诊断步骤**:
```python
# 检查字段数据类型
for field_name in dataset.column_names:
    field_data = dataset[field_name]
    print(f"{field_name}: {type(field_data)} - {type(field_data[0]) if field_data else 'empty'}")
```

**解决方案**:
```python
# 转换数据类型
if isinstance(dataset["text"], str):
    # 如果是单个字符串，转换为列表
    dataset = dataset.map(lambda x: {"text": [x["text"]]})

# 或者使用错误处理策略
from src.evaluation import ErrorHandlingStrategy
handler = ErrorHandlingStrategy(enable_fallback=True)
```

#### A3. 字段长度不一致

**症状**:
```
WARNING - 字段长度不一致: {'text': 100, 'target': 95}
```

**原因**: 不同字段的样本数量不匹配

**诊断步骤**:
```python
# 检查字段长度
field_lengths = {}
for field_name in dataset.column_names:
    field_lengths[field_name] = len(dataset[field_name])

print(f"字段长度: {field_lengths}")

# 找出长度不一致的字段
lengths = list(field_lengths.values())
if len(set(lengths)) > 1:
    print(f"长度不一致: 最小={min(lengths)}, 最大={max(lengths)}")
```

**解决方案**:
```python
# 方案1: 截断到最小长度
min_length = min(len(dataset[field]) for field in dataset.column_names)
dataset = dataset.select(range(min_length))

# 方案2: 填充到最大长度
max_length = max(len(dataset[field]) for field in dataset.column_names)
# 需要自定义填充逻辑

# 方案3: 启用数据清洗
config = {
    "data_processing": {
        "validation": {
            "enable_data_cleaning": True
        }
    }
}
```

### B. 数据质量问题

#### B1. 有效样本比例过低

**症状**:
```
WARNING - 有效样本比例过低: 15.00% (最小要求: 50.00%)
```

**原因**: 数据中包含大量空值或无效数据

**诊断步骤**:
```python
# 分析数据质量
from src.evaluation import BatchDataValidator

validator = BatchDataValidator()
stats = validator.get_batch_statistics(dataset[:100])  # 检查前100个样本

for field_name, field_stats in stats["field_stats"].items():
    print(f"{field_name}:")
    print(f"  总数: {field_stats['length']}")
    print(f"  有效数: {field_stats['valid_count']}")
    print(f"  有效率: {field_stats['valid_ratio']:.2%}")
    print(f"  空值数: {field_stats['empty_count']}")
```

**解决方案**:

**方案1: 调整有效样本比例阈值**
```python
config = {
    "data_processing": {
        "validation": {
            "min_valid_samples_ratio": 0.1  # 降低到10%
        }
    }
}
```

**方案2: 启用数据清洗**
```python
config = {
    "data_processing": {
        "validation": {
            "enable_data_cleaning": True,
            "enable_fallback": True
        }
    }
}
```

**方案3: 预处理数据**
```python
# 过滤空值
def filter_empty(example):
    return example["text"] is not None and str(example["text"]).strip() != ""

dataset = dataset.filter(filter_empty)
```

#### B2. 数据编码问题

**症状**:
```
ERROR - 分词器编码失败: 'utf-8' codec can't decode byte
```

**原因**: 数据包含非UTF-8编码的字符

**诊断步骤**:
```python
# 检查编码问题
import unicodedata

def check_encoding(text):
    try:
        # 尝试编码/解码
        text.encode('utf-8').decode('utf-8')
        return True
    except UnicodeError:
        return False

# 检查数据集中的编码问题
encoding_issues = []
for i, example in enumerate(dataset):
    if not check_encoding(str(example.get("text", ""))):
        encoding_issues.append(i)

print(f"发现 {len(encoding_issues)} 个编码问题")
```

**解决方案**:
```python
# 清理编码问题
def clean_encoding(example):
    text = str(example["text"])
    # 移除或替换非UTF-8字符
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    # 标准化Unicode字符
    text = unicodedata.normalize('NFKC', text)
    example["text"] = text
    return example

dataset = dataset.map(clean_encoding)
```

### C. 性能问题

#### C1. 处理速度慢

**症状**:
```
INFO - 批次处理时间较长，建议优化数据预处理逻辑或减小批次大小
```

**诊断步骤**:
```python
import time
from src.evaluation import DataPreprocessor

# 测量处理时间
preprocessor = DataPreprocessor(config)
times = []

for i in range(0, min(100, len(dataset)), 10):
    batch = dataset[i:i+10]
    
    start_time = time.time()
    result = preprocessor.preprocess_batch(batch, task_name)
    end_time = time.time()
    
    times.append(end_time - start_time)

avg_time = sum(times) / len(times)
print(f"平均批次处理时间: {avg_time*1000:.2f}ms")
print(f"处理速度: {10/avg_time:.1f} samples/s")
```

**解决方案**:

**方案1: 优化批次大小**
```python
# 根据数据大小调整批次大小
if len(dataset) > 10000:
    batch_size = 100
elif len(dataset) > 1000:
    batch_size = 50
else:
    batch_size = 20

config = EvaluationConfig(batch_size=batch_size)
```

**方案2: 禁用详细日志**
```python
config = {
    "data_processing": {
        "diagnostics": {
            "enable_detailed_logging": False,
            "log_batch_statistics": False
        }
    }
}
```

**方案3: 启用并发处理**
```python
engine = create_enhanced_evaluation_engine()
engine.max_workers = 4  # 使用4个工作线程
```

#### C2. 内存使用过高

**症状**:
```
WARNING - 峰值内存使用过高: 2.5GB
```

**诊断步骤**:
```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# 监控内存使用
initial_memory = monitor_memory()
print(f"初始内存: {initial_memory:.1f}MB")

# 处理数据
result = preprocessor.preprocess_batch(large_batch, task_name)

final_memory = monitor_memory()
print(f"最终内存: {final_memory:.1f}MB")
print(f"内存增长: {final_memory - initial_memory:.1f}MB")
```

**解决方案**:

**方案1: 减小批次大小**
```python
config = EvaluationConfig(batch_size=16)  # 减小批次大小
```

**方案2: 分批处理大数据集**
```python
# 分批处理
chunk_size = 1000
for i in range(0, len(large_dataset), chunk_size):
    chunk = large_dataset[i:i+chunk_size]
    result = engine.evaluate_model(model, tokenizer, {"task": chunk})
    # 处理结果...
```

**方案3: 禁用不必要的功能**
```python
config = {
    "enable_efficiency_metrics": False,
    "enable_quality_analysis": False,
    "data_processing": {
        "diagnostics": {
            "save_processing_report": False
        }
    }
}
```

### D. 配置问题

#### D1. 配置文件无效

**症状**:
```
ERROR - 配置验证失败: min_valid_samples_ratio必须在0-1之间
```

**诊断步骤**:
```python
from src.evaluation import ConfigValidator

validator = ConfigValidator()
is_valid = validator.validate_processing_config(your_config)

if not is_valid:
    print("验证错误:")
    for error in validator.get_validation_errors():
        print(f"  - {error}")
    
    print("验证警告:")
    for warning in validator.get_validation_warnings():
        print(f"  - {warning}")
```

**解决方案**:

**方案1: 自动修复配置**
```python
fixed_config, fixes = validator.validate_and_fix_config(your_config)
print(f"应用的修复: {fixes}")
```

**方案2: 使用默认配置**
```python
default_config = validator.get_default_config()
# 合并用户配置和默认配置
merged_config = validator.merge_with_default(your_config)
```

#### D2. 字段映射配置错误

**症状**:
```
WARNING - 任务 'my_task' 没有特定的映射规则，返回原始数据
```

**解决方案**:
```python
# 添加自定义任务的字段映射
custom_mapping = {
    "field_mapping": {
        "my_task": {
            "input_fields": ["my_input_field"],
            "target_fields": ["my_target_field"]
        }
    }
}

config = EvaluationConfig(data_processing=custom_mapping)
```

## 调试技巧

### 1. 启用详细日志

```python
import logging

# 设置日志级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 启用详细日志
config = {
    "data_processing": {
        "diagnostics": {
            "enable_detailed_logging": True
        }
    }
}
```

### 2. 逐步调试

```python
from src.evaluation import DataPreprocessor

preprocessor = DataPreprocessor(config)

# 1. 检查单个样本
sample = dataset[0]
print(f"样本数据: {sample}")

# 2. 检查小批次
small_batch = dataset[:4]
result = preprocessor.preprocess_batch(small_batch, task_name)
print(f"小批次结果: {len(result.inputs)} 个输入")

# 3. 逐步增加批次大小
for batch_size in [4, 8, 16, 32]:
    batch = dataset[:batch_size]
    result = preprocessor.preprocess_batch(batch, task_name)
    print(f"批次大小 {batch_size}: {len(result.inputs)} 个有效输入")
```

### 3. 使用诊断报告

```python
# 生成详细的诊断报告
report = preprocessor.generate_processing_report()

# 保存报告到文件
report_path = preprocessor.save_processing_report("debug_report.json")
print(f"诊断报告已保存到: {report_path}")

# 查看关键信息
print(f"会话统计: {report['session_info']}")
print(f"批次统计: {report['batch_statistics']['summary']}")
print(f"建议: {report['recommendations']}")
```

## 预防措施

### 1. 数据验证

```python
def validate_dataset(dataset, task_name):
    """验证数据集格式"""
    
    # 检查基本结构
    if len(dataset) == 0:
        raise ValueError("数据集为空")
    
    # 检查字段存在性
    required_fields = ["text", "input", "prompt"]  # 根据任务调整
    available_fields = list(dataset.column_names)
    
    if not any(field in available_fields for field in required_fields):
        raise ValueError(f"未找到必需字段。可用字段: {available_fields}")
    
    # 检查数据类型
    for field_name in available_fields:
        field_data = dataset[field_name]
        if not isinstance(field_data, list):
            raise ValueError(f"字段 '{field_name}' 必须是列表类型")
    
    print(f"数据集验证通过: {len(dataset)} 个样本")

# 使用验证函数
validate_dataset(your_dataset, "text_generation")
```

### 2. 配置模板

```python
# 创建配置模板
from src.evaluation import ConfigValidator

validator = ConfigValidator()
template = validator.get_config_template()

# 保存模板到文件
import yaml
with open("config_template.yaml", "w") as f:
    yaml.dump(template, f, default_flow_style=False)
```

### 3. 自动化测试

```python
def test_evaluation_pipeline(model, tokenizer, dataset, task_name):
    """测试评估流程"""
    
    try:
        # 1. 创建引擎
        engine = create_enhanced_evaluation_engine()
        
        # 2. 诊断数据集
        diagnosis = engine.diagnose_dataset(dataset, task_name)
        
        # 3. 检查诊断结果
        if len(diagnosis["recommendations"]) > 0:
            print(f"数据集问题: {diagnosis['recommendations']}")
        
        # 4. 执行评估
        result = engine.evaluate_model_with_diagnostics(
            model, tokenizer, {task_name: dataset}, "test_model"
        )
        
        # 5. 检查结果
        eval_result = result["evaluation_result"]
        if task_name not in eval_result.task_results:
            raise ValueError(f"任务 {task_name} 评估失败")
        
        print("评估流程测试通过")
        return True
        
    except Exception as e:
        print(f"评估流程测试失败: {e}")
        return False

# 运行测试
success = test_evaluation_pipeline(model, tokenizer, dataset, "text_generation")
```

## 联系支持

如果以上方法都无法解决您的问题，请提供以下信息：

1. **错误信息**: 完整的错误日志
2. **数据样本**: 脱敏的数据样本（前几行）
3. **配置信息**: 使用的配置文件
4. **环境信息**: Python版本、依赖包版本
5. **诊断报告**: 使用诊断工具生成的报告

```python
# 收集诊断信息
def collect_diagnostic_info(dataset, task_name, config):
    info = {
        "dataset_info": {
            "size": len(dataset),
            "columns": list(dataset.column_names),
            "sample": dataset[0] if len(dataset) > 0 else None
        },
        "config": config.to_dict() if hasattr(config, 'to_dict') else config,
        "python_version": sys.version,
        "task_name": task_name
    }
    
    # 生成诊断报告
    engine = create_enhanced_evaluation_engine()
    diagnosis = engine.diagnose_dataset(dataset, task_name)
    info["diagnosis"] = diagnosis
    
    return info

# 使用示例
diagnostic_info = collect_diagnostic_info(dataset, task_name, config)
print(json.dumps(diagnostic_info, indent=2, ensure_ascii=False))
```
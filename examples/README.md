# 示例代码

本目录包含了数据拆分和评估系统的各种使用示例。

## 目录结构

```
examples/
├── README.md                    # 本文件
├── basic_usage.py              # 基本使用示例
├── advanced_evaluation.py      # 高级评估示例
├── config_examples/            # 配置文件示例
│   ├── basic_config.yaml       # 基础配置
│   └── advanced_config.yaml    # 高级配置
├── benchmark_examples/         # 基准测试示例
├── custom_tasks/              # 自定义任务示例
└── integration_examples/      # 集成示例
```

## 快速开始

### 1. 基本使用示例

最简单的使用方式，演示核心功能：

```bash
# 使用uv运行
uv run python examples/basic_usage.py

# 或使用python直接运行
python examples/basic_usage.py
```

**功能演示:**
- 数据拆分
- 模型评估
- 实验跟踪
- 报告生成

### 2. 高级评估示例

展示系统的高级功能：

```bash
uv run python examples/advanced_evaluation.py
```

**功能演示:**
- 多模型对比评估
- 质量分析
- 基准测试
- 高级报告生成
- 性能分析

## 配置文件示例

### 基础配置 (basic_config.yaml)

适用于简单的分类任务：

```yaml
data_split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

evaluation:
  tasks: ["classification"]
  metrics: ["accuracy", "f1"]
  batch_size: 8
```

使用方法：
```python
from evaluation.config_manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config("examples/config_examples/basic_config.yaml")
```

### 高级配置 (advanced_config.yaml)

适用于复杂的多任务评估：

```yaml
evaluation:
  tasks: ["text_generation", "classification", "question_answering"]
  metrics: ["bleu", "rouge", "accuracy", "f1", "bertscore"]
  batch_size: 16
  device: "cuda"
  memory_optimization: true
```

## 自定义示例

### 创建自定义评估任务

```python
from evaluation.task_evaluators import CustomTaskEvaluator

class MyCustomEvaluator(CustomTaskEvaluator):
    def evaluate(self, predictions, references, **kwargs):
        # 实现自定义评估逻辑
        custom_score = self.calculate_custom_metric(predictions, references)
        return {"custom_metric": custom_score}
    
    def calculate_custom_metric(self, predictions, references):
        # 自定义指标计算
        return 0.85

# 注册自定义评估器
engine.register_task_evaluator("custom_task", MyCustomEvaluator())
```

### 自定义数据拆分策略

```python
from evaluation import DataSplitter

class CustomDataSplitter(DataSplitter):
    def custom_split_strategy(self, dataset):
        # 实现自定义拆分逻辑
        pass

splitter = CustomDataSplitter()
```

## 基准测试示例

### 运行CLUE基准测试

```python
from evaluation import BenchmarkManager

benchmark_manager = BenchmarkManager()

# 运行CLUE评估
clue_result = benchmark_manager.run_clue_evaluation(
    model=model,
    tokenizer=tokenizer,
    model_name="my_model"
)

print(f"CLUE总分: {clue_result.overall_score:.3f}")
```

### 自定义基准测试

```python
from evaluation.data_models import BenchmarkConfig

custom_config = BenchmarkConfig(
    name="my_benchmark",
    dataset_path="path/to/data.json",
    tasks=["custom_task"],
    evaluation_protocol="standard",
    metrics=["accuracy", "f1"]
)

result = benchmark_manager.run_custom_benchmark(
    config=custom_config,
    model=model,
    tokenizer=tokenizer,
    model_name="my_model"
)
```

## 集成示例

### 与训练流程集成

```python
from evaluation import EvaluationEngine, DataSplitter

# 在训练过程中集成评估
class TrainingWithEvaluation:
    def __init__(self, model, tokenizer, eval_config):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_engine = EvaluationEngine(eval_config)
    
    def train_epoch(self, train_data):
        # 训练逻辑
        pass
    
    def evaluate_epoch(self, val_data):
        # 每个epoch后评估
        result = self.eval_engine.evaluate_model(
            self.model, self.tokenizer, 
            {"validation": val_data}, 
            "training_model"
        )
        return result
```

### 与数据管道集成

```python
from evaluation import DataSplitter, QualityAnalyzer

class DataPipelineWithEvaluation:
    def __init__(self):
        self.splitter = DataSplitter()
        self.quality_analyzer = QualityAnalyzer()
    
    def process_data(self, raw_data):
        # 质量分析
        quality_report = self.quality_analyzer.analyze_data_quality(raw_data)
        
        # 数据拆分
        split_result = self.splitter.split_data(raw_data, "data/splits")
        
        return split_result, quality_report
```

## 性能优化示例

### 批处理优化

```python
# 使用较大的批次大小提高吞吐量
config = EvaluationConfig(
    batch_size=32,  # 增大批次
    memory_optimization=True,
    device="cuda"
)
```

### 并行处理

```python
# 启用并行评估
engine = EvaluationEngine(config, max_workers=4)

# 并行评估多个模型
results = engine.evaluate_multiple_models(models_info, datasets)
```

### 内存优化

```python
# 启用内存优化选项
config = EvaluationConfig(
    memory_optimization=True,
    gradient_checkpointing=True,
    mixed_precision=True
)
```

## 错误处理示例

### 基本错误处理

```python
from evaluation.exceptions import EvaluationError, DataSplitError

try:
    split_result = splitter.split_data(dataset, "output")
except DataSplitError as e:
    print(f"数据拆分失败: {e}")
    # 处理错误
except EvaluationError as e:
    print(f"评估系统错误: {e}")
```

### 自定义错误处理

```python
class CustomErrorHandler:
    def handle_evaluation_error(self, error, context):
        # 记录错误
        self.log_error(error, context)
        
        # 尝试恢复
        if self.can_recover(error):
            return self.recover_from_error(error, context)
        else:
            raise error
```

## 调试技巧

### 启用详细日志

```python
import logging
from evaluation.logging_system import setup_logging

# 设置详细日志
setup_logging(level="DEBUG")
```

### 使用小数据集测试

```python
# 使用小数据集快速测试
test_dataset = dataset.select(range(10))
config = EvaluationConfig(num_samples=5)
```

### 分步调试

```python
# 分别测试各个组件
print("测试数据拆分...")
split_result = splitter.split_data(small_dataset, "debug")

print("测试评估引擎...")
result = engine.evaluate_model(model, tokenizer, {"task": small_dataset})

print("测试报告生成...")
report = generator.generate_evaluation_report(result)
```

## 最佳实践

### 1. 配置管理

- 使用配置文件而不是硬编码参数
- 为不同环境创建不同的配置文件
- 使用环境变量处理敏感信息

### 2. 实验管理

- 为每个实验添加描述性标签
- 定期备份实验数据
- 使用版本控制管理配置文件

### 3. 性能优化

- 根据硬件资源调整批次大小
- 使用GPU加速计算密集型任务
- 启用内存优化选项

### 4. 错误处理

- 实现完善的错误处理机制
- 记录详细的错误信息
- 提供错误恢复策略

## 常见问题

### Q: 如何处理大数据集？

A: 使用以下策略：
- 增大批次大小
- 启用内存优化
- 使用数据流处理
- 分批处理数据

### Q: 如何自定义评估指标？

A: 继承相应的基类：
```python
from evaluation.metrics_calculator import MetricsCalculator

class CustomMetricsCalculator(MetricsCalculator):
    def calculate_custom_metric(self, predictions, references):
        # 实现自定义指标
        pass
```

### Q: 如何集成到现有系统？

A: 使用模块化设计：
- 单独使用各个组件
- 通过配置文件集成
- 使用API接口集成

## 更多资源

- [用户指南](../docs/EVALUATION_USER_GUIDE.md)
- [API参考](../docs/API_REFERENCE.md)
- [配置指南](../docs/CONFIGURATION_GUIDE.md)
- [故障排除](../docs/TROUBLESHOOTING_GUIDE.md)

## 贡献

欢迎提交新的示例代码！请确保：
1. 代码可以正常运行
2. 包含适当的注释
3. 提供使用说明
4. 遵循代码风格规范

---

*示例代码持续更新中，如有问题请参考最新版本。*
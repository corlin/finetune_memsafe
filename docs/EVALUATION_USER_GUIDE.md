# 数据拆分和评估系统用户指南

本指南将帮助您使用Qwen3优化微调系统的数据拆分和模型评估功能。

## 目录

1. [快速开始](#快速开始)
2. [数据拆分](#数据拆分)
3. [模型评估](#模型评估)
4. [基准测试](#基准测试)
5. [实验跟踪](#实验跟踪)
6. [报告生成](#报告生成)
7. [配置管理](#配置管理)
8. [最佳实践](#最佳实践)
9. [故障排除](#故障排除)

## 快速开始

### 安装依赖

使用uv安装依赖（推荐）：
```bash
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
uv sync
```

### 基本使用流程

```python
from evaluation import DataSplitter, EvaluationEngine, EvaluationConfig
from datasets import Dataset

# 1. 准备数据
data = [
    {"text": "这是第一个样本", "label": "positive"},
    {"text": "这是第二个样本", "label": "negative"},
    # ... 更多数据
]
dataset = Dataset.from_list(data)

# 2. 数据拆分
splitter = DataSplitter(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
)
split_result = splitter.split_data(dataset, "data/splits")

# 3. 配置评估
config = EvaluationConfig(
    tasks=["classification"],
    metrics=["accuracy", "f1"],
    batch_size=8,
    num_samples=100
)

# 4. 运行评估
engine = EvaluationEngine(config)
result = engine.evaluate_model(
    model=your_model,
    tokenizer=your_tokenizer,
    datasets={"classification": split_result.test_dataset},
    model_name="my_model"
)

print(f"准确率: {result.metrics['accuracy']:.3f}")
```

## 数据拆分

### 基本数据拆分

```python
from evaluation import DataSplitter
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("your_dataset")["train"]

# 创建数据拆分器
splitter = DataSplitter(
    train_ratio=0.7,      # 训练集比例
    val_ratio=0.15,       # 验证集比例
    test_ratio=0.15,      # 测试集比例
    random_seed=42,       # 随机种子，确保可重现
    stratify_by="label"   # 按标签分层抽样
)

# 执行数据拆分
split_result = splitter.split_data(dataset, "output/data_splits")

# 查看拆分结果
print(f"训练集大小: {len(split_result.train_dataset)}")
print(f"验证集大小: {len(split_result.val_dataset)}")
print(f"测试集大小: {len(split_result.test_dataset)}")
```

### 高级数据拆分选项

```python
# 带质量分析的数据拆分
splitter = DataSplitter(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_by="label",
    min_samples_per_split=100,        # 每个拆分的最小样本数
    enable_quality_analysis=True,     # 启用质量分析
    remove_duplicates=True,           # 移除重复样本
    balance_classes=True              # 平衡类别分布
)

split_result = splitter.split_data(dataset, "output/data_splits")

# 查看分布分析
print("数据分布分析:")
print(split_result.distribution_analysis)
```

### 加载已拆分的数据

```python
# 加载之前保存的数据拆分
split_result = DataSplitter.load_splits("output/data_splits")

train_dataset = split_result.train_dataset
val_dataset = split_result.val_dataset
test_dataset = split_result.test_dataset
```

## 模型评估

### 基本模型评估

```python
from evaluation import EvaluationEngine, EvaluationConfig
from transformers import AutoModel, AutoTokenizer

# 加载模型和分词器
model = AutoModel.from_pretrained("your_model_path")
tokenizer = AutoTokenizer.from_pretrained("your_model_path")

# 配置评估
config = EvaluationConfig(
    tasks=["text_generation", "classification"],
    metrics=["bleu", "rouge", "accuracy", "f1"],
    batch_size=8,
    max_length=512,
    num_samples=200
)

# 创建评估引擎
engine = EvaluationEngine(config)

# 准备评估数据集
datasets = {
    "text_generation": generation_dataset,
    "classification": classification_dataset
}

# 运行评估
result = engine.evaluate_model(
    model=model,
    tokenizer=tokenizer,
    datasets=datasets,
    model_name="qwen3_finetuned"
)

# 查看结果
print("评估结果:")
for metric, value in result.metrics.items():
    print(f"  {metric}: {value:.4f}")
```

### 多模型对比评估

```python
# 准备多个模型
models_info = [
    {
        "model": model1,
        "tokenizer": tokenizer1,
        "name": "baseline_model"
    },
    {
        "model": model2,
        "tokenizer": tokenizer2,
        "name": "finetuned_model"
    }
]

# 批量评估
results = engine.evaluate_multiple_models(models_info, datasets)

# 对比结果
for result in results:
    print(f"\n模型: {result.model_name}")
    print(f"准确率: {result.metrics.get('accuracy', 'N/A')}")
    print(f"F1分数: {result.metrics.get('f1', 'N/A')}")
```

### 自定义评估任务

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

# 使用自定义任务
config = EvaluationConfig(
    tasks=["custom_task"],
    metrics=["custom_metric"],
    batch_size=4
)
```

## 基准测试

### 运行标准基准测试

```python
from evaluation import BenchmarkManager

# 创建基准管理器
benchmark_manager = BenchmarkManager()

# 查看可用基准
available_benchmarks = benchmark_manager.list_available_benchmarks()
print("可用基准:", available_benchmarks)

# 运行CLUE基准测试
clue_result = benchmark_manager.run_clue_evaluation(
    model=model,
    tokenizer=tokenizer,
    model_name="my_model"
)

print(f"CLUE总分: {clue_result.overall_score:.3f}")
print("各任务得分:")
for task, scores in clue_result.task_results.items():
    print(f"  {task}: {scores}")
```

### 运行FewCLUE少样本评估

```python
# FewCLUE评估
few_clue_result = benchmark_manager.run_few_clue_evaluation(
    model=model,
    tokenizer=tokenizer,
    model_name="my_model",
    few_shot_examples=5  # 少样本数量
)

print(f"FewCLUE得分: {few_clue_result.overall_score:.3f}")
```

### 自定义基准测试

```python
from evaluation.data_models import BenchmarkConfig

# 定义自定义基准
custom_config = BenchmarkConfig(
    name="my_custom_benchmark",
    dataset_path="path/to/custom_dataset.json",
    tasks=["custom_task1", "custom_task2"],
    evaluation_protocol="standard",
    metrics=["accuracy", "f1"]
)

# 运行自定义基准
custom_result = benchmark_manager.run_custom_benchmark(
    config=custom_config,
    model=model,
    tokenizer=tokenizer,
    model_name="my_model"
)
```

## 实验跟踪

### 基本实验跟踪

```python
from evaluation import ExperimentTracker
from evaluation.data_models import ExperimentConfig

# 创建实验跟踪器
tracker = ExperimentTracker(experiment_dir="experiments")

# 定义实验配置
experiment_config = ExperimentConfig(
    model_name="qwen3_finetuned",
    dataset_name="my_dataset",
    hyperparameters={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    },
    tags=["baseline", "v1.0"],
    description="基线模型实验"
)

# 跟踪实验
experiment_id = tracker.track_experiment(experiment_config, evaluation_result)
print(f"实验ID: {experiment_id}")
```

### 实验对比分析

```python
# 列出所有实验
experiments = tracker.list_experiments()
print(f"总实验数: {len(experiments)}")

# 按条件过滤实验
filtered_experiments = tracker.list_experiments(
    filters={"dataset_name": "my_dataset"}
)

# 对比多个实验
experiment_ids = [exp["experiment_id"] for exp in experiments[:3]]
comparison = tracker.compare_experiments(experiment_ids)

print("最佳模型:", comparison["best_model"]["model_name"])
print("性能对比:")
for model_comparison in comparison["metric_comparison"]:
    print(f"  {model_comparison}")
```

### 生成排行榜

```python
# 生成准确率排行榜
leaderboard = tracker.generate_leaderboard(metric="accuracy")

print("模型排行榜 (按准确率):")
for i, entry in enumerate(leaderboard[:5], 1):
    print(f"{i}. {entry['model_name']}: {entry['score']:.4f}")
```

### 导出实验结果

```python
# 导出为CSV
csv_path = tracker.export_results("results.csv", format="csv")

# 导出为JSON
json_path = tracker.export_results("results.json", format="json")

# 导出为Excel
excel_path = tracker.export_results("results.xlsx", format="excel")
```

## 报告生成

### 生成评估报告

```python
from evaluation import ReportGenerator

# 创建报告生成器
generator = ReportGenerator(
    output_dir="reports",
    include_plots=True,
    language="zh"
)

# 生成HTML评估报告
html_report = generator.generate_evaluation_report(
    evaluation_result, 
    format="html"
)
print(f"HTML报告: {html_report}")

# 生成JSON报告
json_report = generator.generate_evaluation_report(
    evaluation_result,
    format="json"
)
```

### 生成对比报告

```python
# 多模型对比报告
comparison_report = generator.generate_comparison_report(
    evaluation_results,  # 多个评估结果的列表
    format="html"
)

# 生成LaTeX表格（用于论文）
latex_table = generator.generate_latex_table(
    evaluation_results,
    metrics=["accuracy", "f1", "bleu"]
)
```

### 生成训练报告

```python
# 训练历史数据
training_history = {
    "epochs": [1, 2, 3, 4, 5],
    "train_loss": [2.5, 2.0, 1.8, 1.6, 1.5],
    "val_loss": [2.3, 1.9, 1.7, 1.6, 1.6],
    "train_accuracy": [0.6, 0.7, 0.75, 0.8, 0.82],
    "val_accuracy": [0.65, 0.72, 0.76, 0.78, 0.79]
}

training_config = {
    "model_name": "qwen3_finetuned",
    "dataset": "training_data",
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 5
}

# 生成训练报告
training_report = generator.generate_training_report(
    training_history,
    training_config,
    format="html"
)
```

## 配置管理

### 使用配置文件

创建配置文件 `config.yaml`:
```yaml
data_split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42
  stratify_by: "label"

evaluation:
  tasks:
    - "classification"
    - "text_generation"
  metrics:
    - "accuracy"
    - "f1"
    - "bleu"
    - "rouge"
  batch_size: 8
  max_length: 512
  num_samples: 200

experiment_tracking:
  enabled: true
  experiment_dir: "./experiments"
  auto_save: true

reporting:
  output_dir: "./reports"
  include_plots: true
  language: "zh"
```

加载和使用配置：
```python
from evaluation.config_manager import ConfigManager

# 加载配置
config_manager = ConfigManager()
config = config_manager.load_config("config.yaml")

# 创建组件
splitter = DataSplitter(**config["data_split"])
eval_config = config_manager.create_evaluation_config(config)
engine = EvaluationEngine(eval_config)
```

### 环境变量配置

在配置文件中使用环境变量：
```yaml
evaluation:
  batch_size: ${BATCH_SIZE:8}  # 默认值为8
  model_path: ${MODEL_PATH}
  output_dir: ${OUTPUT_DIR:./outputs}
```

加载时替换环境变量：
```python
config = config_manager.load_config(
    "config.yaml", 
    substitute_env_vars=True
)
```

### 配置验证

```python
# 验证配置
is_valid = config_manager.validate_config(config)
if not is_valid:
    print("配置验证失败")
    
# 获取验证错误
errors = config_manager.get_validation_errors(config)
for error in errors:
    print(f"错误: {error}")
```

## 最佳实践

### 数据拆分最佳实践

1. **确保数据分布一致性**
   ```python
   # 使用分层抽样
   splitter = DataSplitter(stratify_by="label")
   
   # 检查分布一致性
   analysis = split_result.distribution_analysis
   if analysis["consistency_score"] < 0.8:
       print("警告: 数据分布不一致")
   ```

2. **处理不平衡数据集**
   ```python
   splitter = DataSplitter(
       balance_classes=True,
       min_samples_per_class=10
   )
   ```

3. **设置合适的随机种子**
   ```python
   # 使用固定种子确保可重现性
   splitter = DataSplitter(random_seed=42)
   ```

### 评估最佳实践

1. **选择合适的评估指标**
   ```python
   # 分类任务
   classification_metrics = ["accuracy", "precision", "recall", "f1"]
   
   # 生成任务
   generation_metrics = ["bleu", "rouge", "meteor", "bertscore"]
   
   # 语义任务
   semantic_metrics = ["semantic_similarity", "coherence"]
   ```

2. **使用适当的批次大小**
   ```python
   # 根据GPU内存调整批次大小
   config = EvaluationConfig(
       batch_size=8 if gpu_memory < 8 else 16,
       max_length=512
   )
   ```

3. **启用并行评估**
   ```python
   engine = EvaluationEngine(config, max_workers=4)
   ```

### 实验管理最佳实践

1. **使用描述性的实验名称和标签**
   ```python
   experiment_config = ExperimentConfig(
       model_name="qwen3_lr001_bs32_ep10",
       tags=["baseline", "learning_rate_0.001", "batch_size_32"],
       description="基线模型，学习率0.001，批次大小32"
   )
   ```

2. **记录完整的超参数**
   ```python
   hyperparameters = {
       "learning_rate": 0.001,
       "batch_size": 32,
       "epochs": 10,
       "optimizer": "AdamW",
       "weight_decay": 0.01,
       "warmup_steps": 1000
   }
   ```

3. **定期备份实验数据**
   ```python
   # 备份实验数据
   backup_path = tracker.backup_experiments("backup.json")
   ```

## 故障排除

### 常见问题

#### 1. 内存不足错误
```
OutOfMemoryError: CUDA out of memory
```

**解决方案:**
- 减少批次大小
- 使用梯度累积
- 启用内存优化

```python
config = EvaluationConfig(
    batch_size=2,  # 减少批次大小
    memory_optimization=True
)
```

#### 2. 数据拆分失败
```
ValueError: 数据集太小，无法按指定比例拆分
```

**解决方案:**
- 调整最小样本数要求
- 修改拆分比例

```python
splitter = DataSplitter(
    min_samples_per_split=1,  # 降低最小样本数
    train_ratio=0.8,          # 调整比例
    val_ratio=0.1,
    test_ratio=0.1
)
```

#### 3. 评估指标计算错误
```
ValueError: 预测文本和参考文本数量不匹配
```

**解决方案:**
- 检查数据格式
- 确保预测和参考数量一致

```python
# 检查数据长度
assert len(predictions) == len(references)

# 处理缺失数据
predictions = [pred if pred else "" for pred in predictions]
```

#### 4. 配置文件错误
```
yaml.YAMLError: 配置文件格式错误
```

**解决方案:**
- 检查YAML语法
- 使用配置验证

```python
try:
    config = config_manager.load_config("config.yaml")
except yaml.YAMLError as e:
    print(f"配置文件错误: {e}")
```

### 性能优化

#### 1. 加速评估
```python
# 使用并行处理
engine = EvaluationEngine(config, max_workers=4)

# 减少评估样本数
config = EvaluationConfig(num_samples=100)  # 而不是全部数据

# 使用GPU加速
config = EvaluationConfig(device="cuda")
```

#### 2. 内存优化
```python
# 启用内存优化
config = EvaluationConfig(memory_optimization=True)

# 使用较小的批次
config = EvaluationConfig(batch_size=4)

# 清理缓存
import torch
torch.cuda.empty_cache()
```

#### 3. 磁盘空间优化
```python
# 定期清理实验数据
tracker.cleanup_old_experiments(keep_last=50)

# 压缩报告
generator = ReportGenerator(compress_reports=True)
```

### 调试技巧

#### 1. 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或使用评估系统的日志配置
from evaluation.logging_system import setup_logging
setup_logging(level="DEBUG")
```

#### 2. 使用小数据集测试
```python
# 使用小数据集快速测试
test_dataset = dataset.select(range(10))
```

#### 3. 分步骤调试
```python
# 分别测试各个组件
splitter = DataSplitter()
split_result = splitter.split_data(small_dataset, "debug_splits")

engine = EvaluationEngine(simple_config)
result = engine.evaluate_model(model, tokenizer, {"task": small_dataset})
```

## 更多资源

- [API参考文档](API_REFERENCE.md)
- [配置参考](CONFIGURATION_GUIDE.md)
- [部署指南](DEPLOYMENT_GUIDE.md)
- [故障排除指南](TROUBLESHOOTING_GUIDE.md)
- [示例代码](../examples/)

## 支持

如果您遇到问题或有建议，请：
1. 查看[故障排除指南](TROUBLESHOOTING_GUIDE.md)
2. 搜索已知问题
3. 提交问题报告

---

*本指南持续更新中，如有疑问请参考最新版本。*
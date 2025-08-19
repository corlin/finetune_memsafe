# 增强训练Pipeline使用指南

基于现有main.py程序的增强版本，集成了数据拆分、模型训练和性能评估的完整流程。

## 功能特性

### 🔄 数据拆分
- 科学的数据拆分（训练集/验证集/测试集）
- 支持分层抽样和随机拆分
- 数据质量分析和分布一致性检查
- 自动生成数据拆分报告

### 🚀 增强训练
- 基于现有训练引擎的完整训练流程
- 训练过程中的验证集评估
- 早停机制和验证指标跟踪
- 内存优化和错误恢复

### 📊 全面评估
- 多种评估指标（BLEU、ROUGE、准确率等）
- 效率分析（延迟、吞吐量、内存使用）
- 质量分析（流畅性、连贯性、相关性）
- 支持测试集和验证集同时评估

### 🔬 实验跟踪
- 自动记录实验配置和结果
- 实验进度跟踪和状态管理
- 实验历史和对比分析
- 可重现的实验环境

### 📈 报告生成
- HTML、JSON、CSV多格式报告
- 数据拆分分析报告
- 训练过程报告
- 综合评估报告
- 可视化图表和统计信息

### 🛡️ 错误处理
- 智能错误恢复机制
- 分类错误处理策略
- 回退模式支持
- 详细错误报告和建议

## 快速开始

### 1. 基本使用

```bash
# 使用默认配置运行
python enhanced_main.py

# 指定数据目录
python enhanced_main.py --data-dir data/my_data

# 自定义训练参数
python enhanced_main.py --num-epochs 3 --batch-size 8 --learning-rate 1e-4
```

### 2. 使用配置文件

```bash
# 使用完整配置文件
python enhanced_main.py --config enhanced_config_example.yaml

# 使用简化配置文件
python enhanced_main.py --config enhanced_config_simple.yaml
```

### 3. 验证配置

```bash
# 验证配置文件
python validate_config.py enhanced_config_example.yaml

# 创建示例配置
python validate_config.py --create-sample my_config.yaml

# 列出可用示例
python validate_config.py --list-examples
```

## 配置说明

### 基本配置结构

```yaml
# 模型配置
model:
  name: "Qwen/Qwen3-4B-Thinking-2507"
  output_dir: "./enhanced-qwen3-finetuned"

# 数据配置
data:
  data_dir: "data/raw"
  enable_splitting: true
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

# 训练配置
training:
  batch_size: 4
  num_epochs: 5
  learning_rate: 5e-5
  enable_validation: true

# 评估配置
evaluation:
  enable_comprehensive: true
  tasks: ["text_generation"]
  metrics: ["bleu", "rouge", "accuracy"]

# 实验跟踪
experiment:
  enable_tracking: true
  tags: ["enhanced", "pipeline"]

# 报告配置
reports:
  formats: ["html", "json"]
```

### 关键配置项说明

#### 数据拆分配置
- `enable_splitting`: 是否启用数据拆分
- `train_ratio/val_ratio/test_ratio`: 数据拆分比例
- `stratify_by`: 分层抽样字段（可选）
- `split_seed`: 随机种子，确保可重现性

#### 训练配置
- `enable_validation`: 是否在训练中进行验证
- `validation_steps`: 验证评估间隔
- `enable_early_stopping`: 是否启用早停

#### 评估配置
- `enable_comprehensive`: 是否启用全面评估
- `tasks`: 评估任务列表
- `metrics`: 评估指标列表
- `enable_efficiency`: 是否测量效率指标
- `enable_quality`: 是否进行质量分析

## 使用示例

### 示例1：基本训练流程

```python
from enhanced_config import EnhancedApplicationConfig
from enhanced_main import EnhancedQwenFineTuningApplication

# 创建配置
config = EnhancedApplicationConfig(
    model_name="Qwen/Qwen3-4B-Thinking-2507",
    data_dir="data/raw",
    output_dir="./my_model",
    num_epochs=3,
    enable_data_splitting=True,
    enable_comprehensive_evaluation=True
)

# 运行训练
app = EnhancedQwenFineTuningApplication(config)
success = app.run_enhanced_pipeline()
```

### 示例2：自定义数据拆分

```python
config = EnhancedApplicationConfig(
    # 基本配置
    model_name="Qwen/Qwen3-4B-Thinking-2507",
    data_dir="data/raw",
    
    # 自定义数据拆分
    enable_data_splitting=True,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    stratify_by="category",  # 按类别分层
    
    # 训练配置
    num_epochs=5,
    batch_size=8
)
```

### 示例3：评估重点配置

```python
config = EnhancedApplicationConfig(
    # 基本配置
    model_name="Qwen/Qwen3-4B-Thinking-2507",
    data_dir="data/raw",
    
    # 重点评估配置
    enable_comprehensive_evaluation=True,
    evaluation_tasks=["text_generation", "classification"],
    evaluation_metrics=["bleu", "rouge", "accuracy", "f1"],
    enable_efficiency_metrics=True,
    enable_quality_analysis=True,
    
    # 报告配置
    report_formats=["html", "json", "csv"],
    enable_visualization=True
)
```

## 输出文件说明

### 目录结构

```
enhanced-qwen3-finetuned/
├── adapter_config.json          # LoRA适配器配置
├── adapter_model.safetensors    # LoRA权重
├── comprehensive_evaluation.json # 全面评估结果
├── validation_history.json     # 验证集历史
├── validation_evaluation.json  # 验证集详细评估
└── error_report.json          # 错误报告（如有）

data/splits/
├── train/                      # 训练集
├── val/                        # 验证集
├── test/                       # 测试集
└── split_info.json            # 拆分信息

reports/
├── evaluation_report_*.html    # HTML评估报告
├── evaluation_report_*.json    # JSON评估报告
├── data_split_analysis.html    # 数据拆分分析
├── training_process.json       # 训练过程报告
├── comprehensive_report.json   # 综合报告
└── report_index.json          # 报告索引

experiments/
└── [experiment_id]/
    ├── experiment_summary.json # 实验摘要
    └── ...                     # 其他实验文件
```

### 关键输出文件

1. **comprehensive_evaluation.json**: 完整的评估结果
2. **data_split_analysis.html**: 数据拆分可视化分析
3. **validation_history.json**: 训练过程中的验证指标
4. **comprehensive_report.json**: 整个pipeline的综合报告
5. **error_report.json**: 错误和恢复记录

## 命令行参数

### 基本参数
- `--model-name`: 模型名称
- `--output-dir`: 输出目录
- `--data-dir`: 数据目录
- `--config`: 配置文件路径

### 数据拆分参数
- `--enable-data-splitting`: 启用数据拆分
- `--train-ratio`: 训练集比例
- `--val-ratio`: 验证集比例
- `--test-ratio`: 测试集比例
- `--stratify-by`: 分层字段

### 训练参数
- `--num-epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--learning-rate`: 学习率
- `--max-memory-gb`: 最大内存限制

### 评估参数
- `--enable-comprehensive-evaluation`: 启用全面评估
- `--evaluation-tasks`: 评估任务列表
- `--evaluation-metrics`: 评估指标列表

### 实验跟踪参数
- `--enable-experiment-tracking`: 启用实验跟踪
- `--experiment-name`: 实验名称
- `--experiment-tags`: 实验标签

### 报告参数
- `--report-formats`: 报告格式列表
- `--enable-visualization`: 启用可视化

## 最佳实践

### 1. 数据准备
- 确保数据格式正确（支持QA格式）
- 数据量建议至少1000条以上
- 如需分层抽样，确保各类别样本充足

### 2. 配置调优
- 根据GPU内存调整`batch_size`和`max_memory_gb`
- 使用验证集监控训练过程
- 启用早停避免过拟合

### 3. 评估策略
- 使用多种指标全面评估模型
- 关注效率指标，特别是推理延迟
- 定期检查质量分数

### 4. 实验管理
- 为每个实验设置有意义的名称和标签
- 保存重要实验的配置文件
- 定期清理实验目录

### 5. 错误处理
- 启用回退模式提高鲁棒性
- 查看错误报告了解问题原因
- 根据恢复建议调整配置

## 故障排除

### 常见问题

1. **内存不足**
   - 减少`batch_size`
   - 增加`gradient_accumulation_steps`
   - 降低`max_sequence_length`

2. **数据拆分失败**
   - 检查数据格式
   - 确保数据量充足
   - 调整拆分比例

3. **评估失败**
   - 检查模型是否正确保存
   - 减少评估样本数量
   - 跳过失败的指标

4. **训练不收敛**
   - 降低学习率
   - 检查数据质量
   - 启用梯度裁剪

### 日志分析

查看日志文件了解详细执行过程：
- `logs/application.log`: 主要日志
- `[output_dir]/logs/`: TensorBoard日志
- `error_report.json`: 错误详情

### 获取帮助

```bash
# 查看帮助信息
python enhanced_main.py --help

# 验证配置
python validate_config.py --help

# 查看示例配置
python validate_config.py --list-examples
```

## 与原版main.py的区别

| 功能 | 原版main.py | 增强版enhanced_main.py |
|------|-------------|----------------------|
| 数据处理 | 直接使用全部数据 | 科学数据拆分 |
| 训练监控 | 基本训练日志 | 验证集评估+早停 |
| 模型评估 | 简单推理测试 | 全面多指标评估 |
| 实验管理 | 无 | 完整实验跟踪 |
| 报告生成 | 基本日志 | 多格式详细报告 |
| 错误处理 | 基本异常处理 | 智能错误恢复 |
| 配置管理 | 命令行参数 | YAML配置文件 |

增强版本完全兼容原版的所有功能，同时提供了更强大的数据管理、评估和实验跟踪能力。
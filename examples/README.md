# 示例配置和数据集

本目录包含了Qwen3优化微调系统的示例配置文件、数据集和性能基准测试脚本。

## 目录结构

```
examples/
├── configs/                    # 配置文件示例
│   ├── low_memory_8gb.json    # 低内存8GB配置
│   ├── standard_12gb.json     # 标准12GB配置
│   ├── high_performance_16gb.json # 高性能16GB配置
│   └── quick_test.json        # 快速测试配置
├── datasets/                   # 示例数据集
│   ├── programming_qa.md      # 编程问答数据集
│   ├── ai_ml_qa.md           # AI/ML问答数据集
│   └── general_knowledge_qa.md # 通用知识问答数据集
├── benchmark_performance.py    # 性能基准测试脚本
└── README.md                  # 本文件
```

## 配置文件说明

### 1. 低内存8GB配置 (`low_memory_8gb.json`)

适用于显存较少的GPU（如RTX 3070, RTX 4060 Ti等）：

- **最大内存**: 7.5GB
- **批次大小**: 1
- **梯度累积**: 64步
- **序列长度**: 128
- **LoRA参数**: r=4, alpha=8

**使用方法**:
```bash
uv run python main.py --config examples/configs/low_memory_8gb.json
```

### 2. 标准12GB配置 (`standard_12gb.json`)

适用于中等显存的GPU（如RTX 3080 Ti, RTX 4070 Ti等）：

- **最大内存**: 11.5GB
- **批次大小**: 2
- **梯度累积**: 32步
- **序列长度**: 256
- **LoRA参数**: r=6, alpha=12

**使用方法**:
```bash
uv run python main.py --config examples/configs/standard_12gb.json
```

### 3. 高性能16GB配置 (`high_performance_16gb.json`)

适用于高显存的GPU（如RTX 4090, RTX 3090等）：

- **最大内存**: 15.0GB
- **批次大小**: 4
- **梯度累积**: 16步
- **序列长度**: 512
- **LoRA参数**: r=8, alpha=16

**使用方法**:
```bash
uv run python main.py --config examples/configs/high_performance_16gb.json
```

### 4. 快速测试配置 (`quick_test.json`)

适用于快速验证和测试：

- **训练轮数**: 5轮
- **序列长度**: 128
- **学习率**: 1e-4（较高，快速收敛）
- **LoRA参数**: r=4, alpha=8（较小，快速训练）

**使用方法**:
```bash
uv run python main.py --config examples/configs/quick_test.json
```

## 示例数据集

### 1. 编程问答数据集 (`programming_qa.md`)

包含Python编程、数据结构与算法相关的问答对，适用于：
- 编程助手训练
- 技术问答系统
- 代码解释和教学

**数据格式**:
```markdown
### Q1: 如何在Python中创建虚拟环境？
A1: 可以使用venv模块创建虚拟环境：`python -m venv myenv`...
```

### 2. AI/ML问答数据集 (`ai_ml_qa.md`)

包含机器学习、深度学习、自然语言处理相关的问答对，适用于：
- AI教育助手
- 技术咨询系统
- 学术问答

**数据格式**:
```markdown
Q1: 什么是过拟合？如何避免？
A1: 过拟合是指模型在训练数据上表现很好，但在新数据上表现差的现象...
```

### 3. 通用知识问答数据集 (`general_knowledge_qa.md`)

包含科学技术、历史文化、地理环境等通用知识问答对，适用于：
- 通用问答系统
- 教育助手
- 知识问答机器人

**数据格式**:
```markdown
### Q1: 什么是量子计算？
A1: 量子计算利用量子力学原理进行信息处理，使用量子比特(qubit)作为基本单位...
```

## 性能基准测试

### 运行基准测试

```bash
# 运行所有配置的基准测试
uv run python examples/benchmark_performance.py

# 运行单个配置的基准测试
uv run python examples/benchmark_performance.py --config examples/configs/quick_test.json --name "快速测试"

# 指定输出文件
uv run python examples/benchmark_performance.py --output my_benchmark.json --report my_report.md
```

### 基准测试内容

基准测试会评估以下指标：

1. **训练时间**: 完成训练所需的总时间
2. **内存使用**: 系统内存和GPU内存的峰值使用量
3. **训练指标**: 最终训练损失、收敛情况等
4. **推理测试**: 训练后模型的推理能力验证
5. **系统兼容性**: 不同硬件配置下的表现

### 基准测试报告

测试完成后会生成详细的Markdown报告，包含：

- 系统信息汇总
- 各配置的性能对比表格
- 详细的测试结果
- 针对您硬件的优化建议

## 自定义配置

### 创建自定义配置

1. 复制现有配置文件：
```bash
cp examples/configs/standard_12gb.json my_config.json
```

2. 根据需要修改参数：
```json
{
  "model_name": "Qwen/Qwen3-4B-Thinking-2507",
  "max_memory_gb": 10.0,
  "batch_size": 2,
  "num_epochs": 50,
  "learning_rate": 3e-5
}
```

3. 使用自定义配置：
```bash
uv run python main.py --config my_config.json
```

### 配置参数说明

#### 内存相关
- `max_memory_gb`: GPU内存限制
- `batch_size`: 批次大小
- `gradient_accumulation_steps`: 梯度累积步数
- `max_sequence_length`: 最大序列长度

#### 训练相关
- `learning_rate`: 学习率
- `num_epochs`: 训练轮数
- `warmup_ratio`: 预热比例
- `weight_decay`: 权重衰减

#### LoRA相关
- `lora_r`: LoRA秩
- `lora_alpha`: LoRA缩放因子
- `lora_dropout`: LoRA dropout率

## 自定义数据集

### 数据格式要求

支持两种QA格式：

1. **简单格式**:
```markdown
Q1: 问题内容？
A1: 答案内容。

Q2: 问题内容？
A2: 答案内容。
```

2. **标题格式**:
```markdown
### Q1: 问题内容？
A1: 答案内容。

### Q2: 问题内容？
A2: 答案内容。
```

### 数据准备建议

1. **数据质量**: 确保问答对准确、相关、格式一致
2. **数据量**: 建议至少100个问答对，更多数据通常效果更好
3. **数据多样性**: 包含不同类型和难度的问题
4. **编码格式**: 使用UTF-8编码保存文件

### 使用自定义数据集

1. 将数据文件放在 `data/raw/` 目录下
2. 确保文件名以 `.md` 结尾
3. 运行训练时系统会自动加载所有数据文件

## 最佳实践

### 选择合适的配置

1. **GPU内存 < 10GB**: 使用 `low_memory_8gb.json`
2. **GPU内存 10-14GB**: 使用 `standard_12gb.json`
3. **GPU内存 > 14GB**: 使用 `high_performance_16gb.json`
4. **快速测试**: 使用 `quick_test.json`

### 优化训练效果

1. **数据质量优先**: 高质量的小数据集比低质量的大数据集效果更好
2. **逐步调优**: 先用快速配置验证，再用完整配置训练
3. **监控内存**: 使用TensorBoard监控训练过程
4. **多次实验**: 尝试不同的学习率和LoRA参数

### 故障排除

1. **内存不足**: 减少批次大小或序列长度
2. **训练慢**: 增加批次大小或减少梯度累积步数
3. **效果差**: 增加训练轮数或调整学习率
4. **数据错误**: 检查数据格式和编码

## 技术支持

如果在使用示例配置和数据集时遇到问题，请：

1. 查看 [故障排除文档](../docs/TROUBLESHOOTING.md)
2. 运行基准测试诊断系统性能
3. 检查系统日志和错误信息
4. 尝试使用快速测试配置验证环境
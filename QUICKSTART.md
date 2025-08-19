# 快速开始指南

## 5分钟快速体验增强训练Pipeline

### 步骤1: 环境准备

确保已安装必要的依赖：
```bash
pip install torch transformers datasets peft accelerate
pip install pyyaml  # 用于配置文件支持
```

### 步骤2: 准备数据

将训练数据放在 `data/raw` 目录下，支持以下格式：

**QA格式 (推荐)**
```json
[
  {
    "question": "什么是人工智能？",
    "answer": "人工智能是计算机科学的一个分支..."
  }
]
```

**文本格式**
```json
[
  {
    "text": "这是一段训练文本...",
    "target": "这是对应的目标文本..."
  }
]
```

### 步骤3: 快速运行

#### 方法1: 使用默认配置
```bash
python enhanced_main.py
```

#### 方法2: 使用简化配置文件
```bash
python enhanced_main.py --config enhanced_config_simple.yaml
```

#### 方法3: 自定义参数
```bash
python enhanced_main.py \
  --data-dir data/raw \
  --num-epochs 3 \
  --batch-size 4 \
  --enable-data-splitting \
  --enable-comprehensive-evaluation
```

### 步骤4: 查看结果

训练完成后，检查以下输出：

1. **模型文件**: `enhanced-qwen3-finetuned/`
2. **评估报告**: `reports/evaluation_report_*.html`
3. **数据分析**: `reports/data_split_analysis.html`
4. **实验记录**: `experiments/[experiment_id]/`

### 步骤5: 验证配置（可选）

```bash
# 验证配置文件
python validate_config.py enhanced_config_simple.yaml

# 查看示例配置
python validate_config.py --list-examples

# 创建自定义配置
python validate_config.py --create-sample my_config.yaml
```

## 常用配置模板

### 快速测试配置
```yaml
model:
  name: "Qwen/Qwen3-4B-Thinking-2507"
  output_dir: "./quick_test"

data:
  data_dir: "data/raw"
  enable_splitting: true

training:
  batch_size: 2
  num_epochs: 1
  learning_rate: 5e-5

evaluation:
  enable_comprehensive: true
  tasks: ["text_generation"]

experiment:
  enable_tracking: true
  tags: ["quick_test"]
```

### 生产环境配置
```yaml
model:
  name: "Qwen/Qwen3-4B-Thinking-2507"
  output_dir: "./production_model"

data:
  data_dir: "data/raw"
  enable_splitting: true
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

training:
  batch_size: 8
  num_epochs: 10
  learning_rate: 3e-5
  enable_validation: true
  enable_early_stopping: true

evaluation:
  enable_comprehensive: true
  tasks: ["text_generation"]
  metrics: ["bleu", "rouge", "accuracy"]
  enable_efficiency: true
  enable_quality: true

experiment:
  enable_tracking: true
  name: "production_v1"
  tags: ["production", "v1.0"]

reports:
  formats: ["html", "json", "csv"]
  enable_visualization: true
```

## 故障排除

### 问题1: 内存不足
```bash
# 解决方案：减少批次大小
python enhanced_main.py --batch-size 2 --max-memory-gb 8
```

### 问题2: 数据格式错误
```bash
# 解决方案：检查数据格式，或跳过数据拆分
python enhanced_main.py --no-enable-data-splitting
```

### 问题3: 评估失败
```bash
# 解决方案：跳过评估或使用简化评估
python enhanced_main.py --no-enable-comprehensive-evaluation
```

## 下一步

1. 📖 阅读完整文档: `README_enhanced.md`
2. 🔧 运行示例代码: `python example_usage.py`
3. ⚙️ 自定义配置文件
4. 📊 分析生成的报告
5. 🔬 使用实验跟踪功能

## 获取帮助

```bash
# 查看所有可用参数
python enhanced_main.py --help

# 验证配置文件
python validate_config.py --help

# 运行使用示例
python example_usage.py
```

---

**提示**: 首次运行可能需要下载模型，请确保网络连接正常。如果网络受限，可以预先下载模型到本地。
# Qwen3优化微调系统

一个内存优化的Qwen3模型微调系统，支持在13GB或更少GPU内存的硬件上进行高效训练。

## 特性

- **内存优化**: 使用4位量化、LoRA适配器和梯度检查点，将GPU内存使用限制在13GB以下
- **自动化流程**: 完整的端到端微调流程，从数据加载到模型测试
- **错误恢复**: 智能的错误处理和恢复机制
- **全面监控**: TensorBoard集成和详细的日志记录
- **环境管理**: 支持uv环境管理和自动依赖安装

## 系统要求

- Python 3.9+ (推荐3.12)
- CUDA兼容的GPU (推荐12GB+显存，支持CUDA 12.4)
- 15GB+可用磁盘空间
- 16GB+系统内存
- Linux/macOS (推荐，Windows部分功能受限)

## 安装

### 使用uv (推荐)

```bash
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目
git clone <repository-url>
cd qwen3-finetuning

# 初始化项目
uv sync
```

### 使用pip

```bash
# 克隆项目
git clone <repository-url>
cd qwen3-finetuning

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 针对Qwen3-4B-Thinking-2507的快速启动

```bash
# 使用专用启动脚本（推荐）
uv run python start_qwen3_thinking.py

# 或使用配置文件
uv run main.py --config qwen3_4b_thinking_config.json
```

### 基本使用

```bash
# 使用默认配置运行
python main.py

# 或使用uv
uv run main.py
```

### 自定义配置

```bash
# 指定模型和输出目录
python main.py --model-name "Qwen/Qwen3-4B-Thinking-2507" --output-dir "./my-finetuned-model"

# 调整内存和训练参数
python main.py --max-memory-gb 10 --batch-size 2 --num-epochs 50

# 启用自动依赖安装
python main.py --auto-install-deps
```

### 使用配置文件

创建配置文件 `config.json`:

```json
{
  "model_name": "Qwen/Qwen3-4B-Thinking-2507",
  "output_dir": "./qwen3-finetuned",
  "max_memory_gb": 13.0,
  "batch_size": 4,
  "gradient_accumulation_steps": 16,
  "learning_rate": 5e-5,
  "num_epochs": 100,
  "data_dir": "data/raw",
  "auto_install_deps": true
}
```

然后运行:

```bash
python main.py --config config.json
```

## 数据准备

将训练数据放在 `data/raw/` 目录下，支持以下格式的markdown文件:

### 格式1: 简单QA格式
```markdown
Q1: 什么是机器学习？
A1: 机器学习是人工智能的一个分支...

Q2: 如何优化模型性能？
A2: 可以通过以下方法优化模型性能...
```

### 格式2: 标题QA格式
```markdown
### Q1: 什么是机器学习？
A1: 机器学习是人工智能的一个分支...

### Q2: 如何优化模型性能？
A2: 可以通过以下方法优化模型性能...
```

如果没有提供数据文件，系统会自动使用内置的示例数据。

## 命令行参数

### 模型配置
- `--model-name`: 要微调的模型名称 (默认: "Qwen/Qwen3-4B-Thinking-2507")
- `--output-dir`: 输出目录 (默认: "./qwen3-finetuned")

### 内存配置
- `--max-memory-gb`: 最大GPU内存限制(GB) (默认: 13.0)

### 训练配置
- `--batch-size`: 批次大小 (默认: 4)
- `--gradient-accumulation-steps`: 梯度累积步数 (默认: 16)
- `--learning-rate`: 学习率 (默认: 5e-5)
- `--num-epochs`: 训练轮数 (默认: 100)
- `--max-sequence-length`: 最大序列长度 (默认: 256)

### LoRA配置
- `--lora-r`: LoRA rank (默认: 6)
- `--lora-alpha`: LoRA alpha (默认: 12)
- `--lora-dropout`: LoRA dropout (默认: 0.1)

### 系统配置
- `--data-dir`: 训练数据目录 (默认: "data/raw")
- `--log-dir`: 日志目录 (默认: "./logs")
- `--no-tensorboard`: 禁用TensorBoard
- `--no-inference-test`: 禁用推理测试
- `--no-verify-environment`: 跳过环境验证
- `--auto-install-deps`: 自动安装缺少的依赖

## 输出文件

训练完成后，会在输出目录生成以下文件:

```
qwen3-finetuned/
├── adapter_config.json          # LoRA适配器配置
├── adapter_model.safetensors    # LoRA适配器权重
├── config.json                  # 模型配置
├── tokenizer.json              # 分词器文件
├── tokenizer_config.json       # 分词器配置
├── final_application_report.json # 最终训练报告
└── logs/                       # 训练日志
    ├── structured/             # 结构化日志
    ├── tensorboard/           # TensorBoard日志
    └── application.log        # 应用程序日志
```

## 监控训练

### TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir ./qwen3-finetuned/logs/tensorboard

# 在浏览器中访问
http://localhost:6006
```

### 日志文件

- `logs/application.log`: 主应用程序日志
- `logs/structured/qwen3_training_structured.jsonl`: 结构化训练日志
- `qwen3-finetuned/final_application_report.json`: 最终训练报告

## 推理测试

训练完成后，系统会自动进行推理测试。你也可以手动测试:

```python
from src.inference_tester import InferenceTester

# 创建推理测试器
tester = InferenceTester()

# 加载微调模型
tester.load_finetuned_model("./qwen3-finetuned", "Qwen/Qwen3-4B-Thinking-2507")

# 测试推理
response = tester.test_inference("请解释什么是机器学习？")
print(response)
```

## 故障排除

### 内存不足错误

如果遇到GPU内存不足错误，尝试以下解决方案:

1. 减少批次大小: `--batch-size 2`
2. 增加梯度累积: `--gradient-accumulation-steps 32`
3. 减少序列长度: `--max-sequence-length 128`
4. 降低内存限制: `--max-memory-gb 10`

### 依赖安装问题

1. 启用自动安装: `--auto-install-deps`
2. 手动安装PyTorch: 
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. 安装其他依赖:
   ```bash
   pip install transformers peft datasets tensorboard bitsandbytes accelerate
   ```

### 数据加载问题

1. 检查数据目录是否存在: `ls data/raw/`
2. 验证数据文件格式是否正确
3. 检查文件编码是否为UTF-8

### CUDA问题

1. 检查CUDA安装: `nvidia-smi`
2. 验证PyTorch CUDA支持: 
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

## 高级配置

### 自定义LoRA参数

对于不同大小的模型，可能需要调整LoRA参数:

- 小模型 (< 7B): `--lora-r 4 --lora-alpha 8`
- 中等模型 (7B-13B): `--lora-r 6 --lora-alpha 12` (默认)
- 大模型 (> 13B): `--lora-r 8 --lora-alpha 16`

### 内存优化策略

1. **激进内存优化** (适用于低显存):
   ```bash
   python main.py --max-memory-gb 8 --batch-size 1 --gradient-accumulation-steps 64
   ```

2. **平衡配置** (推荐):
   ```bash
   python main.py --max-memory-gb 13 --batch-size 4 --gradient-accumulation-steps 16
   ```

3. **高性能配置** (适用于高显存):
   ```bash
   python main.py --max-memory-gb 20 --batch-size 8 --gradient-accumulation-steps 8
   ```

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License
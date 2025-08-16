# 配置指南

本文档详细介绍了Qwen3优化微调系统的配置选项，适用于不同硬件设置和使用场景。

## 配置文件格式

系统支持JSON格式的配置文件，包含以下主要部分：

```json
{
  "model_name": "Qwen/Qwen3-4B-Thinking-2507",
  "output_dir": "./qwen3-finetuned",
  "max_memory_gb": 13.0,
  "batch_size": 4,
  "gradient_accumulation_steps": 16,
  "learning_rate": 5e-5,
  "num_epochs": 100,
  "max_sequence_length": 256,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01,
  "lora_r": 6,
  "lora_alpha": 12,
  "lora_dropout": 0.1,
  "data_dir": "data/raw",
  "log_dir": "./logs",
  "enable_tensorboard": true,
  "enable_inference_test": true,
  "verify_environment": true,
  "auto_install_deps": false
}
```

## 硬件特定配置

### 低显存GPU (8GB)

适用于RTX 3070, RTX 4060 Ti等显卡：

```json
{
  "model_name": "Qwen/Qwen3-4B-Thinking-2507",
  "max_memory_gb": 7.5,
  "batch_size": 1,
  "gradient_accumulation_steps": 64,
  "max_sequence_length": 128,
  "lora_r": 4,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "learning_rate": 3e-5,
  "num_epochs": 50,
  "warmup_ratio": 0.05,
  "weight_decay": 0.005
}
```

### 中等显存GPU (12GB)

适用于RTX 3080 Ti, RTX 4070 Ti等显卡：

```json
{
  "model_name": "Qwen/Qwen3-4B-Thinking-2507",
  "max_memory_gb": 11.5,
  "batch_size": 2,
  "gradient_accumulation_steps": 32,
  "max_sequence_length": 256,
  "lora_r": 6,
  "lora_alpha": 12,
  "lora_dropout": 0.1,
  "learning_rate": 5e-5,
  "num_epochs": 75,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01
}
```

### 高显存GPU (16GB+)

适用于RTX 4090, RTX 3090等显卡：

```json
{
  "model_name": "Qwen/Qwen3-4B-Thinking-2507",
  "max_memory_gb": 15.0,
  "batch_size": 4,
  "gradient_accumulation_steps": 16,
  "max_sequence_length": 512,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.1,
  "learning_rate": 5e-5,
  "num_epochs": 100,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01
}
```

### 专业级GPU (24GB+)

适用于RTX 4090, A100, V100等显卡：

```json
{
  "model_name": "Qwen/Qwen3-4B-Thinking-2507",
  "max_memory_gb": 22.0,
  "batch_size": 8,
  "gradient_accumulation_steps": 8,
  "max_sequence_length": 1024,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "learning_rate": 5e-5,
  "num_epochs": 100,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01
}
```

## 模型特定配置

### Qwen3-4B-Thinking-2507 (推荐)

```json
{
  "model_name": "Qwen/Qwen3-4B-Thinking-2507",
  "max_memory_gb": 13.0,
  "batch_size": 4,
  "gradient_accumulation_steps": 16,
  "learning_rate": 5e-5,
  "lora_r": 6,
  "lora_alpha": 12,
  "max_sequence_length": 256
}
```

### Qwen3-7B

```json
{
  "model_name": "Qwen/Qwen3-7B",
  "max_memory_gb": 15.0,
  "batch_size": 2,
  "gradient_accumulation_steps": 32,
  "learning_rate": 3e-5,
  "lora_r": 8,
  "lora_alpha": 16,
  "max_sequence_length": 512
}
```

### Qwen3-14B

```json
{
  "model_name": "Qwen/Qwen3-14B",
  "max_memory_gb": 20.0,
  "batch_size": 1,
  "gradient_accumulation_steps": 64,
  "learning_rate": 2e-5,
  "lora_r": 16,
  "lora_alpha": 32,
  "max_sequence_length": 256
}
```

## 训练策略配置

### 快速原型开发

适用于快速测试和验证：

```json
{
  "num_epochs": 10,
  "learning_rate": 1e-4,
  "warmup_ratio": 0.05,
  "max_sequence_length": 128,
  "batch_size": 2,
  "gradient_accumulation_steps": 8
}
```

### 标准训练

适用于大多数使用场景：

```json
{
  "num_epochs": 100,
  "learning_rate": 5e-5,
  "warmup_ratio": 0.1,
  "max_sequence_length": 256,
  "batch_size": 4,
  "gradient_accumulation_steps": 16
}
```

### 高质量训练

适用于追求最佳性能：

```json
{
  "num_epochs": 200,
  "learning_rate": 3e-5,
  "warmup_ratio": 0.15,
  "max_sequence_length": 512,
  "batch_size": 8,
  "gradient_accumulation_steps": 8,
  "weight_decay": 0.02
}
```

## LoRA配置详解

### 参数说明

- **lora_r**: LoRA的秩，控制适配器的复杂度
  - 较小值 (4-8): 更少参数，更快训练，可能性能略低
  - 较大值 (16-32): 更多参数，更好性能，需要更多内存

- **lora_alpha**: LoRA的缩放因子
  - 通常设置为 lora_r 的 1.5-2 倍
  - 影响适配器权重的强度

- **lora_dropout**: LoRA层的dropout率
  - 0.05-0.1: 标准设置
  - 0.0: 无dropout，可能过拟合
  - 0.2+: 高dropout，可能欠拟合

### 不同任务的LoRA配置

#### 问答任务
```json
{
  "lora_r": 6,
  "lora_alpha": 12,
  "lora_dropout": 0.1,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
}
```

#### 文本生成
```json
{
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
```

#### 对话系统
```json
{
  "lora_r": 12,
  "lora_alpha": 24,
  "lora_dropout": 0.1,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
```

## 数据配置

### 数据目录结构
```
data/
├── raw/                    # 原始训练数据
│   ├── qa_dataset_1.md
│   ├── qa_dataset_2.md
│   └── README.md
├── processed/              # 处理后的数据 (自动生成)
└── cache/                  # 缓存文件 (自动生成)
```

### 数据格式配置

#### 简单QA格式
```json
{
  "data_format": "simple_qa",
  "question_prefix": "Q",
  "answer_prefix": "A",
  "separator": ":"
}
```

#### 标题QA格式
```json
{
  "data_format": "header_qa",
  "question_pattern": "### Q\\d+:",
  "answer_pattern": "A\\d+:"
}
```

## 日志和监控配置

### TensorBoard配置
```json
{
  "enable_tensorboard": true,
  "tensorboard_log_dir": "./logs/tensorboard",
  "log_every_n_steps": 10,
  "save_every_n_steps": 100
}
```

### 结构化日志配置
```json
{
  "structured_logging": {
    "enabled": true,
    "log_file": "./logs/structured/training.jsonl",
    "log_level": "INFO",
    "include_memory_stats": true,
    "include_model_stats": true
  }
}
```

## 环境变量配置

### CUDA相关
```bash
export CUDA_VISIBLE_DEVICES=0              # 指定使用的GPU
export CUDA_LAUNCH_BLOCKING=0              # 异步CUDA操作
export TORCH_USE_CUDA_DSA=1                # 启用CUDA设备端断言
```

### PyTorch相关
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 内存分配策略
export TOKENIZERS_PARALLELISM=false        # 禁用tokenizer并行化警告
```

### Transformers相关
```bash
export TRANSFORMERS_CACHE=/path/to/cache   # 模型缓存目录
export HF_HOME=/path/to/hf_cache           # Hugging Face缓存目录
```

## 高级配置选项

### 梯度检查点
```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_kwargs": {
    "use_reentrant": false
  }
}
```

### 混合精度训练
```json
{
  "fp16": false,
  "bf16": true,
  "tf32": true
}
```

### 优化器配置
```json
{
  "optimizer": "paged_adamw_8bit",
  "optimizer_kwargs": {
    "betas": [0.9, 0.999],
    "eps": 1e-8
  }
}
```

### 学习率调度器
```json
{
  "lr_scheduler_type": "cosine",
  "lr_scheduler_kwargs": {
    "num_cycles": 0.5
  }
}
```

## 配置验证

### 自动验证
```bash
# 验证配置文件
uv run python main.py --config config.json --validate-config-only

# 验证硬件兼容性
uv run python main.py --config config.json --validate-hardware-only
```

### 手动验证
```python
from src.model_manager import ModelManager
from src.memory_optimizer import MemoryOptimizer

# 验证内存配置
optimizer = MemoryOptimizer()
status = optimizer.monitor_gpu_memory()
print(f"可用内存: {status[2] - status[0]:.1f} GB")

# 验证模型加载
manager = ModelManager()
config = manager.configure_quantization()
print(f"量化配置: {config}")
```

## 配置模板

### 创建自定义配置
```bash
# 复制基础配置
cp qwen3_4b_thinking_config.json my_config.json

# 编辑配置
nano my_config.json

# 使用自定义配置
uv run python main.py --config my_config.json
```

### 配置继承
```json
{
  "base_config": "qwen3_4b_thinking_config.json",
  "overrides": {
    "max_memory_gb": 10.0,
    "batch_size": 2,
    "num_epochs": 50
  }
}
```
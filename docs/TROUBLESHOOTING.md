# 故障排除指南

本文档提供了Qwen3优化微调系统常见问题的解决方案和调试技巧。

## 内存相关问题

### GPU内存不足 (CUDA Out of Memory)

#### 症状
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB (GPU 0; X.XX GiB total capacity; X.XX GiB already allocated; X.XX GiB free; X.XX GiB reserved in total by PyTorch)
```

#### 解决方案

**1. 立即解决方案**
```bash
# 清理GPU内存
python -c "import torch; torch.cuda.empty_cache()"

# 重启Python进程
pkill -f python
```

**2. 配置调整**
```bash
# 减少批次大小
uv run python main.py --batch-size 1 --gradient-accumulation-steps 64

# 减少序列长度
uv run python main.py --max-sequence-length 128

# 降低内存限制
uv run python main.py --max-memory-gb 8
```

**3. 高级优化**
```json
{
  "max_memory_gb": 8.0,
  "batch_size": 1,
  "gradient_accumulation_steps": 64,
  "max_sequence_length": 128,
  "lora_r": 4,
  "lora_alpha": 8,
  "gradient_checkpointing": true
}
```

### 系统内存不足

#### 症状
```
MemoryError: Unable to allocate X.XX GiB for an array with shape (X, X, X)
```

#### 解决方案

**1. 增加虚拟内存**
```bash
# Linux
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久启用
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**2. 优化数据加载**
```json
{
  "dataloader_num_workers": 0,
  "dataloader_pin_memory": false,
  "remove_unused_columns": true
}
```

### 内存泄漏

#### 症状
- 内存使用持续增长
- 训练过程中系统变慢
- 最终导致OOM错误

#### 解决方案

**1. 启用内存监控**
```bash
# 运行时监控
uv run python main.py --enable-memory-monitoring

# 使用系统监控
watch -n 1 nvidia-smi
```

**2. 定期清理**
```python
# 在训练循环中添加
import gc
import torch

if step % 100 == 0:
    gc.collect()
    torch.cuda.empty_cache()
```

## 模型加载问题

### 模型下载失败

#### 症状
```
OSError: We couldn't connect to 'https://huggingface.co' to load this model
```

#### 解决方案

**1. 网络问题**
```bash
# 设置代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# 或使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

**2. 手动下载**
```bash
# 使用git lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507

# 指定本地路径
uv run python main.py --model-name ./Qwen3-4B-Thinking-2507
```

### 模型格式不兼容

#### 症状
```
ValueError: The model class you are passing has a `config_class` attribute that is not consistent with the config class you passed
```

#### 解决方案

**1. 更新依赖**
```bash
pip install --upgrade transformers peft torch
```

**2. 检查模型兼容性**
```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
print(f"模型类型: {config.model_type}")
print(f"架构: {config.architectures}")
```

### 量化配置错误

#### 症状
```
ImportError: Using `load_in_4bit=True` requires Accelerate: `pip install accelerate` and the latest version of bitsandbytes
```

#### 解决方案

**1. 安装缺失依赖**
```bash
pip install accelerate bitsandbytes
```

**2. Windows特殊处理**
```bash
# Windows可能需要特殊版本
pip install bitsandbytes-windows
```

## 训练过程问题

### 训练停滞

#### 症状
- Loss不再下降
- 验证指标无改善
- 梯度范数过小

#### 解决方案

**1. 调整学习率**
```json
{
  "learning_rate": 1e-4,
  "lr_scheduler_type": "cosine_with_restarts",
  "warmup_ratio": 0.1
}
```

**2. 检查梯度**
```python
# 添加梯度监控
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### 训练不稳定

#### 症状
- Loss剧烈波动
- 梯度爆炸
- NaN值出现

#### 解决方案

**1. 梯度裁剪**
```json
{
  "max_grad_norm": 1.0,
  "gradient_checkpointing": true
}
```

**2. 调整优化器**
```json
{
  "learning_rate": 3e-5,
  "weight_decay": 0.01,
  "warmup_ratio": 0.1
}
```

### 检查点保存失败

#### 症状
```
OSError: [Errno 28] No space left on device
```

#### 解决方案

**1. 清理磁盘空间**
```bash
# 清理缓存
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/

# 清理旧检查点
find ./qwen3-finetuned -name "checkpoint-*" -type d | head -n -3 | xargs rm -rf
```

**2. 限制检查点数量**
```json
{
  "save_total_limit": 2,
  "save_strategy": "epoch",
  "save_only_model": true
}
```

## 数据处理问题

### 数据加载失败

#### 症状
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/qa_data.md'
```

#### 解决方案

**1. 检查数据路径**
```bash
ls -la data/raw/
```

**2. 使用示例数据**
```bash
# 系统会自动回退到示例数据
uv run python main.py --data-dir data/raw
```

### 数据格式错误

#### 症状
```
ValueError: Unable to parse QA data from file
```

#### 解决方案

**1. 验证数据格式**
```python
# 检查文件编码
file -i data/raw/qa_data.md

# 转换编码
iconv -f gbk -t utf-8 data/raw/qa_data.md > data/raw/qa_data_utf8.md
```

**2. 标准化格式**
```markdown
Q1: 这是问题1？
A1: 这是答案1。

Q2: 这是问题2？
A2: 这是答案2。
```

### 分词器问题

#### 症状
```
ValueError: Tokenizer does not have a padding token
```

#### 解决方案

**1. 自动修复**
```python
# 系统会自动设置pad_token
tokenizer.pad_token = tokenizer.eos_token
```

**2. 手动配置**
```json
{
  "tokenizer_kwargs": {
    "padding_side": "left",
    "truncation_side": "left"
  }
}
```

## 推理测试问题

### 推理失败

#### 症状
```
RuntimeError: Expected all tensors to be on the same device
```

#### 解决方案

**1. 设备一致性**
```python
# 确保模型和输入在同一设备
model = model.to("cuda")
inputs = tokenizer(text, return_tensors="pt").to("cuda")
```

**2. 检查适配器加载**
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, adapter_path)
```

### 生成质量差

#### 症状
- 生成重复内容
- 回答不相关
- 格式错误

#### 解决方案

**1. 调整生成参数**
```json
{
  "generation_config": {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": true
  }
}
```

**2. 改进提示格式**
```python
prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
```

## 环境问题

### Python版本不兼容

#### 症状
```
SyntaxError: invalid syntax (match statement requires Python 3.10+)
```

#### 解决方案

**1. 升级Python**
```bash
# 使用uv安装新版本
uv python install 3.12
uv python pin 3.12
```

**2. 检查兼容性**
```python
import sys
print(f"Python版本: {sys.version}")
assert sys.version_info >= (3, 9), "需要Python 3.9+"
```

### 依赖冲突

#### 症状
```
ERROR: pip's dependency resolver does not currently have a solution for this combination of requirements
```

#### 解决方案

**1. 使用uv解决**
```bash
uv sync --resolution=highest
```

**2. 手动解决冲突**
```bash
pip install --upgrade pip
pip install --force-reinstall torch transformers
```

### CUDA版本不匹配

#### 症状
```
RuntimeError: The NVIDIA driver on your system is too old
```

#### 解决方案

**1. 更新驱动**
```bash
# Ubuntu
sudo apt update
sudo apt install nvidia-driver-535

# 重启系统
sudo reboot
```

**2. 安装兼容的PyTorch**
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 性能问题

### 训练速度慢

#### 症状
- 每个epoch耗时过长
- GPU利用率低
- 内存使用不充分

#### 解决方案

**1. 优化数据加载**
```json
{
  "dataloader_num_workers": 4,
  "dataloader_pin_memory": true,
  "dataloader_prefetch_factor": 2
}
```

**2. 调整批次大小**
```json
{
  "batch_size": 8,
  "gradient_accumulation_steps": 8,
  "max_sequence_length": 512
}
```

### 内存使用效率低

#### 症状
- GPU内存使用率低
- 训练速度慢
- 频繁的内存分配

#### 解决方案

**1. 启用内存优化**
```json
{
  "gradient_checkpointing": true,
  "optim": "paged_adamw_8bit",
  "fp16": false,
  "bf16": true
}
```

**2. 调整内存策略**
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## 调试技巧

### 启用详细日志

```bash
# 设置日志级别
export TRANSFORMERS_VERBOSITY=debug
export DATASETS_VERBOSITY=debug

# 运行时启用调试
uv run python main.py --log-level DEBUG
```

### 使用调试模式

```bash
# 快速调试运行
uv run python main.py --num-epochs 1 --batch-size 1 --max-sequence-length 64

# 内存调试
uv run python main.py --enable-memory-profiling
```

### 分步调试

```python
# 单独测试各组件
from src.memory_optimizer import MemoryOptimizer
from src.model_manager import ModelManager
from src.data_pipeline import DataPipeline

# 测试内存优化器
optimizer = MemoryOptimizer()
print(optimizer.monitor_gpu_memory())

# 测试模型加载
manager = ModelManager()
model, tokenizer = manager.load_model_with_quantization("Qwen/Qwen3-4B-Thinking-2507")

# 测试数据管道
pipeline = DataPipeline()
data = pipeline.load_qa_data_from_files("data/raw")
```

## 获取帮助

### 收集系统信息

```bash
# 生成系统报告
uv run python -c "
import torch
import transformers
import sys
import platform

print(f'系统: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

### 创建最小复现示例

```python
# minimal_reproduce.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("测试基本功能...")
model_name = "Qwen/Qwen3-4B-Thinking-2507"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ 分词器加载成功")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto"
    )
    print("✓ 模型加载成功")
    
    inputs = tokenizer("测试", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=10)
    print("✓ 推理测试成功")
    
except Exception as e:
    print(f"✗ 错误: {e}")
```

### 联系支持

如果问题仍未解决，请提供以下信息：

1. 系统信息 (运行上述系统报告)
2. 完整错误日志
3. 使用的配置文件
4. 最小复现示例
5. 尝试过的解决方案
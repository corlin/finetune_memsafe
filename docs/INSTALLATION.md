# 安装指南

本文档提供了Qwen3优化微调系统的详细安装说明，适用于不同的操作系统和硬件配置。

## 系统要求

### 最低要求
- **操作系统**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10/11
- **Python**: 3.9+ (推荐 3.12)
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM)
- **系统内存**: 16GB RAM
- **磁盘空间**: 20GB 可用空间

### 推荐配置
- **操作系统**: Linux (Ubuntu 22.04 LTS)
- **Python**: 3.12
- **GPU**: NVIDIA RTX 4090, RTX 3090, A100, V100 (12GB+ VRAM)
- **系统内存**: 32GB RAM
- **磁盘空间**: 50GB 可用空间 (SSD推荐)

## 环境准备

### 1. CUDA安装

#### Linux (Ubuntu/Debian)
```bash
# 检查GPU驱动
nvidia-smi

# 安装CUDA 12.4 (推荐)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

# 添加到PATH
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Windows
1. 下载并安装 [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-downloads)
2. 下载并安装 [cuDNN](https://developer.nvidia.com/cudnn)
3. 确保CUDA路径已添加到系统环境变量

#### macOS
```bash
# macOS不支持CUDA，建议使用MPS (Metal Performance Shaders)
# 系统会自动检测并使用MPS后端
```

### 2. Python环境

#### 使用uv (推荐)
```bash
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 重启终端或重新加载shell配置
source ~/.bashrc  # Linux/macOS
# 或重启PowerShell (Windows)

# 验证安装
uv --version
```

#### 使用conda
```bash
# 安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建环境
conda create -n qwen3-finetuning python=3.12
conda activate qwen3-finetuning
```

## 项目安装

### 方法1: 使用uv (推荐)

```bash
# 克隆项目
git clone <repository-url>
cd qwen3-finetuning

# 初始化项目环境
uv sync

# 验证安装
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 方法2: 使用pip

```bash
# 克隆项目
git clone <repository-url>
cd qwen3-finetuning

# 创建虚拟环境
python -m venv .venv

# 激活环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

### 方法3: 使用conda

```bash
# 克隆项目
git clone <repository-url>
cd qwen3-finetuning

# 激活环境
conda activate qwen3-finetuning

# 安装依赖
pip install -e .
```

## 依赖验证

### 自动验证
```bash
# 使用内置验证脚本
uv run python verify_pytorch.py

# 或使用主程序的验证功能
uv run python main.py --verify-only
```

### 手动验证

#### 检查PyTorch和CUDA
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

#### 检查Transformers
```python
import transformers
print(f"Transformers版本: {transformers.__version__}")
```

#### 检查其他关键依赖
```python
import peft, datasets, bitsandbytes, accelerate
print("所有依赖已正确安装")
```

## 常见安装问题

### CUDA相关问题

#### 问题: "CUDA out of memory"
**解决方案:**
```bash
# 检查GPU内存使用
nvidia-smi

# 清理GPU内存
python -c "import torch; torch.cuda.empty_cache()"

# 使用更小的内存配置
uv run python main.py --max-memory-gb 8 --batch-size 1
```

#### 问题: "No CUDA-capable device is detected"
**解决方案:**
1. 检查GPU驱动: `nvidia-smi`
2. 重新安装CUDA兼容的PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### 依赖冲突问题

#### 问题: "No module named 'bitsandbytes'"
**解决方案:**
```bash
# Linux/macOS
pip install bitsandbytes

# Windows (可能需要特殊版本)
pip install bitsandbytes-windows
```

#### 问题: "ImportError: cannot import name 'AutoTokenizer'"
**解决方案:**
```bash
pip install --upgrade transformers
```

### 内存问题

#### 问题: "RuntimeError: CUDA out of memory"
**解决方案:**
1. 减少批次大小: `--batch-size 1`
2. 增加梯度累积: `--gradient-accumulation-steps 32`
3. 减少序列长度: `--max-sequence-length 128`
4. 使用更激进的量化设置

#### 问题: 系统内存不足
**解决方案:**
1. 关闭其他应用程序
2. 增加虚拟内存/交换空间
3. 使用更小的模型或数据集

## 性能优化

### GPU优化
```bash
# 设置GPU性能模式
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300  # 设置功率限制为300W

# 设置GPU时钟频率
sudo nvidia-smi -ac 877,1215  # 内存时钟,GPU时钟
```

### 系统优化
```bash
# 增加文件描述符限制
ulimit -n 65536

# 设置环境变量优化
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1
```

## 验证安装

### 快速测试
```bash
# 运行简单测试
uv run python simple_test.py

# 运行内存测试
uv run python -c "
from src.memory_optimizer import MemoryOptimizer
optimizer = MemoryOptimizer()
status = optimizer.monitor_gpu_memory()
print(f'GPU内存状态: {status}')
"
```

### 完整测试
```bash
# 运行所有测试
uv run python -m pytest tests/ -v

# 运行集成测试
uv run python tests/test_integration.py
```

## 下一步

安装完成后，请参考以下文档:
- [配置指南](CONFIGURATION.md) - 了解如何配置系统
- [使用指南](USAGE.md) - 学习如何使用系统
- [故障排除](TROUBLESHOOTING.md) - 解决常见问题
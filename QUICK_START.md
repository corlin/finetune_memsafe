# Qwen3-4B-Thinking-2507 快速启动指南

## 🚀 推荐启动方式（解决uv构建问题）

由于uv的editable安装可能遇到构建问题，推荐使用以下直接运行方式：

### 方式1: 使用Python直接运行（推荐）

```bash
# 直接运行Python脚本
python run_qwen3.py
```

或在Windows上双击 `run_qwen3.bat`

### 方式2: 手动安装依赖后运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行主程序
python main.py --model-name "Qwen/Qwen3-4B-Thinking-2507" --auto-install-deps
```

### 方式3: 使用uv但绕过editable安装

```bash
# 安装依赖到uv环境
uv pip install -r requirements.txt

# 直接运行（不使用项目模式）
uv run --no-project python main.py --model-name "Qwen/Qwen3-4B-Thinking-2507" --auto-install-deps
```

## 🔧 运行模式选择

运行 `python run_qwen3.py` 后，选择适合的模式：

1. **快速测试模式**: 5轮训练，验证环境是否正常
2. **低显存模式**: 适合6-8GB GPU，使用更保守的参数
3. **标准模式**: 推荐配置，适合10GB+GPU
4. **配置文件模式**: 使用 `qwen3_4b_thinking_config.json` 配置

## 📋 系统要求

- Python 3.12
- CUDA兼容的GPU (推荐10GB+显存)
- 15GB+可用磁盘空间
- 16GB+系统内存

## 🛠️ 故障排除

### 如果遇到依赖安装问题：

```bash
# 方法1: 使用pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft datasets bitsandbytes accelerate

# 方法2: 使用uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers peft datasets bitsandbytes accelerate
```

### 如果遇到GPU内存不足：

选择低显存模式，或手动调整参数：

```bash
python main.py \
  --model-name "Qwen/Qwen3-4B-Thinking-2507" \
  --max-memory-gb 6 \
  --batch-size 1 \
  --gradient-accumulation-steps 64 \
  --auto-install-deps
```

### 如果遇到CUDA问题：

```bash
# 检查CUDA
nvidia-smi

# 检查PyTorch CUDA支持
python -c "import torch; print(torch.cuda.is_available())"
```

## 📊 监控训练

训练开始后：

1. **查看实时日志**:
   ```bash
   tail -f logs/application.log
   ```

2. **启动TensorBoard**:
   ```bash
   tensorboard --logdir ./qwen3-finetuned/logs/tensorboard
   # 访问 http://localhost:6006
   ```

3. **检查GPU使用**:
   ```bash
   nvidia-smi
   ```

## 🎯 完成后的操作

训练完成后，您会得到：

- `qwen3-finetuned/` - 微调后的模型文件
- `logs/` - 详细的训练日志
- `final_application_report.json` - 训练报告

测试推理：

```python
from src.inference_tester import InferenceTester

tester = InferenceTester()
tester.load_finetuned_model('./qwen3-finetuned', 'Qwen/Qwen3-4B-Thinking-2507')
response = tester.test_inference('请解释什么是深度学习？')
print(response)
```

## 💡 提示

- 首次运行建议选择"快速测试模式"验证环境
- 如果GPU显存不足，选择"低显存模式"
- 训练过程中可以随时按Ctrl+C中断，状态会自动保存
- 建议在训练前关闭其他占用GPU的程序
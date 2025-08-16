# 使用uv运行Qwen3微调系统

## 1. 安装uv

如果您还没有安装uv，请先安装：

### Windows (PowerShell)
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Linux/macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 使用pip安装
```bash
pip install uv
```

## 2. 初始化项目

项目已经配置好了pyproject.toml文件，您可以直接同步依赖：

```bash
# 同步项目依赖
uv sync

# 或者如果您想创建新的虚拟环境
uv venv
uv sync
```

## 3. 运行微调系统

### 基本使用

```bash
# 使用默认配置运行（推荐新手）
uv run main.py --auto-install-deps

# 或者激活虚拟环境后运行
uv shell
python main.py --auto-install-deps
```

### 自定义配置

```bash
# 指定模型和输出目录
uv run main.py --model-name "Qwen/Qwen3-4B-Thinking-2507" --output-dir "./my-model" --auto-install-deps

# 调整内存和训练参数（适用于低显存GPU）
uv run main.py --max-memory-gb 8 --batch-size 2 --gradient-accumulation-steps 32 --auto-install-deps

# 快速测试（少量训练轮数）
uv run main.py --num-epochs 10 --auto-install-deps
```

### 使用配置文件

创建配置文件 `config.json`：

```json
{
  "model_name": "Qwen/Qwen3-4B-Thinking-2507",
  "output_dir": "./qwen3-finetuned",
  "max_memory_gb": 13.0,
  "batch_size": 4,
  "gradient_accumulation_steps": 16,
  "learning_rate": 5e-5,
  "num_epochs": 50,
  "data_dir": "data/raw",
  "auto_install_deps": true,
  "verify_environment": true
}
```

然后运行：
```bash
uv run main.py --config config.json
```

## 4. 常用命令

### 环境管理
```bash
# 查看项目信息
uv info

# 查看已安装的包
uv pip list

# 添加新依赖
uv add package-name

# 移除依赖
uv remove package-name

# 更新依赖
uv sync --upgrade
```

### 开发工具
```bash
# 代码格式化
uv run black .

# 代码排序
uv run isort .

# 类型检查
uv run mypy src/

# 运行测试
uv run pytest
```

## 5. 推荐的运行流程

### 第一次运行（完整流程）
```bash
# 1. 同步依赖
uv sync

# 2. 运行环境检查
uv run python -c "from src.environment_validator import create_environment_report; create_environment_report()"

# 3. 开始微调（使用保守配置）
uv run main.py --max-memory-gb 10 --batch-size 2 --num-epochs 20 --auto-install-deps
```

### 日常使用
```bash
# 使用默认配置
uv run main.py --auto-install-deps

# 或者使用配置文件
uv run main.py --config config.json
```

## 6. 故障排除

### 依赖问题
```bash
# 重新安装所有依赖
uv sync --reinstall

# 清理缓存
uv cache clean

# 手动安装PyTorch（如果自动安装失败）
# CUDA 12.4版本 (2025年推荐)
uv add torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 --index-url https://download.pytorch.org/whl/cu124

# CPU版本
uv add torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### 内存问题
如果遇到GPU内存不足，尝试以下配置：
```bash
# 最小内存配置
uv run main.py --max-memory-gb 6 --batch-size 1 --gradient-accumulation-steps 64 --max-sequence-length 128 --auto-install-deps
```

### 权限问题
```bash
# 在Windows上，如果遇到权限问题，使用管理员权限运行PowerShell
# 或者设置执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 7. 监控训练

### TensorBoard
```bash
# 启动TensorBoard（在另一个终端中）
uv run tensorboard --logdir ./qwen3-finetuned/logs/tensorboard

# 在浏览器中访问
# http://localhost:6006
```

### 查看日志
```bash
# 查看应用程序日志
tail -f logs/application.log

# 查看结构化日志
tail -f logs/structured/qwen3_training_structured.jsonl
```

## 8. 完成后的文件

训练完成后，您会得到：
- `qwen3-finetuned/` - 微调后的模型文件
- `logs/` - 训练日志
- `final_application_report.json` - 最终训练报告

## 9. 推理测试

```bash
# 系统会自动进行推理测试，您也可以手动测试
uv run python -c "
from src.inference_tester import InferenceTester
tester = InferenceTester()
tester.load_finetuned_model('./qwen3-finetuned', 'Qwen/Qwen3-4B-Thinking-2507')
print(tester.test_inference('请解释什么是机器学习？'))
"
```

这个指南应该能帮助您顺利使用uv运行Qwen3微调系统！
# 故障排除指南

本指南帮助用户诊断和解决使用模型导出优化系统时遇到的常见问题。

## 目录

- [快速诊断](#快速诊断)
- [常见错误及解决方案](#常见错误及解决方案)
- [性能问题](#性能问题)
- [配置问题](#配置问题)
- [环境问题](#环境问题)
- [调试技巧](#调试技巧)
- [日志分析](#日志分析)
- [FAQ](#faq)

## 快速诊断

### 诊断检查清单

在报告问题之前，请先检查以下项目：

- [ ] Python版本 >= 3.8
- [ ] PyTorch版本 >= 1.12.0
- [ ] Transformers版本 >= 4.21.0
- [ ] 足够的GPU内存 (推荐 >= 8GB)
- [ ] 足够的磁盘空间 (模型大小的3-5倍)
- [ ] Checkpoint文件完整性
- [ ] 网络连接正常 (用于下载基座模型)

### 快速诊断命令

```bash
# 检查Python环境
python --version
pip list | grep -E "(torch|transformers|onnx)"

# 检查GPU状态
nvidia-smi

# 检查磁盘空间
df -h

# 检查checkpoint文件
ls -la qwen3-finetuned/
```

## 常见错误及解决方案

### 1. Checkpoint相关错误

#### 错误: Checkpoint目录不存在

```
CheckpointValidationError: Checkpoint目录不存在: ./qwen3-finetuned
```

**原因**: 指定的checkpoint路径不存在或路径错误

**解决方案**:
1. 检查路径是否正确：
   ```bash
   ls -la ./qwen3-finetuned
   ```
2. 使用绝对路径：
   ```python
   config.checkpoint_path = "/absolute/path/to/qwen3-finetuned"
   ```
3. 检查当前工作目录：
   ```python
   import os
   print(f"当前目录: {os.getcwd()}")
   ```

#### 错误: Checkpoint文件不完整

```
CheckpointValidationError: 缺少必要文件: adapter_model.safetensors
```

**原因**: Checkpoint文件损坏或训练未完成

**解决方案**:
1. 检查必要文件是否存在：
   ```bash
   ls qwen3-finetuned/adapter_config.json
   ls qwen3-finetuned/adapter_model.safetensors
   ```
2. 重新训练模型或使用其他checkpoint
3. 手动指定checkpoint路径：
   ```python
   config.checkpoint_path = "./qwen3-finetuned/checkpoint-30"
   ```

### 2. 内存相关错误

#### 错误: CUDA内存不足

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**原因**: GPU内存不足以加载模型

**解决方案**:
1. 启用4bit加载：
   ```python
   config = ExportConfiguration(
       # ... 其他配置
       base_model_load_in_4bit=True
   )
   ```

2. 启用8bit加载：
   ```python
   config = ExportConfiguration(
       # ... 其他配置
       base_model_load_in_8bit=True
   )
   ```

3. 使用CPU加载：
   ```python
   config = ExportConfiguration(
       # ... 其他配置
       device_map="cpu"
   )
   ```

4. 清理GPU缓存：
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

#### 错误: 系统内存不足

```
MemoryError: Unable to allocate array
```

**原因**: 系统RAM不足

**解决方案**:
1. 关闭其他程序释放内存
2. 使用分批处理：
   ```python
   config.batch_size = 1
   config.max_sequence_length = 1024
   ```
3. 启用内存映射：
   ```python
   config.use_memory_mapping = True
   ```

### 3. 模型加载错误

#### 错误: 基座模型下载失败

```
OSError: Can't load tokenizer for 'Qwen/Qwen3-4B-Thinking-2507'
```

**原因**: 网络连接问题或模型不存在

**解决方案**:
1. 检查网络连接
2. 使用本地模型路径：
   ```python
   config.base_model_name = "/path/to/local/model"
   ```
3. 设置代理：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
4. 手动下载模型：
   ```bash
   huggingface-cli download Qwen/Qwen3-4B-Thinking-2507
   ```

#### 错误: LoRA适配器加载失败

```
ValueError: Can't find adapter_config.json in ./qwen3-finetuned
```

**原因**: LoRA适配器文件损坏或格式错误

**解决方案**:
1. 验证adapter_config.json格式：
   ```bash
   cat qwen3-finetuned/adapter_config.json | python -m json.tool
   ```
2. 检查文件权限：
   ```bash
   ls -la qwen3-finetuned/adapter_config.json
   ```
3. 重新生成适配器文件

### 4. ONNX导出错误

#### 错误: 不支持的操作符

```
RuntimeError: Unsupported operator: aten::scaled_dot_product_attention
```

**原因**: ONNX版本不支持某些PyTorch操作符

**解决方案**:
1. 降低ONNX操作集版本：
   ```python
   config.onnx_opset_version = 11
   ```
2. 禁用图优化：
   ```python
   config.onnx_optimize_graph = False
   ```
3. 更新ONNX版本：
   ```bash
   pip install --upgrade onnx onnxruntime
   ```

#### 错误: 动态形状问题

```
RuntimeError: The following model inputs have dynamic shapes that are not supported
```

**原因**: 动态轴配置不正确

**解决方案**:
1. 修正动态轴配置：
   ```python
   config.onnx_dynamic_axes = {
       "input_ids": {0: "batch_size", 1: "sequence_length"},
       "attention_mask": {0: "batch_size", 1: "sequence_length"}
   }
   ```
2. 禁用动态轴：
   ```python
   config.onnx_dynamic_axes = None
   ```

### 5. 验证错误

#### 错误: 输出不一致

```
ValidationError: 模型输出差异过大: 0.001 > 1e-5
```

**原因**: 量化或优化导致精度损失

**解决方案**:
1. 放宽容忍度：
   ```python
   config.validation_tolerance = 1e-3
   ```
2. 降低量化级别：
   ```python
   config.quantization_level = "fp16"  # 从int8改为fp16
   ```
3. 禁用某些优化：
   ```python
   config.remove_training_artifacts = False
   config.compress_weights = False
   ```

#### 错误: 功能测试失败

```
ValidationError: 模型推理失败
```

**原因**: 导出的模型无法正常推理

**解决方案**:
1. 检查模型文件完整性
2. 验证tokenizer配置
3. 使用更简单的测试输入：
   ```python
   config.test_input_samples = ["Hello"]
   ```

## 性能问题

### 导出速度慢

**症状**: 导出过程耗时过长

**诊断**:
```python
# 启用详细日志查看瓶颈
config.log_level = "DEBUG"
config.enable_progress_monitoring = True
```

**解决方案**:
1. 跳过验证：
   ```python
   config.run_validation_tests = False
   ```
2. 减少测试样本：
   ```python
   config.test_input_samples = ["简单测试"]
   ```
3. 禁用ONNX导出：
   ```python
   config.export_onnx = False
   ```
4. 使用SSD存储

### 内存使用过高

**症状**: 系统内存或GPU内存使用率过高

**解决方案**:
1. 启用内存监控：
   ```python
   config.memory_monitoring = True
   ```
2. 使用量化加载：
   ```python
   config.base_model_load_in_4bit = True
   ```
3. 分批处理：
   ```python
   config.batch_size = 1
   ```

## 配置问题

### 配置文件格式错误

**错误**: YAML或JSON解析失败

**解决方案**:
1. 验证YAML格式：
   ```bash
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```
2. 验证JSON格式：
   ```bash
   python -m json.tool config.json
   ```
3. 检查缩进和引号

### 配置参数冲突

**错误**: 配置参数之间存在冲突

**解决方案**:
1. 使用配置验证：
   ```python
   from src.config_validator import validate_config
   errors = validate_config(config)
   ```
2. 查看配置指南了解参数依赖关系

## 环境问题

### 依赖版本冲突

**错误**: 包版本不兼容

**解决方案**:
1. 创建新的虚拟环境：
   ```bash
   python -m venv export_env
   source export_env/bin/activate  # Linux/Mac
   # 或
   export_env\Scripts\activate  # Windows
   ```
2. 安装指定版本：
   ```bash
   pip install torch==1.12.0 transformers==4.21.0
   ```

### CUDA版本问题

**错误**: CUDA版本不匹配

**解决方案**:
1. 检查CUDA版本：
   ```bash
   nvcc --version
   nvidia-smi
   ```
2. 安装匹配的PyTorch版本：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 调试技巧

### 启用详细日志

```python
import logging

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug.log')
    ]
)

config.log_level = "DEBUG"
```

### 保存中间结果

```python
# 保留临时文件用于调试
config.cleanup_temp_files = False

# 保存每个步骤的结果
config.save_intermediate_results = True
```

### 分步调试

```python
# 分别测试各个组件
from src.checkpoint_detector import CheckpointDetector
from src.model_merger import ModelMerger

# 测试checkpoint检测
detector = CheckpointDetector()
checkpoint = detector.detect_latest_checkpoint("./qwen3-finetuned")
print(f"检测到checkpoint: {checkpoint}")

# 测试模型合并
merger = ModelMerger()
base_model = merger.load_base_model("Qwen/Qwen3-4B-Thinking-2507")
print(f"基座模型加载成功: {type(base_model)}")
```

## 日志分析

### 日志级别说明

- **DEBUG**: 详细的调试信息
- **INFO**: 一般信息
- **WARNING**: 警告信息
- **ERROR**: 错误信息
- **CRITICAL**: 严重错误

### 关键日志模式

1. **内存使用**:
   ```
   INFO - Memory usage: 4.2GB / 8.0GB (52.5%)
   ```

2. **处理进度**:
   ```
   INFO - Processing checkpoint: 50% complete
   ```

3. **错误信息**:
   ```
   ERROR - Model merge failed: CUDA out of memory
   ```

### 日志分析工具

```bash
# 查找错误
grep -i "error" export.log

# 查找内存相关信息
grep -i "memory\|oom" export.log

# 查看处理时间
grep -i "duration\|time" export.log
```

## FAQ

### Q1: 为什么导出的模型比原始模型大？

**A**: 这可能是因为：
1. 没有启用压缩：设置 `compress_weights=True`
2. 没有移除训练artifacts：设置 `remove_training_artifacts=True`
3. 使用了较低的量化级别：尝试 `quantization_level="int8"`

### Q2: ONNX模型推理结果与PyTorch不一致怎么办？

**A**: 尝试以下解决方案：
1. 放宽验证容忍度：`validation_tolerance=1e-3`
2. 禁用ONNX图优化：`onnx_optimize_graph=False`
3. 使用较新的ONNX操作集版本：`onnx_opset_version=16`

### Q3: 如何处理"模型太大无法加载"的问题？

**A**: 使用以下策略：
1. 启用4bit量化加载：`load_in_4bit=True`
2. 使用CPU加载：`device_map="cpu"`
3. 分层加载：`device_map="auto"`

### Q4: 验证测试总是失败怎么办？

**A**: 检查以下方面：
1. 测试输入是否合适
2. 容忍度设置是否过严格
3. 模型是否正确导出
4. tokenizer配置是否正确

### Q5: 如何提高导出速度？

**A**: 优化建议：
1. 跳过验证：`run_validation_tests=False`
2. 减少测试样本数量
3. 使用SSD存储
4. 禁用不需要的导出格式

### Q6: 导出过程中断后如何恢复？

**A**: 系统支持断点恢复：
1. 重新运行相同配置
2. 系统会自动检测已完成的步骤
3. 从中断点继续执行

### Q7: 如何确认导出的模型质量？

**A**: 质量检查方法：
1. 运行验证测试
2. 比较输出一致性
3. 进行性能基准测试
4. 在实际应用中测试

### Q8: 支持哪些模型格式？

**A**: 目前支持：
- PyTorch (.bin, .safetensors)
- ONNX (.onnx)
- TensorRT (实验性支持)

### Q9: 如何自定义量化配置？

**A**: 可以通过以下方式：
1. 选择量化级别：`none`, `fp16`, `int8`, `int4`
2. 提供校准数据集
3. 调整量化参数

### Q10: 遇到未知错误怎么办？

**A**: 故障排除步骤：
1. 启用DEBUG日志
2. 保存中间结果
3. 检查系统资源
4. 查看详细错误信息
5. 提交Issue并附上日志

## 获取帮助

如果本指南无法解决你的问题，请：

1. **查看日志文件**: 启用DEBUG级别日志获取详细信息
2. **检查系统资源**: 确保有足够的内存和磁盘空间
3. **更新依赖**: 确保使用最新兼容版本
4. **提交Issue**: 包含错误信息、配置文件和系统信息
5. **社区支持**: 在相关论坛或社区寻求帮助

## 预防措施

为避免常见问题，建议：

1. **定期备份**: 备份重要的checkpoint和配置文件
2. **环境隔离**: 使用虚拟环境避免依赖冲突
3. **资源监控**: 监控系统资源使用情况
4. **配置验证**: 在导出前验证配置文件
5. **渐进测试**: 从简单配置开始，逐步增加复杂性
# 模型导出CLI工具使用指南

## 概述

模型导出CLI工具提供了一个完整的命令行界面，用于将LoRA checkpoint与基座模型合并，并导出为多种优化格式。

## 安装和设置

### 基本使用

```bash
# 直接运行
python model_export_cli.py [command] [options]

# 或者作为模块运行
python -m src.cli [command] [options]
```

### 环境变量配置

可以通过环境变量设置默认值：

```bash
export EXPORT_CHECKPOINT_PATH="qwen3-finetuned"
export EXPORT_OUTPUT_DIR="exported_models"
export EXPORT_QUANTIZATION_LEVEL="int8"
export EXPORT_LOG_LEVEL="INFO"
```

## 命令参考

### 1. export - 执行模型导出

将LoRA checkpoint与基座模型合并并导出为指定格式。

#### 基本语法

```bash
python model_export_cli.py export [options]
```

#### 常用选项

| 选项 | 简写 | 描述 | 默认值 |
|------|------|------|--------|
| `--checkpoint-path` | `-c` | Checkpoint目录路径 | qwen3-finetuned |
| `--base-model` | `-m` | 基座模型名称 | Qwen/Qwen3-4B-Thinking-2507 |
| `--output-dir` | `-o` | 输出目录 | exported_models |
| `--quantization` | `-q` | 量化级别 (none/fp16/int8/int4) | int8 |
| `--config` | `-f` | 配置文件路径 | - |
| `--log-level` | - | 日志级别 | INFO |

#### 优化选项

| 选项 | 描述 |
|------|------|
| `--no-artifacts` | 不移除训练artifacts |
| `--no-compression` | 不压缩权重 |
| `--parallel` | 启用并行导出 |
| `--max-memory` | 最大内存使用量(GB) |

#### 格式选项

| 选项 | 描述 |
|------|------|
| `--pytorch` | 导出PyTorch格式 (默认启用) |
| `--onnx` | 导出ONNX格式 (默认启用) |
| `--tensorrt` | 导出TensorRT格式 |
| `--no-pytorch` | 不导出PyTorch格式 |
| `--no-onnx` | 不导出ONNX格式 |

#### ONNX选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--onnx-opset` | ONNX opset版本 | 14 |
| `--no-onnx-optimize` | 不优化ONNX图 | - |

#### 验证选项

| 选项 | 描述 |
|------|------|
| `--no-validation` | 跳过验证测试 |

#### 使用示例

```bash
# 基本导出
python model_export_cli.py export --checkpoint-path qwen3-finetuned

# 使用配置文件
python model_export_cli.py export --config export_config.yaml

# 自定义量化和格式
python model_export_cli.py export -c qwen3-finetuned -q int4 --onnx --tensorrt

# 高内存环境的并行导出
python model_export_cli.py export --parallel --max-memory 32

# 仅导出PyTorch格式，跳过验证
python model_export_cli.py export --no-onnx --no-validation
```

### 2. config - 配置管理

管理导出配置文件和模板。

#### 子命令

##### create-template - 创建配置模板

```bash
python model_export_cli.py config create-template [--output template.yaml]
```

创建一个包含所有可用选项的配置模板文件。

##### validate - 验证配置文件

```bash
python model_export_cli.py config validate config.yaml
```

验证配置文件的语法和参数有效性。

##### show - 显示当前配置

```bash
python model_export_cli.py config show [--config config.yaml]
```

显示当前生效的配置参数。

##### presets - 管理配置预设

```bash
# 列出可用预设
python model_export_cli.py config presets list

# 创建新预设
python model_export_cli.py config presets create my-preset --description "我的自定义预设"

# 使用预设
python model_export_cli.py config presets use production --output my_config.yaml
```

#### 使用示例

```bash
# 创建配置模板
python model_export_cli.py config create-template --output my_config.yaml

# 验证配置
python model_export_cli.py config validate my_config.yaml

# 显示配置
python model_export_cli.py config show --config my_config.yaml

# 管理预设
python model_export_cli.py config presets list
python model_export_cli.py config presets use mobile
```

### 3. validate - 验证导出的模型

验证已导出模型的功能性和一致性。

#### 基本语法

```bash
python model_export_cli.py validate model_path [options]
```

#### 选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--format` | 模型格式 (pytorch/onnx/tensorrt) | pytorch |
| `--test-samples` | 测试样本数量 | 5 |
| `--compare-with` | 与另一个模型比较输出 | - |
| `--benchmark` | 运行性能基准测试 | - |
| `--output-report` | 验证报告输出路径 | - |

#### 使用示例

```bash
# 基本验证
python model_export_cli.py validate exported_models/qwen3_merged

# 验证ONNX模型并运行基准测试
python model_export_cli.py validate exported_models/qwen3.onnx --format onnx --benchmark

# 比较两个模型的输出
python model_export_cli.py validate model1 --compare-with model2 --output-report comparison.json

# 使用更多测试样本
python model_export_cli.py validate model --test-samples 20
```

### 4. wizard - 交互式配置向导

通过交互式界面创建导出配置。

#### 基本语法

```bash
python model_export_cli.py wizard [options]
```

#### 选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--output` | 配置文件输出路径 | export_config.yaml |
| `--preset` | 基于预设开始配置 | - |

#### 使用示例

```bash
# 启动配置向导
python model_export_cli.py wizard

# 基于预设启动向导
python model_export_cli.py wizard --preset production --output prod_config.yaml
```

## 配置文件格式

### 基本配置文件

```yaml
# 基本配置
checkpoint_path: "qwen3-finetuned"
base_model_name: "Qwen/Qwen3-4B-Thinking-2507"
output_directory: "exported_models"

# 优化配置
quantization_level: "int8"  # none, fp16, int8, int4
remove_training_artifacts: true
compress_weights: true

# 导出格式
export_pytorch: true
export_onnx: true
export_tensorrt: false

# ONNX配置
onnx_opset_version: 14
onnx_optimize_graph: true

# 验证配置
run_validation_tests: true

# 监控配置
enable_progress_monitoring: true
log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
max_memory_usage_gb: 16.0
enable_parallel_export: false
```

### 高级配置文件

```yaml
export:
  # Checkpoint配置
  checkpoint:
    path: "qwen3-finetuned"
    auto_detect_latest: true
    
  # 基座模型配置
  base_model:
    name: "Qwen/Qwen3-4B-Thinking-2507"
    load_in_4bit: false
    trust_remote_code: true
    
  # 优化配置
  optimization:
    quantization: "int8"
    remove_artifacts: true
    compress_weights: true
    
  # 导出格式配置
  formats:
    pytorch:
      enabled: true
      save_tokenizer: true
      save_config: true
      
    onnx:
      enabled: true
      opset_version: 14
      dynamic_axes: true
      optimize_graph: true
      
    tensorrt:
      enabled: false
      precision: "fp16"
      
  # 验证配置
  validation:
    enabled: true
    test_samples: 5
    compare_outputs: true
    benchmark_performance: true
    
  # 输出配置
  output:
    directory: "exported_models"
    naming_pattern: "{model_name}_{timestamp}"
    
  # 监控配置
  monitoring:
    enable_progress: true
    log_level: "INFO"
    max_memory_gb: 16.0
    
  # 高级配置
  advanced:
    enable_parallel_export: false
    retry_attempts: 3
    cleanup_temp_files: true
```

## 预设配置

工具提供了几个预定义的配置预设：

### quick-export
- **用途**: 快速测试和验证
- **特点**: FP16量化，仅导出PyTorch格式，低内存使用

### production
- **用途**: 生产环境部署
- **特点**: INT8量化，多格式导出，完整优化

### mobile
- **用途**: 移动端和资源受限环境
- **特点**: INT4量化，极致压缩，低内存使用

### research
- **用途**: 研究和分析
- **特点**: 无量化，保持最高精度，详细日志

## 环境变量

所有配置选项都可以通过环境变量覆盖：

```bash
# 基本配置
export EXPORT_CHECKPOINT_PATH="qwen3-finetuned"
export EXPORT_BASE_MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
export EXPORT_OUTPUT_DIR="exported_models"

# 优化配置
export EXPORT_QUANTIZATION_LEVEL="int8"
export EXPORT_REMOVE_ARTIFACTS="true"
export EXPORT_COMPRESS_WEIGHTS="true"

# 格式配置
export EXPORT_PYTORCH="true"
export EXPORT_ONNX="true"
export EXPORT_TENSORRT="false"

# ONNX配置
export EXPORT_ONNX_OPSET="14"
export EXPORT_ONNX_OPTIMIZE="true"

# 验证配置
export EXPORT_VALIDATION="true"

# 监控配置
export EXPORT_MONITORING="true"
export EXPORT_LOG_LEVEL="INFO"
export EXPORT_MAX_MEMORY_GB="16.0"
export EXPORT_PARALLEL="false"
```

## 常见使用场景

### 场景1: 快速测试新模型

```bash
# 使用快速导出预设
python model_export_cli.py config presets use quick-export
python model_export_cli.py export --config export_config_quick-export.yaml
```

### 场景2: 生产环境部署

```bash
# 使用生产预设，启用所有优化
python model_export_cli.py export \
  --checkpoint-path production-checkpoint \
  --quantization int8 \
  --pytorch --onnx \
  --parallel \
  --max-memory 32
```

### 场景3: 移动端部署

```bash
# 极致压缩
python model_export_cli.py export \
  --quantization int4 \
  --max-memory 4 \
  --output-dir mobile_models
```

### 场景4: 研究分析

```bash
# 保持最高精度
python model_export_cli.py export \
  --quantization none \
  --no-compression \
  --log-level DEBUG
```

### 场景5: 批量处理

```bash
# 使用配置文件批量处理多个checkpoint
for checkpoint in checkpoint-*; do
  python model_export_cli.py export \
    --checkpoint-path "$checkpoint" \
    --output-dir "exported_$checkpoint" \
    --config batch_config.yaml
done
```

## 故障排除

### 常见问题

1. **内存不足错误**
   ```bash
   # 减少内存使用
   python model_export_cli.py export --max-memory 8 --no-parallel
   ```

2. **ONNX导出失败**
   ```bash
   # 使用兼容的opset版本
   python model_export_cli.py export --onnx-opset 11 --no-onnx-optimize
   ```

3. **验证失败**
   ```bash
   # 跳过验证或使用更宽松的容差
   python model_export_cli.py export --no-validation
   ```

4. **配置文件错误**
   ```bash
   # 验证配置文件
   python model_export_cli.py config validate my_config.yaml
   ```

### 调试模式

```bash
# 启用详细日志
python model_export_cli.py export --log-level DEBUG

# 保存中间结果
export EXPORT_SAVE_INTERMEDIATE="true"
python model_export_cli.py export
```

## 高级用法

### 自定义配置模板

```bash
# 创建自定义模板
python model_export_cli.py config create-template --output custom_template.yaml
# 编辑模板文件
# 使用自定义模板
python model_export_cli.py export --config custom_template.yaml
```

### 配置预设管理

```bash
# 创建基于当前配置的预设
python model_export_cli.py config presets create my-preset --description "我的自定义预设"

# 列出所有预设
python model_export_cli.py config presets list

# 使用预设
python model_export_cli.py config presets use my-preset --output my_config.yaml
```

### 脚本集成

```bash
#!/bin/bash
# 自动化导出脚本

# 设置环境变量
export EXPORT_LOG_LEVEL="INFO"
export EXPORT_MAX_MEMORY_GB="16"

# 执行导出
python model_export_cli.py export \
  --checkpoint-path "$1" \
  --output-dir "$(date +%Y%m%d)_exports" \
  --config production_config.yaml

# 验证结果
if [ $? -eq 0 ]; then
  echo "导出成功"
  python model_export_cli.py validate "$(date +%Y%m%d)_exports/merged_model" --benchmark
else
  echo "导出失败"
  exit 1
fi
```

## 性能优化建议

1. **内存优化**
   - 使用适当的 `--max-memory` 设置
   - 避免在内存不足时启用并行导出

2. **速度优化**
   - 在高内存环境中启用 `--parallel`
   - 跳过不需要的验证测试

3. **存储优化**
   - 使用适当的量化级别
   - 启用权重压缩

4. **质量优化**
   - 运行完整的验证测试
   - 比较不同格式的输出一致性

## 更多资源

- [配置参考文档](CONFIGURATION.md)
- [故障排除指南](TROUBLESHOOTING.md)
- [API文档](API_REFERENCE.md)
- [示例脚本](../examples/)
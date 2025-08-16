# 配置参数说明和最佳实践指南

## 概述

本指南详细介绍了模型导出优化系统的所有配置参数，并提供了针对不同使用场景的最佳实践建议。

## 配置文件格式

系统支持YAML和JSON两种配置文件格式。推荐使用YAML格式，因为它更易读和维护。

### YAML配置示例

```yaml
# export_config.yaml
export:
  # 基本配置
  checkpoint:
    path: "qwen3-finetuned"
    auto_detect_latest: true
    validate_integrity: true
  
  base_model:
    name: "Qwen/Qwen3-4B-Thinking-2507"
    load_in_4bit: false
    load_in_8bit: false
    torch_dtype: "float16"
    device_map: "auto"
  
  # 优化配置
  optimization:
    quantization: "int8"
    remove_artifacts: true
    compress_weights: true
    optimization_level: "standard"
  
  # 导出格式
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
      use_external_data_format: false
    tensorrt:
      enabled: false
      precision: "fp16"
      max_batch_size: 1
  
  # 验证配置
  validation:
    enabled: true
    test_samples: 10
    compare_outputs: true
    benchmark_performance: true
    tolerance: 1e-5
  
  # 输出配置
  output:
    directory: "exported_models"
    naming_pattern: "{model_name}_{timestamp}"
    create_subdirs: true
    cleanup_temp_files: true
  
  # 监控配置
  monitoring:
    enable_progress_bar: true
    log_level: "INFO"
    save_logs: true
    memory_monitoring: true
```

## 详细参数说明

### 基本配置 (Basic Configuration)

#### checkpoint 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | string | 必需 | checkpoint目录路径 |
| `auto_detect_latest` | boolean | true | 是否自动检测最新checkpoint |
| `validate_integrity` | boolean | true | 是否验证checkpoint完整性 |

**最佳实践:**
- 使用相对路径时，确保路径相对于工作目录正确
- 启用`auto_detect_latest`可以自动选择最新的训练checkpoint
- 建议保持`validate_integrity`为true以确保checkpoint文件完整

#### base_model 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | 必需 | 基座模型名称或路径 |
| `load_in_4bit` | boolean | false | 是否以4bit精度加载 |
| `load_in_8bit` | boolean | false | 是否以8bit精度加载 |
| `torch_dtype` | string | "float16" | PyTorch数据类型 |
| `device_map` | string | "auto" | 设备映射策略 |

**最佳实践:**
- 对于大型模型，使用`load_in_4bit`或`load_in_8bit`可以显著减少内存使用
- `torch_dtype`推荐使用"float16"以平衡精度和性能
- `device_map: "auto"`让系统自动分配GPU资源

### 优化配置 (Optimization Configuration)

#### optimization 配置

| 参数 | 类型 | 可选值 | 默认值 | 说明 |
|------|------|--------|--------|------|
| `quantization` | string | "none", "fp16", "int8", "int4" | "int8" | 量化级别 |
| `remove_artifacts` | boolean | - | true | 是否移除训练artifacts |
| `compress_weights` | boolean | - | true | 是否压缩权重 |
| `optimization_level` | string | "minimal", "standard", "aggressive" | "standard" | 优化级别 |

**量化级别说明:**

- **none**: 不进行量化，保持原始精度
  - 优点: 最高精度
  - 缺点: 模型大小最大
  - 适用场景: 对精度要求极高的应用

- **fp16**: 半精度浮点量化
  - 优点: 模型大小减半，精度损失很小
  - 缺点: 需要支持fp16的硬件
  - 适用场景: 现代GPU部署

- **int8**: 8位整数量化
  - 优点: 模型大小减少75%，推理速度快
  - 缺点: 轻微精度损失
  - 适用场景: 生产环境部署

- **int4**: 4位整数量化
  - 优点: 模型大小减少87.5%，极快推理
  - 缺点: 明显精度损失
  - 适用场景: 资源受限环境

**优化级别说明:**

- **minimal**: 最小优化，保持最高兼容性
- **standard**: 标准优化，平衡性能和兼容性
- **aggressive**: 激进优化，最大化性能提升

### 导出格式配置 (Export Format Configuration)

#### pytorch 格式配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | boolean | true | 是否启用PyTorch导出 |
| `save_tokenizer` | boolean | true | 是否保存tokenizer |
| `save_config` | boolean | true | 是否保存模型配置 |

#### onnx 格式配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | boolean | true | 是否启用ONNX导出 |
| `opset_version` | integer | 14 | ONNX操作集版本 |
| `dynamic_axes` | boolean/dict | true | 动态轴配置 |
| `optimize_graph` | boolean | true | 是否优化计算图 |
| `use_external_data_format` | boolean | false | 是否使用外部数据格式 |

**ONNX配置最佳实践:**

```yaml
onnx:
  enabled: true
  opset_version: 14  # 推荐版本，兼容性好
  dynamic_axes:
    input_ids: {0: "batch_size", 1: "sequence_length"}
    attention_mask: {0: "batch_size", 1: "sequence_length"}
    output: {0: "batch_size", 1: "sequence_length"}
  optimize_graph: true
  use_external_data_format: true  # 大模型推荐启用
```

#### tensorrt 格式配置

| 参数 | 类型 | 可选值 | 默认值 | 说明 |
|------|------|--------|--------|------|
| `enabled` | boolean | - | false | 是否启用TensorRT导出 |
| `precision` | string | "fp32", "fp16", "int8" | "fp16" | 精度模式 |
| `max_batch_size` | integer | - | 1 | 最大批次大小 |

### 验证配置 (Validation Configuration)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | boolean | true | 是否启用验证 |
| `test_samples` | integer | 10 | 测试样本数量 |
| `compare_outputs` | boolean | true | 是否比较输出一致性 |
| `benchmark_performance` | boolean | true | 是否进行性能基准测试 |
| `tolerance` | float | 1e-5 | 输出差异容忍度 |

### 输出配置 (Output Configuration)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `directory` | string | "exported_models" | 输出目录 |
| `naming_pattern` | string | "{model_name}_{timestamp}" | 文件命名模式 |
| `create_subdirs` | boolean | true | 是否创建子目录 |
| `cleanup_temp_files` | boolean | true | 是否清理临时文件 |

**命名模式变量:**
- `{model_name}`: 模型名称
- `{timestamp}`: 时间戳
- `{quantization}`: 量化级别
- `{format}`: 导出格式

### 监控配置 (Monitoring Configuration)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_progress_bar` | boolean | true | 是否显示进度条 |
| `log_level` | string | "INFO" | 日志级别 |
| `save_logs` | boolean | true | 是否保存日志文件 |
| `memory_monitoring` | boolean | true | 是否监控内存使用 |

## 环境变量配置

系统支持通过环境变量覆盖配置文件中的参数：

```bash
# 基本配置
export EXPORT_CHECKPOINT_PATH="./qwen3-finetuned"
export EXPORT_BASE_MODEL="Qwen/Qwen3-4B-Thinking-2507"
export EXPORT_OUTPUT_DIR="./exported_models"

# 优化配置
export EXPORT_QUANTIZATION_LEVEL="int8"
export EXPORT_REMOVE_ARTIFACTS="true"
export EXPORT_COMPRESS_WEIGHTS="true"

# 格式配置
export EXPORT_ENABLE_PYTORCH="true"
export EXPORT_ENABLE_ONNX="true"
export EXPORT_ENABLE_TENSORRT="false"

# 监控配置
export EXPORT_LOG_LEVEL="INFO"
export EXPORT_ENABLE_PROGRESS="true"
```

## 使用场景最佳实践

### 场景1: 开发和测试环境

**特点**: 快速迭代，注重调试信息

```yaml
export:
  optimization:
    quantization: "fp16"  # 轻量量化
    optimization_level: "minimal"
  
  formats:
    pytorch:
      enabled: true
    onnx:
      enabled: false  # 跳过ONNX以节省时间
  
  validation:
    enabled: true
    test_samples: 5  # 减少测试样本
  
  monitoring:
    log_level: "DEBUG"  # 详细日志
    enable_progress_bar: true
```

### 场景2: 生产环境部署

**特点**: 优化性能，确保稳定性

```yaml
export:
  optimization:
    quantization: "int8"  # 平衡性能和精度
    optimization_level: "standard"
    remove_artifacts: true
    compress_weights: true
  
  formats:
    pytorch:
      enabled: true
    onnx:
      enabled: true
      optimize_graph: true
  
  validation:
    enabled: true
    test_samples: 20  # 充分验证
    compare_outputs: true
    benchmark_performance: true
  
  monitoring:
    log_level: "INFO"
    save_logs: true
```

### 场景3: 资源受限环境

**特点**: 最小化模型大小和内存使用

```yaml
export:
  base_model:
    load_in_4bit: true  # 4bit加载
  
  optimization:
    quantization: "int4"  # 激进量化
    optimization_level: "aggressive"
    remove_artifacts: true
    compress_weights: true
  
  formats:
    pytorch:
      enabled: true
    onnx:
      enabled: true
      use_external_data_format: true
  
  validation:
    enabled: true
    tolerance: 1e-3  # 放宽容忍度
```

### 场景4: 高精度要求

**特点**: 保持最高精度，性能次要

```yaml
export:
  optimization:
    quantization: "none"  # 不量化
    optimization_level: "minimal"
    remove_artifacts: false
  
  formats:
    pytorch:
      enabled: true
    onnx:
      enabled: true
      opset_version: 16  # 最新版本
  
  validation:
    enabled: true
    test_samples: 50  # 大量测试
    tolerance: 1e-7  # 严格容忍度
```

## 性能调优建议

### 内存优化

1. **使用量化加载**:
   ```yaml
   base_model:
     load_in_4bit: true  # 或 load_in_8bit: true
   ```

2. **启用梯度检查点**:
   ```yaml
   optimization:
     use_gradient_checkpointing: true
   ```

3. **分批处理**:
   ```yaml
   processing:
     batch_size: 1
     max_sequence_length: 2048
   ```

### 速度优化

1. **并行处理**:
   ```yaml
   processing:
     num_workers: 4
     parallel_export: true
   ```

2. **跳过不必要的验证**:
   ```yaml
   validation:
     enabled: false  # 仅在必要时启用
   ```

3. **使用SSD存储**:
   ```yaml
   output:
     directory: "/fast_ssd/exported_models"
   ```

### 精度优化

1. **选择合适的量化级别**:
   ```yaml
   optimization:
     quantization: "int8"  # 平衡点
   ```

2. **使用校准数据集**:
   ```yaml
   optimization:
     calibration_dataset: "path/to/calibration_data"
   ```

3. **启用输出比较**:
   ```yaml
   validation:
     compare_outputs: true
     tolerance: 1e-5
   ```

## 常见配置错误

### 错误1: 内存不足

**症状**: OOM错误
**解决方案**:
```yaml
base_model:
  load_in_4bit: true
  device_map: "auto"
optimization:
  quantization: "int8"
```

### 错误2: ONNX导出失败

**症状**: 不支持的操作符
**解决方案**:
```yaml
onnx:
  opset_version: 14  # 降低版本
  optimize_graph: false  # 禁用优化
```

### 错误3: 精度损失过大

**症状**: 验证失败
**解决方案**:
```yaml
optimization:
  quantization: "fp16"  # 降低量化级别
validation:
  tolerance: 1e-3  # 放宽容忍度
```

### 错误4: 导出速度慢

**症状**: 导出时间过长
**解决方案**:
```yaml
validation:
  enabled: false  # 跳过验证
formats:
  onnx:
    enabled: false  # 跳过ONNX
```

## 配置验证

使用内置的配置验证工具检查配置文件：

```bash
python -m src.cli validate-config export_config.yaml
```

验证工具会检查：
- 参数类型和取值范围
- 路径是否存在
- 硬件兼容性
- 配置冲突

## 配置模板

系统提供了多个预定义的配置模板：

```bash
# 列出可用模板
python -m src.cli list-templates

# 使用模板创建配置
python -m src.cli create-config --template production --output my_config.yaml
```

可用模板：
- `development`: 开发环境配置
- `production`: 生产环境配置
- `minimal`: 最小化配置
- `high-precision`: 高精度配置
- `resource-constrained`: 资源受限配置

## 配置继承

支持配置文件继承，减少重复配置：

```yaml
# base_config.yaml
base_config: &base
  base_model:
    name: "Qwen/Qwen3-4B-Thinking-2507"
    torch_dtype: "float16"
  
  monitoring:
    log_level: "INFO"

# production_config.yaml
<<: *base
export:
  optimization:
    quantization: "int8"
  formats:
    pytorch:
      enabled: true
    onnx:
      enabled: true
```

通过合理的配置，可以显著提升模型导出的效率和质量。建议根据具体使用场景选择合适的配置参数，并在实际使用中不断调优。
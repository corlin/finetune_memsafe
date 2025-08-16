# API参考文档

## 概述

模型导出优化系统提供了一套完整的API，用于将训练完成的LoRA checkpoint与基座模型合并，并导出为多种优化格式。本文档详细介绍了所有可用的API接口和使用方法。

## 核心组件API

### CheckpointDetector

检测和验证训练checkpoint的组件。

#### 类定义

```python
from src.checkpoint_detector import CheckpointDetector

detector = CheckpointDetector()
```

#### 方法

##### `detect_latest_checkpoint(checkpoint_dir: str) -> str`

自动检测指定目录中最新的checkpoint。

**参数:**
- `checkpoint_dir` (str): checkpoint目录路径

**返回值:**
- str: 最新checkpoint的完整路径

**示例:**
```python
latest_checkpoint = detector.detect_latest_checkpoint("./qwen3-finetuned")
print(f"最新checkpoint: {latest_checkpoint}")
```

##### `validate_checkpoint_integrity(checkpoint_path: str) -> bool`

验证checkpoint文件的完整性。

**参数:**
- `checkpoint_path` (str): checkpoint路径

**返回值:**
- bool: 验证结果，True表示完整

**示例:**
```python
is_valid = detector.validate_checkpoint_integrity("./qwen3-finetuned/checkpoint-30")
if not is_valid:
    print("Checkpoint文件不完整或损坏")
```

##### `get_checkpoint_metadata(checkpoint_path: str) -> dict`

获取checkpoint的元数据信息。

**参数:**
- `checkpoint_path` (str): checkpoint路径

**返回值:**
- dict: 包含训练步数、损失值等元数据

**示例:**
```python
metadata = detector.get_checkpoint_metadata("./qwen3-finetuned/checkpoint-30")
print(f"训练步数: {metadata.get('step', 'N/A')}")
print(f"损失值: {metadata.get('loss', 'N/A')}")
```

### ModelMerger

合并LoRA适配器与基座模型的组件。

#### 类定义

```python
from src.model_merger import ModelMerger

merger = ModelMerger()
```

#### 方法

##### `load_base_model(model_name: str, **kwargs) -> AutoModelForCausalLM`

加载基座模型。

**参数:**
- `model_name` (str): 模型名称或路径
- `**kwargs`: 额外的模型加载参数

**返回值:**
- AutoModelForCausalLM: 加载的基座模型

**示例:**
```python
base_model = merger.load_base_model(
    "Qwen/Qwen3-4B-Thinking-2507",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

##### `merge_lora_weights(base_model: AutoModelForCausalLM, adapter_path: str) -> AutoModelForCausalLM`

将LoRA适配器权重合并到基座模型。

**参数:**
- `base_model` (AutoModelForCausalLM): 基座模型
- `adapter_path` (str): LoRA适配器路径

**返回值:**
- AutoModelForCausalLM: 合并后的模型

**示例:**
```python
merged_model = merger.merge_lora_weights(base_model, "./qwen3-finetuned")
```

##### `save_merged_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, output_path: str) -> None`

保存合并后的模型。

**参数:**
- `model` (AutoModelForCausalLM): 要保存的模型
- `tokenizer` (AutoTokenizer): 对应的tokenizer
- `output_path` (str): 输出路径

**示例:**
```python
merger.save_merged_model(merged_model, tokenizer, "./output/merged_model")
```

### OptimizationProcessor

模型优化处理组件。

#### 类定义

```python
from src.optimization_processor import OptimizationProcessor

processor = OptimizationProcessor()
```

#### 方法

##### `apply_quantization(model: AutoModelForCausalLM, quant_config: dict) -> AutoModelForCausalLM`

应用量化优化。

**参数:**
- `model` (AutoModelForCausalLM): 要量化的模型
- `quant_config` (dict): 量化配置

**返回值:**
- AutoModelForCausalLM: 量化后的模型

**示例:**
```python
quant_config = {
    "quantization_level": "int8",
    "calibration_dataset": None
}
quantized_model = processor.apply_quantization(model, quant_config)
```

##### `remove_training_artifacts(model: AutoModelForCausalLM) -> AutoModelForCausalLM`

移除训练相关的冗余参数。

**参数:**
- `model` (AutoModelForCausalLM): 要清理的模型

**返回值:**
- AutoModelForCausalLM: 清理后的模型

**示例:**
```python
cleaned_model = processor.remove_training_artifacts(model)
```

### FormatExporter

多格式模型导出组件。

#### 类定义

```python
from src.format_exporter import FormatExporter

exporter = FormatExporter()
```

#### 方法

##### `export_pytorch_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, output_path: str) -> str`

导出PyTorch格式模型。

**参数:**
- `model` (AutoModelForCausalLM): 要导出的模型
- `tokenizer` (AutoTokenizer): 对应的tokenizer
- `output_path` (str): 输出路径

**返回值:**
- str: 导出模型的路径

**示例:**
```python
pytorch_path = exporter.export_pytorch_model(model, tokenizer, "./output/pytorch_model")
```

##### `export_onnx_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, output_path: str, onnx_config: dict) -> str`

导出ONNX格式模型。

**参数:**
- `model` (AutoModelForCausalLM): 要导出的模型
- `tokenizer` (AutoTokenizer): 对应的tokenizer
- `output_path` (str): 输出路径
- `onnx_config` (dict): ONNX导出配置

**返回值:**
- str: 导出ONNX模型的路径

**示例:**
```python
onnx_config = {
    "opset_version": 14,
    "dynamic_axes": {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"}
    }
}
onnx_path = exporter.export_onnx_model(model, tokenizer, "./output/onnx_model", onnx_config)
```

### ValidationTester

模型验证测试组件。

#### 类定义

```python
from src.validation_tester import ValidationTester

tester = ValidationTester()
```

#### 方法

##### `test_model_functionality(model_path: str, test_inputs: List[str]) -> dict`

测试模型基本功能。

**参数:**
- `model_path` (str): 模型路径
- `test_inputs` (List[str]): 测试输入样本

**返回值:**
- dict: 测试结果

**示例:**
```python
test_inputs = ["你好，请介绍一下自己", "什么是人工智能？"]
results = tester.test_model_functionality("./output/pytorch_model", test_inputs)
```

##### `compare_model_outputs(model1_path: str, model2_path: str, test_inputs: List[str]) -> dict`

比较两个模型的输出一致性。

**参数:**
- `model1_path` (str): 第一个模型路径
- `model2_path` (str): 第二个模型路径
- `test_inputs` (List[str]): 测试输入样本

**返回值:**
- dict: 比较结果

**示例:**
```python
comparison = tester.compare_model_outputs(
    "./output/pytorch_model",
    "./output/onnx_model",
    test_inputs
)
```

### ModelExportController

主导出控制器，整合所有组件。

#### 类定义

```python
from src.model_export_controller import ModelExportController

controller = ModelExportController()
```

#### 方法

##### `export_model(config: ExportConfiguration) -> ExportResult`

执行完整的模型导出流程。

**参数:**
- `config` (ExportConfiguration): 导出配置

**返回值:**
- ExportResult: 导出结果

**示例:**
```python
from src.export_config import ExportConfiguration

config = ExportConfiguration(
    checkpoint_path="./qwen3-finetuned",
    base_model_name="Qwen/Qwen3-4B-Thinking-2507",
    output_directory="./exported_models",
    quantization_level="int8",
    export_pytorch=True,
    export_onnx=True
)

result = controller.export_model(config)
print(f"导出成功: {result.success}")
print(f"PyTorch模型路径: {result.pytorch_model_path}")
print(f"ONNX模型路径: {result.onnx_model_path}")
```

## 配置类API

### ExportConfiguration

导出配置数据类。

#### 属性

```python
@dataclass
class ExportConfiguration:
    # 必需参数
    checkpoint_path: str              # checkpoint路径
    base_model_name: str             # 基座模型名称
    output_directory: str            # 输出目录
    
    # 优化配置
    quantization_level: str = "int8"          # 量化级别: "none", "fp16", "int8", "int4"
    remove_training_artifacts: bool = True    # 是否移除训练artifacts
    compress_weights: bool = True             # 是否压缩权重
    
    # 导出格式
    export_pytorch: bool = True      # 是否导出PyTorch格式
    export_onnx: bool = True        # 是否导出ONNX格式
    export_tensorrt: bool = False   # 是否导出TensorRT格式
    
    # ONNX配置
    onnx_dynamic_axes: dict = None          # ONNX动态轴配置
    onnx_opset_version: int = 14           # ONNX操作集版本
    onnx_optimize_graph: bool = True       # 是否优化ONNX图
    
    # 验证配置
    run_validation_tests: bool = True       # 是否运行验证测试
    test_input_samples: List[str] = None   # 测试输入样本
    
    # 监控配置
    enable_progress_monitoring: bool = True # 是否启用进度监控
    log_level: str = "INFO"                # 日志级别
```

#### 示例

```python
# 基本配置
config = ExportConfiguration(
    checkpoint_path="./qwen3-finetuned",
    base_model_name="Qwen/Qwen3-4B-Thinking-2507",
    output_directory="./exported_models"
)

# 高级配置
advanced_config = ExportConfiguration(
    checkpoint_path="./qwen3-finetuned",
    base_model_name="Qwen/Qwen3-4B-Thinking-2507",
    output_directory="./exported_models",
    quantization_level="int4",
    export_pytorch=True,
    export_onnx=True,
    export_tensorrt=True,
    onnx_dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"}
    },
    test_input_samples=["测试输入1", "测试输入2"],
    log_level="DEBUG"
)
```

### ExportResult

导出结果数据类。

#### 属性

```python
@dataclass
class ExportResult:
    # 基本信息
    export_id: str                    # 导出ID
    timestamp: datetime               # 导出时间戳
    success: bool                     # 是否成功
    
    # 输出路径
    pytorch_model_path: Optional[str] = None    # PyTorch模型路径
    onnx_model_path: Optional[str] = None       # ONNX模型路径
    tensorrt_model_path: Optional[str] = None   # TensorRT模型路径
    
    # 优化统计
    original_size_mb: float           # 原始模型大小(MB)
    optimized_size_mb: float          # 优化后模型大小(MB)
    size_reduction_percentage: float  # 大小减少百分比
    
    # 性能指标
    inference_speed_ms: Optional[float] = None  # 推理速度(毫秒)
    memory_usage_mb: Optional[float] = None     # 内存使用(MB)
    
    # 验证结果
    validation_passed: bool = False             # 验证是否通过
    validation_report_path: Optional[str] = None # 验证报告路径
    
    # 错误信息
    error_message: Optional[str] = None         # 错误消息
    warnings: List[str] = field(default_factory=list) # 警告信息
```

## 异常处理API

### 异常类层次结构

```python
class ModelExportException(Exception):
    """模型导出相关异常的基类"""
    pass

class CheckpointValidationError(ModelExportException):
    """Checkpoint验证错误"""
    pass

class ModelMergeError(ModelExportException):
    """模型合并错误"""
    pass

class FormatExportError(ModelExportException):
    """格式导出错误"""
    pass

class ValidationError(ModelExportException):
    """验证测试错误"""
    pass
```

### 异常处理示例

```python
try:
    result = controller.export_model(config)
except CheckpointValidationError as e:
    print(f"Checkpoint验证失败: {e}")
except ModelMergeError as e:
    print(f"模型合并失败: {e}")
except FormatExportError as e:
    print(f"格式导出失败: {e}")
except ValidationError as e:
    print(f"验证测试失败: {e}")
except ModelExportException as e:
    print(f"导出过程出错: {e}")
```

## 完整使用示例

### 基本使用流程

```python
from src.model_export_controller import ModelExportController
from src.export_config import ExportConfiguration

# 1. 创建配置
config = ExportConfiguration(
    checkpoint_path="./qwen3-finetuned",
    base_model_name="Qwen/Qwen3-4B-Thinking-2507",
    output_directory="./exported_models",
    quantization_level="int8",
    export_pytorch=True,
    export_onnx=True
)

# 2. 创建控制器并执行导出
controller = ModelExportController()
try:
    result = controller.export_model(config)
    
    if result.success:
        print("模型导出成功！")
        print(f"PyTorch模型: {result.pytorch_model_path}")
        print(f"ONNX模型: {result.onnx_model_path}")
        print(f"模型大小减少: {result.size_reduction_percentage:.1f}%")
    else:
        print(f"导出失败: {result.error_message}")
        
except Exception as e:
    print(f"导出过程出错: {e}")
```

### 高级使用示例

```python
import logging
from src.model_export_controller import ModelExportController
from src.export_config import ExportConfiguration

# 配置日志
logging.basicConfig(level=logging.INFO)

# 创建高级配置
config = ExportConfiguration(
    checkpoint_path="./qwen3-finetuned",
    base_model_name="Qwen/Qwen3-4B-Thinking-2507",
    output_directory="./exported_models",
    
    # 优化配置
    quantization_level="int4",
    remove_training_artifacts=True,
    compress_weights=True,
    
    # 多格式导出
    export_pytorch=True,
    export_onnx=True,
    export_tensorrt=False,
    
    # ONNX优化
    onnx_dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"}
    },
    onnx_optimize_graph=True,
    
    # 验证配置
    run_validation_tests=True,
    test_input_samples=[
        "你好，请介绍一下自己",
        "什么是人工智能？",
        "请解释深度学习的基本概念"
    ],
    
    # 监控配置
    enable_progress_monitoring=True,
    log_level="DEBUG"
)

# 执行导出
controller = ModelExportController()
result = controller.export_model(config)

# 处理结果
if result.success:
    print("=== 导出成功 ===")
    print(f"导出ID: {result.export_id}")
    print(f"导出时间: {result.timestamp}")
    
    if result.pytorch_model_path:
        print(f"PyTorch模型: {result.pytorch_model_path}")
    if result.onnx_model_path:
        print(f"ONNX模型: {result.onnx_model_path}")
    
    print(f"\n=== 优化统计 ===")
    print(f"原始大小: {result.original_size_mb:.1f} MB")
    print(f"优化后大小: {result.optimized_size_mb:.1f} MB")
    print(f"大小减少: {result.size_reduction_percentage:.1f}%")
    
    if result.inference_speed_ms:
        print(f"推理速度: {result.inference_speed_ms:.2f} ms")
    if result.memory_usage_mb:
        print(f"内存使用: {result.memory_usage_mb:.1f} MB")
    
    print(f"\n=== 验证结果 ===")
    print(f"验证通过: {result.validation_passed}")
    if result.validation_report_path:
        print(f"验证报告: {result.validation_report_path}")
    
    if result.warnings:
        print(f"\n=== 警告信息 ===")
        for warning in result.warnings:
            print(f"- {warning}")
else:
    print("=== 导出失败 ===")
    print(f"错误信息: {result.error_message}")
    if result.warnings:
        print("警告信息:")
        for warning in result.warnings:
            print(f"- {warning}")
```

## 注意事项

1. **内存管理**: 处理大型模型时，确保有足够的GPU内存和系统内存
2. **路径处理**: 所有路径参数支持相对路径和绝对路径
3. **错误处理**: 建议使用try-catch块处理可能的异常
4. **日志配置**: 可以通过配置日志级别来控制输出详细程度
5. **并发限制**: 避免同时运行多个导出任务，可能导致内存不足

## 版本兼容性

- Python: >= 3.8
- PyTorch: >= 1.12.0
- Transformers: >= 4.21.0
- ONNX: >= 1.12.0
- ONNX Runtime: >= 1.12.0

更多详细信息请参考[配置参数说明](CONFIGURATION_GUIDE.md)和[故障排除指南](TROUBLESHOOTING_GUIDE.md)。
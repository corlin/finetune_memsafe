"""
模型导出系统的核心数据模型

本模块定义了模型导出过程中使用的核心数据结构，包括配置模型和结果模型。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import os


class QuantizationLevel(Enum):
    """量化级别枚举"""
    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ExportConfiguration:
    """导出配置数据模型"""
    
    # 基本配置
    checkpoint_path: str
    base_model_name: str
    output_directory: str
    
    # 优化配置
    quantization_level: QuantizationLevel = QuantizationLevel.INT8
    remove_training_artifacts: bool = True
    compress_weights: bool = True
    
    # 导出格式
    export_pytorch: bool = True
    export_onnx: bool = True
    export_tensorrt: bool = False
    
    # ONNX配置
    onnx_dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    onnx_opset_version: int = 20
    onnx_optimize_graph: bool = True
    
    # 验证配置
    run_validation_tests: bool = True
    test_input_samples: Optional[List[str]] = None
    
    # 监控配置
    enable_progress_monitoring: bool = True
    log_level: LogLevel = LogLevel.INFO
    
    # 高级配置
    auto_detect_latest_checkpoint: bool = True
    save_tokenizer: bool = True
    naming_pattern: str = "{model_name}_{timestamp}"
    max_memory_usage_gb: float = 16.0
    enable_parallel_export: bool = False
    
    def __post_init__(self):
        """初始化后的验证和设置默认值"""
        # 确保输出目录存在
        os.makedirs(self.output_directory, exist_ok=True)
        
        # 设置默认的ONNX动态轴
        if self.onnx_dynamic_axes is None:
            self.onnx_dynamic_axes = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
        
        # 设置默认测试样本
        if self.test_input_samples is None:
            self.test_input_samples = [
                "Hello, how are you?",
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "Write a short story about a robot.",
                "What are the benefits of renewable energy?"
            ]
    
    def validate(self) -> List[str]:
        """验证配置的有效性，返回错误信息列表"""
        errors = []
        
        # 检查必需的路径
        if not self.checkpoint_path:
            errors.append("checkpoint_path不能为空")
        elif not os.path.exists(self.checkpoint_path):
            errors.append(f"checkpoint路径不存在: {self.checkpoint_path}")
        
        if not self.base_model_name:
            errors.append("base_model_name不能为空")
        
        if not self.output_directory:
            errors.append("output_directory不能为空")
        
        # 检查导出格式
        if not any([self.export_pytorch, self.export_onnx, self.export_tensorrt]):
            errors.append("至少需要启用一种导出格式")
        
        # 检查ONNX配置
        if self.export_onnx:
            if self.onnx_opset_version < 11:
                errors.append("ONNX opset版本应该至少为11")
        
        # 检查内存限制
        if self.max_memory_usage_gb <= 0:
            errors.append("max_memory_usage_gb必须大于0")
        
        return errors


@dataclass
class ExportResult:
    """导出结果数据模型"""
    
    # 基本信息
    export_id: str
    timestamp: datetime
    success: bool
    configuration: ExportConfiguration
    
    # 输出路径
    pytorch_model_path: Optional[str] = None
    onnx_model_path: Optional[str] = None
    tensorrt_model_path: Optional[str] = None
    
    # 优化统计
    original_size_mb: float = 0.0
    optimized_size_mb: float = 0.0
    size_reduction_percentage: float = 0.0
    
    # 性能指标
    inference_speed_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # 验证结果
    validation_passed: bool = False
    validation_report_path: Optional[str] = None
    output_consistency_score: Optional[float] = None
    
    # 过程统计
    total_duration_seconds: float = 0.0
    merge_duration_seconds: float = 0.0
    optimization_duration_seconds: float = 0.0
    export_duration_seconds: float = 0.0
    validation_duration_seconds: float = 0.0
    
    # 错误信息
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # 详细日志
    log_entries: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_warning(self, message: str):
        """添加警告信息"""
        self.warnings.append(message)
    
    def add_log_entry(self, level: str, message: str, component: str = ""):
        """添加日志条目"""
        self.log_entries.append({
            'timestamp': datetime.now(),
            'level': level,
            'component': component,
            'message': message
        })
    
    def calculate_size_reduction(self):
        """计算大小减少百分比"""
        if self.original_size_mb > 0:
            self.size_reduction_percentage = (
                (self.original_size_mb - self.optimized_size_mb) / self.original_size_mb * 100
            )
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """转换为摘要字典格式"""
        return {
            'export_id': self.export_id,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'exported_formats': {
                'pytorch': self.pytorch_model_path is not None,
                'onnx': self.onnx_model_path is not None,
                'tensorrt': self.tensorrt_model_path is not None
            },
            'optimization': {
                'original_size_mb': self.original_size_mb,
                'optimized_size_mb': self.optimized_size_mb,
                'size_reduction_percentage': self.size_reduction_percentage
            },
            'performance': {
                'inference_speed_ms': self.inference_speed_ms,
                'memory_usage_mb': self.memory_usage_mb
            },
            'validation': {
                'passed': self.validation_passed,
                'consistency_score': self.output_consistency_score
            },
            'duration': {
                'total_seconds': self.total_duration_seconds,
                'merge_seconds': self.merge_duration_seconds,
                'optimization_seconds': self.optimization_duration_seconds,
                'export_seconds': self.export_duration_seconds,
                'validation_seconds': self.validation_duration_seconds
            },
            'warnings_count': len(self.warnings),
            'error_message': self.error_message
        }


@dataclass
class CheckpointInfo:
    """Checkpoint信息数据模型"""
    
    path: str
    timestamp: datetime
    size_mb: float
    is_valid: bool
    
    # 元数据
    adapter_config: Optional[Dict[str, Any]] = None
    training_args: Optional[Dict[str, Any]] = None
    model_config: Optional[Dict[str, Any]] = None
    
    # 文件信息
    has_adapter_model: bool = False
    has_adapter_config: bool = False
    has_tokenizer: bool = False
    
    # 验证结果
    validation_errors: List[str] = field(default_factory=list)
    
    def add_validation_error(self, error: str):
        """添加验证错误"""
        self.validation_errors.append(error)
        self.is_valid = False


@dataclass
class ValidationResult:
    """验证结果数据模型"""
    
    test_name: str
    success: bool
    score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        if not self.success and not self.error_message:
            self.error_message = "测试失败，未提供具体错误信息"
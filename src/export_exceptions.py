"""
模型导出系统的异常定义

本模块定义了模型导出过程中可能出现的各种异常类型。
"""


class ModelExportException(Exception):
    """模型导出相关异常的基类"""
    
    def __init__(self, message: str, details: str = None, suggestions: list = None):
        super().__init__(message)
        self.message = message
        self.details = details or ""
        self.suggestions = suggestions or []
    
    def __str__(self):
        result = self.message
        if self.details:
            result += f"\n详细信息: {self.details}"
        if self.suggestions:
            result += f"\n建议解决方案: {'; '.join(self.suggestions)}"
        return result


class ModelExportError(ModelExportException):
    """通用模型导出错误"""
    
    def __init__(self, message: str, phase: str = None, export_id: str = None):
        suggestions = [
            "检查日志文件获取详细错误信息",
            "验证输入参数和配置",
            "确保系统资源充足",
            "尝试重新运行导出过程"
        ]
        
        details = ""
        if export_id:
            details += f"导出ID: {export_id}"
        if phase:
            details += f"\n失败阶段: {phase}"
        
        super().__init__(message, details, suggestions)
        self.phase = phase
        self.export_id = export_id


class CheckpointValidationError(ModelExportException):
    """Checkpoint验证错误"""
    
    def __init__(self, message: str, checkpoint_path: str = None, missing_files: list = None):
        suggestions = []
        if missing_files:
            suggestions.append(f"确保以下文件存在: {', '.join(missing_files)}")
        suggestions.extend([
            "检查checkpoint是否完整训练完成",
            "验证checkpoint目录结构是否正确",
            "尝试重新训练或使用其他checkpoint"
        ])
        
        details = f"Checkpoint路径: {checkpoint_path}" if checkpoint_path else None
        if missing_files:
            details += f"\n缺失文件: {', '.join(missing_files)}"
        
        super().__init__(message, details, suggestions)
        self.checkpoint_path = checkpoint_path
        self.missing_files = missing_files or []


class ModelMergeError(ModelExportException):
    """模型合并错误"""
    
    def __init__(self, message: str, base_model: str = None, adapter_path: str = None):
        suggestions = [
            "检查基座模型和LoRA适配器的兼容性",
            "确保有足够的内存进行模型合并",
            "尝试使用较小的批处理大小",
            "检查模型权重的数据类型是否匹配"
        ]
        
        details = ""
        if base_model:
            details += f"基座模型: {base_model}"
        if adapter_path:
            details += f"\nLoRA适配器: {adapter_path}"
        
        super().__init__(message, details, suggestions)
        self.base_model = base_model
        self.adapter_path = adapter_path


class OptimizationError(ModelExportException):
    """模型优化错误"""
    
    def __init__(self, message: str, optimization_type: str = None):
        suggestions = [
            "尝试使用较低的量化级别",
            "检查模型是否支持所选的优化方法",
            "确保有足够的内存进行优化处理",
            "考虑分批处理大型模型"
        ]
        
        details = f"优化类型: {optimization_type}" if optimization_type else None
        
        super().__init__(message, details, suggestions)
        self.optimization_type = optimization_type


class FormatExportError(ModelExportException):
    """格式导出错误"""
    
    def __init__(self, message: str, export_format: str = None, unsupported_ops: list = None):
        suggestions = [
            "检查模型是否包含不支持的操作符",
            "尝试使用不同的ONNX opset版本",
            "考虑简化模型结构",
            "查看详细的转换日志"
        ]
        
        if export_format == "onnx" and unsupported_ops:
            suggestions.insert(0, f"以下操作符不支持: {', '.join(unsupported_ops)}")
        
        details = f"导出格式: {export_format}" if export_format else None
        if unsupported_ops:
            details += f"\n不支持的操作符: {', '.join(unsupported_ops)}"
        
        super().__init__(message, details, suggestions)
        self.export_format = export_format
        self.unsupported_ops = unsupported_ops or []


class ValidationError(ModelExportException):
    """验证测试错误"""
    
    def __init__(self, message: str, test_name: str = None, expected_vs_actual: dict = None):
        suggestions = [
            "检查模型输出的数值精度设置",
            "验证输入数据的预处理是否一致",
            "比较不同格式模型的配置参数",
            "检查随机种子设置"
        ]
        
        details = f"测试名称: {test_name}" if test_name else None
        if expected_vs_actual:
            details += f"\n期望值 vs 实际值: {expected_vs_actual}"
        
        super().__init__(message, details, suggestions)
        self.test_name = test_name
        self.expected_vs_actual = expected_vs_actual or {}


class ConfigurationError(ModelExportException):
    """配置错误"""
    
    def __init__(self, message: str, invalid_params: list = None):
        suggestions = [
            "检查配置文件的语法和格式",
            "验证所有必需参数是否已设置",
            "参考配置模板文件",
            "检查环境变量设置"
        ]
        
        details = None
        if invalid_params:
            details = f"无效参数: {', '.join(invalid_params)}"
        
        super().__init__(message, details, suggestions)
        self.invalid_params = invalid_params or []


class MemoryError(ModelExportException):
    """内存不足错误"""
    
    def __init__(self, message: str, required_memory_gb: float = None, available_memory_gb: float = None):
        suggestions = [
            "增加系统内存或使用更大内存的机器",
            "启用模型量化以减少内存使用",
            "使用CPU卸载功能",
            "分批处理模型组件",
            "关闭其他占用内存的程序"
        ]
        
        details = ""
        if required_memory_gb:
            details += f"需要内存: {required_memory_gb:.1f}GB"
        if available_memory_gb:
            details += f"\n可用内存: {available_memory_gb:.1f}GB"
        
        super().__init__(message, details, suggestions)
        self.required_memory_gb = required_memory_gb
        self.available_memory_gb = available_memory_gb


class DiskSpaceError(ModelExportException):
    """磁盘空间不足错误"""
    
    def __init__(self, message: str, required_space_gb: float = None, available_space_gb: float = None):
        suggestions = [
            "清理磁盘空间",
            "选择其他输出目录",
            "启用模型压缩以减少输出大小",
            "删除不需要的临时文件"
        ]
        
        details = ""
        if required_space_gb:
            details += f"需要空间: {required_space_gb:.1f}GB"
        if available_space_gb:
            details += f"\n可用空间: {available_space_gb:.1f}GB"
        
        super().__init__(message, details, suggestions)
        self.required_space_gb = required_space_gb
        self.available_space_gb = available_space_gb


class DependencyError(ModelExportException):
    """依赖项错误"""
    
    def __init__(self, message: str, missing_packages: list = None, version_conflicts: dict = None):
        suggestions = [
            "安装缺失的依赖包",
            "更新到兼容的包版本",
            "检查虚拟环境配置",
            "参考requirements.txt文件"
        ]
        
        details = ""
        if missing_packages:
            details += f"缺失包: {', '.join(missing_packages)}"
        if version_conflicts:
            details += f"\n版本冲突: {version_conflicts}"
        
        super().__init__(message, details, suggestions)
        self.missing_packages = missing_packages or []
        self.version_conflicts = version_conflicts or {}


class TimeoutError(ModelExportException):
    """超时错误"""
    
    def __init__(self, message: str, timeout_seconds: int = None, operation: str = None):
        suggestions = [
            "增加超时时间限制",
            "检查系统资源使用情况",
            "尝试使用更快的硬件",
            "优化模型大小以减少处理时间"
        ]
        
        details = ""
        if operation:
            details += f"操作: {operation}"
        if timeout_seconds:
            details += f"\n超时时间: {timeout_seconds}秒"
        
        super().__init__(message, details, suggestions)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class NetworkError(ModelExportException):
    """网络错误"""
    
    def __init__(self, message: str, url: str = None, status_code: int = None):
        suggestions = [
            "检查网络连接",
            "验证URL是否正确",
            "尝试使用代理或VPN",
            "检查防火墙设置",
            "稍后重试"
        ]
        
        details = ""
        if url:
            details += f"URL: {url}"
        if status_code:
            details += f"\n状态码: {status_code}"
        
        super().__init__(message, details, suggestions)
        self.url = url
        self.status_code = status_code
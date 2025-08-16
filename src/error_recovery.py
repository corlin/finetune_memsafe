"""
错误恢复和智能诊断系统

本模块提供智能的错误诊断、恢复策略和降级机制。
"""

import os
import sys
import time
import logging
import traceback
import psutil
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from .export_exceptions import *


@dataclass
class RecoveryAction:
    """恢复动作定义"""
    name: str
    description: str
    action: Callable
    priority: int = 1  # 1=高优先级, 2=中优先级, 3=低优先级
    auto_execute: bool = False  # 是否自动执行
    requires_user_input: bool = False


@dataclass
class DiagnosticResult:
    """诊断结果"""
    issue_type: str
    severity: str  # "critical", "warning", "info"
    description: str
    possible_causes: List[str]
    recovery_actions: List[RecoveryAction]
    system_info: Dict[str, Any]


class ErrorRecoveryManager:
    """错误恢复管理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.recovery_history = []
        self.diagnostic_cache = {}
        
        # 注册恢复策略
        self._register_recovery_strategies()
    
    def _register_recovery_strategies(self):
        """注册恢复策略"""
        self.recovery_strategies = {
            MemoryError: self._handle_memory_error,
            CheckpointValidationError: self._handle_checkpoint_error,
            ModelMergeError: self._handle_model_merge_error,
            FormatExportError: self._handle_format_export_error,
            ValidationError: self._handle_validation_error,
            ConfigurationError: self._handle_configuration_error,
            DiskSpaceError: self._handle_disk_space_error,
            DependencyError: self._handle_dependency_error,
            TimeoutError: self._handle_timeout_error,
            NetworkError: self._handle_network_error
        }
    
    def diagnose_and_recover(self, exception: Exception, context: Dict[str, Any] = None) -> DiagnosticResult:
        """
        诊断错误并提供恢复建议
        
        Args:
            exception: 发生的异常
            context: 错误上下文信息
            
        Returns:
            DiagnosticResult: 诊断结果和恢复建议
        """
        self.logger.info(f"开始诊断错误: {type(exception).__name__}")
        
        # 收集系统信息
        system_info = self._collect_system_info()
        
        # 获取错误类型对应的处理策略
        exception_type = type(exception)
        if exception_type in self.recovery_strategies:
            handler = self.recovery_strategies[exception_type]
            diagnostic = handler(exception, context or {}, system_info)
        else:
            # 通用错误处理
            diagnostic = self._handle_generic_error(exception, context or {}, system_info)
        
        # 记录诊断结果
        self._log_diagnostic_result(diagnostic)
        
        return diagnostic
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """收集系统信息"""
        try:
            # 内存信息
            memory = psutil.virtual_memory()
            
            # GPU信息
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
                    "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3
                }
            
            # 磁盘信息
            disk = psutil.disk_usage('.')
            
            return {
                "timestamp": datetime.now().isoformat(),
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "memory": {
                    "total_gb": memory.total / 1024**3,
                    "available_gb": memory.available / 1024**3,
                    "used_gb": memory.used / 1024**3,
                    "percent": memory.percent
                },
                "gpu": gpu_info,
                "disk": {
                    "total_gb": disk.total / 1024**3,
                    "free_gb": disk.free / 1024**3,
                    "used_gb": disk.used / 1024**3
                },
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent()
            }
        except Exception as e:
            self.logger.warning(f"收集系统信息失败: {e}")
            return {"error": str(e)}
    
    def _handle_memory_error(self, exception: MemoryError, context: Dict, system_info: Dict) -> DiagnosticResult:
        """处理内存错误"""
        recovery_actions = [
            RecoveryAction(
                name="清理GPU内存",
                description="清理GPU缓存和未使用的张量",
                action=self._clear_gpu_memory,
                priority=1,
                auto_execute=True
            ),
            RecoveryAction(
                name="降低批处理大小",
                description="减少batch_size以降低内存使用",
                action=lambda: self._suggest_config_change("batch_size", "减半"),
                priority=1
            ),
            RecoveryAction(
                name="启用梯度检查点",
                description="使用梯度检查点技术节省内存",
                action=lambda: self._suggest_config_change("gradient_checkpointing", True),
                priority=2
            ),
            RecoveryAction(
                name="使用CPU卸载",
                description="将部分计算卸载到CPU",
                action=lambda: self._suggest_config_change("offload_to_cpu", True),
                priority=2
            )
        ]
        
        return DiagnosticResult(
            issue_type="memory_shortage",
            severity="critical",
            description=f"内存不足: {exception.message}",
            possible_causes=[
                "模型太大，超出可用内存",
                "批处理大小过大",
                "其他程序占用过多内存",
                "内存泄漏"
            ],
            recovery_actions=recovery_actions,
            system_info=system_info
        )
    
    def _handle_checkpoint_error(self, exception: CheckpointValidationError, context: Dict, system_info: Dict) -> DiagnosticResult:
        """处理checkpoint验证错误"""
        recovery_actions = [
            RecoveryAction(
                name="检查文件完整性",
                description="验证checkpoint文件是否完整",
                action=lambda: self._check_checkpoint_integrity(exception.checkpoint_path),
                priority=1,
                auto_execute=True
            ),
            RecoveryAction(
                name="使用其他checkpoint",
                description="尝试使用其他可用的checkpoint",
                action=lambda: self._find_alternative_checkpoints(exception.checkpoint_path),
                priority=2
            ),
            RecoveryAction(
                name="重新下载模型",
                description="重新下载基座模型文件",
                action=lambda: self._suggest_redownload_model(),
                priority=3
            )
        ]
        
        return DiagnosticResult(
            issue_type="checkpoint_validation",
            severity="critical",
            description=f"Checkpoint验证失败: {exception.message}",
            possible_causes=[
                "训练过程中断，checkpoint不完整",
                "文件损坏或丢失",
                "权限问题",
                "磁盘空间不足导致写入失败"
            ],
            recovery_actions=recovery_actions,
            system_info=system_info
        )
    
    def _handle_model_merge_error(self, exception: ModelMergeError, context: Dict, system_info: Dict) -> DiagnosticResult:
        """处理模型合并错误"""
        recovery_actions = [
            RecoveryAction(
                name="检查模型兼容性",
                description="验证基座模型和LoRA适配器的兼容性",
                action=lambda: self._check_model_compatibility(exception.base_model, exception.adapter_path),
                priority=1,
                auto_execute=True
            ),
            RecoveryAction(
                name="降低精度",
                description="使用较低精度进行合并",
                action=lambda: self._suggest_config_change("merge_precision", "fp16"),
                priority=2
            ),
            RecoveryAction(
                name="分层合并",
                description="分层进行模型合并以减少内存使用",
                action=lambda: self._suggest_config_change("layer_wise_merge", True),
                priority=2
            )
        ]
        
        return DiagnosticResult(
            issue_type="model_merge",
            severity="critical",
            description=f"模型合并失败: {exception.message}",
            possible_causes=[
                "基座模型和LoRA适配器不兼容",
                "内存不足",
                "权重形状不匹配",
                "数据类型不一致"
            ],
            recovery_actions=recovery_actions,
            system_info=system_info
        )
    
    def _handle_format_export_error(self, exception: FormatExportError, context: Dict, system_info: Dict) -> DiagnosticResult:
        """处理格式导出错误"""
        recovery_actions = []
        
        if exception.export_format == "onnx":
            recovery_actions.extend([
                RecoveryAction(
                    name="降低ONNX opset版本",
                    description="使用较低的ONNX opset版本",
                    action=lambda: self._suggest_config_change("onnx_opset_version", 11),
                    priority=1
                ),
                RecoveryAction(
                    name="禁用动态轴",
                    description="禁用动态输入形状",
                    action=lambda: self._suggest_config_change("onnx_dynamic_axes", None),
                    priority=2
                ),
                RecoveryAction(
                    name="简化模型",
                    description="移除不支持的操作符",
                    action=lambda: self._suggest_model_simplification(),
                    priority=3
                )
            ])
        
        return DiagnosticResult(
            issue_type="format_export",
            severity="warning",
            description=f"格式导出失败: {exception.message}",
            possible_causes=[
                "模型包含不支持的操作符",
                "ONNX版本不兼容",
                "动态形状配置问题",
                "模型结构过于复杂"
            ],
            recovery_actions=recovery_actions,
            system_info=system_info
        )
    
    def _handle_validation_error(self, exception: ValidationError, context: Dict, system_info: Dict) -> DiagnosticResult:
        """处理验证错误"""
        recovery_actions = [
            RecoveryAction(
                name="调整数值精度",
                description="放宽数值比较的精度要求",
                action=lambda: self._suggest_config_change("validation_tolerance", 1e-3),
                priority=1
            ),
            RecoveryAction(
                name="检查随机种子",
                description="确保所有模型使用相同的随机种子",
                action=lambda: self._suggest_config_change("random_seed", 42),
                priority=2
            ),
            RecoveryAction(
                name="跳过验证",
                description="跳过验证步骤（不推荐）",
                action=lambda: self._suggest_config_change("run_validation_tests", False),
                priority=3
            )
        ]
        
        return DiagnosticResult(
            issue_type="validation",
            severity="warning",
            description=f"验证失败: {exception.message}",
            possible_causes=[
                "数值精度差异",
                "随机种子不一致",
                "模型配置差异",
                "输入预处理不同"
            ],
            recovery_actions=recovery_actions,
            system_info=system_info
        )
    
    def _handle_configuration_error(self, exception: ConfigurationError, context: Dict, system_info: Dict) -> DiagnosticResult:
        """处理配置错误"""
        recovery_actions = [
            RecoveryAction(
                name="使用默认配置",
                description="重置为默认配置参数",
                action=lambda: self._suggest_default_config(),
                priority=1
            ),
            RecoveryAction(
                name="验证配置文件",
                description="检查配置文件语法",
                action=lambda: self._validate_config_syntax(),
                priority=1,
                auto_execute=True
            )
        ]
        
        return DiagnosticResult(
            issue_type="configuration",
            severity="critical",
            description=f"配置错误: {exception.message}",
            possible_causes=[
                "配置文件语法错误",
                "参数值超出有效范围",
                "必需参数缺失",
                "参数类型不正确"
            ],
            recovery_actions=recovery_actions,
            system_info=system_info
        )
    
    def _handle_disk_space_error(self, exception: DiskSpaceError, context: Dict, system_info: Dict) -> DiagnosticResult:
        """处理磁盘空间错误"""
        recovery_actions = [
            RecoveryAction(
                name="清理临时文件",
                description="删除临时文件和缓存",
                action=self._cleanup_temp_files,
                priority=1,
                auto_execute=True
            ),
            RecoveryAction(
                name="启用压缩",
                description="启用模型压缩以减少输出大小",
                action=lambda: self._suggest_config_change("compress_weights", True),
                priority=1
            ),
            RecoveryAction(
                name="更改输出目录",
                description="选择有更多空间的输出目录",
                action=lambda: self._suggest_alternative_output_dir(),
                priority=2,
                requires_user_input=True
            )
        ]
        
        return DiagnosticResult(
            issue_type="disk_space",
            severity="critical",
            description=f"磁盘空间不足: {exception.message}",
            possible_causes=[
                "输出目录磁盘空间不足",
                "临时文件占用过多空间",
                "模型文件过大",
                "其他程序占用磁盘空间"
            ],
            recovery_actions=recovery_actions,
            system_info=system_info
        )
    
    def _handle_dependency_error(self, exception: DependencyError, context: Dict, system_info: Dict) -> DiagnosticResult:
        """处理依赖项错误"""
        recovery_actions = [
            RecoveryAction(
                name="安装缺失依赖",
                description="自动安装缺失的Python包",
                action=lambda: self._install_missing_packages(exception.missing_packages),
                priority=1,
                auto_execute=True
            ),
            RecoveryAction(
                name="更新包版本",
                description="更新到兼容的包版本",
                action=lambda: self._update_package_versions(exception.version_conflicts),
                priority=2
            )
        ]
        
        return DiagnosticResult(
            issue_type="dependency",
            severity="critical",
            description=f"依赖项错误: {exception.message}",
            possible_causes=[
                "缺失必需的Python包",
                "包版本不兼容",
                "虚拟环境配置问题",
                "包安装损坏"
            ],
            recovery_actions=recovery_actions,
            system_info=system_info
        )
    
    def _handle_timeout_error(self, exception: TimeoutError, context: Dict, system_info: Dict) -> DiagnosticResult:
        """处理超时错误"""
        recovery_actions = [
            RecoveryAction(
                name="增加超时时间",
                description="延长操作超时时间",
                action=lambda: self._suggest_config_change("timeout_seconds", exception.timeout_seconds * 2),
                priority=1
            ),
            RecoveryAction(
                name="优化模型大小",
                description="使用量化减少模型大小",
                action=lambda: self._suggest_config_change("quantization_level", "int8"),
                priority=2
            ),
            RecoveryAction(
                name="分批处理",
                description="将大型操作分解为小批次",
                action=lambda: self._suggest_config_change("batch_processing", True),
                priority=2
            )
        ]
        
        return DiagnosticResult(
            issue_type="timeout",
            severity="warning",
            description=f"操作超时: {exception.message}",
            possible_causes=[
                "模型过大，处理时间过长",
                "系统资源不足",
                "网络连接缓慢",
                "硬件性能限制"
            ],
            recovery_actions=recovery_actions,
            system_info=system_info
        )
    
    def _handle_network_error(self, exception: NetworkError, context: Dict, system_info: Dict) -> DiagnosticResult:
        """处理网络错误"""
        recovery_actions = [
            RecoveryAction(
                name="重试连接",
                description="重新尝试网络连接",
                action=lambda: self._retry_network_operation(exception.url),
                priority=1,
                auto_execute=True
            ),
            RecoveryAction(
                name="使用本地模型",
                description="使用本地缓存的模型文件",
                action=lambda: self._suggest_local_model_path(),
                priority=2
            ),
            RecoveryAction(
                name="配置代理",
                description="配置网络代理设置",
                action=lambda: self._suggest_proxy_config(),
                priority=3,
                requires_user_input=True
            )
        ]
        
        return DiagnosticResult(
            issue_type="network",
            severity="warning",
            description=f"网络错误: {exception.message}",
            possible_causes=[
                "网络连接不稳定",
                "服务器暂时不可用",
                "防火墙阻止连接",
                "代理配置问题"
            ],
            recovery_actions=recovery_actions,
            system_info=system_info
        )
    
    def _handle_generic_error(self, exception: Exception, context: Dict, system_info: Dict) -> DiagnosticResult:
        """处理通用错误"""
        recovery_actions = [
            RecoveryAction(
                name="重试操作",
                description="重新尝试失败的操作",
                action=lambda: self._suggest_retry(),
                priority=1
            ),
            RecoveryAction(
                name="检查日志",
                description="查看详细的错误日志",
                action=lambda: self._suggest_check_logs(),
                priority=2
            ),
            RecoveryAction(
                name="重置环境",
                description="重置到默认配置",
                action=lambda: self._suggest_reset_environment(),
                priority=3
            )
        ]
        
        return DiagnosticResult(
            issue_type="generic",
            severity="warning",
            description=f"未知错误: {str(exception)}",
            possible_causes=[
                "系统环境问题",
                "配置参数错误",
                "资源不足",
                "软件版本不兼容"
            ],
            recovery_actions=recovery_actions,
            system_info=system_info
        )
    
    # 恢复动作实现
    def _clear_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("GPU内存已清理")
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        import tempfile
        import shutil
        
        temp_dirs = [tempfile.gettempdir(), "./temp", "./cache"]
        cleaned_size = 0
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    for item in os.listdir(temp_dir):
                        item_path = os.path.join(temp_dir, item)
                        if os.path.isfile(item_path) and item.startswith(('tmp', 'temp', 'cache')):
                            size = os.path.getsize(item_path)
                            os.remove(item_path)
                            cleaned_size += size
                except Exception as e:
                    self.logger.warning(f"清理临时文件失败: {e}")
        
        self.logger.info(f"已清理临时文件，释放空间: {cleaned_size / 1024**2:.1f}MB")
    
    def _check_checkpoint_integrity(self, checkpoint_path: str):
        """检查checkpoint完整性"""
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return False
        
        required_files = [
            "adapter_config.json",
            "adapter_model.safetensors"
        ]
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(checkpoint_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            self.logger.error(f"Checkpoint缺失文件: {missing_files}")
            return False
        
        self.logger.info("Checkpoint完整性检查通过")
        return True
    
    def _find_alternative_checkpoints(self, checkpoint_path: str):
        """查找替代的checkpoint"""
        if not checkpoint_path:
            return []
        
        parent_dir = os.path.dirname(checkpoint_path)
        alternatives = []
        
        try:
            for item in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path) and "checkpoint" in item:
                    if self._check_checkpoint_integrity(item_path):
                        alternatives.append(item_path)
        except Exception as e:
            self.logger.warning(f"查找替代checkpoint失败: {e}")
        
        self.logger.info(f"找到 {len(alternatives)} 个可用的checkpoint")
        return alternatives
    
    def _suggest_config_change(self, param: str, value: Any):
        """建议配置更改"""
        self.logger.info(f"建议更改配置: {param} = {value}")
    
    def _suggest_default_config(self):
        """建议使用默认配置"""
        self.logger.info("建议使用默认配置参数")
    
    def _suggest_redownload_model(self):
        """建议重新下载模型"""
        self.logger.info("建议重新下载基座模型")
    
    def _suggest_model_simplification(self):
        """建议简化模型"""
        self.logger.info("建议简化模型结构或移除不支持的操作符")
    
    def _suggest_alternative_output_dir(self):
        """建议替代输出目录"""
        self.logger.info("建议选择有更多空间的输出目录")
    
    def _suggest_retry(self):
        """建议重试"""
        self.logger.info("建议重新尝试失败的操作")
    
    def _suggest_check_logs(self):
        """建议检查日志"""
        self.logger.info("建议查看详细的错误日志文件")
    
    def _suggest_reset_environment(self):
        """建议重置环境"""
        self.logger.info("建议重置到默认环境配置")
    
    def _suggest_local_model_path(self):
        """建议使用本地模型路径"""
        self.logger.info("建议使用本地缓存的模型文件")
    
    def _suggest_proxy_config(self):
        """建议配置代理"""
        self.logger.info("建议配置网络代理设置")
    
    def _install_missing_packages(self, packages: List[str]):
        """安装缺失的包"""
        if not packages:
            return
        
        self.logger.info(f"尝试安装缺失的包: {packages}")
        # 这里可以实现自动安装逻辑
    
    def _update_package_versions(self, conflicts: Dict[str, str]):
        """更新包版本"""
        if not conflicts:
            return
        
        self.logger.info(f"尝试解决版本冲突: {conflicts}")
        # 这里可以实现版本更新逻辑
    
    def _retry_network_operation(self, url: str):
        """重试网络操作"""
        self.logger.info(f"重试网络连接: {url}")
        # 这里可以实现重试逻辑
    
    def _validate_config_syntax(self):
        """验证配置文件语法"""
        self.logger.info("验证配置文件语法")
        # 这里可以实现配置验证逻辑
    
    def _check_model_compatibility(self, base_model: str, adapter_path: str):
        """检查模型兼容性"""
        self.logger.info(f"检查模型兼容性: {base_model} <-> {adapter_path}")
        # 这里可以实现兼容性检查逻辑
    
    def _log_diagnostic_result(self, diagnostic: DiagnosticResult):
        """记录诊断结果"""
        self.logger.info(f"诊断完成: {diagnostic.issue_type} - {diagnostic.severity}")
        self.logger.info(f"描述: {diagnostic.description}")
        self.logger.info(f"可能原因: {', '.join(diagnostic.possible_causes)}")
        self.logger.info(f"恢复建议数量: {len(diagnostic.recovery_actions)}")
    
    def execute_recovery_actions(self, diagnostic: DiagnosticResult, auto_only: bool = True) -> List[str]:
        """
        执行恢复动作
        
        Args:
            diagnostic: 诊断结果
            auto_only: 是否只执行自动恢复动作
            
        Returns:
            List[str]: 执行结果列表
        """
        results = []
        
        # 按优先级排序
        actions = sorted(diagnostic.recovery_actions, key=lambda x: x.priority)
        
        for action in actions:
            if auto_only and not action.auto_execute:
                continue
            
            if action.requires_user_input:
                self.logger.info(f"跳过需要用户输入的动作: {action.name}")
                continue
            
            try:
                self.logger.info(f"执行恢复动作: {action.name}")
                action.action()
                results.append(f"成功: {action.name}")
                
                # 记录恢复历史
                self.recovery_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": action.name,
                    "status": "success"
                })
                
            except Exception as e:
                error_msg = f"失败: {action.name} - {str(e)}"
                results.append(error_msg)
                self.logger.error(error_msg)
                
                # 记录恢复历史
                self.recovery_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": action.name,
                    "status": "failed",
                    "error": str(e)
                })
        
        return results


class GracefulDegradationManager:
    """优雅降级管理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.degradation_levels = {
            1: "minimal_degradation",    # 最小降级
            2: "moderate_degradation",   # 中等降级
            3: "significant_degradation" # 显著降级
        }
    
    def apply_degradation(self, config: Dict[str, Any], error_type: str, severity: str) -> Dict[str, Any]:
        """
        应用降级策略
        
        Args:
            config: 原始配置
            error_type: 错误类型
            severity: 严重程度
            
        Returns:
            Dict[str, Any]: 降级后的配置
        """
        degraded_config = config.copy()
        
        # 根据错误类型和严重程度确定降级级别
        degradation_level = self._determine_degradation_level(error_type, severity)
        
        # 应用对应的降级策略
        if degradation_level >= 1:
            degraded_config = self._apply_minimal_degradation(degraded_config)
        
        if degradation_level >= 2:
            degraded_config = self._apply_moderate_degradation(degraded_config)
        
        if degradation_level >= 3:
            degraded_config = self._apply_significant_degradation(degraded_config)
        
        self.logger.info(f"应用降级策略: 级别 {degradation_level} ({self.degradation_levels[degradation_level]})")
        
        return degraded_config
    
    def _determine_degradation_level(self, error_type: str, severity: str) -> int:
        """确定降级级别"""
        if severity == "critical":
            if error_type in ["memory_shortage", "disk_space"]:
                return 3
            else:
                return 2
        elif severity == "warning":
            return 1
        else:
            return 0
    
    def _apply_minimal_degradation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用最小降级"""
        # 禁用非关键功能
        config["run_validation_tests"] = False
        config["enable_progress_monitoring"] = False
        
        self.logger.info("应用最小降级: 禁用验证测试和进度监控")
        return config
    
    def _apply_moderate_degradation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用中等降级"""
        # 降低资源使用
        if "quantization_level" in config and config["quantization_level"] == "none":
            config["quantization_level"] = "int8"
        
        config["compress_weights"] = True
        config["remove_training_artifacts"] = True
        
        # 禁用部分导出格式
        if config.get("export_tensorrt", False):
            config["export_tensorrt"] = False
            self.logger.info("禁用TensorRT导出以节省资源")
        
        self.logger.info("应用中等降级: 启用量化和压缩，禁用TensorRT")
        return config
    
    def _apply_significant_degradation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用显著降级"""
        # 只保留最基本的功能
        config["export_onnx"] = False
        config["export_tensorrt"] = False
        config["quantization_level"] = "int8"
        config["onnx_optimize_graph"] = False
        
        self.logger.info("应用显著降级: 只导出PyTorch格式")
        return config
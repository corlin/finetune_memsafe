#!/usr/bin/env python3
"""
增强错误恢复管理器

为增强训练Pipeline提供错误处理和恢复机制。
"""

import logging
import traceback
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import json


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"          # 轻微错误，可以继续执行
    MEDIUM = "medium"    # 中等错误，需要调整但可以恢复
    HIGH = "high"        # 严重错误，需要用户干预
    CRITICAL = "critical"  # 致命错误，必须停止执行


class ErrorCategory(Enum):
    """错误类别"""
    DATA_SPLIT = "data_split"
    TRAINING = "training"
    EVALUATION = "evaluation"
    MEMORY = "memory"
    CONFIG = "config"
    IO = "io"
    NETWORK = "network"
    SYSTEM = "system"


@dataclass
class ErrorInfo:
    """错误信息"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    context: Optional[Dict[str, Any]] = None
    recovery_suggestions: Optional[List[str]] = None
    timestamp: Optional[str] = None


class ErrorRecoveryManager:
    """
    错误恢复管理器
    
    提供统一的错误处理和恢复机制。
    """
    
    def __init__(self, fallback_mode: bool = True, log_errors: bool = True):
        """
        初始化错误恢复管理器
        
        Args:
            fallback_mode: 是否启用回退模式
            log_errors: 是否记录错误日志
        """
        self.fallback_mode = fallback_mode
        self.log_errors = log_errors
        self.logger = logging.getLogger(__name__)
        
        # 错误历史
        self.error_history: List[ErrorInfo] = []
        
        # 恢复策略映射
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {
            ErrorCategory.DATA_SPLIT: self._handle_data_split_error,
            ErrorCategory.TRAINING: self._handle_training_error,
            ErrorCategory.EVALUATION: self._handle_evaluation_error,
            ErrorCategory.MEMORY: self._handle_memory_error,
            ErrorCategory.CONFIG: self._handle_config_error,
            ErrorCategory.IO: self._handle_io_error,
            ErrorCategory.NETWORK: self._handle_network_error,
            ErrorCategory.SYSTEM: self._handle_system_error
        }
    
    def handle_error(self, 
                    category: ErrorCategory, 
                    exception: Exception,
                    context: Optional[Dict[str, Any]] = None,
                    severity: Optional[ErrorSeverity] = None) -> bool:
        """
        处理错误
        
        Args:
            category: 错误类别
            exception: 异常对象
            context: 错误上下文
            severity: 错误严重程度（如果不指定则自动判断）
            
        Returns:
            bool: 是否可以继续执行
        """
        try:
            # 自动判断严重程度
            if severity is None:
                severity = self._determine_severity(category, exception)
            
            # 创建错误信息
            error_info = ErrorInfo(
                category=category,
                severity=severity,
                message=str(exception),
                exception=exception,
                context=context or {},
                timestamp=self._get_timestamp()
            )
            
            # 记录错误
            self.error_history.append(error_info)
            
            if self.log_errors:
                self._log_error(error_info)
            
            # 尝试恢复
            if category in self.recovery_strategies:
                recovery_result = self.recovery_strategies[category](error_info)
                if recovery_result:
                    self.logger.info(f"错误恢复成功: {category.value}")
                    return True
            
            # 检查是否可以回退
            if self.fallback_mode and severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
                self.logger.warning(f"启用回退模式，跳过 {category.value} 错误")
                return True
            
            # 严重错误，无法继续
            self.logger.error(f"无法恢复的错误: {category.value}, 严重程度: {severity.value}")
            return False
            
        except Exception as e:
            self.logger.error(f"错误处理器本身出错: {e}")
            return False
    
    def _determine_severity(self, category: ErrorCategory, exception: Exception) -> ErrorSeverity:
        """自动判断错误严重程度"""
        error_str = str(exception).lower()
        
        # 内存相关错误通常是严重的
        if category == ErrorCategory.MEMORY or "out of memory" in error_str or "cuda" in error_str:
            return ErrorSeverity.HIGH
        
        # 配置错误通常是中等严重
        if category == ErrorCategory.CONFIG:
            return ErrorSeverity.MEDIUM
        
        # 网络错误通常可以重试
        if category == ErrorCategory.NETWORK:
            return ErrorSeverity.MEDIUM
        
        # 文件IO错误
        if category == ErrorCategory.IO:
            if "permission" in error_str or "access" in error_str:
                return ErrorSeverity.HIGH
            return ErrorSeverity.MEDIUM
        
        # 训练错误
        if category == ErrorCategory.TRAINING:
            if "convergence" in error_str or "nan" in error_str:
                return ErrorSeverity.HIGH
            return ErrorSeverity.MEDIUM
        
        # 默认为中等严重
        return ErrorSeverity.MEDIUM
    
    def _handle_data_split_error(self, error_info: ErrorInfo) -> bool:
        """处理数据拆分错误"""
        try:
            error_str = error_info.message.lower()
            
            # 数据量不足
            if "样本数不足" in error_str or "not enough" in error_str:
                error_info.recovery_suggestions = [
                    "增加数据集大小",
                    "减少最小样本数要求",
                    "调整拆分比例"
                ]
                if self.fallback_mode:
                    self.logger.warning("数据量不足，建议使用更大的数据集")
                    return True
            
            # 分层失败
            if "分层拆分失败" in error_str or "stratify" in error_str:
                error_info.recovery_suggestions = [
                    "检查分层字段是否存在",
                    "使用随机拆分代替分层拆分",
                    "检查标签分布是否均匀"
                ]
                if self.fallback_mode:
                    self.logger.warning("分层拆分失败，将使用随机拆分")
                    return True
            
            # 质量问题
            if "质量" in error_str or "quality" in error_str:
                error_info.recovery_suggestions = [
                    "检查数据格式是否正确",
                    "清理数据中的异常值",
                    "跳过质量分析继续执行"
                ]
                if self.fallback_mode:
                    self.logger.warning("数据质量问题，但继续执行")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"数据拆分错误处理失败: {e}")
            return False
    
    def _handle_training_error(self, error_info: ErrorInfo) -> bool:
        """处理训练错误"""
        try:
            error_str = error_info.message.lower()
            
            # 内存不足
            if "out of memory" in error_str or "cuda" in error_str:
                error_info.recovery_suggestions = [
                    "减少batch_size",
                    "增加gradient_accumulation_steps",
                    "减少max_sequence_length",
                    "启用梯度检查点",
                    "使用更小的模型"
                ]
                # 内存错误通常需要用户调整配置
                return False
            
            # 收敛问题
            if "nan" in error_str or "inf" in error_str:
                error_info.recovery_suggestions = [
                    "降低学习率",
                    "增加梯度裁剪",
                    "检查数据是否包含异常值",
                    "使用更稳定的优化器"
                ]
                return False
            
            # 验证集评估失败
            if "validation" in error_str or "eval" in error_str:
                error_info.recovery_suggestions = [
                    "检查验证集数据格式",
                    "跳过验证集评估",
                    "减少验证集大小"
                ]
                if self.fallback_mode:
                    self.logger.warning("验证集评估失败，但训练可以继续")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"训练错误处理失败: {e}")
            return False
    
    def _handle_evaluation_error(self, error_info: ErrorInfo) -> bool:
        """处理评估错误"""
        try:
            error_str = error_info.message.lower()
            
            # 推理失败
            if "inference" in error_str or "generate" in error_str:
                error_info.recovery_suggestions = [
                    "检查模型是否正确加载",
                    "减少评估批次大小",
                    "跳过部分评估任务",
                    "使用更简单的生成参数"
                ]
                if self.fallback_mode:
                    self.logger.warning("推理失败，跳过部分评估")
                    return True
            
            # 指标计算失败
            if "metric" in error_str or "calculate" in error_str:
                error_info.recovery_suggestions = [
                    "检查预测和参考文本格式",
                    "跳过失败的指标",
                    "使用默认指标值"
                ]
                if self.fallback_mode:
                    self.logger.warning("指标计算失败，使用部分指标")
                    return True
            
            # 效率测量失败
            if "efficiency" in error_str or "latency" in error_str:
                error_info.recovery_suggestions = [
                    "跳过效率指标测量",
                    "减少测试样本数量",
                    "使用简化的效率测试"
                ]
                if self.fallback_mode:
                    self.logger.warning("效率测量失败，跳过效率分析")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"评估错误处理失败: {e}")
            return False
    
    def _handle_memory_error(self, error_info: ErrorInfo) -> bool:
        """处理内存错误"""
        error_info.recovery_suggestions = [
            "减少批次大小",
            "启用梯度检查点",
            "清理GPU内存",
            "使用CPU进行部分计算",
            "减少模型大小或序列长度"
        ]
        # 内存错误通常需要用户调整配置
        return False
    
    def _handle_config_error(self, error_info: ErrorInfo) -> bool:
        """处理配置错误"""
        error_info.recovery_suggestions = [
            "检查配置文件格式",
            "验证所有必需参数",
            "使用默认配置值",
            "检查文件路径是否正确"
        ]
        # 配置错误通常需要修复
        return False
    
    def _handle_io_error(self, error_info: ErrorInfo) -> bool:
        """处理IO错误"""
        error_str = error_info.message.lower()
        
        if "permission" in error_str:
            error_info.recovery_suggestions = [
                "检查文件权限",
                "使用管理员权限运行",
                "更改输出目录"
            ]
            return False
        
        if "not found" in error_str or "no such file" in error_str:
            error_info.recovery_suggestions = [
                "检查文件路径是否正确",
                "创建缺失的目录",
                "使用相对路径"
            ]
            if self.fallback_mode:
                self.logger.warning("文件未找到，尝试创建或跳过")
                return True
        
        return False
    
    def _handle_network_error(self, error_info: ErrorInfo) -> bool:
        """处理网络错误"""
        error_info.recovery_suggestions = [
            "检查网络连接",
            "使用本地模型",
            "配置代理设置",
            "重试下载"
        ]
        if self.fallback_mode:
            self.logger.warning("网络错误，尝试使用本地资源")
            return True
        return False
    
    def _handle_system_error(self, error_info: ErrorInfo) -> bool:
        """处理系统错误"""
        error_info.recovery_suggestions = [
            "检查系统资源",
            "重启相关服务",
            "检查环境变量",
            "更新系统依赖"
        ]
        return False
    
    def _log_error(self, error_info: ErrorInfo):
        """记录错误日志"""
        self.logger.error(f"错误处理 - {error_info.category.value}")
        self.logger.error(f"  严重程度: {error_info.severity.value}")
        self.logger.error(f"  错误信息: {error_info.message}")
        
        if error_info.context:
            self.logger.error(f"  错误上下文: {error_info.context}")
        
        if error_info.exception:
            self.logger.error(f"  异常详情: {traceback.format_exc()}")
        
        if error_info.recovery_suggestions:
            self.logger.error("  恢复建议:")
            for suggestion in error_info.recovery_suggestions:
                self.logger.error(f"    - {suggestion}")
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.error_history:
            return {"total_errors": 0, "categories": {}, "severities": {}}
        
        categories = {}
        severities = {}
        
        for error in self.error_history:
            # 统计类别
            cat = error.category.value
            categories[cat] = categories.get(cat, 0) + 1
            
            # 统计严重程度
            sev = error.severity.value
            severities[sev] = severities.get(sev, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "categories": categories,
            "severities": severities,
            "latest_error": {
                "category": self.error_history[-1].category.value,
                "severity": self.error_history[-1].severity.value,
                "message": self.error_history[-1].message,
                "timestamp": self.error_history[-1].timestamp
            } if self.error_history else None
        }
    
    def save_error_report(self, output_path: str):
        """保存错误报告"""
        try:
            report_data = {
                "summary": self.get_error_summary(),
                "detailed_errors": [
                    {
                        "category": error.category.value,
                        "severity": error.severity.value,
                        "message": error.message,
                        "context": error.context,
                        "recovery_suggestions": error.recovery_suggestions,
                        "timestamp": error.timestamp
                    }
                    for error in self.error_history
                ],
                "generated_at": self._get_timestamp()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"错误报告已保存: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存错误报告失败: {e}")
    
    def clear_error_history(self):
        """清空错误历史"""
        self.error_history.clear()
        self.logger.info("错误历史已清空")
"""
主导出控制器

本模块实现了模型导出的主控制器，负责整合所有组件，编排完整的导出流程，
处理错误和异常，实现断点恢复和重试机制，以及支持并发导出。
"""

import os
import json
import time
import asyncio
import threading
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import traceback
import pickle

from .export_models import ExportConfiguration, ExportResult, QuantizationLevel
from .export_exceptions import (
    ModelExportError, CheckpointValidationError, ModelMergeError,
    OptimizationError, FormatExportError, ValidationError
)
from .checkpoint_detector import CheckpointDetector
from .model_merger import ModelMerger
from .optimization_processor import OptimizationProcessor
from .format_exporter import FormatExporter
from .validation_tester import ValidationTester
from .monitoring_logger import MonitoringLogger, MonitoringLevel
from .export_utils import format_size, ensure_disk_space, create_directory_structure
from .error_recovery import ErrorRecoveryManager, GracefulDegradationManager


@dataclass
class ExportState:
    """导出状态数据模型"""
    
    export_id: str
    current_phase: str = "initialized"
    completed_phases: List[str] = field(default_factory=list)
    failed_phases: List[str] = field(default_factory=list)
    
    # 中间结果
    checkpoint_path: Optional[str] = None
    merged_model_path: Optional[str] = None
    optimized_model_path: Optional[str] = None
    exported_models: Dict[str, str] = field(default_factory=dict)
    
    # 统计信息
    start_time: Optional[datetime] = None
    phase_durations: Dict[str, float] = field(default_factory=dict)
    
    # 错误信息
    last_error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def save_to_file(self, file_path: str):
        """保存状态到文件"""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ExportState':
        """从文件加载状态"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries
    
    def mark_phase_completed(self, phase: str, duration: float):
        """标记阶段完成"""
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)
        self.phase_durations[phase] = duration
        
        # 移除失败记录（如果存在）
        if phase in self.failed_phases:
            self.failed_phases.remove(phase)
    
    def mark_phase_failed(self, phase: str, error_message: str):
        """标记阶段失败"""
        if phase not in self.failed_phases:
            self.failed_phases.append(phase)
        self.last_error = error_message
        self.retry_count += 1


class ModelExportController:
    """模型导出主控制器"""
    
    # 导出阶段定义
    PHASES = [
        "checkpoint_detection",
        "model_merging", 
        "optimization",
        "pytorch_export",
        "onnx_export",
        "tensorrt_export",
        "validation"
    ]
    
    def __init__(self, config: ExportConfiguration):
        """
        初始化导出控制器
        
        Args:
            config: 导出配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.checkpoint_detector = CheckpointDetector()
        self.model_merger = ModelMerger(max_memory_gb=config.max_memory_usage_gb)
        self.optimization_processor = OptimizationProcessor(max_memory_gb=config.max_memory_usage_gb)
        self.format_exporter = FormatExporter(config)
        self.validation_tester = ValidationTester(config)
        
        # 监控系统
        monitoring_level = MonitoringLevel.DETAILED if config.log_level.value == "DEBUG" else MonitoringLevel.STANDARD
        self.monitoring_logger = MonitoringLogger(config, monitoring_level)
        
        # 错误恢复系统
        self.error_recovery_manager = ErrorRecoveryManager(self.logger)
        self.degradation_manager = GracefulDegradationManager(self.logger)
        
        # 导出状态
        self.export_state: Optional[ExportState] = None
        self.state_file_path: Optional[str] = None
        
        # 回调函数
        self.progress_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        
        # 并发控制
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.max_concurrent_exports = 2 if config.enable_parallel_export else 1
        
        self.logger.info("模型导出控制器初始化完成")
    
    def export_model(self, resume_from_checkpoint: bool = False) -> ExportResult:
        """
        执行完整的模型导出流程
        
        Args:
            resume_from_checkpoint: 是否从断点恢复
            
        Returns:
            ExportResult: 导出结果
        """
        export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建导出结果对象
        result = ExportResult(
            export_id=export_id,
            timestamp=datetime.now(),
            success=False,
            configuration=self.config
        )
        
        try:
            with self.monitoring_logger:
                # 初始化或恢复导出状态
                if resume_from_checkpoint:
                    self._resume_from_checkpoint(export_id)
                else:
                    self._initialize_export_state(export_id)
                
                # 开始导出流程
                operation_id = self.monitoring_logger.start_operation(
                    "complete_model_export", 
                    total_steps=len(self.PHASES)
                )
                
                try:
                    # 执行导出阶段
                    self._execute_export_phases(operation_id, result)
                    
                    # 标记成功
                    result.success = True
                    result.total_duration_seconds = (datetime.now() - result.timestamp).total_seconds()
                    
                    self.monitoring_logger.complete_operation(operation_id, success=True)
                    self.logger.info(f"模型导出完成: {export_id}")
                    
                except Exception as e:
                    result.error_message = str(e)
                    result.total_duration_seconds = (datetime.now() - result.timestamp).total_seconds()
                    
                    self.monitoring_logger.complete_operation(operation_id, success=False, error_message=str(e))
                    self.logger.error(f"模型导出失败: {str(e)}")
                    raise
                
                finally:
                    # 清理状态文件（如果成功）
                    if result.success and self.state_file_path and os.path.exists(self.state_file_path):
                        os.remove(self.state_file_path)
        
        except Exception as e:
            if not result.error_message:
                result.error_message = str(e)
            self.logger.error(f"导出过程发生错误: {str(e)}")
            
        finally:
            # 清理资源
            self._cleanup_resources()
        
        return result
    
    def export_multiple_formats_parallel(self, formats: List[str]) -> Dict[str, ExportResult]:
        """
        并发导出多种格式
        
        Args:
            formats: 要导出的格式列表 ['pytorch', 'onnx', 'tensorrt']
            
        Returns:
            Dict[str, ExportResult]: 各格式的导出结果
        """
        if not self.config.enable_parallel_export:
            raise ModelExportError("并发导出未启用，请在配置中设置enable_parallel_export=True")
        
        results = {}
        
        # 首先执行共同的前置步骤（检测、合并、优化）
        with self.monitoring_logger:
            export_id = f"parallel_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._initialize_export_state(export_id)
            
            operation_id = self.monitoring_logger.start_operation(
                "parallel_model_export",
                total_steps=3 + len(formats)  # 前置步骤 + 各格式导出
            )
            
            try:
                # 执行前置步骤
                self._execute_preprocessing_phases(operation_id)
                
                # 并发导出各格式
                with ThreadPoolExecutor(max_workers=self.max_concurrent_exports) as executor:
                    future_to_format = {}
                    
                    for format_name in formats:
                        if self._should_export_format(format_name):
                            future = executor.submit(self._export_single_format, format_name, export_id)
                            future_to_format[future] = format_name
                    
                    # 收集结果
                    for future in as_completed(future_to_format):
                        format_name = future_to_format[future]
                        try:
                            format_result = future.result()
                            results[format_name] = format_result
                            self.logger.info(f"{format_name}格式导出完成")
                        except Exception as e:
                            error_result = ExportResult(
                                export_id=f"{export_id}_{format_name}",
                                timestamp=datetime.now(),
                                success=False,
                                configuration=self.config,
                                error_message=str(e)
                            )
                            results[format_name] = error_result
                            self.logger.error(f"{format_name}格式导出失败: {str(e)}")
                
                self.monitoring_logger.complete_operation(operation_id, success=True)
                
            except Exception as e:
                self.monitoring_logger.complete_operation(operation_id, success=False, error_message=str(e))
                raise
        
        return results
    
    def _initialize_export_state(self, export_id: str):
        """初始化导出状态"""
        self.export_state = ExportState(
            export_id=export_id,
            start_time=datetime.now()
        )
        
        # 创建状态文件路径
        state_dir = Path(self.config.output_directory) / "export_states"
        state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file_path = str(state_dir / f"{export_id}_state.pkl")
        
        # 保存初始状态
        self.export_state.save_to_file(self.state_file_path)
        
        self.logger.info(f"导出状态初始化完成: {export_id}")
    
    def _resume_from_checkpoint(self, export_id: str):
        """从断点恢复导出"""
        state_dir = Path(self.config.output_directory) / "export_states"
        state_file = state_dir / f"{export_id}_state.pkl"
        
        if not state_file.exists():
            raise ModelExportError(f"未找到导出状态文件: {state_file}")
        
        try:
            self.export_state = ExportState.load_from_file(str(state_file))
            self.state_file_path = str(state_file)
            
            self.logger.info(f"从断点恢复导出: {export_id}")
            self.logger.info(f"已完成阶段: {self.export_state.completed_phases}")
            self.logger.info(f"失败阶段: {self.export_state.failed_phases}")
            
        except Exception as e:
            raise ModelExportError(f"加载导出状态失败: {str(e)}")
    
    def _execute_export_phases(self, operation_id: str, result: ExportResult):
        """执行导出阶段"""
        for i, phase in enumerate(self.PHASES):
            # 跳过已完成的阶段
            if phase in self.export_state.completed_phases:
                self.monitoring_logger.update_operation_progress(
                    operation_id, i + 1, f"跳过已完成阶段: {phase}"
                )
                continue
            
            # 检查是否需要执行该阶段
            if not self._should_execute_phase(phase):
                self.export_state.mark_phase_completed(phase, 0.0)
                self.monitoring_logger.update_operation_progress(
                    operation_id, i + 1, f"跳过阶段: {phase}"
                )
                continue
            
            # 执行阶段
            self._execute_single_phase(phase, operation_id, i + 1, result)
    
    def _execute_preprocessing_phases(self, operation_id: str):
        """执行预处理阶段（用于并发导出）"""
        preprocessing_phases = ["checkpoint_detection", "model_merging", "optimization"]
        
        for i, phase in enumerate(preprocessing_phases):
            if phase not in self.export_state.completed_phases:
                self._execute_single_phase(phase, operation_id, i + 1, None)
    
    def _execute_single_phase(self, phase: str, operation_id: str, step_number: int, result: Optional[ExportResult]):
        """执行单个阶段"""
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                self.export_state.current_phase = phase
                self.monitoring_logger.update_operation_progress(
                    operation_id, step_number, f"执行阶段: {phase}"
                )
                
                phase_start_time = time.time()
                
                # 执行具体阶段
                if phase == "checkpoint_detection":
                    self._phase_checkpoint_detection()
                elif phase == "model_merging":
                    self._phase_model_merging()
                elif phase == "optimization":
                    self._phase_optimization()
                elif phase == "pytorch_export":
                    self._phase_pytorch_export(result)
                elif phase == "onnx_export":
                    self._phase_onnx_export(result)
                elif phase == "tensorrt_export":
                    self._phase_tensorrt_export(result)
                elif phase == "validation":
                    self._phase_validation(result)
                else:
                    raise ModelExportError(f"未知阶段: {phase}")
                
                # 标记阶段完成
                phase_duration = time.time() - phase_start_time
                self.export_state.mark_phase_completed(phase, phase_duration)
                
                # 保存状态
                self.export_state.save_to_file(self.state_file_path)
                
                self.logger.info(f"阶段 {phase} 完成，耗时: {phase_duration:.2f}秒")
                break
                
            except Exception as e:
                retry_count += 1
                error_message = f"阶段 {phase} 执行失败 (尝试 {retry_count}/{max_retries + 1}): {str(e)}"
                
                self.export_state.mark_phase_failed(phase, error_message)
                self.monitoring_logger.log_error(operation_id, error_message, e)
                
                if retry_count <= max_retries:
                    self.logger.warning(f"重试阶段 {phase}...")
                    time.sleep(2 ** retry_count)  # 指数退避
                else:
                    self.logger.error(f"阶段 {phase} 最终失败")
                    raise ModelExportError(error_message)
    
    def _should_execute_phase(self, phase: str) -> bool:
        """检查是否应该执行指定阶段"""
        if phase == "pytorch_export":
            return self.config.export_pytorch
        elif phase == "onnx_export":
            return self.config.export_onnx
        elif phase == "tensorrt_export":
            return self.config.export_tensorrt
        elif phase == "validation":
            return self.config.run_validation_tests
        else:
            return True  # 其他阶段总是执行
    
    def _should_export_format(self, format_name: str) -> bool:
        """检查是否应该导出指定格式"""
        if format_name == "pytorch":
            return self.config.export_pytorch
        elif format_name == "onnx":
            return self.config.export_onnx
        elif format_name == "tensorrt":
            return self.config.export_tensorrt
        else:
            return False
    
    def _phase_checkpoint_detection(self):
        """阶段1: Checkpoint检测"""
        self.logger.info("开始Checkpoint检测阶段")
        
        if self.config.auto_detect_latest_checkpoint:
            checkpoint_path = self.checkpoint_detector.detect_latest_checkpoint(self.config.checkpoint_path)
        else:
            checkpoint_path = self.config.checkpoint_path
        
        # 验证checkpoint
        if not self.checkpoint_detector.validate_checkpoint_integrity(checkpoint_path):
            checkpoint_info = self.checkpoint_detector.get_checkpoint_metadata(checkpoint_path)
            raise CheckpointValidationError(
                f"Checkpoint验证失败: {', '.join(checkpoint_info.validation_errors)}",
                checkpoint_path=checkpoint_path
            )
        
        self.export_state.checkpoint_path = checkpoint_path
        self.logger.info(f"Checkpoint检测完成: {checkpoint_path}")
    
    def _phase_model_merging(self):
        """阶段2: 模型合并"""
        self.logger.info("开始模型合并阶段")
        
        # 创建临时目录用于存储合并后的模型
        temp_dir = Path(self.config.output_directory) / "temp" / self.export_state.export_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        merged_model_path = str(temp_dir / "merged_model")
        
        # 执行合并
        merge_result = self.model_merger.merge_and_save(
            base_model_name=self.config.base_model_name,
            adapter_path=self.export_state.checkpoint_path,
            output_path=merged_model_path,
            load_in_4bit=(self.config.quantization_level == QuantizationLevel.INT4),
            load_in_8bit=(self.config.quantization_level == QuantizationLevel.INT8),
            save_tokenizer=self.config.save_tokenizer
        )
        
        if not merge_result['success']:
            raise ModelMergeError(f"模型合并失败: {merge_result.get('error_message', '未知错误')}")
        
        self.export_state.merged_model_path = merged_model_path
        self.logger.info(f"模型合并完成: {merged_model_path}")
    
    def _phase_optimization(self):
        """阶段3: 模型优化"""
        self.logger.info("开始模型优化阶段")
        
        # 加载合并后的模型
        from transformers import AutoModelForCausalLM
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            self.export_state.merged_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # 应用优化
        if self.config.quantization_level != QuantizationLevel.NONE:
            model = self.optimization_processor.apply_quantization(model, self.config.quantization_level)
        
        if self.config.remove_training_artifacts:
            model = self.optimization_processor.remove_training_artifacts(model)
        
        if self.config.compress_weights:
            model = self.optimization_processor.compress_model_weights(model)
        
        # 保存优化后的模型
        temp_dir = Path(self.config.output_directory) / "temp" / self.export_state.export_id
        optimized_model_path = str(temp_dir / "optimized_model")
        
        model.save_pretrained(optimized_model_path, safe_serialization=True)
        
        # 如果有tokenizer，也保存
        if self.config.save_tokenizer:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.export_state.merged_model_path)
                tokenizer.save_pretrained(optimized_model_path)
            except Exception as e:
                self.logger.warning(f"保存tokenizer失败: {str(e)}")
        
        self.export_state.optimized_model_path = optimized_model_path
        self.logger.info(f"模型优化完成: {optimized_model_path}")
        
        # 清理内存
        del model
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _phase_pytorch_export(self, result: Optional[ExportResult]):
        """阶段4: PyTorch格式导出"""
        if not self.config.export_pytorch:
            return
        
        self.logger.info("开始PyTorch格式导出阶段")
        
        # 加载优化后的模型
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            self.export_state.optimized_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        tokenizer = None
        if self.config.save_tokenizer:
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.export_state.optimized_model_path)
            except Exception as e:
                self.logger.warning(f"加载tokenizer失败: {str(e)}")
        
        # 导出PyTorch模型
        pytorch_path = self.format_exporter.export_pytorch_model(model, tokenizer)
        
        self.export_state.exported_models["pytorch"] = pytorch_path
        if result:
            result.pytorch_model_path = pytorch_path
        
        self.logger.info(f"PyTorch格式导出完成: {pytorch_path}")
        
        # 清理内存
        del model
        if tokenizer:
            del tokenizer
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _phase_onnx_export(self, result: Optional[ExportResult]):
        """阶段5: ONNX格式导出"""
        if not self.config.export_onnx:
            return
        
        self.logger.info("开始ONNX格式导出阶段")
        
        # 加载优化后的模型
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            self.export_state.optimized_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.export_state.optimized_model_path)
        
        # 导出ONNX模型
        onnx_path = self.format_exporter.export_onnx_model(model, tokenizer)
        
        self.export_state.exported_models["onnx"] = onnx_path
        if result:
            result.onnx_model_path = onnx_path
        
        self.logger.info(f"ONNX格式导出完成: {onnx_path}")
        
        # 清理内存
        del model, tokenizer
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _phase_tensorrt_export(self, result: Optional[ExportResult]):
        """阶段6: TensorRT格式导出"""
        if not self.config.export_tensorrt:
            return
        
        self.logger.info("开始TensorRT格式导出阶段")
        
        # TensorRT导出需要先有ONNX模型
        if "onnx" not in self.export_state.exported_models:
            raise FormatExportError("TensorRT导出需要先导出ONNX格式")
        
        # 这里可以添加TensorRT导出逻辑
        # 由于TensorRT导出比较复杂且依赖特定环境，这里先记录日志
        self.logger.warning("TensorRT导出功能尚未完全实现")
        
        # 占位符实现
        tensorrt_path = self.export_state.exported_models["onnx"].replace("onnx", "tensorrt")
        self.export_state.exported_models["tensorrt"] = tensorrt_path
        if result:
            result.tensorrt_model_path = tensorrt_path
    
    def _phase_validation(self, result: Optional[ExportResult]):
        """阶段7: 验证测试"""
        if not self.config.run_validation_tests:
            return
        
        self.logger.info("开始验证测试阶段")
        
        # 运行综合验证
        validation_report = self.validation_tester.run_comprehensive_validation(
            self.export_state.exported_models
        )
        
        # 生成验证报告
        report_dir = Path(self.config.output_directory) / "validation_reports" / self.export_state.export_id
        self.validation_tester.generate_validation_report(validation_report, str(report_dir))
        
        if result:
            result.validation_passed = validation_report.success_rate > 0.8
            result.validation_report_path = str(report_dir)
            result.output_consistency_score = validation_report.success_rate
        
        self.logger.info(f"验证测试完成，成功率: {validation_report.success_rate:.2%}")
    
    def _export_single_format(self, format_name: str, export_id: str) -> ExportResult:
        """导出单一格式（用于并发导出）"""
        result = ExportResult(
            export_id=f"{export_id}_{format_name}",
            timestamp=datetime.now(),
            success=False,
            configuration=self.config
        )
        
        try:
            if format_name == "pytorch":
                self._phase_pytorch_export(result)
            elif format_name == "onnx":
                self._phase_onnx_export(result)
            elif format_name == "tensorrt":
                self._phase_tensorrt_export(result)
            else:
                raise FormatExportError(f"不支持的导出格式: {format_name}")
            
            result.success = True
            result.total_duration_seconds = (datetime.now() - result.timestamp).total_seconds()
            
        except Exception as e:
            result.error_message = str(e)
            result.total_duration_seconds = (datetime.now() - result.timestamp).total_seconds()
            raise
        
        return result
    
    def _cleanup_resources(self):
        """清理资源"""
        try:
            # 清理模型合并器
            if hasattr(self.model_merger, 'cleanup'):
                self.model_merger.cleanup()
            
            # 清理优化处理器
            if hasattr(self.optimization_processor, 'cleanup'):
                self.optimization_processor.cleanup()
            
            # 清理临时文件
            if self.export_state and self.export_state.export_id:
                temp_dir = Path(self.config.output_directory) / "temp" / self.export_state.export_id
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
            # 关闭线程池
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            self.logger.info("资源清理完成")
            
        except Exception as e:
            self.logger.warning(f"资源清理时发生错误: {str(e)}")
    
    def add_progress_callback(self, callback: Callable[[str, float, str], None]):
        """添加进度回调函数"""
        self.progress_callbacks.append(callback)
        self.monitoring_logger.add_progress_callback(callback)
    
    def add_status_callback(self, callback: Callable[[str, str], None]):
        """添加状态回调函数"""
        self.status_callbacks.append(callback)
        self.monitoring_logger.add_status_callback(callback)
    
    def get_export_status(self) -> Optional[Dict[str, Any]]:
        """获取当前导出状态"""
        if not self.export_state:
            return None
        
        return {
            'export_id': self.export_state.export_id,
            'current_phase': self.export_state.current_phase,
            'completed_phases': self.export_state.completed_phases,
            'failed_phases': self.export_state.failed_phases,
            'progress_percent': len(self.export_state.completed_phases) / len(self.PHASES) * 100,
            'start_time': self.export_state.start_time.isoformat() if self.export_state.start_time else None,
            'last_error': self.export_state.last_error,
            'retry_count': self.export_state.retry_count,
            'exported_models': self.export_state.exported_models
        }
    
    def cancel_export(self):
        """取消当前导出"""
        if self.export_state:
            self.export_state.current_phase = "cancelled"
            self.logger.info(f"导出已取消: {self.export_state.export_id}")
        
        # 停止监控
        self.monitoring_logger.stop_monitoring()
        
        # 清理资源
        self._cleanup_resources()
    
    def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """列出可用的导出状态检查点"""
        state_dir = Path(self.config.output_directory) / "export_states"
        if not state_dir.exists():
            return []
        
        checkpoints = []
        for state_file in state_dir.glob("*_state.pkl"):
            try:
                state = ExportState.load_from_file(str(state_file))
                checkpoints.append({
                    'export_id': state.export_id,
                    'start_time': state.start_time.isoformat() if state.start_time else None,
                    'current_phase': state.current_phase,
                    'completed_phases': state.completed_phases,
                    'failed_phases': state.failed_phases,
                    'can_resume': len(state.failed_phases) == 0 or state.can_retry(),
                    'file_path': str(state_file)
                })
            except Exception as e:
                self.logger.warning(f"无法加载状态文件 {state_file}: {str(e)}")
        
        return sorted(checkpoints, key=lambda x: x['start_time'], reverse=True)
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self._cleanup_resources()
    
    def _handle_export_error(self, exception: Exception, phase: str, context: Dict[str, Any] = None) -> bool:
        """
        处理导出错误并尝试恢复
        
        Args:
            exception: 发生的异常
            phase: 失败的阶段
            context: 错误上下文
            
        Returns:
            bool: 是否成功恢复
        """
        self.logger.error(f"阶段 '{phase}' 发生错误: {str(exception)}")
        
        # 更新导出状态
        if self.export_state:
            self.export_state.mark_phase_failed(phase, str(exception))
            self._save_export_state()
        
        # 进行错误诊断
        diagnostic = self.error_recovery_manager.diagnose_and_recover(
            exception, 
            context or {"phase": phase, "export_id": self.export_state.export_id if self.export_state else None}
        )
        
        # 记录诊断结果
        self.monitoring_logger.log_error_diagnostic(diagnostic)
        
        # 尝试自动恢复
        recovery_results = self.error_recovery_manager.execute_recovery_actions(diagnostic, auto_only=True)
        
        if recovery_results:
            self.logger.info(f"执行了 {len(recovery_results)} 个恢复动作")
            for result in recovery_results:
                self.logger.info(f"  - {result}")
        
        # 检查是否可以重试
        if self.export_state and self.export_state.can_retry():
            self.logger.info(f"准备重试阶段 '{phase}' (第 {self.export_state.retry_count} 次重试)")
            return True
        
        # 尝试降级处理
        if diagnostic.severity == "critical":
            self.logger.info("尝试应用降级策略")
            degraded_config = self.degradation_manager.apply_degradation(
                self.config.__dict__.copy(),
                diagnostic.issue_type,
                diagnostic.severity
            )
            
            # 更新配置
            for key, value in degraded_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            self.logger.info("已应用降级配置，准备重试")
            return True
        
        return False
    
    def _execute_phase_with_recovery(self, phase: str, phase_func: Callable, *args, **kwargs) -> Any:
        """
        执行阶段并处理错误恢复
        
        Args:
            phase: 阶段名称
            phase_func: 阶段执行函数
            *args, **kwargs: 函数参数
            
        Returns:
            Any: 阶段执行结果
        """
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                self.logger.info(f"执行阶段 '{phase}' (尝试 {attempt + 1}/{max_attempts})")
                
                # 更新当前阶段
                if self.export_state:
                    self.export_state.current_phase = phase
                    self._save_export_state()
                
                # 执行阶段
                start_time = time.time()
                result = phase_func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 标记阶段完成
                if self.export_state:
                    self.export_state.mark_phase_completed(phase, duration)
                    self._save_export_state()
                
                self.logger.info(f"阶段 '{phase}' 完成，耗时 {duration:.2f} 秒")
                return result
                
            except Exception as e:
                attempt += 1
                
                # 尝试错误恢复
                if self._handle_export_error(e, phase, {"attempt": attempt}):
                    if attempt < max_attempts:
                        self.logger.info(f"错误恢复成功，准备重试阶段 '{phase}'")
                        time.sleep(2 ** attempt)  # 指数退避
                        continue
                
                # 如果是最后一次尝试或无法恢复，则抛出异常
                if attempt >= max_attempts:
                    self.logger.error(f"阶段 '{phase}' 在 {max_attempts} 次尝试后仍然失败")
                    raise ModelExportError(
                        f"阶段 '{phase}' 执行失败: {str(e)}",
                        phase=phase,
                        export_id=self.export_state.export_id if self.export_state else None
                    )
        
        # 这里不应该到达
        raise ModelExportError(f"阶段 '{phase}' 执行异常结束")
    
    def _save_export_state(self):
        """保存导出状态"""
        if self.export_state and self.state_file_path:
            try:
                self.export_state.save_to_file(self.state_file_path)
            except Exception as e:
                self.logger.warning(f"保存导出状态失败: {str(e)}")
    
    def _initialize_export_state(self, export_id: str):
        """初始化导出状态"""
        self.export_state = ExportState(
            export_id=export_id,
            start_time=datetime.now()
        )
        
        # 创建状态文件路径
        state_dir = Path(self.config.output_directory) / "export_states"
        state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file_path = str(state_dir / f"{export_id}_state.pkl")
        
        # 保存初始状态
        self._save_export_state()
    
    def _resume_from_checkpoint(self, export_id: str):
        """从检查点恢复"""
        state_file = Path(self.config.output_directory) / "export_states" / f"{export_id}_state.pkl"
        
        if state_file.exists():
            try:
                self.export_state = ExportState.load_from_file(str(state_file))
                self.state_file_path = str(state_file)
                self.logger.info(f"从检查点恢复导出状态: {export_id}")
            except Exception as e:
                self.logger.error(f"恢复导出状态失败: {str(e)}")
                self._initialize_export_state(export_id)
        else:
            self.logger.warning(f"检查点文件不存在: {state_file}")
            self._initialize_export_state(export_id)
    
    def get_recovery_suggestions(self, exception: Exception) -> List[str]:
        """
        获取错误恢复建议
        
        Args:
            exception: 发生的异常
            
        Returns:
            List[str]: 恢复建议列表
        """
        diagnostic = self.error_recovery_manager.diagnose_and_recover(exception)
        
        suggestions = []
        for action in diagnostic.recovery_actions:
            suggestions.append(f"{action.name}: {action.description}")
        
        return suggestions
"""
数据预处理器

统一的数据预处理入口，集成字段检测、数据验证和字段映射功能。
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import traceback

from .data_models import ProcessedBatch, EvaluationConfig
from .data_field_detector import DataFieldDetector
from .batch_data_validator import BatchDataValidator
from .field_mapper import FieldMapper
from .error_handling_strategy import ErrorHandlingStrategy
from .diagnostic_logger import DiagnosticLogger

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    数据预处理器
    
    统一的数据处理入口，整合字段检测、数据验证和字段映射功能，
    提供错误处理和降级处理机制。
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        初始化数据预处理器
        
        Args:
            config: 评估配置
        """
        self.config = config
        self.data_processing_config = config.data_processing
        
        # 初始化组件
        self.field_detector = DataFieldDetector()
        
        # 从配置中获取验证参数
        validation_config = self.data_processing_config.get("validation", {})
        min_valid_ratio = validation_config.get("min_valid_samples_ratio", 0.1)
        self.validator = BatchDataValidator(min_valid_ratio=min_valid_ratio)
        
        # 从配置中获取字段映射
        field_mapping_config = self.data_processing_config.get("field_mapping", {})
        self.field_mapper = FieldMapper(mapping_config=field_mapping_config)
        
        # 初始化错误处理策略
        validation_config = self.data_processing_config.get("validation", {})
        enable_fallback = validation_config.get("enable_fallback", True)
        enable_data_cleaning = validation_config.get("enable_data_cleaning", True)
        self.error_handler = ErrorHandlingStrategy(
            enable_fallback=enable_fallback,
            enable_data_cleaning=enable_data_cleaning
        )
        
        # 诊断配置
        self.diagnostics_config = self.data_processing_config.get("diagnostics", {})
        self.enable_detailed_logging = self.diagnostics_config.get("enable_detailed_logging", False)
        self.log_batch_statistics = self.diagnostics_config.get("log_batch_statistics", True)
        
        # 初始化诊断日志记录器
        self.diagnostic_logger = DiagnosticLogger(
            enable_detailed_logging=self.enable_detailed_logging,
            log_batch_statistics=self.log_batch_statistics,
            save_processing_report=self.diagnostics_config.get("save_processing_report", True)
        )
        
        # 处理统计
        self.processing_stats = {
            "total_batches_processed": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_samples_processed": 0,
            "total_valid_samples": 0
        }
    
    def prepare_inputs(self, batch: Dict[str, List], task_name: str) -> List[str]:
        """
        准备输入数据（简化接口，兼容现有代码）
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            输入文本列表
        """
        try:
            processed_batch = self.preprocess_batch(batch, task_name)
            return processed_batch.inputs
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            if self.enable_detailed_logging:
                logger.error(f"错误详情: {traceback.format_exc()}")
            return []
    
    def preprocess_batch(self, batch: Dict[str, List], task_name: str) -> ProcessedBatch:
        """
        预处理批次数据
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            处理后的批次数据
        """
        self.processing_stats["total_batches_processed"] += 1
        start_time = time.time()
        
        if self.enable_detailed_logging:
            logger.debug(f"开始预处理批次，任务: {task_name}")
            logger.debug(f"批次键: {list(batch.keys()) if batch else 'None'}")
        
        # 记录处理开始事件
        self.diagnostic_logger.log_processing_event(
            "batch_processing_start",
            {"task_name": task_name, "batch_keys": list(batch.keys()) if batch else []}
        )
        
        try:
            # 1. 验证批次数据
            validation_result = self.validator.validate_batch(batch)
            
            if self.log_batch_statistics:
                batch_stats = self.validator.get_batch_statistics(batch)
                logger.info(f"批次统计 - 字段数: {batch_stats['total_fields']}, "
                          f"样本数: {batch_stats['total_samples']}, "
                          f"有效样本: {validation_result.valid_samples_count}")
            
            # 2. 如果数据无效且不能降级处理，返回空结果
            if not validation_result.is_valid and validation_result.valid_samples_count == 0:
                logger.warning(f"批次数据无效且无法恢复: {validation_result.issues}")
                return self._create_empty_result(validation_result.issues)
            
            # 3. 应用字段映射
            mapped_batch = self.field_mapper.map_fields(batch, task_name)
            
            # 4. 检测和提取输入字段
            inputs, valid_indices, warnings = self._extract_inputs(mapped_batch, task_name)
            
            # 5. 数据清洗（如果启用）
            if self.data_processing_config.get("validation", {}).get("enable_data_cleaning", True):
                cleaned_inputs = self.error_handler.clean_data(inputs)
                # 重新计算有效索引
                new_valid_indices = []
                for i, (original_input, cleaned_input) in enumerate(zip(inputs, cleaned_inputs)):
                    if cleaned_input and cleaned_input.strip():
                        new_valid_indices.append(valid_indices[i] if i < len(valid_indices) else i)
                
                inputs = [inp for inp in cleaned_inputs if inp and inp.strip()]
                valid_indices = new_valid_indices[:len(inputs)]
            
            # 6. 更新统计信息
            self.processing_stats["total_samples_processed"] += len(batch.get(list(batch.keys())[0], []) if batch else [])
            self.processing_stats["total_valid_samples"] += len(valid_indices)
            
            if inputs:
                self.processing_stats["successful_batches"] += 1
            else:
                self.processing_stats["failed_batches"] += 1
            
            # 7. 创建处理结果
            skipped_indices = self._calculate_skipped_indices(batch, valid_indices)
            
            processing_stats = {
                "task_name": task_name,
                "original_sample_count": len(batch.get(list(batch.keys())[0], []) if batch else []),
                "valid_sample_count": len(valid_indices),
                "skipped_sample_count": len(skipped_indices),
                "validation_issues": validation_result.issues,
                "field_detection_used": True,
                "error_handling_stats": self.error_handler.get_error_statistics()
            }
            
            result = ProcessedBatch(
                inputs=inputs,
                valid_indices=valid_indices,
                skipped_indices=skipped_indices,
                processing_stats=processing_stats,
                warnings=warnings + validation_result.issues
            )
            
            if self.enable_detailed_logging:
                logger.debug(f"预处理完成 - 输入数: {len(inputs)}, 有效索引: {len(valid_indices)}")
            
            # 记录批次处理完成
            processing_time = time.time() - start_time
            batch_info = {
                "total_samples": len(batch.get(list(batch.keys())[0], []) if batch else []),
                "available_fields": list(batch.keys()) if batch else [],
                "field_count": len(batch) if batch else 0
            }
            
            self.diagnostic_logger.log_batch_processing(
                batch_info=batch_info,
                processing_result=processing_stats,
                task_name=task_name,
                processing_time=processing_time
            )
            
            # 记录数据质量指标
            if validation_result.valid_samples_count > 0:
                quality_metrics = {
                    "valid_sample_ratio": validation_result.valid_samples_count / max(validation_result.total_samples_count, 1),
                    "field_detection_success": len(inputs) > 0,
                    "data_cleaning_applied": self.data_processing_config.get("validation", {}).get("enable_data_cleaning", True)
                }
                self.diagnostic_logger.log_data_quality_metrics(task_name, quality_metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"批次预处理失败: {e}")
            if self.enable_detailed_logging:
                logger.error(f"错误详情: {traceback.format_exc()}")
            
            # 记录错误事件
            self.diagnostic_logger.log_processing_event(
                "batch_processing_error",
                {"task_name": task_name, "error": str(e), "traceback": traceback.format_exc()},
                level="error"
            )
            
            self.processing_stats["failed_batches"] += 1
            return self._create_error_result(str(e))
    
    def _extract_inputs(self, batch: Dict[str, List], task_name: str) -> Tuple[List[str], List[int], List[str]]:
        """
        提取输入数据
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            (输入列表, 有效索引列表, 警告列表)
        """
        inputs = []
        valid_indices = []
        warnings = []
        
        if not batch:
            warnings.append("批次数据为空")
            return inputs, valid_indices, warnings
        
        try:
            # 尝试创建任务特定的组合输入
            combined_inputs = self.field_mapper.create_combined_input(batch, task_name)
            if combined_inputs:
                inputs = combined_inputs
                valid_indices = list(range(len(inputs)))
                if self.enable_detailed_logging:
                    logger.debug(f"使用组合输入，任务: {task_name}")
            else:
                # 查找最佳输入字段
                input_field = self.field_mapper.find_best_input_field(batch, task_name)
                
                if input_field:
                    field_data = batch[input_field]
                    inputs, valid_indices = self._process_field_data(field_data)
                    if self.enable_detailed_logging:
                        logger.debug(f"使用输入字段: {input_field}")
                else:
                    # 使用错误处理策略进行降级处理
                    fallback_inputs = self.error_handler.handle_missing_fields(batch, task_name)
                    if fallback_inputs:
                        inputs = fallback_inputs
                        valid_indices = list(range(len(inputs)))
                        warnings.append("使用错误处理策略的降级字段")
                    else:
                        # 最后的降级处理：尝试所有可能的字段
                        inputs, valid_indices, field_used = self._fallback_field_extraction(batch, task_name)
                        if field_used:
                            warnings.append(f"使用最后降级字段: {field_used}")
                        else:
                            warnings.append("未找到有效的输入字段")
            
        except Exception as e:
            logger.error(f"输入提取失败: {e}")
            warnings.append(f"输入提取错误: {str(e)}")
        
        return inputs, valid_indices, warnings
    
    def _process_field_data(self, field_data: List[Any]) -> Tuple[List[str], List[int]]:
        """
        处理字段数据
        
        Args:
            field_data: 字段数据
            
        Returns:
            (处理后的输入列表, 有效索引列表)
        """
        inputs = []
        valid_indices = []
        
        for i, value in enumerate(field_data):
            if self._is_valid_input(value):
                inputs.append(str(value).strip())
                valid_indices.append(i)
        
        return inputs, valid_indices
    
    def _fallback_field_extraction(self, batch: Dict[str, List], task_name: str) -> Tuple[List[str], List[int], Optional[str]]:
        """
        降级字段提取
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            (输入列表, 有效索引列表, 使用的字段名)
        """
        # 通用字段候选列表
        fallback_candidates = [
            "text", "input", "prompt", "content", "source",
            "question", "query", "sentence", "document"
        ]
        
        for candidate in fallback_candidates:
            if candidate in batch:
                field_data = batch[candidate]
                if isinstance(field_data, list) and field_data:
                    inputs, valid_indices = self._process_field_data(field_data)
                    if inputs:  # 如果找到有效输入
                        return inputs, valid_indices, candidate
        
        # 如果还是没有找到，尝试第一个非空列表字段
        for field_name, field_data in batch.items():
            if isinstance(field_data, list) and field_data:
                inputs, valid_indices = self._process_field_data(field_data)
                if inputs:
                    logger.warning(f"使用未知字段作为输入: {field_name}")
                    return inputs, valid_indices, field_name
        
        return [], [], None
    
    def _clean_inputs(self, inputs: List[str], valid_indices: List[int]) -> Tuple[List[str], List[int]]:
        """
        清洗输入数据
        
        Args:
            inputs: 输入列表
            valid_indices: 有效索引列表
            
        Returns:
            (清洗后的输入列表, 清洗后的有效索引列表)
        """
        cleaned_inputs = []
        cleaned_indices = []
        
        for i, (input_text, original_index) in enumerate(zip(inputs, valid_indices)):
            # 基本清洗
            cleaned_text = input_text.strip()
            
            # 过滤过短的文本
            if len(cleaned_text) < 3:
                continue
            
            # 过滤重复的空白字符
            cleaned_text = ' '.join(cleaned_text.split())
            
            cleaned_inputs.append(cleaned_text)
            cleaned_indices.append(original_index)
        
        return cleaned_inputs, cleaned_indices
    
    def _is_valid_input(self, value: Any) -> bool:
        """检查输入值是否有效"""
        if value is None:
            return False
        
        if isinstance(value, str):
            return bool(value.strip())
        
        if isinstance(value, (int, float)):
            return True
        
        if isinstance(value, (list, dict)):
            return len(value) > 0
        
        return True
    
    def _calculate_skipped_indices(self, batch: Dict[str, List], valid_indices: List[int]) -> List[int]:
        """计算跳过的索引"""
        if not batch:
            return []
        
        # 获取总样本数
        total_samples = 0
        for field_data in batch.values():
            if isinstance(field_data, list):
                total_samples = max(total_samples, len(field_data))
        
        # 计算跳过的索引
        all_indices = set(range(total_samples))
        valid_indices_set = set(valid_indices)
        skipped_indices = list(all_indices - valid_indices_set)
        
        return sorted(skipped_indices)
    
    def _create_empty_result(self, issues: List[str]) -> ProcessedBatch:
        """创建空的处理结果"""
        return ProcessedBatch(
            inputs=[],
            valid_indices=[],
            skipped_indices=[],
            processing_stats={
                "original_sample_count": 0,
                "valid_sample_count": 0,
                "skipped_sample_count": 0,
                "validation_issues": issues,
                "processing_status": "empty_batch"
            },
            warnings=issues
        )
    
    def _create_error_result(self, error_message: str) -> ProcessedBatch:
        """创建错误处理结果"""
        return ProcessedBatch(
            inputs=[],
            valid_indices=[],
            skipped_indices=[],
            processing_stats={
                "original_sample_count": 0,
                "valid_sample_count": 0,
                "skipped_sample_count": 0,
                "processing_error": error_message,
                "processing_status": "error"
            },
            warnings=[f"处理错误: {error_message}"]
        )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            处理统计信息
        """
        stats = self.processing_stats.copy()
        
        if stats["total_batches_processed"] > 0:
            stats["success_rate"] = stats["successful_batches"] / stats["total_batches_processed"]
            stats["failure_rate"] = stats["failed_batches"] / stats["total_batches_processed"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        if stats["total_samples_processed"] > 0:
            stats["valid_sample_rate"] = stats["total_valid_samples"] / stats["total_samples_processed"]
        else:
            stats["valid_sample_rate"] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """重置处理统计信息"""
        self.processing_stats = {
            "total_batches_processed": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_samples_processed": 0,
            "total_valid_samples": 0
        }
        logger.info("处理统计信息已重置")
    
    def update_config(self, config: EvaluationConfig):
        """
        更新配置
        
        Args:
            config: 新的评估配置
        """
        self.config = config
        self.data_processing_config = config.data_processing
        
        # 更新组件配置
        validation_config = self.data_processing_config.get("validation", {})
        min_valid_ratio = validation_config.get("min_valid_samples_ratio", 0.1)
        self.validator = BatchDataValidator(min_valid_ratio=min_valid_ratio)
        
        field_mapping_config = self.data_processing_config.get("field_mapping", {})
        self.field_mapper = FieldMapper(mapping_config=field_mapping_config)
        
        # 更新错误处理策略
        validation_config = self.data_processing_config.get("validation", {})
        enable_fallback = validation_config.get("enable_fallback", True)
        enable_data_cleaning = validation_config.get("enable_data_cleaning", True)
        self.error_handler.configure_strategy(
            enable_fallback=enable_fallback,
            enable_data_cleaning=enable_data_cleaning
        )
        
        # 更新诊断配置
        self.diagnostics_config = self.data_processing_config.get("diagnostics", {})
        self.enable_detailed_logging = self.diagnostics_config.get("enable_detailed_logging", False)
        self.log_batch_statistics = self.diagnostics_config.get("log_batch_statistics", True)
        
        # 更新诊断日志记录器配置
        self.diagnostic_logger.enable_detailed_logging = self.enable_detailed_logging
        self.diagnostic_logger.log_batch_statistics = self.log_batch_statistics
        self.diagnostic_logger.save_processing_report = self.diagnostics_config.get("save_processing_report", True)
        
        logger.info("数据预处理器配置已更新")
    
    def diagnose_batch(self, batch: Dict[str, List], task_name: str) -> Dict[str, Any]:
        """
        诊断批次数据
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            诊断信息
        """
        diagnosis = {
            "batch_info": {
                "is_empty": not batch,
                "field_count": len(batch) if batch else 0,
                "available_fields": list(batch.keys()) if batch else []
            },
            "validation_result": None,
            "field_detection_result": None,
            "field_mapping_info": None,
            "recommendations": []
        }
        
        if not batch:
            diagnosis["recommendations"].append("批次数据为空，检查数据加载过程")
            return diagnosis
        
        # 验证结果
        validation_result = self.validator.validate_batch(batch)
        diagnosis["validation_result"] = validation_result.to_dict()
        
        # 字段检测结果
        field_detection_result = self.field_detector.detect_input_fields(batch, task_name)
        diagnosis["field_detection_result"] = field_detection_result.to_dict()
        
        # 字段映射信息
        input_field = self.field_mapper.find_best_input_field(batch, task_name)
        target_field = self.field_mapper.find_best_target_field(batch, task_name)
        diagnosis["field_mapping_info"] = {
            "recommended_input_field": input_field,
            "recommended_target_field": target_field,
            "input_candidates": self.field_mapper.get_input_field_candidates(task_name),
            "target_candidates": self.field_mapper.get_target_field_candidates(task_name)
        }
        
        # 生成建议
        recommendations = []
        if not validation_result.is_valid:
            recommendations.extend(validation_result.suggestions)
        
        if not field_detection_result.detected_fields:
            recommendations.append("未检测到有效字段，考虑使用自定义字段映射")
        
        if not input_field:
            recommendations.append("未找到合适的输入字段，检查字段名称是否正确")
        
        # 添加错误处理策略的建议
        error_suggestions = self.error_handler.suggest_fixes(batch, task_name)
        recommendations.extend(error_suggestions)
        
        diagnosis["recommendations"] = list(dict.fromkeys(recommendations))  # 去重
        diagnosis["error_handling_stats"] = self.error_handler.get_error_statistics()
        diagnosis["diagnostic_stats"] = self.diagnostic_logger.get_current_statistics()
        
        return diagnosis
    
    def generate_processing_report(self) -> Dict[str, Any]:
        """
        生成数据处理报告
        
        Returns:
            处理报告字典
        """
        return self.diagnostic_logger.generate_comprehensive_report()
    
    def save_processing_report(self, filename: Optional[str] = None) -> str:
        """
        保存处理报告到文件
        
        Args:
            filename: 文件名（可选）
            
        Returns:
            保存的文件路径
        """
        report = self.generate_processing_report()
        return self.diagnostic_logger.save_report_to_file(report, filename)
    
    def get_diagnostic_statistics(self) -> Dict[str, Any]:
        """
        获取诊断统计信息
        
        Returns:
            诊断统计信息
        """
        return {
            "processing_stats": self.get_processing_statistics(),
            "error_handling_stats": self.error_handler.get_error_statistics(),
            "diagnostic_logger_stats": self.diagnostic_logger.get_current_statistics()
        }
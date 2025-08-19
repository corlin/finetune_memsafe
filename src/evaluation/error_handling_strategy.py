"""
错误处理策略

提供各种数据处理错误的处理方法和降级处理机制。
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class ErrorHandlingStrategy:
    """
    错误处理策略
    
    提供各种错误情况的处理方法，包括空批次、缺失字段、
    无效数据类型和部分数据等问题的降级处理机制。
    """
    
    def __init__(self, enable_fallback: bool = True, enable_data_cleaning: bool = True):
        """
        初始化错误处理策略
        
        Args:
            enable_fallback: 是否启用降级处理
            enable_data_cleaning: 是否启用数据清洗
        """
        self.enable_fallback = enable_fallback
        self.enable_data_cleaning = enable_data_cleaning
        
        # 错误处理统计
        self.error_stats = {
            "empty_batch_count": 0,
            "missing_fields_count": 0,
            "invalid_data_types_count": 0,
            "partial_data_count": 0,
            "fallback_used_count": 0,
            "data_cleaning_applied_count": 0
        }
    
    def handle_empty_batch(self, batch: Dict[str, List]) -> List[str]:
        """
        处理空批次
        
        Args:
            batch: 批次数据
            
        Returns:
            处理后的输入列表
        """
        self.error_stats["empty_batch_count"] += 1
        
        if not batch:
            logger.warning("批次数据完全为空")
            return []
        
        # 检查是否有任何非空字段
        non_empty_fields = []
        for field_name, field_data in batch.items():
            if isinstance(field_data, list) and field_data:
                non_empty_fields.append(field_name)
        
        if not non_empty_fields:
            logger.warning("批次中所有字段都为空")
            return []
        
        logger.info(f"发现非空字段: {non_empty_fields}")
        
        # 尝试从非空字段中提取数据
        if self.enable_fallback:
            return self._extract_from_any_field(batch, non_empty_fields)
        
        return []
    
    def handle_missing_fields(self, batch: Dict[str, List], task_name: str) -> List[str]:
        """
        处理缺失字段
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            处理后的输入列表
        """
        self.error_stats["missing_fields_count"] += 1
        
        if not batch:
            return []
        
        logger.warning(f"任务 {task_name} 的预期字段缺失")
        
        if not self.enable_fallback:
            return []
        
        self.error_stats["fallback_used_count"] += 1
        
        # 尝试通用字段名称
        generic_field_names = [
            "text", "input", "content", "prompt", "source",
            "question", "query", "sentence", "document"
        ]
        
        for field_name in generic_field_names:
            if field_name in batch:
                field_data = batch[field_name]
                if isinstance(field_data, list) and field_data:
                    logger.info(f"使用降级字段: {field_name}")
                    return self._process_field_data(field_data)
        
        # 如果通用字段也没有，尝试第一个可用字段
        for field_name, field_data in batch.items():
            if isinstance(field_data, list) and field_data:
                logger.warning(f"使用未知字段作为降级选项: {field_name}")
                return self._process_field_data(field_data)
        
        return []
    
    def handle_invalid_data_types(self, batch: Dict[str, List], field_name: str) -> List[str]:
        """
        处理无效数据类型
        
        Args:
            batch: 批次数据
            field_name: 字段名称
            
        Returns:
            处理后的输入列表
        """
        self.error_stats["invalid_data_types_count"] += 1
        
        if field_name not in batch:
            return []
        
        field_data = batch[field_name]
        logger.warning(f"字段 {field_name} 包含无效数据类型")
        
        # 尝试类型转换
        converted_data = []
        for i, value in enumerate(field_data):
            try:
                if value is None:
                    converted_data.append("")
                elif isinstance(value, str):
                    converted_data.append(value)
                elif isinstance(value, (int, float)):
                    converted_data.append(str(value))
                elif isinstance(value, dict):
                    # 尝试提取字典中的文本内容
                    text_content = self._extract_text_from_dict(value)
                    converted_data.append(text_content)
                elif isinstance(value, list):
                    # 将列表转换为字符串
                    converted_data.append(" ".join(str(item) for item in value))
                else:
                    converted_data.append(str(value))
            except Exception as e:
                logger.warning(f"无法转换索引 {i} 的值: {e}")
                converted_data.append("")
        
        if self.enable_data_cleaning:
            self.error_stats["data_cleaning_applied_count"] += 1
            return self._clean_converted_data(converted_data)
        
        return converted_data
    
    def handle_partial_data(self, batch: Dict[str, List], field_name: str) -> List[str]:
        """
        处理部分数据
        
        Args:
            batch: 批次数据
            field_name: 字段名称
            
        Returns:
            处理后的输入列表
        """
        self.error_stats["partial_data_count"] += 1
        
        if field_name not in batch:
            return []
        
        field_data = batch[field_name]
        logger.warning(f"字段 {field_name} 包含部分无效数据")
        
        # 统计数据质量
        total_count = len(field_data)
        valid_count = sum(1 for value in field_data if self._is_valid_value(value))
        
        logger.info(f"字段 {field_name} 数据质量: {valid_count}/{total_count} ({valid_count/total_count:.2%})")
        
        if valid_count == 0:
            logger.warning("字段中没有有效数据")
            return []
        
        # 处理部分数据
        processed_data = []
        for value in field_data:
            if self._is_valid_value(value):
                processed_data.append(str(value).strip())
            else:
                # 使用默认值填充
                if self.enable_fallback:
                    processed_data.append(self._get_default_value())
                else:
                    processed_data.append("")
        
        if self.enable_data_cleaning:
            self.error_stats["data_cleaning_applied_count"] += 1
            return self._clean_processed_data(processed_data)
        
        return processed_data
    
    def apply_field_fallback(self, batch: Dict[str, List], primary_fields: List[str]) -> Optional[str]:
        """
        应用字段回退机制
        
        Args:
            batch: 批次数据
            primary_fields: 主要字段列表
            
        Returns:
            找到的字段名称，如果没有找到则返回None
        """
        if not self.enable_fallback:
            return None
        
        # 首先尝试主要字段
        for field_name in primary_fields:
            if field_name in batch and self._is_valid_field(batch[field_name]):
                return field_name
        
        # 然后尝试通用字段
        fallback_fields = [
            "text", "input", "content", "prompt", "source",
            "question", "query", "sentence", "document"
        ]
        
        for field_name in fallback_fields:
            if field_name in batch and self._is_valid_field(batch[field_name]):
                logger.info(f"使用回退字段: {field_name}")
                self.error_stats["fallback_used_count"] += 1
                return field_name
        
        # 最后尝试任何可用字段
        for field_name, field_data in batch.items():
            if self._is_valid_field(field_data):
                logger.warning(f"使用未知字段作为最后回退: {field_name}")
                self.error_stats["fallback_used_count"] += 1
                return field_name
        
        return None
    
    def clean_data(self, data: List[str]) -> List[str]:
        """
        清洗数据
        
        Args:
            data: 原始数据列表
            
        Returns:
            清洗后的数据列表
        """
        if not self.enable_data_cleaning:
            return data
        
        self.error_stats["data_cleaning_applied_count"] += 1
        
        cleaned_data = []
        for item in data:
            if isinstance(item, str):
                # 清理空白字符
                cleaned_item = item.strip()
                
                # 移除多余的空白字符
                cleaned_item = " ".join(cleaned_item.split())
                
                # 过滤过短的文本
                if len(cleaned_item) >= 3:
                    cleaned_data.append(cleaned_item)
                else:
                    cleaned_data.append("")
            else:
                cleaned_data.append(str(item) if item is not None else "")
        
        return cleaned_data
    
    def fill_missing_values(self, data: List[str], fill_strategy: str = "empty") -> List[str]:
        """
        填充缺失值
        
        Args:
            data: 数据列表
            fill_strategy: 填充策略 ("empty", "default", "interpolate")
            
        Returns:
            填充后的数据列表
        """
        filled_data = []
        
        for item in data:
            if not self._is_valid_value(item):
                if fill_strategy == "empty":
                    filled_data.append("")
                elif fill_strategy == "default":
                    filled_data.append(self._get_default_value())
                elif fill_strategy == "interpolate":
                    # 简单的插值：使用前一个有效值
                    if filled_data:
                        last_valid = None
                        for prev_item in reversed(filled_data):
                            if self._is_valid_value(prev_item):
                                last_valid = prev_item
                                break
                        filled_data.append(last_valid or self._get_default_value())
                    else:
                        filled_data.append(self._get_default_value())
                else:
                    filled_data.append("")
            else:
                filled_data.append(item)
        
        return filled_data
    
    def _extract_from_any_field(self, batch: Dict[str, List], field_names: List[str]) -> List[str]:
        """从任何可用字段中提取数据"""
        for field_name in field_names:
            field_data = batch[field_name]
            if isinstance(field_data, list) and field_data:
                processed_data = self._process_field_data(field_data)
                if processed_data:
                    logger.info(f"从字段 {field_name} 提取了 {len(processed_data)} 个样本")
                    return processed_data
        
        return []
    
    def _process_field_data(self, field_data: List[Any]) -> List[str]:
        """处理字段数据"""
        processed = []
        for value in field_data:
            if self._is_valid_value(value):
                processed.append(str(value).strip())
        
        return processed
    
    def _extract_text_from_dict(self, data: dict) -> str:
        """从字典中提取文本内容"""
        # 常见的文本字段名
        text_fields = ["text", "content", "input", "prompt", "question", "answer"]
        
        for field in text_fields:
            if field in data and data[field]:
                return str(data[field])
        
        # 如果没有找到标准字段，尝试第一个字符串值
        for value in data.values():
            if isinstance(value, str) and value.strip():
                return value
        
        # 最后返回字典的字符串表示
        return str(data)
    
    def _clean_converted_data(self, data: List[str]) -> List[str]:
        """清洗转换后的数据"""
        cleaned = []
        for item in data:
            if isinstance(item, str):
                cleaned_item = item.strip()
                if cleaned_item:
                    cleaned.append(cleaned_item)
                else:
                    cleaned.append("")
            else:
                cleaned.append("")
        
        return cleaned
    
    def _clean_processed_data(self, data: List[str]) -> List[str]:
        """清洗处理后的数据"""
        return self.clean_data(data)
    
    def _is_valid_value(self, value: Any) -> bool:
        """检查值是否有效"""
        if value is None:
            return False
        
        if isinstance(value, str):
            return bool(value.strip())
        
        if isinstance(value, (int, float)):
            return True
        
        if isinstance(value, (list, dict)):
            return len(value) > 0
        
        return True
    
    def _is_valid_field(self, field_data: Any) -> bool:
        """检查字段是否有效"""
        if not isinstance(field_data, list):
            return False
        
        if len(field_data) == 0:
            return False
        
        # 检查是否有有效值
        valid_count = sum(1 for value in field_data if self._is_valid_value(value))
        return valid_count > 0
    
    def _get_default_value(self) -> str:
        """获取默认值"""
        return "[缺失数据]"
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        获取错误处理统计信息
        
        Returns:
            错误统计信息
        """
        total_errors = sum(self.error_stats.values())
        
        stats = self.error_stats.copy()
        stats["total_errors_handled"] = total_errors
        stats["fallback_usage_rate"] = (
            self.error_stats["fallback_used_count"] / max(total_errors, 1)
        )
        stats["data_cleaning_usage_rate"] = (
            self.error_stats["data_cleaning_applied_count"] / max(total_errors, 1)
        )
        
        return stats
    
    def reset_statistics(self):
        """重置错误统计信息"""
        self.error_stats = {
            "empty_batch_count": 0,
            "missing_fields_count": 0,
            "invalid_data_types_count": 0,
            "partial_data_count": 0,
            "fallback_used_count": 0,
            "data_cleaning_applied_count": 0
        }
        logger.info("错误处理统计信息已重置")
    
    def configure_strategy(self, enable_fallback: bool = None, enable_data_cleaning: bool = None):
        """
        配置错误处理策略
        
        Args:
            enable_fallback: 是否启用降级处理
            enable_data_cleaning: 是否启用数据清洗
        """
        if enable_fallback is not None:
            self.enable_fallback = enable_fallback
            logger.info(f"降级处理已{'启用' if enable_fallback else '禁用'}")
        
        if enable_data_cleaning is not None:
            self.enable_data_cleaning = enable_data_cleaning
            logger.info(f"数据清洗已{'启用' if enable_data_cleaning else '禁用'}")
    
    def suggest_fixes(self, batch: Dict[str, List], task_name: str) -> List[str]:
        """
        建议修复方案
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            修复建议列表
        """
        suggestions = []
        
        if not batch:
            suggestions.append("批次数据为空，检查数据加载过程")
            return suggestions
        
        # 检查字段问题
        available_fields = list(batch.keys())
        
        # 检查是否有常见字段
        common_fields = ["text", "input", "prompt", "question", "content"]
        has_common_field = any(field in available_fields for field in common_fields)
        
        if not has_common_field:
            suggestions.append(f"未找到常见输入字段，可用字段: {available_fields}")
            suggestions.append("考虑使用自定义字段映射配置")
        
        # 检查数据类型问题
        for field_name, field_data in batch.items():
            if not isinstance(field_data, list):
                suggestions.append(f"字段 '{field_name}' 不是列表类型，考虑数据格式转换")
            elif len(field_data) == 0:
                suggestions.append(f"字段 '{field_name}' 为空，检查数据完整性")
            else:
                # 检查数据质量
                valid_count = sum(1 for value in field_data if self._is_valid_value(value))
                if valid_count == 0:
                    suggestions.append(f"字段 '{field_name}' 没有有效数据，考虑数据清洗")
                elif valid_count < len(field_data) * 0.5:
                    suggestions.append(f"字段 '{field_name}' 有效数据比例较低，考虑启用数据清洗")
        
        return suggestions
"""
批次数据验证器

验证批次数据的完整性和有效性，提供数据质量统计和问题检测。
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

from .data_models import ValidationResult

logger = logging.getLogger(__name__)


class BatchDataValidator:
    """
    批次数据验证器
    
    负责验证批次数据的结构完整性、字段一致性和数据质量，
    提供详细的验证报告和改进建议。
    """
    
    def __init__(self, min_valid_ratio: float = 0.1):
        """
        初始化验证器
        
        Args:
            min_valid_ratio: 最小有效样本比例阈值
        """
        self.min_valid_ratio = min_valid_ratio
    
    def validate_batch(self, batch: Dict[str, List]) -> ValidationResult:
        """
        验证批次数据
        
        Args:
            batch: 批次数据
            
        Returns:
            验证结果
        """
        if not batch:
            return ValidationResult(
                is_valid=False,
                valid_samples_count=0,
                total_samples_count=0,
                available_fields=[],
                issues=["批次数据为空"],
                suggestions=["检查数据加载过程，确保数据正确传递"]
            )
        
        available_fields = list(batch.keys())
        issues = []
        suggestions = []
        
        # 检查基本结构
        structure_issues = self._check_basic_structure(batch)
        issues.extend(structure_issues)
        
        # 检查字段一致性
        consistency_issues, total_samples = self._check_field_consistency(batch)
        issues.extend(consistency_issues)
        
        # 统计有效样本数量
        valid_samples_count = self._count_valid_samples(batch)
        
        # 检查数据质量
        quality_issues = self._check_data_quality(batch, valid_samples_count, total_samples)
        issues.extend(quality_issues)
        
        # 生成建议
        suggestions = self._generate_suggestions(batch, issues, valid_samples_count, total_samples)
        
        # 判断整体有效性
        is_valid = (
            len(issues) == 0 or 
            (valid_samples_count > 0 and valid_samples_count / max(total_samples, 1) >= self.min_valid_ratio)
        )
        
        return ValidationResult(
            is_valid=is_valid,
            valid_samples_count=valid_samples_count,
            total_samples_count=total_samples,
            available_fields=available_fields,
            issues=issues,
            suggestions=suggestions
        )
    
    def check_field_consistency(self, batch: Dict[str, List]) -> bool:
        """
        检查字段一致性
        
        Args:
            batch: 批次数据
            
        Returns:
            是否一致
        """
        if not batch:
            return False
        
        field_lengths = [len(field_data) for field_data in batch.values() if isinstance(field_data, list)]
        
        if not field_lengths:
            return False
        
        # 检查所有字段长度是否一致
        return len(set(field_lengths)) <= 1
    
    def get_valid_samples_count(self, batch: Dict[str, List], field_name: str) -> int:
        """
        获取指定字段的有效样本数量
        
        Args:
            batch: 批次数据
            field_name: 字段名称
            
        Returns:
            有效样本数量
        """
        if field_name not in batch:
            return 0
        
        field_data = batch[field_name]
        if not isinstance(field_data, list):
            return 0
        
        valid_count = 0
        for value in field_data:
            if self._is_valid_sample(value):
                valid_count += 1
        
        return valid_count
    
    def _check_basic_structure(self, batch: Dict[str, List]) -> List[str]:
        """检查基本数据结构"""
        issues = []
        
        if not isinstance(batch, dict):
            issues.append(f"批次数据类型错误，期望dict，实际{type(batch).__name__}")
            return issues
        
        if len(batch) == 0:
            issues.append("批次数据字典为空")
            return issues
        
        # 检查每个字段的数据类型
        for field_name, field_data in batch.items():
            if not isinstance(field_data, list):
                issues.append(f"字段 '{field_name}' 数据类型错误，期望list，实际{type(field_data).__name__}")
        
        return issues
    
    def _check_field_consistency(self, batch: Dict[str, List]) -> Tuple[List[str], int]:
        """检查字段一致性"""
        issues = []
        
        # 获取所有列表字段的长度
        field_lengths = {}
        for field_name, field_data in batch.items():
            if isinstance(field_data, list):
                field_lengths[field_name] = len(field_data)
        
        if not field_lengths:
            issues.append("没有找到有效的列表字段")
            return issues, 0
        
        # 检查长度一致性
        lengths = list(field_lengths.values())
        total_samples = max(lengths) if lengths else 0
        
        if len(set(lengths)) > 1:
            issues.append(f"字段长度不一致: {field_lengths}")
        
        return issues, total_samples
    
    def _count_valid_samples(self, batch: Dict[str, List]) -> int:
        """统计有效样本数量"""
        if not batch:
            return 0
        
        # 找到最可能的输入字段
        input_candidates = ["text", "input", "prompt", "question", "content"]
        input_field = None
        
        for candidate in input_candidates:
            if candidate in batch and isinstance(batch[candidate], list):
                input_field = candidate
                break
        
        # 如果没有找到标准输入字段，使用第一个列表字段
        if input_field is None:
            for field_name, field_data in batch.items():
                if isinstance(field_data, list):
                    input_field = field_name
                    break
        
        if input_field is None:
            return 0
        
        return self.get_valid_samples_count(batch, input_field)
    
    def _check_data_quality(self, batch: Dict[str, List], valid_count: int, total_count: int) -> List[str]:
        """检查数据质量"""
        issues = []
        
        if total_count == 0:
            issues.append("批次中没有样本数据")
            return issues
        
        # 检查有效样本比例
        valid_ratio = valid_count / total_count if total_count > 0 else 0
        if valid_ratio < self.min_valid_ratio:
            issues.append(f"有效样本比例过低: {valid_ratio:.2%} (最小要求: {self.min_valid_ratio:.2%})")
        
        # 检查每个字段的数据质量
        for field_name, field_data in batch.items():
            if isinstance(field_data, list):
                field_issues = self._check_field_quality(field_name, field_data)
                issues.extend(field_issues)
        
        return issues
    
    def _check_field_quality(self, field_name: str, field_data: List[Any]) -> List[str]:
        """检查单个字段的数据质量"""
        issues = []
        
        if not field_data:
            issues.append(f"字段 '{field_name}' 为空列表")
            return issues
        
        # 统计数据类型
        type_counter = Counter(type(value).__name__ for value in field_data)
        
        # 检查数据类型一致性
        if len(type_counter) > 2:  # 允许None和一种主要类型
            issues.append(f"字段 '{field_name}' 数据类型不一致: {dict(type_counter)}")
        
        # 统计空值
        none_count = sum(1 for value in field_data if value is None)
        empty_str_count = sum(1 for value in field_data if isinstance(value, str) and not value.strip())
        
        total_empty = none_count + empty_str_count
        empty_ratio = total_empty / len(field_data)
        
        if empty_ratio > 0.5:
            issues.append(f"字段 '{field_name}' 空值比例过高: {empty_ratio:.2%}")
        
        # 检查字符串字段的长度
        if any(isinstance(value, str) for value in field_data):
            str_values = [value for value in field_data if isinstance(value, str) and value.strip()]
            if str_values:
                avg_length = sum(len(value) for value in str_values) / len(str_values)
                if avg_length < 5:
                    issues.append(f"字段 '{field_name}' 平均文本长度过短: {avg_length:.1f}")
        
        return issues
    
    def _generate_suggestions(self, 
                            batch: Dict[str, List], 
                            issues: List[str], 
                            valid_count: int, 
                            total_count: int) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if not batch:
            suggestions.append("检查数据加载流程，确保批次数据正确传递")
            return suggestions
        
        # 基于问题类型生成建议
        for issue in issues:
            if "批次数据为空" in issue:
                suggestions.append("检查数据集是否正确加载，验证数据文件路径和格式")
            elif "字段长度不一致" in issue:
                suggestions.append("检查数据预处理步骤，确保所有字段具有相同的样本数量")
            elif "数据类型错误" in issue:
                suggestions.append("检查数据格式，确保字段数据为列表类型")
            elif "有效样本比例过低" in issue:
                suggestions.append("检查数据质量，清理空值和无效数据")
            elif "空值比例过高" in issue:
                suggestions.append("考虑数据清洗或使用默认值填充空字段")
            elif "平均文本长度过短" in issue:
                suggestions.append("检查文本数据是否完整，考虑过滤过短的样本")
        
        # 通用建议
        if valid_count == 0:
            suggestions.extend([
                "检查字段名称是否正确，常见的输入字段名: text, input, prompt, question",
                "验证数据格式是否符合预期的结构",
                "考虑使用自定义字段映射配置"
            ])
        elif valid_count < total_count * 0.8:
            suggestions.append("考虑启用数据清洗功能以提高数据质量")
        
        # 去重
        suggestions = list(dict.fromkeys(suggestions))
        
        return suggestions
    
    def _is_valid_sample(self, value: Any) -> bool:
        """判断单个样本是否有效"""
        if value is None:
            return False
        
        if isinstance(value, str):
            return bool(value.strip())
        
        if isinstance(value, (int, float)):
            return True
        
        if isinstance(value, (list, dict)):
            return len(value) > 0
        
        return True
    
    def get_batch_statistics(self, batch: Dict[str, List]) -> Dict[str, Any]:
        """
        获取批次数据统计信息
        
        Args:
            batch: 批次数据
            
        Returns:
            统计信息字典
        """
        if not batch:
            return {"total_fields": 0, "total_samples": 0, "field_stats": {}}
        
        stats = {
            "total_fields": len(batch),
            "total_samples": 0,
            "field_stats": {}
        }
        
        for field_name, field_data in batch.items():
            if isinstance(field_data, list):
                field_stats = {
                    "length": len(field_data),
                    "valid_count": self.get_valid_samples_count(batch, field_name),
                    "data_types": dict(Counter(type(value).__name__ for value in field_data)),
                    "empty_count": sum(1 for value in field_data if not self._is_valid_sample(value))
                }
                
                field_stats["valid_ratio"] = field_stats["valid_count"] / max(field_stats["length"], 1)
                stats["field_stats"][field_name] = field_stats
                
                # 更新总样本数（使用最大长度）
                stats["total_samples"] = max(stats["total_samples"], field_stats["length"])
        
        return stats
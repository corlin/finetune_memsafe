"""
数据字段检测器

智能检测批次数据中的有效字段，支持多种数据格式和任务类型。
"""

import logging
from typing import Dict, List, Any, Optional, Set
from collections import Counter

from .data_models import FieldDetectionResult

logger = logging.getLogger(__name__)


class DataFieldDetector:
    """
    智能数据字段检测器
    
    负责检测批次数据中的有效字段，根据任务类型确定字段优先级，
    并分析批次数据结构。
    """
    
    def __init__(self):
        """初始化字段检测器"""
        # 预定义的字段优先级映射
        self.field_priorities = {
            "text_generation": {
                "input": ["text", "input", "prompt", "source", "content"],
                "target": ["target", "answer", "output", "response", "label"]
            },
            "question_answering": {
                "input": ["question", "query", "q"],
                "context": ["context", "passage", "document", "text"],
                "target": ["answer", "target", "a", "response"]
            },
            "classification": {
                "input": ["text", "input", "sentence", "content"],
                "target": ["label", "target", "class", "category"]
            },
            "similarity": {
                "input": ["text1", "sentence1", "text_a"],
                "input2": ["text2", "sentence2", "text_b"],
                "target": ["label", "score", "similarity"]
            }
        }
        
        # 通用字段名称（作为回退选项）
        self.generic_fields = {
            "input": ["text", "input", "content", "source", "prompt"],
            "target": ["target", "label", "answer", "output", "response"]
        }
    
    def detect_input_fields(self, batch: Dict[str, List], task_name: str) -> FieldDetectionResult:
        """
        检测批次数据中的输入字段
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            字段检测结果
        """
        if not batch:
            return FieldDetectionResult(
                detected_fields=[],
                recommended_field=None,
                field_analysis={},
                confidence_scores={}
            )
        
        available_fields = list(batch.keys())
        logger.debug(f"可用字段: {available_fields}")
        
        # 分析每个字段
        field_analysis = self.analyze_batch_structure(batch)
        
        # 获取任务特定的字段优先级
        priority_fields = self.get_field_priority(task_name)
        
        # 检测有效的输入字段
        detected_fields = []
        confidence_scores = {}
        
        for field_type, candidates in priority_fields.items():
            for candidate in candidates:
                if candidate in available_fields:
                    # 计算置信度分数
                    confidence = self._calculate_field_confidence(
                        batch, candidate, field_type, field_analysis
                    )
                    
                    if confidence > 0.1:  # 最低置信度阈值
                        detected_fields.append(candidate)
                        confidence_scores[candidate] = confidence
                        logger.debug(f"检测到字段 '{candidate}' (类型: {field_type}, 置信度: {confidence:.2f})")
        
        # 如果没有检测到任何字段，尝试通用字段
        if not detected_fields:
            logger.warning("未检测到任务特定字段，尝试通用字段")
            for field_type, candidates in self.generic_fields.items():
                for candidate in candidates:
                    if candidate in available_fields:
                        confidence = self._calculate_field_confidence(
                            batch, candidate, field_type, field_analysis
                        )
                        if confidence > 0.05:  # 更低的阈值
                            detected_fields.append(candidate)
                            confidence_scores[candidate] = confidence
        
        # 推荐最佳字段
        recommended_field = None
        if confidence_scores:
            recommended_field = max(confidence_scores.keys(), key=lambda x: confidence_scores[x])
        
        return FieldDetectionResult(
            detected_fields=detected_fields,
            recommended_field=recommended_field,
            field_analysis=field_analysis,
            confidence_scores=confidence_scores
        )
    
    def get_field_priority(self, task_name: str) -> Dict[str, List[str]]:
        """
        获取任务特定的字段优先级
        
        Args:
            task_name: 任务名称
            
        Returns:
            字段优先级字典
        """
        # 标准化任务名称
        normalized_task = task_name.lower()
        
        # 匹配任务类型
        for task_type, priorities in self.field_priorities.items():
            if task_type in normalized_task:
                return priorities
        
        # 默认返回文本生成的优先级
        logger.debug(f"未找到任务 '{task_name}' 的特定优先级，使用默认优先级")
        return self.field_priorities["text_generation"]
    
    def analyze_batch_structure(self, batch: Dict[str, List]) -> Dict[str, Dict[str, Any]]:
        """
        分析批次数据结构
        
        Args:
            batch: 批次数据
            
        Returns:
            字段分析结果
        """
        analysis = {}
        
        for field_name, field_data in batch.items():
            field_info = {
                "data_type": type(field_data).__name__,
                "length": len(field_data) if hasattr(field_data, '__len__') else 0,
                "sample_values": [],
                "non_empty_count": 0,
                "data_types": Counter(),
                "avg_length": 0,
                "has_text_content": False
            }
            
            # 分析字段内容
            if isinstance(field_data, list) and field_data:
                # 采样前几个值进行分析
                sample_size = min(5, len(field_data))
                sample_values = field_data[:sample_size]
                field_info["sample_values"] = [str(v)[:100] for v in sample_values]  # 限制长度
                
                # 统计数据类型和非空值
                text_lengths = []
                for value in field_data:
                    field_info["data_types"][type(value).__name__] += 1
                    
                    if value is not None and str(value).strip():
                        field_info["non_empty_count"] += 1
                        
                        # 如果是字符串，计算长度
                        if isinstance(value, str):
                            text_lengths.append(len(value))
                            field_info["has_text_content"] = True
                
                # 计算平均长度
                if text_lengths:
                    field_info["avg_length"] = sum(text_lengths) / len(text_lengths)
                
                # 计算非空比例
                field_info["non_empty_ratio"] = field_info["non_empty_count"] / len(field_data)
            
            analysis[field_name] = field_info
        
        return analysis
    
    def _calculate_field_confidence(self, 
                                  batch: Dict[str, List], 
                                  field_name: str, 
                                  field_type: str,
                                  field_analysis: Dict[str, Dict[str, Any]]) -> float:
        """
        计算字段的置信度分数
        
        Args:
            batch: 批次数据
            field_name: 字段名称
            field_type: 字段类型
            field_analysis: 字段分析结果
            
        Returns:
            置信度分数 (0-1)
        """
        if field_name not in field_analysis:
            return 0.0
        
        info = field_analysis[field_name]
        confidence = 0.0
        
        # 基础分数：字段存在且有数据
        if info["length"] > 0:
            confidence += 0.3
        
        # 非空数据比例
        confidence += info.get("non_empty_ratio", 0) * 0.4
        
        # 数据类型匹配
        if field_type in ["input", "target"] and info.get("has_text_content", False):
            confidence += 0.2
        
        # 字段名称匹配度
        name_score = self._calculate_name_similarity(field_name, field_type)
        confidence += name_score * 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_name_similarity(self, field_name: str, field_type: str) -> float:
        """
        计算字段名称与类型的相似度
        
        Args:
            field_name: 字段名称
            field_type: 字段类型
            
        Returns:
            相似度分数 (0-1)
        """
        field_name_lower = field_name.lower()
        
        # 直接匹配
        if field_type == "input":
            input_keywords = ["text", "input", "prompt", "question", "query", "content"]
            if any(keyword in field_name_lower for keyword in input_keywords):
                return 1.0
        elif field_type == "target":
            target_keywords = ["target", "answer", "label", "output", "response"]
            if any(keyword in field_name_lower for keyword in target_keywords):
                return 1.0
        elif field_type == "context":
            context_keywords = ["context", "passage", "document"]
            if any(keyword in field_name_lower for keyword in context_keywords):
                return 1.0
        
        return 0.0
    
    def get_recommended_fields_for_task(self, 
                                      batch: Dict[str, List], 
                                      task_name: str) -> Dict[str, Optional[str]]:
        """
        为特定任务获取推荐的字段映射
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            字段映射字典
        """
        detection_result = self.detect_input_fields(batch, task_name)
        field_analysis = detection_result.field_analysis
        confidence_scores = detection_result.confidence_scores
        
        recommendations = {}
        priority_fields = self.get_field_priority(task_name)
        
        for field_type, candidates in priority_fields.items():
            best_field = None
            best_score = 0.0
            
            for candidate in candidates:
                if candidate in batch and candidate in confidence_scores:
                    score = confidence_scores[candidate]
                    if score > best_score:
                        best_score = score
                        best_field = candidate
            
            recommendations[field_type] = best_field
        
        return recommendations
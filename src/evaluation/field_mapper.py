"""
字段映射器

提供灵活的字段映射机制，支持自定义映射配置和任务特定的字段处理。
"""

import logging
from typing import Dict, List, Any, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


class FieldMapper:
    """
    可配置字段映射器
    
    支持自定义字段映射、任务特定的映射规则和默认映射回退机制。
    """
    
    def __init__(self, mapping_config: Optional[Dict[str, Any]] = None):
        """
        初始化字段映射器
        
        Args:
            mapping_config: 自定义映射配置
        """
        self.mapping_config = mapping_config or {}
        
        # 默认映射配置
        self.default_mapping = {
            "text_generation": {
                "input_fields": ["text", "input", "prompt", "source", "content"],
                "target_fields": ["target", "answer", "output", "response", "label"]
            },
            "question_answering": {
                "input_fields": ["question", "query", "q"],
                "context_fields": ["context", "passage", "document", "text"],
                "target_fields": ["answer", "target", "a", "response"]
            },
            "classification": {
                "input_fields": ["text", "input", "sentence", "content"],
                "target_fields": ["label", "target", "class", "category"]
            },
            "similarity": {
                "input_fields": ["text1", "sentence1", "text_a"],
                "input2_fields": ["text2", "sentence2", "text_b"],
                "target_fields": ["label", "score", "similarity"]
            }
        }
    
    def map_fields(self, batch: Dict[str, List], task_name: str) -> Dict[str, List]:
        """
        应用字段映射
        
        Args:
            batch: 原始批次数据
            task_name: 任务名称
            
        Returns:
            映射后的批次数据
        """
        if not batch:
            logger.warning("批次数据为空，无法进行字段映射")
            return batch
        
        # 获取映射规则
        mapping_rules = self._get_mapping_rules(task_name)
        if not mapping_rules:
            logger.debug(f"任务 '{task_name}' 没有特定的映射规则，返回原始数据")
            return batch
        
        mapped_batch = deepcopy(batch)
        
        # 应用映射规则
        for target_field_type, source_candidates in mapping_rules.items():
            if not isinstance(source_candidates, list):
                continue
            
            # 查找第一个存在的源字段
            source_field = None
            for candidate in source_candidates:
                if candidate in batch:
                    source_field = candidate
                    break
            
            if source_field:
                # 创建标准化的字段名
                standard_field_name = self._get_standard_field_name(target_field_type)
                if standard_field_name and standard_field_name != source_field:
                    mapped_batch[standard_field_name] = batch[source_field]
                    logger.debug(f"映射字段: {source_field} -> {standard_field_name}")
        
        return mapped_batch
    
    def get_mapped_field_name(self, original_name: str, task_name: str) -> str:
        """
        获取映射后的字段名称
        
        Args:
            original_name: 原始字段名
            task_name: 任务名称
            
        Returns:
            映射后的字段名称
        """
        mapping_rules = self._get_mapping_rules(task_name)
        
        for target_field_type, source_candidates in mapping_rules.items():
            if isinstance(source_candidates, list) and original_name in source_candidates:
                return self._get_standard_field_name(target_field_type)
        
        return original_name
    
    def get_input_field_candidates(self, task_name: str) -> List[str]:
        """
        获取输入字段候选列表
        
        Args:
            task_name: 任务名称
            
        Returns:
            输入字段候选列表
        """
        mapping_rules = self._get_mapping_rules(task_name)
        
        candidates = []
        for field_type, field_list in mapping_rules.items():
            if "input" in field_type.lower() and isinstance(field_list, list):
                candidates.extend(field_list)
        
        # 去重并保持顺序
        return list(dict.fromkeys(candidates))
    
    def get_target_field_candidates(self, task_name: str) -> List[str]:
        """
        获取目标字段候选列表
        
        Args:
            task_name: 任务名称
            
        Returns:
            目标字段候选列表
        """
        mapping_rules = self._get_mapping_rules(task_name)
        
        candidates = []
        for field_type, field_list in mapping_rules.items():
            if "target" in field_type.lower() and isinstance(field_list, list):
                candidates.extend(field_list)
        
        # 去重并保持顺序
        return list(dict.fromkeys(candidates))
    
    def find_best_input_field(self, batch: Dict[str, List], task_name: str) -> Optional[str]:
        """
        找到最佳的输入字段
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            最佳输入字段名称，如果没有找到则返回None
        """
        candidates = self.get_input_field_candidates(task_name)
        
        for candidate in candidates:
            if candidate in batch and self._is_valid_field(batch[candidate]):
                return candidate
        
        # 如果没有找到，尝试通用字段
        generic_candidates = ["text", "input", "content", "prompt"]
        for candidate in generic_candidates:
            if candidate in batch and self._is_valid_field(batch[candidate]):
                return candidate
        
        return None
    
    def find_best_target_field(self, batch: Dict[str, List], task_name: str) -> Optional[str]:
        """
        找到最佳的目标字段
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            最佳目标字段名称，如果没有找到则返回None
        """
        candidates = self.get_target_field_candidates(task_name)
        
        for candidate in candidates:
            if candidate in batch and self._is_valid_field(batch[candidate]):
                return candidate
        
        # 如果没有找到，尝试通用字段
        generic_candidates = ["target", "answer", "label", "output"]
        for candidate in generic_candidates:
            if candidate in batch and self._is_valid_field(batch[candidate]):
                return candidate
        
        return None
    
    def create_combined_input(self, batch: Dict[str, List], task_name: str) -> Optional[List[str]]:
        """
        为特定任务创建组合输入
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            组合后的输入列表
        """
        if task_name.lower() == "question_answering":
            return self._create_qa_input(batch)
        elif task_name.lower() == "similarity":
            return self._create_similarity_input(batch)
        else:
            # 对于其他任务，返回最佳输入字段
            input_field = self.find_best_input_field(batch, task_name)
            if input_field and input_field in batch:
                return batch[input_field]
        
        return None
    
    def _get_mapping_rules(self, task_name: str) -> Dict[str, List[str]]:
        """获取任务的映射规则"""
        # 首先检查自定义配置
        if self.mapping_config and task_name in self.mapping_config:
            return self.mapping_config[task_name]
        
        # 标准化任务名称并匹配默认规则
        normalized_task = task_name.lower()
        
        for task_type, rules in self.default_mapping.items():
            if task_type in normalized_task:
                return rules
        
        # 如果没有匹配，返回文本生成的默认规则
        return self.default_mapping.get("text_generation", {})
    
    def _get_standard_field_name(self, field_type: str) -> str:
        """获取标准化的字段名称"""
        field_type_lower = field_type.lower()
        
        if "input" in field_type_lower:
            return "input"
        elif "target" in field_type_lower:
            return "target"
        elif "context" in field_type_lower:
            return "context"
        elif "question" in field_type_lower:
            return "question"
        elif "answer" in field_type_lower:
            return "answer"
        else:
            return field_type
    
    def _is_valid_field(self, field_data: Any) -> bool:
        """检查字段数据是否有效"""
        if not isinstance(field_data, list):
            return False
        
        if len(field_data) == 0:
            return False
        
        # 检查是否有非空值
        valid_count = 0
        for value in field_data:
            if value is not None and str(value).strip():
                valid_count += 1
        
        # 至少要有10%的有效值
        return valid_count / len(field_data) >= 0.1
    
    def _create_qa_input(self, batch: Dict[str, List]) -> Optional[List[str]]:
        """创建问答任务的组合输入"""
        question_field = None
        context_field = None
        
        # 查找问题字段
        question_candidates = ["question", "query", "q"]
        for candidate in question_candidates:
            if candidate in batch and self._is_valid_field(batch[candidate]):
                question_field = candidate
                break
        
        # 查找上下文字段
        context_candidates = ["context", "passage", "document", "text"]
        for candidate in context_candidates:
            if candidate in batch and self._is_valid_field(batch[candidate]):
                context_field = candidate
                break
        
        if not question_field:
            return None
        
        questions = batch[question_field]
        contexts = batch.get(context_field, [""] * len(questions)) if context_field else [""] * len(questions)
        
        # 确保长度一致
        min_length = min(len(questions), len(contexts))
        questions = questions[:min_length]
        contexts = contexts[:min_length]
        
        # 组合问题和上下文
        combined_inputs = []
        for q, c in zip(questions, contexts):
            if c and str(c).strip():
                combined_input = f"问题: {q}\n上下文: {c}"
            else:
                combined_input = f"问题: {q}"
            combined_inputs.append(combined_input)
        
        return combined_inputs
    
    def _create_similarity_input(self, batch: Dict[str, List]) -> Optional[List[str]]:
        """创建相似度任务的组合输入"""
        text1_field = None
        text2_field = None
        
        # 查找第一个文本字段
        text1_candidates = ["text1", "sentence1", "text_a"]
        for candidate in text1_candidates:
            if candidate in batch and self._is_valid_field(batch[candidate]):
                text1_field = candidate
                break
        
        # 查找第二个文本字段
        text2_candidates = ["text2", "sentence2", "text_b"]
        for candidate in text2_candidates:
            if candidate in batch and self._is_valid_field(batch[candidate]):
                text2_field = candidate
                break
        
        if not text1_field or not text2_field:
            return None
        
        text1_list = batch[text1_field]
        text2_list = batch[text2_field]
        
        # 确保长度一致
        min_length = min(len(text1_list), len(text2_list))
        text1_list = text1_list[:min_length]
        text2_list = text2_list[:min_length]
        
        # 组合两个文本
        combined_inputs = []
        for t1, t2 in zip(text1_list, text2_list):
            combined_input = f"文本1: {t1}\n文本2: {t2}"
            combined_inputs.append(combined_input)
        
        return combined_inputs
    
    def update_mapping_config(self, task_name: str, mapping_rules: Dict[str, List[str]]):
        """
        更新映射配置
        
        Args:
            task_name: 任务名称
            mapping_rules: 新的映射规则
        """
        if not self.mapping_config:
            self.mapping_config = {}
        
        self.mapping_config[task_name] = mapping_rules
        logger.info(f"已更新任务 '{task_name}' 的映射配置")
    
    def get_mapping_summary(self) -> Dict[str, Any]:
        """
        获取映射配置摘要
        
        Returns:
            映射配置摘要
        """
        summary = {
            "custom_mappings": list(self.mapping_config.keys()) if self.mapping_config else [],
            "default_mappings": list(self.default_mapping.keys()),
            "total_tasks": len(set(list(self.mapping_config.keys() if self.mapping_config else []) + 
                                 list(self.default_mapping.keys())))
        }
        
        return summary
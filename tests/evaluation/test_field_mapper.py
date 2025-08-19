"""
FieldMapper单元测试
"""

import unittest
from unittest.mock import patch, MagicMock

from src.evaluation.field_mapper import FieldMapper


class TestFieldMapper(unittest.TestCase):
    """FieldMapper测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.mapper = FieldMapper()
    
    def test_init_with_custom_mapping(self):
        """测试使用自定义映射初始化"""
        custom_mapping = {
            "custom_task": {
                "input_fields": ["custom_input"],
                "target_fields": ["custom_target"]
            }
        }
        
        mapper = FieldMapper(mapping_config=custom_mapping)
        self.assertEqual(mapper.mapping_config, custom_mapping)
    
    def test_map_fields_text_generation(self):
        """测试文本生成任务的字段映射"""
        batch = {
            "prompt": ["Hello", "World"],
            "response": ["Hi", "Earth"]
        }
        
        mapped_batch = self.mapper.map_fields(batch, "text_generation")
        
        # 原始字段应该保留
        self.assertIn("prompt", mapped_batch)
        self.assertIn("response", mapped_batch)
        
        # 可能会添加标准化字段名
        # 具体行为取决于实现细节
    
    def test_map_fields_empty_batch(self):
        """测试空批次的字段映射"""
        mapped_batch = self.mapper.map_fields({}, "text_generation")
        self.assertEqual(mapped_batch, {})
    
    def test_get_mapped_field_name(self):
        """测试获取映射后的字段名称"""
        # 测试输入字段映射
        mapped_name = self.mapper.get_mapped_field_name("text", "text_generation")
        self.assertIsInstance(mapped_name, str)
        
        # 测试未知字段
        unknown_mapped = self.mapper.get_mapped_field_name("unknown_field", "text_generation")
        self.assertEqual(unknown_mapped, "unknown_field")
    
    def test_get_input_field_candidates(self):
        """测试获取输入字段候选列表"""
        candidates = self.mapper.get_input_field_candidates("text_generation")
        
        self.assertIsInstance(candidates, list)
        self.assertIn("text", candidates)
        self.assertIn("input", candidates)
        self.assertIn("prompt", candidates)
    
    def test_get_target_field_candidates(self):
        """测试获取目标字段候选列表"""
        candidates = self.mapper.get_target_field_candidates("text_generation")
        
        self.assertIsInstance(candidates, list)
        self.assertIn("target", candidates)
        self.assertIn("answer", candidates)
        self.assertIn("output", candidates)
    
    def test_find_best_input_field(self):
        """测试查找最佳输入字段"""
        batch = {
            "text": ["Hello", "World"],
            "empty_field": ["", ""],
            "numbers": [1, 2]
        }
        
        best_field = self.mapper.find_best_input_field(batch, "text_generation")
        self.assertEqual(best_field, "text")
    
    def test_find_best_input_field_no_valid_field(self):
        """测试没有有效输入字段的情况"""
        batch = {
            "empty_field": ["", ""],
            "none_field": [None, None]
        }
        
        best_field = self.mapper.find_best_input_field(batch, "text_generation")
        self.assertIsNone(best_field)
    
    def test_find_best_target_field(self):
        """测试查找最佳目标字段"""
        batch = {
            "answer": ["Response1", "Response2"],
            "empty_field": ["", ""],
            "numbers": [1, 2]
        }
        
        best_field = self.mapper.find_best_target_field(batch, "text_generation")
        self.assertEqual(best_field, "answer")
    
    def test_create_combined_input_qa(self):
        """测试创建问答任务的组合输入"""
        batch = {
            "question": ["What is AI?", "How does ML work?"],
            "context": ["AI is artificial intelligence", "ML uses algorithms"]
        }
        
        combined = self.mapper.create_combined_input(batch, "question_answering")
        
        self.assertIsInstance(combined, list)
        self.assertEqual(len(combined), 2)
        self.assertIn("问题:", combined[0])
        self.assertIn("上下文:", combined[0])
        self.assertIn("What is AI?", combined[0])
        self.assertIn("AI is artificial intelligence", combined[0])
    
    def test_create_combined_input_qa_no_context(self):
        """测试没有上下文的问答任务组合输入"""
        batch = {
            "question": ["What is AI?", "How does ML work?"]
        }
        
        combined = self.mapper.create_combined_input(batch, "question_answering")
        
        self.assertIsInstance(combined, list)
        self.assertEqual(len(combined), 2)
        self.assertIn("问题:", combined[0])
        self.assertNotIn("上下文:", combined[0])
    
    def test_create_combined_input_similarity(self):
        """测试创建相似度任务的组合输入"""
        batch = {
            "text1": ["Hello world", "Good morning"],
            "text2": ["Hi earth", "Good day"]
        }
        
        combined = self.mapper.create_combined_input(batch, "similarity")
        
        self.assertIsInstance(combined, list)
        self.assertEqual(len(combined), 2)
        self.assertIn("文本1:", combined[0])
        self.assertIn("文本2:", combined[0])
        self.assertIn("Hello world", combined[0])
        self.assertIn("Hi earth", combined[0])
    
    def test_create_combined_input_other_task(self):
        """测试其他任务的组合输入"""
        batch = {
            "text": ["Hello", "World"]
        }
        
        combined = self.mapper.create_combined_input(batch, "text_generation")
        
        self.assertEqual(combined, ["Hello", "World"])
    
    def test_is_valid_field(self):
        """测试字段有效性检查"""
        # 有效字段
        self.assertTrue(self.mapper._is_valid_field(["Hello", "World"]))
        self.assertTrue(self.mapper._is_valid_field(["Hello", "", "World"]))  # 部分有效
        
        # 无效字段
        self.assertFalse(self.mapper._is_valid_field([]))  # 空列表
        self.assertFalse(self.mapper._is_valid_field(["", "", ""]))  # 全空
        self.assertFalse(self.mapper._is_valid_field([None, None]))  # 全None
        self.assertFalse(self.mapper._is_valid_field("not a list"))  # 非列表
    
    def test_update_mapping_config(self):
        """测试更新映射配置"""
        new_mapping = {
            "input_fields": ["custom_input"],
            "target_fields": ["custom_target"]
        }
        
        self.mapper.update_mapping_config("custom_task", new_mapping)
        
        self.assertIn("custom_task", self.mapper.mapping_config)
        self.assertEqual(self.mapper.mapping_config["custom_task"], new_mapping)
    
    def test_get_mapping_summary(self):
        """测试获取映射配置摘要"""
        # 添加自定义映射
        self.mapper.update_mapping_config("custom_task", {"input_fields": ["custom"]})
        
        summary = self.mapper.get_mapping_summary()
        
        self.assertIn("custom_mappings", summary)
        self.assertIn("default_mappings", summary)
        self.assertIn("total_tasks", summary)
        self.assertIn("custom_task", summary["custom_mappings"])
    
    def test_create_qa_input_with_length_mismatch(self):
        """测试长度不匹配的问答输入创建"""
        batch = {
            "question": ["Q1", "Q2", "Q3"],
            "context": ["C1", "C2"]  # 长度不匹配
        }
        
        combined = self.mapper._create_qa_input(batch)
        
        # 应该处理长度不匹配，使用较短的长度
        self.assertEqual(len(combined), 2)
    
    def test_create_similarity_input_with_length_mismatch(self):
        """测试长度不匹配的相似度输入创建"""
        batch = {
            "text1": ["T1", "T2", "T3"],
            "text2": ["T1", "T2"]  # 长度不匹配
        }
        
        combined = self.mapper._create_similarity_input(batch)
        
        # 应该处理长度不匹配，使用较短的长度
        self.assertEqual(len(combined), 2)
    
    def test_fallback_to_generic_fields(self):
        """测试回退到通用字段"""
        batch = {
            "content": ["Hello", "World"],  # 通用字段
            "unknown_field": ["A", "B"]
        }
        
        best_input = self.mapper.find_best_input_field(batch, "unknown_task")
        self.assertEqual(best_input, "content")


if __name__ == '__main__':
    unittest.main()
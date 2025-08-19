"""
DataFieldDetector单元测试
"""

import unittest
from unittest.mock import patch, MagicMock

from src.evaluation.data_field_detector import DataFieldDetector
from src.evaluation.data_models import FieldDetectionResult


class TestDataFieldDetector(unittest.TestCase):
    """DataFieldDetector测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.detector = DataFieldDetector()
    
    def test_detect_input_fields_empty_batch(self):
        """测试空批次的字段检测"""
        result = self.detector.detect_input_fields({}, "text_generation")
        
        self.assertIsInstance(result, FieldDetectionResult)
        self.assertEqual(result.detected_fields, [])
        self.assertIsNone(result.recommended_field)
        self.assertEqual(result.field_analysis, {})
        self.assertEqual(result.confidence_scores, {})
    
    def test_detect_input_fields_text_generation(self):
        """测试文本生成任务的字段检测"""
        batch = {
            "text": ["Hello world", "How are you?", "Good morning"],
            "target": ["Bonjour monde", "Comment allez-vous?", "Bonjour"],
            "id": [1, 2, 3]
        }
        
        result = self.detector.detect_input_fields(batch, "text_generation")
        
        self.assertIn("text", result.detected_fields)
        self.assertEqual(result.recommended_field, "text")
        self.assertIn("text", result.field_analysis)
        self.assertIn("text", result.confidence_scores)
        self.assertGreater(result.confidence_scores["text"], 0.5)
    
    def test_detect_input_fields_question_answering(self):
        """测试问答任务的字段检测"""
        batch = {
            "question": ["What is AI?", "How does ML work?"],
            "context": ["AI is artificial intelligence", "ML uses algorithms"],
            "answer": ["Artificial Intelligence", "Machine Learning algorithms"]
        }
        
        result = self.detector.detect_input_fields(batch, "question_answering")
        
        self.assertIn("question", result.detected_fields)
        self.assertIn("question", result.confidence_scores)
        self.assertGreater(result.confidence_scores["question"], 0.5)
    
    def test_detect_input_fields_with_empty_values(self):
        """测试包含空值的字段检测"""
        batch = {
            "text": ["Hello", "", None, "World"],
            "empty_field": ["", "", "", ""],
            "mixed_field": ["Good", None, "", "Morning"]
        }
        
        result = self.detector.detect_input_fields(batch, "text_generation")
        
        # text字段应该被检测到，因为有有效值
        self.assertIn("text", result.detected_fields)
        # empty_field不应该被检测到
        self.assertNotIn("empty_field", result.detected_fields)
        # mixed_field可能被检测到，但置信度较低
        if "mixed_field" in result.detected_fields:
            self.assertLess(result.confidence_scores["mixed_field"], 
                           result.confidence_scores["text"])
    
    def test_get_field_priority_different_tasks(self):
        """测试不同任务的字段优先级"""
        # 文本生成任务
        text_gen_priority = self.detector.get_field_priority("text_generation")
        self.assertIn("input", text_gen_priority)
        self.assertIn("text", text_gen_priority["input"])
        
        # 问答任务
        qa_priority = self.detector.get_field_priority("question_answering")
        self.assertIn("input", qa_priority)
        self.assertIn("question", qa_priority["input"])
        
        # 分类任务
        cls_priority = self.detector.get_field_priority("classification")
        self.assertIn("input", cls_priority)
        self.assertIn("text", cls_priority["input"])
    
    def test_analyze_batch_structure(self):
        """测试批次结构分析"""
        batch = {
            "text": ["Hello world", "Good morning", "How are you?"],
            "numbers": [1, 2, 3],
            "mixed": ["text", 123, None],
            "empty": []
        }
        
        analysis = self.detector.analyze_batch_structure(batch)
        
        # 检查分析结果结构
        self.assertIn("text", analysis)
        self.assertIn("numbers", analysis)
        self.assertIn("mixed", analysis)
        self.assertIn("empty", analysis)
        
        # 检查text字段分析
        text_analysis = analysis["text"]
        self.assertEqual(text_analysis["length"], 3)
        self.assertEqual(text_analysis["non_empty_count"], 3)
        self.assertTrue(text_analysis["has_text_content"])
        self.assertGreater(text_analysis["avg_length"], 0)
        
        # 检查empty字段分析
        empty_analysis = analysis["empty"]
        self.assertEqual(empty_analysis["length"], 0)
        self.assertEqual(empty_analysis["non_empty_count"], 0)
    
    def test_calculate_field_confidence(self):
        """测试字段置信度计算"""
        batch = {
            "text": ["Hello", "World", "Test"],
            "empty": ["", "", ""],
            "partial": ["Good", "", "Morning"]
        }
        
        analysis = self.detector.analyze_batch_structure(batch)
        
        # 计算不同字段的置信度
        text_confidence = self.detector._calculate_field_confidence(
            batch, "text", "input", analysis
        )
        empty_confidence = self.detector._calculate_field_confidence(
            batch, "empty", "input", analysis
        )
        partial_confidence = self.detector._calculate_field_confidence(
            batch, "partial", "input", analysis
        )
        
        # text字段应该有最高置信度
        self.assertGreater(text_confidence, empty_confidence)
        self.assertGreater(text_confidence, partial_confidence)
        self.assertGreater(partial_confidence, empty_confidence)
    
    def test_get_recommended_fields_for_task(self):
        """测试任务推荐字段"""
        batch = {
            "question": ["What is AI?"],
            "context": ["AI is artificial intelligence"],
            "answer": ["Artificial Intelligence"]
        }
        
        recommendations = self.detector.get_recommended_fields_for_task(
            batch, "question_answering"
        )
        
        self.assertIn("input", recommendations)
        self.assertIn("target", recommendations)
        self.assertEqual(recommendations["input"], "question")
        self.assertEqual(recommendations["target"], "answer")
    
    def test_fallback_to_generic_fields(self):
        """测试回退到通用字段"""
        batch = {
            "custom_input": ["Hello world", "Good morning"],
            "custom_output": ["Bonjour monde", "Bonjour"]
        }
        
        result = self.detector.detect_input_fields(batch, "unknown_task")
        
        # 应该检测到custom_input字段（因为包含"input"关键词）
        self.assertTrue(len(result.detected_fields) > 0)
    
    def test_name_similarity_calculation(self):
        """测试字段名称相似度计算"""
        # 测试输入字段名称匹配
        input_score = self.detector._calculate_name_similarity("text", "input")
        self.assertEqual(input_score, 1.0)
        
        input_score2 = self.detector._calculate_name_similarity("prompt", "input")
        self.assertEqual(input_score2, 1.0)
        
        # 测试目标字段名称匹配
        target_score = self.detector._calculate_name_similarity("answer", "target")
        self.assertEqual(target_score, 1.0)
        
        # 测试不匹配的情况
        no_match_score = self.detector._calculate_name_similarity("random", "input")
        self.assertEqual(no_match_score, 0.0)


if __name__ == '__main__':
    unittest.main()
"""
DataPreprocessor单元测试
"""

import unittest
from unittest.mock import patch, MagicMock, Mock

from src.evaluation.data_preprocessor import DataPreprocessor
from src.evaluation.data_models import EvaluationConfig, ProcessedBatch


class TestDataPreprocessor(unittest.TestCase):
    """DataPreprocessor测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = EvaluationConfig()
        self.preprocessor = DataPreprocessor(self.config)
    
    def test_init_with_config(self):
        """测试使用配置初始化"""
        custom_config = EvaluationConfig(
            data_processing={
                "validation": {
                    "min_valid_samples_ratio": 0.2,
                    "enable_data_cleaning": False
                }
            }
        )
        
        preprocessor = DataPreprocessor(custom_config)
        self.assertEqual(preprocessor.config, custom_config)
    
    def test_prepare_inputs_valid_batch(self):
        """测试有效批次的输入准备"""
        batch = {
            "text": ["Hello world", "Good morning", "How are you?"]
        }
        
        inputs = self.preprocessor.prepare_inputs(batch, "text_generation")
        
        self.assertIsInstance(inputs, list)
        self.assertEqual(len(inputs), 3)
        self.assertEqual(inputs[0], "Hello world")
        self.assertEqual(inputs[1], "Good morning")
        self.assertEqual(inputs[2], "How are you?")
    
    def test_prepare_inputs_empty_batch(self):
        """测试空批次的输入准备"""
        inputs = self.preprocessor.prepare_inputs({}, "text_generation")
        
        self.assertIsInstance(inputs, list)
        self.assertEqual(len(inputs), 0)
    
    def test_prepare_inputs_with_empty_values(self):
        """测试包含空值的输入准备"""
        batch = {
            "text": ["Hello", "", None, "World", "   "]
        }
        
        inputs = self.preprocessor.prepare_inputs(batch, "text_generation")
        
        # 应该过滤掉空值
        self.assertGreater(len(inputs), 0)
        self.assertLess(len(inputs), 5)  # 少于原始数量
        self.assertIn("Hello", inputs)
        self.assertIn("World", inputs)
    
    def test_preprocess_batch_detailed(self):
        """测试详细的批次预处理"""
        batch = {
            "text": ["Hello world", "Good morning", ""],
            "target": ["Bonjour monde", "Bonjour", "Vide"]
        }
        
        result = self.preprocessor.preprocess_batch(batch, "text_generation")
        
        self.assertIsInstance(result, ProcessedBatch)
        self.assertIsInstance(result.inputs, list)
        self.assertIsInstance(result.valid_indices, list)
        self.assertIsInstance(result.skipped_indices, list)
        self.assertIsInstance(result.processing_stats, dict)
        self.assertIsInstance(result.warnings, list)
        
        # 检查处理统计
        self.assertIn("task_name", result.processing_stats)
        self.assertIn("valid_sample_count", result.processing_stats)
        self.assertEqual(result.processing_stats["task_name"], "text_generation")
    
    def test_preprocess_batch_question_answering(self):
        """测试问答任务的批次预处理"""
        batch = {
            "question": ["What is AI?", "How does ML work?"],
            "context": ["AI is artificial intelligence", "ML uses algorithms"],
            "answer": ["Artificial Intelligence", "Machine Learning"]
        }
        
        result = self.preprocessor.preprocess_batch(batch, "question_answering")
        
        self.assertGreater(len(result.inputs), 0)
        # 问答任务应该组合问题和上下文
        self.assertIn("问题:", result.inputs[0])
        self.assertIn("上下文:", result.inputs[0])
    
    def test_extract_inputs_with_fallback(self):
        """测试带降级处理的输入提取"""
        batch = {
            "custom_field": ["Hello", "World"],  # 非标准字段名
            "empty_field": ["", ""]
        }
        
        inputs, valid_indices, warnings = self.preprocessor._extract_inputs(batch, "text_generation")
        
        # 应该能够提取到输入，即使字段名不标准
        self.assertGreater(len(inputs), 0)
        self.assertGreater(len(valid_indices), 0)
    
    def test_process_field_data(self):
        """测试字段数据处理"""
        field_data = ["Hello", "", None, "World", 123, []]
        
        inputs, valid_indices = self.preprocessor._process_field_data(field_data)
        
        # 应该只保留有效的数据
        self.assertIn("Hello", inputs)
        self.assertIn("World", inputs)
        self.assertIn("123", inputs)  # 数字应该转换为字符串
        self.assertEqual(len(inputs), len(valid_indices))
    
    def test_clean_inputs(self):
        """测试输入清洗"""
        inputs = ["  Hello  ", "World", "Hi", "  ", "Good morning  "]
        valid_indices = [0, 1, 2, 3, 4]
        
        cleaned_inputs, cleaned_indices = self.preprocessor._clean_inputs(inputs, valid_indices)
        
        # 应该清理空白字符和过短文本
        self.assertIn("Hello", cleaned_inputs)
        self.assertIn("World", cleaned_inputs)
        self.assertIn("Good morning", cleaned_inputs)
        self.assertNotIn("Hi", cleaned_inputs)  # 过短
        self.assertEqual(len(cleaned_inputs), len(cleaned_indices))
    
    def test_is_valid_input(self):
        """测试输入有效性判断"""
        # 有效输入
        self.assertTrue(self.preprocessor._is_valid_input("Hello"))
        self.assertTrue(self.preprocessor._is_valid_input(123))
        self.assertTrue(self.preprocessor._is_valid_input([1, 2, 3]))
        
        # 无效输入
        self.assertFalse(self.preprocessor._is_valid_input(None))
        self.assertFalse(self.preprocessor._is_valid_input(""))
        self.assertFalse(self.preprocessor._is_valid_input("   "))
    
    def test_calculate_skipped_indices(self):
        """测试跳过索引计算"""
        batch = {
            "text": ["A", "B", "C", "D", "E"]
        }
        valid_indices = [0, 2, 4]  # 索引1和3被跳过
        
        skipped_indices = self.preprocessor._calculate_skipped_indices(batch, valid_indices)
        
        self.assertEqual(set(skipped_indices), {1, 3})
    
    def test_get_processing_statistics(self):
        """测试获取处理统计信息"""
        # 先处理一些批次
        batch = {"text": ["Hello", "World"]}
        self.preprocessor.preprocess_batch(batch, "text_generation")
        
        stats = self.preprocessor.get_processing_statistics()
        
        self.assertIn("total_batches_processed", stats)
        self.assertIn("successful_batches", stats)
        self.assertIn("failed_batches", stats)
        self.assertIn("success_rate", stats)
        self.assertGreater(stats["total_batches_processed"], 0)
    
    def test_reset_statistics(self):
        """测试重置统计信息"""
        # 先处理一些批次
        batch = {"text": ["Hello", "World"]}
        self.preprocessor.preprocess_batch(batch, "text_generation")
        
        # 重置统计
        self.preprocessor.reset_statistics()
        
        stats = self.preprocessor.get_processing_statistics()
        self.assertEqual(stats["total_batches_processed"], 0)
        self.assertEqual(stats["successful_batches"], 0)
        self.assertEqual(stats["failed_batches"], 0)
    
    def test_update_config(self):
        """测试更新配置"""
        new_config = EvaluationConfig(
            data_processing={
                "validation": {
                    "min_valid_samples_ratio": 0.3,
                    "enable_data_cleaning": False
                }
            }
        )
        
        self.preprocessor.update_config(new_config)
        
        self.assertEqual(self.preprocessor.config, new_config)
    
    def test_diagnose_batch(self):
        """测试批次诊断"""
        batch = {
            "text": ["Hello", "World"],
            "empty_field": ["", ""]
        }
        
        diagnosis = self.preprocessor.diagnose_batch(batch, "text_generation")
        
        self.assertIn("batch_info", diagnosis)
        self.assertIn("validation_result", diagnosis)
        self.assertIn("field_detection_result", diagnosis)
        self.assertIn("field_mapping_info", diagnosis)
        self.assertIn("recommendations", diagnosis)
        
        # 检查批次信息
        batch_info = diagnosis["batch_info"]
        self.assertFalse(batch_info["is_empty"])
        self.assertEqual(batch_info["field_count"], 2)
        self.assertEqual(batch_info["available_fields"], ["text", "empty_field"])
    
    def test_error_handling_in_preprocess_batch(self):
        """测试预处理中的错误处理"""
        # 模拟一个会导致错误的情况
        with patch.object(self.preprocessor.validator, 'validate_batch', side_effect=Exception("Test error")):
            result = self.preprocessor.preprocess_batch({"text": ["Hello"]}, "text_generation")
            
            self.assertIsInstance(result, ProcessedBatch)
            self.assertEqual(len(result.inputs), 0)
            self.assertIn("processing_error", result.processing_stats)
    
    def test_fallback_field_extraction(self):
        """测试降级字段提取"""
        batch = {
            "unknown_field": ["Hello", "World"],
            "content": ["Good", "Morning"]
        }
        
        inputs, valid_indices, field_used = self.preprocessor._fallback_field_extraction(batch, "text_generation")
        
        # 应该找到一个可用的字段
        self.assertGreater(len(inputs), 0)
        self.assertIsNotNone(field_used)
        self.assertIn(field_used, ["content", "unknown_field"])  # content是通用字段，优先级更高
    
    def test_create_empty_result(self):
        """测试创建空结果"""
        issues = ["Test issue 1", "Test issue 2"]
        result = self.preprocessor._create_empty_result(issues)
        
        self.assertIsInstance(result, ProcessedBatch)
        self.assertEqual(len(result.inputs), 0)
        self.assertEqual(len(result.valid_indices), 0)
        self.assertEqual(result.warnings, issues)
        self.assertIn("processing_status", result.processing_stats)
        self.assertEqual(result.processing_stats["processing_status"], "empty_batch")
    
    def test_create_error_result(self):
        """测试创建错误结果"""
        error_message = "Test error message"
        result = self.preprocessor._create_error_result(error_message)
        
        self.assertIsInstance(result, ProcessedBatch)
        self.assertEqual(len(result.inputs), 0)
        self.assertEqual(len(result.valid_indices), 0)
        self.assertIn(error_message, result.warnings[0])
        self.assertIn("processing_error", result.processing_stats)
        self.assertEqual(result.processing_stats["processing_status"], "error")


if __name__ == '__main__':
    unittest.main()
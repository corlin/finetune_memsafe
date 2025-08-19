"""
BatchDataValidator单元测试
"""

import unittest
from unittest.mock import patch, MagicMock

from src.evaluation.batch_data_validator import BatchDataValidator
from src.evaluation.data_models import ValidationResult


class TestBatchDataValidator(unittest.TestCase):
    """BatchDataValidator测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.validator = BatchDataValidator(min_valid_ratio=0.1)
    
    def test_validate_empty_batch(self):
        """测试空批次验证"""
        result = self.validator.validate_batch({})
        
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.valid_samples_count, 0)
        self.assertEqual(result.total_samples_count, 0)
        self.assertEqual(result.available_fields, [])
        self.assertIn("批次数据为空", result.issues)
    
    def test_validate_valid_batch(self):
        """测试有效批次验证"""
        batch = {
            "text": ["Hello world", "Good morning", "How are you?"],
            "target": ["Bonjour monde", "Bonjour", "Comment allez-vous?"]
        }
        
        result = self.validator.validate_batch(batch)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.valid_samples_count, 3)
        self.assertEqual(result.total_samples_count, 3)
        self.assertEqual(set(result.available_fields), {"text", "target"})
        self.assertEqual(len(result.issues), 0)
    
    def test_validate_batch_with_inconsistent_lengths(self):
        """测试长度不一致的批次"""
        batch = {
            "text": ["Hello", "World"],
            "target": ["Bonjour", "Monde", "Extra"]
        }
        
        result = self.validator.validate_batch(batch)
        
        self.assertFalse(result.is_valid)
        self.assertIn("字段长度不一致", str(result.issues))
    
    def test_validate_batch_with_empty_values(self):
        """测试包含空值的批次"""
        batch = {
            "text": ["Hello", "", None, "World", "   "],
            "target": ["Bonjour", "Vide", "Nul", "Monde", "Espace"]
        }
        
        result = self.validator.validate_batch(batch)
        
        # 应该检测到有效样本（Hello, World）
        self.assertGreaterEqual(result.valid_samples_count, 2)
        self.assertEqual(result.total_samples_count, 5)
    
    def test_validate_batch_with_wrong_data_types(self):
        """测试错误数据类型的批次"""
        batch = {
            "text": "not a list",  # 应该是列表
            "numbers": [1, 2, 3]
        }
        
        result = self.validator.validate_batch(batch)
        
        self.assertFalse(result.is_valid)
        self.assertIn("数据类型错误", str(result.issues))
    
    def test_check_field_consistency_valid(self):
        """测试字段一致性检查 - 有效情况"""
        batch = {
            "text": ["A", "B", "C"],
            "target": ["X", "Y", "Z"]
        }
        
        is_consistent = self.validator.check_field_consistency(batch)
        self.assertTrue(is_consistent)
    
    def test_check_field_consistency_invalid(self):
        """测试字段一致性检查 - 无效情况"""
        batch = {
            "text": ["A", "B"],
            "target": ["X", "Y", "Z"]
        }
        
        is_consistent = self.validator.check_field_consistency(batch)
        self.assertFalse(is_consistent)
    
    def test_get_valid_samples_count(self):
        """测试有效样本计数"""
        batch = {
            "text": ["Hello", "", None, "World", "   ", "Good"],
            "numbers": [1, 2, 3, 4, 5, 6]
        }
        
        text_count = self.validator.get_valid_samples_count(batch, "text")
        numbers_count = self.validator.get_valid_samples_count(batch, "numbers")
        missing_count = self.validator.get_valid_samples_count(batch, "missing")
        
        self.assertEqual(text_count, 3)  # "Hello", "World", "Good"
        self.assertEqual(numbers_count, 6)  # 所有数字都有效
        self.assertEqual(missing_count, 0)  # 字段不存在
    
    def test_validate_with_different_min_ratio(self):
        """测试不同最小有效比例的验证"""
        batch = {
            "text": ["Hello", "", "", "World"]  # 50%有效
        }
        
        # 低阈值验证器应该通过
        low_threshold_validator = BatchDataValidator(min_valid_ratio=0.3)
        result_low = low_threshold_validator.validate_batch(batch)
        self.assertTrue(result_low.is_valid)
        
        # 高阈值验证器应该失败
        high_threshold_validator = BatchDataValidator(min_valid_ratio=0.8)
        result_high = high_threshold_validator.validate_batch(batch)
        self.assertFalse(result_high.is_valid)
    
    def test_get_batch_statistics(self):
        """测试批次统计信息获取"""
        batch = {
            "text": ["Hello", "", "World"],
            "numbers": [1, 2, 3],
            "mixed": ["A", None, 3]
        }
        
        stats = self.validator.get_batch_statistics(batch)
        
        self.assertEqual(stats["total_fields"], 3)
        self.assertEqual(stats["total_samples"], 3)
        self.assertIn("field_stats", stats)
        
        # 检查字段统计
        text_stats = stats["field_stats"]["text"]
        self.assertEqual(text_stats["length"], 3)
        self.assertEqual(text_stats["valid_count"], 2)  # "Hello", "World"
        self.assertAlmostEqual(text_stats["valid_ratio"], 2/3, places=2)
    
    def test_is_valid_sample(self):
        """测试单个样本有效性判断"""
        # 有效样本
        self.assertTrue(self.validator._is_valid_sample("Hello"))
        self.assertTrue(self.validator._is_valid_sample(123))
        self.assertTrue(self.validator._is_valid_sample([1, 2, 3]))
        self.assertTrue(self.validator._is_valid_sample({"key": "value"}))
        
        # 无效样本
        self.assertFalse(self.validator._is_valid_sample(None))
        self.assertFalse(self.validator._is_valid_sample(""))
        self.assertFalse(self.validator._is_valid_sample("   "))
        self.assertFalse(self.validator._is_valid_sample([]))
        self.assertFalse(self.validator._is_valid_sample({}))
    
    def test_generate_suggestions_for_empty_batch(self):
        """测试空批次的建议生成"""
        result = self.validator.validate_batch({})
        
        self.assertIn("检查数据加载流程", str(result.suggestions))
    
    def test_generate_suggestions_for_low_quality_data(self):
        """测试低质量数据的建议生成"""
        batch = {
            "text": ["", "", "", "One valid sample"]  # 25%有效
        }
        
        result = self.validator.validate_batch(batch)
        
        # 应该包含数据质量相关的建议
        suggestions_str = str(result.suggestions)
        self.assertTrue(
            "数据质量" in suggestions_str or 
            "清洗" in suggestions_str or 
            "有效样本" in suggestions_str
        )
    
    def test_validate_batch_with_mixed_data_types(self):
        """测试混合数据类型的批次"""
        batch = {
            "mixed_field": ["text", 123, None, True, [1, 2], {"key": "value"}]
        }
        
        result = self.validator.validate_batch(batch)
        
        # 应该检测到数据类型不一致的问题
        issues_str = str(result.issues)
        self.assertIn("数据类型不一致", issues_str)


if __name__ == '__main__':
    unittest.main()
"""
数据管道测试模块
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from src.data_pipeline import DataPipeline, QAData


class TestDataPipeline:
    """数据管道测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "test_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """测试初始化"""
        pipeline = DataPipeline(str(self.data_dir))
        assert pipeline.data_dir == self.data_dir
        assert pipeline.max_sequence_length == 256
        assert pipeline.qa_data == []
    
    def test_load_qa_data_with_format1(self):
        """测试加载格式1的QA数据"""
        # 创建测试文件
        test_content = """Q1: 什么是密码学？
A1: 密码学是研究编制密码和破译密码的技术科学。

Q2: 什么是对称加密？
A2: 对称加密是指加密和解密使用相同密钥的加密方式。
"""
        test_file = self.data_dir / "test1.md"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        pipeline = DataPipeline(str(self.data_dir))
        qa_data = pipeline.load_qa_data_from_files()
        
        assert len(qa_data) == 2
        assert qa_data[0].question == "什么是密码学？"
        assert qa_data[0].answer == "密码学是研究编制密码和破译密码的技术科学。"
        assert qa_data[1].question == "什么是对称加密？"
        assert qa_data[1].answer == "对称加密是指加密和解密使用相同密钥的加密方式。"
    
    def test_load_qa_data_with_format2(self):
        """测试加载格式2的QA数据"""
        test_content = """# 测试QA

### Q1: 什么是非对称加密？

A1: 非对称加密是指加密和解密使用不同密钥的加密方式。

### Q2: 什么是数字签名？

A2: 数字签名是使用私钥对数据进行签名的技术。
"""
        test_file = self.data_dir / "test2.md"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        pipeline = DataPipeline(str(self.data_dir))
        qa_data = pipeline.load_qa_data_from_files()
        
        assert len(qa_data) == 2
        assert qa_data[0].question == "什么是非对称加密？"
        assert qa_data[0].answer == "非对称加密是指加密和解密使用不同密钥的加密方式。"
    
    def test_load_qa_data_fallback_to_example(self):
        """测试回退到示例数据"""
        # 使用不存在的目录
        pipeline = DataPipeline("nonexistent_dir")
        qa_data = pipeline.load_qa_data_from_files()
        
        # 应该返回示例数据
        assert len(qa_data) > 0
        assert all(qa.source == "example_data" for qa in qa_data)
    
    def test_format_for_qwen(self):
        """测试Qwen格式化"""
        qa_data = [
            QAData(
                question="测试问题",
                answer="测试答案",
                source="test"
            )
        ]
        
        pipeline = DataPipeline(str(self.data_dir))
        dataset = pipeline.format_for_qwen(qa_data)
        
        assert len(dataset) == 1
        expected_text = "<|im_start|>user\n测试问题<|im_end|>\n<|im_start|>assistant\n测试答案<|im_end|>"
        assert dataset[0]["text"] == expected_text
        assert dataset[0]["question"] == "测试问题"
        assert dataset[0]["answer"] == "测试答案"
    
    def test_tokenize_dataset(self):
        """测试数据集分词"""
        # 创建模拟的tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "attention_mask": [[1, 1, 1, 1, 1]]
        }
        
        # 创建测试数据集
        from datasets import Dataset
        dataset = Dataset.from_dict({
            "text": ["<|im_start|>user\n测试<|im_end|>\n<|im_start|>assistant\n答案<|im_end|>"]
        })
        
        pipeline = DataPipeline(str(self.data_dir))
        
        with patch.object(dataset, 'map') as mock_map:
            mock_map.return_value.filter.return_value = dataset
            result = pipeline.tokenize_dataset(dataset, mock_tokenizer)
            mock_map.assert_called_once()
    
    def test_create_data_collator(self):
        """测试创建数据整理器"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        
        pipeline = DataPipeline(str(self.data_dir))
        collator = pipeline.create_data_collator(mock_tokenizer)
        
        assert collator is not None
        assert mock_tokenizer.pad_token == "</s>"
    
    def test_get_dataset_stats(self):
        """测试获取数据集统计信息"""
        from datasets import Dataset
        
        dataset = Dataset.from_dict({
            "input_ids": [[1, 2, 3], [4, 5, 6, 7]],
            "source": ["test1", "test2"]
        })
        
        pipeline = DataPipeline(str(self.data_dir))
        stats = pipeline.get_dataset_stats(dataset)
        
        assert stats["total_samples"] == 2
        assert stats["avg_sequence_length"] == 3.5
        assert stats["min_sequence_length"] == 3
        assert stats["max_sequence_length"] == 4
        assert stats["total_tokens"] == 7
        assert stats["source_distribution"] == {"test1": 1, "test2": 1}
    
    def test_validate_and_clean_data(self):
        """测试数据验证和清理"""
        qa_data = [
            QAData(question="", answer="答案"),  # 空问题，应被过滤
            QAData(question="问题", answer=""),  # 空答案，应被过滤
            QAData(question="短", answer="短"),  # 过短，应被过滤
            QAData(question="相同内容", answer="相同内容"),  # 问答相同，应被过滤
            QAData(question="有效问题", answer="有效答案"),  # 有效数据
        ]
        
        pipeline = DataPipeline(str(self.data_dir))
        validated = pipeline._validate_and_clean_data(qa_data)
        
        assert len(validated) == 1
        assert validated[0].question == "有效问题"
        assert validated[0].answer == "有效答案"
    
    def test_clean_text(self):
        """测试文本清理"""
        pipeline = DataPipeline(str(self.data_dir))
        
        # 测试各种清理情况
        assert pipeline._clean_text("  多余空格  ") == "多余空格"
        assert pipeline._clean_text("*粗体*文本") == "粗体文本"
        assert pipeline._clean_text("问题：") == "问题"
        assert pipeline._clean_text("") == ""
        assert pipeline._clean_text(None) == ""


if __name__ == "__main__":
    pytest.main([__file__])
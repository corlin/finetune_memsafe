"""
数据管道测试模块

测试DataPipeline类的数据加载、格式化、分词和数据整理功能。
"""

import pytest
import tempfile
import shutil
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset

from src.data_pipeline import (
    DataPipeline, 
    QAData, 
    EnhancedDataCollatorForLanguageModeling,
    TensorCreationErrorHandler,
    create_safe_data_collator
)


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


class TestEnhancedDataCollatorForLanguageModeling:
    """测试增强的数据整理器"""
    
    def setup_method(self):
        """测试前设置"""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token = "<pad>"
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token = "</s>"
        
        self.collator = EnhancedDataCollatorForLanguageModeling(
            tokenizer=self.mock_tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
    
    def test_initialization(self):
        """测试数据整理器初始化"""
        assert self.collator.tokenizer == self.mock_tokenizer
        assert self.collator.mlm is False
        assert self.collator.pad_to_multiple_of == 8
        assert self.collator.return_tensors == "pt"
    
    def test_setup_pad_token_with_eos(self):
        """测试使用EOS token设置pad token"""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"
        
        collator = EnhancedDataCollatorForLanguageModeling(tokenizer)
        
        assert tokenizer.pad_token == "</s>"
    
    def test_setup_pad_token_without_eos(self):
        """测试在没有EOS token时添加pad token"""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = None
        
        collator = EnhancedDataCollatorForLanguageModeling(tokenizer)
        
        tokenizer.add_special_tokens.assert_called_once_with({'pad_token': '<|pad|>'})
    
    def test_call_success(self):
        """测试成功的数据整理"""
        features = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "labels": [4, 5]}
        ]
        
        batch = self.collator(features)
        
        # 验证返回的批次结构
        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch
        
        # 验证张量类型
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)
        assert isinstance(batch["attention_mask"], torch.Tensor)
        
        # 验证形状一致性
        assert batch["input_ids"].shape == batch["labels"].shape
        assert batch["input_ids"].shape == batch["attention_mask"].shape
    
    def test_call_with_padding(self):
        """测试带填充的数据整理"""
        features = [
            {"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]},
            {"input_ids": [6, 7], "labels": [6, 7]}
        ]
        
        batch = self.collator(features)
        
        # 验证填充后的形状
        assert batch["input_ids"].shape[1] == 8  # 填充到8的倍数
        assert batch["labels"].shape[1] == 8
        
        # 验证填充值
        assert batch["input_ids"][1, 2] == 0  # pad_token_id
        assert batch["labels"][1, 2] == -100  # 忽略索引
    
    def test_validate_features_empty(self):
        """测试空特征验证"""
        with pytest.raises(ValueError, match="特征列表为空"):
            self.collator._validate_features([])
    
    def test_validate_features_missing_keys(self):
        """测试缺少必需键的特征验证"""
        features = [{"input_ids": [1, 2, 3]}]  # 缺少labels
        
        with pytest.raises(ValueError, match="缺少必需的键"):
            self.collator._validate_features(features)
    
    def test_validate_features_wrong_type(self):
        """测试错误类型的特征验证"""
        features = [{"input_ids": "not a list", "labels": [1, 2, 3]}]
        
        with pytest.raises(ValueError, match="必须是列表"):
            self.collator._validate_features(features)
    
    def test_pad_sequences(self):
        """测试序列填充"""
        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        padded = self.collator._pad_sequences(sequences, 0)
        
        # 验证所有序列长度相同（应该是8，因为pad_to_multiple_of=8）
        expected_length = 8  # 最长序列是4，向上取整到8的倍数
        assert all(len(seq) == expected_length for seq in padded)
        
        # 验证填充值
        assert padded[1] == [4, 5, 0, 0, 0, 0, 0, 0]
    
    def test_pad_sequences_with_multiple_of(self):
        """测试填充到指定倍数"""
        sequences = [[1, 2, 3]]
        padded = self.collator._pad_sequences(sequences, 0)
        
        # 应该填充到8的倍数
        assert len(padded[0]) == 8


class TestTensorCreationErrorHandler:
    """测试张量创建错误处理器"""
    
    def setup_method(self):
        """测试前设置"""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.eos_token_id = 2
        
        self.handler = TensorCreationErrorHandler(
            tokenizer=self.mock_tokenizer,
            max_length=256
        )
    
    def test_fix_sequence_nested_list(self):
        """测试修复嵌套列表"""
        nested_sequence = [[1, 2], [3, 4]]
        fixed = self.handler._fix_sequence(nested_sequence)
        
        assert fixed == [1, 2, 3, 4]
    
    def test_fix_sequence_non_integers(self):
        """测试修复非整数元素"""
        sequence = [1, 2.5, 3, "invalid", 4]
        fixed = self.handler._fix_sequence(sequence)
        
        assert fixed == [1, 2, 3, 4]  # 移除了非整数元素
    
    def test_fix_sequence_too_long(self):
        """测试修复过长序列"""
        long_sequence = list(range(300))  # 超过max_length=256
        fixed = self.handler._fix_sequence(long_sequence)
        
        assert len(fixed) == 256
        assert fixed == list(range(256))
    
    def test_fix_sequence_empty(self):
        """测试修复空序列"""
        empty_sequence = []
        fixed = self.handler._fix_sequence(empty_sequence)
        
        assert len(fixed) == 1
        assert fixed[0] == 2  # eos_token_id
    
    def test_fix_features_batch(self):
        """测试修复特征批次"""
        features = [
            {"input_ids": [[1, 2], [3, 4]], "labels": [[1, 2], [3, 4]]},
            {"input_ids": [5, 6], "labels": [5, 6]}
        ]
        
        fixed_features = self.handler.fix_features_batch(features)
        
        assert len(fixed_features) == 2
        assert fixed_features[0]["input_ids"] == [1, 2, 3, 4]
        assert fixed_features[0]["labels"] == [1, 2, 3, 4]
        assert fixed_features[1]["input_ids"] == [5, 6]
        assert fixed_features[1]["labels"] == [5, 6]
    
    def test_fix_features_batch_invalid_feature(self):
        """测试修复包含无效特征的批次"""
        features = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
            {"input_ids": [], "labels": []},  # 无效特征
            {"input_ids": [4, 5], "labels": [4, 5]}
        ]
        
        fixed_features = self.handler.fix_features_batch(features)
        
        # 应该跳过无效特征，但保留有效的
        assert len(fixed_features) >= 2


class TestCreateSafeDataCollator:
    """测试安全数据整理器创建函数"""
    
    def test_create_safe_data_collator(self):
        """测试创建安全数据整理器"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        
        collator = create_safe_data_collator(
            tokenizer=mock_tokenizer,
            max_length=256,
            pad_to_multiple_of=8
        )
        
        # 验证返回的是可调用对象
        assert callable(collator)
    
    def test_safe_collator_success(self):
        """测试安全整理器成功处理"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        
        collator = create_safe_data_collator(mock_tokenizer)
        
        features = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "labels": [4, 5]}
        ]
        
        batch = collator(features)
        
        # 验证成功处理
        assert "input_ids" in batch
        assert "labels" in batch
        assert isinstance(batch["input_ids"], torch.Tensor)
    
    def test_safe_collator_with_recovery(self):
        """测试安全整理器的错误恢复"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        
        collator = create_safe_data_collator(mock_tokenizer)
        
        # 创建有问题的特征（嵌套列表）
        features = [
            {"input_ids": [[1, 2], [3, 4]], "labels": [[1, 2], [3, 4]]},
            {"input_ids": [5, 6], "labels": [5, 6]}
        ]
        
        batch = collator(features)
        
        # 应该通过错误恢复成功处理
        assert "input_ids" in batch
        assert "labels" in batch
        assert isinstance(batch["input_ids"], torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__])
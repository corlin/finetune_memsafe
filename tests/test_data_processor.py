"""
数据预处理模块测试
"""

import pytest
import json
import tempfile
import csv
from pathlib import Path
from industry_evaluation.utils.data_processor import TextDataProcessor, DataValidator
from industry_evaluation.models.data_models import Dataset, DataSample


class TestTextDataProcessor:
    """文本数据处理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.processor = TextDataProcessor()
    
    def test_clean_text(self):
        """测试文本清理"""
        # 测试空白字符处理
        text = "  这是   一个  测试   "
        cleaned = self.processor._clean_text(text)
        assert cleaned == "这是 一个 测试"
        
        # 测试标点符号标准化
        text = "你好，世界！"
        cleaned = self.processor._clean_text(text)
        assert cleaned == "你好,世界!"
        
        # 测试特殊字符移除
        text = "Hello@#$%World123"
        cleaned = self.processor._clean_text(text)
        assert cleaned == "HelloWorld123"
    
    def test_process_dict(self):
        """测试字典数据处理"""
        data = {
            "name": "测试数据集",
            "industry_domain": "金融",
            "samples": [
                {
                    "id": "sample_001",
                    "input": "什么是股票？",
                    "expected": "股票是股份公司发行的所有权凭证。",
                    "context": {"domain": "finance"}
                },
                {
                    "id": "sample_002",
                    "input": "如何投资？",
                    "expected": "投资需要制定合理的投资策略。"
                }
            ]
        }
        
        dataset = self.processor._process_dict(data)
        
        assert dataset.name == "测试数据集"
        assert dataset.industry_domain == "金融"
        assert len(dataset.samples) == 2
        assert dataset.samples[0].sample_id == "sample_001"
        assert dataset.samples[0].input_text == "什么是股票?"
        assert dataset.samples[0].expected_output == "股票是股份公司发行的所有权凭证."
    
    def test_process_list(self):
        """测试列表数据处理"""
        data = [
            {
                "id": "item_001",
                "input": "测试输入1",
                "expected": "测试输出1"
            },
            {
                "id": "item_002",
                "input": "测试输入2",
                "expected": "测试输出2"
            }
        ]
        
        dataset = self.processor._process_list(data)
        
        assert dataset.name == "列表数据集"
        assert len(dataset.samples) == 2
        assert dataset.samples[0].sample_id == "item_001"
        assert dataset.samples[1].sample_id == "item_002"
    
    def test_process_json_file(self):
        """测试JSON文件处理"""
        data = {
            "name": "JSON测试数据集",
            "industry_domain": "医疗",
            "samples": [
                {
                    "id": "json_001",
                    "input": "什么是高血压？",
                    "expected": "高血压是一种常见的心血管疾病。"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            dataset = self.processor._process_json_file(Path(temp_path))
            assert dataset.name == "JSON测试数据集"
            assert dataset.industry_domain == "医疗"
            assert len(dataset.samples) == 1
        finally:
            Path(temp_path).unlink()
    
    def test_process_csv_file(self):
        """测试CSV文件处理"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'input', 'expected', 'context'])
            writer.writerow(['csv_001', '什么是AI？', 'AI是人工智能的缩写。', '{"domain": "tech"}'])
            writer.writerow(['csv_002', '机器学习是什么？', '机器学习是AI的一个分支。', '{}'])
            temp_path = f.name
        
        try:
            dataset = self.processor._process_csv_file(Path(temp_path))
            assert len(dataset.samples) == 2
            assert dataset.samples[0].sample_id == "csv_001"
            assert dataset.samples[0].context == {"domain": "tech"}
        finally:
            Path(temp_path).unlink()
    
    def test_process_txt_file(self):
        """测试文本文件处理"""
        content = "第一行测试文本\n第二行测试文本\n第三行测试文本"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            dataset = self.processor._process_txt_file(Path(temp_path))
            assert len(dataset.samples) == 3
            assert dataset.samples[0].input_text == "第一行测试文本"
            assert dataset.samples[0].metadata["line_number"] == 1
        finally:
            Path(temp_path).unlink()
    
    def test_validate_dict(self):
        """测试字典验证"""
        # 有效数据
        valid_data = {
            "samples": [
                {"input": "测试输入", "expected": "测试输出"}
            ]
        }
        assert self.processor._validate_dict(valid_data) == True
        
        # 无效数据 - 缺少samples
        invalid_data1 = {"name": "测试"}
        assert self.processor._validate_dict(invalid_data1) == False
        
        # 无效数据 - samples不是列表
        invalid_data2 = {"samples": "not a list"}
        assert self.processor._validate_dict(invalid_data2) == False
        
        # 无效数据 - 样本缺少input
        invalid_data3 = {
            "samples": [
                {"expected": "测试输出"}
            ]
        }
        assert self.processor._validate_dict(invalid_data3) == False
    
    def test_validate_list(self):
        """测试列表验证"""
        # 有效数据
        valid_data = [{"input": "测试"}]
        assert self.processor._validate_list(valid_data) == True
        
        # 无效数据 - 空列表
        invalid_data1 = []
        assert self.processor._validate_list(invalid_data1) == False
        
        # 无效数据 - 不是列表
        invalid_data2 = "not a list"
        assert self.processor._validate_list(invalid_data2) == False
    
    def test_add_remove_text_cleaner(self):
        """测试添加和移除文本清理器"""
        def custom_cleaner(text):
            return text.replace("测试", "test")
        
        # 添加清理器
        original_count = len(self.processor.text_cleaners)
        self.processor.add_text_cleaner(custom_cleaner)
        assert len(self.processor.text_cleaners) == original_count + 1
        
        # 测试清理效果
        text = "这是测试文本"
        cleaned = self.processor._clean_text(text)
        assert "test" in cleaned
        
        # 移除清理器
        self.processor.remove_text_cleaner(custom_cleaner)
        assert len(self.processor.text_cleaners) == original_count


class TestDataValidator:
    """数据验证器测试"""
    
    def test_validate_sample(self):
        """测试样本验证"""
        # 有效样本
        valid_sample = DataSample(
            sample_id="test_001",
            input_text="这是测试输入",
            expected_output="这是测试输出"
        )
        errors = DataValidator.validate_sample(valid_sample)
        assert len(errors) == 0
        
        # 无效样本 - 空ID
        invalid_sample1 = DataSample(
            sample_id="",
            input_text="测试输入",
            expected_output="测试输出"
        )
        errors = DataValidator.validate_sample(invalid_sample1)
        assert len(errors) > 0
        assert "样本ID不能为空" in errors
        
        # 无效样本 - 空输入
        invalid_sample2 = DataSample(
            sample_id="test_002",
            input_text="",
            expected_output="测试输出"
        )
        errors = DataValidator.validate_sample(invalid_sample2)
        assert len(errors) > 0
        assert "输入文本不能为空" in errors
        
        # 无效样本 - 文本过长
        invalid_sample3 = DataSample(
            sample_id="test_003",
            input_text="x" * 10001,
            expected_output="测试输出"
        )
        errors = DataValidator.validate_sample(invalid_sample3)
        assert len(errors) > 0
        assert "输入文本过长" in errors[0]
    
    def test_validate_dataset(self):
        """测试数据集验证"""
        # 有效数据集
        valid_dataset = Dataset(
            name="测试数据集",
            industry_domain="测试领域",
            samples=[
                DataSample("sample_001", "输入1", "输出1"),
                DataSample("sample_002", "输入2", "输出2")
            ]
        )
        errors = DataValidator.validate_dataset(valid_dataset)
        assert len(errors) == 0
        
        # 无效数据集 - 空名称
        invalid_dataset1 = Dataset(
            name="",
            industry_domain="测试领域",
            samples=[DataSample("sample_001", "输入1", "输出1")]
        )
        errors = DataValidator.validate_dataset(invalid_dataset1)
        assert len(errors) > 0
        assert "数据集名称不能为空" in errors
        
        # 无效数据集 - 重复ID
        invalid_dataset2 = Dataset(
            name="测试数据集",
            industry_domain="测试领域",
            samples=[
                DataSample("sample_001", "输入1", "输出1"),
                DataSample("sample_001", "输入2", "输出2")  # 重复ID
            ]
        )
        errors = DataValidator.validate_dataset(invalid_dataset2)
        assert len(errors) > 0
        assert "存在重复的样本ID" in errors
    
    def test_check_data_quality(self):
        """测试数据质量检查"""
        dataset = Dataset(
            name="质量测试数据集",
            industry_domain="测试",
            samples=[
                DataSample("sample_001", "短输入", "短输出"),
                DataSample("sample_002", "这是一个较长的输入文本", "这是一个较长的输出文本"),
                DataSample("sample_003", "", ""),  # 空文本
                DataSample("sample_004", "重复输入", "重复输出"),
                DataSample("sample_005", "重复输入", "不同输出")  # 重复输入
            ]
        )
        
        report = DataValidator.check_data_quality(dataset)
        
        assert report["total_samples"] == 5
        assert report["empty_inputs"] == 1
        assert report["empty_outputs"] == 1
        assert report["unique_inputs"] == 4  # 有一个重复
        assert report["unique_outputs"] == 4
        assert report["avg_input_length"] > 0
        assert report["min_input_length"] == 0
        assert report["max_input_length"] > report["min_input_length"]
    
    def test_check_data_quality_empty_dataset(self):
        """测试空数据集的质量检查"""
        empty_dataset = Dataset(
            name="空数据集",
            industry_domain="测试",
            samples=[]
        )
        
        report = DataValidator.check_data_quality(empty_dataset)
        
        assert report["total_samples"] == 0
        assert report["empty_inputs"] == 0
        assert report["empty_outputs"] == 0
        assert report["avg_input_length"] == 0
        assert report["avg_output_length"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
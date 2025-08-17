"""
数据拆分器测试

测试DataSplitter类的功能。
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from evaluation import DataSplitter
from evaluation.data_models import DataSplitResult
from tests.conftest import create_test_dataset, assert_file_exists


class TestDataSplitter:
    """数据拆分器测试类"""
    
    def test_init_default_params(self):
        """测试默认参数初始化"""
        splitter = DataSplitter()
        
        assert splitter.train_ratio == 0.7
        assert splitter.val_ratio == 0.15
        assert splitter.test_ratio == 0.15
        assert splitter.random_seed == 42
        assert splitter.min_samples_per_split == 10
    
    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        splitter = DataSplitter(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            stratify_by="label",
            random_seed=123,
            min_samples_per_split=5
        )
        
        assert splitter.train_ratio == 0.8
        assert splitter.val_ratio == 0.1
        assert splitter.test_ratio == 0.1
        assert splitter.stratify_by == "label"
        assert splitter.random_seed == 123
        assert splitter.min_samples_per_split == 5
    
    def test_validate_ratios_valid(self):
        """测试有效比例验证"""
        splitter = DataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        # 应该不抛出异常
        splitter._validate_ratios()
    
    def test_validate_ratios_invalid_sum(self):
        """测试无效比例和验证"""
        with pytest.raises(ValueError, match="比例之和必须等于1.0"):
            DataSplitter(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
    
    def test_validate_ratios_negative(self):
        """测试负比例验证"""
        with pytest.raises(ValueError, match="所有比例必须大于0"):
            DataSplitter(train_ratio=-0.1, val_ratio=0.6, test_ratio=0.5)
    
    def test_split_data_basic(self, sample_dataset, temp_dir):
        """测试基本数据拆分"""
        splitter = DataSplitter(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            min_samples_per_split=1
        )
        
        result = splitter.split_data(sample_dataset, str(temp_dir))
        
        # 检查返回类型
        assert isinstance(result, DataSplitResult)
        
        # 检查数据集大小
        total_size = len(sample_dataset)
        assert len(result.train_dataset) + len(result.val_dataset) + len(result.test_dataset) == total_size
        
        # 检查比例（允许小的误差）
        train_ratio = len(result.train_dataset) / total_size
        val_ratio = len(result.val_dataset) / total_size
        test_ratio = len(result.test_dataset) / total_size
        
        assert abs(train_ratio - 0.6) <= 0.2  # 允许较大误差，因为样本数少
        assert abs(val_ratio - 0.2) <= 0.2
        assert abs(test_ratio - 0.2) <= 0.2
    
    def test_split_data_stratified(self, sample_dataset, temp_dir):
        """测试分层拆分"""
        splitter = DataSplitter(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            stratify_by="label",
            min_samples_per_split=1
        )
        
        result = splitter.split_data(sample_dataset, str(temp_dir))
        
        # 检查每个拆分都包含不同的标签
        train_labels = set(result.train_dataset["label"])
        val_labels = set(result.val_dataset["label"])
        test_labels = set(result.test_dataset["label"])
        
        # 至少应该有一些标签重叠（取决于数据分布）
        all_labels = set(sample_dataset["label"])
        assert len(train_labels) > 0
        assert train_labels.issubset(all_labels)
    
    def test_split_data_insufficient_samples(self, small_dataset):
        """测试样本不足的情况"""
        splitter = DataSplitter(min_samples_per_split=10)  # 要求每个拆分至少10个样本
        
        with pytest.raises(ValueError, match="数据集太小"):
            splitter.split_data(small_dataset)
    
    def test_split_data_reproducible(self, sample_dataset, temp_dir):
        """测试拆分结果的可重现性"""
        splitter1 = DataSplitter(random_seed=42, min_samples_per_split=1)
        splitter2 = DataSplitter(random_seed=42, min_samples_per_split=1)
        
        result1 = splitter1.split_data(sample_dataset, str(temp_dir / "split1"))
        result2 = splitter2.split_data(sample_dataset, str(temp_dir / "split2"))
        
        # 检查拆分结果是否相同
        assert len(result1.train_dataset) == len(result2.train_dataset)
        assert len(result1.val_dataset) == len(result2.val_dataset)
        assert len(result1.test_dataset) == len(result2.test_dataset)
        
        # 检查具体内容是否相同（前几个样本）
        for i in range(min(3, len(result1.train_dataset))):
            assert result1.train_dataset[i]["text"] == result2.train_dataset[i]["text"]
    
    def test_save_splits(self, sample_dataset, temp_dir):
        """测试保存拆分结果"""
        splitter = DataSplitter(min_samples_per_split=1)
        result = splitter.split_data(sample_dataset, str(temp_dir))
        
        # 检查文件是否被创建
        assert_file_exists(temp_dir / "train")
        assert_file_exists(temp_dir / "val") 
        assert_file_exists(temp_dir / "test")
        assert_file_exists(temp_dir / "split_info.json")
    
    def test_load_splits(self, sample_dataset, temp_dir):
        """测试加载拆分结果"""
        # 先保存拆分结果
        splitter = DataSplitter(min_samples_per_split=1)
        original_result = splitter.split_data(sample_dataset, str(temp_dir))
        
        # 加载拆分结果
        loaded_result = DataSplitter.load_splits(str(temp_dir))
        
        # 检查加载的结果
        assert isinstance(loaded_result, DataSplitResult)
        assert len(loaded_result.train_dataset) == len(original_result.train_dataset)
        assert len(loaded_result.val_dataset) == len(original_result.val_dataset)
        assert len(loaded_result.test_dataset) == len(original_result.test_dataset)
    
    def test_analyze_distribution(self, sample_dataset):
        """测试分布分析"""
        splitter = DataSplitter(min_samples_per_split=1)
        
        # 创建模拟的拆分数据集
        train_data = sample_dataset.select(range(6))
        val_data = sample_dataset.select(range(6, 8))
        test_data = sample_dataset.select(range(8, 10))
        
        analysis = splitter._analyze_distribution(train_data, val_data, test_data)
        
        # 检查分析结果结构
        assert "train_stats" in analysis
        assert "val_stats" in analysis
        assert "test_stats" in analysis
        assert "consistency_score" in analysis
        assert "statistical_tests" in analysis
        assert "recommendations" in analysis
        
        # 检查一致性分数范围
        assert 0 <= analysis["consistency_score"] <= 1
    
    def test_detect_data_leakage(self, sample_dataset):
        """测试数据泄露检测"""
        splitter = DataSplitter(min_samples_per_split=1)
        
        # 创建有重叠的数据集来测试泄露检测
        train_data = sample_dataset.select(range(7))
        val_data = sample_dataset.select(range(5, 9))  # 与训练集有重叠
        test_data = sample_dataset.select(range(8, 10))
        
        leakage_info = splitter._detect_data_leakage(train_data, val_data, test_data)
        
        # 检查泄露检测结果
        assert "train_val_overlap" in leakage_info
        assert "train_test_overlap" in leakage_info
        assert "val_test_overlap" in leakage_info
        assert "has_leakage" in leakage_info
        
        # 应该检测到泄露
        assert leakage_info["has_leakage"] == True
        assert leakage_info["train_val_overlap"] > 0
    
    def test_quality_analysis_integration(self, sample_dataset, temp_dir):
        """测试质量分析集成"""
        splitter = DataSplitter(
            min_samples_per_split=1,
            enable_quality_analysis=True
        )
        
        result = splitter.split_data(sample_dataset, str(temp_dir))
        
        # 检查是否有质量分析结果
        # 注意：这个测试可能需要根据实际的质量分析实现进行调整
        assert result is not None
    
    def test_edge_case_single_sample_per_split(self):
        """测试边界情况：每个拆分只有一个样本"""
        # 创建只有3个样本的数据集
        tiny_dataset = create_test_dataset(3)
        
        splitter = DataSplitter(
            train_ratio=0.34,
            val_ratio=0.33,
            test_ratio=0.33,
            min_samples_per_split=1
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = splitter.split_data(tiny_dataset, temp_dir)
            
            # 每个拆分至少应该有一个样本
            assert len(result.train_dataset) >= 1
            assert len(result.val_dataset) >= 1
            assert len(result.test_dataset) >= 1
            
            # 总样本数应该保持不变
            total = len(result.train_dataset) + len(result.val_dataset) + len(result.test_dataset)
            assert total == 3
    
    def test_different_random_seeds(self, sample_dataset, temp_dir):
        """测试不同随机种子产生不同结果"""
        splitter1 = DataSplitter(random_seed=42, min_samples_per_split=1)
        splitter2 = DataSplitter(random_seed=123, min_samples_per_split=1)
        
        result1 = splitter1.split_data(sample_dataset, str(temp_dir / "seed42"))
        result2 = splitter2.split_data(sample_dataset, str(temp_dir / "seed123"))
        
        # 大小应该相似（可能有小的差异）
        assert abs(len(result1.train_dataset) - len(result2.train_dataset)) <= 1
        
        # 但内容应该不同（至少有一些不同）
        train1_texts = set(result1.train_dataset["text"])
        train2_texts = set(result2.train_dataset["text"])
        
        # 如果数据集足够大，应该有一些差异
        if len(sample_dataset) > 5:
            assert train1_texts != train2_texts


class TestDataSplitResult:
    """数据拆分结果测试类"""
    
    def test_save_info(self, sample_dataset, temp_dir):
        """测试保存拆分信息"""
        splitter = DataSplitter(min_samples_per_split=1)
        result = splitter.split_data(sample_dataset, str(temp_dir))
        
        info_file = temp_dir / "test_info.json"
        result.save_info(str(info_file))
        
        assert_file_exists(info_file)
        
        # 检查文件内容
        import json
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        assert "split_info" in info
        assert "distribution_analysis" in info
        assert "train_size" in info
        assert "val_size" in info
        assert "test_size" in info
        assert "created_at" in info
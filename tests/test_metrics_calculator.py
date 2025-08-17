"""
指标计算器测试

测试MetricsCalculator类的功能。
"""

import pytest
import numpy as np

from evaluation import MetricsCalculator
from tests.conftest import assert_metrics_valid


class TestMetricsCalculator:
    """指标计算器测试类"""
    
    def test_init_default_params(self):
        """测试默认参数初始化"""
        calculator = MetricsCalculator()
        
        assert calculator.language == "zh"
        assert calculator.device == "cpu"
    
    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        calculator = MetricsCalculator(
            language="en",
            device="cuda",
            cache_dir="./test_cache"
        )
        
        assert calculator.language == "en"
        assert calculator.device == "cuda"
        assert calculator.cache_dir == "./test_cache"
    
    def test_calculate_bleu_basic(self, metrics_calculator):
        """测试基本BLEU计算"""
        predictions = ["这是一个测试", "另一个测试"]
        references = ["这是一个测试", "这是另一个测试"]
        
        result = metrics_calculator.calculate_bleu(predictions, references)
        
        assert "bleu" in result
        assert "bleu_std" in result
        assert "bleu_scores" in result
        assert isinstance(result["bleu"], float)
        assert 0 <= result["bleu"] <= 1
        assert len(result["bleu_scores"]) == len(predictions)
    
    def test_calculate_bleu_empty_input(self, metrics_calculator):
        """测试空输入的BLEU计算"""
        predictions = ["", "测试"]
        references = ["参考", ""]
        
        result = metrics_calculator.calculate_bleu(predictions, references)
        
        assert "bleu" in result
        assert isinstance(result["bleu"], float)
        # 空输入应该得到较低的分数
        assert result["bleu"] >= 0
    
    def test_calculate_bleu_mismatched_length(self, metrics_calculator):
        """测试长度不匹配的BLEU计算"""
        predictions = ["测试1", "测试2"]
        references = ["参考1"]  # 长度不匹配
        
        with pytest.raises(ValueError, match="预测文本和参考文本数量不匹配"):
            metrics_calculator.calculate_bleu(predictions, references)
    
    def test_calculate_rouge_basic(self, metrics_calculator):
        """测试基本ROUGE计算"""
        predictions = ["这是一个测试文本", "另一个测试文本"]
        references = ["这是测试文本", "这是另一个测试"]
        
        result = metrics_calculator.calculate_rouge(predictions, references)
        
        # 检查ROUGE指标
        expected_metrics = ["rouge1", "rouge2", "rougeL"]
        for metric in expected_metrics:
            assert metric in result
            assert f"{metric}_std" in result
            assert isinstance(result[metric], float)
            assert 0 <= result[metric] <= 1
    
    def test_calculate_classification_metrics_basic(self, metrics_calculator):
        """测试基本分类指标计算"""
        predictions = ["positive", "negative", "positive", "neutral"]
        references = ["positive", "positive", "negative", "neutral"]
        
        result = metrics_calculator.calculate_classification_metrics(predictions, references)
        
        # 检查基本指标
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "num_classes" in result
        assert "total_samples" in result
        
        # 检查指标范围
        assert 0 <= result["accuracy"] <= 1
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1"] <= 1
        
        # 检查样本数和类别数
        assert result["total_samples"] == 4
        assert result["num_classes"] == 3  # positive, negative, neutral
    
    def test_calculate_classification_metrics_perfect(self, metrics_calculator):
        """测试完美分类的指标计算"""
        predictions = ["A", "B", "C"]
        references = ["A", "B", "C"]
        
        result = metrics_calculator.calculate_classification_metrics(predictions, references)
        
        # 完美分类应该得到1.0的分数
        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
    
    def test_calculate_classification_metrics_worst(self, metrics_calculator):
        """测试最差分类的指标计算"""
        predictions = ["A", "A", "A"]
        references = ["B", "C", "D"]
        
        result = metrics_calculator.calculate_classification_metrics(predictions, references)
        
        # 最差分类应该得到0.0的准确率
        assert result["accuracy"] == 0.0
    
    def test_semantic_similarity_cosine(self, metrics_calculator):
        """测试余弦相似度计算"""
        text1_list = ["这是第一个文本", "这是第二个文本"]
        text2_list = ["这是第一个文本", "这是完全不同的文本"]
        
        result = metrics_calculator.calculate_semantic_similarity(
            text1_list, text2_list, method="cosine"
        )
        
        assert "cosine_similarity" in result
        assert "cosine_similarity_std" in result
        assert "similarities" in result
        assert isinstance(result["cosine_similarity"], float)
        assert 0 <= result["cosine_similarity"] <= 1
        assert len(result["similarities"]) == 2
    
    def test_semantic_similarity_jaccard(self, metrics_calculator):
        """测试Jaccard相似度计算"""
        text1_list = ["这是测试文本", "另一个测试"]
        text2_list = ["这是测试", "完全不同的文本"]
        
        result = metrics_calculator.calculate_semantic_similarity(
            text1_list, text2_list, method="jaccard"
        )
        
        assert "jaccard_similarity" in result
        assert "jaccard_similarity_std" in result
        assert isinstance(result["jaccard_similarity"], float)
        assert 0 <= result["jaccard_similarity"] <= 1
    
    def test_semantic_similarity_invalid_method(self, metrics_calculator):
        """测试无效相似度方法"""
        text1_list = ["文本1"]
        text2_list = ["文本2"]
        
        with pytest.raises(ValueError, match="不支持的相似度计算方法"):
            metrics_calculator.calculate_semantic_similarity(
                text1_list, text2_list, method="invalid_method"
            )
    
    def test_calculate_all_metrics_generation(self, metrics_calculator):
        """测试生成任务的所有指标计算"""
        predictions = ["生成的文本1", "生成的文本2"]
        references = ["参考文本1", "参考文本2"]
        
        result = metrics_calculator.calculate_all_metrics(
            predictions, references, task_type="generation"
        )
        
        # 检查基本字段
        assert "num_samples" in result
        assert "task_type" in result
        assert result["task_type"] == "generation"
        assert result["num_samples"] == 2
        
        # 应该包含一些生成任务的指标
        # 注意：具体的指标可能取决于可用的库
        assert isinstance(result, dict)
        assert len(result) > 2  # 至少有num_samples和task_type
    
    def test_calculate_all_metrics_classification(self, metrics_calculator):
        """测试分类任务的所有指标计算"""
        predictions = ["positive", "negative", "neutral"]
        references = ["positive", "positive", "neutral"]
        
        result = metrics_calculator.calculate_all_metrics(
            predictions, references, task_type="classification"
        )
        
        # 检查分类指标
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert result["task_type"] == "classification"
        
        # 验证指标有效性
        assert_metrics_valid({
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"]
        })
    
    def test_tokenize_chinese(self, metrics_calculator):
        """测试中文分词"""
        text = "这是一个中文测试文本"
        tokens = metrics_calculator._tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # 中文应该按字符分词
        assert len(tokens) == len(text.replace(" ", ""))
    
    def test_tokenize_english(self):
        """测试英文分词"""
        calculator = MetricsCalculator(language="en")
        text = "This is an English test text"
        tokens = calculator._tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # 英文应该按单词分词
        expected_tokens = text.lower().split()
        assert len(tokens) >= len(expected_tokens)
    
    def test_cosine_similarity_identical(self, metrics_calculator):
        """测试相同文本的余弦相似度"""
        text = "这是测试文本"
        similarity = metrics_calculator._cosine_similarity(text, text)
        
        # 相同文本的相似度应该是1.0
        assert abs(similarity - 1.0) < 0.001
    
    def test_cosine_similarity_different(self, metrics_calculator):
        """测试不同文本的余弦相似度"""
        text1 = "这是第一个文本"
        text2 = "完全不同的内容"
        similarity = metrics_calculator._cosine_similarity(text1, text2)
        
        # 不同文本的相似度应该小于1.0
        assert 0 <= similarity < 1.0
    
    def test_jaccard_similarity_identical(self, metrics_calculator):
        """测试相同文本的Jaccard相似度"""
        text = "这是测试文本"
        similarity = metrics_calculator._jaccard_similarity(text, text)
        
        # 相同文本的Jaccard相似度应该是1.0
        assert similarity == 1.0
    
    def test_jaccard_similarity_no_overlap(self, metrics_calculator):
        """测试无重叠文本的Jaccard相似度"""
        text1 = "第一个文本"
        text2 = "完全不同内容"
        similarity = metrics_calculator._jaccard_similarity(text1, text2)
        
        # 无重叠文本的相似度应该是0.0
        assert similarity == 0.0
    
    def test_confusion_matrix(self, metrics_calculator):
        """测试混淆矩阵计算"""
        predictions = ["A", "B", "A", "C"]
        references = ["A", "A", "B", "C"]
        labels = ["A", "B", "C"]
        
        matrix = metrics_calculator._calculate_confusion_matrix(predictions, references, labels)
        
        # 检查矩阵结构
        assert isinstance(matrix, dict)
        for true_label in labels:
            assert true_label in matrix
            for pred_label in labels:
                assert pred_label in matrix[true_label]
                assert isinstance(matrix[true_label][pred_label], int)
        
        # 检查具体值
        assert matrix["A"]["A"] == 1  # 正确预测A为A
        assert matrix["A"]["B"] == 1  # 错误预测A为B
        assert matrix["B"]["A"] == 1  # 错误预测B为A
        assert matrix["C"]["C"] == 1  # 正确预测C为C
"""
质量分析器测试

测试QualityAnalyzer类的功能。
"""

import pytest
from datasets import Dataset

from evaluation import QualityAnalyzer
from evaluation.data_models import DataQualityReport, ResponseQualityReport
from tests.conftest import create_test_dataset


class TestQualityAnalyzer:
    """质量分析器测试类"""
    
    def test_init_default_params(self):
        """测试默认参数初始化"""
        analyzer = QualityAnalyzer()
        
        assert analyzer.min_length == 5
        assert analyzer.max_length == 2048
        assert analyzer.length_outlier_threshold == 3.0
        assert analyzer.vocab_diversity_threshold == 0.5
        assert analyzer.language == "zh"
    
    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        analyzer = QualityAnalyzer(
            min_length=10,
            max_length=1000,
            length_outlier_threshold=2.5,
            vocab_diversity_threshold=0.7,
            language="en"
        )
        
        assert analyzer.min_length == 10
        assert analyzer.max_length == 1000
        assert analyzer.length_outlier_threshold == 2.5
        assert analyzer.vocab_diversity_threshold == 0.7
        assert analyzer.language == "en"
    
    def test_analyze_data_quality_basic(self, quality_analyzer, sample_dataset):
        """测试基本数据质量分析"""
        report = quality_analyzer.analyze_data_quality(sample_dataset)
        
        # 检查报告类型和基本字段
        assert isinstance(report, DataQualityReport)
        assert hasattr(report, 'total_samples')
        assert hasattr(report, 'length_stats')
        assert hasattr(report, 'vocab_diversity')
        assert hasattr(report, 'class_distribution')
        assert hasattr(report, 'quality_issues')
        assert hasattr(report, 'quality_score')
        
        # 检查基本统计
        assert report.total_samples == len(sample_dataset)
        assert 0 <= report.quality_score <= 1
    
    def test_analyze_length_distribution(self, quality_analyzer):
        """测试长度分布分析"""
        texts = ["短", "中等长度文本", "这是一个相对较长的测试文本内容"]
        
        stats = quality_analyzer._analyze_length_distribution(texts)
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "outliers" in stats
        
        # 检查统计值合理性
        assert stats["min"] <= stats["mean"] <= stats["max"]
        assert stats["min"] <= stats["median"] <= stats["max"]
        assert stats["std"] >= 0
    
    def test_analyze_vocab_diversity(self, quality_analyzer):
        """测试词汇多样性分析"""
        texts = [
            "这是第一个测试文本",
            "这是第二个测试文本", 
            "完全不同的内容和词汇",
            "重复 重复 重复 重复"
        ]
        
        diversity = quality_analyzer._analyze_vocab_diversity(texts)
        
        assert "unique_tokens" in diversity
        assert "total_tokens" in diversity
        assert "diversity_ratio" in diversity
        assert "avg_tokens_per_text" in diversity
        
        # 检查多样性比例范围
        assert 0 <= diversity["diversity_ratio"] <= 1
        assert diversity["unique_tokens"] <= diversity["total_tokens"]
    
    def test_analyze_class_distribution(self, quality_analyzer, sample_dataset):
        """测试类别分布分析"""
        distribution = quality_analyzer._analyze_class_distribution(sample_dataset, "label")
        
        assert "class_counts" in distribution
        assert "class_ratios" in distribution
        assert "num_classes" in distribution
        assert "balance_score" in distribution
        
        # 检查平衡分数范围
        assert 0 <= distribution["balance_score"] <= 1
        
        # 检查类别数量
        unique_labels = set(sample_dataset["label"])
        assert distribution["num_classes"] == len(unique_labels)
    
    def test_detect_quality_issues(self, quality_analyzer):
        """测试质量问题检测"""
        # 创建有问题的数据集
        problematic_data = [
            {"text": "", "label": "empty"},  # 空文本
            {"text": "a", "label": "too_short"},  # 过短
            {"text": "x" * 3000, "label": "too_long"},  # 过长
            {"text": "正常文本", "label": "normal"},
            {"text": "重复 重复 重复 重复 重复", "label": "repetitive"}  # 重复内容
        ]
        dataset = Dataset.from_list(problematic_data)
        
        issues = quality_analyzer._detect_quality_issues(dataset)
        
        assert isinstance(issues, list)
        
        # 检查是否检测到各种问题
        issue_types = [issue["type"] for issue in issues]
        assert "empty_text" in issue_types
        assert "text_too_short" in issue_types
        assert "text_too_long" in issue_types
    
    def test_suggest_improvements(self, quality_analyzer):
        """测试改进建议"""
        # 创建模拟质量报告
        from evaluation.data_models import DataQualityReport
        
        report = DataQualityReport(
            total_samples=100,
            length_stats={"mean": 50, "std": 20, "outliers": 5},
            vocab_diversity={"diversity_ratio": 0.3},
            class_distribution={"balance_score": 0.4},
            quality_issues=[
                {"type": "text_too_short", "count": 10},
                {"type": "low_diversity", "count": 1}
            ],
            quality_score=0.6
        )
        
        suggestions = quality_analyzer.suggest_improvements(report)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # 检查建议内容
        suggestion_text = " ".join(suggestions)
        assert "过短" in suggestion_text or "多样性" in suggestion_text
    
    def test_analyze_response_quality_basic(self, quality_analyzer):
        """测试响应质量分析"""
        responses = [
            "这是一个流畅且相关的回答",
            "回答内容连贯性较好",
            "简短回答",
            "这个回答可能不太相关或者质量较低"
        ]
        
        report = quality_analyzer.analyze_response_quality(responses)
        
        assert isinstance(report, ResponseQualityReport)
        assert hasattr(report, 'fluency_score')
        assert hasattr(report, 'coherence_score')
        assert hasattr(report, 'relevance_score')
        assert hasattr(report, 'overall_score')
        
        # 检查分数范围
        assert 0 <= report.fluency_score <= 1
        assert 0 <= report.coherence_score <= 1
        assert 0 <= report.relevance_score <= 1
        assert 0 <= report.overall_score <= 1
    
    def test_analyze_response_quality_with_references(self, quality_analyzer):
        """测试带参考答案的响应质量分析"""
        responses = ["这是预测回答", "另一个预测"]
        references = ["这是参考答案", "另一个参考"]
        
        report = quality_analyzer.analyze_response_quality(responses, references)
        
        assert isinstance(report, ResponseQualityReport)
        # 有参考答案时应该有更准确的相关性评分
        assert hasattr(report, 'relevance_score')
    
    def test_calculate_fluency_score(self, quality_analyzer):
        """测试流畅度分数计算"""
        texts = [
            "这是一个流畅的中文句子。",
            "不太流畅 的 句子 结构",
            "完全不通顺语法错误很多"
        ]
        
        scores = [quality_analyzer._calculate_fluency_score(text) for text in texts]
        
        # 检查分数范围
        for score in scores:
            assert 0 <= score <= 1
        
        # 第一个句子应该比后面的更流畅
        assert scores[0] >= scores[1]
        assert scores[0] >= scores[2]
    
    def test_calculate_coherence_score(self, quality_analyzer):
        """测试连贯性分数计算"""
        texts = [
            "这是第一句。这是相关的第二句。",
            "这是第一句。完全不相关的内容。",
            "单独一句话。"
        ]
        
        scores = [quality_analyzer._calculate_coherence_score(text) for text in texts]
        
        # 检查分数范围
        for score in scores:
            assert 0 <= score <= 1
    
    def test_detect_repetitive_content(self, quality_analyzer):
        """测试重复内容检测"""
        repetitive_text = "重复内容 重复内容 重复内容 重复内容"
        normal_text = "这是正常的文本内容，没有过度重复"
        
        is_repetitive_1 = quality_analyzer._detect_repetitive_content(repetitive_text)
        is_repetitive_2 = quality_analyzer._detect_repetitive_content(normal_text)
        
        assert is_repetitive_1 == True
        assert is_repetitive_2 == False
    
    def test_calculate_quality_score(self, quality_analyzer):
        """测试质量分数计算"""
        # 创建模拟统计数据
        length_stats = {"outliers": 2}
        vocab_diversity = {"diversity_ratio": 0.7}
        class_distribution = {"balance_score": 0.8}
        issues = [{"type": "text_too_short", "count": 1}]
        total_samples = 100
        
        score = quality_analyzer._calculate_quality_score(
            length_stats, vocab_diversity, class_distribution, issues, total_samples
        )
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
    
    def test_empty_dataset(self, quality_analyzer):
        """测试空数据集处理"""
        empty_dataset = Dataset.from_list([])
        
        with pytest.raises(ValueError, match="数据集为空"):
            quality_analyzer.analyze_data_quality(empty_dataset)
    
    def test_missing_text_field(self, quality_analyzer):
        """测试缺少文本字段的数据集"""
        dataset_without_text = Dataset.from_list([
            {"label": "A", "other_field": "value"},
            {"label": "B", "other_field": "value2"}
        ])
        
        with pytest.raises(ValueError, match="数据集必须包含.*字段"):
            quality_analyzer.analyze_data_quality(dataset_without_text)
    
    def test_custom_text_field(self, quality_analyzer):
        """测试自定义文本字段"""
        dataset = Dataset.from_list([
            {"content": "文本内容1", "label": "A"},
            {"content": "文本内容2", "label": "B"}
        ])
        
        report = quality_analyzer.analyze_data_quality(dataset, text_field="content")
        
        assert isinstance(report, DataQualityReport)
        assert report.total_samples == 2
    
    def test_generate_quality_report_html(self, quality_analyzer, sample_dataset, temp_dir):
        """测试生成HTML质量报告"""
        report = quality_analyzer.analyze_data_quality(sample_dataset)
        
        output_path = temp_dir / "quality_report.html"
        quality_analyzer.generate_quality_report(report, str(output_path), format="html")
        
        assert output_path.exists()
        
        # 检查HTML内容
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "<html>" in content
        assert "质量分析报告" in content
        assert str(report.total_samples) in content
    
    def test_generate_quality_report_json(self, quality_analyzer, sample_dataset, temp_dir):
        """测试生成JSON质量报告"""
        report = quality_analyzer.analyze_data_quality(sample_dataset)
        
        output_path = temp_dir / "quality_report.json"
        quality_analyzer.generate_quality_report(report, str(output_path), format="json")
        
        assert output_path.exists()
        
        # 检查JSON内容
        import json
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "total_samples" in data
        assert data["total_samples"] == report.total_samples
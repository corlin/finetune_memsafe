"""
评估结果聚合器单元测试
"""

import pytest
from datetime import datetime
from industry_evaluation.core.result_aggregator import ResultAggregator
from industry_evaluation.models.data_models import (
    EvaluationScore, SampleResult, EvaluationConfig, ErrorAnalysis
)


class TestResultAggregator:
    """结果聚合器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.weight_config = {
            "knowledge": 0.4,
            "terminology": 0.3,
            "reasoning": 0.3
        }
        self.aggregator = ResultAggregator(self.weight_config)
    
    def test_initialization(self):
        """测试初始化"""
        assert self.aggregator.weight_config == self.weight_config
        
        # 测试无权重配置的初始化
        aggregator = ResultAggregator()
        assert aggregator.weight_config == {}
    
    def test_aggregate_scores_empty(self):
        """测试聚合空分数列表"""
        result = self.aggregator.aggregate_scores([])
        
        assert result.overall_score == 0.0
        assert result.dimension_scores == {}
        assert result.confidence == 0.0
    
    def test_aggregate_scores(self):
        """测试聚合分数"""
        scores = [
            EvaluationScore(
                overall_score=0.8,
                dimension_scores={"knowledge": 0.9, "terminology": 0.7},
                confidence=0.9
            ),
            EvaluationScore(
                overall_score=0.7,
                dimension_scores={"knowledge": 0.8, "terminology": 0.6},
                confidence=0.8
            )
        ]
        
        result = self.aggregator.aggregate_scores(scores)
        
        # 检查维度分数（应该是平均值）
        assert result.dimension_scores["knowledge"] == 0.85  # (0.9 + 0.8) / 2
        assert result.dimension_scores["terminology"] == 0.65  # (0.7 + 0.6) / 2
        
        # 检查置信度（应该是平均值）
        assert result.confidence == 0.85  # (0.9 + 0.8) / 2
        
        # 检查总分（应该是加权平均）
        expected_overall = 0.85 * 0.4 + 0.65 * 0.3  # knowledge * weight + terminology * weight
        assert abs(result.overall_score - expected_overall) < 0.01
    
    def test_calculate_weighted_score_no_weights(self):
        """测试无权重配置的分数计算"""
        aggregator = ResultAggregator()  # 无权重配置
        
        dimension_scores = {"knowledge": 0.8, "terminology": 0.6}
        score = aggregator._calculate_weighted_score(dimension_scores)
        
        # 应该是平均分
        assert score == 0.7  # (0.8 + 0.6) / 2
    
    def test_calculate_weighted_score_with_weights(self):
        """测试有权重配置的分数计算"""
        dimension_scores = {"knowledge": 0.8, "terminology": 0.6, "reasoning": 0.9}
        score = self.aggregator._calculate_weighted_score(dimension_scores)
        
        # 加权平均：0.8*0.4 + 0.6*0.3 + 0.9*0.3 = 0.32 + 0.18 + 0.27 = 0.77
        expected = 0.8 * 0.4 + 0.6 * 0.3 + 0.9 * 0.3
        assert abs(score - expected) < 0.01
    
    def test_calculate_weighted_score_empty(self):
        """测试空维度分数的计算"""
        score = self.aggregator._calculate_weighted_score({})
        assert score == 0.0
    
    def test_aggregate_sample_results_empty(self):
        """测试聚合空样本结果"""
        result = self.aggregator.aggregate_sample_results([])
        assert result == {}
    
    def test_aggregate_sample_results(self):
        """测试聚合样本结果"""
        sample_results = [
            SampleResult(
                sample_id="001",
                input_text="输入1",
                model_output="输出1",
                expected_output="期望1",
                dimension_scores={"knowledge": 0.8, "terminology": 0.7},
                error_types=["knowledge_error"],
                processing_time=1.5
            ),
            SampleResult(
                sample_id="002",
                input_text="输入2",
                model_output="输出2",
                expected_output="期望2",
                dimension_scores={"knowledge": 0.9, "terminology": 0.6},
                error_types=[],
                processing_time=2.0
            )
        ]
        
        result = self.aggregator.aggregate_sample_results(sample_results)
        
        assert result["total_samples"] == 2
        
        # 检查维度统计
        knowledge_stats = result["dimension_statistics"]["knowledge"]
        assert knowledge_stats["mean"] == 0.85  # (0.8 + 0.9) / 2
        assert knowledge_stats["min"] == 0.8
        assert knowledge_stats["max"] == 0.9
        
        # 检查错误统计
        error_stats = result["error_statistics"]
        assert error_stats["total_errors"] == 1
        assert error_stats["samples_with_errors"] == 1
        assert error_stats["error_rate"] == 0.5  # 1/2
        
        # 检查性能统计
        perf_stats = result["performance_statistics"]
        assert perf_stats["avg_processing_time"] == 1.75  # (1.5 + 2.0) / 2
        assert perf_stats["total_processing_time"] == 3.5
    
    def test_create_evaluation_result_empty(self):
        """测试创建空评估结果"""
        config = EvaluationConfig(
            industry_domain="测试",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={}
        )
        
        result = self.aggregator.create_evaluation_result("task1", "model1", [], config)
        
        assert result.task_id == "task1"
        assert result.model_id == "model1"
        assert result.overall_score == 0.0
        assert result.dimension_scores == {}
        assert len(result.detailed_results) == 0
    
    def test_create_evaluation_result(self):
        """测试创建评估结果"""
        sample_results = [
            SampleResult(
                sample_id="001",
                input_text="输入1",
                model_output="输出1",
                expected_output="期望1",
                dimension_scores={"knowledge": 0.8},
                error_types=["knowledge_error"]
            )
        ]
        
        config = EvaluationConfig(
            industry_domain="测试",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={}
        )
        
        result = self.aggregator.create_evaluation_result("task1", "model1", sample_results, config)
        
        assert result.task_id == "task1"
        assert result.model_id == "model1"
        assert result.overall_score == 0.8
        assert result.dimension_scores["knowledge"] == 0.8
        assert len(result.detailed_results) == 1
        assert len(result.improvement_suggestions) > 0
    
    def test_calculate_error_statistics(self):
        """测试错误统计计算"""
        sample_results = [
            SampleResult("001", "输入1", "输出1", "期望1", {}, ["knowledge_error", "terminology_error"]),
            SampleResult("002", "输入2", "输出2", "期望2", {}, ["knowledge_error"]),
            SampleResult("003", "输入3", "输出3", "期望3", {}, [])
        ]
        
        stats = self.aggregator._calculate_error_statistics(sample_results)
        
        assert stats["total_errors"] == 3
        assert stats["samples_with_errors"] == 2
        assert stats["error_rate"] == 2/3  # 2个样本有错误，共3个样本
        assert stats["avg_errors_per_sample"] == 1.0  # 3个错误，3个样本
        assert stats["error_distribution"]["knowledge_error"] == 2
        assert stats["error_distribution"]["terminology_error"] == 1
    
    def test_identify_error_patterns(self):
        """测试错误模式识别"""
        sample_results = [
            SampleResult("001", "短", "短输出", "期望1", {}, ["knowledge_error"]),
            SampleResult("002", "短", "短输出", "期望2", {}, ["knowledge_error"]),
            SampleResult("003", "长" * 100, "长输出" * 50, "期望3", {}, ["terminology_error"]),
            SampleResult("004", "长" * 100, "长输出" * 50, "期望4", {}, ["terminology_error"])
        ]
        
        patterns = self.aggregator._identify_error_patterns(sample_results)
        
        # 应该识别出错误组合模式
        assert len(patterns) > 0
        # 应该识别出短文本或长文本的错误模式
        pattern_text = " ".join(patterns)
        assert "错误组合" in pattern_text or "文本" in pattern_text
    
    def test_determine_severity_levels(self):
        """测试错误严重程度确定"""
        error_distribution = {
            "critical_error": 50,  # 50%
            "high_error": 30,      # 30%
            "medium_error": 15,    # 15%
            "low_error": 5         # 5%
        }
        
        severity = self.aggregator._determine_severity_levels(error_distribution)
        
        assert severity["critical_error"] == "critical"
        assert severity["high_error"] == "high"
        assert severity["medium_error"] == "medium"
        assert severity["low_error"] == "low"
    
    def test_identify_improvement_areas(self):
        """测试改进领域识别"""
        error_distribution = {"knowledge_error": 10, "terminology_error": 5}
        sample_results = [
            SampleResult("001", "输入1", "输出1", "期望1", {"knowledge": 0.3, "terminology": 0.8}, [])
        ]
        
        areas = self.aggregator._identify_improvement_areas(error_distribution, sample_results)
        
        assert len(areas) > 0
        # 应该包含基于错误类型的改进领域
        assert "专业知识掌握" in areas or "术语使用准确性" in areas
        # 应该包含基于低分维度的改进领域（knowledge分数0.3很低）
        assert "knowledge" in areas
    
    def test_generate_improvement_suggestions(self):
        """测试改进建议生成"""
        dimension_scores = {"knowledge": 0.5, "terminology": 0.8}  # knowledge分数较低
        error_analysis = ErrorAnalysis(
            error_distribution={"knowledge_error": 10},
            common_patterns=[],
            severity_levels={"knowledge_error": "critical"},
            improvement_areas=["专业知识掌握"]
        )
        
        suggestions = self.aggregator._generate_improvement_suggestions(dimension_scores, error_analysis)
        
        assert len(suggestions) > 0
        # 应该包含针对低分维度的建议
        suggestion_text = " ".join(suggestions)
        assert "专业知识" in suggestion_text or "knowledge" in suggestion_text
        # 应该包含针对严重错误的建议
        assert "知识" in suggestion_text
    
    def test_aggregate_details(self):
        """测试详细信息聚合"""
        details_list = [
            {"score": 0.8, "length": 100, "type": "text"},
            {"score": 0.6, "length": 150, "type": "text"},
            {"score": 0.9, "length": 80}  # 缺少type字段
        ]
        
        result = self.aggregator._aggregate_details(details_list)
        
        assert result["count"] == 3
        # score和length是公共的数值字段，应该被聚合
        assert "score" in result["aggregated_values"]
        assert "length" in result["aggregated_values"]
        
        # 检查聚合的统计值
        score_stats = result["aggregated_values"]["score"]
        assert score_stats["mean"] == 0.7666666666666667  # (0.8 + 0.6 + 0.9) / 3
        assert score_stats["min"] == 0.6
        assert score_stats["max"] == 0.9


if __name__ == "__main__":
    pytest.main([__file__])
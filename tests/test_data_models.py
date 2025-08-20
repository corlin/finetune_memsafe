"""
数据模型单元测试
"""

import pytest
from datetime import datetime
from industry_evaluation.models.data_models import (
    EvaluationConfig, EvaluationScore, SampleResult, ErrorAnalysis,
    EvaluationResult, DataSample, Dataset, ProgressInfo, Criterion,
    Explanation, Report, EvaluationStatus, ErrorType
)


class TestEvaluationConfig:
    """评估配置测试"""
    
    def test_valid_config(self):
        """测试有效配置"""
        config = EvaluationConfig(
            industry_domain="金融",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.6, "terminology": 0.4},
            threshold_config={"pass": 0.6}
        )
        assert config.industry_domain == "金融"
        assert len(config.evaluation_dimensions) == 2
    
    def test_invalid_domain(self):
        """测试无效领域"""
        with pytest.raises(ValueError, match="行业领域不能为空"):
            EvaluationConfig(
                industry_domain="",
                evaluation_dimensions=["knowledge"],
                weight_config={"knowledge": 1.0},
                threshold_config={}
            )
    
    def test_invalid_weights(self):
        """测试无效权重"""
        with pytest.raises(ValueError, match="权重总和应为1.0"):
            EvaluationConfig(
                industry_domain="金融",
                evaluation_dimensions=["knowledge"],
                weight_config={"knowledge": 0.5},  # 总和不为1.0
                threshold_config={}
            )
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = EvaluationConfig(
            industry_domain="金融",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"pass": 0.6}
        )
        data = config.to_dict()
        assert data["industry_domain"] == "金融"
        assert "evaluation_dimensions" in data
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "industry_domain": "金融",
            "evaluation_dimensions": ["knowledge"],
            "weight_config": {"knowledge": 1.0},
            "threshold_config": {"pass": 0.6}
        }
        config = EvaluationConfig.from_dict(data)
        assert config.industry_domain == "金融"


class TestEvaluationScore:
    """评估分数测试"""
    
    def test_valid_score(self):
        """测试有效分数"""
        score = EvaluationScore(
            overall_score=0.8,
            dimension_scores={"knowledge": 0.9, "terminology": 0.7}
        )
        assert score.overall_score == 0.8
        assert score.confidence == 1.0
    
    def test_invalid_overall_score(self):
        """测试无效总分"""
        with pytest.raises(ValueError, match="总分应在0-1之间"):
            EvaluationScore(
                overall_score=1.5,
                dimension_scores={}
            )
    
    def test_invalid_dimension_score(self):
        """测试无效维度分数"""
        with pytest.raises(ValueError, match="维度.*的分数应在0-1之间"):
            EvaluationScore(
                overall_score=0.8,
                dimension_scores={"knowledge": 1.2}
            )


class TestSampleResult:
    """样本结果测试"""
    
    def test_sample_result_creation(self):
        """测试样本结果创建"""
        result = SampleResult(
            sample_id="test_001",
            input_text="测试输入",
            model_output="模型输出",
            expected_output="期望输出",
            dimension_scores={"knowledge": 0.8}
        )
        assert result.sample_id == "test_001"
        assert len(result.error_types) == 0
    
    def test_get_overall_score(self):
        """测试计算总分"""
        result = SampleResult(
            sample_id="test_001",
            input_text="测试输入",
            model_output="模型输出",
            expected_output="期望输出",
            dimension_scores={"knowledge": 0.8, "terminology": 0.6}
        )
        
        # 无权重情况
        score = result.get_overall_score({})
        assert score == 0.7  # (0.8 + 0.6) / 2
        
        # 有权重情况
        weights = {"knowledge": 0.7, "terminology": 0.3}
        score = result.get_overall_score(weights)
        assert score == 0.74  # 0.8 * 0.7 + 0.6 * 0.3
    
    def test_to_dict(self):
        """测试转换为字典"""
        result = SampleResult(
            sample_id="test_001",
            input_text="测试输入",
            model_output="模型输出",
            expected_output="期望输出",
            dimension_scores={"knowledge": 0.8}
        )
        data = result.to_dict()
        assert data["sample_id"] == "test_001"
        assert "timestamp" in data


class TestDataset:
    """数据集测试"""
    
    def test_dataset_creation(self):
        """测试数据集创建"""
        samples = [
            DataSample("001", "输入1", "输出1"),
            DataSample("002", "输入2", "输出2")
        ]
        dataset = Dataset(
            name="测试数据集",
            samples=samples,
            industry_domain="金融"
        )
        assert len(dataset) == 2
        assert dataset.name == "测试数据集"
    
    def test_dataset_validation(self):
        """测试数据集验证"""
        # 有效数据集
        samples = [DataSample("001", "输入1", "输出1")]
        dataset = Dataset("测试", samples, "金融")
        assert dataset.validate() is True
        
        # 无效数据集（空名称）
        dataset = Dataset("", samples, "金融")
        assert dataset.validate() is False
        
        # 无效数据集（空样本）
        dataset = Dataset("测试", [], "金融")
        assert dataset.validate() is False
    
    def test_get_sample_by_id(self):
        """测试根据ID获取样本"""
        samples = [
            DataSample("001", "输入1", "输出1"),
            DataSample("002", "输入2", "输出2")
        ]
        dataset = Dataset("测试", samples, "金融")
        
        sample = dataset.get_sample_by_id("001")
        assert sample is not None
        assert sample.sample_id == "001"
        
        sample = dataset.get_sample_by_id("999")
        assert sample is None


class TestProgressInfo:
    """进度信息测试"""
    
    def test_progress_calculation(self):
        """测试进度计算"""
        progress = ProgressInfo(
            task_id="task_001",
            current_step=1,
            total_steps=10,
            current_sample=25,
            total_samples=100,
            status=EvaluationStatus.RUNNING
        )
        
        assert progress.get_progress_percentage() == 25.0
        assert progress.get_elapsed_time() >= 0
    
    def test_zero_samples(self):
        """测试零样本情况"""
        progress = ProgressInfo(
            task_id="task_001",
            current_step=1,
            total_steps=10,
            current_sample=0,
            total_samples=0,
            status=EvaluationStatus.PENDING
        )
        
        assert progress.get_progress_percentage() == 0.0


class TestErrorAnalysis:
    """错误分析测试"""
    
    def test_error_analysis(self):
        """测试错误分析"""
        analysis = ErrorAnalysis(
            error_distribution={"knowledge": 5, "terminology": 3},
            common_patterns=["概念混淆", "术语错误"],
            severity_levels={"knowledge": "high", "terminology": "medium"},
            improvement_areas=["专业知识", "术语使用"]
        )
        
        assert analysis.get_total_errors() == 8
        assert analysis.get_error_rate(100) == 0.08
        assert analysis.get_error_rate(0) == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
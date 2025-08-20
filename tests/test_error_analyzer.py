"""
错误分析器单元测试
"""

import pytest
from datetime import datetime
from industry_evaluation.analysis.error_analyzer import (
    ErrorAnalysisEngine, ErrorClassifier, ErrorStatisticsAnalyzer,
    ErrorInstance, ErrorPattern, ErrorSeverity
)
from industry_evaluation.models.data_models import SampleResult, ErrorAnalysis


class TestErrorClassifier:
    """错误分类器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.classifier = ErrorClassifier()
    
    def test_classify_errors_basic(self):
        """测试基本错误分类"""
        sample_results = [
            SampleResult(
                sample_id="sample_001",
                input_text="什么是机器学习？",
                model_output="机器学习是深度学习的一种",  # 概念错误
                expected_output="机器学习是人工智能的分支",
                dimension_scores={"knowledge": 0.3},
                error_types=["knowledge_error"]
            ),
            SampleResult(
                sample_id="sample_002", 
                input_text="解释神经网络",
                model_output="神经网络使用ML算法和机器学习方法",  # 术语不一致
                expected_output="神经网络是深度学习的基础",
                dimension_scores={"terminology": 0.4},
                error_types=["terminology_error"]
            )
        ]
        
        classified_errors = self.classifier.classify_errors(sample_results)
        
        assert "knowledge_error" in classified_errors
        assert "terminology_error" in classified_errors
        assert len(classified_errors["knowledge_error"]) > 0
        assert len(classified_errors["terminology_error"]) > 0
    
    def test_analyze_knowledge_errors(self):
        """测试知识错误分析"""
        sample = SampleResult(
            sample_id="test_001",
            input_text="机器学习的历史",
            model_output="机器学习在1990年被发明，深度学习是机器学习的父类",
            expected_output="机器学习概念在1950年代提出，深度学习是机器学习的子领域",
            dimension_scores={"knowledge": 0.2},
            error_types=["knowledge_error"]
        )
        
        errors = self.classifier._analyze_knowledge_errors(sample)
        
        assert len(errors) > 0
        assert all(error.error_type == "knowledge_error" for error in errors)
        assert any("概念" in error.error_message for error in errors)
    
    def test_analyze_terminology_errors(self):
        """测试术语错误分析"""
        sample = SampleResult(
            sample_id="test_002",
            input_text="介绍AI技术",
            model_output="AI智能技术包括机器学习和ML算法",  # 重复表达和术语不一致
            expected_output="人工智能技术包括机器学习算法",
            dimension_scores={"terminology": 0.3},
            error_types=["terminology_error"]
        )
        
        errors = self.classifier._analyze_terminology_errors(sample)
        
        assert len(errors) > 0
        assert all(error.error_type == "terminology_error" for error in errors)
    
    def test_analyze_reasoning_errors(self):
        """测试推理错误分析"""
        sample = SampleResult(
            sample_id="test_003",
            input_text="分析因果关系",
            model_output="机器学习不能处理数据，但是机器学习可以处理大量数据",  # 逻辑矛盾
            expected_output="机器学习可以有效处理大量数据",
            dimension_scores={"reasoning": 0.1},
            error_types=["reasoning_error"]
        )
        
        errors = self.classifier._analyze_reasoning_errors(sample)
        
        assert len(errors) > 0
        assert all(error.error_type == "reasoning_error" for error in errors)
    
    def test_analyze_context_errors(self):
        """测试上下文错误分析"""
        sample = SampleResult(
            sample_id="test_004",
            input_text="什么是深度学习？请详细解释其原理和应用",
            model_output="如何使用深度学习：首先安装框架",  # 回答了错误的问题
            expected_output="深度学习是基于神经网络的机器学习方法",
            dimension_scores={"context": 0.2},
            error_types=["context_error"]
        )
        
        errors = self.classifier._analyze_context_errors(sample)
        
        assert len(errors) > 0
        assert all(error.error_type == "context_error" for error in errors)
    
    def test_analyze_format_errors(self):
        """测试格式错误分析"""
        sample = SampleResult(
            sample_id="test_005",
            input_text="请分段说明机器学习的步骤",
            model_output="机器学习包括数据收集模型训练模型评估这些步骤都很重要",  # 缺少标点和段落
            expected_output="机器学习包括以下步骤：\n\n1. 数据收集\n2. 模型训练\n3. 模型评估",
            dimension_scores={"format": 0.4},
            error_types=["format_error"]
        )
        
        errors = self.classifier._analyze_format_errors(sample)
        
        assert len(errors) > 0
        assert all(error.error_type == "format_error" for error in errors)
    
    def test_detect_factual_errors(self):
        """测试事实错误检测"""
        model_output = "机器学习在2025年被发明，准确率达到150%"
        expected_output = "机器学习在1950年代提出，准确率通常在90%以下"
        
        errors = self.classifier._detect_factual_errors(model_output, expected_output)
        
        assert len(errors) > 0
        # 应该检测到年份和百分比错误
        error_messages = [error["message"] for error in errors]
        assert any("年份" in msg for msg in error_messages)
        assert any("百分比" in msg for msg in error_messages)
    
    def test_detect_concept_confusion(self):
        """测试概念混淆检测"""
        model_output = "监督学习不需要标注数据"
        expected_output = "无监督学习不需要标注数据"
        
        errors = self.classifier._detect_concept_confusion(model_output, expected_output)
        
        assert len(errors) > 0
        assert any("混淆" in error["message"] for error in errors)
    
    def test_detect_terminology_inconsistency(self):
        """测试术语不一致检测"""
        text = "机器学习算法很重要，ML方法也很有效，机学技术发展迅速"
        
        errors = self.classifier._detect_terminology_inconsistency(text)
        
        assert len(errors) > 0
        assert any("不一致" in error["message"] for error in errors)
    
    def test_detect_logic_errors(self):
        """测试逻辑错误检测"""
        model_output = "深度学习不是机器学习方法，但深度学习是机器学习的重要方法"
        input_text = "解释深度学习"
        
        errors = self.classifier._detect_logic_errors(model_output, input_text)
        
        assert len(errors) > 0
        assert any("矛盾" in error["message"] for error in errors)


class TestErrorStatisticsAnalyzer:
    """错误统计分析器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.analyzer = ErrorStatisticsAnalyzer()
    
    def test_analyze_error_statistics(self):
        """测试错误统计分析"""
        error_instances = {
            "knowledge_error": [
                ErrorInstance("sample_001", "knowledge_error", "事实错误", ErrorSeverity.HIGH),
                ErrorInstance("sample_002", "knowledge_error", "概念混淆", ErrorSeverity.MEDIUM),
                ErrorInstance("sample_003", "knowledge_error", "事实错误", ErrorSeverity.HIGH)
            ],
            "terminology_error": [
                ErrorInstance("sample_001", "terminology_error", "术语误用", ErrorSeverity.MEDIUM),
                ErrorInstance("sample_004", "terminology_error", "术语不一致", ErrorSeverity.LOW)
            ]
        }
        
        stats = self.analyzer.analyze_error_statistics(error_instances)
        
        assert stats["total_errors"] == 5
        assert stats["error_type_distribution"]["knowledge_error"] == 3
        assert stats["error_type_distribution"]["terminology_error"] == 2
        assert "high" in stats["severity_distribution"]
        assert "medium" in stats["severity_distribution"]
        assert len(stats["error_frequency"]) > 0
        assert len(stats["top_error_patterns"]) > 0
    
    def test_identify_top_patterns(self):
        """测试识别顶级错误模式"""
        errors = [
            ErrorInstance("s1", "knowledge_error", "事实错误", ErrorSeverity.HIGH),
            ErrorInstance("s2", "knowledge_error", "事实错误", ErrorSeverity.HIGH),
            ErrorInstance("s3", "terminology_error", "术语误用", ErrorSeverity.MEDIUM),
            ErrorInstance("s4", "terminology_error", "术语误用", ErrorSeverity.MEDIUM),
            ErrorInstance("s5", "terminology_error", "术语误用", ErrorSeverity.MEDIUM)
        ]
        
        patterns = self.analyzer._identify_top_patterns(errors)
        
        assert len(patterns) > 0
        # 术语误用应该是最频繁的模式
        assert patterns[0]["error_type"] == "terminology_error"
        assert patterns[0]["frequency"] == 3
        assert patterns[0]["percentage"] == 60.0
    
    def test_analyze_error_trends(self):
        """测试错误趋势分析"""
        error_instances = {
            "knowledge_error": [
                ErrorInstance("s1", "knowledge_error", "错误1", ErrorSeverity.HIGH, confidence=0.9),
                ErrorInstance("s2", "knowledge_error", "错误2", ErrorSeverity.MEDIUM, confidence=0.7)
            ],
            "terminology_error": [
                ErrorInstance("s3", "terminology_error", "错误3", ErrorSeverity.LOW, confidence=0.8)
            ]
        }
        
        trends = self.analyzer._analyze_error_trends(error_instances)
        
        assert "knowledge_error" in trends
        assert "terminology_error" in trends
        assert trends["knowledge_error"]["total_count"] == 2
        assert trends["knowledge_error"]["avg_confidence"] == 0.8
        assert trends["terminology_error"]["total_count"] == 1
    
    def test_analyze_error_correlations(self):
        """测试错误相关性分析"""
        error_instances = {
            "knowledge_error": [
                ErrorInstance("sample_001", "knowledge_error", "错误1", ErrorSeverity.HIGH),
                ErrorInstance("sample_002", "knowledge_error", "错误2", ErrorSeverity.MEDIUM)
            ],
            "terminology_error": [
                ErrorInstance("sample_001", "terminology_error", "错误3", ErrorSeverity.MEDIUM),  # 与sample_001的知识错误共现
                ErrorInstance("sample_003", "terminology_error", "错误4", ErrorSeverity.LOW)
            ]
        }
        
        correlations = self.analyzer._analyze_error_correlations(error_instances)
        
        assert "knowledge_error__terminology_error" in correlations
        assert correlations["knowledge_error__terminology_error"]["cooccurrence_count"] == 1
        assert correlations["knowledge_error__terminology_error"]["correlation_strength"] > 0


class TestErrorAnalysisEngine:
    """错误分析引擎测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.engine = ErrorAnalysisEngine()
    
    def test_analyze_errors(self):
        """测试错误分析"""
        sample_results = [
            SampleResult(
                sample_id="sample_001",
                input_text="什么是机器学习？",
                model_output="机器学习是深度学习的一种",
                expected_output="机器学习是人工智能的分支",
                dimension_scores={"knowledge": 0.3},
                error_types=["knowledge_error"]
            ),
            SampleResult(
                sample_id="sample_002",
                input_text="解释神经网络",
                model_output="神经网络使用ML和机器学习",
                expected_output="神经网络是深度学习的基础",
                dimension_scores={"terminology": 0.4},
                error_types=["terminology_error"]
            ),
            SampleResult(
                sample_id="sample_003",
                input_text="正常样本",
                model_output="正确的输出",
                expected_output="正确的输出",
                dimension_scores={"overall": 0.9},
                error_types=[]
            )
        ]
        
        error_analysis = self.engine.analyze_errors(sample_results)
        
        assert isinstance(error_analysis, ErrorAnalysis)
        assert len(error_analysis.error_distribution) > 0
        assert len(error_analysis.common_patterns) > 0
        assert len(error_analysis.severity_levels) > 0
        assert len(error_analysis.improvement_areas) > 0
        
        # 检查错误分布
        assert "knowledge_error" in error_analysis.error_distribution
        assert "terminology_error" in error_analysis.error_distribution
        
        # 检查严重程度
        assert all(severity in ["critical", "high", "medium", "low"] 
                  for severity in error_analysis.severity_levels.values())
    
    def test_identify_improvement_areas(self):
        """测试识别改进领域"""
        error_stats = {
            "error_type_distribution": {
                "knowledge_error": 5,
                "terminology_error": 3,
                "reasoning_error": 2,
                "context_error": 1
            },
            "severity_distribution": {
                "critical": 2,
                "high": 4,
                "medium": 3,
                "low": 2
            }
        }
        
        classified_errors = {
            "knowledge_error": [ErrorInstance("s1", "knowledge_error", "错误", ErrorSeverity.HIGH)],
            "terminology_error": [ErrorInstance("s2", "terminology_error", "错误", ErrorSeverity.MEDIUM)]
        }
        
        improvement_areas = self.engine._identify_improvement_areas(error_stats, classified_errors)
        
        assert len(improvement_areas) > 0
        assert "关键错误修复" in improvement_areas  # 因为有critical错误
        assert "专业知识准确性" in improvement_areas  # 因为knowledge_error最多
        assert len(improvement_areas) <= 8
    
    def test_generate_error_report(self):
        """测试生成错误报告"""
        sample_results = [
            SampleResult("s1", "输入1", "输出1", "期望1", {"score": 0.5}, ["knowledge_error"]),
            SampleResult("s2", "输入2", "输出2", "期望2", {"score": 0.7}, ["terminology_error"]),
            SampleResult("s3", "输入3", "输出3", "期望3", {"score": 0.9}, [])
        ]
        
        error_analysis = ErrorAnalysis(
            error_distribution={"knowledge_error": 1, "terminology_error": 1},
            common_patterns=["模式1", "模式2"],
            severity_levels={"knowledge_error": "high", "terminology_error": "medium"},
            improvement_areas=["专业知识", "术语使用"]
        )
        
        report = self.engine.generate_error_report(error_analysis, sample_results)
        
        assert "summary" in report
        assert "error_distribution" in report
        assert "severity_analysis" in report
        assert "common_patterns" in report
        assert "improvement_recommendations" in report
        assert "detailed_breakdown" in report
        
        # 检查摘要信息
        summary = report["summary"]
        assert summary["total_samples"] == 3
        assert summary["samples_with_errors"] == 2
        assert summary["error_rate"] == 2/3
        assert summary["total_error_instances"] == 2
        
        # 检查详细分解
        breakdown = report["detailed_breakdown"]
        assert "knowledge_error" in breakdown
        assert "terminology_error" in breakdown
        assert breakdown["knowledge_error"]["count"] == 1
        assert breakdown["knowledge_error"]["percentage"] == 100/3
    
    def test_empty_sample_results(self):
        """测试空样本结果"""
        sample_results = []
        
        error_analysis = self.engine.analyze_errors(sample_results)
        
        assert error_analysis.error_distribution == {}
        assert error_analysis.common_patterns == []
        assert error_analysis.severity_levels == {}
        assert error_analysis.improvement_areas == []
    
    def test_no_errors_sample_results(self):
        """测试无错误样本结果"""
        sample_results = [
            SampleResult("s1", "输入1", "输出1", "期望1", {"score": 0.9}, []),
            SampleResult("s2", "输入2", "输出2", "期望2", {"score": 0.8}, [])
        ]
        
        error_analysis = self.engine.analyze_errors(sample_results)
        
        assert error_analysis.error_distribution == {}
        assert error_analysis.common_patterns == []
        assert error_analysis.severity_levels == {}
        assert len(error_analysis.improvement_areas) == 0


class TestErrorInstance:
    """错误实例测试"""
    
    def test_error_instance_creation(self):
        """测试错误实例创建"""
        error = ErrorInstance(
            sample_id="test_sample",
            error_type="knowledge_error",
            error_message="事实性错误",
            severity=ErrorSeverity.HIGH,
            location="第2段",
            context="上下文信息",
            suggested_fix="建议修复方案",
            confidence=0.9
        )
        
        assert error.sample_id == "test_sample"
        assert error.error_type == "knowledge_error"
        assert error.error_message == "事实性错误"
        assert error.severity == ErrorSeverity.HIGH
        assert error.location == "第2段"
        assert error.context == "上下文信息"
        assert error.suggested_fix == "建议修复方案"
        assert error.confidence == 0.9


class TestErrorPattern:
    """错误模式测试"""
    
    def test_error_pattern_creation(self):
        """测试错误模式创建"""
        pattern = ErrorPattern(
            pattern_id="pattern_001",
            pattern_type="factual_error",
            description="事实性错误模式",
            regex_pattern=r'\d{4}\s*年',
            keywords=["错误", "不正确"],
            severity=ErrorSeverity.HIGH,
            frequency=5,
            examples=["例子1", "例子2"]
        )
        
        assert pattern.pattern_id == "pattern_001"
        assert pattern.pattern_type == "factual_error"
        assert pattern.description == "事实性错误模式"
        assert pattern.regex_pattern == r'\d{4}\s*年'
        assert pattern.keywords == ["错误", "不正确"]
        assert pattern.severity == ErrorSeverity.HIGH
        assert pattern.frequency == 5
        assert pattern.examples == ["例子1", "例子2"]


if __name__ == "__main__":
    pytest.main([__file__])
"""
改进建议生成器单元测试
"""

import pytest
from datetime import datetime
from industry_evaluation.analysis.improvement_generator import (
    ImprovementSuggestionGenerator, ImprovementSuggestion, SuggestionPriority,
    SuggestionCategory, SuggestionTemplateManager
)
from industry_evaluation.models.data_models import (
    EvaluationResult, SampleResult, ErrorAnalysis, EvaluationConfig
)


class TestSuggestionTemplateManager:
    """建议模板管理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.template_manager = SuggestionTemplateManager()
    
    def test_get_template(self):
        """测试获取模板"""
        template = self.template_manager.get_template("knowledge_accuracy")
        
        assert template is not None
        assert template["title"] == "提升专业知识准确性"
        assert template["category"] == SuggestionCategory.KNOWLEDGE_BASE
        assert "base_description" in template
        assert "action_templates" in template
        assert "impact_range" in template
        assert "effort_range" in template
    
    def test_get_nonexistent_template(self):
        """测试获取不存在的模板"""
        template = self.template_manager.get_template("nonexistent_template")
        assert template is None
    
    def test_get_all_templates(self):
        """测试获取所有模板"""
        templates = self.template_manager.get_all_templates()
        
        assert len(templates) > 0
        assert "knowledge_accuracy" in templates
        assert "terminology_consistency" in templates
        assert "reasoning_logic" in templates
        
        # 确保返回的是副本
        templates["test"] = "test_value"
        original_templates = self.template_manager.get_all_templates()
        assert "test" not in original_templates


class TestImprovementSuggestion:
    """改进建议测试"""
    
    def test_suggestion_creation(self):
        """测试建议创建"""
        suggestion = ImprovementSuggestion(
            suggestion_id="test_001",
            title="测试建议",
            description="这是一个测试建议",
            category=SuggestionCategory.KNOWLEDGE_BASE,
            priority=SuggestionPriority.HIGH,
            impact_score=0.8,
            effort_score=0.6,
            evidence=["证据1", "证据2"],
            action_items=["行动1", "行动2"],
            expected_improvement="预期改进效果",
            resources_needed=["资源1", "资源2"],
            timeline="2-4周"
        )
        
        assert suggestion.suggestion_id == "test_001"
        assert suggestion.title == "测试建议"
        assert suggestion.category == SuggestionCategory.KNOWLEDGE_BASE
        assert suggestion.priority == SuggestionPriority.HIGH
        assert suggestion.impact_score == 0.8
        assert suggestion.effort_score == 0.6
        assert len(suggestion.evidence) == 2
        assert len(suggestion.action_items) == 2
    
    def test_get_roi_score(self):
        """测试ROI分数计算"""
        # 高影响低努力
        suggestion1 = ImprovementSuggestion(
            suggestion_id="test_001",
            title="高ROI建议",
            description="测试",
            category=SuggestionCategory.KNOWLEDGE_BASE,
            priority=SuggestionPriority.HIGH,
            impact_score=0.9,
            effort_score=0.3
        )
        
        assert suggestion1.get_roi_score() == 3.0  # 0.9 / 0.3
        
        # 零努力情况
        suggestion2 = ImprovementSuggestion(
            suggestion_id="test_002",
            title="零努力建议",
            description="测试",
            category=SuggestionCategory.KNOWLEDGE_BASE,
            priority=SuggestionPriority.HIGH,
            impact_score=0.8,
            effort_score=0.0
        )
        
        assert suggestion2.get_roi_score() == 0.8  # 直接返回影响分数


class TestImprovementSuggestionGenerator:
    """改进建议生成器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.generator = ImprovementSuggestionGenerator()
    
    def create_test_evaluation_result(self, overall_score=0.7, dimension_scores=None):
        """创建测试评估结果"""
        if dimension_scores is None:
            dimension_scores = {"knowledge": 0.6, "terminology": 0.8, "reasoning": 0.5}
        
        config = EvaluationConfig(
            industry_domain="测试",
            evaluation_dimensions=list(dimension_scores.keys()),
            weight_config={k: 1.0/len(dimension_scores) for k in dimension_scores.keys()},
            threshold_config={}
        )
        
        sample_results = [
            SampleResult(
                sample_id="sample_001",
                input_text="测试输入1",
                model_output="测试输出1",
                expected_output="期望输出1",
                dimension_scores=dimension_scores,
                error_types=["knowledge_error"] if dimension_scores.get("knowledge", 1.0) < 0.6 else [],
                processing_time=1.5
            ),
            SampleResult(
                sample_id="sample_002",
                input_text="测试输入2",
                model_output="测试输出2",
                expected_output="期望输出2",
                dimension_scores=dimension_scores,
                error_types=["reasoning_error"] if dimension_scores.get("reasoning", 1.0) < 0.6 else [],
                processing_time=2.0
            )
        ]
        
        return EvaluationResult(
            task_id="test_task",
            model_id="test_model",
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            detailed_results=sample_results,
            error_analysis=ErrorAnalysis({}, [], {}, []),
            improvement_suggestions=[],
            evaluation_config=config
        )
    
    def test_generate_suggestions_basic(self):
        """测试基本建议生成"""
        evaluation_result = self.create_test_evaluation_result(
            overall_score=0.5,
            dimension_scores={"knowledge": 0.4, "terminology": 0.7, "reasoning": 0.3}
        )
        
        suggestions = self.generator.generate_suggestions(evaluation_result)
        
        assert len(suggestions) > 0
        assert len(suggestions) <= 10  # 最多返回10个建议
        
        # 检查是否包含低分维度的建议
        suggestion_titles = [s.title for s in suggestions]
        assert any("知识" in title for title in suggestion_titles)  # knowledge维度分数低
        assert any("推理" in title for title in suggestion_titles)  # reasoning维度分数低
    
    def test_generate_score_based_suggestions(self):
        """测试基于分数的建议生成"""
        evaluation_result = self.create_test_evaluation_result(
            overall_score=0.4,
            dimension_scores={"knowledge": 0.3, "terminology": 0.8}
        )
        
        suggestions = self.generator._generate_score_based_suggestions(evaluation_result)
        
        assert len(suggestions) > 0
        
        # 应该包含knowledge维度的建议（分数0.3很低）
        knowledge_suggestions = [s for s in suggestions if "知识" in s.title]
        assert len(knowledge_suggestions) > 0
        
        # 应该包含整体改进建议（整体分数0.4很低）
        overall_suggestions = [s for s in suggestions if "整体" in s.title]
        assert len(overall_suggestions) > 0
    
    def test_generate_error_based_suggestions(self):
        """测试基于错误的建议生成"""
        error_analysis = ErrorAnalysis(
            error_distribution={"knowledge_error": 5, "terminology_error": 2},
            common_patterns=["模式1", "模式2"],
            severity_levels={"knowledge_error": "critical", "terminology_error": "medium"},
            improvement_areas=["专业知识", "术语使用"]
        )
        
        suggestions = self.generator._generate_error_based_suggestions(error_analysis)
        
        assert len(suggestions) > 0
        
        # 应该包含知识错误的建议
        knowledge_suggestions = [s for s in suggestions if "知识" in s.title]
        assert len(knowledge_suggestions) > 0
        
        # 应该包含关键错误的建议（因为有critical级别的错误）
        critical_suggestions = [s for s in suggestions if "关键" in s.title or "紧急" in s.title]
        assert len(critical_suggestions) > 0
    
    def test_generate_sample_based_suggestions(self):
        """测试基于样本的建议生成"""
        # 创建有质量问题的样本
        sample_results = [
            SampleResult(
                sample_id="sample_001",
                input_text="测试输入",
                model_output="短",  # 很短的输出
                expected_output="期望的长输出内容",
                dimension_scores={"overall": 0.3},  # 低分
                processing_time=8.0  # 处理时间长
            ),
            SampleResult(
                sample_id="sample_002",
                input_text="测试输入",
                model_output="另一个短输出",
                expected_output="期望输出",
                dimension_scores={"overall": 0.2},  # 低分
                processing_time=7.0
            )
        ]
        
        suggestions = self.generator._generate_sample_based_suggestions(sample_results)
        
        assert len(suggestions) > 0
        
        # 应该包含性能优化建议（处理时间长）
        performance_suggestions = [s for s in suggestions if "性能" in s.title or "处理" in s.title]
        assert len(performance_suggestions) > 0
        
        # 应该包含输出质量建议（输出过短）
        output_suggestions = [s for s in suggestions if "输出" in s.title or "完整" in s.title]
        assert len(output_suggestions) > 0
    
    def test_create_dimension_improvement_suggestion(self):
        """测试创建维度改进建议"""
        suggestion = self.generator._create_dimension_improvement_suggestion("knowledge", 0.3)
        
        assert suggestion is not None
        assert suggestion.priority == SuggestionPriority.CRITICAL  # 分数0.3很低
        assert "知识" in suggestion.title
        assert suggestion.category == SuggestionCategory.KNOWLEDGE_BASE
        assert suggestion.impact_score > 0.7  # 低分应该有高影响
        assert len(suggestion.action_items) > 0
        assert "0.3" in suggestion.description  # 应该包含当前分数
    
    def test_create_overall_improvement_suggestion(self):
        """测试创建整体改进建议"""
        evaluation_result = self.create_test_evaluation_result(overall_score=0.4)
        
        suggestion = self.generator._create_overall_improvement_suggestion(evaluation_result)
        
        assert suggestion.title == "提升整体评估性能"
        assert suggestion.priority == SuggestionPriority.HIGH  # 分数0.4较低
        assert suggestion.category == SuggestionCategory.MODEL_TRAINING
        assert "0.4" in suggestion.description
        assert len(suggestion.action_items) > 0
        assert len(suggestion.resources_needed) > 0
    
    def test_create_error_type_suggestion(self):
        """测试创建错误类型建议"""
        error_analysis = ErrorAnalysis(
            error_distribution={"knowledge_error": 3},
            common_patterns=[],
            severity_levels={"knowledge_error": "high"},
            improvement_areas=[]
        )
        
        suggestion = self.generator._create_error_type_suggestion(
            "knowledge_error", 3, error_analysis
        )
        
        assert suggestion is not None
        assert suggestion.priority == SuggestionPriority.HIGH
        assert "知识" in suggestion.title
        assert "3个" in suggestion.description
        assert "high" in suggestion.description
        assert len(suggestion.evidence) > 0
    
    def test_create_critical_error_suggestion(self):
        """测试创建关键错误建议"""
        critical_errors = ["knowledge_error", "reasoning_error"]
        
        suggestion = self.generator._create_critical_error_suggestion(critical_errors)
        
        assert suggestion.priority == SuggestionPriority.CRITICAL
        assert "关键错误" in suggestion.title or "紧急" in suggestion.title
        assert "knowledge_error" in suggestion.description
        assert "reasoning_error" in suggestion.description
        assert suggestion.impact_score >= 0.9
        assert "1-2周" in suggestion.timeline
    
    def test_analyze_sample_quality(self):
        """测试样本质量分析"""
        sample_results = [
            SampleResult("s1", "输入", "短", "期望输出", {}, processing_time=6.0),
            SampleResult("s2", "输入", "也很短", "期望输出", {}, processing_time=7.0),
            SampleResult("s3", "输入", "正常长度的输出内容", "期望输出", {}, processing_time=2.0)
        ]
        
        quality_issues = self.generator._analyze_sample_quality(sample_results)
        
        # 应该检测到处理时间慢的问题
        assert "slow_processing" in quality_issues
        assert quality_issues["slow_processing"]["avg_time"] > 5.0
        
        # 应该检测到输出过短的问题
        assert "short_outputs" in quality_issues
        assert quality_issues["short_outputs"]["short_count"] == 2  # 两个短输出
    
    def test_analyze_performance_patterns(self):
        """测试性能模式分析"""
        sample_results = [
            SampleResult("s1", "输入", "输出", "期望", {"score": 0.3}),  # 低分
            SampleResult("s2", "输入", "输出", "期望", {"score": 0.4}),  # 低分
            SampleResult("s3", "输入", "输出", "期望", {"score": 0.2}),  # 低分
            SampleResult("s4", "输入", "输出", "期望", {"score": 0.8})   # 高分
        ]
        
        patterns = self.generator._analyze_performance_patterns(sample_results)
        
        # 应该检测到低分模式（75%的样本分数低于0.5）
        assert "low_score_pattern" in patterns
        assert patterns["low_score_pattern"]["low_score_count"] == 3
        assert patterns["low_score_pattern"]["percentage"] == 75.0
    
    def test_deduplicate_suggestions(self):
        """测试建议去重"""
        suggestions = [
            ImprovementSuggestion(
                suggestion_id="s1",
                title="重复标题",
                description="描述1",
                category=SuggestionCategory.KNOWLEDGE_BASE,
                priority=SuggestionPriority.HIGH,
                impact_score=0.8,
                effort_score=0.5,
                evidence=["证据1"],
                action_items=["行动1"]
            ),
            ImprovementSuggestion(
                suggestion_id="s2",
                title="重复标题",
                description="描述2",
                category=SuggestionCategory.KNOWLEDGE_BASE,
                priority=SuggestionPriority.HIGH,
                impact_score=0.7,
                effort_score=0.4,
                evidence=["证据2"],
                action_items=["行动2"]
            ),
            ImprovementSuggestion(
                suggestion_id="s3",
                title="唯一标题",
                description="描述3",
                category=SuggestionCategory.TERMINOLOGY,
                priority=SuggestionPriority.MEDIUM,
                impact_score=0.6,
                effort_score=0.3
            )
        ]
        
        unique_suggestions = self.generator._deduplicate_suggestions(suggestions)
        
        assert len(unique_suggestions) == 2  # 去重后只有2个
        
        # 找到合并后的建议
        merged_suggestion = next(s for s in unique_suggestions if s.title == "重复标题")
        assert len(merged_suggestion.evidence) == 2  # 合并了证据
        assert len(merged_suggestion.action_items) == 2  # 合并了行动项
    
    def test_prioritize_suggestions(self):
        """测试建议优先级排序"""
        evaluation_result = self.create_test_evaluation_result(overall_score=0.5)
        
        suggestions = [
            ImprovementSuggestion(
                suggestion_id="s1",
                title="低优先级建议",
                description="描述",
                category=SuggestionCategory.OUTPUT_FORMAT,
                priority=SuggestionPriority.LOW,
                impact_score=0.3,
                effort_score=0.2
            ),
            ImprovementSuggestion(
                suggestion_id="s2",
                title="高优先级建议",
                description="描述",
                category=SuggestionCategory.KNOWLEDGE_BASE,
                priority=SuggestionPriority.CRITICAL,
                impact_score=0.9,
                effort_score=0.4
            ),
            ImprovementSuggestion(
                suggestion_id="s3",
                title="中等优先级建议",
                description="描述",
                category=SuggestionCategory.TERMINOLOGY,
                priority=SuggestionPriority.MEDIUM,
                impact_score=0.6,
                effort_score=0.3
            )
        ]
        
        prioritized = self.generator._prioritize_suggestions(suggestions, evaluation_result)
        
        # 高优先级建议应该排在前面
        assert prioritized[0].priority == SuggestionPriority.CRITICAL
        assert prioritized[-1].priority == SuggestionPriority.LOW
    
    def test_generate_action_plan(self):
        """测试生成行动计划"""
        suggestions = [
            ImprovementSuggestion(
                suggestion_id="s1",
                title="关键建议",
                description="描述",
                category=SuggestionCategory.KNOWLEDGE_BASE,
                priority=SuggestionPriority.CRITICAL,
                impact_score=0.9,
                effort_score=0.4,
                resources_needed=["专家", "数据"],
                timeline="1-2周"
            ),
            ImprovementSuggestion(
                suggestion_id="s2",
                title="高影响建议",
                description="描述",
                category=SuggestionCategory.MODEL_TRAINING,
                priority=SuggestionPriority.HIGH,
                impact_score=0.8,
                effort_score=0.3,
                resources_needed=["团队", "计算资源"],
                timeline="2-4周"
            )
        ]
        
        action_plan = self.generator.generate_action_plan(suggestions)
        
        assert "summary" in action_plan
        assert "priority_groups" in action_plan
        assert "category_groups" in action_plan
        assert "timeline" in action_plan
        assert "quick_wins" in action_plan
        assert "high_impact" in action_plan
        assert "resource_requirements" in action_plan
        
        # 检查摘要信息
        summary = action_plan["summary"]
        assert summary["total_suggestions"] == 2
        assert "CRITICAL" in summary["priority_distribution"]
        assert "HIGH" in summary["priority_distribution"]
        
        # 检查时间线
        timeline = action_plan["timeline"]
        assert "immediate" in timeline
        assert "short_term" in timeline
        
        # 检查快速胜利（高ROI）
        quick_wins = action_plan["quick_wins"]
        assert len(quick_wins) > 0  # 应该有高ROI的建议
        
        # 检查高影响建议
        high_impact = action_plan["high_impact"]
        assert len(high_impact) > 0  # 应该有高影响的建议
    
    def test_generate_suggestions_with_error_analysis(self):
        """测试带错误分析的建议生成"""
        evaluation_result = self.create_test_evaluation_result()
        
        error_analysis = ErrorAnalysis(
            error_distribution={"knowledge_error": 3, "terminology_error": 1},
            common_patterns=["知识错误模式"],
            severity_levels={"knowledge_error": "high", "terminology_error": "low"},
            improvement_areas=["专业知识准确性", "术语使用规范性"]
        )
        
        suggestions = self.generator.generate_suggestions(evaluation_result, error_analysis)
        
        assert len(suggestions) > 0
        
        # 应该包含基于错误分析的建议
        suggestion_descriptions = [s.description for s in suggestions]
        assert any("knowledge_error" in desc for desc in suggestion_descriptions)
    
    def test_empty_evaluation_result(self):
        """测试空评估结果"""
        config = EvaluationConfig(
            industry_domain="测试",
            evaluation_dimensions=[],
            weight_config={},
            threshold_config={}
        )
        
        evaluation_result = EvaluationResult(
            task_id="test",
            model_id="test",
            overall_score=1.0,  # 完美分数
            dimension_scores={},
            detailed_results=[],
            error_analysis=ErrorAnalysis({}, [], {}, []),
            improvement_suggestions=[],
            evaluation_config=config
        )
        
        suggestions = self.generator.generate_suggestions(evaluation_result)
        
        # 完美分数应该生成很少或没有建议
        assert len(suggestions) == 0 or all(s.priority.value <= 2 for s in suggestions)


if __name__ == "__main__":
    pytest.main([__file__])
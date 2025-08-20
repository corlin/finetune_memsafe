"""
报告生成器单元测试
"""

import pytest
import tempfile
import os
from datetime import datetime
from industry_evaluation.reporting.report_generator import (
    ReportGenerator, ReportTemplateManager, ChartGenerator,
    ReportType, ReportFormat, ReportTemplate
)
from industry_evaluation.models.data_models import (
    EvaluationResult, EvaluationConfig, SampleResult, ErrorAnalysis
)


class TestReportTemplateManager:
    """报告模板管理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.template_manager = ReportTemplateManager()
    
    def test_get_template(self):
        """测试获取模板"""
        template = self.template_manager.get_template("summary_report")
        
        assert template is not None
        assert template.template_id == "summary_report"
        assert template.name == "评估摘要报告"
        assert template.report_type == ReportType.SUMMARY
        assert len(template.template_content) > 0
        assert "evaluation_result" in template.required_data
    
    def test_get_nonexistent_template(self):
        """测试获取不存在的模板"""
        template = self.template_manager.get_template("nonexistent_template")
        assert template is None
    
    def test_get_templates_by_type(self):
        """测试根据类型获取模板"""
        summary_templates = self.template_manager.get_templates_by_type(ReportType.SUMMARY)
        assert len(summary_templates) > 0
        assert all(t.report_type == ReportType.SUMMARY for t in summary_templates)
        
        detailed_templates = self.template_manager.get_templates_by_type(ReportType.DETAILED)
        assert len(detailed_templates) > 0
        assert all(t.report_type == ReportType.DETAILED for t in detailed_templates)
    
    def test_add_template(self):
        """测试添加模板"""
        custom_template = ReportTemplate(
            template_id="custom_test",
            name="自定义测试模板",
            description="测试用的自定义模板",
            report_type=ReportType.SUMMARY,
            template_content="<html><body>{{model_id}}</body></html>",
            required_data=["evaluation_result"],
            supported_formats=[ReportFormat.HTML]
        )
        
        self.template_manager.add_template(custom_template)
        
        retrieved_template = self.template_manager.get_template("custom_test")
        assert retrieved_template is not None
        assert retrieved_template.name == "自定义测试模板"
    
    def test_remove_template(self):
        """测试移除模板"""
        # 先添加一个模板
        test_template = ReportTemplate(
            template_id="test_remove",
            name="待删除模板",
            description="测试删除",
            report_type=ReportType.SUMMARY,
            template_content="test",
            required_data=[]
        )
        
        self.template_manager.add_template(test_template)
        assert self.template_manager.get_template("test_remove") is not None
        
        # 删除模板
        success = self.template_manager.remove_template("test_remove")
        assert success is True
        assert self.template_manager.get_template("test_remove") is None
        
        # 删除不存在的模板
        success = self.template_manager.remove_template("nonexistent")
        assert success is False


class TestChartGenerator:
    """图表生成器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.chart_generator = ChartGenerator()
    
    def test_generate_score_distribution_chart(self):
        """测试生成分数分布图"""
        scores = {
            "知识准确性": 0.85,
            "术语使用": 0.78,
            "逻辑推理": 0.82,
            "上下文理解": 0.75
        }
        
        chart_data = self.chart_generator.generate_score_distribution_chart(
            scores, "测试分数分布"
        )
        
        assert isinstance(chart_data, str)
        assert len(chart_data) > 0
        # Base64编码的图片数据应该是有效的
        import base64
        try:
            base64.b64decode(chart_data)
            assert True
        except Exception:
            assert False, "生成的图表数据不是有效的Base64编码"
    
    def test_generate_comparison_chart(self):
        """测试生成对比图表"""
        comparison_data = {
            "模型A": {"知识": 0.8, "术语": 0.7, "推理": 0.9},
            "模型B": {"知识": 0.7, "术语": 0.8, "推理": 0.6},
            "模型C": {"知识": 0.9, "术语": 0.6, "推理": 0.8}
        }
        
        chart_data = self.chart_generator.generate_comparison_chart(
            comparison_data, "模型对比测试"
        )
        
        assert isinstance(chart_data, str)
        assert len(chart_data) > 0
    
    def test_generate_trend_chart(self):
        """测试生成趋势图表"""
        trend_data = {
            "总分": [0.6, 0.65, 0.7, 0.75, 0.8],
            "知识": [0.5, 0.6, 0.7, 0.8, 0.85]
        }
        labels = ["v1.0", "v1.1", "v1.2", "v1.3", "v1.4"]
        
        chart_data = self.chart_generator.generate_trend_chart(
            trend_data, labels, "性能趋势"
        )
        
        assert isinstance(chart_data, str)
        assert len(chart_data) > 0
    
    def test_generate_error_distribution_pie(self):
        """测试生成错误分布饼图"""
        error_data = {
            "知识错误": 15,
            "术语错误": 8,
            "推理错误": 12,
            "格式错误": 5
        }
        
        chart_data = self.chart_generator.generate_error_distribution_pie(
            error_data, "错误分布测试"
        )
        
        assert isinstance(chart_data, str)
        assert len(chart_data) > 0
    
    def test_generate_heatmap(self):
        """测试生成热力图"""
        data = [
            [0.8, 0.6, 0.7],
            [0.9, 0.5, 0.8],
            [0.7, 0.8, 0.6]
        ]
        row_labels = ["模型A", "模型B", "模型C"]
        col_labels = ["知识", "术语", "推理"]
        
        chart_data = self.chart_generator.generate_heatmap(
            data, row_labels, col_labels, "相关性测试"
        )
        
        assert isinstance(chart_data, str)
        assert len(chart_data) > 0


class TestReportGenerator:
    """报告生成器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.report_generator = ReportGenerator()
        
        # 创建测试用的评估结果
        config = EvaluationConfig(
            industry_domain="测试领域",
            evaluation_dimensions=["knowledge", "terminology", "reasoning"],
            weight_config={"knowledge": 0.4, "terminology": 0.3, "reasoning": 0.3},
            threshold_config={}
        )
        
        sample_results = [
            SampleResult(
                sample_id="sample_001",
                input_text="什么是机器学习？",
                model_output="机器学习是人工智能的一个分支",
                expected_output="机器学习是AI的重要技术",
                dimension_scores={"knowledge": 0.8, "terminology": 0.7, "reasoning": 0.9}
            ),
            SampleResult(
                sample_id="sample_002",
                input_text="解释深度学习",
                model_output="深度学习使用神经网络",
                expected_output="深度学习是机器学习的子领域",
                dimension_scores={"knowledge": 0.7, "terminology": 0.8, "reasoning": 0.6}
            )
        ]
        
        self.evaluation_result = EvaluationResult(
            task_id="test_task",
            model_id="test_model",
            overall_score=0.75,
            dimension_scores={"knowledge": 0.75, "terminology": 0.75, "reasoning": 0.75},
            detailed_results=sample_results,
            error_analysis=ErrorAnalysis({}, [], {}, []),
            improvement_suggestions=["提升专业知识准确性", "改善术语使用一致性"],
            evaluation_config=config
        )
    
    def test_generate_summary_report(self):
        """测试生成摘要报告"""
        report = self.report_generator.generate_report(
            self.evaluation_result,
            report_type=ReportType.SUMMARY,
            report_format=ReportFormat.HTML
        )
        
        assert report.title.startswith("评估摘要报告")
        assert report.format_type == "html"
        assert len(report.content) > 0
        assert "test_model" in report.content
        assert "0.750" in report.content  # 总分应该在内容中
        assert len(report.charts) > 0  # 应该有图表
    
    def test_generate_detailed_report(self):
        """测试生成详细报告"""
        report = self.report_generator.generate_report(
            self.evaluation_result,
            report_type=ReportType.DETAILED,
            report_format=ReportFormat.HTML
        )
        
        assert "详细评估报告" in report.title
        assert report.format_type == "html"
        assert len(report.content) > 0
        assert "sample_001" in report.content or "sample_002" in report.content  # 应该包含样本信息
    
    def test_generate_comparative_report(self):
        """测试生成对比报告"""
        # 创建对比数据
        additional_data = {
            "model_count": 2,
            "model_comparison": [
                {
                    "model_id": "model_A",
                    "overall_score": "0.750",
                    "overall_class": "good",
                    "dimension_scores": [
                        {"score": "0.8", "class": "good"},
                        {"score": "0.7", "class": "medium"}
                    ],
                    "rank": 1
                },
                {
                    "model_id": "model_B", 
                    "overall_score": "0.650",
                    "overall_class": "medium",
                    "dimension_scores": [
                        {"score": "0.6", "class": "medium"},
                        {"score": "0.7", "class": "medium"}
                    ],
                    "rank": 2
                }
            ],
            "dimensions": ["knowledge", "terminology"]
        }
        
        report = self.report_generator.generate_report(
            self.evaluation_result,
            report_type=ReportType.COMPARATIVE,
            report_format=ReportFormat.HTML,
            additional_data=additional_data
        )
        
        assert "对比" in report.title
        assert "model_A" in report.content
        assert "model_B" in report.content
    
    def test_generate_error_analysis_report(self):
        """测试生成错误分析报告"""
        # 创建错误分析数据
        additional_data = {
            "total_errors": 25,
            "error_rate": 12.5,
            "affected_samples": 15,
            "error_types": [
                {
                    "type_name": "知识错误",
                    "count": 10,
                    "severity": "高",
                    "description": "专业知识不准确"
                },
                {
                    "type_name": "术语错误",
                    "count": 8,
                    "severity": "中",
                    "description": "术语使用不当"
                }
            ],
            "error_patterns": [
                {
                    "pattern_name": "概念混淆",
                    "frequency": 15,
                    "description": "经常混淆相关概念"
                }
            ],
            "recommendations": [
                "加强专业知识训练",
                "建立术语词典"
            ]
        }
        
        report = self.report_generator.generate_report(
            self.evaluation_result,
            report_type=ReportType.ERROR_ANALYSIS,
            report_format=ReportFormat.HTML,
            additional_data=additional_data
        )
        
        assert "错误分析" in report.title
        assert "25" in report.content  # 总错误数
        assert "知识错误" in report.content
    
    def test_generate_improvement_plan_report(self):
        """测试生成改进计划报告"""
        additional_data = {
            "total_suggestions": 5,
            "high_priority_count": 2,
            "expected_improvement": "15-25%",
            "suggestions": [
                {
                    "title": "提升专业知识准确性",
                    "priority": "高",
                    "priority_class": "high",
                    "impact_score": "0.8",
                    "description": "加强专业知识训练",
                    "action_items": ["补充训练数据", "专家审核"],
                    "timeline": "4-6周"
                }
            ],
            "timeline": [
                {
                    "phase": "第一阶段 (1-2周)",
                    "tasks": ["数据收集", "专家咨询"]
                },
                {
                    "phase": "第二阶段 (3-4周)",
                    "tasks": ["模型训练", "初步测试"]
                }
            ]
        }
        
        report = self.report_generator.generate_report(
            self.evaluation_result,
            report_type=ReportType.IMPROVEMENT_PLAN,
            report_format=ReportFormat.HTML,
            additional_data=additional_data
        )
        
        assert "改进计划" in report.title
        assert "专业知识准确性" in report.content
        assert "第一阶段" in report.content
    
    def test_export_report_html(self):
        """测试导出HTML报告"""
        report = self.report_generator.generate_report(
            self.evaluation_result,
            report_type=ReportType.SUMMARY
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            self.report_generator.export_report(report, temp_path, ReportFormat.HTML)
            
            # 验证文件存在且有内容
            assert os.path.exists(temp_path)
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 0
                assert "test_model" in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_report_json(self):
        """测试导出JSON报告"""
        report = self.report_generator.generate_report(
            self.evaluation_result,
            report_type=ReportType.SUMMARY
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.report_generator.export_report(report, temp_path, ReportFormat.JSON)
            
            # 验证文件存在且是有效JSON
            assert os.path.exists(temp_path)
            import json
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert "title" in data
                assert "evaluation_result" in data
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_report_markdown(self):
        """测试导出Markdown报告"""
        report = self.report_generator.generate_report(
            self.evaluation_result,
            report_type=ReportType.SUMMARY
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_path = f.name
        
        try:
            self.report_generator.export_report(report, temp_path, ReportFormat.MARKDOWN)
            
            # 验证文件存在且有内容
            assert os.path.exists(temp_path)
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 0
                # Markdown应该包含标题标记
                assert "#" in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_performance_level(self):
        """测试性能等级判断"""
        assert self.report_generator._get_performance_level(0.95) == "优秀"
        assert self.report_generator._get_performance_level(0.85) == "良好"
        assert self.report_generator._get_performance_level(0.65) == "及格"
        assert self.report_generator._get_performance_level(0.45) == "较差"
        assert self.report_generator._get_performance_level(0.25) == "很差"
    
    def test_prepare_report_data(self):
        """测试准备报告数据"""
        additional_data = {"custom_field": "custom_value"}
        
        data = self.report_generator._prepare_report_data(
            self.evaluation_result, additional_data
        )
        
        assert data["model_id"] == "test_model"
        assert data["task_id"] == "test_task"
        assert data["overall_score"] == "0.750"
        assert data["sample_count"] == 2
        assert len(data["dimension_scores"]) == 3
        assert len(data["improvement_suggestions"]) == 2
        assert data["custom_field"] == "custom_value"
        assert "best_samples" in data
        assert "worst_samples" in data
    
    def test_invalid_template_id(self):
        """测试无效模板ID"""
        with pytest.raises(ValueError, match="未找到适合的模板"):
            self.report_generator.generate_report(
                self.evaluation_result,
                template_id="nonexistent_template"
            )
    
    def test_unsupported_export_format(self):
        """测试不支持的导出格式"""
        report = self.report_generator.generate_report(self.evaluation_result)
        
        with pytest.raises(ValueError, match="不支持的导出格式"):
            # 创建一个不支持的格式枚举值（这里用PDF作为示例，因为我们没有实现PDF导出）
            self.report_generator.export_report(report, "test.pdf", ReportFormat.PDF)


if __name__ == "__main__":
    pytest.main([__file__])
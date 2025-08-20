"""
评估报告生成器
"""

import json
import base64
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
import pandas as pd
from io import BytesIO
from industry_evaluation.models.data_models import EvaluationResult, Report


class ReportType(Enum):
    """报告类型"""
    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPARATIVE = "comparative"
    TREND_ANALYSIS = "trend_analysis"
    ERROR_ANALYSIS = "error_analysis"
    IMPROVEMENT_PLAN = "improvement_plan"


class ReportFormat(Enum):
    """报告格式"""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class ReportTemplate:
    """报告模板"""
    template_id: str
    name: str
    description: str
    report_type: ReportType
    template_content: str
    required_data: List[str] = field(default_factory=list)
    optional_data: List[str] = field(default_factory=list)
    supported_formats: List[ReportFormat] = field(default_factory=list)
    created_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "report_type": self.report_type.value,
            "template_content": self.template_content,
            "required_data": self.required_data,
            "optional_data": self.optional_data,
            "supported_formats": [f.value for f in self.supported_formats],
            "created_time": self.created_time.isoformat()
        }


class ReportTemplateManager:
    """报告模板管理器"""
    
    def __init__(self):
        """初始化模板管理器"""
        self.templates = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """初始化默认模板"""
        # 摘要报告模板
        summary_template = ReportTemplate(
            template_id="summary_report",
            name="评估摘要报告",
            description="提供评估结果的简要摘要",
            report_type=ReportType.SUMMARY,
            template_content=self._get_summary_template(),
            required_data=["evaluation_result"],
            optional_data=["comparison_data", "historical_data"],
            supported_formats=[ReportFormat.HTML, ReportFormat.PDF, ReportFormat.MARKDOWN]
        )
        
        # 详细报告模板
        detailed_template = ReportTemplate(
            template_id="detailed_report",
            name="详细评估报告",
            description="提供完整的评估结果和分析",
            report_type=ReportType.DETAILED,
            template_content=self._get_detailed_template(),
            required_data=["evaluation_result", "sample_results"],
            optional_data=["error_analysis", "improvement_suggestions"],
            supported_formats=[ReportFormat.HTML, ReportFormat.PDF]
        )
        
        # 对比报告模板
        comparative_template = ReportTemplate(
            template_id="comparative_report",
            name="对比评估报告",
            description="对比多个模型的评估结果",
            report_type=ReportType.COMPARATIVE,
            template_content=self._get_comparative_template(),
            required_data=["evaluation_results"],
            optional_data=["baseline_results"],
            supported_formats=[ReportFormat.HTML, ReportFormat.PDF]
        )
        
        # 错误分析报告模板
        error_analysis_template = ReportTemplate(
            template_id="error_analysis_report",
            name="错误分析报告",
            description="详细分析评估中发现的错误",
            report_type=ReportType.ERROR_ANALYSIS,
            template_content=self._get_error_analysis_template(),
            required_data=["error_analysis", "sample_results"],
            optional_data=["expert_annotations"],
            supported_formats=[ReportFormat.HTML, ReportFormat.MARKDOWN]
        )
        
        # 改进计划报告模板
        improvement_plan_template = ReportTemplate(
            template_id="improvement_plan_report",
            name="改进计划报告",
            description="基于评估结果生成改进建议和计划",
            report_type=ReportType.IMPROVEMENT_PLAN,
            template_content=self._get_improvement_plan_template(),
            required_data=["improvement_suggestions", "evaluation_result"],
            optional_data=["action_plan", "resource_requirements"],
            supported_formats=[ReportFormat.HTML, ReportFormat.MARKDOWN]
        )
        
        # 注册模板
        for template in [summary_template, detailed_template, comparative_template, 
                        error_analysis_template, improvement_plan_template]:
            self.templates[template.template_id] = template
    
    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """获取模板"""
        return self.templates.get(template_id)
    
    def get_templates_by_type(self, report_type: ReportType) -> List[ReportTemplate]:
        """根据类型获取模板"""
        return [template for template in self.templates.values() 
                if template.report_type == report_type]
    
    def add_template(self, template: ReportTemplate):
        """添加模板"""
        self.templates[template.template_id] = template
    
    def remove_template(self, template_id: str) -> bool:
        """移除模板"""
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False
    
    def _get_summary_template(self) -> str:
        """获取摘要报告模板"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>评估摘要报告</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; margin-bottom: 30px; }
        .summary-box { background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .score-display { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .dimension-scores { display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }
        .dimension-item { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .chart-container { text-align: center; margin: 30px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>模型评估摘要报告</h1>
        <p>模型ID: {{model_id}} | 任务ID: {{task_id}}</p>
        <p>生成时间: {{generated_time}}</p>
    </div>
    
    <div class="summary-box">
        <h2>总体评估结果</h2>
        <div class="score-display">总分: {{overall_score}}</div>
        <p>评估状态: {{status}}</p>
        <p>样本数量: {{sample_count}}</p>
        <p>评估耗时: {{duration}} 秒</p>
    </div>
    
    <div class="dimension-scores">
        <h3>各维度得分</h3>
        {{#dimension_scores}}
        <div class="dimension-item">
            <h4>{{dimension}}</h4>
            <div class="score-display">{{score}}</div>
        </div>
        {{/dimension_scores}}
    </div>
    
    <div class="chart-container">
        {{#charts}}
        <img src="data:image/png;base64,{{chart_data}}" alt="{{chart_title}}" />
        <p>{{chart_title}}</p>
        {{/charts}}
    </div>
    
    {{#improvement_suggestions}}
    <div>
        <h3>主要改进建议</h3>
        <ul>
        {{#suggestions}}
        <li>{{.}}</li>
        {{/suggestions}}
        </ul>
    </div>
    {{/improvement_suggestions}}
</body>
</html>
        """
    
    def _get_detailed_template(self) -> str:
        """获取详细报告模板"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>详细评估报告</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { text-align: center; margin-bottom: 40px; border-bottom: 2px solid #3498db; padding-bottom: 20px; }
        .section { margin: 30px 0; }
        .section h2 { color: #2c3e50; border-left: 4px solid #3498db; padding-left: 15px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .score-large { font-size: 36px; font-weight: bold; color: #27ae60; }
        .score-medium { font-size: 24px; font-weight: bold; color: #3498db; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #ecf0f1; font-weight: bold; }
        .sample-details { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .error-highlight { background: #ffebee; border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; }
        .success-highlight { background: #e8f5e8; border-left: 4px solid #27ae60; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>详细评估报告</h1>
        <p><strong>模型:</strong> {{model_id}} | <strong>任务:</strong> {{task_id}}</p>
        <p><strong>评估时间:</strong> {{start_time}} - {{end_time}}</p>
        <p><strong>报告生成时间:</strong> {{generated_time}}</p>
    </div>
    
    <div class="section">
        <h2>执行摘要</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>总体得分</h3>
                <div class="score-large">{{overall_score}}</div>
                <p>评估等级: {{performance_level}}</p>
            </div>
            <div class="metric-card">
                <h3>样本统计</h3>
                <p><strong>总样本数:</strong> {{sample_count}}</p>
                <p><strong>通过率:</strong> {{pass_rate}}%</p>
                <p><strong>平均处理时间:</strong> {{avg_processing_time}}ms</p>
            </div>
            <div class="metric-card">
                <h3>质量指标</h3>
                <p><strong>准确性:</strong> {{accuracy_score}}</p>
                <p><strong>一致性:</strong> {{consistency_score}}</p>
                <p><strong>完整性:</strong> {{completeness_score}}</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>维度分析</h2>
        <table>
            <thead>
                <tr>
                    <th>评估维度</th>
                    <th>得分</th>
                    <th>权重</th>
                    <th>贡献度</th>
                    <th>评估等级</th>
                </tr>
            </thead>
            <tbody>
            {{#dimension_analysis}}
                <tr>
                    <td>{{dimension}}</td>
                    <td><span class="score-medium">{{score}}</span></td>
                    <td>{{weight}}%</td>
                    <td>{{contribution}}%</td>
                    <td>{{level}}</td>
                </tr>
            {{/dimension_analysis}}
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>样本分析</h2>
        <h3>表现最佳样本</h3>
        {{#best_samples}}
        <div class="success-highlight">
            <p><strong>样本ID:</strong> {{sample_id}} | <strong>得分:</strong> {{score}}</p>
            <p><strong>输入:</strong> {{input_text}}</p>
            <p><strong>输出:</strong> {{model_output}}</p>
        </div>
        {{/best_samples}}
        
        <h3>需要改进样本</h3>
        {{#worst_samples}}
        <div class="error-highlight">
            <p><strong>样本ID:</strong> {{sample_id}} | <strong>得分:</strong> {{score}}</p>
            <p><strong>输入:</strong> {{input_text}}</p>
            <p><strong>输出:</strong> {{model_output}}</p>
            <p><strong>问题:</strong> {{issues}}</p>
        </div>
        {{/worst_samples}}
    </div>
    
    <div class="section">
        <h2>图表分析</h2>
        {{#charts}}
        <div style="text-align: center; margin: 30px 0;">
            <img src="data:image/png;base64,{{chart_data}}" alt="{{chart_title}}" style="max-width: 100%;" />
            <h4>{{chart_title}}</h4>
            <p>{{chart_description}}</p>
        </div>
        {{/charts}}
    </div>
    
    <div class="section">
        <h2>改进建议</h2>
        {{#improvement_suggestions}}
        <div class="metric-card">
            <h4>{{title}}</h4>
            <p>{{description}}</p>
            <p><strong>优先级:</strong> {{priority}} | <strong>预期影响:</strong> {{impact}}</p>
            <ul>
            {{#action_items}}
            <li>{{.}}</li>
            {{/action_items}}
            </ul>
        </div>
        {{/improvement_suggestions}}
    </div>
</body>
</html>
        """
    
    def _get_comparative_template(self) -> str:
        """获取对比报告模板"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>模型对比评估报告</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; margin-bottom: 30px; }
        .comparison-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .comparison-table th, .comparison-table td { padding: 12px; text-align: center; border: 1px solid #ddd; }
        .comparison-table th { background-color: #f2f2f2; }
        .best-score { background-color: #d4edda; font-weight: bold; }
        .worst-score { background-color: #f8d7da; }
        .chart-section { margin: 30px 0; text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <h1>模型对比评估报告</h1>
        <p>对比模型数量: {{model_count}} | 生成时间: {{generated_time}}</p>
    </div>
    
    <h2>总体对比</h2>
    <table class="comparison-table">
        <thead>
            <tr>
                <th>模型</th>
                <th>总分</th>
                {{#dimensions}}
                <th>{{.}}</th>
                {{/dimensions}}
                <th>排名</th>
            </tr>
        </thead>
        <tbody>
        {{#model_comparison}}
            <tr>
                <td>{{model_id}}</td>
                <td class="{{overall_class}}">{{overall_score}}</td>
                {{#dimension_scores}}
                <td class="{{class}}">{{score}}</td>
                {{/dimension_scores}}
                <td>{{rank}}</td>
            </tr>
        {{/model_comparison}}
        </tbody>
    </table>
    
    <div class="chart-section">
        {{#charts}}
        <img src="data:image/png;base64,{{chart_data}}" alt="{{chart_title}}" />
        <h3>{{chart_title}}</h3>
        {{/charts}}
    </div>
    
    <h2>详细分析</h2>
    {{#detailed_analysis}}
    <h3>{{dimension}}</h3>
    <p>{{analysis}}</p>
    {{/detailed_analysis}}
</body>
</html>
        """
    
    def _get_error_analysis_template(self) -> str:
        """获取错误分析报告模板"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>错误分析报告</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .error-summary { background: #fff3cd; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .error-type { margin: 20px 0; padding: 15px; border-left: 4px solid #dc3545; background: #f8f9fa; }
        .pattern-box { background: #e7f3ff; padding: 15px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>错误分析报告</h1>
    <p>生成时间: {{generated_time}}</p>
    
    <div class="error-summary">
        <h2>错误概览</h2>
        <p>总错误数: {{total_errors}}</p>
        <p>错误率: {{error_rate}}%</p>
        <p>影响样本数: {{affected_samples}}</p>
    </div>
    
    <h2>错误类型分布</h2>
    {{#error_types}}
    <div class="error-type">
        <h3>{{type_name}} ({{count}}个)</h3>
        <p>严重程度: {{severity}}</p>
        <p>{{description}}</p>
    </div>
    {{/error_types}}
    
    <h2>常见错误模式</h2>
    {{#error_patterns}}
    <div class="pattern-box">
        <h4>{{pattern_name}}</h4>
        <p>出现频率: {{frequency}}</p>
        <p>{{description}}</p>
    </div>
    {{/error_patterns}}
    
    <h2>改进建议</h2>
    <ul>
    {{#recommendations}}
    <li>{{.}}</li>
    {{/recommendations}}
    </ul>
</body>
</html>
        """
    
    def _get_improvement_plan_template(self) -> str:
        """获取改进计划报告模板"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>改进计划报告</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .plan-summary { background: #e8f5e8; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .suggestion-card { background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .priority-high { border-left: 4px solid #dc3545; }
        .priority-medium { border-left: 4px solid #ffc107; }
        .priority-low { border-left: 4px solid #28a745; }
        .timeline { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>模型改进计划报告</h1>
    <p>基于评估结果生成 | 生成时间: {{generated_time}}</p>
    
    <div class="plan-summary">
        <h2>改进计划概览</h2>
        <p>总建议数: {{total_suggestions}}</p>
        <p>高优先级: {{high_priority_count}}</p>
        <p>预期总体提升: {{expected_improvement}}</p>
    </div>
    
    <h2>改进建议</h2>
    {{#suggestions}}
    <div class="suggestion-card priority-{{priority_class}}">
        <h3>{{title}}</h3>
        <p><strong>优先级:</strong> {{priority}} | <strong>预期影响:</strong> {{impact_score}}</p>
        <p>{{description}}</p>
        
        <h4>行动项目</h4>
        <ul>
        {{#action_items}}
        <li>{{.}}</li>
        {{/action_items}}
        </ul>
        
        <div class="timeline">
            <strong>时间线:</strong> {{timeline}}
        </div>
    </div>
    {{/suggestions}}
    
    <h2>实施时间表</h2>
    {{#timeline}}
    <h3>{{phase}}</h3>
    <ul>
    {{#tasks}}
    <li>{{.}}</li>
    {{/tasks}}
    </ul>
    {{/timeline}}
</body>
</html>
        """


class ChartGenerator:
    """图表生成器"""
    
    def __init__(self):
        """初始化图表生成器"""
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
    
    def generate_score_distribution_chart(self, scores: Dict[str, float], 
                                        title: str = "评估分数分布") -> str:
        """
        生成分数分布图
        
        Args:
            scores: 分数字典
            title: 图表标题
            
        Returns:
            str: Base64编码的图片数据
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dimensions = list(scores.keys())
        values = list(scores.values())
        
        bars = ax.bar(dimensions, values, color='skyblue', alpha=0.7)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('分数', fontsize=12)
        ax.set_ylim(0, 1)
        
        # 在柱子上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def generate_comparison_chart(self, comparison_data: Dict[str, Dict[str, float]], 
                                title: str = "模型对比") -> str:
        """
        生成对比图表
        
        Args:
            comparison_data: 对比数据 {model_id: {dimension: score}}
            title: 图表标题
            
        Returns:
            str: Base64编码的图片数据
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 准备数据
        models = list(comparison_data.keys())
        dimensions = list(next(iter(comparison_data.values())).keys())
        
        x = range(len(dimensions))
        width = 0.8 / len(models)
        
        colors = plt.cm.Set3(range(len(models)))
        
        for i, model in enumerate(models):
            scores = [comparison_data[model][dim] for dim in dimensions]
            ax.bar([xi + i * width for xi in x], scores, width, 
                  label=model, color=colors[i], alpha=0.8)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('分数', fontsize=12)
        ax.set_xlabel('评估维度', fontsize=12)
        ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
        ax.set_xticklabels(dimensions, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def generate_trend_chart(self, trend_data: Dict[str, List[float]], 
                           labels: List[str], title: str = "趋势分析") -> str:
        """
        生成趋势图表
        
        Args:
            trend_data: 趋势数据 {series_name: [values]}
            labels: X轴标签
            title: 图表标题
            
        Returns:
            str: Base64编码的图片数据
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for series_name, values in trend_data.items():
            ax.plot(labels, values, marker='o', linewidth=2, label=series_name)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('分数', fontsize=12)
        ax.set_xlabel('时间/版本', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def generate_error_distribution_pie(self, error_data: Dict[str, int], 
                                      title: str = "错误类型分布") -> str:
        """
        生成错误分布饼图
        
        Args:
            error_data: 错误数据 {error_type: count}
            title: 图表标题
            
        Returns:
            str: Base64编码的图片数据
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = list(error_data.keys())
        sizes = list(error_data.values())
        colors = plt.cm.Set3(range(len(labels)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def generate_heatmap(self, data: List[List[float]], 
                        row_labels: List[str], col_labels: List[str],
                        title: str = "相关性热力图") -> str:
        """
        生成热力图
        
        Args:
            data: 二维数据
            row_labels: 行标签
            col_labels: 列标签
            title: 图表标题
            
        Returns:
            str: Base64编码的图片数据
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto')
        
        # 设置标签
        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值标注
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{data[i][j]:.2f}', 
                             ha="center", va="center", color="black")
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('数值', rotation=270, labelpad=15)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """将matplotlib图形转换为base64字符串"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close(fig)
        return image_base64


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        """初始化报告生成器"""
        self.template_manager = ReportTemplateManager()
        self.chart_generator = ChartGenerator()
    
    def generate_report(self, evaluation_result: EvaluationResult, 
                       report_type: ReportType = ReportType.SUMMARY,
                       report_format: ReportFormat = ReportFormat.HTML,
                       template_id: Optional[str] = None,
                       additional_data: Optional[Dict[str, Any]] = None) -> Report:
        """
        生成评估报告
        
        Args:
            evaluation_result: 评估结果
            report_type: 报告类型
            report_format: 报告格式
            template_id: 模板ID
            additional_data: 额外数据
            
        Returns:
            Report: 生成的报告
        """
        # 选择模板
        if template_id:
            template = self.template_manager.get_template(template_id)
        else:
            templates = self.template_manager.get_templates_by_type(report_type)
            template = templates[0] if templates else None
        
        if not template:
            raise ValueError(f"未找到适合的模板: {report_type}")
        
        # 准备数据
        report_data = self._prepare_report_data(evaluation_result, additional_data or {})
        
        # 生成图表
        charts = self._generate_charts(evaluation_result, report_type)
        report_data['charts'] = charts
        
        # 渲染报告
        content = self._render_template(template.template_content, report_data)
        
        # 创建报告对象
        report = Report(
            title=f"{template.name} - {evaluation_result.model_id}",
            evaluation_result=evaluation_result,
            format_type=report_format.value,
            content=content,
            charts=[{"title": chart["title"], "data": chart["data"]} for chart in charts]
        )
        
        return report
    
    def _prepare_report_data(self, evaluation_result: EvaluationResult, 
                           additional_data: Dict[str, Any]) -> Dict[str, Any]:
        """准备报告数据"""
        data = {
            # 基本信息
            "model_id": evaluation_result.model_id,
            "task_id": evaluation_result.task_id,
            "overall_score": f"{evaluation_result.overall_score:.3f}",
            "generated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start_time": evaluation_result.start_time.strftime("%Y-%m-%d %H:%M:%S") if evaluation_result.start_time else "",
            "end_time": evaluation_result.end_time.strftime("%Y-%m-%d %H:%M:%S") if evaluation_result.end_time else "",
            "duration": f"{evaluation_result.get_duration():.2f}",
            "sample_count": evaluation_result.get_sample_count(),
            "pass_rate": f"{evaluation_result.get_pass_rate() * 100:.1f}",
            "status": evaluation_result.status.value if hasattr(evaluation_result, 'status') else "completed",
            
            # 维度分数
            "dimension_scores": [
                {"dimension": dim, "score": f"{score:.3f}"} 
                for dim, score in evaluation_result.dimension_scores.items()
            ],
            
            # 改进建议
            "improvement_suggestions": [
                {"title": suggestion, "description": suggestion, "priority": "medium", "impact": "0.7"}
                for suggestion in evaluation_result.improvement_suggestions
            ],
            
            # 性能等级
            "performance_level": self._get_performance_level(evaluation_result.overall_score),
            
            # 维度分析
            "dimension_analysis": [
                {
                    "dimension": dim,
                    "score": f"{score:.3f}",
                    "weight": "25",  # 简化处理
                    "contribution": f"{score * 25:.1f}",
                    "level": self._get_performance_level(score)
                }
                for dim, score in evaluation_result.dimension_scores.items()
            ]
        }
        
        # 样本分析
        if evaluation_result.detailed_results:
            # 找出最佳和最差样本
            sorted_samples = sorted(evaluation_result.detailed_results, 
                                  key=lambda x: x.get_overall_score(evaluation_result.evaluation_config.weight_config), 
                                  reverse=True)
            
            data["best_samples"] = [
                {
                    "sample_id": sample.sample_id,
                    "score": f"{sample.get_overall_score(evaluation_result.evaluation_config.weight_config):.3f}",
                    "input_text": sample.input_text[:100] + "..." if len(sample.input_text) > 100 else sample.input_text,
                    "model_output": sample.model_output[:100] + "..." if len(sample.model_output) > 100 else sample.model_output
                }
                for sample in sorted_samples[:3]
            ]
            
            data["worst_samples"] = [
                {
                    "sample_id": sample.sample_id,
                    "score": f"{sample.get_overall_score(evaluation_result.evaluation_config.weight_config):.3f}",
                    "input_text": sample.input_text[:100] + "..." if len(sample.input_text) > 100 else sample.input_text,
                    "model_output": sample.model_output[:100] + "..." if len(sample.model_output) > 100 else sample.model_output,
                    "issues": ", ".join(sample.error_types) if sample.error_types else "无明显问题"
                }
                for sample in sorted_samples[-3:]
            ]
        
        # 合并额外数据
        data.update(additional_data)
        
        return data
    
    def _generate_charts(self, evaluation_result: EvaluationResult, 
                        report_type: ReportType) -> List[Dict[str, Any]]:
        """生成图表"""
        charts = []
        
        # 维度分数分布图
        if evaluation_result.dimension_scores:
            chart_data = self.chart_generator.generate_score_distribution_chart(
                evaluation_result.dimension_scores, "各维度评估分数"
            )
            charts.append({
                "title": "各维度评估分数",
                "data": chart_data,
                "description": "展示模型在各个评估维度上的表现"
            })
        
        # 错误分布图（如果有错误分析）
        if hasattr(evaluation_result, 'error_analysis') and evaluation_result.error_analysis:
            if evaluation_result.error_analysis.error_distribution:
                chart_data = self.chart_generator.generate_error_distribution_pie(
                    evaluation_result.error_analysis.error_distribution, "错误类型分布"
                )
                charts.append({
                    "title": "错误类型分布",
                    "data": chart_data,
                    "description": "显示不同类型错误的分布情况"
                })
        
        return charts
    
    def _render_template(self, template_content: str, data: Dict[str, Any]) -> str:
        """渲染模板"""
        # 简单的模板渲染实现
        content = template_content
        
        # 替换简单变量
        for key, value in data.items():
            if isinstance(value, (str, int, float)):
                content = content.replace(f"{{{{{key}}}}}", str(value))
        
        # 处理列表渲染（简化实现）
        if "dimension_scores" in data:
            dimension_html = ""
            for item in data["dimension_scores"]:
                dimension_html += f"""
                <div class="dimension-item">
                    <h4>{item['dimension']}</h4>
                    <div class="score-display">{item['score']}</div>
                </div>
                """
            content = content.replace("{{#dimension_scores}}", "").replace("{{/dimension_scores}}", "")
            content = content.replace('<div class="dimension-item">', dimension_html)
        
        # 处理图表渲染
        if "charts" in data:
            chart_html = ""
            for chart in data["charts"]:
                chart_html += f'<img src="data:image/png;base64,{chart["data"]}" alt="{chart["title"]}" /><p>{chart["title"]}</p>'
            content = content.replace("{{#charts}}", "").replace("{{/charts}}", "")
            content = content.replace('<img src="data:image/png;base64,{{chart_data}}" alt="{{chart_title}}" />', chart_html)
        
        return content
    
    def _get_performance_level(self, score: float) -> str:
        """获取性能等级"""
        if score >= 0.9:
            return "优秀"
        elif score >= 0.8:
            return "良好"
        elif score >= 0.6:
            return "及格"
        elif score >= 0.4:
            return "较差"
        else:
            return "很差"
    
    def export_report(self, report: Report, file_path: str, 
                     format_type: Optional[ReportFormat] = None):
        """
        导出报告到文件
        
        Args:
            report: 报告对象
            file_path: 文件路径
            format_type: 导出格式
        """
        export_format = format_type or ReportFormat(report.format_type)
        
        if export_format == ReportFormat.HTML:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report.content)
        
        elif export_format == ReportFormat.JSON:
            report_dict = report.to_dict()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
        elif export_format == ReportFormat.MARKDOWN:
            # 简单的HTML到Markdown转换
            markdown_content = self._html_to_markdown(report.content)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
        
        else:
            raise ValueError(f"不支持的导出格式: {export_format}")
    
    def _html_to_markdown(self, html_content: str) -> str:
        """简单的HTML到Markdown转换"""
        # 这里实现一个简化的转换
        content = html_content
        
        # 替换标题
        content = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1', content)
        content = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1', content)
        content = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1', content)
        content = re.sub(r'<h4[^>]*>(.*?)</h4>', r'#### \1', content)
        
        # 替换段落
        content = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', content)
        
        # 替换列表
        content = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1', content)
        content = re.sub(r'<ul[^>]*>|</ul>', '', content)
        content = re.sub(r'<ol[^>]*>|</ol>', '', content)
        
        # 移除其他HTML标签
        content = re.sub(r'<[^>]+>', '', content)
        
        # 清理多余的空行
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content.strip()
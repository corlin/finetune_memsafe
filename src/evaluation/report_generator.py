"""
报告生成器

实现标准化报告生成，支持MLflow/Weights & Biases标准格式，多种导出格式和可视化图表生成。
"""

import logging
import json
import csv
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available, some export formats will be disabled")

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logging.warning("Jinja2 not available, HTML template rendering will be limited")

from .data_models import (
    EvaluationResult, BenchmarkResult, ComparisonResult, 
    DataQualityReport, convert_numpy_types
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    报告生成器
    
    提供标准化报告生成功能：
    - 符合MLflow/Weights & Biases标准的报告格式
    - JSON、CSV、Excel格式的结果导出
    - LaTeX表格格式用于学术论文
    - 可视化图表和性能曲线生成
    """
    
    def __init__(self, 
                 output_dir: str = "./reports",
                 template_dir: Optional[str] = None):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
            template_dir: 模板目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.template_dir = Path(template_dir) if template_dir else self.output_dir / "templates"
        self.template_dir.mkdir(exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        logger.info(f"ReportGenerator初始化完成，输出目录: {output_dir}")
    
    def generate_evaluation_report(self, 
                                 result: EvaluationResult,
                                 format_type: str = "html",
                                 include_charts: bool = True) -> str:
        """
        生成评估报告
        
        Args:
            result: 评估结果
            format_type: 报告格式 ('html', 'json', 'csv', 'latex', 'mlflow')
            include_charts: 是否包含图表
            
        Returns:
            报告文件路径
        """
        logger.info(f"生成评估报告: {result.model_name}, 格式: {format_type}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if format_type == "html":
                return self._generate_html_evaluation_report(result, timestamp, include_charts)
            elif format_type == "json":
                return self._generate_json_report(result, timestamp)
            elif format_type == "csv":
                return self._generate_csv_report(result, timestamp)
            elif format_type == "latex":
                return self._generate_latex_report(result, timestamp)
            elif format_type == "mlflow":
                return self._generate_mlflow_report(result, timestamp)
            else:
                raise ValueError(f"不支持的报告格式: {format_type}")
                
        except Exception as e:
            logger.error(f"生成评估报告失败: {e}")
            return ""
    
    def generate_benchmark_report(self, 
                                result: BenchmarkResult,
                                format_type: str = "html",
                                include_charts: bool = True) -> str:
        """
        生成基准测试报告
        
        Args:
            result: 基准测试结果
            format_type: 报告格式
            include_charts: 是否包含图表
            
        Returns:
            报告文件路径
        """
        logger.info(f"生成基准测试报告: {result.benchmark_name}, 格式: {format_type}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if format_type == "html":
                return self._generate_html_benchmark_report(result, timestamp, include_charts)
            elif format_type == "json":
                return self._generate_json_report(result, timestamp)
            elif format_type == "csv":
                return self._generate_benchmark_csv_report(result, timestamp)
            elif format_type == "latex":
                return self._generate_benchmark_latex_report(result, timestamp)
            else:
                raise ValueError(f"不支持的报告格式: {format_type}")
                
        except Exception as e:
            logger.error(f"生成基准测试报告失败: {e}")
            return ""
    
    def generate_comparison_report(self, 
                                 comparison_result: ComparisonResult,
                                 format_type: str = "html",
                                 include_charts: bool = True) -> str:
        """
        生成比较报告
        
        Args:
            comparison_result: 比较结果
            format_type: 报告格式
            include_charts: 是否包含图表
            
        Returns:
            报告文件路径
        """
        logger.info(f"生成比较报告，模型数量: {len(comparison_result.models)}, 格式: {format_type}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if format_type == "html":
                return self._generate_html_comparison_report(comparison_result, timestamp, include_charts)
            elif format_type == "json":
                return self._generate_json_report(comparison_result, timestamp)
            elif format_type == "csv":
                return self._generate_comparison_csv_report(comparison_result, timestamp)
            elif format_type == "latex":
                return self._generate_comparison_latex_report(comparison_result, timestamp)
            else:
                raise ValueError(f"不支持的报告格式: {format_type}")
                
        except Exception as e:
            logger.error(f"生成比较报告失败: {e}")
            return ""
    
    def export_to_excel(self, 
                       data: Union[EvaluationResult, BenchmarkResult, ComparisonResult],
                       output_path: Optional[str] = None) -> str:
        """
        导出到Excel格式
        
        Args:
            data: 要导出的数据
            output_path: 输出路径
            
        Returns:
            Excel文件路径
        """
        if not PANDAS_AVAILABLE:
            logger.error("Pandas不可用，无法导出Excel格式")
            return ""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_path is None:
                output_path = self.output_dir / f"export_{timestamp}.xlsx"
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                if isinstance(data, EvaluationResult):
                    self._export_evaluation_to_excel(data, writer)
                elif isinstance(data, BenchmarkResult):
                    self._export_benchmark_to_excel(data, writer)
                elif isinstance(data, ComparisonResult):
                    self._export_comparison_to_excel(data, writer)
            
            logger.info(f"Excel导出完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Excel导出失败: {e}")
            return ""
    
    def create_performance_charts(self, 
                                data: Union[EvaluationResult, BenchmarkResult, ComparisonResult],
                                chart_types: List[str] = ["bar", "line", "radar"]) -> List[str]:
        """
        创建性能图表
        
        Args:
            data: 数据
            chart_types: 图表类型列表
            
        Returns:
            图表文件路径列表
        """
        chart_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            for chart_type in chart_types:
                chart_path = self.output_dir / f"chart_{chart_type}_{timestamp}.png"
                
                if isinstance(data, ComparisonResult):
                    success = self._create_comparison_chart(data, chart_type, chart_path)
                elif isinstance(data, EvaluationResult):
                    success = self._create_evaluation_chart(data, chart_type, chart_path)
                elif isinstance(data, BenchmarkResult):
                    success = self._create_benchmark_chart(data, chart_type, chart_path)
                else:
                    success = False
                
                if success:
                    chart_paths.append(str(chart_path))
            
            logger.info(f"生成图表完成，共 {len(chart_paths)} 个")
            return chart_paths
            
        except Exception as e:
            logger.error(f"创建性能图表失败: {e}")
            return []
    
    def generate_latex_table(self, 
                            data: Union[EvaluationResult, BenchmarkResult, ComparisonResult],
                            caption: str = "",
                            label: str = "") -> str:
        """
        生成LaTeX表格
        
        Args:
            data: 数据
            caption: 表格标题
            label: 表格标签
            
        Returns:
            LaTeX表格代码
        """
        try:
            if isinstance(data, ComparisonResult):
                return self._generate_comparison_latex_table(data, caption, label)
            elif isinstance(data, EvaluationResult):
                return self._generate_evaluation_latex_table(data, caption, label)
            elif isinstance(data, BenchmarkResult):
                return self._generate_benchmark_latex_table(data, caption, label)
            else:
                return ""
                
        except Exception as e:
            logger.error(f"生成LaTeX表格失败: {e}")
            return ""
    
    def _generate_html_evaluation_report(self, 
                                       result: EvaluationResult, 
                                       timestamp: str,
                                       include_charts: bool) -> str:
        """生成HTML评估报告"""
        output_path = self.output_dir / f"evaluation_report_{result.model_name}_{timestamp}.html"
        
        # 生成图表
        chart_paths = []
        if include_charts:
            chart_paths = self.create_performance_charts(result, ["bar", "radar"])
        
        # 转换图表为base64
        chart_images = {}
        for i, chart_path in enumerate(chart_paths):
            with open(chart_path, 'rb') as f:
                chart_data = base64.b64encode(f.read()).decode()
                chart_images[f"chart_{i}"] = chart_data
        
        html_content = self._create_evaluation_html_template(result, chart_images)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _generate_html_benchmark_report(self, 
                                      result: BenchmarkResult, 
                                      timestamp: str,
                                      include_charts: bool) -> str:
        """生成HTML基准测试报告"""
        output_path = self.output_dir / f"benchmark_report_{result.benchmark_name}_{timestamp}.html"
        
        # 生成图表
        chart_paths = []
        if include_charts:
            chart_paths = self.create_performance_charts(result, ["bar"])
        
        # 转换图表为base64
        chart_images = {}
        for i, chart_path in enumerate(chart_paths):
            with open(chart_path, 'rb') as f:
                chart_data = base64.b64encode(f.read()).decode()
                chart_images[f"chart_{i}"] = chart_data
        
        html_content = self._create_benchmark_html_template(result, chart_images)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _generate_html_comparison_report(self, 
                                       comparison_result: ComparisonResult, 
                                       timestamp: str,
                                       include_charts: bool) -> str:
        """生成HTML比较报告"""
        output_path = self.output_dir / f"comparison_report_{timestamp}.html"
        
        # 生成图表
        chart_paths = []
        if include_charts:
            chart_paths = self.create_performance_charts(comparison_result, ["bar", "radar", "heatmap"])
        
        # 转换图表为base64
        chart_images = {}
        for i, chart_path in enumerate(chart_paths):
            with open(chart_path, 'rb') as f:
                chart_data = base64.b64encode(f.read()).decode()
                chart_images[f"chart_{i}"] = chart_data
        
        html_content = self._create_comparison_html_template(comparison_result, chart_images)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _generate_json_report(self, data: Any, timestamp: str) -> str:
        """生成JSON报告"""
        if hasattr(data, 'get_summary'):
            report_data = data.get_summary()
        elif hasattr(data, 'to_dict'):
            report_data = data.to_dict()
        else:
            report_data = data.__dict__
        
        output_path = self.output_dir / f"report_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(report_data), f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def _generate_csv_report(self, result: EvaluationResult, timestamp: str) -> str:
        """生成CSV评估报告"""
        output_path = self.output_dir / f"evaluation_report_{timestamp}.csv"
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入基本信息
            writer.writerow(["模型名称", result.model_name])
            writer.writerow(["评估时间", result.evaluation_time.isoformat()])
            writer.writerow([])
            
            # 写入整体指标
            writer.writerow(["整体指标"])
            writer.writerow(["指标名称", "数值"])
            for metric, value in result.metrics.items():
                writer.writerow([metric, value])
            writer.writerow([])
            
            # 写入任务结果
            for task_name, task_result in result.task_results.items():
                writer.writerow([f"任务: {task_name}"])
                writer.writerow(["指标名称", "数值"])
                for metric, value in task_result.metrics.items():
                    writer.writerow([metric, value])
                writer.writerow([])
        
        return str(output_path)
    
    def _generate_benchmark_csv_report(self, result: BenchmarkResult, timestamp: str) -> str:
        """生成CSV基准测试报告"""
        output_path = self.output_dir / f"benchmark_report_{timestamp}.csv"
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入基本信息
            writer.writerow(["基准测试", result.benchmark_name])
            writer.writerow(["模型名称", result.model_name])
            writer.writerow(["总分", result.overall_score])
            writer.writerow(["评估时间", result.evaluation_time.isoformat()])
            writer.writerow([])
            
            # 写入任务结果
            writer.writerow(["任务名称", "主要指标", "分数"])
            for task_name, task_result in result.task_results.items():
                main_metric = "accuracy" if "accuracy" in task_result.metrics else list(task_result.metrics.keys())[0]
                score = task_result.metrics.get(main_metric, 0)
                writer.writerow([task_name, main_metric, score])
        
        return str(output_path)
    
    def _generate_comparison_csv_report(self, comparison_result: ComparisonResult, timestamp: str) -> str:
        """生成CSV比较报告"""
        output_path = self.output_dir / f"comparison_report_{timestamp}.csv"
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            header = ["模型名称"] + list(comparison_result.metrics.keys())
            writer.writerow(header)
            
            # 写入数据
            for i, model in enumerate(comparison_result.models):
                row = [model]
                for metric, values in comparison_result.metrics.items():
                    value = values[i] if i < len(values) else 0
                    row.append(value)
                writer.writerow(row)
        
        return str(output_path)
    
    def _generate_latex_report(self, result: EvaluationResult, timestamp: str) -> str:
        """生成LaTeX评估报告"""
        output_path = self.output_dir / f"evaluation_report_{timestamp}.tex"
        
        latex_content = self._generate_evaluation_latex_table(
            result, 
            f"模型 {result.model_name} 评估结果",
            f"tab:eval_{result.model_name}"
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        return str(output_path)
    
    def _generate_benchmark_latex_report(self, result: BenchmarkResult, timestamp: str) -> str:
        """生成LaTeX基准测试报告"""
        output_path = self.output_dir / f"benchmark_report_{timestamp}.tex"
        
        latex_content = self._generate_benchmark_latex_table(
            result,
            f"{result.benchmark_name} 基准测试结果",
            f"tab:benchmark_{result.benchmark_name}"
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        return str(output_path)
    
    def _generate_comparison_latex_report(self, comparison_result: ComparisonResult, timestamp: str) -> str:
        """生成LaTeX比较报告"""
        output_path = self.output_dir / f"comparison_report_{timestamp}.tex"
        
        latex_content = self._generate_comparison_latex_table(
            comparison_result,
            "模型性能比较",
            "tab:model_comparison"
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        return str(output_path)
    
    def _generate_mlflow_report(self, result: EvaluationResult, timestamp: str) -> str:
        """生成MLflow格式报告"""
        output_path = self.output_dir / f"mlflow_report_{timestamp}.json"
        
        mlflow_data = {
            "run_id": f"run_{timestamp}",
            "experiment_id": "0",
            "status": "FINISHED",
            "start_time": int(result.evaluation_time.timestamp() * 1000),
            "end_time": int(datetime.now().timestamp() * 1000),
            "artifact_uri": str(self.output_dir),
            "lifecycle_stage": "active",
            "data": {
                "metrics": convert_numpy_types(result.metrics),
                "params": {
                    "model_name": result.model_name,
                    "evaluation_config": result.config.to_dict()
                },
                "tags": {
                    "mlflow.runName": f"evaluation_{result.model_name}",
                    "mlflow.source.type": "LOCAL",
                    "evaluation_type": "model_evaluation"
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mlflow_data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def _create_evaluation_html_template(self, result: EvaluationResult, chart_images: Dict[str, str]) -> str:
        """创建评估HTML模板"""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>模型评估报告 - {model_name}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-name {{ color: #7f8c8d; font-size: 14px; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .task-result {{ border-left: 4px solid #3498db; padding-left: 15px; margin: 15px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .efficiency-section {{ background: #e8f5e8; }}
                .quality-section {{ background: #fff3cd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>模型评估报告</h1>
                <h2>{model_name}</h2>
                <p>评估时间: {evaluation_time}</p>
            </div>
            
            <div class="section">
                <h2>整体指标</h2>
                <div class="metric-grid">
                    {overall_metrics_html}
                </div>
            </div>
            
            {charts_html}
            
            <div class="section">
                <h2>任务详细结果</h2>
                {task_results_html}
            </div>
            
            <div class="section efficiency-section">
                <h2>效率指标</h2>
                {efficiency_html}
            </div>
            
            <div class="section quality-section">
                <h2>质量分数</h2>
                {quality_html}
            </div>
        </body>
        </html>
        """
        
        # 生成各部分内容
        overall_metrics_html = self._generate_metrics_cards_html(result.metrics)
        charts_html = self._generate_charts_html(chart_images)
        task_results_html = self._generate_task_results_html(result.task_results)
        efficiency_html = self._generate_efficiency_html(result.efficiency_metrics)
        quality_html = self._generate_quality_html(result.quality_scores)
        
        return template.format(
            model_name=result.model_name,
            evaluation_time=result.evaluation_time.strftime("%Y-%m-%d %H:%M:%S"),
            overall_metrics_html=overall_metrics_html,
            charts_html=charts_html,
            task_results_html=task_results_html,
            efficiency_html=efficiency_html,
            quality_html=quality_html
        )
    
    def _create_benchmark_html_template(self, result: BenchmarkResult, chart_images: Dict[str, str]) -> str:
        """创建基准测试HTML模板"""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>基准测试报告 - {benchmark_name}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .score-display {{ text-align: center; font-size: 48px; font-weight: bold; color: #e74c3c; margin: 20px 0; }}
                .task-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
                .task-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; }}
                .task-score {{ font-size: 20px; font-weight: bold; color: #2c3e50; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>基准测试报告</h1>
                <h2>{benchmark_name}</h2>
                <p>模型: {model_name}</p>
                <p>评估时间: {evaluation_time}</p>
            </div>
            
            <div class="section">
                <h2>总体分数</h2>
                <div class="score-display">{overall_score:.4f}</div>
            </div>
            
            {charts_html}
            
            <div class="section">
                <h2>任务分数</h2>
                <div class="task-grid">
                    {task_scores_html}
                </div>
            </div>
            
            <div class="section">
                <h2>详细结果</h2>
                {detailed_results_html}
            </div>
        </body>
        </html>
        """
        
        # 生成各部分内容
        charts_html = self._generate_charts_html(chart_images)
        task_scores_html = self._generate_task_scores_html(result.task_results)
        detailed_results_html = self._generate_benchmark_detailed_html(result.task_results)
        
        return template.format(
            benchmark_name=result.benchmark_name,
            model_name=result.model_name,
            evaluation_time=result.evaluation_time.strftime("%Y-%m-%d %H:%M:%S"),
            overall_score=result.overall_score,
            charts_html=charts_html,
            task_scores_html=task_scores_html,
            detailed_results_html=detailed_results_html
        )
    
    def _create_comparison_html_template(self, comparison_result: ComparisonResult, chart_images: Dict[str, str]) -> str:
        """创建比较HTML模板"""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>模型比较报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .model-count {{ font-size: 24px; font-weight: bold; text-align: center; margin: 20px 0; }}
                .ranking-section {{ background: #e8f5e8; }}
                .best-models-section {{ background: #fff3cd; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .best-score {{ background-color: #d4edda; font-weight: bold; }}
                .ranking-list {{ list-style: none; padding: 0; }}
                .ranking-item {{ background: #f8f9fa; margin: 5px 0; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>模型比较报告</h1>
                <p>生成时间: {timestamp}</p>
            </div>
            
            <div class="section">
                <div class="model-count">比较模型数量: {num_models}</div>
                <p>参与比较的模型: {models_list}</p>
            </div>
            
            {charts_html}
            
            <div class="section ranking-section">
                <h2>模型排名</h2>
                {rankings_html}
            </div>
            
            <div class="section">
                <h2>详细指标对比</h2>
                {metrics_table_html}
            </div>
            
            <div class="section best-models-section">
                <h2>各指标最佳模型</h2>
                {best_models_html}
            </div>
        </body>
        </html>
        """
        
        # 生成各部分内容
        charts_html = self._generate_charts_html(chart_images)
        rankings_html = self._generate_rankings_html(comparison_result.rankings)
        metrics_table_html = self._generate_comparison_metrics_table_html(comparison_result)
        best_models_html = self._generate_best_models_html(comparison_result.best_model)
        
        return template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            num_models=len(comparison_result.models),
            models_list=", ".join(comparison_result.models),
            charts_html=charts_html,
            rankings_html=rankings_html,
            metrics_table_html=metrics_table_html,
            best_models_html=best_models_html
        )
    
    def _generate_metrics_cards_html(self, metrics: Dict[str, float]) -> str:
        """生成指标卡片HTML"""
        html = ""
        for metric, value in metrics.items():
            html += f'''
            <div class="metric-card">
                <div class="metric-value">{value:.4f}</div>
                <div class="metric-name">{metric}</div>
            </div>
            '''
        return html
    
    def _generate_charts_html(self, chart_images: Dict[str, str]) -> str:
        """生成图表HTML"""
        if not chart_images:
            return ""
        
        html = '<div class="section"><h2>性能图表</h2>'
        for chart_name, chart_data in chart_images.items():
            html += f'''
            <div class="chart">
                <img src="data:image/png;base64,{chart_data}" alt="{chart_name}" style="max-width: 100%; height: auto;">
            </div>
            '''
        html += '</div>'
        return html
    
    def _generate_task_results_html(self, task_results: Dict[str, Any]) -> str:
        """生成任务结果HTML"""
        html = ""
        for task_name, result in task_results.items():
            html += f'''
            <div class="task-result">
                <h3>{task_name}</h3>
                <table>
                    <tr><th>指标</th><th>数值</th></tr>
            '''
            for metric, value in result.metrics.items():
                html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
            html += "</table></div>"
        return html
    
    def _generate_efficiency_html(self, efficiency_metrics) -> str:
        """生成效率指标HTML"""
        if not efficiency_metrics:
            return "<p>无效率指标数据</p>"
        
        return f'''
        <table>
            <tr><th>指标</th><th>数值</th></tr>
            <tr><td>推理延迟 (ms)</td><td>{efficiency_metrics.inference_latency:.2f}</td></tr>
            <tr><td>吞吐量 (tokens/s)</td><td>{efficiency_metrics.throughput:.2f}</td></tr>
            <tr><td>内存使用 (GB)</td><td>{efficiency_metrics.memory_usage:.2f}</td></tr>
            <tr><td>模型大小 (MB)</td><td>{efficiency_metrics.model_size:.2f}</td></tr>
        </table>
        '''
    
    def _generate_quality_html(self, quality_scores) -> str:
        """生成质量分数HTML"""
        if not quality_scores:
            return "<p>无质量分数数据</p>"
        
        return f'''
        <table>
            <tr><th>质量维度</th><th>分数</th></tr>
            <tr><td>流畅度</td><td>{quality_scores.fluency:.4f}</td></tr>
            <tr><td>连贯性</td><td>{quality_scores.coherence:.4f}</td></tr>
            <tr><td>相关性</td><td>{quality_scores.relevance:.4f}</td></tr>
            <tr><td>事实性</td><td>{quality_scores.factuality:.4f}</td></tr>
            <tr><td>总体质量</td><td>{quality_scores.overall:.4f}</td></tr>
        </table>
        '''
    
    def _generate_task_scores_html(self, task_results: Dict[str, Any]) -> str:
        """生成任务分数HTML"""
        html = ""
        for task_name, result in task_results.items():
            # 获取主要指标
            main_metric = "accuracy" if "accuracy" in result.metrics else list(result.metrics.keys())[0]
            score = result.metrics.get(main_metric, 0)
            
            html += f'''
            <div class="task-card">
                <h4>{task_name}</h4>
                <div class="task-score">{score:.4f}</div>
                <div>主要指标: {main_metric}</div>
            </div>
            '''
        return html
    
    def _generate_benchmark_detailed_html(self, task_results: Dict[str, Any]) -> str:
        """生成基准测试详细结果HTML"""
        html = "<table><tr><th>任务</th><th>指标</th><th>分数</th></tr>"
        
        for task_name, result in task_results.items():
            for metric, value in result.metrics.items():
                html += f"<tr><td>{task_name}</td><td>{metric}</td><td>{value:.4f}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_rankings_html(self, rankings: Dict[str, List[str]]) -> str:
        """生成排名HTML"""
        html = ""
        for metric, model_list in rankings.items():
            html += f'''
            <div class="ranking-item">
                <strong>{metric}:</strong>
                <ol>
            '''
            for model in model_list:
                html += f"<li>{model}</li>"
            html += "</ol></div>"
        return html
    
    def _generate_comparison_metrics_table_html(self, comparison_result: ComparisonResult) -> str:
        """生成比较指标表格HTML"""
        html = '<table><tr><th>模型</th>'
        
        # 表头
        for metric in comparison_result.metrics.keys():
            html += f"<th>{metric}</th>"
        html += "</tr>"
        
        # 数据行
        for i, model in enumerate(comparison_result.models):
            html += f"<tr><td>{model}</td>"
            for metric, values in comparison_result.metrics.items():
                value = values[i] if i < len(values) else 0.0
                # 检查是否是最佳分数
                is_best = (comparison_result.best_model.get(metric) == model)
                css_class = ' class="best-score"' if is_best else ''
                html += f"<td{css_class}>{value:.4f}</td>"
            html += "</tr>"
        
        html += "</table>"
        return html
    
    def _generate_best_models_html(self, best_model: Dict[str, str]) -> str:
        """生成最佳模型HTML"""
        html = "<ul>"
        for metric, model in best_model.items():
            html += f"<li><strong>{metric}:</strong> {model}</li>"
        html += "</ul>"
        return html
    
    def _generate_evaluation_latex_table(self, result: EvaluationResult, caption: str, label: str) -> str:
        """生成评估LaTeX表格"""
        latex = f'''
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{|l|r|}}
\\hline
\\textbf{{指标}} & \\textbf{{数值}} \\\\
\\hline
'''
        
        for metric, value in result.metrics.items():
            latex += f"{metric} & {value:.4f} \\\\\n"
        
        latex += '''\\hline
\\end{tabular}
\\end{table}
'''
        return latex
    
    def _generate_benchmark_latex_table(self, result: BenchmarkResult, caption: str, label: str) -> str:
        """生成基准测试LaTeX表格"""
        latex = f'''
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{|l|r|}}
\\hline
\\textbf{{任务}} & \\textbf{{分数}} \\\\
\\hline
'''
        
        for task_name, task_result in result.task_results.items():
            main_metric = "accuracy" if "accuracy" in task_result.metrics else list(task_result.metrics.keys())[0]
            score = task_result.metrics.get(main_metric, 0)
            latex += f"{task_name} & {score:.4f} \\\\\n"
        
        latex += f'''\\hline
\\textbf{{总分}} & \\textbf{{{result.overall_score:.4f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
'''
        return latex
    
    def _generate_comparison_latex_table(self, comparison_result: ComparisonResult, caption: str, label: str) -> str:
        """生成比较LaTeX表格"""
        metrics_list = list(comparison_result.metrics.keys())
        
        # 构建表格列格式
        col_format = "|l|" + "r|" * len(metrics_list)
        
        latex = f'''
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_format}}}
\\hline
\\textbf{{模型}}'''
        
        # 表头
        for metric in metrics_list:
            latex += f" & \\textbf{{{metric}}}"
        latex += " \\\\\n\\hline\n"
        
        # 数据行
        for i, model in enumerate(comparison_result.models):
            latex += model
            for metric in metrics_list:
                values = comparison_result.metrics[metric]
                value = values[i] if i < len(values) else 0.0
                # 检查是否是最佳分数
                is_best = (comparison_result.best_model.get(metric) == model)
                if is_best:
                    latex += f" & \\textbf{{{value:.4f}}}"
                else:
                    latex += f" & {value:.4f}"
            latex += " \\\\\n"
        
        latex += '''\\hline
\\end{tabular}
\\end{table}
'''
        return latex
    
    def _create_comparison_chart(self, comparison_result: ComparisonResult, chart_type: str, output_path: Path) -> bool:
        """创建比较图表"""
        try:
            if chart_type == "bar":
                return self._create_comparison_bar_chart(comparison_result, output_path)
            elif chart_type == "radar":
                return self._create_comparison_radar_chart(comparison_result, output_path)
            elif chart_type == "heatmap":
                return self._create_comparison_heatmap(comparison_result, output_path)
            else:
                return False
        except Exception as e:
            logger.error(f"创建比较图表失败: {e}")
            return False
    
    def _create_comparison_bar_chart(self, comparison_result: ComparisonResult, output_path: Path) -> bool:
        """创建比较柱状图"""
        plt.figure(figsize=(12, 6))
        
        # 选择前5个指标
        metrics_subset = list(comparison_result.metrics.items())[:5]
        x = np.arange(len(comparison_result.models))
        width = 0.8 / len(metrics_subset)
        
        for i, (metric, values) in enumerate(metrics_subset):
            plt.bar(x + i * width, values, width, label=metric)
        
        plt.title('模型性能比较')
        plt.xlabel('模型')
        plt.ylabel('分数')
        plt.xticks(x + width * (len(metrics_subset) - 1) / 2, comparison_result.models, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    
    def _create_comparison_radar_chart(self, comparison_result: ComparisonResult, output_path: Path) -> bool:
        """创建比较雷达图"""
        metrics_names = list(comparison_result.metrics.keys())[:6]  # 最多6个指标
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        for i, model in enumerate(comparison_result.models[:3]):  # 最多显示3个模型
            values = []
            for metric in metrics_names:
                if metric in comparison_result.metrics:
                    model_values = comparison_result.metrics[metric]
                    values.append(model_values[i] if i < len(model_values) else 0)
                else:
                    values.append(0)
            
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_title('模型性能雷达图', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    
    def _create_comparison_heatmap(self, comparison_result: ComparisonResult, output_path: Path) -> bool:
        """创建比较热力图"""
        # 准备数据矩阵
        metrics_names = list(comparison_result.metrics.keys())
        models = comparison_result.models
        
        data_matrix = []
        for model_idx in range(len(models)):
            row = []
            for metric in metrics_names:
                values = comparison_result.metrics[metric]
                value = values[model_idx] if model_idx < len(values) else 0
                row.append(value)
            data_matrix.append(row)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(data_matrix, 
                   xticklabels=metrics_names,
                   yticklabels=models,
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd')
        
        plt.title('模型性能热力图')
        plt.xlabel('指标')
        plt.ylabel('模型')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    
    def _create_evaluation_chart(self, result: EvaluationResult, chart_type: str, output_path: Path) -> bool:
        """创建评估图表"""
        try:
            if chart_type == "bar":
                plt.figure(figsize=(10, 6))
                metrics = list(result.metrics.items())
                names, values = zip(*metrics)
                plt.bar(names, values)
                plt.title(f'模型评估结果 - {result.model_name}')
                plt.ylabel('分数')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"创建评估图表失败: {e}")
            return False
    
    def _create_benchmark_chart(self, result: BenchmarkResult, chart_type: str, output_path: Path) -> bool:
        """创建基准测试图表"""
        try:
            if chart_type == "bar":
                plt.figure(figsize=(10, 6))
                
                task_names = []
                scores = []
                
                for task_name, task_result in result.task_results.items():
                    task_names.append(task_name)
                    main_metric = "accuracy" if "accuracy" in task_result.metrics else list(task_result.metrics.keys())[0]
                    scores.append(task_result.metrics.get(main_metric, 0))
                
                plt.bar(task_names, scores)
                plt.title(f'{result.benchmark_name} 基准测试结果')
                plt.ylabel('分数')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"创建基准测试图表失败: {e}")
            return False
    
    def _export_evaluation_to_excel(self, result: EvaluationResult, writer):
        """导出评估结果到Excel"""
        # 基本信息
        basic_info = pd.DataFrame([
            ["模型名称", result.model_name],
            ["评估时间", result.evaluation_time.isoformat()]
        ], columns=["项目", "值"])
        basic_info.to_excel(writer, sheet_name="基本信息", index=False)
        
        # 整体指标
        metrics_df = pd.DataFrame(list(result.metrics.items()), columns=["指标", "数值"])
        metrics_df.to_excel(writer, sheet_name="整体指标", index=False)
        
        # 任务结果
        for task_name, task_result in result.task_results.items():
            task_df = pd.DataFrame(list(task_result.metrics.items()), columns=["指标", "数值"])
            sheet_name = f"任务_{task_name}"[:31]  # Excel工作表名称限制
            task_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def _export_benchmark_to_excel(self, result: BenchmarkResult, writer):
        """导出基准测试结果到Excel"""
        # 基本信息
        basic_info = pd.DataFrame([
            ["基准测试", result.benchmark_name],
            ["模型名称", result.model_name],
            ["总分", result.overall_score],
            ["评估时间", result.evaluation_time.isoformat()]
        ], columns=["项目", "值"])
        basic_info.to_excel(writer, sheet_name="基本信息", index=False)
        
        # 任务分数
        task_data = []
        for task_name, task_result in result.task_results.items():
            main_metric = "accuracy" if "accuracy" in task_result.metrics else list(task_result.metrics.keys())[0]
            score = task_result.metrics.get(main_metric, 0)
            task_data.append([task_name, main_metric, score])
        
        task_df = pd.DataFrame(task_data, columns=["任务", "主要指标", "分数"])
        task_df.to_excel(writer, sheet_name="任务分数", index=False)
    
    def _export_comparison_to_excel(self, comparison_result: ComparisonResult, writer):
        """导出比较结果到Excel"""
        # 模型比较
        data = []
        for i, model in enumerate(comparison_result.models):
            row = [model]
            for metric, values in comparison_result.metrics.items():
                value = values[i] if i < len(values) else 0
                row.append(value)
            data.append(row)
        
        columns = ["模型"] + list(comparison_result.metrics.keys())
        comparison_df = pd.DataFrame(data, columns=columns)
        comparison_df.to_excel(writer, sheet_name="模型比较", index=False)
        
        # 排名
        ranking_data = []
        for metric, model_list in comparison_result.rankings.items():
            for rank, model in enumerate(model_list, 1):
                ranking_data.append([metric, rank, model])
        
        ranking_df = pd.DataFrame(ranking_data, columns=["指标", "排名", "模型"])
        ranking_df.to_excel(writer, sheet_name="排名", index=False)
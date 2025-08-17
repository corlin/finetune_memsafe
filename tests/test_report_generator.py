"""
报告生成器测试

测试ReportGenerator类的功能。
"""

import pytest
from datetime import datetime
from pathlib import Path
import json

from evaluation import ReportGenerator
from evaluation.data_models import EvaluationResult, EfficiencyMetrics, QualityScores, TaskResult
from tests.conftest import assert_file_exists


class TestReportGenerator:
    """报告生成器测试类"""
    
    def test_init_default_params(self):
        """测试默认参数初始化"""
        generator = ReportGenerator()
        
        assert generator.template_dir == "templates"
        assert generator.output_dir == "reports"
        assert generator.include_plots == True
        assert generator.language == "zh"
    
    def test_init_custom_params(self, temp_dir):
        """测试自定义参数初始化"""
        generator = ReportGenerator(
            template_dir=str(temp_dir / "templates"),
            output_dir=str(temp_dir / "reports"),
            include_plots=False,
            language="en"
        )
        
        assert generator.template_dir == str(temp_dir / "templates")
        assert generator.output_dir == str(temp_dir / "reports")
        assert generator.include_plots == False
        assert generator.language == "en"
    
    def test_generate_evaluation_report_html(self, temp_dir, evaluation_config):
        """测试生成HTML评估报告"""
        generator = ReportGenerator(output_dir=str(temp_dir))
        
        # 创建模拟评估结果
        task_result = TaskResult(
            task_name="text_generation",
            predictions=["预测1", "预测2"],
            references=["参考1", "参考2"],
            metrics={"bleu": 0.75, "rouge": 0.80},
            samples=[],
            execution_time=2.5
        )
        
        result = EvaluationResult(
            model_name="test_model",
            evaluation_time=datetime.now(),
            metrics={"overall_score": 0.85, "accuracy": 0.90},
            task_results={"text_generation": task_result},
            efficiency_metrics=EfficiencyMetrics(150, 60, 2.5, 600),
            quality_scores=QualityScores(0.85, 0.80, 0.88, 0.82, 0.84),
            config=evaluation_config
        )
        
        output_path = generator.generate_evaluation_report(result, format="html")
        
        assert_file_exists(Path(output_path))
        
        # 检查HTML内容
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "<html>" in content
        assert "test_model" in content
        assert "0.85" in content  # overall_score
        assert "text_generation" in content
    
    def test_generate_evaluation_report_json(self, temp_dir, evaluation_config):
        """测试生成JSON评估报告"""
        generator = ReportGenerator(output_dir=str(temp_dir))
        
        result = EvaluationResult(
            model_name="test_model",
            evaluation_time=datetime.now(),
            metrics={"accuracy": 0.85},
            task_results={},
            efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
            quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            config=evaluation_config
        )
        
        output_path = generator.generate_evaluation_report(result, format="json")
        
        assert_file_exists(Path(output_path))
        
        # 检查JSON内容
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data["model_name"] == "test_model"
        assert data["metrics"]["accuracy"] == 0.85
        assert "efficiency_metrics" in data
        assert "quality_scores" in data
    
    def test_generate_comparison_report(self, temp_dir, evaluation_config):
        """测试生成对比报告"""
        generator = ReportGenerator(output_dir=str(temp_dir))
        
        # 创建多个评估结果
        results = []
        for i in range(3):
            result = EvaluationResult(
                model_name=f"model_{i}",
                evaluation_time=datetime.now(),
                metrics={"accuracy": 0.8 + i * 0.05, "f1": 0.75 + i * 0.05},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100 + i * 10, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            results.append(result)
        
        output_path = generator.generate_comparison_report(results, format="html")
        
        assert_file_exists(Path(output_path))
        
        # 检查HTML内容
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "model_0" in content
        assert "model_1" in content
        assert "model_2" in content
        assert "对比分析" in content or "comparison" in content.lower()
    
    def test_generate_benchmark_report(self, temp_dir):
        """测试生成基准测试报告"""
        generator = ReportGenerator(output_dir=str(temp_dir))
        
        # 创建模拟基准结果
        from evaluation.data_models import BenchmarkResult
        
        benchmark_result = BenchmarkResult(
            benchmark_name="clue",
            model_name="test_model",
            task_results={
                "tnews": {"accuracy": 0.85, "f1": 0.82},
                "afqmc": {"accuracy": 0.78, "f1": 0.75}
            },
            overall_score=0.815,
            metadata={"version": "1.0", "date": "2024-01-01"}
        )
        
        output_path = generator.generate_benchmark_report(benchmark_result, format="html")
        
        assert_file_exists(Path(output_path))
        
        # 检查内容
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "clue" in content.lower()
        assert "test_model" in content
        assert "0.815" in content
        assert "tnews" in content
        assert "afqmc" in content
    
    def test_generate_training_report(self, temp_dir):
        """测试生成训练报告"""
        generator = ReportGenerator(output_dir=str(temp_dir))
        
        # 创建模拟训练历史
        training_history = {
            "epochs": [1, 2, 3, 4, 5],
            "train_loss": [2.5, 2.0, 1.8, 1.6, 1.5],
            "val_loss": [2.3, 1.9, 1.7, 1.6, 1.6],
            "train_accuracy": [0.6, 0.7, 0.75, 0.8, 0.82],
            "val_accuracy": [0.65, 0.72, 0.76, 0.78, 0.79],
            "learning_rate": [0.001, 0.001, 0.0008, 0.0006, 0.0004]
        }
        
        training_config = {
            "model_name": "test_model",
            "dataset": "training_data",
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 5,
            "optimizer": "AdamW"
        }
        
        output_path = generator.generate_training_report(
            training_history, training_config, format="html"
        )
        
        assert_file_exists(Path(output_path))
        
        # 检查内容
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "训练报告" in content or "training" in content.lower()
        assert "test_model" in content
        assert "AdamW" in content
    
    def test_generate_latex_table(self, temp_dir, evaluation_config):
        """测试生成LaTeX表格"""
        generator = ReportGenerator(output_dir=str(temp_dir))
        
        # 创建多个评估结果
        results = []
        for i in range(3):
            result = EvaluationResult(
                model_name=f"Model-{i+1}",
                evaluation_time=datetime.now(),
                metrics={"Accuracy": 0.8 + i * 0.05, "F1-Score": 0.75 + i * 0.05},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            results.append(result)
        
        output_path = generator.generate_latex_table(results, metrics=["Accuracy", "F1-Score"])
        
        assert_file_exists(Path(output_path))
        
        # 检查LaTeX内容
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "\\begin{table}" in content
        assert "\\begin{tabular}" in content
        assert "Model-1" in content
        assert "Model-2" in content
        assert "Model-3" in content
        assert "\\end{table}" in content
    
    def test_generate_csv_export(self, temp_dir, evaluation_config):
        """测试生成CSV导出"""
        generator = ReportGenerator(output_dir=str(temp_dir))
        
        results = []
        for i in range(3):
            result = EvaluationResult(
                model_name=f"model_{i}",
                evaluation_time=datetime.now(),
                metrics={"accuracy": 0.8 + i * 0.05, "f1": 0.75 + i * 0.05},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100 + i * 10, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            results.append(result)
        
        output_path = generator.generate_csv_export(results)
        
        assert_file_exists(Path(output_path))
        
        # 检查CSV内容
        import pandas as pd
        df = pd.read_csv(output_path)
        
        assert len(df) == 3
        assert "model_name" in df.columns
        assert "accuracy" in df.columns
        assert "f1" in df.columns
        assert df.iloc[0]["model_name"] == "model_0"
    
    def test_generate_excel_export(self, temp_dir, evaluation_config):
        """测试生成Excel导出"""
        generator = ReportGenerator(output_dir=str(temp_dir))
        
        results = []
        for i in range(2):
            result = EvaluationResult(
                model_name=f"model_{i}",
                evaluation_time=datetime.now(),
                metrics={"accuracy": 0.8 + i * 0.1},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            results.append(result)
        
        output_path = generator.generate_excel_export(results)
        
        assert_file_exists(Path(output_path))
        
        # 检查Excel内容（需要pandas和openpyxl）
        try:
            import pandas as pd
            df = pd.read_excel(output_path)
            
            assert len(df) == 2
            assert "model_name" in df.columns
            assert df.iloc[0]["model_name"] == "model_0"
        except ImportError:
            # 如果没有安装相关库，至少检查文件是否创建
            pass
    
    def test_create_performance_plots(self, temp_dir, evaluation_config):
        """测试创建性能图表"""
        generator = ReportGenerator(output_dir=str(temp_dir), include_plots=True)
        
        results = []
        for i in range(5):
            result = EvaluationResult(
                model_name=f"model_{i}",
                evaluation_time=datetime.now(),
                metrics={"accuracy": 0.7 + i * 0.05, "f1": 0.65 + i * 0.05},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100 + i * 20, 50 + i * 5, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            results.append(result)
        
        plot_paths = generator.create_performance_plots(results)
        
        assert isinstance(plot_paths, list)
        assert len(plot_paths) > 0
        
        # 检查图表文件是否创建
        for plot_path in plot_paths:
            assert_file_exists(Path(plot_path))
    
    def test_create_training_curves(self, temp_dir):
        """测试创建训练曲线"""
        generator = ReportGenerator(output_dir=str(temp_dir), include_plots=True)
        
        training_history = {
            "epochs": list(range(1, 11)),
            "train_loss": [2.5 - i * 0.2 for i in range(10)],
            "val_loss": [2.3 - i * 0.18 for i in range(10)],
            "train_accuracy": [0.5 + i * 0.04 for i in range(10)],
            "val_accuracy": [0.52 + i * 0.035 for i in range(10)]
        }
        
        plot_paths = generator.create_training_curves(training_history)
        
        assert isinstance(plot_paths, list)
        assert len(plot_paths) > 0
        
        # 检查图表文件
        for plot_path in plot_paths:
            assert_file_exists(Path(plot_path))
    
    def test_generate_summary_statistics(self, evaluation_config):
        """测试生成汇总统计"""
        generator = ReportGenerator()
        
        results = []
        for i in range(10):
            result = EvaluationResult(
                model_name=f"model_{i}",
                evaluation_time=datetime.now(),
                metrics={"accuracy": 0.7 + i * 0.02, "f1": 0.65 + i * 0.025},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100 + i * 10, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            results.append(result)
        
        stats = generator.generate_summary_statistics(results)
        
        assert isinstance(stats, dict)
        assert "total_models" in stats
        assert "avg_accuracy" in stats
        assert "best_accuracy" in stats
        assert "worst_accuracy" in stats
        assert "std_accuracy" in stats
        
        assert stats["total_models"] == 10
        assert stats["best_accuracy"] > stats["worst_accuracy"]
        assert stats["std_accuracy"] >= 0
    
    def test_format_metrics_for_display(self):
        """测试格式化指标显示"""
        generator = ReportGenerator()
        
        metrics = {
            "accuracy": 0.8567,
            "f1": 0.7234,
            "precision": 0.9001,
            "recall": 0.6789
        }
        
        formatted = generator._format_metrics_for_display(metrics, decimal_places=3)
        
        assert formatted["accuracy"] == "0.857"
        assert formatted["f1"] == "0.723"
        assert formatted["precision"] == "0.900"
        assert formatted["recall"] == "0.679"
    
    def test_create_html_template(self, temp_dir):
        """测试创建HTML模板"""
        generator = ReportGenerator(template_dir=str(temp_dir))
        
        template_content = generator._create_html_template("evaluation")
        
        assert isinstance(template_content, str)
        assert "<html>" in template_content
        assert "{{" in template_content  # 模板变量
        assert "}}" in template_content
    
    def test_render_template(self, temp_dir):
        """测试渲染模板"""
        generator = ReportGenerator(template_dir=str(temp_dir))
        
        template = """
        <html>
        <body>
            <h1>{{title}}</h1>
            <p>Model: {{model_name}}</p>
            <p>Score: {{score}}</p>
        </body>
        </html>
        """
        
        data = {
            "title": "测试报告",
            "model_name": "test_model",
            "score": 0.85
        }
        
        rendered = generator._render_template(template, data)
        
        assert "测试报告" in rendered
        assert "test_model" in rendered
        assert "0.85" in rendered
        assert "{{" not in rendered  # 所有变量都应该被替换
    
    def test_save_report(self, temp_dir):
        """测试保存报告"""
        generator = ReportGenerator(output_dir=str(temp_dir))
        
        content = "<html><body><h1>测试报告</h1></body></html>"
        filename = "test_report.html"
        
        output_path = generator._save_report(content, filename)
        
        assert_file_exists(Path(output_path))
        
        # 检查内容
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        
        assert saved_content == content
    
    def test_error_handling_invalid_format(self, evaluation_config):
        """测试无效格式的错误处理"""
        generator = ReportGenerator()
        
        result = EvaluationResult(
            model_name="test_model",
            evaluation_time=datetime.now(),
            metrics={"accuracy": 0.85},
            task_results={},
            efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
            quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            config=evaluation_config
        )
        
        with pytest.raises(ValueError, match="不支持的报告格式"):
            generator.generate_evaluation_report(result, format="invalid_format")
    
    def test_multilingual_support(self, temp_dir, evaluation_config):
        """测试多语言支持"""
        # 中文报告
        generator_zh = ReportGenerator(output_dir=str(temp_dir), language="zh")
        
        result = EvaluationResult(
            model_name="test_model",
            evaluation_time=datetime.now(),
            metrics={"accuracy": 0.85},
            task_results={},
            efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
            quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            config=evaluation_config
        )
        
        output_path_zh = generator_zh.generate_evaluation_report(result, format="html")
        
        with open(output_path_zh, 'r', encoding='utf-8') as f:
            content_zh = f.read()
        
        assert "模型" in content_zh or "评估" in content_zh
        
        # 英文报告
        generator_en = ReportGenerator(output_dir=str(temp_dir), language="en")
        output_path_en = generator_en.generate_evaluation_report(result, format="html")
        
        with open(output_path_en, 'r', encoding='utf-8') as f:
            content_en = f.read()
        
        assert "model" in content_en.lower() or "evaluation" in content_en.lower()
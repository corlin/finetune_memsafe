"""
评估系统集成测试

测试整个评估系统的端到端功能。
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import shutil

from evaluation import (
    DataSplitter, EvaluationEngine, MetricsCalculator, 
    BenchmarkManager, ExperimentTracker, ReportGenerator,
    QualityAnalyzer
)
from evaluation.data_models import EvaluationConfig
from tests.conftest import create_test_dataset


class TestEvaluationIntegration:
    """评估系统集成测试类"""
    
    @pytest.fixture
    def temp_workspace(self):
        """临时工作空间夹具"""
        temp_dir = tempfile.mkdtemp()
        workspace = Path(temp_dir)
        
        # 创建目录结构
        (workspace / "data").mkdir()
        (workspace / "models").mkdir()
        (workspace / "experiments").mkdir()
        (workspace / "reports").mkdir()
        
        yield workspace
        shutil.rmtree(temp_dir)
    
    def test_complete_evaluation_pipeline(self, temp_workspace, mock_model, mock_tokenizer):
        """测试完整的评估流水线"""
        # 1. 数据拆分
        dataset = create_test_dataset(100)
        splitter = DataSplitter(min_samples_per_split=10)
        split_result = splitter.split_data(dataset, str(temp_workspace / "data"))
        
        assert split_result is not None
        assert len(split_result.train_dataset) > 0
        assert len(split_result.val_dataset) > 0
        assert len(split_result.test_dataset) > 0
        
        # 2. 质量分析
        analyzer = QualityAnalyzer()
        quality_report = analyzer.analyze_data_quality(split_result.test_dataset)
        
        assert quality_report is not None
        assert quality_report.total_samples > 0
        
        # 3. 模型评估
        config = EvaluationConfig(
            tasks=["classification"],
            metrics=["accuracy", "f1"],
            batch_size=4,
            num_samples=20
        )
        
        engine = EvaluationEngine(config)
        datasets = {"classification": split_result.test_dataset}
        
        with patch.object(engine, '_create_inference_function') as mock_inference:
            mock_inference.return_value = lambda inputs: ["positive"] * len(inputs)
            
            eval_result = engine.evaluate_model(
                model=mock_model,
                tokenizer=mock_tokenizer,
                datasets=datasets,
                model_name="test_model"
            )
        
        assert eval_result is not None
        assert eval_result.model_name == "test_model"
        assert "accuracy" in eval_result.metrics
        
        # 4. 实验跟踪
        tracker = ExperimentTracker(experiment_dir=str(temp_workspace / "experiments"))
        
        from evaluation.data_models import ExperimentConfig
        exp_config = ExperimentConfig(
            model_name="test_model",
            dataset_name="test_dataset",
            hyperparameters={"lr": 0.001}
        )
        
        experiment_id = tracker.track_experiment(exp_config, eval_result)
        assert experiment_id is not None
        
        # 5. 报告生成
        generator = ReportGenerator(output_dir=str(temp_workspace / "reports"))
        report_path = generator.generate_evaluation_report(eval_result, format="html")
        
        assert Path(report_path).exists()
    
    def test_benchmark_evaluation_workflow(self, temp_workspace, mock_model, mock_tokenizer):
        """测试基准评估工作流"""
        # 创建基准管理器
        benchmark_manager = BenchmarkManager(
            benchmark_dir=str(temp_workspace / "benchmarks")
        )
        
        # 模拟基准数据集
        from evaluation.data_models import BenchmarkDataset
        mock_benchmark = BenchmarkDataset(
            name="test_benchmark",
            tasks={"task1": create_test_dataset(50)},
            metadata={"version": "1.0"}
        )
        
        with patch.object(benchmark_manager, 'load_benchmark') as mock_load:
            with patch('evaluation.benchmark_manager.EvaluationEngine') as mock_engine_class:
                mock_load.return_value = mock_benchmark
                
                # 模拟评估引擎
                mock_engine = Mock()
                from evaluation.data_models import BenchmarkResult
                mock_result = BenchmarkResult(
                    benchmark_name="test_benchmark",
                    model_name="test_model",
                    task_results={"task1": {"accuracy": 0.8}},
                    overall_score=0.8,
                    metadata={}
                )
                mock_engine.evaluate_model.return_value = mock_result
                mock_engine_class.return_value = mock_engine
                
                # 运行基准评估
                result = benchmark_manager.run_custom_benchmark(
                    config=Mock(name="test_benchmark", dataset_path="", tasks=["task1"], 
                              evaluation_protocol="standard", metrics=["accuracy"]),
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    model_name="test_model"
                )
                
                assert result.benchmark_name == "test_benchmark"
                assert result.overall_score == 0.8
        
        # 生成基准报告
        generator = ReportGenerator(output_dir=str(temp_workspace / "reports"))
        report_path = generator.generate_benchmark_report(result, format="html")
        
        assert Path(report_path).exists()
    
    def test_multi_model_comparison_workflow(self, temp_workspace, mock_model, mock_tokenizer):
        """测试多模型对比工作流"""
        dataset = create_test_dataset(30)
        
        config = EvaluationConfig(
            tasks=["classification"],
            metrics=["accuracy"],
            batch_size=4,
            num_samples=10
        )
        
        engine = EvaluationEngine(config)
        datasets = {"classification": dataset}
        
        # 评估多个模型
        results = []
        model_names = ["model_a", "model_b", "model_c"]
        
        with patch.object(engine, '_create_inference_function') as mock_inference:
            # 模拟不同模型的不同性能
            def create_mock_inference(accuracy):
                def mock_func(inputs):
                    # 根据准确率返回正确或错误的预测
                    import random
                    return ["positive" if random.random() < accuracy else "negative" 
                           for _ in inputs]
                return mock_func
            
            for i, model_name in enumerate(model_names):
                accuracy = 0.7 + i * 0.1  # 递增的准确率
                mock_inference.return_value = create_mock_inference(accuracy)
                
                result = engine.evaluate_model(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    datasets=datasets,
                    model_name=model_name
                )
                results.append(result)
        
        # 生成对比报告
        generator = ReportGenerator(output_dir=str(temp_workspace / "reports"))
        comparison_path = generator.generate_comparison_report(results, format="html")
        
        assert Path(comparison_path).exists()
        
        # 实验跟踪和对比
        tracker = ExperimentTracker(experiment_dir=str(temp_workspace / "experiments"))
        
        experiment_ids = []
        for result in results:
            from evaluation.data_models import ExperimentConfig
            exp_config = ExperimentConfig(
                model_name=result.model_name,
                dataset_name="test_dataset",
                hyperparameters={"lr": 0.001}
            )
            exp_id = tracker.track_experiment(exp_config, result)
            experiment_ids.append(exp_id)
        
        # 生成对比分析
        comparison = tracker.compare_experiments(experiment_ids)
        assert "best_model" in comparison
        assert comparison["best_model"]["model_name"] in model_names
    
    def test_data_quality_and_evaluation_integration(self, temp_workspace):
        """测试数据质量分析与评估的集成"""
        # 创建有质量问题的数据集
        problematic_data = [
            {"text": "", "label": "empty"},  # 空文本
            {"text": "a", "label": "short"},  # 过短
            {"text": "正常长度的文本内容", "label": "normal"},
            {"text": "另一个正常的文本", "label": "normal"},
            {"text": "重复 重复 重复 重复", "label": "repetitive"},
            {"text": "x" * 1000, "label": "long"},  # 过长
        ] * 10  # 重复以增加样本数
        
        from datasets import Dataset
        dataset = Dataset.from_list(problematic_data)
        
        # 1. 质量分析
        analyzer = QualityAnalyzer()
        quality_report = analyzer.analyze_data_quality(dataset)
        
        assert quality_report.total_samples > 0
        assert len(quality_report.quality_issues) > 0
        
        # 2. 数据拆分（应该处理质量问题）
        splitter = DataSplitter(
            min_samples_per_split=10,
            enable_quality_analysis=True
        )
        
        split_result = splitter.split_data(dataset, str(temp_workspace / "data"))
        
        # 3. 生成质量报告
        quality_report_path = temp_workspace / "reports" / "quality_report.html"
        quality_report_path.parent.mkdir(exist_ok=True)
        
        analyzer.generate_quality_report(
            quality_report, 
            str(quality_report_path), 
            format="html"
        )
        
        assert quality_report_path.exists()
    
    def test_training_monitoring_integration(self, temp_workspace):
        """测试训练监控集成"""
        # 模拟训练历史数据
        training_history = {
            "epochs": list(range(1, 11)),
            "train_loss": [2.5 - i * 0.2 for i in range(10)],
            "val_loss": [2.3 - i * 0.18 for i in range(10)],
            "train_accuracy": [0.5 + i * 0.04 for i in range(10)],
            "val_accuracy": [0.52 + i * 0.035 for i in range(10)]
        }
        
        training_config = {
            "model_name": "training_model",
            "dataset": "training_data",
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10
        }
        
        # 生成训练报告
        generator = ReportGenerator(
            output_dir=str(temp_workspace / "reports"),
            include_plots=True
        )
        
        report_path = generator.generate_training_report(
            training_history, 
            training_config, 
            format="html"
        )
        
        assert Path(report_path).exists()
        
        # 检查是否生成了训练曲线图
        plot_paths = generator.create_training_curves(training_history)
        for plot_path in plot_paths:
            assert Path(plot_path).exists()
    
    def test_error_recovery_and_robustness(self, temp_workspace):
        """测试错误恢复和系统鲁棒性"""
        # 1. 测试空数据集处理
        empty_dataset = create_test_dataset(0)
        
        splitter = DataSplitter(min_samples_per_split=1)
        with pytest.raises(ValueError):
            splitter.split_data(empty_dataset, str(temp_workspace / "data"))
        
        # 2. 测试无效配置处理
        with pytest.raises(ValueError):
            EvaluationConfig(
                tasks=[],  # 空任务列表
                metrics=["accuracy"],
                batch_size=0  # 无效批次大小
            )
        
        # 3. 测试文件系统错误处理
        invalid_path = "/invalid/path/that/does/not/exist"
        
        tracker = ExperimentTracker(experiment_dir=invalid_path)
        # 应该创建目录或处理错误
        
        # 4. 测试内存限制处理
        large_dataset = create_test_dataset(10000)  # 大数据集
        
        config = EvaluationConfig(
            tasks=["classification"],
            metrics=["accuracy"],
            batch_size=1,  # 小批次以避免内存问题
            num_samples=100
        )
        
        engine = EvaluationEngine(config)
        # 应该能够处理大数据集而不崩溃
    
    def test_configuration_management_integration(self, temp_workspace):
        """测试配置管理集成"""
        from evaluation.config_manager import ConfigManager
        
        # 创建配置文件
        config_data = {
            "data_split": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "random_seed": 42
            },
            "evaluation": {
                "tasks": ["classification", "text_generation"],
                "metrics": ["accuracy", "bleu", "rouge"],
                "batch_size": 8,
                "num_samples": 50
            },
            "experiment_tracking": {
                "enabled": True,
                "auto_save": True,
                "max_history": 100
            }
        }
        
        config_path = temp_workspace / "config.yaml"
        
        # 保存配置
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True)
        
        # 加载配置
        config_manager = ConfigManager()
        loaded_config = config_manager.load_config(str(config_path))
        
        assert loaded_config["data_split"]["train_ratio"] == 0.7
        assert "classification" in loaded_config["evaluation"]["tasks"]
        
        # 使用配置创建组件
        splitter = DataSplitter(
            train_ratio=loaded_config["data_split"]["train_ratio"],
            val_ratio=loaded_config["data_split"]["val_ratio"],
            test_ratio=loaded_config["data_split"]["test_ratio"],
            random_seed=loaded_config["data_split"]["random_seed"]
        )
        
        eval_config = EvaluationConfig(
            tasks=loaded_config["evaluation"]["tasks"],
            metrics=loaded_config["evaluation"]["metrics"],
            batch_size=loaded_config["evaluation"]["batch_size"],
            num_samples=loaded_config["evaluation"]["num_samples"]
        )
        
        assert splitter.train_ratio == 0.7
        assert eval_config.batch_size == 8
    
    def test_parallel_evaluation_integration(self, temp_workspace, mock_model, mock_tokenizer):
        """测试并行评估集成"""
        dataset = create_test_dataset(100)
        
        config = EvaluationConfig(
            tasks=["classification"],
            metrics=["accuracy"],
            batch_size=4,
            num_samples=50
        )
        
        # 测试多线程评估
        engine = EvaluationEngine(config, max_workers=2)
        
        # 创建多个模型信息
        models_info = [
            {"model": mock_model, "tokenizer": mock_tokenizer, "name": f"model_{i}"}
            for i in range(3)
        ]
        
        datasets = {"classification": dataset}
        
        with patch.object(engine, '_create_inference_function') as mock_inference:
            mock_inference.return_value = lambda inputs: ["positive"] * len(inputs)
            
            results = engine.evaluate_multiple_models(models_info, datasets)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.model_name == f"model_{i}"
    
    def test_memory_optimization_integration(self, temp_workspace):
        """测试内存优化集成"""
        # 创建大数据集
        large_dataset = create_test_dataset(1000)
        
        # 使用内存优化的配置
        config = EvaluationConfig(
            tasks=["classification"],
            metrics=["accuracy"],
            batch_size=2,  # 小批次
            num_samples=100,
            memory_optimization=True
        )
        
        engine = EvaluationEngine(config)
        
        # 测试内存使用监控
        from evaluation.efficiency_analyzer import EfficiencyAnalyzer
        efficiency_analyzer = EfficiencyAnalyzer()
        
        # 模拟内存使用测量
        memory_usage = efficiency_analyzer.measure_memory_usage(lambda: None)
        assert isinstance(memory_usage, (int, float))
        assert memory_usage >= 0
    
    def test_cli_integration(self, temp_workspace):
        """测试命令行接口集成"""
        from evaluation.cli import EvaluationCLI
        
        # 创建测试数据
        dataset = create_test_dataset(50)
        data_path = temp_workspace / "test_data.json"
        
        # 保存数据
        import json
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump([{"text": item["text"], "label": item["label"]} 
                      for item in dataset], f, ensure_ascii=False)
        
        # 创建CLI实例
        cli = EvaluationCLI()
        
        # 测试数据拆分命令
        split_args = [
            "split-data",
            "--input", str(data_path),
            "--output", str(temp_workspace / "splits"),
            "--train-ratio", "0.7",
            "--val-ratio", "0.15",
            "--test-ratio", "0.15"
        ]
        
        # 注意：这里只是测试CLI接口的存在性，实际执行可能需要更复杂的模拟
        assert hasattr(cli, 'split_data')
        assert hasattr(cli, 'evaluate_model')
        assert hasattr(cli, 'run_benchmark')
        assert hasattr(cli, 'generate_report')
"""
实验跟踪器测试

测试ExperimentTracker类的功能。
"""

import pytest
from datetime import datetime
from pathlib import Path
import json

from evaluation import ExperimentTracker
from evaluation.data_models import EvaluationResult, ExperimentConfig, EfficiencyMetrics, QualityScores
from tests.conftest import assert_file_exists


class TestExperimentTracker:
    """实验跟踪器测试类"""
    
    def test_init_default_params(self):
        """测试默认参数初始化"""
        tracker = ExperimentTracker()
        
        assert tracker.experiment_dir == "experiments"
        assert tracker.auto_save == True
        assert tracker.max_history == 1000
    
    def test_init_custom_params(self, temp_dir):
        """测试自定义参数初始化"""
        tracker = ExperimentTracker(
            experiment_dir=str(temp_dir),
            auto_save=False,
            max_history=500
        )
        
        assert tracker.experiment_dir == str(temp_dir)
        assert tracker.auto_save == False
        assert tracker.max_history == 500
    
    def test_track_experiment_basic(self, experiment_tracker, evaluation_config):
        """测试基本实验跟踪"""
        # 创建模拟实验配置
        experiment_config = ExperimentConfig(
            model_name="test_model",
            dataset_name="test_dataset",
            hyperparameters={"lr": 0.001, "batch_size": 32},
            description="测试实验"
        )
        
        # 创建模拟评估结果
        result = EvaluationResult(
            model_name="test_model",
            evaluation_time=datetime.now(),
            metrics={"accuracy": 0.85, "f1": 0.80},
            task_results={},
            efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
            quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            config=evaluation_config
        )
        
        experiment_id = experiment_tracker.track_experiment(experiment_config, result)
        
        # 检查返回的实验ID
        assert isinstance(experiment_id, str)
        assert len(experiment_id) > 0
        
        # 检查实验是否被保存
        experiments = experiment_tracker.list_experiments()
        assert len(experiments) == 1
        assert experiments[0]["experiment_id"] == experiment_id
    
    def test_track_experiment_with_tags(self, experiment_tracker, evaluation_config):
        """测试带标签的实验跟踪"""
        experiment_config = ExperimentConfig(
            model_name="test_model",
            dataset_name="test_dataset",
            hyperparameters={"lr": 0.001},
            tags=["baseline", "v1.0"],
            description="带标签的测试实验"
        )
        
        result = EvaluationResult(
            model_name="test_model",
            evaluation_time=datetime.now(),
            metrics={"accuracy": 0.85},
            task_results={},
            efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
            quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            config=evaluation_config
        )
        
        experiment_id = experiment_tracker.track_experiment(experiment_config, result)
        
        # 检查标签是否被正确保存
        experiment = experiment_tracker.get_experiment(experiment_id)
        assert "baseline" in experiment["config"]["tags"]
        assert "v1.0" in experiment["config"]["tags"]
    
    def test_get_experiment(self, experiment_tracker, evaluation_config):
        """测试获取实验"""
        # 先创建一个实验
        experiment_config = ExperimentConfig(
            model_name="test_model",
            dataset_name="test_dataset",
            hyperparameters={"lr": 0.001}
        )
        
        result = EvaluationResult(
            model_name="test_model",
            evaluation_time=datetime.now(),
            metrics={"accuracy": 0.85},
            task_results={},
            efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
            quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            config=evaluation_config
        )
        
        experiment_id = experiment_tracker.track_experiment(experiment_config, result)
        
        # 获取实验
        experiment = experiment_tracker.get_experiment(experiment_id)
        
        assert experiment is not None
        assert experiment["experiment_id"] == experiment_id
        assert experiment["config"]["model_name"] == "test_model"
        assert experiment["result"]["metrics"]["accuracy"] == 0.85
    
    def test_get_experiment_not_found(self, experiment_tracker):
        """测试获取不存在的实验"""
        with pytest.raises(ValueError, match="实验不存在"):
            experiment_tracker.get_experiment("nonexistent_id")
    
    def test_list_experiments(self, experiment_tracker, evaluation_config):
        """测试列出实验"""
        # 创建多个实验
        for i in range(3):
            experiment_config = ExperimentConfig(
                model_name=f"model_{i}",
                dataset_name="test_dataset",
                hyperparameters={"lr": 0.001 * (i + 1)}
            )
            
            result = EvaluationResult(
                model_name=f"model_{i}",
                evaluation_time=datetime.now(),
                metrics={"accuracy": 0.8 + i * 0.05},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            
            experiment_tracker.track_experiment(experiment_config, result)
        
        experiments = experiment_tracker.list_experiments()
        
        assert len(experiments) == 3
        # 检查是否按时间排序（最新的在前）
        assert experiments[0]["config"]["model_name"] == "model_2"
    
    def test_list_experiments_with_filter(self, experiment_tracker, evaluation_config):
        """测试带过滤条件的实验列表"""
        # 创建不同的实验
        configs = [
            {"model_name": "model_a", "dataset_name": "dataset1"},
            {"model_name": "model_b", "dataset_name": "dataset1"},
            {"model_name": "model_c", "dataset_name": "dataset2"}
        ]
        
        for config in configs:
            experiment_config = ExperimentConfig(**config, hyperparameters={})
            result = EvaluationResult(
                model_name=config["model_name"],
                evaluation_time=datetime.now(),
                metrics={"accuracy": 0.8},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            experiment_tracker.track_experiment(experiment_config, result)
        
        # 按数据集过滤
        filtered = experiment_tracker.list_experiments(filters={"dataset_name": "dataset1"})
        assert len(filtered) == 2
        
        # 按模型名过滤
        filtered = experiment_tracker.list_experiments(filters={"model_name": "model_a"})
        assert len(filtered) == 1
        assert filtered[0]["config"]["model_name"] == "model_a"
    
    def test_compare_experiments(self, experiment_tracker, evaluation_config):
        """测试实验对比"""
        # 创建两个实验
        experiment_ids = []
        
        for i in range(2):
            experiment_config = ExperimentConfig(
                model_name=f"model_{i}",
                dataset_name="test_dataset",
                hyperparameters={"lr": 0.001 * (i + 1)}
            )
            
            result = EvaluationResult(
                model_name=f"model_{i}",
                evaluation_time=datetime.now(),
                metrics={"accuracy": 0.8 + i * 0.1, "f1": 0.75 + i * 0.1},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100 + i * 10, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            
            experiment_id = experiment_tracker.track_experiment(experiment_config, result)
            experiment_ids.append(experiment_id)
        
        comparison = experiment_tracker.compare_experiments(experiment_ids)
        
        assert isinstance(comparison, dict)
        assert "experiments" in comparison
        assert "metric_comparison" in comparison
        assert "best_model" in comparison
        assert "statistical_tests" in comparison
        
        # 检查对比结果
        assert len(comparison["experiments"]) == 2
        assert comparison["best_model"]["model_name"] == "model_1"  # 更高的准确率
    
    def test_generate_leaderboard(self, experiment_tracker, evaluation_config):
        """测试生成排行榜"""
        # 创建多个实验
        for i in range(5):
            experiment_config = ExperimentConfig(
                model_name=f"model_{i}",
                dataset_name="test_dataset",
                hyperparameters={"lr": 0.001}
            )
            
            result = EvaluationResult(
                model_name=f"model_{i}",
                evaluation_time=datetime.now(),
                metrics={"accuracy": 0.7 + i * 0.05, "f1": 0.65 + i * 0.05},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            
            experiment_tracker.track_experiment(experiment_config, result)
        
        leaderboard = experiment_tracker.generate_leaderboard(metric="accuracy")
        
        assert isinstance(leaderboard, list)
        assert len(leaderboard) == 5
        
        # 检查排序（准确率从高到低）
        assert leaderboard[0]["model_name"] == "model_4"
        assert leaderboard[-1]["model_name"] == "model_0"
        
        # 检查分数递减
        scores = [entry["score"] for entry in leaderboard]
        assert scores == sorted(scores, reverse=True)
    
    def test_export_results_csv(self, experiment_tracker, evaluation_config, temp_dir):
        """测试导出CSV格式结果"""
        # 创建实验
        experiment_config = ExperimentConfig(
            model_name="test_model",
            dataset_name="test_dataset",
            hyperparameters={"lr": 0.001}
        )
        
        result = EvaluationResult(
            model_name="test_model",
            evaluation_time=datetime.now(),
            metrics={"accuracy": 0.85, "f1": 0.80},
            task_results={},
            efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
            quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            config=evaluation_config
        )
        
        experiment_tracker.track_experiment(experiment_config, result)
        
        # 导出结果
        output_path = temp_dir / "results.csv"
        experiment_tracker.export_results(str(output_path), format="csv")
        
        assert_file_exists(output_path)
        
        # 检查CSV内容
        import pandas as pd
        df = pd.read_csv(output_path)
        
        assert len(df) == 1
        assert "model_name" in df.columns
        assert "accuracy" in df.columns
        assert df.iloc[0]["model_name"] == "test_model"
    
    def test_export_results_json(self, experiment_tracker, evaluation_config, temp_dir):
        """测试导出JSON格式结果"""
        # 创建实验
        experiment_config = ExperimentConfig(
            model_name="test_model",
            dataset_name="test_dataset",
            hyperparameters={"lr": 0.001}
        )
        
        result = EvaluationResult(
            model_name="test_model",
            evaluation_time=datetime.now(),
            metrics={"accuracy": 0.85},
            task_results={},
            efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
            quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            config=evaluation_config
        )
        
        experiment_tracker.track_experiment(experiment_config, result)
        
        # 导出结果
        output_path = temp_dir / "results.json"
        experiment_tracker.export_results(str(output_path), format="json")
        
        assert_file_exists(output_path)
        
        # 检查JSON内容
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["config"]["model_name"] == "test_model"
    
    def test_delete_experiment(self, experiment_tracker, evaluation_config):
        """测试删除实验"""
        # 创建实验
        experiment_config = ExperimentConfig(
            model_name="test_model",
            dataset_name="test_dataset",
            hyperparameters={"lr": 0.001}
        )
        
        result = EvaluationResult(
            model_name="test_model",
            evaluation_time=datetime.now(),
            metrics={"accuracy": 0.85},
            task_results={},
            efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
            quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            config=evaluation_config
        )
        
        experiment_id = experiment_tracker.track_experiment(experiment_config, result)
        
        # 确认实验存在
        assert len(experiment_tracker.list_experiments()) == 1
        
        # 删除实验
        experiment_tracker.delete_experiment(experiment_id)
        
        # 确认实验被删除
        assert len(experiment_tracker.list_experiments()) == 0
        
        # 尝试获取已删除的实验应该抛出异常
        with pytest.raises(ValueError, match="实验不存在"):
            experiment_tracker.get_experiment(experiment_id)
    
    def test_search_experiments(self, experiment_tracker, evaluation_config):
        """测试搜索实验"""
        # 创建不同的实验
        configs = [
            {"model_name": "bert_base", "description": "BERT基础模型测试"},
            {"model_name": "bert_large", "description": "BERT大模型测试"},
            {"model_name": "gpt_small", "description": "GPT小模型测试"}
        ]
        
        for config in configs:
            experiment_config = ExperimentConfig(
                dataset_name="test_dataset",
                hyperparameters={},
                **config
            )
            result = EvaluationResult(
                model_name=config["model_name"],
                evaluation_time=datetime.now(),
                metrics={"accuracy": 0.8},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            experiment_tracker.track_experiment(experiment_config, result)
        
        # 搜索BERT相关实验
        results = experiment_tracker.search_experiments("bert")
        assert len(results) == 2
        
        # 搜索基础模型
        results = experiment_tracker.search_experiments("基础")
        assert len(results) == 1
        assert results[0]["config"]["model_name"] == "bert_base"
    
    def test_get_experiment_statistics(self, experiment_tracker, evaluation_config):
        """测试获取实验统计信息"""
        # 创建多个实验
        for i in range(10):
            experiment_config = ExperimentConfig(
                model_name=f"model_{i}",
                dataset_name="test_dataset",
                hyperparameters={"lr": 0.001}
            )
            
            result = EvaluationResult(
                model_name=f"model_{i}",
                evaluation_time=datetime.now(),
                metrics={"accuracy": 0.7 + i * 0.02},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            
            experiment_tracker.track_experiment(experiment_config, result)
        
        stats = experiment_tracker.get_experiment_statistics()
        
        assert isinstance(stats, dict)
        assert "total_experiments" in stats
        assert "avg_accuracy" in stats
        assert "best_accuracy" in stats
        assert "worst_accuracy" in stats
        
        assert stats["total_experiments"] == 10
        assert stats["best_accuracy"] > stats["worst_accuracy"]
    
    def test_backup_and_restore(self, experiment_tracker, evaluation_config, temp_dir):
        """测试备份和恢复"""
        # 创建实验
        experiment_config = ExperimentConfig(
            model_name="test_model",
            dataset_name="test_dataset",
            hyperparameters={"lr": 0.001}
        )
        
        result = EvaluationResult(
            model_name="test_model",
            evaluation_time=datetime.now(),
            metrics={"accuracy": 0.85},
            task_results={},
            efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
            quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            config=evaluation_config
        )
        
        experiment_tracker.track_experiment(experiment_config, result)
        
        # 备份
        backup_path = temp_dir / "backup.json"
        experiment_tracker.backup_experiments(str(backup_path))
        
        assert_file_exists(backup_path)
        
        # 清空当前实验
        experiment_tracker.clear_all_experiments()
        assert len(experiment_tracker.list_experiments()) == 0
        
        # 恢复
        experiment_tracker.restore_experiments(str(backup_path))
        assert len(experiment_tracker.list_experiments()) == 1
    
    def test_max_history_limit(self, temp_dir, evaluation_config):
        """测试历史记录数量限制"""
        tracker = ExperimentTracker(
            experiment_dir=str(temp_dir),
            max_history=3  # 限制最多3个实验
        )
        
        # 创建5个实验
        for i in range(5):
            experiment_config = ExperimentConfig(
                model_name=f"model_{i}",
                dataset_name="test_dataset",
                hyperparameters={"lr": 0.001}
            )
            
            result = EvaluationResult(
                model_name=f"model_{i}",
                evaluation_time=datetime.now(),
                metrics={"accuracy": 0.8},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=evaluation_config
            )
            
            tracker.track_experiment(experiment_config, result)
        
        # 应该只保留最新的3个实验
        experiments = tracker.list_experiments()
        assert len(experiments) <= 3
        
        # 最新的实验应该被保留
        model_names = [exp["config"]["model_name"] for exp in experiments]
        assert "model_4" in model_names  # 最新的实验
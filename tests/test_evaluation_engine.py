"""
评估引擎测试

测试EvaluationEngine类的功能。
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from evaluation import EvaluationEngine, EvaluationConfig
from evaluation.data_models import EvaluationResult


class TestEvaluationEngine:
    """评估引擎测试类"""
    
    def test_init_default_params(self, evaluation_config):
        """测试默认参数初始化"""
        engine = EvaluationEngine(evaluation_config)
        
        assert engine.config == evaluation_config
        assert engine.device == "cpu"
        assert engine.max_workers == 4
        assert engine.metrics_calculator is not None
        assert engine.efficiency_analyzer is not None
    
    def test_init_custom_params(self, evaluation_config):
        """测试自定义参数初始化"""
        engine = EvaluationEngine(
            config=evaluation_config,
            device="cuda",
            max_workers=8
        )
        
        assert engine.device == "cuda"
        assert engine.max_workers == 8
    
    def test_evaluate_model_basic(self, evaluation_config, mock_model, mock_tokenizer, sample_dataset):
        """测试基本模型评估"""
        engine = EvaluationEngine(evaluation_config)
        datasets = {"test_task": sample_dataset}
        
        # 模拟推理函数
        with patch.object(engine, '_create_inference_function') as mock_inference:
            mock_inference.return_value = lambda inputs: ["预测结果"] * len(inputs)
            
            result = engine.evaluate_model(
                model=mock_model,
                tokenizer=mock_tokenizer,
                datasets=datasets,
                model_name="test_model"
            )
        
        # 检查结果类型和基本字段
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "test_model"
        assert isinstance(result.evaluation_time, datetime)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.task_results, dict)
    
    def test_evaluate_multiple_models(self, evaluation_config, mock_model, mock_tokenizer, sample_dataset):
        """测试多模型评估"""
        engine = EvaluationEngine(evaluation_config, max_workers=1)  # 使用单线程避免复杂性
        
        models_info = [
            {"model": mock_model, "tokenizer": mock_tokenizer, "name": "model1"},
            {"model": mock_model, "tokenizer": mock_tokenizer, "name": "model2"}
        ]
        datasets = {"test_task": sample_dataset}
        
        with patch.object(engine, '_create_inference_function') as mock_inference:
            mock_inference.return_value = lambda inputs: ["预测结果"] * len(inputs)
            
            results = engine.evaluate_multiple_models(models_info, datasets)
        
        # 检查结果
        assert len(results) == 2
        assert all(isinstance(result, EvaluationResult) for result in results)
        assert results[0].model_name == "model1"
        assert results[1].model_name == "model2"
    
    def test_prepare_inputs_text_generation(self, evaluation_config):
        """测试文本生成任务的输入准备"""
        engine = EvaluationEngine(evaluation_config)
        
        batch = {"text": ["输入文本1", "输入文本2"]}
        inputs = engine._prepare_inputs(batch, "text_generation")
        
        assert inputs == ["输入文本1", "输入文本2"]
    
    def test_prepare_inputs_question_answering(self, evaluation_config):
        """测试问答任务的输入准备"""
        engine = EvaluationEngine(evaluation_config)
        
        batch = {
            "question": ["问题1", "问题2"],
            "context": ["上下文1", "上下文2"]
        }
        inputs = engine._prepare_inputs(batch, "question_answering")
        
        expected = [
            "问题: 问题1\n上下文: 上下文1",
            "问题: 问题2\n上下文: 上下文2"
        ]
        assert inputs == expected
    
    def test_get_task_type(self, evaluation_config):
        """测试任务类型识别"""
        engine = EvaluationEngine(evaluation_config)
        
        assert engine._get_task_type("text_generation") == "generation"
        assert engine._get_task_type("classification_task") == "classification"
        assert engine._get_task_type("similarity_task") == "similarity"
        assert engine._get_task_type("unknown_task") == "generation"  # 默认
    
    def test_calculate_overall_metrics(self, evaluation_config):
        """测试整体指标计算"""
        engine = EvaluationEngine(evaluation_config)
        
        # 创建模拟任务结果
        from evaluation.data_models import TaskResult
        task_results = {
            "task1": TaskResult(
                task_name="task1",
                predictions=[],
                references=[],
                metrics={"accuracy": 0.8, "f1": 0.75},
                samples=[],
                execution_time=1.0
            ),
            "task2": TaskResult(
                task_name="task2", 
                predictions=[],
                references=[],
                metrics={"accuracy": 0.9, "f1": 0.85},
                samples=[],
                execution_time=1.5
            )
        }
        
        overall_metrics = engine._calculate_overall_metrics(task_results)
        
        # 检查平均指标
        assert "avg_accuracy" in overall_metrics
        assert "avg_f1" in overall_metrics
        assert "std_accuracy" in overall_metrics
        assert "std_f1" in overall_metrics
        assert "overall_score" in overall_metrics
        
        # 检查计算正确性
        assert overall_metrics["avg_accuracy"] == 0.85  # (0.8 + 0.9) / 2
        assert overall_metrics["avg_f1"] == 0.8  # (0.75 + 0.85) / 2
    
    def test_calculate_overall_metrics_empty(self, evaluation_config):
        """测试空任务结果的整体指标计算"""
        engine = EvaluationEngine(evaluation_config)
        
        overall_metrics = engine._calculate_overall_metrics({})
        
        assert overall_metrics == {}
    
    def test_evaluation_history(self, evaluation_config, mock_model, mock_tokenizer, sample_dataset):
        """测试评估历史记录"""
        engine = EvaluationEngine(evaluation_config)
        datasets = {"test_task": sample_dataset}
        
        # 初始历史应该为空
        assert len(engine.get_evaluation_history()) == 0
        
        with patch.object(engine, '_create_inference_function') as mock_inference:
            mock_inference.return_value = lambda inputs: ["预测结果"] * len(inputs)
            
            # 执行评估
            result = engine.evaluate_model(mock_model, mock_tokenizer, datasets, "test_model")
        
        # 检查历史记录
        history = engine.get_evaluation_history()
        assert len(history) == 1
        assert history[0] == result
        
        # 清空历史
        engine.clear_evaluation_history()
        assert len(engine.get_evaluation_history()) == 0
    
    def test_save_evaluation_result(self, evaluation_config, temp_dir):
        """测试保存评估结果"""
        engine = EvaluationEngine(evaluation_config)
        
        # 创建模拟评估结果
        from evaluation.data_models import EfficiencyMetrics, QualityScores
        
        result = EvaluationResult(
            model_name="test_model",
            evaluation_time=datetime.now(),
            metrics={"accuracy": 0.8},
            task_results={},
            efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
            quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            config=evaluation_config
        )
        
        output_path = temp_dir / "test_result.json"
        engine.save_evaluation_result(result, str(output_path))
        
        # 检查文件是否创建
        assert output_path.exists()
        
        # 检查文件内容
        import json
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert "model_name" in saved_data
        assert saved_data["model_name"] == "test_model"
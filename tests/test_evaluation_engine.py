"""
测试评估引擎主控制器
"""

import pytest
import time
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from industry_evaluation.core.evaluation_engine import (
    EvaluationOrchestrator,
    IndustryEvaluationEngine,
    EvaluationTask,
    EvaluationStatus
)
from industry_evaluation.core.batch_evaluator import (
    BatchEvaluator,
    BatchEvaluationConfig,
    DatasetLoader,
    BatchResultManager
)
from industry_evaluation.core.interfaces import EvaluationConfig, EvaluationResult, SampleResult
from industry_evaluation.adapters.model_adapter import ModelManager


class MockEvaluator:
    """模拟评估器"""
    
    def __init__(self, dimension_name: str, score: float = 0.8):
        self.dimension_name = dimension_name
        self.score = score
    
    def evaluate(self, input_text: str, model_output: str, expected_output: str, context: dict):
        mock_score = Mock()
        mock_score.score = self.score
        mock_score.error_types = []
        mock_score.explanation = f"{self.dimension_name}评估结果"
        return mock_score


class MockModelAdapter:
    """模拟模型适配器"""
    
    def __init__(self, model_id: str, response: str = "模拟响应"):
        self.model_id = model_id
        self.response = response
    
    def predict(self, input_text: str, context=None):
        return f"{self.response}: {input_text}"
    
    def is_available(self):
        return True


class TestEvaluationOrchestrator:
    """测试评估编排器"""
    
    def setup_method(self):
        """设置测试环境"""
        # 创建模拟组件
        self.model_manager = ModelManager()
        self.model_manager.adapters["test_model"] = MockModelAdapter("test_model")
        
        self.evaluators = {
            "knowledge": MockEvaluator("knowledge", 0.8),
            "terminology": MockEvaluator("terminology", 0.7)
        }
        
        self.result_aggregator = Mock()
        self.result_aggregator.aggregate_results.return_value = EvaluationResult(
            overall_score=0.75,
            dimension_scores={"knowledge": 0.8, "terminology": 0.7},
            detailed_results=[],
            error_analysis=None,
            improvement_suggestions=[]
        )
        
        self.report_generator = Mock()
        
        self.orchestrator = EvaluationOrchestrator(
            model_manager=self.model_manager,
            evaluators=self.evaluators,
            result_aggregator=self.result_aggregator,
            report_generator=self.report_generator,
            max_workers=2
        )
    
    def test_create_evaluation_task(self):
        """测试创建评估任务"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.6, "terminology": 0.4},
            threshold_config={"knowledge": 0.7, "terminology": 0.6}
        )
        
        dataset = [
            {"input": "测试输入1", "expected_output": "期望输出1"},
            {"input": "测试输入2", "expected_output": "期望输出2"}
        ]
        
        task = self.orchestrator.create_evaluation_task(
            task_id="test_task",
            model_id="test_model",
            dataset=dataset,
            config=config
        )
        
        assert task.task_id == "test_task"
        assert task.model_id == "test_model"
        assert task.status == EvaluationStatus.PENDING
        assert len(task.dataset) == 2
        assert task.task_id in self.orchestrator.tasks
    
    def test_start_evaluation_success(self):
        """测试成功启动评估"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.7}
        )
        
        dataset = [{"input": "测试输入", "expected_output": "期望输出"}]
        
        task = self.orchestrator.create_evaluation_task(
            task_id="test_task",
            model_id="test_model",
            dataset=dataset,
            config=config
        )
        
        success = self.orchestrator.start_evaluation("test_task")
        
        assert success == True
        assert task.status == EvaluationStatus.RUNNING
        assert task.started_at is not None
    
    def test_start_evaluation_model_not_found(self):
        """测试模型不存在时启动评估"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.7}
        )
        
        dataset = [{"input": "测试输入", "expected_output": "期望输出"}]
        
        task = self.orchestrator.create_evaluation_task(
            task_id="test_task",
            model_id="nonexistent_model",
            dataset=dataset,
            config=config
        )
        
        success = self.orchestrator.start_evaluation("test_task")
        
        assert success == False
        assert task.status == EvaluationStatus.FAILED
        assert "模型不存在" in task.error
    
    def test_get_task_status(self):
        """测试获取任务状态"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.7}
        )
        
        dataset = [{"input": "测试输入", "expected_output": "期望输出"}]
        
        task = self.orchestrator.create_evaluation_task(
            task_id="test_task",
            model_id="test_model",
            dataset=dataset,
            config=config
        )
        
        retrieved_task = self.orchestrator.get_task_status("test_task")
        
        assert retrieved_task is not None
        assert retrieved_task.task_id == "test_task"
        assert retrieved_task == task
    
    def test_cancel_task(self):
        """测试取消任务"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.7}
        )
        
        dataset = [{"input": "测试输入", "expected_output": "期望输出"}]
        
        task = self.orchestrator.create_evaluation_task(
            task_id="test_task",
            model_id="test_model",
            dataset=dataset,
            config=config
        )
        
        success = self.orchestrator.cancel_task("test_task")
        
        assert success == True
        assert task.status == EvaluationStatus.CANCELLED
        assert task.completed_at is not None
    
    def test_list_tasks(self):
        """测试列出任务"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.7}
        )
        
        dataset = [{"input": "测试输入", "expected_output": "期望输出"}]
        
        # 创建多个任务
        task1 = self.orchestrator.create_evaluation_task(
            task_id="task1",
            model_id="test_model",
            dataset=dataset,
            config=config
        )
        
        task2 = self.orchestrator.create_evaluation_task(
            task_id="task2",
            model_id="test_model",
            dataset=dataset,
            config=config
        )
        
        # 列出所有任务
        all_tasks = self.orchestrator.list_tasks()
        assert len(all_tasks) == 2
        
        # 按状态过滤
        pending_tasks = self.orchestrator.list_tasks(EvaluationStatus.PENDING)
        assert len(pending_tasks) == 2
        
        # 取消一个任务后再次过滤
        self.orchestrator.cancel_task("task1")
        pending_tasks = self.orchestrator.list_tasks(EvaluationStatus.PENDING)
        assert len(pending_tasks) == 1
        
        cancelled_tasks = self.orchestrator.list_tasks(EvaluationStatus.CANCELLED)
        assert len(cancelled_tasks) == 1
    
    def test_callback_registration(self):
        """测试回调函数注册"""
        callback_called = False
        
        def test_callback(task):
            nonlocal callback_called
            callback_called = True
        
        self.orchestrator.register_callback("on_task_start", test_callback)
        
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.7}
        )
        
        dataset = [{"input": "测试输入", "expected_output": "期望输出"}]
        
        task = self.orchestrator.create_evaluation_task(
            task_id="test_task",
            model_id="test_model",
            dataset=dataset,
            config=config
        )
        
        self.orchestrator.start_evaluation("test_task")
        
        # 等待一小段时间让回调执行
        time.sleep(0.1)
        
        assert callback_called == True


class TestIndustryEvaluationEngine:
    """测试行业评估引擎"""
    
    def setup_method(self):
        """设置测试环境"""
        self.model_manager = ModelManager()
        self.model_manager.adapters["test_model"] = MockModelAdapter("test_model")
        
        self.evaluators = {
            "knowledge": MockEvaluator("knowledge", 0.8)
        }
        
        self.result_aggregator = Mock()
        self.result_aggregator.aggregate_results.return_value = EvaluationResult(
            overall_score=0.8,
            dimension_scores={"knowledge": 0.8},
            detailed_results=[],
            error_analysis=None,
            improvement_suggestions=[]
        )
        
        self.report_generator = Mock()
        self.report_generator.generate_report.return_value = "测试报告"
        
        self.engine = IndustryEvaluationEngine(
            model_manager=self.model_manager,
            evaluators=self.evaluators,
            result_aggregator=self.result_aggregator,
            report_generator=self.report_generator
        )
    
    def test_evaluate_model(self):
        """测试评估模型"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.7}
        )
        
        dataset = [{"input": "测试输入", "expected_output": "期望输出"}]
        
        task_id = self.engine.evaluate_model(
            model_id="test_model",
            dataset=dataset,
            evaluation_config=config
        )
        
        assert task_id is not None
        assert isinstance(task_id, str)
    
    def test_get_evaluation_progress(self):
        """测试获取评估进度"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.7}
        )
        
        dataset = [{"input": "测试输入", "expected_output": "期望输出"}]
        
        task_id = self.engine.evaluate_model(
            model_id="test_model",
            dataset=dataset,
            evaluation_config=config
        )
        
        # 立即获取进度（可能为None，因为任务刚开始）
        progress = self.engine.get_evaluation_progress(task_id)
        # 进度可能为None或包含进度信息
        assert progress is None or hasattr(progress, 'status')
    
    def test_list_evaluations(self):
        """测试列出评估任务"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.7}
        )
        
        dataset = [{"input": "测试输入", "expected_output": "期望输出"}]
        
        # 创建评估任务
        task_id = self.engine.evaluate_model(
            model_id="test_model",
            dataset=dataset,
            evaluation_config=config
        )
        
        # 列出评估任务
        evaluations = self.engine.list_evaluations()
        
        assert len(evaluations) >= 1
        assert any(eval_info["task_id"] == task_id for eval_info in evaluations)
    
    def test_cancel_evaluation(self):
        """测试取消评估"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.7}
        )
        
        dataset = [{"input": "测试输入", "expected_output": "期望输出"}]
        
        task_id = self.engine.evaluate_model(
            model_id="test_model",
            dataset=dataset,
            evaluation_config=config
        )
        
        success = self.engine.cancel_evaluation(task_id)
        
        assert success == True


class TestDatasetLoader:
    """测试数据集加载器"""
    
    def test_load_json_dataset(self):
        """测试加载JSON数据集"""
        test_data = [
            {"input": "测试1", "expected_output": "输出1"},
            {"input": "测试2", "expected_output": "输出2"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            chunks = list(DatasetLoader.load_dataset(temp_path))
            
            assert len(chunks) == 1
            assert chunks[0] == test_data
        finally:
            import os
            os.unlink(temp_path)
    
    def test_load_jsonl_dataset(self):
        """测试加载JSONL数据集"""
        test_data = [
            {"input": "测试1", "expected_output": "输出1"},
            {"input": "测试2", "expected_output": "输出2"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_path = f.name
        
        try:
            chunks = list(DatasetLoader.load_dataset(temp_path))
            
            assert len(chunks) == 1
            assert chunks[0] == test_data
        finally:
            import os
            os.unlink(temp_path)
    
    def test_load_dataset_with_chunks(self):
        """测试分块加载数据集"""
        test_data = [
            {"input": f"测试{i}", "expected_output": f"输出{i}"}
            for i in range(5)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            chunks = list(DatasetLoader.load_dataset(temp_path, chunk_size=2))
            
            assert len(chunks) == 3  # 5个样本，每块2个，共3块
            assert len(chunks[0]) == 2
            assert len(chunks[1]) == 2
            assert len(chunks[2]) == 1
        finally:
            import os
            os.unlink(temp_path)
    
    def test_count_samples(self):
        """测试计算样本数量"""
        test_data = [
            {"input": "测试1", "expected_output": "输出1"},
            {"input": "测试2", "expected_output": "输出2"},
            {"input": "测试3", "expected_output": "输出3"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            count = DatasetLoader.count_samples(temp_path)
            assert count == 3
        finally:
            import os
            os.unlink(temp_path)


class TestBatchResultManager:
    """测试批量结果管理器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.result_manager = BatchResultManager(self.temp_dir)
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_intermediate_result(self):
        """测试保存和加载中间结果"""
        result = EvaluationResult(
            overall_score=0.8,
            dimension_scores={"knowledge": 0.8},
            detailed_results=[],
            error_analysis=None,
            improvement_suggestions=[]
        )
        
        # 保存中间结果
        self.result_manager.save_intermediate_result(
            task_id="test_task",
            model_id="test_model",
            batch_index=0,
            result=result
        )
        
        # 加载中间结果
        loaded_results = self.result_manager.load_intermediate_results(
            task_id="test_task",
            model_id="test_model"
        )
        
        assert len(loaded_results) == 1
        assert loaded_results[0].overall_score == 0.8


class TestBatchEvaluator:
    """测试批量评估器"""
    
    def setup_method(self):
        """设置测试环境"""
        # 创建模拟评估引擎
        self.evaluation_engine = Mock()
        self.evaluation_engine.evaluate_model.return_value = "mock_task_id"
        
        # 模拟进度和结果
        mock_progress = Mock()
        mock_progress.status = "completed"
        self.evaluation_engine.get_evaluation_progress.return_value = mock_progress
        
        mock_result = EvaluationResult(
            overall_score=0.8,
            dimension_scores={"knowledge": 0.8},
            detailed_results=[],
            error_analysis=None,
            improvement_suggestions=[]
        )
        self.evaluation_engine.get_evaluation_result.return_value = mock_result
        
        self.batch_evaluator = BatchEvaluator(self.evaluation_engine)
    
    def test_create_batch_task(self):
        """测试创建批量任务"""
        # 创建临时数据集文件
        test_data = [
            {"input": "测试1", "expected_output": "输出1"},
            {"input": "测试2", "expected_output": "输出2"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            config = EvaluationConfig(
                industry_domain="finance",
                evaluation_dimensions=["knowledge"],
                weight_config={"knowledge": 1.0},
                threshold_config={"knowledge": 0.7}
            )
            
            batch_config = BatchEvaluationConfig(
                batch_size=10,
                max_concurrent_tasks=2
            )
            
            batch_task = self.batch_evaluator.create_batch_task(
                task_id="batch_test",
                model_ids=["model1", "model2"],
                dataset_path=temp_path,
                evaluation_config=config,
                batch_config=batch_config
            )
            
            assert batch_task.task_id == "batch_test"
            assert batch_task.model_ids == ["model1", "model2"]
            assert batch_task.total_samples == 2
            assert batch_task.status == "pending"
        finally:
            import os
            os.unlink(temp_path)
    
    def test_get_batch_task_status(self):
        """测试获取批量任务状态"""
        # 创建临时数据集文件
        test_data = [{"input": "测试", "expected_output": "输出"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            config = EvaluationConfig(
                industry_domain="finance",
                evaluation_dimensions=["knowledge"],
                weight_config={"knowledge": 1.0},
                threshold_config={"knowledge": 0.7}
            )
            
            batch_config = BatchEvaluationConfig()
            
            batch_task = self.batch_evaluator.create_batch_task(
                task_id="batch_test",
                model_ids=["model1"],
                dataset_path=temp_path,
                evaluation_config=config,
                batch_config=batch_config
            )
            
            retrieved_task = self.batch_evaluator.get_batch_task_status("batch_test")
            
            assert retrieved_task is not None
            assert retrieved_task.task_id == "batch_test"
            assert retrieved_task == batch_task
        finally:
            import os
            os.unlink(temp_path)
    
    def test_cancel_batch_task(self):
        """测试取消批量任务"""
        # 创建临时数据集文件
        test_data = [{"input": "测试", "expected_output": "输出"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            config = EvaluationConfig(
                industry_domain="finance",
                evaluation_dimensions=["knowledge"],
                weight_config={"knowledge": 1.0},
                threshold_config={"knowledge": 0.7}
            )
            
            batch_config = BatchEvaluationConfig()
            
            batch_task = self.batch_evaluator.create_batch_task(
                task_id="batch_test",
                model_ids=["model1"],
                dataset_path=temp_path,
                evaluation_config=config,
                batch_config=batch_config
            )
            
            success = self.batch_evaluator.cancel_batch_task("batch_test")
            
            assert success == True
            assert batch_task.status == "cancelled"
            assert batch_task.completed_at is not None
        finally:
            import os
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])
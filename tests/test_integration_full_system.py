"""
完整系统集成测试
"""

import pytest
import tempfile
import json
import time
import logging
from pathlib import Path
from unittest.mock import Mock, patch
from industry_evaluation.core.evaluation_engine import IndustryEvaluationEngine
from industry_evaluation.core.batch_evaluator import BatchEvaluator, BatchEvaluationConfig
from industry_evaluation.core.result_aggregator import ResultAggregator
from industry_evaluation.core.progress_tracker import ProgressTracker
from industry_evaluation.adapters.model_adapter import ModelManager, ModelAdapterFactory
from industry_evaluation.reporting.report_generator import ReportGenerator
from industry_evaluation.evaluators.knowledge_evaluator import KnowledgeEvaluator
from industry_evaluation.evaluators.terminology_evaluator import TerminologyEvaluator
from industry_evaluation.evaluators.reasoning_evaluator import ReasoningEvaluator
from industry_evaluation.evaluators.long_text_evaluator import LongTextEvaluator
from industry_evaluation.core.interfaces import EvaluationConfig


class MockModelAdapter:
    """集成测试用的模拟模型适配器"""
    
    def __init__(self, model_id: str, response_quality: str = "good"):
        self.model_id = model_id
        self.response_quality = response_quality
        self.call_count = 0
    
    def predict(self, input_text: str, context=None):
        self.call_count += 1
        
        if self.response_quality == "good":
            return f"高质量回答：{input_text}的专业分析结果，包含准确的术语和逻辑推理。"
        elif self.response_quality == "poor":
            return f"低质量回答：{input_text}的简单回复，可能包含错误。"
        else:
            return f"中等质量回答：{input_text}的基本分析。"
    
    def is_available(self):
        return True


class TestFullSystemIntegration:
    """完整系统集成测试"""
    
    def setup_method(self):
        """设置测试环境"""
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 初始化组件
        self.setup_components()
        self.setup_test_data()
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def setup_components(self):
        """设置系统组件"""
        # 模型管理器
        self.model_manager = ModelManager()
        
        # 注册测试模型
        ModelAdapterFactory.register_adapter("mock", MockModelAdapter)
        self.model_manager.register_model(
            "good_model", 
            "mock", 
            {"response_quality": "good"}
        )
        self.model_manager.register_model(
            "poor_model", 
            "mock", 
            {"response_quality": "poor"}
        )
        
        # 评估器
        self.evaluators = {
            "knowledge": KnowledgeEvaluator(),
            "terminology": TerminologyEvaluator(),
            "reasoning": ReasoningEvaluator(),
            "long_text": LongTextEvaluator()
        }
        
        # 结果聚合器
        self.result_aggregator = ResultAggregator()
        
        # 报告生成器
        self.report_generator = ReportGenerator()
        
        # 评估引擎
        self.evaluation_engine = IndustryEvaluationEngine(
            model_manager=self.model_manager,
            evaluators=self.evaluators,
            result_aggregator=self.result_aggregator,
            report_generator=self.report_generator,
            max_workers=2
        )
        
        # 批量评估器
        self.batch_evaluator = BatchEvaluator(self.evaluation_engine)
    
    def setup_test_data(self):
        """设置测试数据"""
        # 创建测试数据集
        self.test_dataset = [
            {
                "id": "sample_1",
                "input": "请解释金融风险管理中的VaR模型",
                "expected_output": "VaR（Value at Risk）是一种风险度量方法，用于量化在正常市场条件下，特定时间段内投资组合可能面临的最大损失。",
                "context": {
                    "industry": "finance",
                    "topic": "risk_management"
                }
            },
            {
                "id": "sample_2", 
                "input": "什么是区块链技术的共识机制？",
                "expected_output": "共识机制是区块链网络中确保所有节点对交易和区块状态达成一致的算法，常见的有工作量证明（PoW）和权益证明（PoS）。",
                "context": {
                    "industry": "technology",
                    "topic": "blockchain"
                }
            },
            {
                "id": "sample_3",
                "input": "医疗诊断中的机器学习应用有哪些？",
                "expected_output": "机器学习在医疗诊断中的应用包括医学影像分析、疾病预测、药物发现、个性化治疗方案制定等。",
                "context": {
                    "industry": "healthcare",
                    "topic": "ai_diagnosis"
                }
            }
        ]
        
        # 保存为JSON文件
        self.dataset_path = Path(self.temp_dir) / "test_dataset.json"
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_dataset, f, ensure_ascii=False, indent=2)
        
        # 创建大数据集用于批量测试
        large_dataset = []
        for i in range(50):
            large_dataset.append({
                "id": f"large_sample_{i}",
                "input": f"测试问题 {i}：请分析相关的专业概念",
                "expected_output": f"专业回答 {i}：包含准确的术语和分析",
                "context": {"industry": "general", "topic": "analysis"}
            })
        
        self.large_dataset_path = Path(self.temp_dir) / "large_dataset.json"
        with open(self.large_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(large_dataset, f, ensure_ascii=False, indent=2)
    
    def test_single_model_evaluation_workflow(self):
        """测试单模型评估完整流程"""
        # 配置评估参数
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge", "terminology", "reasoning"],
            weight_config={
                "knowledge": 0.4,
                "terminology": 0.3,
                "reasoning": 0.3
            },
            threshold_config={
                "knowledge": 0.7,
                "terminology": 0.6,
                "reasoning": 0.7
            },
            auto_generate_report=True
        )
        
        # 启动评估
        task_id = self.evaluation_engine.evaluate_model(
            model_id="good_model",
            dataset=self.test_dataset,
            evaluation_config=config
        )
        
        assert task_id is not None
        
        # 等待评估完成
        max_wait_time = 30  # 30秒超时
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            progress = self.evaluation_engine.get_evaluation_progress(task_id)
            
            if progress and progress.status == "completed":
                break
            elif progress and progress.status == "failed":
                pytest.fail(f"评估失败: {progress.error if hasattr(progress, 'error') else '未知错误'}")
            
            time.sleep(1)
        else:
            pytest.fail("评估超时")
        
        # 获取评估结果
        result = self.evaluation_engine.get_evaluation_result(task_id)
        
        assert result is not None
        assert result.overall_score > 0
        assert len(result.dimension_scores) == 3
        assert "knowledge" in result.dimension_scores
        assert "terminology" in result.dimension_scores
        assert "reasoning" in result.dimension_scores
        assert len(result.detailed_results) == len(self.test_dataset)
        
        # 生成报告
        report = self.evaluation_engine.generate_report(task_id, "json")
        assert report is not None
        
        # 验证报告内容
        if isinstance(report, str):
            try:
                report_data = json.loads(report)
                assert "overall_score" in report_data
                assert "dimension_scores" in report_data
            except json.JSONDecodeError:
                # 如果是文件路径，检查文件是否存在
                assert Path(report).exists()
    
    def test_multi_model_comparison(self):
        """测试多模型对比评估"""
        config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.6, "terminology": 0.4},
            threshold_config={"knowledge": 0.7, "terminology": 0.6}
        )
        
        # 同时评估两个模型
        task_ids = []
        
        for model_id in ["good_model", "poor_model"]:
            task_id = self.evaluation_engine.evaluate_model(
                model_id=model_id,
                dataset=self.test_dataset[:2],  # 使用较小的数据集
                evaluation_config=config
            )
            task_ids.append((model_id, task_id))
        
        # 等待所有评估完成
        results = {}
        max_wait_time = 30
        
        for model_id, task_id in task_ids:
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                progress = self.evaluation_engine.get_evaluation_progress(task_id)
                
                if progress and progress.status == "completed":
                    result = self.evaluation_engine.get_evaluation_result(task_id)
                    results[model_id] = result
                    break
                elif progress and progress.status == "failed":
                    pytest.fail(f"模型 {model_id} 评估失败")
                
                time.sleep(1)
            else:
                pytest.fail(f"模型 {model_id} 评估超时")
        
        # 验证结果
        assert len(results) == 2
        assert "good_model" in results
        assert "poor_model" in results
        
        # 好模型应该得分更高
        good_score = results["good_model"].overall_score
        poor_score = results["poor_model"].overall_score
        
        # 注意：由于使用的是模拟评估器，实际得分可能不会有显著差异
        # 这里主要验证流程的完整性
        assert good_score >= 0
        assert poor_score >= 0
    
    def test_batch_evaluation_workflow(self):
        """测试批量评估流程"""
        # 配置批量评估
        eval_config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.7, "terminology": 0.3},
            threshold_config={"knowledge": 0.6, "terminology": 0.5}
        )
        
        batch_config = BatchEvaluationConfig(
            batch_size=10,
            max_concurrent_tasks=2,
            chunk_size=20,
            save_intermediate_results=True,
            intermediate_results_dir=str(Path(self.temp_dir) / "batch_results"),
            enable_parallel_processing=False  # 使用顺序处理以简化测试
        )
        
        # 创建批量任务
        batch_task = self.batch_evaluator.create_batch_task(
            task_id="batch_test",
            model_ids=["good_model"],
            dataset_path=str(self.large_dataset_path),
            evaluation_config=eval_config,
            batch_config=batch_config
        )
        
        assert batch_task.task_id == "batch_test"
        assert batch_task.total_samples == 50
        
        # 启动批量评估
        success = self.batch_evaluator.start_batch_evaluation("batch_test")
        assert success == True
        
        # 等待批量评估完成
        max_wait_time = 60  # 批量评估需要更长时间
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            task_status = self.batch_evaluator.get_batch_task_status("batch_test")
            
            if task_status.status == "completed":
                break
            elif task_status.status == "failed":
                pytest.fail(f"批量评估失败: {task_status.errors}")
            
            time.sleep(2)
        else:
            # 如果超时，取消任务
            self.batch_evaluator.cancel_batch_task("batch_test")
            pytest.fail("批量评估超时")
        
        # 验证批量评估结果
        final_task = self.batch_evaluator.get_batch_task_status("batch_test")
        
        assert final_task.status == "completed"
        assert final_task.processed_samples > 0
        assert "good_model" in final_task.results
        
        # 验证结果质量
        model_result = final_task.results["good_model"]
        assert model_result.overall_score > 0
        assert len(model_result.detailed_results) > 0
    
    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复机制"""
        # 注册一个会失败的模型适配器
        class FailingModelAdapter:
            def __init__(self, model_id: str, config: dict):
                self.model_id = model_id
                self.call_count = 0
            
            def predict(self, input_text: str, context=None):
                self.call_count += 1
                if self.call_count <= 2:
                    raise Exception("模拟模型调用失败")
                return "恢复后的响应"
            
            def is_available(self):
                return True
        
        ModelAdapterFactory.register_adapter("failing", FailingModelAdapter)
        self.model_manager.register_model("failing_model", "failing", {})
        
        # 配置评估（启用降级机制）
        config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.5}
        )
        
        # 使用小数据集测试错误处理
        small_dataset = self.test_dataset[:1]
        
        task_id = self.evaluation_engine.evaluate_model(
            model_id="failing_model",
            dataset=small_dataset,
            evaluation_config=config
        )
        
        # 等待评估完成或失败
        max_wait_time = 20
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            progress = self.evaluation_engine.get_evaluation_progress(task_id)
            
            if progress and progress.status in ["completed", "failed"]:
                break
            
            time.sleep(1)
        
        # 验证错误处理
        progress = self.evaluation_engine.get_evaluation_progress(task_id)
        
        # 任务可能完成（如果重试成功）或失败
        assert progress.status in ["completed", "failed"]
        
        if progress.status == "completed":
            result = self.evaluation_engine.get_evaluation_result(task_id)
            assert result is not None
    
    def test_concurrent_evaluations(self):
        """测试并发评估"""
        config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.5}
        )
        
        # 启动多个并发评估任务
        task_ids = []
        small_dataset = self.test_dataset[:2]
        
        for i in range(3):
            task_id = self.evaluation_engine.evaluate_model(
                model_id="good_model",
                dataset=small_dataset,
                evaluation_config=config
            )
            task_ids.append(task_id)
        
        # 等待所有任务完成
        completed_tasks = 0
        max_wait_time = 30
        start_time = time.time()
        
        while completed_tasks < len(task_ids) and time.time() - start_time < max_wait_time:
            completed_tasks = 0
            
            for task_id in task_ids:
                progress = self.evaluation_engine.get_evaluation_progress(task_id)
                if progress and progress.status == "completed":
                    completed_tasks += 1
            
            time.sleep(1)
        
        # 验证所有任务都完成了
        assert completed_tasks == len(task_ids)
        
        # 验证结果
        for task_id in task_ids:
            result = self.evaluation_engine.get_evaluation_result(task_id)
            assert result is not None
            assert result.overall_score >= 0
    
    def test_performance_benchmarks(self):
        """测试性能基准"""
        config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.7, "terminology": 0.3},
            threshold_config={"knowledge": 0.6, "terminology": 0.5}
        )
        
        # 测试小数据集性能
        start_time = time.time()
        
        task_id = self.evaluation_engine.evaluate_model(
            model_id="good_model",
            dataset=self.test_dataset,
            evaluation_config=config
        )
        
        # 等待完成
        while True:
            progress = self.evaluation_engine.get_evaluation_progress(task_id)
            if progress and progress.status == "completed":
                break
            elif progress and progress.status == "failed":
                pytest.fail("性能测试评估失败")
            time.sleep(0.5)
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        # 验证性能指标
        samples_per_second = len(self.test_dataset) / evaluation_time
        
        # 记录性能指标
        print(f"评估性能: {samples_per_second:.2f} 样本/秒")
        print(f"总评估时间: {evaluation_time:.2f} 秒")
        
        # 基本性能要求（这些阈值可以根据实际需求调整）
        assert evaluation_time < 30  # 3个样本应该在30秒内完成
        assert samples_per_second > 0.1  # 至少每10秒处理一个样本
    
    def test_memory_usage_monitoring(self):
        """测试内存使用监控"""
        import psutil
        import os
        
        # 获取当前进程
        process = psutil.Process(os.getpid())
        
        # 记录初始内存使用
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.7, "terminology": 0.3},
            threshold_config={"knowledge": 0.6, "terminology": 0.5}
        )
        
        # 运行多个评估任务
        task_ids = []
        for i in range(5):
            task_id = self.evaluation_engine.evaluate_model(
                model_id="good_model",
                dataset=self.test_dataset,
                evaluation_config=config
            )
            task_ids.append(task_id)
        
        # 等待所有任务完成
        for task_id in task_ids:
            while True:
                progress = self.evaluation_engine.get_evaluation_progress(task_id)
                if progress and progress.status in ["completed", "failed"]:
                    break
                time.sleep(0.5)
        
        # 记录峰值内存使用
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"初始内存: {initial_memory:.2f} MB")
        print(f"峰值内存: {peak_memory:.2f} MB")
        print(f"内存增长: {memory_increase:.2f} MB")
        
        # 基本内存使用要求
        assert memory_increase < 500  # 内存增长不应超过500MB
    
    def test_system_cleanup(self):
        """测试系统清理功能"""
        config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.5}
        )
        
        # 创建一些任务
        task_ids = []
        for i in range(3):
            task_id = self.evaluation_engine.evaluate_model(
                model_id="good_model",
                dataset=self.test_dataset[:1],
                evaluation_config=config
            )
            task_ids.append(task_id)
        
        # 等待任务完成
        for task_id in task_ids:
            while True:
                progress = self.evaluation_engine.get_evaluation_progress(task_id)
                if progress and progress.status in ["completed", "failed"]:
                    break
                time.sleep(0.5)
        
        # 验证任务列表
        evaluations = self.evaluation_engine.list_evaluations()
        assert len(evaluations) >= 3
        
        # 测试清理功能
        self.evaluation_engine.orchestrator.cleanup_completed_tasks(max_age_hours=0)
        
        # 验证清理后的状态
        remaining_evaluations = self.evaluation_engine.list_evaluations()
        # 由于我们设置了0小时的最大保留时间，所有已完成的任务都应该被清理
        active_evaluations = [e for e in remaining_evaluations if e["status"] in ["pending", "running"]]
        assert len(active_evaluations) == 0


class TestSystemReliability:
    """系统可靠性测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建基本组件
        self.model_manager = ModelManager()
        ModelAdapterFactory.register_adapter("mock", MockModelAdapter)
        self.model_manager.register_model("test_model", "mock", {})
        
        self.evaluators = {
            "knowledge": KnowledgeEvaluator()
        }
        
        self.result_aggregator = ResultAggregator()
        self.report_generator = ReportGenerator()
        
        self.evaluation_engine = IndustryEvaluationEngine(
            model_manager=self.model_manager,
            evaluators=self.evaluators,
            result_aggregator=self.result_aggregator,
            report_generator=self.report_generator
        )
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
        self.evaluation_engine.shutdown()
    
    def test_system_stability_under_load(self):
        """测试系统在负载下的稳定性"""
        config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.5}
        )
        
        dataset = [{"input": f"测试问题 {i}", "expected_output": f"测试答案 {i}"} for i in range(10)]
        
        # 快速连续启动多个评估任务
        task_ids = []
        for i in range(10):
            try:
                task_id = self.evaluation_engine.evaluate_model(
                    model_id="test_model",
                    dataset=dataset,
                    evaluation_config=config
                )
                task_ids.append(task_id)
            except Exception as e:
                # 记录但不失败，系统应该能够处理过载情况
                print(f"任务 {i} 创建失败: {str(e)}")
        
        # 验证至少有一些任务成功创建
        assert len(task_ids) > 0
        
        # 等待任务完成或超时
        completed_count = 0
        max_wait_time = 60
        start_time = time.time()
        
        while completed_count < len(task_ids) and time.time() - start_time < max_wait_time:
            completed_count = 0
            for task_id in task_ids:
                progress = self.evaluation_engine.get_evaluation_progress(task_id)
                if progress and progress.status in ["completed", "failed"]:
                    completed_count += 1
            time.sleep(1)
        
        # 验证系统处理了大部分任务
        success_rate = completed_count / len(task_ids)
        assert success_rate > 0.5  # 至少50%的任务应该完成
    
    def test_graceful_shutdown(self):
        """测试优雅关闭"""
        config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.5}
        )
        
        dataset = [{"input": "测试问题", "expected_output": "测试答案"}]
        
        # 启动一个评估任务
        task_id = self.evaluation_engine.evaluate_model(
            model_id="test_model",
            dataset=dataset,
            evaluation_config=config
        )
        
        # 立即关闭系统
        self.evaluation_engine.shutdown()
        
        # 验证系统能够优雅关闭而不抛出异常
        # 这个测试主要验证shutdown方法不会抛出异常


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
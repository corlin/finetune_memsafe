"""
评估引擎主控制器
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from industry_evaluation.core.interfaces import (
    EvaluationEngine, 
    BaseEvaluator, 
    ModelAdapter,
    EvaluationConfig,
    EvaluationResult,
    SampleResult,
    ProgressInfo
)
from industry_evaluation.core.result_aggregator import ResultAggregator
from industry_evaluation.core.progress_tracker import ProgressTracker
from industry_evaluation.adapters.model_adapter import ModelManager
from industry_evaluation.reporting.report_generator import ReportGenerator


class EvaluationStatus(Enum):
    """评估状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EvaluationTask:
    """评估任务"""
    task_id: str
    model_id: str
    dataset: List[Dict[str, Any]]
    config: EvaluationConfig
    status: EvaluationStatus = EvaluationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[EvaluationResult] = None
    error: Optional[str] = None
    progress: float = 0.0


class EvaluationOrchestrator:
    """评估流程编排器"""
    
    def __init__(self, 
                 model_manager: ModelManager,
                 evaluators: Dict[str, BaseEvaluator],
                 result_aggregator: ResultAggregator,
                 report_generator: ReportGenerator,
                 max_workers: int = 4):
        """
        初始化评估编排器
        
        Args:
            model_manager: 模型管理器
            evaluators: 评估器字典
            result_aggregator: 结果聚合器
            report_generator: 报告生成器
            max_workers: 最大工作线程数
        """
        self.model_manager = model_manager
        self.evaluators = evaluators
        self.result_aggregator = result_aggregator
        self.report_generator = report_generator
        self.max_workers = max_workers
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.progress_tracker = ProgressTracker()
        
        # 任务管理
        self.tasks: Dict[str, EvaluationTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            "on_task_start": [],
            "on_task_complete": [],
            "on_task_error": [],
            "on_progress_update": []
        }
    
    def register_callback(self, event: str, callback: Callable):
        """
        注册回调函数
        
        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """触发回调函数"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"回调函数执行失败: {str(e)}")
    
    def create_evaluation_task(self, 
                             task_id: str,
                             model_id: str, 
                             dataset: List[Dict[str, Any]], 
                             config: EvaluationConfig) -> EvaluationTask:
        """
        创建评估任务
        
        Args:
            task_id: 任务ID
            model_id: 模型ID
            dataset: 数据集
            config: 评估配置
            
        Returns:
            EvaluationTask: 评估任务
        """
        task = EvaluationTask(
            task_id=task_id,
            model_id=model_id,
            dataset=dataset,
            config=config
        )
        
        self.tasks[task_id] = task
        self.logger.info(f"创建评估任务: {task_id}")
        
        return task
    
    def start_evaluation(self, task_id: str) -> bool:
        """
        启动评估任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 启动是否成功
        """
        if task_id not in self.tasks:
            self.logger.error(f"任务不存在: {task_id}")
            return False
        
        task = self.tasks[task_id]
        
        if task.status != EvaluationStatus.PENDING:
            self.logger.error(f"任务状态不正确: {task.status}")
            return False
        
        # 验证模型是否可用
        model_adapter = self.model_manager.get_model(task.model_id)
        if not model_adapter:
            task.status = EvaluationStatus.FAILED
            task.error = f"模型不存在: {task.model_id}"
            return False
        
        if not model_adapter.is_available():
            task.status = EvaluationStatus.FAILED
            task.error = f"模型不可用: {task.model_id}"
            return False
        
        # 提交任务到线程池
        future = self.executor.submit(self._execute_evaluation, task)
        
        task.status = EvaluationStatus.RUNNING
        task.started_at = time.time()
        
        self._trigger_callbacks("on_task_start", task)
        self.logger.info(f"启动评估任务: {task_id}")
        
        return True
    
    def _execute_evaluation(self, task: EvaluationTask):
        """
        执行评估任务
        
        Args:
            task: 评估任务
        """
        try:
            self.logger.info(f"开始执行评估任务: {task.task_id}")
            
            # 获取模型适配器
            model_adapter = self.model_manager.get_model(task.model_id)
            
            # 初始化进度跟踪
            total_samples = len(task.dataset)
            self.progress_tracker.start_task(task.task_id, total_samples)
            
            # 执行评估
            sample_results = []
            
            for i, sample in enumerate(task.dataset):
                try:
                    # 获取模型输出
                    input_text = sample.get("input", "")
                    expected_output = sample.get("expected_output", "")
                    context = sample.get("context", {})
                    
                    model_output = model_adapter.predict(input_text, context)
                    
                    # 执行各维度评估
                    dimension_scores = {}
                    error_types = []
                    explanations = []
                    
                    for dimension, evaluator in self.evaluators.items():
                        if dimension in task.config.evaluation_dimensions:
                            try:
                                score = evaluator.evaluate(
                                    input_text, 
                                    model_output, 
                                    expected_output, 
                                    context
                                )
                                dimension_scores[dimension] = score.score
                                
                                if hasattr(score, 'error_types'):
                                    error_types.extend(score.error_types)
                                
                                if hasattr(score, 'explanation'):
                                    explanations.append(f"{dimension}: {score.explanation}")
                                    
                            except Exception as e:
                                self.logger.error(f"评估器 {dimension} 执行失败: {str(e)}")
                                dimension_scores[dimension] = 0.0
                                error_types.append(f"{dimension}_error")
                    
                    # 创建样本结果
                    sample_result = SampleResult(
                        sample_id=sample.get("id", f"sample_{i}"),
                        input_text=input_text,
                        model_output=model_output,
                        expected_output=expected_output,
                        dimension_scores=dimension_scores,
                        error_types=error_types,
                        explanation="; ".join(explanations)
                    )
                    
                    sample_results.append(sample_result)
                    
                    # 更新进度
                    self.progress_tracker.update_progress(task.task_id, i + 1)
                    task.progress = (i + 1) / total_samples
                    
                    self._trigger_callbacks("on_progress_update", task)
                    
                except Exception as e:
                    self.logger.error(f"处理样本 {i} 失败: {str(e)}")
                    # 创建错误样本结果
                    error_result = SampleResult(
                        sample_id=sample.get("id", f"sample_{i}"),
                        input_text=sample.get("input", ""),
                        model_output="",
                        expected_output=sample.get("expected_output", ""),
                        dimension_scores={dim: 0.0 for dim in task.config.evaluation_dimensions},
                        error_types=["processing_error"],
                        explanation=f"处理失败: {str(e)}"
                    )
                    sample_results.append(error_result)
            
            # 聚合结果
            evaluation_result = self.result_aggregator.aggregate_results(
                sample_results, 
                task.config
            )
            
            # 完成任务
            task.result = evaluation_result
            task.status = EvaluationStatus.COMPLETED
            task.completed_at = time.time()
            task.progress = 1.0
            
            self.progress_tracker.complete_task(task.task_id)
            self._trigger_callbacks("on_task_complete", task)
            
            self.logger.info(f"评估任务完成: {task.task_id}")
            
        except Exception as e:
            # 任务失败
            task.status = EvaluationStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            self.progress_tracker.fail_task(task.task_id, str(e))
            self._trigger_callbacks("on_task_error", task, e)
            
            self.logger.error(f"评估任务失败: {task.task_id}, 错误: {str(e)}")
    
    def get_task_status(self, task_id: str) -> Optional[EvaluationTask]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[EvaluationTask]: 任务信息
        """
        return self.tasks.get(task_id)
    
    def get_task_progress(self, task_id: str) -> Optional[ProgressInfo]:
        """
        获取任务进度
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[ProgressInfo]: 进度信息
        """
        return self.progress_tracker.get_progress(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 取消是否成功
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED]:
            return False
        
        task.status = EvaluationStatus.CANCELLED
        task.completed_at = time.time()
        
        self.progress_tracker.cancel_task(task_id)
        self.logger.info(f"取消评估任务: {task_id}")
        
        return True
    
    def list_tasks(self, status_filter: Optional[EvaluationStatus] = None) -> List[EvaluationTask]:
        """
        列出任务
        
        Args:
            status_filter: 状态过滤器
            
        Returns:
            List[EvaluationTask]: 任务列表
        """
        tasks = list(self.tasks.values())
        
        if status_filter:
            tasks = [task for task in tasks if task.status == status_filter]
        
        return sorted(tasks, key=lambda x: x.created_at, reverse=True)
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """
        清理已完成的任务
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            if (task.status in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED] and
                task.completed_at and 
                current_time - task.completed_at > max_age_seconds):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
            self.progress_tracker.remove_task(task_id)
        
        if tasks_to_remove:
            self.logger.info(f"清理了 {len(tasks_to_remove)} 个过期任务")
    
    def shutdown(self):
        """关闭编排器"""
        self.logger.info("关闭评估编排器")
        self.executor.shutdown(wait=True)


class IndustryEvaluationEngine(EvaluationEngine):
    """行业评估引擎主控制器"""
    
    def __init__(self, 
                 model_manager: ModelManager,
                 evaluators: Dict[str, BaseEvaluator],
                 result_aggregator: ResultAggregator,
                 report_generator: ReportGenerator,
                 max_workers: int = 4):
        """
        初始化评估引擎
        
        Args:
            model_manager: 模型管理器
            evaluators: 评估器字典
            result_aggregator: 结果聚合器
            report_generator: 报告生成器
            max_workers: 最大工作线程数
        """
        self.orchestrator = EvaluationOrchestrator(
            model_manager=model_manager,
            evaluators=evaluators,
            result_aggregator=result_aggregator,
            report_generator=report_generator,
            max_workers=max_workers
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 注册默认回调
        self.orchestrator.register_callback("on_task_complete", self._on_task_complete)
        self.orchestrator.register_callback("on_task_error", self._on_task_error)
    
    def evaluate_model(self, 
                      model_id: str, 
                      dataset: List[Dict[str, Any]], 
                      evaluation_config: EvaluationConfig) -> str:
        """
        评估模型
        
        Args:
            model_id: 模型ID
            dataset: 数据集
            evaluation_config: 评估配置
            
        Returns:
            str: 任务ID
        """
        import uuid
        task_id = str(uuid.uuid4())
        
        # 创建评估任务
        task = self.orchestrator.create_evaluation_task(
            task_id=task_id,
            model_id=model_id,
            dataset=dataset,
            config=evaluation_config
        )
        
        # 启动评估
        success = self.orchestrator.start_evaluation(task_id)
        
        if not success:
            raise RuntimeError(f"启动评估任务失败: {task_id}")
        
        return task_id
    
    def get_evaluation_progress(self, task_id: str) -> Optional[ProgressInfo]:
        """
        获取评估进度
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[ProgressInfo]: 进度信息
        """
        return self.orchestrator.get_task_progress(task_id)
    
    def get_evaluation_result(self, task_id: str) -> Optional[EvaluationResult]:
        """
        获取评估结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[EvaluationResult]: 评估结果
        """
        task = self.orchestrator.get_task_status(task_id)
        
        if task and task.status == EvaluationStatus.COMPLETED:
            return task.result
        
        return None
    
    def generate_report(self, task_id: str, report_format: str = "json") -> Optional[str]:
        """
        生成评估报告
        
        Args:
            task_id: 任务ID
            report_format: 报告格式
            
        Returns:
            Optional[str]: 报告内容或文件路径
        """
        task = self.orchestrator.get_task_status(task_id)
        
        if not task or task.status != EvaluationStatus.COMPLETED or not task.result:
            return None
        
        try:
            report = self.orchestrator.report_generator.generate_report(
                task.result, 
                report_format
            )
            return report
        except Exception as e:
            self.logger.error(f"生成报告失败: {str(e)}")
            return None
    
    def cancel_evaluation(self, task_id: str) -> bool:
        """
        取消评估
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 取消是否成功
        """
        return self.orchestrator.cancel_task(task_id)
    
    def list_evaluations(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出评估任务
        
        Args:
            status_filter: 状态过滤器
            
        Returns:
            List[Dict[str, Any]]: 任务列表
        """
        status_enum = None
        if status_filter:
            try:
                status_enum = EvaluationStatus(status_filter)
            except ValueError:
                pass
        
        tasks = self.orchestrator.list_tasks(status_enum)
        
        return [
            {
                "task_id": task.task_id,
                "model_id": task.model_id,
                "status": task.status.value,
                "progress": task.progress,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error": task.error
            }
            for task in tasks
        ]
    
    def _on_task_complete(self, task: EvaluationTask):
        """任务完成回调"""
        self.logger.info(f"评估任务完成: {task.task_id}")
        
        # 可以在这里添加自动报告生成等逻辑
        if task.config.auto_generate_report:
            try:
                self.generate_report(task.task_id)
            except Exception as e:
                self.logger.error(f"自动生成报告失败: {str(e)}")
    
    def _on_task_error(self, task: EvaluationTask, error: Exception):
        """任务错误回调"""
        self.logger.error(f"评估任务失败: {task.task_id}, 错误: {str(error)}")
    
    def shutdown(self):
        """关闭评估引擎"""
        self.orchestrator.shutdown()
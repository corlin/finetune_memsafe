"""
评估任务调度器
"""

import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from industry_evaluation.models.data_models import EvaluationConfig, EvaluationResult, Dataset
from industry_evaluation.core.progress_tracker import ProgressTracker, get_global_tracker


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class EvaluationTask:
    """评估任务"""
    task_id: str
    model_id: str
    dataset: Dataset
    config: EvaluationConfig
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_time: datetime = field(default_factory=datetime.now)
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    result: Optional[EvaluationResult] = None
    error: Optional[str] = None
    progress: float = 0.0
    callback: Optional[Callable] = None
    
    def get_duration(self) -> float:
        """获取任务执行时间"""
        if self.started_time and self.completed_time:
            return (self.completed_time - self.started_time).total_seconds()
        elif self.started_time:
            return (datetime.now() - self.started_time).total_seconds()
        return 0.0


class ResourceManager:
    """资源管理器"""
    
    def __init__(self, max_concurrent_tasks: int = 4, max_memory_mb: int = 2048):
        """
        初始化资源管理器
        
        Args:
            max_concurrent_tasks: 最大并发任务数
            max_memory_mb: 最大内存使用量(MB)
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_memory_mb = max_memory_mb
        self.current_tasks = 0
        self.current_memory_mb = 0
        self._lock = threading.Lock()
    
    def can_allocate_resources(self, estimated_memory_mb: int = 256) -> bool:
        """
        检查是否可以分配资源
        
        Args:
            estimated_memory_mb: 预估内存使用量
            
        Returns:
            bool: 是否可以分配
        """
        with self._lock:
            return (self.current_tasks < self.max_concurrent_tasks and 
                   self.current_memory_mb + estimated_memory_mb <= self.max_memory_mb)
    
    def allocate_resources(self, estimated_memory_mb: int = 256) -> bool:
        """
        分配资源
        
        Args:
            estimated_memory_mb: 预估内存使用量
            
        Returns:
            bool: 分配是否成功
        """
        with self._lock:
            if self.can_allocate_resources(estimated_memory_mb):
                self.current_tasks += 1
                self.current_memory_mb += estimated_memory_mb
                return True
            return False
    
    def release_resources(self, memory_mb: int = 256):
        """
        释放资源
        
        Args:
            memory_mb: 释放的内存量
        """
        with self._lock:
            self.current_tasks = max(0, self.current_tasks - 1)
            self.current_memory_mb = max(0, self.current_memory_mb - memory_mb)
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        with self._lock:
            return {
                "current_tasks": self.current_tasks,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "current_memory_mb": self.current_memory_mb,
                "max_memory_mb": self.max_memory_mb,
                "task_utilization": self.current_tasks / self.max_concurrent_tasks,
                "memory_utilization": self.current_memory_mb / self.max_memory_mb
            }


class TaskQueue:
    """任务队列"""
    
    def __init__(self):
        """初始化任务队列"""
        self._queue = []
        self._lock = threading.Lock()
    
    def add_task(self, task: EvaluationTask):
        """添加任务到队列"""
        with self._lock:
            self._queue.append(task)
            # 按优先级和创建时间排序
            self._queue.sort(key=lambda t: (-t.priority.value, t.created_time))
    
    def get_next_task(self) -> Optional[EvaluationTask]:
        """获取下一个任务"""
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
            return None
    
    def remove_task(self, task_id: str) -> bool:
        """移除任务"""
        with self._lock:
            for i, task in enumerate(self._queue):
                if task.task_id == task_id:
                    self._queue.pop(i)
                    return True
            return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        with self._lock:
            return {
                "total_tasks": len(self._queue),
                "priority_distribution": {
                    priority.name: len([t for t in self._queue if t.priority == priority])
                    for priority in TaskPriority
                },
                "oldest_task_age": (datetime.now() - min(t.created_time for t in self._queue)).total_seconds() if self._queue else 0
            }
    
    def is_empty(self) -> bool:
        """检查队列是否为空"""
        with self._lock:
            return len(self._queue) == 0


class EvaluationScheduler:
    """评估任务调度器"""
    
    def __init__(self, max_concurrent_tasks: int = 4, max_memory_mb: int = 2048,
                 progress_tracker: Optional[ProgressTracker] = None):
        """
        初始化调度器
        
        Args:
            max_concurrent_tasks: 最大并发任务数
            max_memory_mb: 最大内存使用量
            progress_tracker: 进度跟踪器
        """
        self.resource_manager = ResourceManager(max_concurrent_tasks, max_memory_mb)
        self.task_queue = TaskQueue()
        self.running_tasks: Dict[str, EvaluationTask] = {}
        self.completed_tasks: Dict[str, EvaluationTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.futures: Dict[str, Future] = {}
        self.progress_tracker = progress_tracker or get_global_tracker()
        
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start(self):
        """启动调度器"""
        if not self._running:
            self._running = True
            self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self._scheduler_thread.start()
    
    def stop(self):
        """停止调度器"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        
        # 取消所有运行中的任务
        for task_id in list(self.running_tasks.keys()):
            self.cancel_task(task_id)
        
        self.executor.shutdown(wait=True)
    
    def submit_task(self, task: EvaluationTask) -> str:
        """
        提交评估任务
        
        Args:
            task: 评估任务
            
        Returns:
            str: 任务ID
        """
        # 在进度跟踪器中创建任务
        total_samples = len(task.dataset.samples) if task.dataset else 0
        self.progress_tracker.create_task(
            task.task_id, 
            total_steps=5,  # 假设有5个评估步骤
            total_samples=total_samples
        )
        
        self.task_queue.add_task(task)
        return task.task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 取消是否成功
        """
        # 尝试从队列中移除
        if self.task_queue.remove_task(task_id):
            # 在进度跟踪器中取消任务
            try:
                self.progress_tracker.cancel_task(task_id)
            except ValueError:
                pass  # 任务可能还未在跟踪器中创建
            return True
        
        # 尝试取消运行中的任务
        with self._lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                
                # 取消Future
                if task_id in self.futures:
                    self.futures[task_id].cancel()
                
                # 在进度跟踪器中取消任务
                try:
                    self.progress_tracker.cancel_task(task_id)
                except ValueError:
                    pass  # 任务可能已经完成
                
                return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Dict[str, Any]]: 任务状态信息
        """
        # 检查运行中的任务
        with self._lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                return self._task_to_status_dict(task)
        
        # 检查已完成的任务
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return self._task_to_status_dict(task)
        
        # 检查队列中的任务
        with self.task_queue._lock:
            for task in self.task_queue._queue:
                if task.task_id == task_id:
                    return self._task_to_status_dict(task)
        
        return None
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        queue_status = self.task_queue.get_queue_status()
        resource_usage = self.resource_manager.get_resource_usage()
        
        with self._lock:
            running_tasks_count = len(self.running_tasks)
        
        return {
            "is_running": self._running,
            "queue_status": queue_status,
            "resource_usage": resource_usage,
            "running_tasks_count": running_tasks_count,
            "completed_tasks_count": len(self.completed_tasks),
            "total_tasks_processed": len(self.completed_tasks) + running_tasks_count
        }
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self._running:
            try:
                # 清理已完成的任务
                self._cleanup_completed_tasks()
                
                # 检查是否可以启动新任务
                if self.resource_manager.can_allocate_resources():
                    next_task = self.task_queue.get_next_task()
                    
                    if next_task:
                        self._start_task(next_task)
                
                # 短暂休眠
                time.sleep(0.1)
                
            except Exception as e:
                print(f"调度器循环出错: {e}")
                time.sleep(1.0)
    
    def _start_task(self, task: EvaluationTask):
        """启动任务"""
        if not self.resource_manager.allocate_resources():
            # 资源分配失败，重新加入队列
            self.task_queue.add_task(task)
            return
        
        task.status = TaskStatus.RUNNING
        task.started_time = datetime.now()
        
        # 在进度跟踪器中开始任务
        self.progress_tracker.start_task(task.task_id)
        
        with self._lock:
            self.running_tasks[task.task_id] = task
        
        # 提交到线程池执行
        future = self.executor.submit(self._execute_task, task)
        self.futures[task.task_id] = future
    
    def _execute_task(self, task: EvaluationTask):
        """执行任务"""
        try:
            # 这里应该调用实际的评估逻辑
            # 为了演示，我们创建一个模拟的评估结果
            
            total_samples = len(task.dataset.samples) if task.dataset else 10
            
            # 模拟评估过程
            for step in range(5):  # 5个评估步骤
                if task.status == TaskStatus.CANCELLED:
                    self.progress_tracker.cancel_task(task.task_id)
                    return
                
                step_name = ["数据预处理", "模型加载", "评估执行", "结果分析", "报告生成"][step]
                self.progress_tracker.update_progress(
                    task.task_id,
                    current_step=step + 1,
                    step_name=step_name,
                    message=f"正在执行: {step_name}"
                )
                
                # 模拟处理样本
                samples_per_step = total_samples // 5
                for sample in range(samples_per_step):
                    if task.status == TaskStatus.CANCELLED:
                        self.progress_tracker.cancel_task(task.task_id)
                        return
                    
                    current_sample = step * samples_per_step + sample + 1
                    self.progress_tracker.update_progress(
                        task.task_id,
                        current_sample=current_sample,
                        message=f"处理样本 {current_sample}/{total_samples}",
                        metrics={"processed_samples": current_sample}
                    )
                    
                    task.progress = current_sample / total_samples
                    time.sleep(0.01)  # 模拟处理时间
            
            # 创建模拟结果
            from industry_evaluation.models.data_models import ErrorAnalysis
            
            task.result = EvaluationResult(
                task_id=task.task_id,
                model_id=task.model_id,
                overall_score=0.85,
                dimension_scores={"knowledge": 0.8, "terminology": 0.9},
                detailed_results=[],
                error_analysis=ErrorAnalysis({}, [], {}, []),
                improvement_suggestions=["建议改进专业知识"],
                evaluation_config=task.config
            )
            
            task.status = TaskStatus.COMPLETED
            task.completed_time = datetime.now()
            
            # 在进度跟踪器中完成任务
            self.progress_tracker.complete_task(task.task_id, "评估任务成功完成")
            
            # 调用回调函数
            if task.callback:
                try:
                    task.callback(task)
                except Exception as e:
                    print(f"回调函数执行失败: {e}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_time = datetime.now()
            
            # 在进度跟踪器中标记任务失败
            self.progress_tracker.fail_task(task.task_id, str(e))
        
        finally:
            # 释放资源
            self.resource_manager.release_resources()
            
            # 移动到已完成任务
            with self._lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
            
            self.completed_tasks[task.task_id] = task
            
            # 清理Future
            if task.task_id in self.futures:
                del self.futures[task.task_id]
    
    def _cleanup_completed_tasks(self):
        """清理已完成的任务"""
        # 清理超过1小时的已完成任务
        current_time = datetime.now()
        to_remove = []
        
        for task_id, task in self.completed_tasks.items():
            if task.completed_time and (current_time - task.completed_time).total_seconds() > 3600:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.completed_tasks[task_id]
    
    def _task_to_status_dict(self, task: EvaluationTask) -> Dict[str, Any]:
        """将任务转换为状态字典"""
        return {
            "task_id": task.task_id,
            "model_id": task.model_id,
            "status": task.status.value,
            "priority": task.priority.name,
            "progress": task.progress,
            "created_time": task.created_time.isoformat(),
            "started_time": task.started_time.isoformat() if task.started_time else None,
            "completed_time": task.completed_time.isoformat() if task.completed_time else None,
            "duration": task.get_duration(),
            "error": task.error,
            "has_result": task.result is not None
        }
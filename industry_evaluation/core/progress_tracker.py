"""
进度跟踪和状态管理模块
"""

import json
import sqlite3
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from industry_evaluation.models.data_models import ProgressInfo, EvaluationStatus


class ProgressEventType(Enum):
    """进度事件类型"""
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    SAMPLE_PROCESSED = "sample_processed"


@dataclass
class ProgressEvent:
    """进度事件"""
    event_id: str
    task_id: str
    event_type: ProgressEventType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class TaskProgress:
    """任务进度信息"""
    task_id: str
    status: EvaluationStatus
    current_step: int = 0
    total_steps: int = 0
    current_sample: int = 0
    total_samples: int = 0
    progress_percentage: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    last_update_time: datetime = field(default_factory=datetime.now)
    estimated_remaining_time: Optional[float] = None
    current_step_name: str = ""
    message: str = ""
    error_message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def update_progress(self, current_sample: Optional[int] = None, 
                       current_step: Optional[int] = None,
                       message: str = ""):
        """更新进度"""
        if current_sample is not None:
            self.current_sample = current_sample
        
        if current_step is not None:
            self.current_step = current_step
        
        if message:
            self.message = message
        
        # 计算进度百分比
        if self.total_samples > 0:
            sample_progress = self.current_sample / self.total_samples
            if self.total_steps > 0:
                step_progress = self.current_step / self.total_steps
                self.progress_percentage = (step_progress + sample_progress / self.total_steps) * 100
            else:
                self.progress_percentage = sample_progress * 100
        elif self.total_steps > 0:
            self.progress_percentage = (self.current_step / self.total_steps) * 100
        
        self.last_update_time = datetime.now()
        
        # 估算剩余时间
        self._estimate_remaining_time()
    
    def _estimate_remaining_time(self):
        """估算剩余时间"""
        if self.progress_percentage <= 0:
            self.estimated_remaining_time = None
            return
        
        elapsed_time = (self.last_update_time - self.start_time).total_seconds()
        if elapsed_time <= 0:
            self.estimated_remaining_time = None
            return
        
        # 基于当前进度估算总时间
        estimated_total_time = elapsed_time / (self.progress_percentage / 100)
        self.estimated_remaining_time = max(0, estimated_total_time - elapsed_time)
    
    def to_progress_info(self) -> ProgressInfo:
        """转换为ProgressInfo对象"""
        return ProgressInfo(
            task_id=self.task_id,
            current_step=self.current_step,
            total_steps=self.total_steps,
            current_sample=self.current_sample,
            total_samples=self.total_samples,
            status=self.status,
            message=self.message,
            start_time=self.start_time,
            estimated_remaining_time=self.estimated_remaining_time
        )


class ProgressStorage:
    """进度存储管理"""
    
    def __init__(self, db_path: str = "evaluation_progress.db"):
        """
        初始化进度存储
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_database()
        self._lock = threading.Lock()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_progress (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    current_step INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 0,
                    current_sample INTEGER DEFAULT 0,
                    total_samples INTEGER DEFAULT 0,
                    progress_percentage REAL DEFAULT 0.0,
                    start_time TEXT NOT NULL,
                    last_update_time TEXT NOT NULL,
                    estimated_remaining_time REAL,
                    current_step_name TEXT DEFAULT '',
                    message TEXT DEFAULT '',
                    error_message TEXT DEFAULT '',
                    metrics TEXT DEFAULT '{}'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS progress_events (
                    event_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data TEXT DEFAULT '{}',
                    message TEXT DEFAULT '',
                    FOREIGN KEY (task_id) REFERENCES task_progress (task_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_events 
                ON progress_events (task_id, timestamp)
            """)
    
    def save_progress(self, progress: TaskProgress):
        """保存进度信息"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO task_progress 
                    (task_id, status, current_step, total_steps, current_sample, 
                     total_samples, progress_percentage, start_time, last_update_time,
                     estimated_remaining_time, current_step_name, message, 
                     error_message, metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    progress.task_id,
                    progress.status.value,
                    progress.current_step,
                    progress.total_steps,
                    progress.current_sample,
                    progress.total_samples,
                    progress.progress_percentage,
                    progress.start_time.isoformat(),
                    progress.last_update_time.isoformat(),
                    progress.estimated_remaining_time,
                    progress.current_step_name,
                    progress.message,
                    progress.error_message,
                    json.dumps(progress.metrics)
                ))
    
    def load_progress(self, task_id: str) -> Optional[TaskProgress]:
        """加载进度信息"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM task_progress WHERE task_id = ?
                """, (task_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return TaskProgress(
                    task_id=row[0],
                    status=EvaluationStatus(row[1]),
                    current_step=row[2],
                    total_steps=row[3],
                    current_sample=row[4],
                    total_samples=row[5],
                    progress_percentage=row[6],
                    start_time=datetime.fromisoformat(row[7]),
                    last_update_time=datetime.fromisoformat(row[8]),
                    estimated_remaining_time=row[9],
                    current_step_name=row[10],
                    message=row[11],
                    error_message=row[12],
                    metrics=json.loads(row[13])
                )
    
    def save_event(self, event: ProgressEvent):
        """保存进度事件"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO progress_events 
                    (event_id, task_id, event_type, timestamp, data, message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.task_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    json.dumps(event.data),
                    event.message
                ))
    
    def get_task_events(self, task_id: str, limit: int = 100) -> List[ProgressEvent]:
        """获取任务事件历史"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM progress_events 
                    WHERE task_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (task_id, limit))
                
                events = []
                for row in cursor.fetchall():
                    events.append(ProgressEvent(
                        event_id=row[0],
                        task_id=row[1],
                        event_type=ProgressEventType(row[2]),
                        timestamp=datetime.fromisoformat(row[3]),
                        data=json.loads(row[4]),
                        message=row[5]
                    ))
                
                return events
    
    def get_all_active_tasks(self) -> List[TaskProgress]:
        """获取所有活跃任务"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM task_progress 
                    WHERE status IN ('pending', 'running')
                    ORDER BY start_time DESC
                """)
                
                tasks = []
                for row in cursor.fetchall():
                    tasks.append(TaskProgress(
                        task_id=row[0],
                        status=EvaluationStatus(row[1]),
                        current_step=row[2],
                        total_steps=row[3],
                        current_sample=row[4],
                        total_samples=row[5],
                        progress_percentage=row[6],
                        start_time=datetime.fromisoformat(row[7]),
                        last_update_time=datetime.fromisoformat(row[8]),
                        estimated_remaining_time=row[9],
                        current_step_name=row[10],
                        message=row[11],
                        error_message=row[12],
                        metrics=json.loads(row[13])
                    ))
                
                return tasks
    
    def cleanup_old_data(self, days: int = 30):
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # 删除旧的已完成任务
                conn.execute("""
                    DELETE FROM task_progress 
                    WHERE status IN ('completed', 'failed', 'cancelled')
                    AND last_update_time < ?
                """, (cutoff_date.isoformat(),))
                
                # 删除旧的事件
                conn.execute("""
                    DELETE FROM progress_events 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, storage: Optional[ProgressStorage] = None):
        """
        初始化进度跟踪器
        
        Args:
            storage: 进度存储实例
        """
        self.storage = storage or ProgressStorage()
        self.active_tasks: Dict[str, TaskProgress] = {}
        self.event_listeners: List[Callable[[ProgressEvent], None]] = []
        self._lock = threading.Lock()
        
        # 加载活跃任务
        self._load_active_tasks()
    
    def _load_active_tasks(self):
        """加载活跃任务"""
        active_tasks = self.storage.get_all_active_tasks()
        for task in active_tasks:
            self.active_tasks[task.task_id] = task
    
    def create_task(self, task_id: str, total_steps: int = 0, 
                   total_samples: int = 0) -> TaskProgress:
        """
        创建新任务
        
        Args:
            task_id: 任务ID
            total_steps: 总步骤数
            total_samples: 总样本数
            
        Returns:
            TaskProgress: 任务进度对象
        """
        with self._lock:
            progress = TaskProgress(
                task_id=task_id,
                status=EvaluationStatus.PENDING,
                total_steps=total_steps,
                total_samples=total_samples
            )
            
            self.active_tasks[task_id] = progress
            self.storage.save_progress(progress)
            
            # 触发事件
            self._emit_event(ProgressEvent(
                event_id=f"{task_id}_created_{datetime.now().timestamp()}",
                task_id=task_id,
                event_type=ProgressEventType.TASK_CREATED,
                timestamp=datetime.now(),
                data={"total_steps": total_steps, "total_samples": total_samples},
                message=f"任务 {task_id} 已创建"
            ))
            
            return progress
    
    def start_task(self, task_id: str):
        """开始任务"""
        with self._lock:
            if task_id not in self.active_tasks:
                raise ValueError(f"任务 {task_id} 不存在")
            
            progress = self.active_tasks[task_id]
            progress.status = EvaluationStatus.RUNNING
            progress.start_time = datetime.now()
            progress.last_update_time = datetime.now()
            
            self.storage.save_progress(progress)
            
            # 触发事件
            self._emit_event(ProgressEvent(
                event_id=f"{task_id}_started_{datetime.now().timestamp()}",
                task_id=task_id,
                event_type=ProgressEventType.TASK_STARTED,
                timestamp=datetime.now(),
                message=f"任务 {task_id} 已开始"
            ))
    
    def update_progress(self, task_id: str, current_sample: Optional[int] = None,
                       current_step: Optional[int] = None, message: str = "",
                       step_name: str = "", metrics: Optional[Dict[str, Any]] = None):
        """
        更新任务进度
        
        Args:
            task_id: 任务ID
            current_sample: 当前样本数
            current_step: 当前步骤
            message: 进度消息
            step_name: 步骤名称
            metrics: 指标数据
        """
        with self._lock:
            if task_id not in self.active_tasks:
                raise ValueError(f"任务 {task_id} 不存在")
            
            progress = self.active_tasks[task_id]
            
            # 更新步骤名称
            if step_name:
                progress.current_step_name = step_name
            
            # 更新指标
            if metrics:
                progress.metrics.update(metrics)
            
            # 更新进度
            progress.update_progress(current_sample, current_step, message)
            
            self.storage.save_progress(progress)
            
            # 触发事件
            event_data = {}
            if current_sample is not None:
                event_data["current_sample"] = current_sample
            if current_step is not None:
                event_data["current_step"] = current_step
            if metrics:
                event_data["metrics"] = metrics
            
            self._emit_event(ProgressEvent(
                event_id=f"{task_id}_progress_{datetime.now().timestamp()}",
                task_id=task_id,
                event_type=ProgressEventType.TASK_PROGRESS,
                timestamp=datetime.now(),
                data=event_data,
                message=message or f"任务 {task_id} 进度更新: {progress.progress_percentage:.1f}%"
            ))
    
    def complete_task(self, task_id: str, message: str = ""):
        """完成任务"""
        with self._lock:
            if task_id not in self.active_tasks:
                raise ValueError(f"任务 {task_id} 不存在")
            
            progress = self.active_tasks[task_id]
            progress.status = EvaluationStatus.COMPLETED
            progress.progress_percentage = 100.0
            progress.last_update_time = datetime.now()
            progress.estimated_remaining_time = 0.0
            
            if message:
                progress.message = message
            
            self.storage.save_progress(progress)
            
            # 从活跃任务中移除
            del self.active_tasks[task_id]
            
            # 触发事件
            self._emit_event(ProgressEvent(
                event_id=f"{task_id}_completed_{datetime.now().timestamp()}",
                task_id=task_id,
                event_type=ProgressEventType.TASK_COMPLETED,
                timestamp=datetime.now(),
                message=message or f"任务 {task_id} 已完成"
            ))
    
    def fail_task(self, task_id: str, error_message: str):
        """任务失败"""
        with self._lock:
            if task_id not in self.active_tasks:
                raise ValueError(f"任务 {task_id} 不存在")
            
            progress = self.active_tasks[task_id]
            progress.status = EvaluationStatus.FAILED
            progress.error_message = error_message
            progress.last_update_time = datetime.now()
            
            self.storage.save_progress(progress)
            
            # 从活跃任务中移除
            del self.active_tasks[task_id]
            
            # 触发事件
            self._emit_event(ProgressEvent(
                event_id=f"{task_id}_failed_{datetime.now().timestamp()}",
                task_id=task_id,
                event_type=ProgressEventType.TASK_FAILED,
                timestamp=datetime.now(),
                data={"error": error_message},
                message=f"任务 {task_id} 失败: {error_message}"
            ))
    
    def cancel_task(self, task_id: str):
        """取消任务"""
        with self._lock:
            if task_id not in self.active_tasks:
                raise ValueError(f"任务 {task_id} 不存在")
            
            progress = self.active_tasks[task_id]
            progress.status = EvaluationStatus.CANCELLED
            progress.last_update_time = datetime.now()
            
            self.storage.save_progress(progress)
            
            # 从活跃任务中移除
            del self.active_tasks[task_id]
            
            # 触发事件
            self._emit_event(ProgressEvent(
                event_id=f"{task_id}_cancelled_{datetime.now().timestamp()}",
                task_id=task_id,
                event_type=ProgressEventType.TASK_CANCELLED,
                timestamp=datetime.now(),
                message=f"任务 {task_id} 已取消"
            ))
    
    def get_progress(self, task_id: str) -> Optional[ProgressInfo]:
        """
        获取任务进度
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[ProgressInfo]: 进度信息
        """
        # 先检查活跃任务
        with self._lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].to_progress_info()
        
        # 从存储中加载
        progress = self.storage.load_progress(task_id)
        if progress:
            return progress.to_progress_info()
        
        return None
    
    def get_all_active_progress(self) -> List[ProgressInfo]:
        """获取所有活跃任务的进度"""
        with self._lock:
            return [progress.to_progress_info() for progress in self.active_tasks.values()]
    
    def get_task_events(self, task_id: str, limit: int = 100) -> List[ProgressEvent]:
        """获取任务事件历史"""
        return self.storage.get_task_events(task_id, limit)
    
    def add_event_listener(self, listener: Callable[[ProgressEvent], None]):
        """添加事件监听器"""
        self.event_listeners.append(listener)
    
    def remove_event_listener(self, listener: Callable[[ProgressEvent], None]):
        """移除事件监听器"""
        if listener in self.event_listeners:
            self.event_listeners.remove(listener)
    
    def _emit_event(self, event: ProgressEvent):
        """触发事件"""
        # 保存事件
        self.storage.save_event(event)
        
        # 通知监听器
        for listener in self.event_listeners:
            try:
                listener(event)
            except Exception as e:
                print(f"事件监听器执行失败: {e}")
    
    def cleanup_old_data(self, days: int = 30):
        """清理旧数据"""
        self.storage.cleanup_old_data(days)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            active_count = len(self.active_tasks)
            
            # 按状态统计
            status_counts = {}
            for progress in self.active_tasks.values():
                status = progress.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # 计算平均进度
            if active_count > 0:
                avg_progress = sum(p.progress_percentage for p in self.active_tasks.values()) / active_count
            else:
                avg_progress = 0.0
            
            return {
                "active_tasks": active_count,
                "status_distribution": status_counts,
                "average_progress": avg_progress,
                "tasks_by_status": {
                    status.value: [
                        task_id for task_id, progress in self.active_tasks.items()
                        if progress.status == status
                    ]
                    for status in EvaluationStatus
                }
            }


# 全局进度跟踪器实例
_global_tracker: Optional[ProgressTracker] = None


def get_global_tracker() -> ProgressTracker:
    """获取全局进度跟踪器"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProgressTracker()
    return _global_tracker


def set_global_tracker(tracker: ProgressTracker):
    """设置全局进度跟踪器"""
    global _global_tracker
    _global_tracker = tracker
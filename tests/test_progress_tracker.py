"""
进度跟踪器单元测试
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from industry_evaluation.core.progress_tracker import (
    ProgressTracker, ProgressStorage, TaskProgress, ProgressEvent,
    ProgressEventType, get_global_tracker, set_global_tracker
)
from industry_evaluation.models.data_models import EvaluationStatus


class TestProgressStorage:
    """进度存储测试"""
    
    def setup_method(self):
        """测试前准备"""
        # 使用临时数据库文件
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.storage = ProgressStorage(self.temp_db.name)
    
    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_save_and_load_progress(self):
        """测试保存和加载进度"""
        progress = TaskProgress(
            task_id="test_task_001",
            status=EvaluationStatus.RUNNING,
            current_step=2,
            total_steps=10,
            current_sample=50,
            total_samples=100,
            progress_percentage=25.0,
            current_step_name="数据预处理",
            message="正在处理数据",
            metrics={"accuracy": 0.85, "loss": 0.15}
        )
        
        # 保存进度
        self.storage.save_progress(progress)
        
        # 加载进度
        loaded_progress = self.storage.load_progress("test_task_001")
        
        assert loaded_progress is not None
        assert loaded_progress.task_id == "test_task_001"
        assert loaded_progress.status == EvaluationStatus.RUNNING
        assert loaded_progress.current_step == 2
        assert loaded_progress.total_steps == 10
        assert loaded_progress.current_sample == 50
        assert loaded_progress.total_samples == 100
        assert loaded_progress.progress_percentage == 25.0
        assert loaded_progress.current_step_name == "数据预处理"
        assert loaded_progress.message == "正在处理数据"
        assert loaded_progress.metrics["accuracy"] == 0.85
    
    def test_load_nonexistent_progress(self):
        """测试加载不存在的进度"""
        progress = self.storage.load_progress("nonexistent_task")
        assert progress is None
    
    def test_save_and_get_events(self):
        """测试保存和获取事件"""
        event = ProgressEvent(
            event_id="event_001",
            task_id="test_task_001",
            event_type=ProgressEventType.TASK_STARTED,
            timestamp=datetime.now(),
            data={"step": "initialization"},
            message="任务开始"
        )
        
        # 保存事件
        self.storage.save_event(event)
        
        # 获取事件
        events = self.storage.get_task_events("test_task_001")
        
        assert len(events) == 1
        assert events[0].event_id == "event_001"
        assert events[0].task_id == "test_task_001"
        assert events[0].event_type == ProgressEventType.TASK_STARTED
        assert events[0].message == "任务开始"
        assert events[0].data["step"] == "initialization"
    
    def test_get_all_active_tasks(self):
        """测试获取所有活跃任务"""
        # 创建多个任务
        tasks = [
            TaskProgress("task_001", EvaluationStatus.RUNNING),
            TaskProgress("task_002", EvaluationStatus.PENDING),
            TaskProgress("task_003", EvaluationStatus.COMPLETED),  # 非活跃
            TaskProgress("task_004", EvaluationStatus.RUNNING)
        ]
        
        for task in tasks:
            self.storage.save_progress(task)
        
        # 获取活跃任务
        active_tasks = self.storage.get_all_active_tasks()
        
        assert len(active_tasks) == 3  # 排除已完成的任务
        active_task_ids = {task.task_id for task in active_tasks}
        assert "task_001" in active_task_ids
        assert "task_002" in active_task_ids
        assert "task_004" in active_task_ids
        assert "task_003" not in active_task_ids
    
    def test_cleanup_old_data(self):
        """测试清理旧数据"""
        # 创建旧任务
        old_time = datetime.now() - timedelta(days=35)
        old_progress = TaskProgress(
            task_id="old_task",
            status=EvaluationStatus.COMPLETED,
            start_time=old_time,
            last_update_time=old_time
        )
        
        # 创建新任务
        new_progress = TaskProgress(
            task_id="new_task",
            status=EvaluationStatus.COMPLETED
        )
        
        self.storage.save_progress(old_progress)
        self.storage.save_progress(new_progress)
        
        # 创建旧事件
        old_event = ProgressEvent(
            event_id="old_event",
            task_id="old_task",
            event_type=ProgressEventType.TASK_COMPLETED,
            timestamp=old_time
        )
        
        self.storage.save_event(old_event)
        
        # 清理旧数据
        self.storage.cleanup_old_data(days=30)
        
        # 验证旧数据被清理
        assert self.storage.load_progress("old_task") is None
        assert self.storage.load_progress("new_task") is not None
        
        old_events = self.storage.get_task_events("old_task")
        assert len(old_events) == 0


class TestTaskProgress:
    """任务进度测试"""
    
    def test_update_progress(self):
        """测试更新进度"""
        progress = TaskProgress(
            task_id="test_task",
            status=EvaluationStatus.RUNNING,
            total_steps=5,
            total_samples=100
        )
        
        # 更新进度
        progress.update_progress(current_sample=25, current_step=2, message="处理中")
        
        assert progress.current_sample == 25
        assert progress.current_step == 2
        assert progress.message == "处理中"
        assert progress.progress_percentage == 45.0  # (2/5 + 0.25/5) * 100
        assert progress.estimated_remaining_time is not None
    
    def test_progress_calculation_samples_only(self):
        """测试仅基于样本的进度计算"""
        progress = TaskProgress(
            task_id="test_task",
            status=EvaluationStatus.RUNNING,
            total_samples=100
        )
        
        progress.update_progress(current_sample=30)
        assert progress.progress_percentage == 30.0
    
    def test_progress_calculation_steps_only(self):
        """测试仅基于步骤的进度计算"""
        progress = TaskProgress(
            task_id="test_task",
            status=EvaluationStatus.RUNNING,
            total_steps=4
        )
        
        progress.update_progress(current_step=1)
        assert progress.progress_percentage == 25.0
    
    def test_to_progress_info(self):
        """测试转换为ProgressInfo"""
        progress = TaskProgress(
            task_id="test_task",
            status=EvaluationStatus.RUNNING,
            current_step=3,
            total_steps=10,
            current_sample=75,
            total_samples=100,
            message="正在评估"
        )
        
        progress_info = progress.to_progress_info()
        
        assert progress_info.task_id == "test_task"
        assert progress_info.status == EvaluationStatus.RUNNING
        assert progress_info.current_step == 3
        assert progress_info.total_steps == 10
        assert progress_info.current_sample == 75
        assert progress_info.total_samples == 100
        assert progress_info.message == "正在评估"


class TestProgressTracker:
    """进度跟踪器测试"""
    
    def setup_method(self):
        """测试前准备"""
        # 使用临时数据库
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        storage = ProgressStorage(self.temp_db.name)
        self.tracker = ProgressTracker(storage)
        
        # 事件收集器
        self.received_events = []
        self.tracker.add_event_listener(self._event_collector)
    
    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def _event_collector(self, event: ProgressEvent):
        """事件收集器"""
        self.received_events.append(event)
    
    def test_create_task(self):
        """测试创建任务"""
        progress = self.tracker.create_task("test_task_001", total_steps=5, total_samples=100)
        
        assert progress.task_id == "test_task_001"
        assert progress.status == EvaluationStatus.PENDING
        assert progress.total_steps == 5
        assert progress.total_samples == 100
        
        # 检查事件
        assert len(self.received_events) == 1
        assert self.received_events[0].event_type == ProgressEventType.TASK_CREATED
        assert self.received_events[0].task_id == "test_task_001"
    
    def test_start_task(self):
        """测试开始任务"""
        self.tracker.create_task("test_task_002")
        self.tracker.start_task("test_task_002")
        
        progress = self.tracker.get_progress("test_task_002")
        assert progress.status == EvaluationStatus.RUNNING
        
        # 检查事件
        start_events = [e for e in self.received_events if e.event_type == ProgressEventType.TASK_STARTED]
        assert len(start_events) == 1
    
    def test_update_progress(self):
        """测试更新进度"""
        self.tracker.create_task("test_task_003", total_samples=100)
        self.tracker.start_task("test_task_003")
        
        self.tracker.update_progress(
            "test_task_003",
            current_sample=25,
            message="处理中",
            step_name="数据预处理",
            metrics={"accuracy": 0.85}
        )
        
        progress = self.tracker.get_progress("test_task_003")
        assert progress.current_sample == 25
        assert progress.message == "处理中"
        
        # 检查事件
        progress_events = [e for e in self.received_events if e.event_type == ProgressEventType.TASK_PROGRESS]
        assert len(progress_events) == 1
        assert progress_events[0].data["current_sample"] == 25
        assert progress_events[0].data["metrics"]["accuracy"] == 0.85
    
    def test_complete_task(self):
        """测试完成任务"""
        self.tracker.create_task("test_task_004")
        self.tracker.start_task("test_task_004")
        self.tracker.complete_task("test_task_004", "任务成功完成")
        
        progress = self.tracker.get_progress("test_task_004")
        assert progress.status == EvaluationStatus.COMPLETED
        assert progress.progress_percentage == 100.0
        assert progress.message == "任务成功完成"
        
        # 任务应该从活跃列表中移除
        assert "test_task_004" not in self.tracker.active_tasks
        
        # 检查事件
        complete_events = [e for e in self.received_events if e.event_type == ProgressEventType.TASK_COMPLETED]
        assert len(complete_events) == 1
    
    def test_fail_task(self):
        """测试任务失败"""
        self.tracker.create_task("test_task_005")
        self.tracker.start_task("test_task_005")
        self.tracker.fail_task("test_task_005", "评估器初始化失败")
        
        progress = self.tracker.get_progress("test_task_005")
        assert progress.status == EvaluationStatus.FAILED
        
        # 任务应该从活跃列表中移除
        assert "test_task_005" not in self.tracker.active_tasks
        
        # 检查事件
        fail_events = [e for e in self.received_events if e.event_type == ProgressEventType.TASK_FAILED]
        assert len(fail_events) == 1
        assert fail_events[0].data["error"] == "评估器初始化失败"
    
    def test_cancel_task(self):
        """测试取消任务"""
        self.tracker.create_task("test_task_006")
        self.tracker.start_task("test_task_006")
        self.tracker.cancel_task("test_task_006")
        
        progress = self.tracker.get_progress("test_task_006")
        assert progress.status == EvaluationStatus.CANCELLED
        
        # 任务应该从活跃列表中移除
        assert "test_task_006" not in self.tracker.active_tasks
        
        # 检查事件
        cancel_events = [e for e in self.received_events if e.event_type == ProgressEventType.TASK_CANCELLED]
        assert len(cancel_events) == 1
    
    def test_get_all_active_progress(self):
        """测试获取所有活跃进度"""
        # 创建多个任务
        self.tracker.create_task("active_task_001")
        self.tracker.create_task("active_task_002")
        self.tracker.start_task("active_task_001")
        
        # 完成一个任务
        self.tracker.create_task("completed_task")
        self.tracker.start_task("completed_task")
        self.tracker.complete_task("completed_task")
        
        active_progress = self.tracker.get_all_active_progress()
        
        assert len(active_progress) == 2
        task_ids = {p.task_id for p in active_progress}
        assert "active_task_001" in task_ids
        assert "active_task_002" in task_ids
        assert "completed_task" not in task_ids
    
    def test_get_task_events(self):
        """测试获取任务事件"""
        self.tracker.create_task("event_task")
        self.tracker.start_task("event_task")
        self.tracker.update_progress("event_task", current_sample=10)
        self.tracker.complete_task("event_task")
        
        events = self.tracker.get_task_events("event_task")
        
        assert len(events) >= 4  # 创建、开始、进度、完成
        event_types = {e.event_type for e in events}
        assert ProgressEventType.TASK_CREATED in event_types
        assert ProgressEventType.TASK_STARTED in event_types
        assert ProgressEventType.TASK_PROGRESS in event_types
        assert ProgressEventType.TASK_COMPLETED in event_types
    
    def test_event_listeners(self):
        """测试事件监听器"""
        events_received = []
        
        def custom_listener(event):
            events_received.append(event.event_type)
        
        self.tracker.add_event_listener(custom_listener)
        
        self.tracker.create_task("listener_test")
        self.tracker.start_task("listener_test")
        
        assert ProgressEventType.TASK_CREATED in events_received
        assert ProgressEventType.TASK_STARTED in events_received
        
        # 移除监听器
        self.tracker.remove_event_listener(custom_listener)
        
        events_before_removal = len(events_received)
        self.tracker.update_progress("listener_test", current_sample=5)
        
        # 移除后不应该收到新事件
        assert len(events_received) == events_before_removal
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        # 创建不同状态的任务
        self.tracker.create_task("pending_task")
        self.tracker.create_task("running_task_1")
        self.tracker.create_task("running_task_2")
        self.tracker.start_task("running_task_1")
        self.tracker.start_task("running_task_2")
        
        stats = self.tracker.get_statistics()
        
        assert stats["active_tasks"] == 3
        assert stats["status_distribution"]["pending"] == 1
        assert stats["status_distribution"]["running"] == 2
        assert 0 <= stats["average_progress"] <= 100
        
        # 检查按状态分组的任务
        assert "pending_task" in stats["tasks_by_status"]["pending"]
        assert "running_task_1" in stats["tasks_by_status"]["running"]
        assert "running_task_2" in stats["tasks_by_status"]["running"]
    
    def test_nonexistent_task_operations(self):
        """测试对不存在任务的操作"""
        with pytest.raises(ValueError, match="任务 nonexistent 不存在"):
            self.tracker.start_task("nonexistent")
        
        with pytest.raises(ValueError, match="任务 nonexistent 不存在"):
            self.tracker.update_progress("nonexistent", current_sample=10)
        
        with pytest.raises(ValueError, match="任务 nonexistent 不存在"):
            self.tracker.complete_task("nonexistent")
        
        with pytest.raises(ValueError, match="任务 nonexistent 不存在"):
            self.tracker.fail_task("nonexistent", "error")
        
        with pytest.raises(ValueError, match="任务 nonexistent 不存在"):
            self.tracker.cancel_task("nonexistent")


class TestGlobalTracker:
    """全局跟踪器测试"""
    
    def test_get_global_tracker(self):
        """测试获取全局跟踪器"""
        tracker1 = get_global_tracker()
        tracker2 = get_global_tracker()
        
        # 应该返回同一个实例
        assert tracker1 is tracker2
    
    def test_set_global_tracker(self):
        """测试设置全局跟踪器"""
        # 创建临时数据库
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        try:
            storage = ProgressStorage(temp_db.name)
            custom_tracker = ProgressTracker(storage)
            
            set_global_tracker(custom_tracker)
            
            retrieved_tracker = get_global_tracker()
            assert retrieved_tracker is custom_tracker
            
        finally:
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)


if __name__ == "__main__":
    pytest.main([__file__])
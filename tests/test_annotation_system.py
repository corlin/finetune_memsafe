"""
专家标注系统单元测试
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from industry_evaluation.annotation.annotation_manager import (
    AnnotationManager, AnnotationDatabase, TaskAssignmentEngine,
    AnnotationTask, Annotation, Expert, AnnotationStatus, AnnotationTaskType
)
from industry_evaluation.annotation.quality_control import (
    QualityControlManager, AnnotationConsistencyChecker, ConsistencyLevel,
    QualityIssue, QualityIssueType
)
from industry_evaluation.models.data_models import SampleResult


class TestAnnotationDatabase:
    """标注数据库测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.database = AnnotationDatabase(self.temp_db.name)
    
    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_save_and_get_expert(self):
        """测试保存和获取专家"""
        expert = Expert(
            expert_id="expert_001",
            name="张三",
            email="zhangsan@example.com",
            expertise_areas=["机器学习", "自然语言处理"],
            qualification_level=3,
            max_concurrent_tasks=5,
            preferred_task_types=[AnnotationTaskType.QUALITY_ASSESSMENT]
        )
        
        # 保存专家
        self.database.save_expert(expert)
        
        # 获取专家
        retrieved_expert = self.database.get_expert("expert_001")
        
        assert retrieved_expert is not None
        assert retrieved_expert.expert_id == "expert_001"
        assert retrieved_expert.name == "张三"
        assert retrieved_expert.email == "zhangsan@example.com"
        assert retrieved_expert.expertise_areas == ["机器学习", "自然语言处理"]
        assert retrieved_expert.qualification_level == 3
        assert retrieved_expert.max_concurrent_tasks == 5
        assert AnnotationTaskType.QUALITY_ASSESSMENT in retrieved_expert.preferred_task_types
    
    def test_get_all_experts(self):
        """测试获取所有专家"""
        # 创建活跃专家
        active_expert = Expert(
            expert_id="active_001",
            name="活跃专家",
            email="active@example.com",
            expertise_areas=["AI"],
            active=True
        )
        
        # 创建非活跃专家
        inactive_expert = Expert(
            expert_id="inactive_001",
            name="非活跃专家",
            email="inactive@example.com",
            expertise_areas=["AI"],
            active=False
        )
        
        self.database.save_expert(active_expert)
        self.database.save_expert(inactive_expert)
        
        # 获取所有专家
        all_experts = self.database.get_all_experts(active_only=False)
        assert len(all_experts) == 2
        
        # 只获取活跃专家
        active_experts = self.database.get_all_experts(active_only=True)
        assert len(active_experts) == 1
        assert active_experts[0].expert_id == "active_001"
    
    def test_save_and_get_task(self):
        """测试保存和获取任务"""
        task = AnnotationTask(
            task_id="task_001",
            task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
            title="质量评估任务",
            description="评估模型输出质量",
            sample_id="sample_001",
            input_text="输入文本",
            model_output="模型输出",
            expected_output="期望输出",
            instructions="请评估输出质量",
            priority=3,
            estimated_time=30
        )
        
        # 保存任务
        self.database.save_task(task)
        
        # 获取任务
        retrieved_task = self.database.get_task("task_001")
        
        assert retrieved_task is not None
        assert retrieved_task.task_id == "task_001"
        assert retrieved_task.task_type == AnnotationTaskType.QUALITY_ASSESSMENT
        assert retrieved_task.title == "质量评估任务"
        assert retrieved_task.priority == 3
        assert retrieved_task.estimated_time == 30
    
    def test_get_tasks_by_status(self):
        """测试按状态获取任务"""
        # 创建不同状态的任务
        pending_task = AnnotationTask(
            task_id="pending_001",
            task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
            title="待处理任务",
            description="描述",
            sample_id="sample_001",
            input_text="输入",
            model_output="输出",
            status=AnnotationStatus.PENDING
        )
        
        completed_task = AnnotationTask(
            task_id="completed_001",
            task_type=AnnotationTaskType.ERROR_IDENTIFICATION,
            title="已完成任务",
            description="描述",
            sample_id="sample_002",
            input_text="输入",
            model_output="输出",
            status=AnnotationStatus.COMPLETED
        )
        
        self.database.save_task(pending_task)
        self.database.save_task(completed_task)
        
        # 获取待处理任务
        pending_tasks = self.database.get_tasks_by_status(AnnotationStatus.PENDING)
        assert len(pending_tasks) == 1
        assert pending_tasks[0].task_id == "pending_001"
        
        # 获取已完成任务
        completed_tasks = self.database.get_tasks_by_status(AnnotationStatus.COMPLETED)
        assert len(completed_tasks) == 1
        assert completed_tasks[0].task_id == "completed_001"
    
    def test_save_and_get_annotation(self):
        """测试保存和获取标注"""
        annotation = Annotation(
            annotation_id="annotation_001",
            task_id="task_001",
            expert_id="expert_001",
            annotation_data={"quality_score": 0.8, "issues": ["语法错误"]},
            quality_score=0.8,
            confidence=0.9,
            comments="输出质量较好，但有小错误",
            time_spent=25
        )
        
        # 保存标注
        self.database.save_annotation(annotation)
        
        # 获取任务的标注
        annotations = self.database.get_annotations_by_task("task_001")
        assert len(annotations) == 1
        assert annotations[0].annotation_id == "annotation_001"
        assert annotations[0].quality_score == 0.8
        assert annotations[0].confidence == 0.9
        assert annotations[0].time_spent == 25
        
        # 获取专家的标注
        expert_annotations = self.database.get_annotations_by_expert("expert_001")
        assert len(expert_annotations) == 1
        assert expert_annotations[0].annotation_id == "annotation_001"


class TestTaskAssignmentEngine:
    """任务分配引擎测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.database = AnnotationDatabase(self.temp_db.name)
        self.assignment_engine = TaskAssignmentEngine(self.database)
        
        # 创建测试专家
        self.expert = Expert(
            expert_id="expert_001",
            name="测试专家",
            email="expert@example.com",
            expertise_areas=["机器学习"],
            qualification_level=3,
            max_concurrent_tasks=3,
            preferred_task_types=[AnnotationTaskType.QUALITY_ASSESSMENT]
        )
        self.database.save_expert(self.expert)
    
    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_assign_task_to_specific_expert(self):
        """测试分配任务给指定专家"""
        task = AnnotationTask(
            task_id="task_001",
            task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
            title="测试任务",
            description="描述",
            sample_id="sample_001",
            input_text="输入",
            model_output="输出"
        )
        
        # 分配任务
        success = self.assignment_engine.assign_task(task, "expert_001")
        
        assert success is True
        assert task.assigned_expert == "expert_001"
        assert task.status == AnnotationStatus.IN_PROGRESS
        assert task.assigned_time is not None
    
    def test_assign_task_auto_assignment(self):
        """测试自动分配任务"""
        task = AnnotationTask(
            task_id="task_002",
            task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
            title="自动分配任务",
            description="描述",
            sample_id="sample_002",
            input_text="输入",
            model_output="输出"
        )
        
        # 自动分配任务
        success = self.assignment_engine.assign_task(task)
        
        assert success is True
        assert task.assigned_expert == "expert_001"
        assert task.status == AnnotationStatus.IN_PROGRESS
    
    def test_cannot_assign_to_overloaded_expert(self):
        """测试不能分配给超负荷专家"""
        # 创建多个任务使专家达到最大负载
        for i in range(3):
            task = AnnotationTask(
                task_id=f"task_{i}",
                task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
                title=f"任务{i}",
                description="描述",
                sample_id=f"sample_{i}",
                input_text="输入",
                model_output="输出",
                assigned_expert="expert_001",
                status=AnnotationStatus.IN_PROGRESS
            )
            self.database.save_task(task)
        
        # 尝试分配新任务
        new_task = AnnotationTask(
            task_id="task_new",
            task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
            title="新任务",
            description="描述",
            sample_id="sample_new",
            input_text="输入",
            model_output="输出"
        )
        
        success = self.assignment_engine.assign_task(new_task, "expert_001")
        assert success is False  # 应该分配失败
    
    def test_calculate_expert_score(self):
        """测试专家适合度分数计算"""
        task = AnnotationTask(
            task_id="task_score",
            task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
            title="评分任务",
            description="描述",
            sample_id="sample_score",
            input_text="输入",
            model_output="输出"
        )
        
        score = self.assignment_engine._calculate_expert_score(self.expert, task)
        
        assert isinstance(score, float)
        assert score > 0  # 应该有正分数
        
        # 专家偏好的任务类型应该有更高分数
        assert score > 0.5  # 因为任务类型匹配专家偏好


class TestAnnotationManager:
    """标注管理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.manager = AnnotationManager(self.temp_db.name)
    
    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_create_expert(self):
        """测试创建专家"""
        expert = self.manager.create_expert(
            name="李四",
            email="lisi@example.com",
            expertise_areas=["深度学习", "计算机视觉"],
            qualification_level=4,
            max_concurrent_tasks=8,
            preferred_task_types=[AnnotationTaskType.ERROR_IDENTIFICATION]
        )
        
        assert expert.name == "李四"
        assert expert.email == "lisi@example.com"
        assert expert.qualification_level == 4
        assert expert.max_concurrent_tasks == 8
        assert AnnotationTaskType.ERROR_IDENTIFICATION in expert.preferred_task_types
        
        # 验证专家已保存到数据库
        retrieved_expert = self.manager.database.get_expert(expert.expert_id)
        assert retrieved_expert is not None
        assert retrieved_expert.name == "李四"
    
    def test_create_annotation_task(self):
        """测试创建标注任务"""
        sample_result = SampleResult(
            sample_id="sample_test",
            input_text="测试输入文本",
            model_output="测试模型输出",
            expected_output="测试期望输出",
            dimension_scores={"quality": 0.7}
        )
        
        task = self.manager.create_annotation_task(
            sample_result=sample_result,
            task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
            title="质量评估",
            description="请评估模型输出的质量",
            instructions="按照1-5分评分",
            priority=2,
            estimated_time=20
        )
        
        assert task.sample_id == "sample_test"
        assert task.input_text == "测试输入文本"
        assert task.model_output == "测试模型输出"
        assert task.expected_output == "测试期望输出"
        assert task.task_type == AnnotationTaskType.QUALITY_ASSESSMENT
        assert task.priority == 2
        assert task.estimated_time == 20
    
    def test_submit_annotation(self):
        """测试提交标注"""
        # 先创建专家和任务
        expert = self.manager.create_expert(
            name="标注专家",
            email="annotator@example.com",
            expertise_areas=["AI"]
        )
        
        sample_result = SampleResult(
            sample_id="sample_annotation",
            input_text="输入",
            model_output="输出",
            expected_output="期望",
            dimension_scores={}
        )
        
        task = self.manager.create_annotation_task(
            sample_result=sample_result,
            task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
            title="标注任务",
            description="描述"
        )
        
        # 提交标注
        annotation = self.manager.submit_annotation(
            task_id=task.task_id,
            expert_id=expert.expert_id,
            annotation_data={"score": 4, "comments": "质量不错"},
            quality_score=0.8,
            confidence=0.9,
            comments="整体质量良好",
            time_spent=15
        )
        
        assert annotation.task_id == task.task_id
        assert annotation.expert_id == expert.expert_id
        assert annotation.quality_score == 0.8
        assert annotation.confidence == 0.9
        assert annotation.time_spent == 15
        
        # 验证任务状态已更新
        updated_task = self.manager.database.get_task(task.task_id)
        assert updated_task.status == AnnotationStatus.COMPLETED
        assert updated_task.completed_time is not None
    
    def test_get_expert_workload(self):
        """测试获取专家工作负载"""
        # 创建专家
        expert = self.manager.create_expert(
            name="工作负载测试专家",
            email="workload@example.com",
            expertise_areas=["AI"]
        )
        
        # 创建一些任务和标注
        for i in range(3):
            sample_result = SampleResult(
                sample_id=f"sample_{i}",
                input_text=f"输入{i}",
                model_output=f"输出{i}",
                expected_output=f"期望{i}",
                dimension_scores={}
            )
            
            task = self.manager.create_annotation_task(
                sample_result=sample_result,
                task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
                title=f"任务{i}",
                description="描述"
            )
            
            if i < 2:  # 完成前两个任务
                self.manager.submit_annotation(
                    task_id=task.task_id,
                    expert_id=expert.expert_id,
                    annotation_data={"score": 3 + i},
                    quality_score=0.6 + i * 0.1,
                    time_spent=10 + i * 5
                )
        
        # 获取工作负载
        workload = self.manager.get_expert_workload(expert.expert_id)
        
        assert workload["total_tasks"] == 3
        assert workload["total_annotations"] == 2
        assert workload["active_tasks"] == 0  # 没有进行中的任务
        assert workload["avg_quality_score"] > 0


class TestAnnotationConsistencyChecker:
    """标注一致性检查器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.checker = AnnotationConsistencyChecker()
    
    def test_check_high_consistency(self):
        """测试高一致性标注"""
        annotations = [
            Annotation(
                annotation_id="ann_1",
                task_id="task_001",
                expert_id="expert_1",
                annotation_data={"score": 4},
                quality_score=0.8,
                confidence=0.9
            ),
            Annotation(
                annotation_id="ann_2",
                task_id="task_001",
                expert_id="expert_2",
                annotation_data={"score": 4},
                quality_score=0.82,
                confidence=0.85
            ),
            Annotation(
                annotation_id="ann_3",
                task_id="task_001",
                expert_id="expert_3",
                annotation_data={"score": 4},
                quality_score=0.78,
                confidence=0.9
            )
        ]
        
        report = self.checker.check_task_consistency(annotations)
        
        assert report.consistency_level in [ConsistencyLevel.HIGH, ConsistencyLevel.VERY_HIGH]
        assert report.consistency_score > 0.7
        assert len(report.outliers) == 0
        assert report.consensus_annotation is not None
    
    def test_check_low_consistency(self):
        """测试低一致性标注"""
        annotations = [
            Annotation(
                annotation_id="ann_1",
                task_id="task_001",
                expert_id="expert_1",
                annotation_data={"score": 1},
                quality_score=0.2,
                confidence=0.9
            ),
            Annotation(
                annotation_id="ann_2",
                task_id="task_001",
                expert_id="expert_2",
                annotation_data={"score": 5},
                quality_score=0.9,
                confidence=0.8
            ),
            Annotation(
                annotation_id="ann_3",
                task_id="task_001",
                expert_id="expert_3",
                annotation_data={"score": 3},
                quality_score=0.6,
                confidence=0.7
            )
        ]
        
        report = self.checker.check_task_consistency(annotations)
        
        assert report.consistency_level in [ConsistencyLevel.LOW, ConsistencyLevel.VERY_LOW]
        assert report.consistency_score < 0.5
        assert len(report.outliers) > 0  # 应该有异常值
    
    def test_single_annotation(self):
        """测试单个标注的情况"""
        annotations = [
            Annotation(
                annotation_id="ann_1",
                task_id="task_001",
                expert_id="expert_1",
                annotation_data={"score": 4},
                quality_score=0.8,
                confidence=0.9
            )
        ]
        
        report = self.checker.check_task_consistency(annotations)
        
        assert report.consistency_level == ConsistencyLevel.VERY_HIGH
        assert report.consistency_score == 1.0
        assert len(report.outliers) == 0


class TestQualityControlManager:
    """质量控制管理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.database = AnnotationDatabase(self.temp_db.name)
        self.quality_manager = QualityControlManager(self.database)
        
        # 创建测试数据
        self.expert = Expert(
            expert_id="expert_quality",
            name="质量测试专家",
            email="quality@example.com",
            expertise_areas=["AI"]
        )
        self.database.save_expert(self.expert)
    
    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_analyze_annotation_quality(self):
        """测试分析标注质量"""
        # 创建任务
        task = AnnotationTask(
            task_id="quality_task",
            task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
            title="质量测试任务",
            description="描述",
            sample_id="sample_quality",
            input_text="输入",
            model_output="输出",
            status=AnnotationStatus.COMPLETED
        )
        self.database.save_task(task)
        
        # 创建一致性较好的标注
        annotations = [
            Annotation(
                annotation_id="ann_q1",
                task_id="quality_task",
                expert_id="expert_quality",
                annotation_data={"score": 4},
                quality_score=0.8,
                confidence=0.9
            ),
            Annotation(
                annotation_id="ann_q2",
                task_id="quality_task",
                expert_id="expert_quality",
                annotation_data={"score": 4},
                quality_score=0.82,
                confidence=0.85
            )
        ]
        
        for annotation in annotations:
            self.database.save_annotation(annotation)
        
        # 分析质量
        report = self.quality_manager.analyze_annotation_quality("quality_task")
        
        assert report.task_id == "quality_task"
        assert len(report.annotations) == 2
        assert report.consistency_level in [ConsistencyLevel.HIGH, ConsistencyLevel.VERY_HIGH]
    
    def test_identify_quality_issues(self):
        """测试识别质量问题"""
        # 创建有问题的标注数据
        
        # 1. 创建只有一个标注的任务（标注不足）
        insufficient_task = AnnotationTask(
            task_id="insufficient_task",
            task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
            title="标注不足任务",
            description="描述",
            sample_id="sample_insufficient",
            input_text="输入",
            model_output="输出",
            status=AnnotationStatus.COMPLETED
        )
        self.database.save_task(insufficient_task)
        
        single_annotation = Annotation(
            annotation_id="single_ann",
            task_id="insufficient_task",
            expert_id="expert_quality",
            annotation_data={"score": 3},
            quality_score=0.6
        )
        self.database.save_annotation(single_annotation)
        
        # 2. 创建一致性差的任务
        inconsistent_task = AnnotationTask(
            task_id="inconsistent_task",
            task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
            title="不一致任务",
            description="描述",
            sample_id="sample_inconsistent",
            input_text="输入",
            model_output="输出",
            status=AnnotationStatus.COMPLETED
        )
        self.database.save_task(inconsistent_task)
        
        inconsistent_annotations = [
            Annotation(
                annotation_id="inc_ann1",
                task_id="inconsistent_task",
                expert_id="expert_quality",
                annotation_data={"score": 1},
                quality_score=0.1
            ),
            Annotation(
                annotation_id="inc_ann2",
                task_id="inconsistent_task",
                expert_id="expert_quality",
                annotation_data={"score": 5},
                quality_score=0.9
            )
        ]
        
        for annotation in inconsistent_annotations:
            self.database.save_annotation(annotation)
        
        # 识别质量问题
        issues = self.quality_manager.identify_quality_issues()
        
        assert len(issues) > 0
        
        # 检查是否识别出标注不足问题
        insufficient_issues = [i for i in issues if i.issue_type == QualityIssueType.INSUFFICIENT_ANNOTATIONS]
        assert len(insufficient_issues) > 0
        
        # 检查是否识别出一致性问题
        consistency_issues = [i for i in issues if i.issue_type == QualityIssueType.LOW_CONSISTENCY]
        assert len(consistency_issues) > 0
    
    def test_generate_quality_report(self):
        """测试生成质量报告"""
        # 创建一些测试数据
        task = AnnotationTask(
            task_id="report_task",
            task_type=AnnotationTaskType.QUALITY_ASSESSMENT,
            title="报告测试任务",
            description="描述",
            sample_id="sample_report",
            input_text="输入",
            model_output="输出",
            status=AnnotationStatus.COMPLETED
        )
        self.database.save_task(task)
        
        annotation = Annotation(
            annotation_id="report_ann",
            task_id="report_task",
            expert_id="expert_quality",
            annotation_data={"score": 4},
            quality_score=0.8
        )
        self.database.save_annotation(annotation)
        
        # 生成报告
        report = self.quality_manager.generate_quality_report()
        
        assert "summary" in report
        assert "issues_by_severity" in report
        assert "issues_by_type" in report
        assert "expert_quality" in report
        assert "recommendations" in report
        
        # 检查专家质量信息
        assert "expert_quality" in report["expert_quality"]
        expert_info = report["expert_quality"]["expert_quality"]
        assert expert_info["name"] == "质量测试专家"
        assert expert_info["total_annotations"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
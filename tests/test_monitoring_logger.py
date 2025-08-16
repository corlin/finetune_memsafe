"""
监控和日志系统测试

本模块测试MonitoringLogger类的各种功能，包括系统监控、操作跟踪、
日志记录和报告生成功能。
"""

import unittest
import tempfile
import shutil
import os
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# 导入被测试的模块
from src.monitoring_logger import (
    MonitoringLogger, MonitoringLevel, OperationStatus, 
    SystemMetrics, OperationMetrics, MonitoringReport
)
from src.export_models import ExportConfiguration, LogLevel


class TestSystemMetrics(unittest.TestCase):
    """系统指标测试类"""
    
    def test_system_metrics_creation(self):
        """测试系统指标创建"""
        timestamp = datetime.now()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_percent=70.0
        )
        
        self.assertEqual(metrics.timestamp, timestamp)
        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.memory_percent, 60.0)
        self.assertEqual(metrics.disk_percent, 70.0)


class TestOperationMetrics(unittest.TestCase):
    """操作指标测试类"""
    
    def test_operation_metrics_creation(self):
        """测试操作指标创建"""
        start_time = datetime.now()
        operation = OperationMetrics(
            operation_id="test_op_001",
            operation_name="test_operation",
            status=OperationStatus.PENDING,
            start_time=start_time,
            total_steps=10
        )
        
        self.assertEqual(operation.operation_id, "test_op_001")
        self.assertEqual(operation.operation_name, "test_operation")
        self.assertEqual(operation.status, OperationStatus.PENDING)
        self.assertEqual(operation.start_time, start_time)
        self.assertEqual(operation.total_steps, 10)
        self.assertEqual(operation.progress_percent, 0.0)
    
    def test_update_progress(self):
        """测试进度更新"""
        operation = OperationMetrics(
            operation_id="test_op",
            operation_name="test",
            status=OperationStatus.PENDING,
            start_time=datetime.now(),
            total_steps=10
        )
        
        # 更新进度
        operation.update_progress(5, "处理中...")
        
        self.assertEqual(operation.current_step, 5)
        self.assertEqual(operation.progress_percent, 50.0)
        self.assertEqual(operation.current_task, "处理中...")
        self.assertEqual(operation.status, OperationStatus.RUNNING)
    
    def test_complete_success(self):
        """测试成功完成"""
        start_time = datetime.now()
        operation = OperationMetrics(
            operation_id="test_op",
            operation_name="test",
            status=OperationStatus.RUNNING,
            start_time=start_time
        )
        
        # 等待一小段时间
        time.sleep(0.01)
        
        # 完成操作
        operation.complete(success=True)
        
        self.assertEqual(operation.status, OperationStatus.COMPLETED)
        self.assertEqual(operation.progress_percent, 100.0)
        self.assertIsNotNone(operation.end_time)
        self.assertGreater(operation.duration_seconds, 0)
        self.assertIsNone(operation.error_message)
    
    def test_complete_failure(self):
        """测试失败完成"""
        operation = OperationMetrics(
            operation_id="test_op",
            operation_name="test",
            status=OperationStatus.RUNNING,
            start_time=datetime.now()
        )
        
        # 完成操作（失败）
        operation.complete(success=False, error_message="测试错误")
        
        self.assertEqual(operation.status, OperationStatus.FAILED)
        self.assertIsNotNone(operation.end_time)
        self.assertEqual(operation.error_message, "测试错误")


class TestMonitoringReport(unittest.TestCase):
    """监控报告测试类"""
    
    def test_monitoring_report_creation(self):
        """测试监控报告创建"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=5)
        
        report = MonitoringReport(
            report_id="test_report",
            start_time=start_time,
            end_time=end_time
        )
        
        self.assertEqual(report.report_id, "test_report")
        self.assertEqual(report.start_time, start_time)
        self.assertEqual(report.end_time, end_time)
        self.assertEqual(report.total_operations, 0)
    
    def test_calculate_summary(self):
        """测试摘要计算"""
        report = MonitoringReport(
            report_id="test_report",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=5)
        )
        
        # 添加操作
        op1 = OperationMetrics(
            operation_id="op1",
            operation_name="operation1",
            status=OperationStatus.COMPLETED,
            start_time=datetime.now(),
            peak_memory_mb=100.0,
            avg_memory_mb=80.0,
            peak_cpu_percent=50.0,
            avg_cpu_percent=30.0,
            duration_seconds=10.0
        )
        
        op2 = OperationMetrics(
            operation_id="op2",
            operation_name="operation2",
            status=OperationStatus.FAILED,
            start_time=datetime.now(),
            peak_memory_mb=150.0,
            avg_memory_mb=120.0,
            peak_cpu_percent=70.0,
            avg_cpu_percent=50.0,
            duration_seconds=5.0,
            error_message="测试错误"
        )
        
        report.operations = [op1, op2]
        report.calculate_summary()
        
        self.assertEqual(report.total_operations, 2)
        self.assertEqual(report.successful_operations, 1)
        self.assertEqual(report.failed_operations, 1)
        self.assertEqual(report.peak_memory_usage_mb, 150.0)
        self.assertEqual(report.avg_memory_usage_mb, 100.0)
        self.assertEqual(report.peak_cpu_usage_percent, 70.0)
        self.assertEqual(report.avg_cpu_usage_percent, 40.0)
        self.assertEqual(report.avg_operation_duration_seconds, 7.5)
        self.assertIn("operation2: 测试错误", report.errors)


class TestMonitoringLogger(unittest.TestCase):
    """监控日志器测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ExportConfiguration(
            checkpoint_path="test_checkpoint",
            base_model_name="test_model",
            output_directory=self.temp_dir,
            enable_progress_monitoring=True,
            log_level=LogLevel.INFO
        )
    
    def tearDown(self):
        """测试后的清理"""
        # 清理所有日志处理器以释放文件锁
        import logging
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)
        
        # 清理所有以monitoring_开头的logger
        for name in list(logging.Logger.manager.loggerDict.keys()):
            if name.startswith('monitoring_'):
                logger = logging.getLogger(name)
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
        
        # 等待一小段时间让文件句柄释放
        import time
        time.sleep(0.1)
        
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # 如果仍然无法删除，忽略错误
                pass
    
    def test_monitoring_logger_initialization(self):
        """测试监控日志器初始化"""
        logger = MonitoringLogger(self.config, MonitoringLevel.STANDARD)
        
        self.assertEqual(logger.config, self.config)
        self.assertEqual(logger.monitoring_level, MonitoringLevel.STANDARD)
        self.assertFalse(logger.is_monitoring)
        self.assertIsNone(logger.current_operation)
        self.assertEqual(len(logger.operations), 0)
        self.assertEqual(len(logger.system_metrics_history), 0)
    
    def test_start_stop_monitoring(self):
        """测试开始和停止监控"""
        logger = MonitoringLogger(self.config, MonitoringLevel.STANDARD)
        
        # 开始监控
        logger.start_monitoring()
        self.assertTrue(logger.is_monitoring)
        self.assertIsNotNone(logger.monitoring_thread)
        
        # 等待一小段时间让监控线程运行
        time.sleep(0.1)
        
        # 停止监控
        logger.stop_monitoring()
        self.assertFalse(logger.is_monitoring)
    
    def test_start_operation(self):
        """测试开始操作"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        operation_id = logger.start_operation("test_operation", total_steps=5)
        
        self.assertIsNotNone(operation_id)
        self.assertIn(operation_id, logger.operations)
        self.assertEqual(logger.current_operation, operation_id)
        
        operation = logger.operations[operation_id]
        self.assertEqual(operation.operation_name, "test_operation")
        self.assertEqual(operation.total_steps, 5)
        self.assertEqual(operation.status, OperationStatus.PENDING)
    
    def test_update_operation_progress(self):
        """测试更新操作进度"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        operation_id = logger.start_operation("test_operation", total_steps=10)
        logger.update_operation_progress(operation_id, 3, "处理步骤3")
        
        operation = logger.operations[operation_id]
        self.assertEqual(operation.current_step, 3)
        self.assertEqual(operation.progress_percent, 30.0)
        self.assertEqual(operation.current_task, "处理步骤3")
        self.assertEqual(operation.status, OperationStatus.RUNNING)
    
    def test_complete_operation_success(self):
        """测试成功完成操作"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        operation_id = logger.start_operation("test_operation")
        logger.complete_operation(operation_id, success=True)
        
        operation = logger.operations[operation_id]
        self.assertEqual(operation.status, OperationStatus.COMPLETED)
        self.assertEqual(operation.progress_percent, 100.0)
        self.assertIsNone(operation.error_message)
        self.assertIsNone(logger.current_operation)
    
    def test_complete_operation_failure(self):
        """测试失败完成操作"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        operation_id = logger.start_operation("test_operation")
        logger.complete_operation(operation_id, success=False, error_message="测试错误")
        
        operation = logger.operations[operation_id]
        self.assertEqual(operation.status, OperationStatus.FAILED)
        self.assertEqual(operation.error_message, "测试错误")
        self.assertIsNone(logger.current_operation)
    
    def test_log_warning(self):
        """测试记录警告"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        operation_id = logger.start_operation("test_operation")
        logger.log_warning(operation_id, "这是一个警告")
        
        operation = logger.operations[operation_id]
        self.assertIn("这是一个警告", operation.warnings)
    
    def test_log_error(self):
        """测试记录错误"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        operation_id = logger.start_operation("test_operation")
        logger.log_error(operation_id, "这是一个错误")
        
        operation = logger.operations[operation_id]
        self.assertEqual(operation.error_message, "这是一个错误")
    
    def test_log_error_with_exception(self):
        """测试记录带异常的错误"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        operation_id = logger.start_operation("test_operation")
        test_exception = ValueError("测试异常")
        logger.log_error(operation_id, "操作失败", test_exception)
        
        operation = logger.operations[operation_id]
        self.assertIn("操作失败: 测试异常", operation.error_message)
    
    @patch('src.monitoring_logger.psutil')
    def test_get_current_metrics(self, mock_psutil):
        """测试获取当前系统指标"""
        # 设置mock返回值
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.cpu_count.return_value = 4
        
        mock_memory = Mock()
        mock_memory.total = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.used = 4 * 1024 * 1024 * 1024   # 4GB
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.total = 100 * 1024 * 1024 * 1024  # 100GB
        mock_disk.used = 50 * 1024 * 1024 * 1024    # 50GB
        mock_disk.free = 50 * 1024 * 1024 * 1024    # 50GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_process = Mock()
        mock_process_memory = Mock()
        mock_process_memory.rss = 512 * 1024 * 1024  # 512MB
        mock_process.memory_info.return_value = mock_process_memory
        mock_process.cpu_percent.return_value = 25.0
        mock_psutil.Process.return_value = mock_process
        
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        metrics = logger.get_current_metrics()
        
        self.assertIsInstance(metrics, SystemMetrics)
        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.cpu_count, 4)
        self.assertEqual(metrics.memory_percent, 50.0)
        self.assertEqual(metrics.disk_percent, 50.0)
        self.assertEqual(metrics.process_memory_mb, 512.0)
        self.assertEqual(metrics.process_cpu_percent, 25.0)
    
    def test_monitor_operation_context_manager_success(self):
        """测试操作监控上下文管理器（成功）"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        with logger.monitor_operation("test_context_operation", total_steps=3) as operation_id:
            self.assertIsNotNone(operation_id)
            self.assertIn(operation_id, logger.operations)
            
            # 模拟一些工作
            logger.update_operation_progress(operation_id, 1, "步骤1")
            logger.update_operation_progress(operation_id, 2, "步骤2")
            logger.update_operation_progress(operation_id, 3, "步骤3")
        
        # 检查操作是否成功完成
        operation = logger.operations[operation_id]
        self.assertEqual(operation.status, OperationStatus.COMPLETED)
        self.assertEqual(operation.progress_percent, 100.0)
    
    def test_monitor_operation_context_manager_failure(self):
        """测试操作监控上下文管理器（失败）"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        with self.assertRaises(ValueError):
            with logger.monitor_operation("test_context_operation") as operation_id:
                self.assertIsNotNone(operation_id)
                # 模拟异常
                raise ValueError("测试异常")
        
        # 检查操作是否标记为失败
        operation = logger.operations[operation_id]
        self.assertEqual(operation.status, OperationStatus.FAILED)
        self.assertIn("测试异常", operation.error_message)
    
    def test_progress_callback(self):
        """测试进度回调"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        callback_calls = []
        
        def progress_callback(op_id, progress, task):
            callback_calls.append((op_id, progress, task))
        
        logger.add_progress_callback(progress_callback)
        
        operation_id = logger.start_operation("test_operation", total_steps=2)
        logger.update_operation_progress(operation_id, 1, "任务1")
        
        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(callback_calls[0][0], operation_id)
        self.assertEqual(callback_calls[0][1], 50.0)
        self.assertEqual(callback_calls[0][2], "任务1")
    
    def test_status_callback(self):
        """测试状态回调"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        callback_calls = []
        
        def status_callback(op_id, status):
            callback_calls.append((op_id, status))
        
        logger.add_status_callback(status_callback)
        
        operation_id = logger.start_operation("test_operation")
        logger.complete_operation(operation_id, success=True)
        
        self.assertEqual(len(callback_calls), 2)  # started 和 completed
        self.assertEqual(callback_calls[0][1], "started")
        self.assertEqual(callback_calls[1][1], "completed")
    
    def test_generate_monitoring_report(self):
        """测试生成监控报告"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        # 创建一些操作
        op1_id = logger.start_operation("operation1", total_steps=2)
        logger.update_operation_progress(op1_id, 1, "步骤1")
        logger.update_operation_progress(op1_id, 2, "步骤2")
        logger.complete_operation(op1_id, success=True)
        
        op2_id = logger.start_operation("operation2")
        logger.complete_operation(op2_id, success=False, error_message="测试错误")
        
        # 生成报告
        report = logger.generate_monitoring_report()
        
        self.assertIsInstance(report, MonitoringReport)
        self.assertEqual(report.total_operations, 2)
        self.assertEqual(report.successful_operations, 1)
        self.assertEqual(report.failed_operations, 1)
        self.assertIn("operation2: 测试错误", report.errors)
    
    def test_generate_monitoring_report_with_output(self):
        """测试生成监控报告并保存"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        # 创建一个操作
        operation_id = logger.start_operation("test_operation")
        logger.complete_operation(operation_id, success=True)
        
        # 生成报告并保存
        report_dir = os.path.join(self.temp_dir, "reports")
        report = logger.generate_monitoring_report(report_dir)
        
        # 检查报告文件是否生成
        self.assertTrue(os.path.exists(report_dir))
        
        json_file = Path(report_dir) / f"{report.report_id}.json"
        html_file = Path(report_dir) / f"{report.report_id}.html"
        
        self.assertTrue(json_file.exists())
        self.assertTrue(html_file.exists())
        
        # 检查JSON报告内容
        with open(json_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
            self.assertEqual(report_data['summary']['total_operations'], 1)
            self.assertEqual(report_data['summary']['successful_operations'], 1)
    
    def test_get_operation_status(self):
        """测试获取操作状态"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        operation_id = logger.start_operation("test_operation")
        
        # 获取操作状态
        status = logger.get_operation_status(operation_id)
        self.assertIsNotNone(status)
        self.assertEqual(status.operation_id, operation_id)
        
        # 获取不存在的操作状态
        non_existent_status = logger.get_operation_status("non_existent")
        self.assertIsNone(non_existent_status)
    
    def test_get_all_operations(self):
        """测试获取所有操作"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        # 初始状态
        operations = logger.get_all_operations()
        self.assertEqual(len(operations), 0)
        
        # 创建操作
        op1_id = logger.start_operation("operation1")
        op2_id = logger.start_operation("operation2")
        
        operations = logger.get_all_operations()
        self.assertEqual(len(operations), 2)
        
        operation_ids = [op.operation_id for op in operations]
        self.assertIn(op1_id, operation_ids)
        self.assertIn(op2_id, operation_ids)
    
    def test_clear_completed_operations(self):
        """测试清理已完成的操作"""
        logger = MonitoringLogger(self.config, MonitoringLevel.MINIMAL)
        
        # 创建不同状态的操作
        op1_id = logger.start_operation("completed_op")
        logger.complete_operation(op1_id, success=True)
        
        op2_id = logger.start_operation("failed_op")
        logger.complete_operation(op2_id, success=False, error_message="错误")
        
        op3_id = logger.start_operation("running_op")
        # 不完成这个操作
        
        # 清理前有3个操作
        self.assertEqual(len(logger.operations), 3)
        
        # 清理已完成的操作
        logger.clear_completed_operations()
        
        # 清理后只剩1个运行中的操作
        self.assertEqual(len(logger.operations), 1)
        self.assertIn(op3_id, logger.operations)
        self.assertNotIn(op1_id, logger.operations)
        self.assertNotIn(op2_id, logger.operations)
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with MonitoringLogger(self.config, MonitoringLevel.MINIMAL) as logger:
            self.assertTrue(logger.is_monitoring)
            
            # 创建一个操作
            operation_id = logger.start_operation("context_test")
            logger.complete_operation(operation_id, success=True)
        
        # 退出上下文后监控应该停止
        self.assertFalse(logger.is_monitoring)
        
        # 检查报告是否生成
        report_dir = Path(self.config.output_directory) / "monitoring_reports"
        self.assertTrue(report_dir.exists())


if __name__ == '__main__':
    unittest.main()
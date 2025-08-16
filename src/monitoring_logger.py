"""
监控和日志系统

本模块实现了完整的监控和日志功能，包括实时进度监控、内存和磁盘使用情况监控、
详细的操作日志记录和错误追踪，以及导出过程的状态报告和摘要生成。
"""

import os
import time
import json
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import logging.handlers
from contextlib import contextmanager
import queue
import traceback

from .export_models import ExportConfiguration, LogLevel
from .export_utils import format_size


class MonitoringLevel(Enum):
    """监控级别"""
    MINIMAL = "minimal"      # 最小监控
    STANDARD = "standard"    # 标准监控
    DETAILED = "detailed"    # 详细监控
    DEBUG = "debug"          # 调试监控


class OperationStatus(Enum):
    """操作状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    
    # CPU指标
    cpu_percent: float = 0.0
    cpu_count: int = 0
    
    # 内存指标
    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_percent: float = 0.0
    
    # 磁盘指标
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_percent: float = 0.0
    
    # 进程指标
    process_memory_mb: float = 0.0
    process_cpu_percent: float = 0.0
    
    # GPU指标（如果可用）
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_percent: float = 0.0


@dataclass
class OperationMetrics:
    """操作指标"""
    operation_id: str
    operation_name: str
    status: OperationStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # 进度信息
    current_step: int = 0
    total_steps: int = 0
    progress_percent: float = 0.0
    current_task: str = ""
    
    # 性能指标
    duration_seconds: float = 0.0
    throughput: float = 0.0  # 处理速度
    
    # 资源使用
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    
    # 错误信息
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def update_progress(self, current_step: int, task_description: str = ""):
        """更新进度"""
        self.current_step = current_step
        self.progress_percent = (current_step / self.total_steps * 100) if self.total_steps > 0 else 0
        self.current_task = task_description
        
        if self.status == OperationStatus.PENDING:
            self.status = OperationStatus.RUNNING
    
    def complete(self, success: bool = True, error_message: str = None):
        """完成操作"""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        if success:
            self.status = OperationStatus.COMPLETED
            self.progress_percent = 100.0
        else:
            self.status = OperationStatus.FAILED
            self.error_message = error_message


@dataclass
class MonitoringReport:
    """监控报告"""
    report_id: str
    start_time: datetime
    end_time: datetime
    
    # 操作摘要
    operations: List[OperationMetrics] = field(default_factory=list)
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    
    # 系统资源摘要
    peak_memory_usage_mb: float = 0.0
    avg_memory_usage_mb: float = 0.0
    peak_cpu_usage_percent: float = 0.0
    avg_cpu_usage_percent: float = 0.0
    total_disk_usage_gb: float = 0.0
    
    # 性能摘要
    total_duration_seconds: float = 0.0
    avg_operation_duration_seconds: float = 0.0
    
    # 问题和警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def calculate_summary(self):
        """计算报告摘要"""
        self.total_operations = len(self.operations)
        self.successful_operations = sum(1 for op in self.operations if op.status == OperationStatus.COMPLETED)
        self.failed_operations = sum(1 for op in self.operations if op.status == OperationStatus.FAILED)
        
        if self.operations:
            self.peak_memory_usage_mb = max(op.peak_memory_mb for op in self.operations)
            self.avg_memory_usage_mb = sum(op.avg_memory_mb for op in self.operations) / len(self.operations)
            self.peak_cpu_usage_percent = max(op.peak_cpu_percent for op in self.operations)
            self.avg_cpu_usage_percent = sum(op.avg_cpu_percent for op in self.operations) / len(self.operations)
            
            completed_ops = [op for op in self.operations if op.duration_seconds > 0]
            if completed_ops:
                self.avg_operation_duration_seconds = sum(op.duration_seconds for op in completed_ops) / len(completed_ops)
        
        self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        # 收集错误和警告
        for op in self.operations:
            if op.error_message:
                self.errors.append(f"{op.operation_name}: {op.error_message}")
            self.warnings.extend([f"{op.operation_name}: {w}" for w in op.warnings])


class MonitoringLogger:
    """监控日志器"""
    
    def __init__(self, config: ExportConfiguration, monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD):
        """
        初始化监控日志器
        
        Args:
            config: 导出配置
            monitoring_level: 监控级别
        """
        self.config = config
        self.monitoring_level = monitoring_level
        
        # 监控状态
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # 操作跟踪
        self.operations: Dict[str, OperationMetrics] = {}
        self.current_operation: Optional[str] = None
        
        # 系统指标历史
        self.system_metrics_history: List[SystemMetrics] = []
        self.metrics_collection_interval = 1.0  # 秒
        
        # 日志设置
        self.logger = self._setup_logger()
        
        # 回调函数
        self.progress_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        
        # 报告生成
        self.monitoring_start_time = datetime.now()
        
        self.logger.info(f"监控日志器初始化完成，监控级别: {monitoring_level.value}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(f"monitoring_{id(self)}")
        logger.setLevel(getattr(logging, self.config.log_level.value))
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 创建日志目录
        log_dir = Path(self.config.output_directory) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件处理器 - 详细日志
        log_file = log_dir / f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器 - 简化输出
        if self.config.enable_progress_monitoring:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
        
        return logger
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        self.monitoring_start_time = datetime.now()
        
        if self.monitoring_level in [MonitoringLevel.STANDARD, MonitoringLevel.DETAILED, MonitoringLevel.DEBUG]:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
        
        self.logger.info("开始系统监控")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("停止系统监控")
    
    def start_operation(self, operation_name: str, total_steps: int = 0, operation_id: str = None) -> str:
        """
        开始操作监控
        
        Args:
            operation_name: 操作名称
            total_steps: 总步骤数
            operation_id: 操作ID（可选）
            
        Returns:
            str: 操作ID
        """
        if operation_id is None:
            operation_id = f"{operation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        operation = OperationMetrics(
            operation_id=operation_id,
            operation_name=operation_name,
            status=OperationStatus.PENDING,
            start_time=datetime.now(),
            total_steps=total_steps
        )
        
        self.operations[operation_id] = operation
        self.current_operation = operation_id
        
        self.logger.info(f"开始操作: {operation_name} (ID: {operation_id})")
        self._notify_status_callbacks(operation_id, "started")
        
        return operation_id
    
    def update_operation_progress(self, operation_id: str, current_step: int, task_description: str = ""):
        """
        更新操作进度
        
        Args:
            operation_id: 操作ID
            current_step: 当前步骤
            task_description: 任务描述
        """
        if operation_id not in self.operations:
            return
        
        operation = self.operations[operation_id]
        operation.update_progress(current_step, task_description)
        
        # 更新资源使用情况
        if self.system_metrics_history:
            latest_metrics = self.system_metrics_history[-1]
            operation.peak_memory_mb = max(operation.peak_memory_mb, latest_metrics.process_memory_mb)
            operation.peak_cpu_percent = max(operation.peak_cpu_percent, latest_metrics.process_cpu_percent)
        
        self.logger.info(f"操作进度更新: {operation.operation_name} - {operation.progress_percent:.1f}% - {task_description}")
        self._notify_progress_callbacks(operation_id, operation.progress_percent, task_description)
    
    def complete_operation(self, operation_id: str, success: bool = True, error_message: str = None):
        """
        完成操作
        
        Args:
            operation_id: 操作ID
            success: 是否成功
            error_message: 错误信息（如果失败）
        """
        if operation_id not in self.operations:
            return
        
        operation = self.operations[operation_id]
        operation.complete(success, error_message)
        
        # 计算平均资源使用
        if self.system_metrics_history:
            recent_metrics = [m for m in self.system_metrics_history 
                            if m.timestamp >= operation.start_time]
            if recent_metrics:
                operation.avg_memory_mb = sum(m.process_memory_mb for m in recent_metrics) / len(recent_metrics)
                operation.avg_cpu_percent = sum(m.process_cpu_percent for m in recent_metrics) / len(recent_metrics)
        
        status_msg = "成功完成" if success else f"失败: {error_message}"
        self.logger.info(f"操作完成: {operation.operation_name} - {status_msg} (耗时: {operation.duration_seconds:.2f}秒)")
        
        self._notify_status_callbacks(operation_id, "completed" if success else "failed")
        
        if operation_id == self.current_operation:
            self.current_operation = None
    
    def log_warning(self, operation_id: str, warning_message: str):
        """
        记录警告
        
        Args:
            operation_id: 操作ID
            warning_message: 警告信息
        """
        if operation_id in self.operations:
            self.operations[operation_id].warnings.append(warning_message)
        
        self.logger.warning(f"操作警告 [{operation_id}]: {warning_message}")
    
    def log_error(self, operation_id: str, error_message: str, exception: Exception = None):
        """
        记录错误
        
        Args:
            operation_id: 操作ID
            error_message: 错误信息
            exception: 异常对象（可选）
        """
        if exception:
            error_details = f"{error_message}: {str(exception)}"
            if self.monitoring_level == MonitoringLevel.DEBUG:
                error_details += f"\n{traceback.format_exc()}"
        else:
            error_details = error_message
        
        if operation_id in self.operations:
            self.operations[operation_id].error_message = error_details
        
        self.logger.error(f"操作错误 [{operation_id}]: {error_details}")
    
    def get_current_metrics(self) -> SystemMetrics:
        """获取当前系统指标"""
        try:
            # CPU指标
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # 内存指标
            memory = psutil.virtual_memory()
            memory_total_mb = memory.total / (1024 * 1024)
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            memory_percent = memory.percent
            
            # 磁盘指标
            disk = psutil.disk_usage(self.config.output_directory)
            disk_total_gb = disk.total / (1024 * 1024 * 1024)
            disk_used_gb = disk.used / (1024 * 1024 * 1024)
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            disk_percent = (disk.used / disk.total) * 100
            
            # 进程指标
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024 * 1024)
            process_cpu_percent = process.cpu_percent()
            
            # GPU指标（如果可用）
            gpu_memory_used_mb = 0.0
            gpu_memory_total_mb = 0.0
            gpu_utilization_percent = 0.0
            
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    gpu_utilization_percent = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
            except ImportError:
                pass
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                memory_total_mb=memory_total_mb,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                memory_percent=memory_percent,
                disk_total_gb=disk_total_gb,
                disk_used_gb=disk_used_gb,
                disk_free_gb=disk_free_gb,
                disk_percent=disk_percent,
                process_memory_mb=process_memory_mb,
                process_cpu_percent=process_cpu_percent,
                gpu_memory_used_mb=gpu_memory_used_mb,
                gpu_memory_total_mb=gpu_memory_total_mb,
                gpu_utilization_percent=gpu_utilization_percent
            )
            
        except Exception as e:
            self.logger.warning(f"获取系统指标失败: {str(e)}")
            return SystemMetrics(timestamp=datetime.now())
    
    def _monitoring_loop(self):
        """监控循环"""
        while not self.stop_event.wait(self.metrics_collection_interval):
            try:
                metrics = self.get_current_metrics()
                self.system_metrics_history.append(metrics)
                
                # 限制历史记录数量
                if len(self.system_metrics_history) > 3600:  # 保留1小时的数据
                    self.system_metrics_history = self.system_metrics_history[-3600:]
                
                # 检查资源使用警告
                self._check_resource_warnings(metrics)
                
                # 详细监控模式下记录指标
                if self.monitoring_level == MonitoringLevel.DETAILED:
                    self.logger.debug(f"系统指标 - CPU: {metrics.cpu_percent:.1f}%, "
                                    f"内存: {metrics.memory_percent:.1f}%, "
                                    f"磁盘: {metrics.disk_percent:.1f}%")
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {str(e)}")
    
    def _check_resource_warnings(self, metrics: SystemMetrics):
        """检查资源使用警告"""
        # 内存使用警告
        if metrics.memory_percent > 90:
            if self.current_operation:
                self.log_warning(self.current_operation, f"内存使用率过高: {metrics.memory_percent:.1f}%")
        
        # 磁盘空间警告
        if metrics.disk_percent > 90:
            if self.current_operation:
                self.log_warning(self.current_operation, f"磁盘使用率过高: {metrics.disk_percent:.1f}%")
        
        # CPU使用警告
        if metrics.cpu_percent > 95:
            if self.current_operation:
                self.log_warning(self.current_operation, f"CPU使用率过高: {metrics.cpu_percent:.1f}%")
    
    def add_progress_callback(self, callback: Callable[[str, float, str], None]):
        """
        添加进度回调函数
        
        Args:
            callback: 回调函数，参数为 (operation_id, progress_percent, task_description)
        """
        self.progress_callbacks.append(callback)
    
    def add_status_callback(self, callback: Callable[[str, str], None]):
        """
        添加状态回调函数
        
        Args:
            callback: 回调函数，参数为 (operation_id, status)
        """
        self.status_callbacks.append(callback)
    
    def _notify_progress_callbacks(self, operation_id: str, progress_percent: float, task_description: str):
        """通知进度回调"""
        for callback in self.progress_callbacks:
            try:
                callback(operation_id, progress_percent, task_description)
            except Exception as e:
                self.logger.warning(f"进度回调执行失败: {str(e)}")
    
    def _notify_status_callbacks(self, operation_id: str, status: str):
        """通知状态回调"""
        for callback in self.status_callbacks:
            try:
                callback(operation_id, status)
            except Exception as e:
                self.logger.warning(f"状态回调执行失败: {str(e)}")
    
    @contextmanager
    def monitor_operation(self, operation_name: str, total_steps: int = 0):
        """
        操作监控上下文管理器
        
        Args:
            operation_name: 操作名称
            total_steps: 总步骤数
        """
        operation_id = self.start_operation(operation_name, total_steps)
        try:
            yield operation_id
            self.complete_operation(operation_id, success=True)
        except Exception as e:
            self.complete_operation(operation_id, success=False, error_message=str(e))
            raise
    
    def generate_monitoring_report(self, output_path: str = None) -> MonitoringReport:
        """
        生成监控报告
        
        Args:
            output_path: 输出路径（可选）
            
        Returns:
            MonitoringReport: 监控报告
        """
        end_time = datetime.now()
        
        report = MonitoringReport(
            report_id=f"monitoring_{end_time.strftime('%Y%m%d_%H%M%S')}",
            start_time=self.monitoring_start_time,
            end_time=end_time,
            operations=list(self.operations.values())
        )
        
        report.calculate_summary()
        
        # 保存报告
        if output_path:
            self._save_monitoring_report(report, output_path)
        
        return report
    
    def _save_monitoring_report(self, report: MonitoringReport, output_path: str):
        """保存监控报告"""
        try:
            report_dir = Path(output_path)
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成JSON报告
            json_report_path = report_dir / f"{report.report_id}.json"
            self._generate_json_monitoring_report(report, json_report_path)
            
            # 生成HTML报告
            html_report_path = report_dir / f"{report.report_id}.html"
            self._generate_html_monitoring_report(report, html_report_path)
            
            # 生成系统指标CSV
            if self.system_metrics_history:
                csv_report_path = report_dir / f"{report.report_id}_metrics.csv"
                self._generate_metrics_csv(csv_report_path)
            
            self.logger.info(f"监控报告已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存监控报告失败: {str(e)}")
    
    def _generate_json_monitoring_report(self, report: MonitoringReport, output_path: Path):
        """生成JSON监控报告"""
        report_data = {
            'report_id': report.report_id,
            'start_time': report.start_time.isoformat(),
            'end_time': report.end_time.isoformat(),
            'summary': {
                'total_operations': report.total_operations,
                'successful_operations': report.successful_operations,
                'failed_operations': report.failed_operations,
                'total_duration_seconds': report.total_duration_seconds,
                'avg_operation_duration_seconds': report.avg_operation_duration_seconds,
                'peak_memory_usage_mb': report.peak_memory_usage_mb,
                'avg_memory_usage_mb': report.avg_memory_usage_mb,
                'peak_cpu_usage_percent': report.peak_cpu_usage_percent,
                'avg_cpu_usage_percent': report.avg_cpu_usage_percent
            },
            'operations': [
                {
                    'operation_id': op.operation_id,
                    'operation_name': op.operation_name,
                    'status': op.status.value,
                    'start_time': op.start_time.isoformat(),
                    'end_time': op.end_time.isoformat() if op.end_time else None,
                    'duration_seconds': op.duration_seconds,
                    'progress_percent': op.progress_percent,
                    'peak_memory_mb': op.peak_memory_mb,
                    'avg_memory_mb': op.avg_memory_mb,
                    'peak_cpu_percent': op.peak_cpu_percent,
                    'avg_cpu_percent': op.avg_cpu_percent,
                    'error_message': op.error_message,
                    'warnings': op.warnings
                }
                for op in report.operations
            ],
            'errors': report.errors,
            'warnings': report.warnings
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    def _generate_html_monitoring_report(self, report: MonitoringReport, output_path: Path):
        """生成HTML监控报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>监控报告 - {report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; }}
        .metric h3 {{ margin: 0; color: #333; }}
        .metric p {{ margin: 5px 0; font-size: 18px; font-weight: bold; }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .status-completed {{ background-color: #d4edda; }}
        .status-failed {{ background-color: #f8d7da; }}
        .status-running {{ background-color: #fff3cd; }}
        .progress-bar {{ width: 100%; height: 20px; background-color: #e9ecef; border-radius: 10px; }}
        .progress-fill {{ height: 100%; background-color: #007bff; border-radius: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>监控报告</h1>
        <p><strong>报告ID:</strong> {report.report_id}</p>
        <p><strong>监控时间:</strong> {report.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {report.end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>总时长:</strong> {report.total_duration_seconds:.2f}秒</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>总操作数</h3>
            <p>{report.total_operations}</p>
        </div>
        <div class="metric">
            <h3>成功操作</h3>
            <p class="success">{report.successful_operations}</p>
        </div>
        <div class="metric">
            <h3>失败操作</h3>
            <p class="failure">{report.failed_operations}</p>
        </div>
        <div class="metric">
            <h3>峰值内存</h3>
            <p>{report.peak_memory_usage_mb:.1f} MB</p>
        </div>
        <div class="metric">
            <h3>峰值CPU</h3>
            <p>{report.peak_cpu_usage_percent:.1f}%</p>
        </div>
    </div>
    
    <h2>操作详情</h2>
    <table>
        <tr>
            <th>操作名称</th>
            <th>状态</th>
            <th>进度</th>
            <th>耗时(秒)</th>
            <th>峰值内存(MB)</th>
            <th>峰值CPU(%)</th>
            <th>警告数</th>
        </tr>
        {''.join([f'''
        <tr class="status-{op.status.value}">
            <td>{op.operation_name}</td>
            <td>{op.status.value}</td>
            <td>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {op.progress_percent}%"></div>
                </div>
                {op.progress_percent:.1f}%
            </td>
            <td>{op.duration_seconds:.2f}</td>
            <td>{op.peak_memory_mb:.1f}</td>
            <td>{op.peak_cpu_percent:.1f}</td>
            <td>{len(op.warnings)}</td>
        </tr>
        ''' for op in report.operations])}
    </table>
    
    {f'''
    <h2>错误信息</h2>
    <ul>
        {''.join([f'<li class="failure">{error}</li>' for error in report.errors])}
    </ul>
    ''' if report.errors else ''}
    
    {f'''
    <h2>警告信息</h2>
    <ul>
        {''.join([f'<li class="warning">{warning}</li>' for warning in report.warnings])}
    </ul>
    ''' if report.warnings else ''}
    
</body>
</html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_metrics_csv(self, output_path: Path):
        """生成系统指标CSV文件"""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入标题行
            writer.writerow([
                'timestamp', 'cpu_percent', 'memory_percent', 'memory_used_mb',
                'disk_percent', 'disk_used_gb', 'process_memory_mb', 'process_cpu_percent',
                'gpu_memory_used_mb', 'gpu_utilization_percent'
            ])
            
            # 写入数据行
            for metrics in self.system_metrics_history:
                writer.writerow([
                    metrics.timestamp.isoformat(),
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.memory_used_mb,
                    metrics.disk_percent,
                    metrics.disk_used_gb,
                    metrics.process_memory_mb,
                    metrics.process_cpu_percent,
                    metrics.gpu_memory_used_mb,
                    metrics.gpu_utilization_percent
                ])
    
    def get_operation_status(self, operation_id: str) -> Optional[OperationMetrics]:
        """
        获取操作状态
        
        Args:
            operation_id: 操作ID
            
        Returns:
            Optional[OperationMetrics]: 操作指标
        """
        return self.operations.get(operation_id)
    
    def get_all_operations(self) -> List[OperationMetrics]:
        """获取所有操作"""
        return list(self.operations.values())
    
    def clear_completed_operations(self):
        """清理已完成的操作"""
        completed_ops = [op_id for op_id, op in self.operations.items() 
                        if op.status in [OperationStatus.COMPLETED, OperationStatus.FAILED]]
        
        for op_id in completed_ops:
            del self.operations[op_id]
        
        self.logger.info(f"清理了{len(completed_ops)}个已完成的操作")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
        
        # 生成最终报告
        report_path = Path(self.config.output_directory) / "monitoring_reports"
        self.generate_monitoring_report(str(report_path))
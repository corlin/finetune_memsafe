"""
进度监控和报告系统

编写显示训练进度和内存状态的函数，创建训练期间的定期内存使用报告，
实现带性能指标的最终训练摘要。
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

try:
    from .memory_optimizer import MemoryOptimizer, MemoryStatus
    from .logging_system import LoggingSystem
except ImportError:
    from memory_optimizer import MemoryOptimizer, MemoryStatus
    from logging_system import LoggingSystem


@dataclass
class ProgressSnapshot:
    """进度快照数据类"""
    timestamp: datetime
    epoch: float
    step: int
    total_steps: int
    loss: float
    learning_rate: float
    memory_status: MemoryStatus
    elapsed_time: timedelta
    estimated_remaining: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        try:
            data = asdict(self)
            data['timestamp'] = self.timestamp.isoformat()
            data['elapsed_time'] = str(self.elapsed_time) if self.elapsed_time else "0:00:00"
            data['estimated_remaining'] = str(self.estimated_remaining) if self.estimated_remaining else None
            
            # 安全处理memory_status
            if hasattr(self, 'memory_status') and self.memory_status:
                data['memory_status'] = asdict(self.memory_status)
                if hasattr(self.memory_status, 'timestamp'):
                    data['memory_status']['timestamp'] = self.memory_status.timestamp.isoformat()
            
            return data
        except Exception as e:
            # 如果转换失败，返回基本信息
            return {
                'timestamp': self.timestamp.isoformat() if hasattr(self, 'timestamp') else datetime.now().isoformat(),
                'epoch': float(self.epoch) if hasattr(self, 'epoch') and self.epoch is not None else 0.0,
                'step': int(self.step) if hasattr(self, 'step') and self.step is not None else 0,
                'total_steps': int(self.total_steps) if hasattr(self, 'total_steps') and self.total_steps is not None else 0,
                'loss': float(self.loss) if hasattr(self, 'loss') and self.loss is not None else 0.0,
                'learning_rate': float(self.learning_rate) if hasattr(self, 'learning_rate') and self.learning_rate is not None else 0.0,
                'elapsed_time': "0:00:00",
                'estimated_remaining': None,
                'error': str(e)
            }


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    avg_steps_per_second: float
    avg_samples_per_second: float
    peak_memory_gb: float
    avg_memory_gb: float
    memory_efficiency: float  # 内存使用效率 (0-1)
    training_stability: float  # 训练稳定性 (基于损失变化)
    total_training_time: timedelta
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['total_training_time'] = str(self.total_training_time)
        return data


class ProgressMonitor:
    """
    进度监控器
    
    显示训练进度和内存状态，创建定期内存使用报告。
    """
    
    def __init__(self, 
                 memory_optimizer: MemoryOptimizer,
                 logging_system: Optional[LoggingSystem] = None,
                 update_interval: float = 5.0,
                 enable_rich_display: bool = True):
        """
        初始化进度监控器
        
        Args:
            memory_optimizer: 内存优化器
            logging_system: 日志记录系统
            update_interval: 更新间隔（秒）
            enable_rich_display: 是否启用Rich显示
        """
        self.memory_optimizer = memory_optimizer
        self.logging_system = logging_system
        self.update_interval = update_interval
        self.enable_rich_display = enable_rich_display
        
        # 进度跟踪
        self.start_time: Optional[datetime] = None
        self.current_epoch: float = 0.0
        self.current_step: int = 0
        self.total_steps: int = 0
        self.current_loss: float = 0.0
        self.current_lr: float = 0.0
        
        # 历史数据
        self.progress_history: List[ProgressSnapshot] = []
        self.memory_reports: List[Dict[str, Any]] = []
        
        # 监控线程
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring_event = threading.Event()
        
        # Rich显示组件
        self.console = Console() if enable_rich_display else None
        self.progress_bar: Optional[Progress] = None
        self.task_id: Optional[TaskID] = None
        self.live_display: Optional[Live] = None
        
        # 性能统计
        self.step_times: List[float] = []
        self.loss_history: List[float] = []
        self.memory_usage_history: List[float] = []
    
    def start_monitoring(self, total_steps: int) -> None:
        """
        开始监控
        
        Args:
            total_steps: 总训练步数
        """
        self.start_time = datetime.now()
        self.total_steps = total_steps
        self.stop_monitoring_event.clear()
        
        if self.logging_system:
            self.logging_system.info("开始进度监控", "PROGRESS_MONITOR", 
                                    {"total_steps": total_steps})
        
        # 启动Rich显示
        if self.enable_rich_display and self.console:
            self._setup_rich_display()
        
        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        try:
            self.stop_monitoring_event.set()
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2.0)
            
            if self.live_display:
                try:
                    self.live_display.stop()
                except Exception as e:
                    if self.logging_system:
                        self.logging_system.warning(f"停止Rich显示时出错: {e}", "PROGRESS_MONITOR")
            
            if self.logging_system:
                self.logging_system.info("进度监控已停止", "PROGRESS_MONITOR")
        except Exception as e:
            if self.logging_system:
                self.logging_system.error(f"停止监控时出错: {e}", "PROGRESS_MONITOR")
    
    def update_progress(self, epoch: float, step: int, loss: float, learning_rate: float) -> None:
        """
        更新进度信息
        
        Args:
            epoch: 当前轮次
            step: 当前步数
            loss: 当前损失
            learning_rate: 当前学习率
        """
        # 安全地处理可能为None的值
        self.current_epoch = float(epoch) if epoch is not None else 0.0
        self.current_step = int(step) if step is not None else 0
        self.current_loss = float(loss) if loss is not None else 0.0
        self.current_lr = float(learning_rate) if learning_rate is not None else 0.0
        
        # 记录步骤时间
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if step > 0:
                avg_step_time = elapsed / step
                self.step_times.append(avg_step_time)
        
        # 记录损失历史
        self.loss_history.append(loss)
        
        # 简化的进度显示 - 不使用Rich进度条
        if step is not None and step > 0:
            # 每10步或每轮结束时输出进度信息
            if step % 10 == 0 or (hasattr(self, 'total_steps') and self.total_steps and step >= self.total_steps):
                progress_msg = f"训练进度: 步骤 {step}"
                if hasattr(self, 'total_steps') and self.total_steps:
                    progress_pct = (step / self.total_steps) * 100
                    progress_msg += f"/{self.total_steps} ({progress_pct:.1f}%)"
                if loss is not None:
                    progress_msg += f", 损失: {loss:.4f}"
                
                if self.logging_system:
                    self.logging_system.info(progress_msg, "PROGRESS_MONITOR")
    
    def _setup_rich_display(self) -> None:
        """设置Rich显示 - 完全禁用Rich避免内部错误"""
        # 完全禁用Rich显示，使用简单的文本输出
        self.progress_bar = None
        self.live_display = None
        self.task_id = None
        
        if hasattr(self, 'logging_system') and self.logging_system:
            self.logging_system.info("使用简化进度显示模式", "PROGRESS_MONITOR")
    
    def _create_layout(self) -> Layout:
        """创建显示布局"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="progress", size=3),
            Layout(name="metrics", size=10),
            Layout(name="memory", size=8)
        )
        
        return layout
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while not self.stop_monitoring_event.is_set():
            try:
                # 创建进度快照
                snapshot = self._create_progress_snapshot()
                self.progress_history.append(snapshot)
                
                # 创建内存报告
                memory_report = self._create_memory_report()
                self.memory_reports.append(memory_report)
                
                # 更新显示
                if self.live_display and self.enable_rich_display:
                    self._update_display()
                
                # 记录到日志系统
                if self.logging_system:
                    self._log_progress_update(snapshot, memory_report)
                
                # 等待下次更新
                self.stop_monitoring_event.wait(self.update_interval)
                
            except Exception as e:
                if self.logging_system:
                    self.logging_system.error(f"监控循环错误: {e}", "PROGRESS_MONITOR")
                time.sleep(1.0)
    
    def _create_progress_snapshot(self) -> ProgressSnapshot:
        """创建进度快照"""
        now = datetime.now()
        elapsed = now - self.start_time if self.start_time else timedelta(0)
        
        # 估算剩余时间 - 添加安全检查
        estimated_remaining = None
        try:
            if (self.current_step and self.current_step > 0 and 
                self.total_steps and self.total_steps > 0 and 
                elapsed and elapsed.total_seconds() > 0):
                
                avg_step_time = elapsed.total_seconds() / self.current_step
                remaining_steps = self.total_steps - self.current_step
                
                # 确保计算结果是有效的
                if avg_step_time > 0 and remaining_steps > 0:
                    estimated_seconds = avg_step_time * remaining_steps
                    # 限制最大预估时间，避免异常值
                    if estimated_seconds < 86400:  # 小于24小时
                        estimated_remaining = timedelta(seconds=estimated_seconds)
        except Exception as e:
            # 如果时间计算失败，记录但不中断
            if hasattr(self, 'logging_system') and self.logging_system:
                self.logging_system.warning(f"时间估算计算失败: {e}", "PROGRESS_MONITOR")
        
        # 获取内存状态
        memory_status = self.memory_optimizer.get_memory_status()
        self.memory_usage_history.append(memory_status.allocated_gb)
        
        return ProgressSnapshot(
            timestamp=now,
            epoch=float(self.current_epoch) if self.current_epoch is not None else 0.0,
            step=int(self.current_step) if self.current_step is not None else 0,
            total_steps=int(self.total_steps) if self.total_steps is not None else 0,
            loss=float(self.current_loss) if self.current_loss is not None else 0.0,
            learning_rate=float(self.current_lr) if self.current_lr is not None else 0.0,
            memory_status=memory_status,
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining
        )
    
    def _create_memory_report(self) -> Dict[str, Any]:
        """创建内存使用报告"""
        memory_status = self.memory_optimizer.get_memory_status()
        
        # 计算内存统计
        if self.memory_usage_history:
            avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history)
            peak_memory = max(self.memory_usage_history)
            min_memory = min(self.memory_usage_history)
        else:
            avg_memory = peak_memory = min_memory = memory_status.allocated_gb
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_memory_gb": memory_status.allocated_gb,
            "cached_memory_gb": memory_status.cached_gb,
            "available_memory_gb": memory_status.available_gb,
            "total_memory_gb": memory_status.total_gb,
            "is_safe": memory_status.is_safe,
            "avg_memory_gb": avg_memory,
            "peak_memory_gb": peak_memory,
            "min_memory_gb": min_memory,
            "memory_usage_percent": (memory_status.allocated_gb / memory_status.total_gb) * 100,
            "step": self.current_step
        }
    
    def _update_display(self) -> None:
        """更新Rich显示"""
        if not self.live_display or not self.console:
            return
        
        try:
            # 简化显示，直接显示进度条和基本信息
            if self.progress_bar:
                # 只显示进度条，避免复杂的Layout问题
                self.live_display.update(self.progress_bar)
            
        except Exception as e:
            if self.logging_system:
                self.logging_system.error(f"更新显示失败: {e}", "PROGRESS_MONITOR")
    
    def _create_metrics_table(self) -> Table:
        """创建指标表"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("当前值", style="green")
        table.add_column("统计", style="yellow")
        
        # 计算统计信息
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        steps_per_sec = self.current_step / elapsed if elapsed > 0 else 0
        
        # 添加行
        table.add_row("轮次", f"{self.current_epoch:.2f}", "")
        table.add_row("步骤", f"{self.current_step:,}", f"{self.current_step}/{self.total_steps:,}")
        table.add_row("损失", f"{self.current_loss:.6f}", 
                     f"最近10步平均: {sum(self.loss_history[-10:])/len(self.loss_history[-10:]):.6f}" if len(self.loss_history) >= 10 else "")
        table.add_row("学习率", f"{self.current_lr:.2e}", "")
        table.add_row("速度", f"{steps_per_sec:.2f} 步/秒", "")
        
        if elapsed > 0:
            table.add_row("已用时间", str(timedelta(seconds=int(elapsed))), "")
        
        return table
    
    def _create_memory_table(self) -> Table:
        """创建内存表"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("内存类型", style="cyan")
        table.add_column("当前值", style="green")
        table.add_column("统计", style="yellow")
        
        memory_status = self.memory_optimizer.get_memory_status()
        
        # 计算统计
        if self.memory_usage_history:
            avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history)
            peak_memory = max(self.memory_usage_history)
        else:
            avg_memory = peak_memory = memory_status.allocated_gb
        
        # 添加行
        table.add_row("已分配", f"{memory_status.allocated_gb:.2f} GB", f"峰值: {peak_memory:.2f} GB")
        table.add_row("缓存", f"{memory_status.cached_gb:.2f} GB", "")
        table.add_row("可用", f"{memory_status.available_gb:.2f} GB", f"平均使用: {avg_memory:.2f} GB")
        table.add_row("总计", f"{memory_status.total_gb:.2f} GB", "")
        table.add_row("使用率", f"{(memory_status.allocated_gb/memory_status.total_gb)*100:.1f}%", "")
        table.add_row("安全状态", "✅ 安全" if memory_status.is_safe else "⚠️ 警告", "")
        
        return table
    
    def _log_progress_update(self, snapshot: ProgressSnapshot, memory_report: Dict[str, Any]) -> None:
        """记录进度更新到日志系统"""
        if not self.logging_system:
            return
        
        # 每50步记录一次详细进度
        if self.current_step % 50 == 0:
            self.logging_system.info(
                f"训练进度更新 - 步骤 {self.current_step}/{self.total_steps}",
                "PROGRESS_MONITOR",
                {
                    "epoch": snapshot.epoch,
                    "step": snapshot.step,
                    "loss": snapshot.loss,
                    "learning_rate": snapshot.learning_rate,
                    "elapsed_time": str(snapshot.elapsed_time),
                    "estimated_remaining": str(snapshot.estimated_remaining) if snapshot.estimated_remaining else None,
                    "memory_usage_gb": snapshot.memory_status.allocated_gb,
                    "memory_is_safe": snapshot.memory_status.is_safe
                }
            )
        
        # 每100步记录一次内存报告
        if self.current_step % 100 == 0:
            self.logging_system.info(
                f"内存使用报告 - 步骤 {self.current_step}",
                "MEMORY_MONITOR",
                memory_report
            )
    
    def generate_training_summary(self) -> Dict[str, Any]:
        """
        生成带性能指标的最终训练摘要
        
        Returns:
            包含训练摘要和性能指标的字典
        """
        if not self.start_time:
            return {"error": "监控未启动"}
        
        end_time = datetime.now()
        total_time = end_time - self.start_time
        
        # 计算性能指标
        performance_metrics = self._calculate_performance_metrics(total_time)
        
        # 创建摘要
        summary = {
            "training_summary": {
                "start_time": self.start_time.isoformat() if self.start_time else datetime.now().isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration": str(total_time),
                "total_steps": int(self.current_step) if self.current_step is not None else 0,
                "final_epoch": float(self.current_epoch) if self.current_epoch is not None else 0.0,
                "final_loss": float(self.current_loss) if self.current_loss is not None else 0.0,
                "final_learning_rate": float(self.current_lr) if self.current_lr is not None else 0.0
            },
            "performance_metrics": performance_metrics.to_dict(),
            "memory_summary": self._generate_memory_summary(),
            "progress_statistics": self._generate_progress_statistics()
        }
        
        # 记录到日志系统
        if self.logging_system:
            self.logging_system.info("训练摘要生成完成", "PROGRESS_MONITOR", summary)
        
        return summary
    
    def _calculate_performance_metrics(self, total_time: timedelta) -> PerformanceMetrics:
        """计算性能指标"""
        # 安全地处理total_time，确保不为None
        if total_time is None:
            total_time = timedelta(0)
        
        # 计算步骤速度
        total_seconds = total_time.total_seconds()
        current_step = self.current_step if self.current_step is not None else 0
        avg_steps_per_second = current_step / total_seconds if total_seconds > 0 else 0
        
        # 假设每步处理一个样本（可以根据实际批次大小调整）
        avg_samples_per_second = avg_steps_per_second  # 简化假设
        
        # 内存统计
        if self.memory_usage_history:
            peak_memory_gb = max(self.memory_usage_history)
            avg_memory_gb = sum(self.memory_usage_history) / len(self.memory_usage_history)
        else:
            peak_memory_gb = avg_memory_gb = 0.0
        
        # 内存效率（使用的内存与总内存的比例）
        try:
            total_memory = self.memory_optimizer.get_memory_status().total_gb
            memory_efficiency = avg_memory_gb / total_memory if total_memory > 0 else 0.0
        except:
            memory_efficiency = 0.0
        
        # 训练稳定性（基于损失变化的标准差）
        training_stability = 1.0
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-50:]  # 最近50步
            if len(recent_losses) > 1:
                try:
                    import statistics
                    loss_std = statistics.stdev(recent_losses)
                    loss_mean = statistics.mean(recent_losses)
                    # 稳定性 = 1 - (标准差/均值)，限制在0-1之间
                    training_stability = max(0.0, min(1.0, 1.0 - (loss_std / loss_mean if loss_mean > 0 else 1.0)))
                except:
                    training_stability = 1.0
        
        return PerformanceMetrics(
            avg_steps_per_second=avg_steps_per_second,
            avg_samples_per_second=avg_samples_per_second,
            peak_memory_gb=peak_memory_gb,
            avg_memory_gb=avg_memory_gb,
            memory_efficiency=memory_efficiency,
            training_stability=training_stability,
            total_training_time=total_time
        )
    
    def _generate_memory_summary(self) -> Dict[str, Any]:
        """生成内存使用摘要"""
        if not self.memory_usage_history:
            return {}
        
        import statistics
        
        return {
            "peak_memory_gb": max(self.memory_usage_history),
            "min_memory_gb": min(self.memory_usage_history),
            "avg_memory_gb": statistics.mean(self.memory_usage_history),
            "median_memory_gb": statistics.median(self.memory_usage_history),
            "memory_std_gb": statistics.stdev(self.memory_usage_history) if len(self.memory_usage_history) > 1 else 0.0,
            "total_memory_reports": len(self.memory_reports),
            "memory_warnings": sum(1 for report in self.memory_reports if not report.get("is_safe", True))
        }
    
    def _generate_progress_statistics(self) -> Dict[str, Any]:
        """生成进度统计"""
        if not self.progress_history:
            return {}
        
        return {
            "total_snapshots": len(self.progress_history),
            "monitoring_duration": str(self.progress_history[-1].timestamp - self.progress_history[0].timestamp) if len(self.progress_history) > 1 else "0:00:00",
            "avg_update_interval": self.update_interval,
            "loss_trend": self._calculate_loss_trend(),
            "step_consistency": self._calculate_step_consistency()
        }
    
    def _calculate_loss_trend(self) -> str:
        """计算损失趋势"""
        if len(self.loss_history) < 10:
            return "insufficient_data"
        
        # 安全地过滤None值
        valid_losses = [x for x in self.loss_history if x is not None]
        if len(valid_losses) < 10:
            return "insufficient_data"
        
        recent_losses = valid_losses[-20:]
        try:
            early_avg = sum(recent_losses[:10]) / 10
            late_avg = sum(recent_losses[-10:]) / 10
            
            if late_avg < early_avg * 0.95:
                return "decreasing"
            elif late_avg > early_avg * 1.05:
                return "increasing"
            else:
                return "stable"
        except (TypeError, ZeroDivisionError):
            return "insufficient_data"
    
    def _calculate_step_consistency(self) -> float:
        """计算步骤一致性（基于步骤时间的变异系数）"""
        if len(self.step_times) < 10:
            return 1.0
        
        # 安全地过滤None值
        valid_times = [x for x in self.step_times if x is not None]
        if len(valid_times) < 2:
            return 1.0
        
        try:
            import statistics
            recent_times = valid_times[-50:]  # 最近50步
            if len(recent_times) > 1:
                mean_time = statistics.mean(recent_times)
                std_time = statistics.stdev(recent_times)
                # 一致性 = 1 - 变异系数
                consistency = max(0.0, min(1.0, 1.0 - (std_time / mean_time if mean_time > 0 else 1.0)))
                return consistency
        except (TypeError, statistics.StatisticsError):
            return 1.0
        
        return 1.0
    
    def save_progress_report(self, output_path: str) -> None:
        """
        保存进度报告到文件
        
        Args:
            output_path: 输出文件路径
        """
        try:
            summary = self.generate_training_summary()
            
            # 安全地处理进度历史
            progress_history = []
            for snapshot in self.progress_history:
                try:
                    progress_history.append(snapshot.to_dict())
                except Exception as e:
                    # 如果单个快照转换失败，记录错误但继续
                    if self.logging_system:
                        self.logging_system.warning(f"跳过损坏的进度快照: {e}", "PROGRESS_MONITOR")
                    continue
            
            # 安全地处理其他数据
            safe_memory_reports = []
            for report in self.memory_reports:
                try:
                    if isinstance(report, dict):
                        safe_memory_reports.append(report)
                    else:
                        safe_memory_reports.append(str(report))
                except:
                    continue
            
            # 安全地处理数值列表
            def safe_float_convert(x):
                """安全地将值转换为浮点数"""
                if x is None:
                    return 0.0
                try:
                    return float(x)
                except (ValueError, TypeError):
                    return 0.0
            
            safe_loss_history = [safe_float_convert(x) for x in self.loss_history]
            safe_memory_history = [safe_float_convert(x) for x in self.memory_usage_history]
            
            # 添加详细的历史数据
            detailed_report = {
                "summary": summary,
                "progress_history": progress_history,
                "memory_reports": safe_memory_reports,
                "loss_history": safe_loss_history,
                "memory_usage_history": safe_memory_history,
                "report_generated_at": datetime.now().isoformat()
            }
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用自定义JSON编码器处理datetime对象
            def json_serializer(obj):
                """JSON序列化器，处理datetime和其他特殊对象"""
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, timedelta):
                    return str(obj)
                elif hasattr(obj, 'isoformat'):  # 其他日期时间对象
                    return obj.isoformat()
                elif hasattr(obj, '__dict__'):  # 自定义对象
                    return obj.__dict__
                else:
                    return str(obj)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_report, f, indent=2, ensure_ascii=False, default=json_serializer)
            
            if self.logging_system:
                self.logging_system.info(f"进度报告已保存到 {output_path}", "PROGRESS_MONITOR")
        
        except Exception as e:
            if self.logging_system:
                self.logging_system.error(f"保存进度报告失败: {e}", "PROGRESS_MONITOR")
            # 不再抛出异常，避免中断训练
            try:
                # 尝试保存一个最小的报告
                minimal_report = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "status": "report_generation_failed"
                }
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal_report, f, indent=2, ensure_ascii=False)
            except:
                pass  # 如果连最小报告都无法保存，就放弃
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        if not self.start_time:
            return {"status": "not_started"}
        
        elapsed = datetime.now() - self.start_time
        memory_status = self.memory_optimizer.get_memory_status()
        
        return {
            "status": "running",
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0,
            "current_loss": self.current_loss,
            "current_lr": self.current_lr,
            "elapsed_time": str(elapsed),
            "memory_usage_gb": memory_status.allocated_gb,
            "memory_is_safe": memory_status.is_safe,
            "steps_per_second": self.current_step / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        }
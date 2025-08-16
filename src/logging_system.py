"""
日志记录和监控系统 - 带TensorBoard集成的全面日志记录

提供训练指标和内存使用的日志配置，创建损失、学习率和内存统计的TensorBoard日志，
实现带时间戳和严重性级别的结构化日志。
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading
from queue import Queue, Empty

import torch
from torch.utils.tensorboard import SummaryWriter

from .memory_optimizer import MemoryStatus


@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    epoch: float
    step: int
    loss: float
    learning_rate: float
    memory_usage: MemoryStatus
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        # 转换datetime为字符串
        data['timestamp'] = self.timestamp.isoformat()
        # 转换MemoryStatus为字典
        data['memory_usage'] = asdict(self.memory_usage)
        data['memory_usage']['timestamp'] = self.memory_usage.timestamp.isoformat()
        return data


@dataclass
class LogEntry:
    """结构化日志条目"""
    timestamp: datetime
    level: str
    message: str
    component: str
    metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'message': self.message,
            'component': self.component
        }
        if self.metrics:
            data['metrics'] = self.metrics
        return data


class StructuredLogger:
    """
    结构化日志记录器
    
    提供带时间戳和严重性级别的结构化日志记录功能。
    """
    
    def __init__(self, name: str, log_dir: str, log_level: int = logging.INFO):
        """
        初始化结构化日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志目录
            log_level: 日志级别
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建Python标准日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 创建文件处理器
        log_file = self.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 结构化日志存储
        self.structured_log_file = self.log_dir / f"{name}_structured.jsonl"
        self.log_entries: List[LogEntry] = []
        
        self.logger.info(f"结构化日志记录器初始化完成: {name}")
    
    def log_structured(self, level: str, message: str, component: str, 
                      metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        记录结构化日志条目
        
        Args:
            level: 日志级别
            message: 日志消息
            component: 组件名称
            metrics: 可选的指标数据
        """
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            component=component,
            metrics=metrics
        )
        
        # 添加到内存存储
        self.log_entries.append(entry)
        
        # 写入结构化日志文件
        with open(self.structured_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + '\n')
        
        # 同时记录到标准日志
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        if metrics:
            log_method(f"[{component}] {message} | Metrics: {metrics}")
        else:
            log_method(f"[{component}] {message}")
    
    def info(self, message: str, component: str = "SYSTEM", 
             metrics: Optional[Dict[str, Any]] = None) -> None:
        """记录INFO级别日志"""
        self.log_structured("INFO", message, component, metrics)
    
    def warning(self, message: str, component: str = "SYSTEM", 
                metrics: Optional[Dict[str, Any]] = None) -> None:
        """记录WARNING级别日志"""
        self.log_structured("WARNING", message, component, metrics)
    
    def error(self, message: str, component: str = "SYSTEM", 
              metrics: Optional[Dict[str, Any]] = None) -> None:
        """记录ERROR级别日志"""
        self.log_structured("ERROR", message, component, metrics)
    
    def debug(self, message: str, component: str = "SYSTEM", 
              metrics: Optional[Dict[str, Any]] = None) -> None:
        """记录DEBUG级别日志"""
        self.log_structured("DEBUG", message, component, metrics)
    
    def get_log_entries(self, level: Optional[str] = None, 
                       component: Optional[str] = None) -> List[LogEntry]:
        """
        获取日志条目
        
        Args:
            level: 可选的日志级别过滤
            component: 可选的组件过滤
            
        Returns:
            过滤后的日志条目列表
        """
        entries = self.log_entries
        
        if level:
            entries = [e for e in entries if e.level == level]
        
        if component:
            entries = [e for e in entries if e.component == component]
        
        return entries


class TensorBoardLogger:
    """
    TensorBoard日志记录器
    
    创建损失、学习率和内存统计的TensorBoard日志。
    """
    
    def __init__(self, log_dir: str, run_name: Optional[str] = None):
        """
        初始化TensorBoard日志记录器
        
        Args:
            log_dir: TensorBoard日志目录
            run_name: 可选的运行名称
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建运行特定的目录
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_dir = self.log_dir / run_name
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        
        # 日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 指标缓存
        self.metrics_cache: Dict[str, List[float]] = {}
        
        self.logger.info(f"TensorBoard日志记录器初始化完成: {self.run_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        记录标量值到TensorBoard
        
        Args:
            tag: 标签名称
            value: 标量值
            step: 步骤数
        """
        try:
            self.writer.add_scalar(tag, value, step)
            
            # 添加到缓存
            if tag not in self.metrics_cache:
                self.metrics_cache[tag] = []
            self.metrics_cache[tag].append(value)
            
        except Exception as e:
            self.logger.error(f"记录标量值失败 {tag}: {e}")
    
    def log_training_metrics(self, metrics: TrainingMetrics) -> None:
        """
        记录训练指标到TensorBoard
        
        Args:
            metrics: 训练指标对象
        """
        try:
            step = metrics.step
            
            # 记录训练损失
            self.log_scalar("Training/Loss", metrics.loss, step)
            
            # 记录学习率
            self.log_scalar("Training/LearningRate", metrics.learning_rate, step)
            
            # 记录轮次
            self.log_scalar("Training/Epoch", metrics.epoch, step)
            
            # 记录内存使用情况
            memory = metrics.memory_usage
            self.log_scalar("Memory/Allocated_GB", memory.allocated_gb, step)
            self.log_scalar("Memory/Cached_GB", memory.cached_gb, step)
            self.log_scalar("Memory/Available_GB", memory.available_gb, step)
            self.log_scalar("Memory/Total_GB", memory.total_gb, step)
            self.log_scalar("Memory/IsSafe", 1.0 if memory.is_safe else 0.0, step)
            
            # 计算内存使用率
            memory_usage_percent = (memory.allocated_gb / memory.total_gb) * 100
            self.log_scalar("Memory/Usage_Percent", memory_usage_percent, step)
            
        except Exception as e:
            self.logger.error(f"记录训练指标失败: {e}")
    
    def log_memory_status(self, memory_status: MemoryStatus, step: int, 
                         prefix: str = "Memory") -> None:
        """
        记录内存状态到TensorBoard
        
        Args:
            memory_status: 内存状态对象
            step: 步骤数
            prefix: 标签前缀
        """
        try:
            self.log_scalar(f"{prefix}/Allocated_GB", memory_status.allocated_gb, step)
            self.log_scalar(f"{prefix}/Cached_GB", memory_status.cached_gb, step)
            self.log_scalar(f"{prefix}/Available_GB", memory_status.available_gb, step)
            self.log_scalar(f"{prefix}/Total_GB", memory_status.total_gb, step)
            self.log_scalar(f"{prefix}/IsSafe", 1.0 if memory_status.is_safe else 0.0, step)
            
            # 计算使用率
            usage_percent = (memory_status.allocated_gb / memory_status.total_gb) * 100
            self.log_scalar(f"{prefix}/Usage_Percent", usage_percent, step)
            
        except Exception as e:
            self.logger.error(f"记录内存状态失败: {e}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """
        记录超参数和指标
        
        Args:
            hparams: 超参数字典
            metrics: 指标字典
        """
        try:
            self.writer.add_hparams(hparams, metrics)
        except Exception as e:
            self.logger.error(f"记录超参数失败: {e}")
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """
        记录直方图到TensorBoard
        
        Args:
            tag: 标签名称
            values: 张量值
            step: 步骤数
        """
        try:
            self.writer.add_histogram(tag, values, step)
        except Exception as e:
            self.logger.error(f"记录直方图失败 {tag}: {e}")
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        记录文本到TensorBoard
        
        Args:
            tag: 标签名称
            text: 文本内容
            step: 步骤数
        """
        try:
            self.writer.add_text(tag, text, step)
        except Exception as e:
            self.logger.error(f"记录文本失败 {tag}: {e}")
    
    def flush(self) -> None:
        """刷新TensorBoard写入器"""
        try:
            self.writer.flush()
        except Exception as e:
            self.logger.error(f"刷新TensorBoard写入器失败: {e}")
    
    def close(self) -> None:
        """关闭TensorBoard写入器"""
        try:
            self.writer.close()
            self.logger.info("TensorBoard写入器已关闭")
        except Exception as e:
            self.logger.error(f"关闭TensorBoard写入器失败: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        获取指标摘要统计
        
        Returns:
            包含各指标统计信息的字典
        """
        summary = {}
        
        for tag, values in self.metrics_cache.items():
            if values:
                summary[tag] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "last": values[-1]
                }
        
        return summary


class AsyncLogger:
    """
    异步日志记录器
    
    使用后台线程处理日志记录，避免阻塞主训练循环。
    """
    
    def __init__(self, structured_logger: StructuredLogger, 
                 tensorboard_logger: TensorBoardLogger):
        """
        初始化异步日志记录器
        
        Args:
            structured_logger: 结构化日志记录器
            tensorboard_logger: TensorBoard日志记录器
        """
        self.structured_logger = structured_logger
        self.tensorboard_logger = tensorboard_logger
        
        # 日志队列
        self.log_queue: Queue = Queue()
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        self.logger = logging.getLogger(__name__)
    
    def start(self) -> None:
        """启动异步日志处理"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        self.logger.info("异步日志记录器已启动")
    
    def stop(self) -> None:
        """停止异步日志处理"""
        if not self.running:
            return
        
        self.running = False
        
        # 发送停止信号
        self.log_queue.put(None)
        
        # 等待工作线程结束
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        self.logger.info("异步日志记录器已停止")
    
    def _worker(self) -> None:
        """后台工作线程"""
        while self.running:
            try:
                # 从队列获取日志任务
                task = self.log_queue.get(timeout=1.0)
                
                if task is None:  # 停止信号
                    break
                
                # 处理日志任务
                self._process_log_task(task)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"异步日志处理错误: {e}")
    
    def _process_log_task(self, task: Dict[str, Any]) -> None:
        """
        处理日志任务
        
        Args:
            task: 日志任务字典
        """
        try:
            task_type = task.get("type")
            
            if task_type == "structured":
                self.structured_logger.log_structured(
                    task["level"], task["message"], task["component"], task.get("metrics")
                )
            
            elif task_type == "training_metrics":
                self.tensorboard_logger.log_training_metrics(task["metrics"])
            
            elif task_type == "scalar":
                self.tensorboard_logger.log_scalar(task["tag"], task["value"], task["step"])
            
            elif task_type == "memory_status":
                self.tensorboard_logger.log_memory_status(
                    task["memory_status"], task["step"], task.get("prefix", "Memory")
                )
            
            elif task_type == "flush":
                self.tensorboard_logger.flush()
            
        except Exception as e:
            self.logger.error(f"处理日志任务失败: {e}")
    
    def log_structured_async(self, level: str, message: str, component: str, 
                           metrics: Optional[Dict[str, Any]] = None) -> None:
        """异步记录结构化日志"""
        task = {
            "type": "structured",
            "level": level,
            "message": message,
            "component": component,
            "metrics": metrics
        }
        self.log_queue.put(task)
    
    def log_training_metrics_async(self, metrics: TrainingMetrics) -> None:
        """异步记录训练指标"""
        task = {
            "type": "training_metrics",
            "metrics": metrics
        }
        self.log_queue.put(task)
    
    def log_scalar_async(self, tag: str, value: float, step: int) -> None:
        """异步记录标量值"""
        task = {
            "type": "scalar",
            "tag": tag,
            "value": value,
            "step": step
        }
        self.log_queue.put(task)
    
    def log_memory_status_async(self, memory_status: MemoryStatus, step: int, 
                              prefix: str = "Memory") -> None:
        """异步记录内存状态"""
        task = {
            "type": "memory_status",
            "memory_status": memory_status,
            "step": step,
            "prefix": prefix
        }
        self.log_queue.put(task)
    
    def flush_async(self) -> None:
        """异步刷新"""
        task = {"type": "flush"}
        self.log_queue.put(task)


class LoggingSystem:
    """
    综合日志记录系统
    
    整合结构化日志记录和TensorBoard日志记录功能。
    """
    
    def __init__(self, log_dir: str, run_name: Optional[str] = None, 
                 enable_async: bool = True):
        """
        初始化日志记录系统
        
        Args:
            log_dir: 日志目录
            run_name: 可选的运行名称
            enable_async: 是否启用异步日志记录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成运行名称
        if run_name is None:
            run_name = f"qwen3_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_name = run_name
        
        # 创建子日志记录器
        self.structured_logger = StructuredLogger(
            name="qwen3_training",
            log_dir=str(self.log_dir / "structured")
        )
        
        self.tensorboard_logger = TensorBoardLogger(
            log_dir=str(self.log_dir / "tensorboard"),
            run_name=run_name
        )
        
        # 异步日志记录器
        self.async_logger: Optional[AsyncLogger] = None
        if enable_async:
            self.async_logger = AsyncLogger(
                self.structured_logger, 
                self.tensorboard_logger
            )
            self.async_logger.start()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志记录系统初始化完成: {run_name}")
    
    def log_training_start(self, config: Dict[str, Any]) -> None:
        """记录训练开始"""
        self.info("训练开始", "TRAINING", {"config": config})
        
        # 记录超参数到TensorBoard
        hparams = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
        self.tensorboard_logger.log_hyperparameters(hparams, {})
    
    def log_training_metrics(self, metrics: TrainingMetrics) -> None:
        """记录训练指标"""
        if self.async_logger:
            self.async_logger.log_training_metrics_async(metrics)
        else:
            self.tensorboard_logger.log_training_metrics(metrics)
        
        # 同时记录结构化日志
        metrics_dict = metrics.to_dict()
        self.info(f"训练指标 - 步骤 {metrics.step}", "TRAINING", metrics_dict)
    
    def log_memory_status(self, memory_status: MemoryStatus, step: int, 
                         prefix: str = "Memory") -> None:
        """记录内存状态"""
        if self.async_logger:
            self.async_logger.log_memory_status_async(memory_status, step, prefix)
        else:
            self.tensorboard_logger.log_memory_status(memory_status, step, prefix)
    
    def info(self, message: str, component: str = "SYSTEM", 
             metrics: Optional[Dict[str, Any]] = None) -> None:
        """记录INFO级别日志"""
        if self.async_logger:
            self.async_logger.log_structured_async("INFO", message, component, metrics)
        else:
            self.structured_logger.info(message, component, metrics)
    
    def warning(self, message: str, component: str = "SYSTEM", 
                metrics: Optional[Dict[str, Any]] = None) -> None:
        """记录WARNING级别日志"""
        if self.async_logger:
            self.async_logger.log_structured_async("WARNING", message, component, metrics)
        else:
            self.structured_logger.warning(message, component, metrics)
    
    def error(self, message: str, component: str = "SYSTEM", 
              metrics: Optional[Dict[str, Any]] = None) -> None:
        """记录ERROR级别日志"""
        if self.async_logger:
            self.async_logger.log_structured_async("ERROR", message, component, metrics)
        else:
            self.structured_logger.error(message, component, metrics)
    
    def debug(self, message: str, component: str = "SYSTEM", 
              metrics: Optional[Dict[str, Any]] = None) -> None:
        """记录DEBUG级别日志"""
        if self.async_logger:
            self.async_logger.log_structured_async("DEBUG", message, component, metrics)
        else:
            self.structured_logger.debug(message, component, metrics)
    
    def flush(self) -> None:
        """刷新所有日志记录器"""
        if self.async_logger:
            self.async_logger.flush_async()
        else:
            self.tensorboard_logger.flush()
    
    def close(self) -> None:
        """关闭日志记录系统"""
        if self.async_logger:
            self.async_logger.stop()
        
        self.tensorboard_logger.close()
        self.logger.info("日志记录系统已关闭")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """获取日志摘要"""
        return {
            "run_name": self.run_name,
            "log_dir": str(self.log_dir),
            "tensorboard_metrics": self.tensorboard_logger.get_metrics_summary(),
            "structured_log_entries": len(self.structured_logger.log_entries)
        }
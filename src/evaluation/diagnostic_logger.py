"""
诊断日志记录器

提供详细的诊断信息记录、数据质量报告生成和性能监控功能。
"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class DiagnosticLogger:
    """
    诊断日志记录器
    
    提供详细的诊断信息记录、批次处理统计、数据质量指标收集
    和性能指标监控功能。
    """
    
    def __init__(self, 
                 enable_detailed_logging: bool = False,
                 log_batch_statistics: bool = True,
                 save_processing_report: bool = True,
                 output_dir: Optional[str] = None):
        """
        初始化诊断日志记录器
        
        Args:
            enable_detailed_logging: 是否启用详细日志
            log_batch_statistics: 是否记录批次统计
            save_processing_report: 是否保存处理报告
            output_dir: 输出目录
        """
        self.enable_detailed_logging = enable_detailed_logging
        self.log_batch_statistics = log_batch_statistics
        self.save_processing_report = save_processing_report
        self.output_dir = Path(output_dir) if output_dir else Path("logs/diagnostics")
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.batch_statistics = []
        self.data_quality_metrics = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        self.processing_timeline = []
        self.error_log = []
        
        # 会话信息
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = time.time()
        
        logger.info(f"诊断日志记录器已初始化，会话ID: {self.session_id}")
    
    def log_batch_processing(self, 
                           batch_info: Dict[str, Any],
                           processing_result: Dict[str, Any],
                           task_name: str,
                           processing_time: float) -> None:
        """
        记录批次处理信息
        
        Args:
            batch_info: 批次信息
            processing_result: 处理结果
            task_name: 任务名称
            processing_time: 处理时间
        """
        timestamp = datetime.now()
        
        batch_stat = {
            "timestamp": timestamp.isoformat(),
            "task_name": task_name,
            "batch_info": batch_info,
            "processing_result": processing_result,
            "processing_time_ms": processing_time * 1000,
            "session_id": self.session_id
        }
        
        self.batch_statistics.append(batch_stat)
        
        if self.log_batch_statistics:
            logger.info(f"批次处理完成 - 任务: {task_name}, "
                       f"样本数: {batch_info.get('total_samples', 0)}, "
                       f"有效样本: {processing_result.get('valid_sample_count', 0)}, "
                       f"处理时间: {processing_time*1000:.1f}ms")
        
        if self.enable_detailed_logging:
            logger.debug(f"详细批次信息: {json.dumps(batch_stat, indent=2, ensure_ascii=False)}")
    
    def log_data_quality_metrics(self, 
                                task_name: str,
                                metrics: Dict[str, Any]) -> None:
        """
        记录数据质量指标
        
        Args:
            task_name: 任务名称
            metrics: 质量指标
        """
        timestamp = datetime.now()
        
        quality_metric = {
            "timestamp": timestamp.isoformat(),
            "task_name": task_name,
            "metrics": metrics,
            "session_id": self.session_id
        }
        
        self.data_quality_metrics[task_name].append(quality_metric)
        
        if self.enable_detailed_logging:
            logger.debug(f"数据质量指标 - 任务: {task_name}, 指标: {metrics}")
    
    def log_performance_metrics(self, 
                              operation: str,
                              metrics: Dict[str, Any]) -> None:
        """
        记录性能指标
        
        Args:
            operation: 操作名称
            metrics: 性能指标
        """
        timestamp = datetime.now()
        
        perf_metric = {
            "timestamp": timestamp.isoformat(),
            "operation": operation,
            "metrics": metrics,
            "session_id": self.session_id
        }
        
        self.performance_metrics[operation].append(perf_metric)
        
        if self.enable_detailed_logging:
            logger.debug(f"性能指标 - 操作: {operation}, 指标: {metrics}")
    
    def log_processing_event(self, 
                           event_type: str,
                           event_data: Dict[str, Any],
                           level: str = "info") -> None:
        """
        记录处理事件
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
            level: 日志级别
        """
        timestamp = datetime.now()
        
        event = {
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "event_data": event_data,
            "level": level,
            "session_id": self.session_id
        }
        
        self.processing_timeline.append(event)
        
        if level == "error":
            self.error_log.append(event)
            logger.error(f"处理错误 - {event_type}: {event_data}")
        elif level == "warning":
            logger.warning(f"处理警告 - {event_type}: {event_data}")
        elif self.enable_detailed_logging:
            logger.debug(f"处理事件 - {event_type}: {event_data}")
    
    def generate_batch_statistics_report(self) -> Dict[str, Any]:
        """
        生成批次处理统计报告
        
        Returns:
            统计报告字典
        """
        if not self.batch_statistics:
            return {"message": "没有批次处理统计数据"}
        
        # 基本统计
        total_batches = len(self.batch_statistics)
        total_samples = sum(stat["batch_info"].get("total_samples", 0) for stat in self.batch_statistics)
        total_valid_samples = sum(stat["processing_result"].get("valid_sample_count", 0) for stat in self.batch_statistics)
        total_processing_time = sum(stat["processing_time_ms"] for stat in self.batch_statistics)
        
        # 按任务统计
        task_stats = defaultdict(lambda: {
            "batch_count": 0,
            "total_samples": 0,
            "valid_samples": 0,
            "processing_time_ms": 0
        })
        
        for stat in self.batch_statistics:
            task_name = stat["task_name"]
            task_stats[task_name]["batch_count"] += 1
            task_stats[task_name]["total_samples"] += stat["batch_info"].get("total_samples", 0)
            task_stats[task_name]["valid_samples"] += stat["processing_result"].get("valid_sample_count", 0)
            task_stats[task_name]["processing_time_ms"] += stat["processing_time_ms"]
        
        # 计算比率
        for task_name, stats in task_stats.items():
            if stats["total_samples"] > 0:
                stats["valid_sample_ratio"] = stats["valid_samples"] / stats["total_samples"]
            else:
                stats["valid_sample_ratio"] = 0.0
            
            if stats["batch_count"] > 0:
                stats["avg_processing_time_ms"] = stats["processing_time_ms"] / stats["batch_count"]
            else:
                stats["avg_processing_time_ms"] = 0.0
        
        report = {
            "session_id": self.session_id,
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_batches": total_batches,
                "total_samples": total_samples,
                "total_valid_samples": total_valid_samples,
                "overall_valid_ratio": total_valid_samples / max(total_samples, 1),
                "total_processing_time_ms": total_processing_time,
                "avg_processing_time_per_batch_ms": total_processing_time / max(total_batches, 1)
            },
            "task_statistics": dict(task_stats),
            "processing_timeline": self._create_processing_timeline()
        }
        
        return report
    
    def generate_data_quality_report(self) -> Dict[str, Any]:
        """
        生成数据质量报告
        
        Returns:
            数据质量报告字典
        """
        if not self.data_quality_metrics:
            return {"message": "没有数据质量指标数据"}
        
        report = {
            "session_id": self.session_id,
            "report_timestamp": datetime.now().isoformat(),
            "quality_metrics_by_task": {}
        }
        
        for task_name, metrics_list in self.data_quality_metrics.items():
            if not metrics_list:
                continue
            
            # 聚合指标
            aggregated_metrics = self._aggregate_quality_metrics(metrics_list)
            
            report["quality_metrics_by_task"][task_name] = {
                "metric_count": len(metrics_list),
                "aggregated_metrics": aggregated_metrics,
                "latest_metrics": metrics_list[-1]["metrics"] if metrics_list else {}
            }
        
        return report
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        生成性能报告
        
        Returns:
            性能报告字典
        """
        if not self.performance_metrics:
            return {"message": "没有性能指标数据"}
        
        report = {
            "session_id": self.session_id,
            "report_timestamp": datetime.now().isoformat(),
            "performance_metrics_by_operation": {}
        }
        
        for operation, metrics_list in self.performance_metrics.items():
            if not metrics_list:
                continue
            
            # 聚合性能指标
            aggregated_metrics = self._aggregate_performance_metrics(metrics_list)
            
            report["performance_metrics_by_operation"][operation] = {
                "measurement_count": len(metrics_list),
                "aggregated_metrics": aggregated_metrics,
                "latest_metrics": metrics_list[-1]["metrics"] if metrics_list else {}
            }
        
        return report
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        生成综合报告
        
        Returns:
            综合报告字典
        """
        session_duration = time.time() - self.session_start_time
        
        comprehensive_report = {
            "session_info": {
                "session_id": self.session_id,
                "start_time": datetime.fromtimestamp(self.session_start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": session_duration
            },
            "batch_statistics": self.generate_batch_statistics_report(),
            "data_quality": self.generate_data_quality_report(),
            "performance": self.generate_performance_report(),
            "error_summary": self._generate_error_summary(),
            "recommendations": self._generate_recommendations()
        }
        
        return comprehensive_report
    
    def save_report_to_file(self, 
                           report: Dict[str, Any],
                           filename: Optional[str] = None) -> str:
        """
        保存报告到文件
        
        Args:
            report: 报告数据
            filename: 文件名（可选）
            
        Returns:
            保存的文件路径
        """
        if not self.save_processing_report:
            logger.info("处理报告保存已禁用")
            return ""
        
        if filename is None:
            filename = f"diagnostic_report_{self.session_id}.json"
        
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"诊断报告已保存到: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"保存诊断报告失败: {e}")
            return ""
    
    def _create_processing_timeline(self) -> List[Dict[str, Any]]:
        """创建处理时间线"""
        timeline = []
        
        for event in self.processing_timeline:
            timeline.append({
                "timestamp": event["timestamp"],
                "event_type": event["event_type"],
                "level": event["level"]
            })
        
        return timeline
    
    def _aggregate_quality_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合数据质量指标"""
        if not metrics_list:
            return {}
        
        # 收集所有指标值
        all_metrics = defaultdict(list)
        for metric_entry in metrics_list:
            metrics = metric_entry["metrics"]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)
        
        # 计算聚合统计
        aggregated = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregated[metric_name] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return aggregated
    
    def _aggregate_performance_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合性能指标"""
        if not metrics_list:
            return {}
        
        # 收集所有性能指标值
        all_metrics = defaultdict(list)
        for metric_entry in metrics_list:
            metrics = metric_entry["metrics"]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)
        
        # 计算聚合统计
        aggregated = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregated[metric_name] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "total": sum(values) if metric_name.endswith(('_time', '_count', '_size')) else None
                }
        
        return aggregated
    
    def _generate_error_summary(self) -> Dict[str, Any]:
        """生成错误摘要"""
        if not self.error_log:
            return {"total_errors": 0, "error_types": {}}
        
        error_types = Counter(error["event_type"] for error in self.error_log)
        
        return {
            "total_errors": len(self.error_log),
            "error_types": dict(error_types),
            "recent_errors": self.error_log[-5:] if len(self.error_log) > 5 else self.error_log
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于批次统计的建议
        if self.batch_statistics:
            batch_report = self.generate_batch_statistics_report()
            summary = batch_report.get("summary", {})
            
            valid_ratio = summary.get("overall_valid_ratio", 0)
            if valid_ratio < 0.5:
                recommendations.append("整体有效样本比例较低，建议检查数据质量和字段映射配置")
            
            avg_processing_time = summary.get("avg_processing_time_per_batch_ms", 0)
            if avg_processing_time > 1000:  # 超过1秒
                recommendations.append("批次处理时间较长，建议优化数据预处理逻辑或减小批次大小")
        
        # 基于错误日志的建议
        if self.error_log:
            error_summary = self._generate_error_summary()
            if error_summary["total_errors"] > 10:
                recommendations.append("错误数量较多，建议启用详细日志进行问题诊断")
            
            common_errors = error_summary.get("error_types", {})
            if "field_detection_failed" in common_errors:
                recommendations.append("字段检测失败较多，建议检查字段映射配置")
            if "data_validation_failed" in common_errors:
                recommendations.append("数据验证失败较多，建议启用数据清洗功能")
        
        # 基于性能指标的建议
        if self.performance_metrics:
            perf_report = self.generate_performance_report()
            for operation, metrics in perf_report.get("performance_metrics_by_operation", {}).items():
                aggregated = metrics.get("aggregated_metrics", {})
                if "processing_time_ms" in aggregated:
                    avg_time = aggregated["processing_time_ms"].get("mean", 0)
                    if avg_time > 500:  # 超过500ms
                        recommendations.append(f"操作 '{operation}' 处理时间较长，建议进行性能优化")
        
        return recommendations
    
    def reset_statistics(self):
        """重置所有统计信息"""
        self.batch_statistics.clear()
        self.data_quality_metrics.clear()
        self.performance_metrics.clear()
        self.processing_timeline.clear()
        self.error_log.clear()
        
        # 重新初始化会话
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = time.time()
        
        logger.info(f"诊断统计信息已重置，新会话ID: {self.session_id}")
    
    def get_current_statistics(self) -> Dict[str, Any]:
        """
        获取当前统计信息
        
        Returns:
            当前统计信息字典
        """
        return {
            "session_id": self.session_id,
            "batch_count": len(self.batch_statistics),
            "quality_metrics_count": sum(len(metrics) for metrics in self.data_quality_metrics.values()),
            "performance_metrics_count": sum(len(metrics) for metrics in self.performance_metrics.values()),
            "event_count": len(self.processing_timeline),
            "error_count": len(self.error_log),
            "session_duration_seconds": time.time() - self.session_start_time
        }
"""
效率分析器

实现模型效率和性能指标的测量，包括推理延迟、吞吐量、内存使用等。
"""

import logging
import time
import psutil
import threading
from typing import List, Dict, Any, Optional, Callable, Tuple
from contextlib import contextmanager
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, GPU metrics will be disabled")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available, GPU monitoring will be disabled")

from .data_models import EfficiencyMetrics, convert_numpy_types

logger = logging.getLogger(__name__)


class EfficiencyAnalyzer:
    """
    效率分析器
    
    提供模型效率和性能指标的测量功能：
    - 推理延迟和吞吐量测量
    - 内存使用监控
    - FLOPs计算
    - 模型大小统计
    - 效率-准确率权衡分析
    """
    
    def __init__(self, device: str = "cpu", monitor_interval: float = 0.1):
        """
        初始化效率分析器
        
        Args:
            device: 计算设备
            monitor_interval: 监控间隔（秒）
        """
        self.device = device
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.memory_usage = []
        self.gpu_usage = []
        
        logger.info(f"EfficiencyAnalyzer初始化完成，设备: {device}")
    
    @contextmanager
    def measure_inference_time(self):
        """
        测量推理时间的上下文管理器
        
        Returns:
            推理时间（秒）
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            self.last_inference_time = end_time - start_time
    
    def measure_latency_and_throughput(self, 
                                     inference_func: Callable,
                                     inputs: List[Any],
                                     batch_sizes: List[int] = [1, 4, 8, 16],
                                     num_runs: int = 10) -> Dict[str, Any]:
        """
        测量推理延迟和吞吐量
        
        Args:
            inference_func: 推理函数
            inputs: 输入数据列表
            batch_sizes: 批次大小列表
            num_runs: 每个批次大小的运行次数
            
        Returns:
            包含延迟和吞吐量指标的字典
        """
        # 验证输入
        if not inputs:
            logger.warning("延迟测量收到空输入，返回默认结果")
            return {
                "latency": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "throughput": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "batch_results": {}
            }
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"测量批次大小 {batch_size} 的性能...")
            
            latencies = []
            throughputs = []
            
            # 准备批次数据
            batches = self._create_batches(inputs, batch_size)
            
            for run in range(num_runs):
                batch = batches[run % len(batches)]
                
                # 预热
                if run == 0:
                    try:
                        _ = inference_func(batch)
                    except Exception as e:
                        logger.warning(f"预热运行失败: {e}")
                
                # 测量推理时间
                start_time = time.perf_counter()
                try:
                    outputs = inference_func(batch)
                    end_time = time.perf_counter()
                    
                    latency = end_time - start_time
                    throughput = len(batch) / latency if latency > 0 else 0
                    
                    latencies.append(latency)
                    throughputs.append(throughput)
                    
                except Exception as e:
                    logger.error(f"推理失败: {e}")
                    continue
            
            if latencies:
                results[f"batch_size_{batch_size}"] = {
                    "avg_latency_ms": float(np.mean(latencies) * 1000),
                    "std_latency_ms": float(np.std(latencies) * 1000),
                    "min_latency_ms": float(np.min(latencies) * 1000),
                    "max_latency_ms": float(np.max(latencies) * 1000),
                    "avg_throughput": float(np.mean(throughputs)),
                    "std_throughput": float(np.std(throughputs)),
                    "max_throughput": float(np.max(throughputs))
                }
        
        # 计算整体指标
        if results:
            all_latencies = []
            all_throughputs = []
            
            for batch_result in results.values():
                # 使用平均延迟计算每个样本的延迟
                per_sample_latency = batch_result["avg_latency_ms"] / int(batch_result.get("batch_size", 1))
                all_latencies.append(per_sample_latency)
                all_throughputs.append(batch_result["avg_throughput"])
            
            results["overall"] = {
                "avg_latency_per_sample_ms": float(np.mean(all_latencies)),
                "best_throughput": float(np.max(all_throughputs)),
                "optimal_batch_size": int(batch_sizes[np.argmax(all_throughputs)])
            }
        
        return convert_numpy_types(results)
    
    def monitor_memory_usage(self, 
                           inference_func: Callable,
                           inputs: List[Any],
                           duration: float = 60.0) -> Dict[str, Any]:
        """
        监控内存使用情况
        
        Args:
            inference_func: 推理函数
            inputs: 输入数据
            duration: 监控持续时间（秒）
            
        Returns:
            内存使用统计
        """
        self.memory_usage = []
        self.gpu_usage = []
        self.monitoring = True
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=self._memory_monitor)
        monitor_thread.start()
        
        # 记录初始内存
        initial_memory = self._get_memory_usage()
        
        try:
            # 验证输入
            if not inputs:
                logger.warning("内存监控收到空输入，跳过监控")
                return MemoryUsageResult(
                    peak_memory_mb=0.0,
                    average_memory_mb=0.0,
                    memory_samples=[]
                )
            
            # 运行推理
            start_time = time.time()
            while time.time() - start_time < duration:
                try:
                    test_input = inputs[:1] if inputs else ["测试输入"]
                    _ = inference_func(test_input)
                except Exception as e:
                    logger.warning(f"推理过程中出错: {e}")
                    break
        finally:
            # 停止监控
            self.monitoring = False
            monitor_thread.join()
        
        # 记录最终内存
        final_memory = self._get_memory_usage()
        
        # 分析内存使用
        if self.memory_usage:
            memory_stats = {
                "initial_memory_mb": initial_memory["ram_mb"],
                "final_memory_mb": final_memory["ram_mb"],
                "peak_memory_mb": float(np.max([m["ram_mb"] for m in self.memory_usage])),
                "avg_memory_mb": float(np.mean([m["ram_mb"] for m in self.memory_usage])),
                "memory_increase_mb": final_memory["ram_mb"] - initial_memory["ram_mb"]
            }
            
            # GPU内存统计
            if TORCH_AVAILABLE and torch.cuda.is_available():
                memory_stats.update({
                    "initial_gpu_memory_mb": initial_memory.get("gpu_mb", 0),
                    "final_gpu_memory_mb": final_memory.get("gpu_mb", 0),
                    "peak_gpu_memory_mb": float(np.max([m.get("gpu_mb", 0) for m in self.memory_usage])),
                    "avg_gpu_memory_mb": float(np.mean([m.get("gpu_mb", 0) for m in self.memory_usage])),
                    "gpu_memory_increase_mb": final_memory.get("gpu_mb", 0) - initial_memory.get("gpu_mb", 0)
                })
        else:
            memory_stats = {"error": "无法获取内存使用数据"}
        
        return convert_numpy_types(memory_stats)
    
    def calculate_model_size(self, model) -> Dict[str, Any]:
        """
        计算模型大小和参数量
        
        Args:
            model: 模型对象
            
        Returns:
            模型大小统计
        """
        stats = {}
        
        if TORCH_AVAILABLE and hasattr(model, 'parameters'):
            # PyTorch模型
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 估算模型大小（假设float32）
            model_size_mb = total_params * 4 / (1024 * 1024)
            
            stats = {
                "total_parameters": int(total_params),
                "trainable_parameters": int(trainable_params),
                "non_trainable_parameters": int(total_params - trainable_params),
                "model_size_mb": float(model_size_mb),
                "model_size_gb": float(model_size_mb / 1024)
            }
            
            # 按层统计
            layer_stats = {}
            for name, param in model.named_parameters():
                layer_name = name.split('.')[0]  # 获取层名
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {"params": 0, "size_mb": 0}
                
                layer_stats[layer_name]["params"] += param.numel()
                layer_stats[layer_name]["size_mb"] += param.numel() * 4 / (1024 * 1024)
            
            stats["layer_statistics"] = {
                name: {
                    "parameters": int(info["params"]),
                    "size_mb": float(info["size_mb"])
                }
                for name, info in layer_stats.items()
            }
        
        else:
            # 尝试其他方法获取模型信息
            try:
                import sys
                model_size_bytes = sys.getsizeof(model)
                stats = {
                    "estimated_size_mb": float(model_size_bytes / (1024 * 1024)),
                    "note": "使用sys.getsizeof估算，可能不准确"
                }
            except Exception as e:
                stats = {"error": f"无法计算模型大小: {e}"}
        
        return convert_numpy_types(stats)
    
    def estimate_flops(self, 
                      model,
                      input_shape: Tuple[int, ...],
                      num_samples: int = 1) -> Dict[str, Any]:
        """
        估算FLOPs（浮点运算次数）
        
        Args:
            model: 模型对象
            input_shape: 输入形状
            num_samples: 样本数量
            
        Returns:
            FLOPs统计
        """
        try:
            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                # 使用简单的参数量估算
                total_params = sum(p.numel() for p in model.parameters())
                
                # 粗略估算：每个参数大约需要2个FLOPs（乘法和加法）
                estimated_flops = total_params * 2 * num_samples
                
                # 考虑输入大小的影响
                input_size = np.prod(input_shape)
                flops_per_sample = estimated_flops / num_samples
                
                stats = {
                    "estimated_flops": int(estimated_flops),
                    "flops_per_sample": int(flops_per_sample),
                    "gflops": float(estimated_flops / 1e9),
                    "input_size": int(input_size),
                    "note": "基于参数量的粗略估算"
                }
            else:
                stats = {"error": "无法估算FLOPs，不支持的模型类型"}
                
        except Exception as e:
            stats = {"error": f"FLOPs估算失败: {e}"}
        
        return convert_numpy_types(stats)
    
    def analyze_efficiency_accuracy_tradeoff(self, 
                                           models_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析效率-准确率权衡（Pareto前沿）
        
        Args:
            models_data: 模型数据列表，每个包含accuracy, latency, memory等字段
            
        Returns:
            权衡分析结果
        """
        if not models_data:
            return {"error": "没有提供模型数据"}
        
        # 提取数据
        accuracies = [data.get("accuracy", 0) for data in models_data]
        latencies = [data.get("latency_ms", float('inf')) for data in models_data]
        memory_usage = [data.get("memory_mb", float('inf')) for data in models_data]
        model_names = [data.get("name", f"model_{i}") for i, data in enumerate(models_data)]
        
        # 计算效率分数（越高越好）
        efficiency_scores = []
        for acc, lat, mem in zip(accuracies, latencies, memory_usage):
            # 归一化指标
            lat_norm = lat / max(latencies) if max(latencies) > 0 else 0
            mem_norm = mem / max(memory_usage) if max(memory_usage) > 0 else 0
            
            # 效率分数：准确率高，延迟和内存使用低
            efficiency = acc / (1 + lat_norm + mem_norm)
            efficiency_scores.append(efficiency)
        
        # 找到Pareto前沿
        pareto_indices = self._find_pareto_frontier(accuracies, latencies)
        
        # 排序模型
        sorted_indices = sorted(range(len(efficiency_scores)), 
                              key=lambda i: efficiency_scores[i], 
                              reverse=True)
        
        results = {
            "model_rankings": [
                {
                    "rank": rank + 1,
                    "name": model_names[idx],
                    "accuracy": float(accuracies[idx]),
                    "latency_ms": float(latencies[idx]),
                    "memory_mb": float(memory_usage[idx]),
                    "efficiency_score": float(efficiency_scores[idx]),
                    "is_pareto_optimal": idx in pareto_indices
                }
                for rank, idx in enumerate(sorted_indices)
            ],
            "pareto_frontier": [
                {
                    "name": model_names[idx],
                    "accuracy": float(accuracies[idx]),
                    "latency_ms": float(latencies[idx]),
                    "memory_mb": float(memory_usage[idx])
                }
                for idx in pareto_indices
            ],
            "best_accuracy": {
                "name": model_names[np.argmax(accuracies)],
                "accuracy": float(max(accuracies))
            },
            "best_efficiency": {
                "name": model_names[np.argmax(efficiency_scores)],
                "efficiency_score": float(max(efficiency_scores))
            },
            "fastest_model": {
                "name": model_names[np.argmin(latencies)],
                "latency_ms": float(min(latencies))
            }
        }
        
        return convert_numpy_types(results)
    
    def create_efficiency_metrics(self, 
                                inference_latency: float,
                                throughput: float,
                                memory_usage: float,
                                model_size: float,
                                flops: Optional[int] = None) -> EfficiencyMetrics:
        """
        创建效率指标对象
        
        Args:
            inference_latency: 推理延迟（ms）
            throughput: 吞吐量（tokens/s）
            memory_usage: 内存使用（GB）
            model_size: 模型大小（MB）
            flops: FLOPs
            
        Returns:
            EfficiencyMetrics对象
        """
        return EfficiencyMetrics(
            inference_latency=float(inference_latency),
            throughput=float(throughput),
            memory_usage=float(memory_usage),
            model_size=float(model_size),
            flops=int(flops) if flops is not None else None
        )
    
    def _create_batches(self, inputs: List[Any], batch_size: int) -> List[List[Any]]:
        """创建批次数据"""
        batches = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def _memory_monitor(self):
        """内存监控线程"""
        while self.monitoring:
            memory_info = self._get_memory_usage()
            self.memory_usage.append(memory_info)
            time.sleep(self.monitor_interval)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        memory_info = {}
        
        # RAM使用情况
        process = psutil.Process()
        memory_info["ram_mb"] = process.memory_info().rss / (1024 * 1024)
        
        # GPU内存使用情况
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                memory_info["gpu_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            except Exception as e:
                logger.warning(f"获取GPU内存信息失败: {e}")
        
        return memory_info
    
    def _find_pareto_frontier(self, 
                            accuracies: List[float], 
                            latencies: List[float]) -> List[int]:
        """
        找到Pareto前沿
        
        Args:
            accuracies: 准确率列表（越高越好）
            latencies: 延迟列表（越低越好）
            
        Returns:
            Pareto最优解的索引列表
        """
        pareto_indices = []
        
        for i, (acc1, lat1) in enumerate(zip(accuracies, latencies)):
            is_pareto = True
            
            for j, (acc2, lat2) in enumerate(zip(accuracies, latencies)):
                if i != j:
                    # 如果存在另一个解在两个维度上都不差于当前解，且至少在一个维度上更好
                    if (acc2 >= acc1 and lat2 <= lat1) and (acc2 > acc1 or lat2 < lat1):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
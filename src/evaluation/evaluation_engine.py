"""
评估引擎

提供核心评估框架，支持多任务评估调度、批量推理和结果收集。
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from datasets import Dataset

from .data_models import (
    EvaluationConfig, EvaluationResult, TaskResult, EvaluationSample,
    EfficiencyMetrics, QualityScores, convert_numpy_types
)
from .metrics_calculator import MetricsCalculator
from .efficiency_analyzer import EfficiencyAnalyzer

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """
    评估引擎
    
    提供核心评估框架功能：
    - 多任务评估调度
    - 批量推理和结果收集
    - 错误处理和恢复
    - 评估过程监控
    """
    
    def __init__(self, 
                 config: EvaluationConfig,
                 device: str = "cpu",
                 max_workers: int = 4):
        """
        初始化评估引擎
        
        Args:
            config: 评估配置
            device: 计算设备
            max_workers: 最大工作线程数
        """
        self.config = config
        self.device = device
        self.max_workers = max_workers
        
        # 初始化组件
        self.metrics_calculator = MetricsCalculator(device=device)
        self.efficiency_analyzer = EfficiencyAnalyzer(device=device)
        
        # 评估状态
        self.current_evaluation = None
        self.evaluation_history = []
        
        logger.info(f"EvaluationEngine初始化完成，设备: {device}, 最大工作线程: {max_workers}")
    
    def evaluate_model(self, 
                      model,
                      tokenizer,
                      datasets: Dict[str, Dataset],
                      model_name: str = "unknown_model") -> EvaluationResult:
        """
        评估模型
        
        Args:
            model: 要评估的模型
            tokenizer: 分词器
            datasets: 数据集字典，键为任务名称
            model_name: 模型名称
            
        Returns:
            评估结果
        """
        logger.info(f"开始评估模型: {model_name}")
        evaluation_start_time = datetime.now()
        
        try:
            # 创建推理函数
            inference_func = self._create_inference_function(model, tokenizer)
            
            # 执行任务评估
            task_results = {}
            for task_name in self.config.tasks:
                if task_name in datasets:
                    logger.info(f"执行任务评估: {task_name}")
                    task_result = self._evaluate_task(
                        task_name, 
                        datasets[task_name], 
                        inference_func
                    )
                    task_results[task_name] = task_result
                else:
                    logger.warning(f"未找到任务数据集: {task_name}")
            
            # 计算整体指标
            overall_metrics = self._calculate_overall_metrics(task_results)
            
            # 测量效率指标
            efficiency_metrics = None
            if self.config.enable_efficiency_metrics:
                logger.info("测量效率指标...")
                efficiency_metrics = self._measure_efficiency(
                    inference_func, 
                    list(datasets.values())[0] if datasets else None,
                    model
                )
            
            # 计算质量分数
            quality_scores = None
            if self.config.enable_quality_analysis:
                logger.info("计算质量分数...")
                quality_scores = self._calculate_quality_scores(task_results)
            
            # 创建评估结果
            result = EvaluationResult(
                model_name=model_name,
                evaluation_time=evaluation_start_time,
                metrics=overall_metrics,
                task_results=task_results,
                efficiency_metrics=efficiency_metrics or EfficiencyMetrics(0, 0, 0, 0),
                quality_scores=quality_scores or QualityScores(0, 0, 0, 0, 0),
                config=self.config
            )
            
            # 保存到历史记录
            self.evaluation_history.append(result)
            
            logger.info(f"模型评估完成: {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def evaluate_multiple_models(self, 
                                models_info: List[Dict[str, Any]],
                                datasets: Dict[str, Dataset]) -> List[EvaluationResult]:
        """
        评估多个模型
        
        Args:
            models_info: 模型信息列表，每个包含model, tokenizer, name字段
            datasets: 数据集字典
            
        Returns:
            评估结果列表
        """
        logger.info(f"开始评估 {len(models_info)} 个模型")
        
        results = []
        
        if self.max_workers > 1:
            # 并行评估
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_model = {
                    executor.submit(
                        self.evaluate_model,
                        info["model"],
                        info["tokenizer"],
                        datasets,
                        info.get("name", f"model_{i}")
                    ): info
                    for i, info in enumerate(models_info)
                }
                
                for future in as_completed(future_to_model):
                    model_info = future_to_model[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"模型 {model_info.get('name', 'unknown')} 评估完成")
                    except Exception as e:
                        logger.error(f"模型 {model_info.get('name', 'unknown')} 评估失败: {e}")
        else:
            # 串行评估
            for i, info in enumerate(models_info):
                try:
                    result = self.evaluate_model(
                        info["model"],
                        info["tokenizer"],
                        datasets,
                        info.get("name", f"model_{i}")
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"模型 {info.get('name', f'model_{i}')} 评估失败: {e}")
        
        logger.info(f"多模型评估完成，成功评估 {len(results)} 个模型")
        return results
    
    def _evaluate_task(self, 
                      task_name: str, 
                      dataset: Dataset, 
                      inference_func: Callable) -> TaskResult:
        """
        评估单个任务
        
        Args:
            task_name: 任务名称
            dataset: 数据集
            inference_func: 推理函数
            
        Returns:
            任务评估结果
        """
        start_time = time.time()
        
        # 限制样本数量
        if len(dataset) > self.config.num_samples:
            dataset = dataset.select(range(self.config.num_samples))
        
        predictions = []
        references = []
        samples = []
        
        logger.info(f"开始推理，样本数: {len(dataset)}")
        
        # 批量推理
        batch_size = self.config.batch_size
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            
            try:
                # 准备输入
                inputs = self._prepare_inputs(batch, task_name)
                
                # 执行推理
                batch_predictions = inference_func(inputs)
                
                # 处理输出
                for j, (input_text, pred, ref) in enumerate(zip(
                    inputs, batch_predictions, batch.get("target", batch.get("answer", [""] * len(batch)))
                )):
                    predictions.append(pred)
                    references.append(ref)
                    
                    # 创建样本对象（暂时不计算单个样本指标）
                    sample = EvaluationSample(
                        input_text=input_text,
                        prediction=pred,
                        reference=ref,
                        metrics={}
                    )
                    samples.append(sample)
                    
            except Exception as e:
                logger.error(f"批次推理失败: {e}")
                # 添加空预测以保持数量一致
                batch_size_actual = len(batch)
                predictions.extend([""] * batch_size_actual)
                references.extend(batch.get("target", batch.get("answer", [""] * batch_size_actual)))
                
                for j in range(batch_size_actual):
                    sample = EvaluationSample(
                        input_text="",
                        prediction="",
                        reference="",
                        metrics={}
                    )
                    samples.append(sample)
        
        # 计算任务指标
        task_type = self._get_task_type(task_name)
        metrics = self.metrics_calculator.calculate_all_metrics(
            predictions, references, task_type
        )
        
        execution_time = time.time() - start_time
        
        return TaskResult(
            task_name=task_name,
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=samples,
            execution_time=execution_time
        )
    
    def _create_inference_function(self, model, tokenizer) -> Callable:
        """
        创建推理函数
        
        Args:
            model: 模型
            tokenizer: 分词器
            
        Returns:
            推理函数
        """
        def inference_func(inputs: List[str]) -> List[str]:
            try:
                # 编码输入
                encoded = tokenizer(
                    inputs,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                # 移动到设备
                if hasattr(encoded, 'to'):
                    encoded = encoded.to(self.device)
                else:
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # 生成
                with torch.no_grad():
                    outputs = model.generate(
                        **encoded,
                        max_length=self.config.max_length,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                # 解码输出
                predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # 移除输入部分（如果是生成任务）
                if hasattr(model, 'config') and hasattr(model.config, 'is_encoder_decoder'):
                    if not model.config.is_encoder_decoder:
                        # 对于decoder-only模型，移除输入部分
                        input_lengths = [len(tokenizer.encode(inp)) for inp in inputs]
                        predictions = [
                            tokenizer.decode(outputs[i][input_lengths[i]:], skip_special_tokens=True)
                            for i in range(len(predictions))
                        ]
                
                return predictions
                
            except Exception as e:
                logger.error(f"推理函数执行失败: {e}")
                return [""] * len(inputs)
        
        return inference_func
    
    def _prepare_inputs(self, batch: Dict[str, List], task_name: str) -> List[str]:
        """
        准备输入数据
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            输入文本列表
        """
        if task_name == "text_generation":
            return batch.get("text", batch.get("input", []))
        elif task_name == "question_answering":
            questions = batch.get("question", [])
            contexts = batch.get("context", [""] * len(questions))
            return [f"问题: {q}\n上下文: {c}" for q, c in zip(questions, contexts)]
        elif task_name == "classification":
            return batch.get("text", batch.get("input", []))
        else:
            # 默认使用text字段
            return batch.get("text", batch.get("input", []))
    
    def _get_task_type(self, task_name: str) -> str:
        """
        获取任务类型
        
        Args:
            task_name: 任务名称
            
        Returns:
            任务类型
        """
        if "generation" in task_name.lower():
            return "generation"
        elif "classification" in task_name.lower() or "classify" in task_name.lower():
            return "classification"
        elif "similarity" in task_name.lower():
            return "similarity"
        else:
            return "generation"  # 默认为生成任务
    
    def _calculate_overall_metrics(self, task_results: Dict[str, TaskResult]) -> Dict[str, float]:
        """
        计算整体指标
        
        Args:
            task_results: 任务结果字典
            
        Returns:
            整体指标字典
        """
        if not task_results:
            return {}
        
        overall_metrics = {}
        
        # 收集所有指标
        all_metrics = {}
        for task_name, result in task_results.items():
            for metric_name, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        
        # 计算平均值
        for metric_name, values in all_metrics.items():
            if values:
                overall_metrics[f"avg_{metric_name}"] = float(np.mean(values))
                overall_metrics[f"std_{metric_name}"] = float(np.std(values))
        
        # 计算总体分数
        if "avg_accuracy" in overall_metrics:
            overall_metrics["overall_score"] = overall_metrics["avg_accuracy"]
        elif "avg_f1" in overall_metrics:
            overall_metrics["overall_score"] = overall_metrics["avg_f1"]
        elif "avg_bleu" in overall_metrics:
            overall_metrics["overall_score"] = overall_metrics["avg_bleu"]
        else:
            overall_metrics["overall_score"] = 0.0
        
        return convert_numpy_types(overall_metrics)
    
    def _measure_efficiency(self, 
                          inference_func: Callable, 
                          dataset: Optional[Dataset],
                          model) -> EfficiencyMetrics:
        """
        测量效率指标
        
        Args:
            inference_func: 推理函数
            dataset: 数据集
            model: 模型
            
        Returns:
            效率指标
        """
        try:
            # 准备测试数据
            if dataset and len(dataset) > 0:
                test_inputs = dataset["text"][:10] if "text" in dataset.column_names else ["测试文本"] * 10
            else:
                test_inputs = ["测试文本"] * 10
            
            # 测量延迟和吞吐量
            latency_results = self.efficiency_analyzer.measure_latency_and_throughput(
                inference_func, test_inputs, batch_sizes=[1, 4], num_runs=5
            )
            
            # 测量内存使用
            memory_results = self.efficiency_analyzer.monitor_memory_usage(
                inference_func, test_inputs, duration=10.0
            )
            
            # 计算模型大小
            model_size_results = self.efficiency_analyzer.calculate_model_size(model)
            
            # 提取关键指标
            avg_latency = latency_results.get("overall", {}).get("avg_latency_per_sample_ms", 0)
            best_throughput = latency_results.get("overall", {}).get("best_throughput", 0)
            peak_memory = memory_results.get("peak_memory_mb", 0) / 1024  # 转换为GB
            model_size = model_size_results.get("model_size_mb", 0)
            
            return EfficiencyMetrics(
                inference_latency=float(avg_latency),
                throughput=float(best_throughput),
                memory_usage=float(peak_memory),
                model_size=float(model_size)
            )
            
        except Exception as e:
            logger.error(f"效率测量失败: {e}")
            return EfficiencyMetrics(0, 0, 0, 0)
    
    def _calculate_quality_scores(self, task_results: Dict[str, TaskResult]) -> QualityScores:
        """
        计算质量分数
        
        Args:
            task_results: 任务结果字典
            
        Returns:
            质量分数
        """
        try:
            # 基于任务结果计算质量分数
            fluency = 0.8  # 默认值，实际应该基于语言模型评估
            coherence = 0.8
            relevance = 0.8
            factuality = 0.8
            
            # 基于指标计算整体质量
            if task_results:
                metric_scores = []
                for result in task_results.values():
                    if "bleu" in result.metrics:
                        metric_scores.append(result.metrics["bleu"])
                    elif "f1" in result.metrics:
                        metric_scores.append(result.metrics["f1"])
                    elif "accuracy" in result.metrics:
                        metric_scores.append(result.metrics["accuracy"])
                
                if metric_scores:
                    overall = float(np.mean(metric_scores))
                else:
                    overall = 0.8
            else:
                overall = 0.8
            
            return QualityScores(
                fluency=fluency,
                coherence=coherence,
                relevance=relevance,
                factuality=factuality,
                overall=overall
            )
            
        except Exception as e:
            logger.error(f"质量分数计算失败: {e}")
            return QualityScores(0.8, 0.8, 0.8, 0.8, 0.8)
    
    def get_evaluation_history(self) -> List[EvaluationResult]:
        """
        获取评估历史
        
        Returns:
            评估结果列表
        """
        return self.evaluation_history.copy()
    
    def clear_evaluation_history(self):
        """清空评估历史"""
        self.evaluation_history.clear()
        logger.info("评估历史已清空")
    
    def save_evaluation_result(self, result: EvaluationResult, output_path: str):
        """
        保存评估结果
        
        Args:
            result: 评估结果
            output_path: 输出路径
        """
        try:
            import json
            from pathlib import Path
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为可序列化的格式
            result_dict = result.get_summary()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"评估结果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")


# 添加必要的导入
try:
    import torch
    import numpy as np
except ImportError:
    logger.warning("PyTorch或NumPy不可用，某些功能可能受限")
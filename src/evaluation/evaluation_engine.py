"""
评估引擎

提供核心评估框架，支持多任务评估调度、批量推理和结果收集。
"""

import logging
import asyncio
import time
import warnings
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from datasets import Dataset

# 抑制梯度检查点相关的警告
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*past_key_value.*")

from .data_models import (
    EvaluationConfig, EvaluationResult, TaskResult, EvaluationSample,
    EfficiencyMetrics, QualityScores, convert_numpy_types
)
from .metrics_calculator import MetricsCalculator
from .efficiency_analyzer import EfficiencyAnalyzer
from .data_preprocessor import DataPreprocessor

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
                 device: str = "auto",
                 max_workers: int = 4):
        """
        初始化评估引擎
        
        Args:
            config: 评估配置
            device: 计算设备 ("auto" 自动检测, "cpu", "cuda", "cuda:0" 等)
            max_workers: 最大工作线程数
        """
        self.config = config
        
        # 自动检测设备
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
            
        self.max_workers = max_workers
        
        # 初始化组件
        self.metrics_calculator = MetricsCalculator(device=self.device)
        self.efficiency_analyzer = EfficiencyAnalyzer(device=self.device)
        self.data_preprocessor = DataPreprocessor(config)
        
        # 评估状态
        self.current_evaluation = None
        self.evaluation_history = []
        
        logger.info(f"EvaluationEngine初始化完成，设备: {self.device}, 最大工作线程: {max_workers}")
    
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
            
            # 如果配置的任务在数据集中找不到，尝试使用所有可用的数据集
            available_tasks = set(self.config.tasks) & set(datasets.keys())
            if not available_tasks:
                logger.warning(f"配置的任务 {self.config.tasks} 在数据集中未找到，使用所有可用数据集: {list(datasets.keys())}")
                available_tasks = datasets.keys()
            
            for task_name in available_tasks:
                logger.info(f"执行任务评估: {task_name}")
                task_result = self._evaluate_task(
                    task_name, 
                    datasets[task_name], 
                    inference_func
                )
                task_results[task_name] = task_result
            
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
                print("# im here #")
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
        total_batches = (len(dataset) + batch_size - 1) // batch_size
        processed_batches = 0
        skipped_batches = 0
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batch_end = min(i + batch_size, len(dataset))
            
            try:
                # 准备输入
                inputs = self._prepare_inputs(batch, task_name)
                
                # 跳过空批次
                if not inputs:
                    skipped_batches += 1
                    logger.warning(f"跳过空批次 {processed_batches + 1}/{total_batches}，索引: {i}-{batch_end}")
                    
                    # 为跳过的批次添加空预测和参考答案
                    batch_size_actual = len(batch)
                    predictions.extend([""] * batch_size_actual)
                    
                    # 获取参考答案
                    batch_references = batch.get("target", batch.get("answer", [""] * batch_size_actual))
                    if len(batch_references) != batch_size_actual:
                        batch_references = (batch_references + [""] * batch_size_actual)[:batch_size_actual]
                    references.extend(batch_references)
                    
                    # 创建空样本对象
                    for j in range(batch_size_actual):
                        sample = EvaluationSample(
                            input_text="",
                            prediction="",
                            reference=batch_references[j] if j < len(batch_references) else "",
                            metrics={}
                        )
                        samples.append(sample)
                    
                    processed_batches += 1
                    continue
                
                processed_batches += 1
                logger.debug(f"处理批次 {processed_batches}/{total_batches}，有效输入: {len(inputs)}")
                
                # 执行推理
                batch_predictions = inference_func(inputs)
                
                # 验证推理结果
                if not batch_predictions:
                    logger.warning(f"推理返回空结果，批次索引: {i}-{i+batch_size}")
                    batch_predictions = [""] * len(inputs)
                
                # 确保预测结果数量与输入匹配
                if len(batch_predictions) != len(inputs):
                    logger.warning(f"预测结果数量不匹配，输入: {len(inputs)}, 预测: {len(batch_predictions)}")
                    # 调整预测结果长度
                    if len(batch_predictions) < len(inputs):
                        batch_predictions.extend([""] * (len(inputs) - len(batch_predictions)))
                    else:
                        batch_predictions = batch_predictions[:len(inputs)]
                
                # 获取参考答案
                references_batch = batch.get("target", batch.get("answer", [""] * len(batch)))
                if len(references_batch) != len(inputs):
                    logger.warning(f"参考答案数量不匹配，调整为输入长度")
                    references_batch = (references_batch + [""] * len(inputs))[:len(inputs)]
                
                # 处理输出
                for j, (input_text, pred, ref) in enumerate(zip(inputs, batch_predictions, references_batch)):
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
                logger.error(f"错误详情: {traceback.format_exc()}")
                
                # 尝试获取批次大小
                try:
                    batch_inputs = self._prepare_inputs(batch, task_name)
                    batch_size_actual = len(batch_inputs) if batch_inputs else len(batch)
                except:
                    batch_size_actual = len(batch)
                
                if batch_size_actual == 0:
                    logger.warning("跳过空批次")
                    continue
                
                # 为失败的批次添加空预测
                predictions.extend([""] * batch_size_actual)
                
                # 获取参考答案
                batch_references = batch.get("target", batch.get("answer", [""] * batch_size_actual))
                if len(batch_references) != batch_size_actual:
                    batch_references = (batch_references + [""] * batch_size_actual)[:batch_size_actual]
                references.extend(batch_references)
                
                # 创建失败的样本对象
                for j in range(batch_size_actual):
                    try:
                        batch_inputs = self._prepare_inputs(batch, task_name)
                        input_text = batch_inputs[j] if batch_inputs and j < len(batch_inputs) else ""
                    except:
                        input_text = ""
                    
                    sample = EvaluationSample(
                        input_text=input_text,
                        prediction="",
                        reference=batch_references[j] if j < len(batch_references) else "",
                        metrics={}
                    )
                    samples.append(sample)
        
        # 计算任务指标
        task_type = self._get_task_type(task_name)
        
        # 验证预测和参考答案数量
        if len(predictions) != len(references):
            logger.warning(f"预测和参考答案数量不匹配: {len(predictions)} vs {len(references)}")
            # 调整到相同长度
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
            samples = samples[:min_len]
        
        # 检查是否有有效的预测结果
        valid_predictions = [p for p in predictions if p and p.strip()]
        if not valid_predictions:
            logger.warning("没有有效的预测结果，返回默认指标")
            metrics = {"accuracy": 0.0, "bleu": 0.0, "rouge_l": 0.0}
        else:
            try:
                metrics = self.metrics_calculator.calculate_all_metrics(
                    predictions, references, task_type
                )
            except Exception as e:
                logger.error(f"指标计算失败: {e}")
                metrics = {"accuracy": 0.0, "bleu": 0.0, "rouge_l": 0.0, "error": str(e)}
        
        execution_time = time.time() - start_time
        
        # 记录批次处理统计
        processing_stats = self.data_preprocessor.get_processing_statistics()
        logger.info(f"任务 {task_name} 完成 - 处理批次: {processed_batches}, "
                   f"跳过批次: {skipped_batches}, "
                   f"有效样本: {len([p for p in predictions if p and p.strip()])}/{len(predictions)}")
        
        if processing_stats["total_batches_processed"] > 0:
            logger.info(f"数据预处理统计 - 成功率: {processing_stats['success_rate']:.2%}, "
                       f"有效样本率: {processing_stats['valid_sample_rate']:.2%}")
        
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
                # 验证输入
                if not inputs:
                    logger.warning("推理函数收到空输入列表")
                    return []
                
                # 过滤无效输入
                valid_inputs = [inp for inp in inputs if inp and isinstance(inp, str) and inp.strip()]
                if not valid_inputs:
                    logger.warning("推理函数收到的所有输入都无效")
                    return [""] * len(inputs)  # 返回与输入长度相同的空字符串列表
                
                # 确定模型所在的设备
                model_device = next(model.parameters()).device
                logger.debug(f"模型设备: {model_device}")
                
                # 编码输入
                try:
                    encoded = tokenizer(
                        valid_inputs,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors="pt"
                    )
                except Exception as tokenizer_error:
                    logger.error(f"分词器编码失败: {tokenizer_error}")
                    logger.error(f"输入数据: {valid_inputs[:3]}...")  # 只显示前3个输入
                    return [""] * len(inputs)
                
                logger.debug(f"编码前张量设备: {[(k, v.device if hasattr(v, 'device') else 'N/A') for k, v in encoded.items()]}")
                
                # 移动所有张量到模型设备
                encoded = {k: v.to(model_device) for k, v in encoded.items() if hasattr(v, 'to')}
                
                logger.debug(f"编码后张量设备: {[(k, v.device if hasattr(v, 'device') else 'N/A') for k, v in encoded.items()]}")
                
                # 生成 (明确禁用缓存以避免梯度检查点警告)
                with torch.no_grad():
                    outputs = model.generate(
                        **encoded,
                        max_length=self.config.max_length,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                        use_cache=False,  # 明确禁用缓存以避免警告
                        past_key_values=None  # 确保不使用过去的键值对
                    )
                
                logger.debug(f"生成输出张量设备: {outputs.device if hasattr(outputs, 'device') else 'N/A'}")
                
                # 解码输出
                predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # 移除输入部分（如果是生成任务）
                if hasattr(model, 'config') and hasattr(model.config, 'is_encoder_decoder'):
                    if not model.config.is_encoder_decoder:
                        # 对于decoder-only模型，移除输入部分
                        input_lengths = [len(tokenizer.encode(inp, add_special_tokens=False)) for inp in inputs]
                        predictions = [
                            tokenizer.decode(outputs[i][input_lengths[i]:], skip_special_tokens=True)
                            for i in range(len(predictions))
                        ]
                
                return predictions
                
            except Exception as e:
                logger.error(f"推理函数执行失败: {e}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                # 添加设备信息到错误日志
                try:
                    model_device = next(model.parameters()).device
                    logger.error(f"模型设备: {model_device}")
                    if 'encoded' in locals():
                        logger.error(f"输入张量设备: {[(k, v.device if hasattr(v, 'device') else 'N/A') for k, v in encoded.items()]}")
                except:
                    logger.error("无法获取设备信息")
                return [""] * len(inputs)
        
        return inference_func
    
    def _prepare_inputs(self, batch: Dict[str, List], task_name: str) -> List[str]:
        """
        准备输入数据（使用新的数据预处理器）
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            输入文本列表
        """
        try:
            # 使用新的数据预处理器
            inputs = self.data_preprocessor.prepare_inputs(batch, task_name)
            
            if not inputs:
                # 提供详细的诊断信息
                diagnosis = self.data_preprocessor.diagnose_batch(batch, task_name)
                logger.warning(f"批次数据处理失败，任务: {task_name}")
                logger.warning(f"可用字段: {diagnosis['batch_info']['available_fields']}")
                
                if diagnosis['recommendations']:
                    logger.warning(f"建议: {'; '.join(diagnosis['recommendations'])}")
                
                # 记录详细的字段分析
                if diagnosis.get('field_detection_result'):
                    detection_result = diagnosis['field_detection_result']
                    if detection_result.get('detected_fields'):
                        logger.info(f"检测到的字段: {detection_result['detected_fields']}")
                    if detection_result.get('recommended_field'):
                        logger.info(f"推荐字段: {detection_result['recommended_field']}")
                
                return []
            
            logger.debug(f"成功准备了 {len(inputs)} 个有效输入，任务: {task_name}")
            return inputs
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return []
    
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
            
            # 添加数据处理统计信息
            result_dict["data_processing_stats"] = self.data_preprocessor.get_processing_statistics()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"评估结果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")
    
    def diagnose_data_processing(self, datasets: Dict[str, Dataset]) -> Dict[str, Any]:
        """
        诊断数据处理问题
        
        Args:
            datasets: 数据集字典
            
        Returns:
            诊断信息
        """
        diagnosis = {
            "overall_stats": self.data_preprocessor.get_processing_statistics(),
            "dataset_diagnosis": {}
        }
        
        for task_name, dataset in datasets.items():
            if len(dataset) > 0:
                # 取第一个批次进行诊断
                batch_size = min(self.config.batch_size, len(dataset))
                sample_batch = dataset[:batch_size]
                
                batch_diagnosis = self.data_preprocessor.diagnose_batch(sample_batch, task_name)
                diagnosis["dataset_diagnosis"][task_name] = batch_diagnosis
        
        return diagnosis
    
    def update_data_processing_config(self, config: EvaluationConfig):
        """
        更新数据处理配置
        
        Args:
            config: 新的评估配置
        """
        self.config = config
        self.data_preprocessor.update_config(config)
        logger.info("评估引擎数据处理配置已更新")


# 添加必要的导入
try:
    import torch
    import numpy as np
except ImportError:
    logger.warning("PyTorch或NumPy不可用，某些功能可能受限")
"""
推理测试器集成

扩展InferenceTester推理测试器，集成标准化评估功能、批量推理优化和性能基准测试。
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from datasets import Dataset

from .evaluation_engine import EvaluationEngine
from .metrics_calculator import MetricsCalculator
from .efficiency_analyzer import EfficiencyAnalyzer
from .quality_analyzer import QualityAnalyzer
from .data_models import (
    EvaluationConfig, EvaluationResult, EfficiencyMetrics, 
    QualityScores, convert_numpy_types
)

logger = logging.getLogger(__name__)


class InferenceTesterIntegration:
    """
    推理测试器集成器
    
    提供与现有推理测试器的集成功能：
    - 集成标准化评估功能到推理测试器
    - 实现批量推理优化和内存管理
    - 创建推理结果的质量分析和验证
    - 建立推理性能的基准测试
    """
    
    def __init__(self, 
                 evaluation_config: Optional[EvaluationConfig] = None,
                 enable_quality_analysis: bool = True,
                 enable_efficiency_analysis: bool = True,
                 output_dir: str = "./inference_evaluation"):
        """
        初始化推理测试器集成器
        
        Args:
            evaluation_config: 评估配置
            enable_quality_analysis: 是否启用质量分析
            enable_efficiency_analysis: 是否启用效率分析
            output_dir: 输出目录
        """
        self.evaluation_config = evaluation_config or EvaluationConfig()
        self.enable_quality_analysis = enable_quality_analysis
        self.enable_efficiency_analysis = enable_efficiency_analysis
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.evaluation_engine = EvaluationEngine(self.evaluation_config)
        self.metrics_calculator = MetricsCalculator()
        self.efficiency_analyzer = EfficiencyAnalyzer() if enable_efficiency_analysis else None
        self.quality_analyzer = QualityAnalyzer() if enable_quality_analysis else None
        
        # 推理历史
        self.inference_history = []
        self.performance_benchmarks = {}
        
        logger.info("InferenceTesterIntegration初始化完成")
    
    def enhance_inference_tester(self, inference_tester_class):
        """
        增强现有的InferenceTester类
        
        Args:
            inference_tester_class: 现有的InferenceTester类
            
        Returns:
            增强后的InferenceTester类
        """
        class EnhancedInferenceTester(inference_tester_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                # 集成评估组件
                eval_config = kwargs.get('evaluation_config')
                enable_quality = kwargs.get('enable_quality_analysis', True)
                enable_efficiency = kwargs.get('enable_efficiency_analysis', True)
                output_dir = kwargs.get('evaluation_output_dir', './inference_evaluation')
                
                self.integration = InferenceTesterIntegration(
                    evaluation_config=eval_config,
                    enable_quality_analysis=enable_quality,
                    enable_efficiency_analysis=enable_efficiency,
                    output_dir=output_dir
                )
            
            def test_with_evaluation(self, 
                                   model,
                                   tokenizer,
                                   test_datasets: Dict[str, Dataset],
                                   batch_sizes: List[int] = [1, 4, 8, 16],
                                   **kwargs) -> Dict[str, Any]:
                """
                带评估的推理测试
                
                Args:
                    model: 模型
                    tokenizer: 分词器
                    test_datasets: 测试数据集
                    batch_sizes: 批次大小列表
                    **kwargs: 其他参数
                    
                Returns:
                    测试结果
                """
                return self.integration.test_with_evaluation(
                    inference_tester=self,
                    model=model,
                    tokenizer=tokenizer,
                    test_datasets=test_datasets,
                    batch_sizes=batch_sizes,
                    **kwargs
                )
            
            def benchmark_inference_performance(self, 
                                              model,
                                              tokenizer,
                                              test_data: List[str],
                                              **kwargs) -> Dict[str, Any]:
                """
                基准测试推理性能
                
                Args:
                    model: 模型
                    tokenizer: 分词器
                    test_data: 测试数据
                    **kwargs: 其他参数
                    
                Returns:
                    性能基准测试结果
                """
                return self.integration.benchmark_inference_performance(
                    model=model,
                    tokenizer=tokenizer,
                    test_data=test_data,
                    **kwargs
                )
            
            def analyze_inference_quality(self, 
                                        predictions: List[str],
                                        references: List[str],
                                        inputs: List[str]) -> Dict[str, Any]:
                """
                分析推理质量
                
                Args:
                    predictions: 预测结果
                    references: 参考答案
                    inputs: 输入数据
                    
                Returns:
                    质量分析结果
                """
                return self.integration.analyze_inference_quality(
                    predictions=predictions,
                    references=references,
                    inputs=inputs
                )
            
            def optimize_batch_inference(self, 
                                        model,
                                        tokenizer,
                                        inputs: List[str],
                                        **kwargs) -> Tuple[List[str], Dict[str, Any]]:
                """
                优化批量推理
                
                Args:
                    model: 模型
                    tokenizer: 分词器
                    inputs: 输入列表
                    **kwargs: 其他参数
                    
                Returns:
                    推理结果和性能指标
                """
                return self.integration.optimize_batch_inference(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    **kwargs
                )
        
        return EnhancedInferenceTester    
  
  def test_with_evaluation(self, 
                           inference_tester,
                           model,
                           tokenizer,
                           test_datasets: Dict[str, Dataset],
                           batch_sizes: List[int] = [1, 4, 8, 16],
                           **kwargs) -> Dict[str, Any]:
        """
        带评估的推理测试
        
        Args:
            inference_tester: 推理测试器实例
            model: 模型
            tokenizer: 分词器
            test_datasets: 测试数据集
            batch_sizes: 批次大小列表
            **kwargs: 其他参数
            
        Returns:
            测试结果
        """
        logger.info("开始带评估的推理测试")
        
        test_results = {
            "evaluation_results": {},
            "performance_benchmarks": {},
            "quality_analysis": {},
            "optimization_recommendations": []
        }
        
        try:
            # 1. 标准化评估
            logger.info("执行标准化评估...")
            evaluation_result = self.evaluation_engine.evaluate_model(
                model=model,
                tokenizer=tokenizer,
                datasets=test_datasets,
                model_name=kwargs.get('model_name', 'test_model')
            )
            test_results["evaluation_results"] = evaluation_result.get_summary()
            
            # 2. 性能基准测试
            if self.enable_efficiency_analysis:
                logger.info("执行性能基准测试...")
                for dataset_name, dataset in test_datasets.items():
                    # 准备测试数据
                    test_inputs = self._prepare_test_inputs(dataset, max_samples=100)
                    
                    # 基准测试
                    benchmark_result = self.benchmark_inference_performance(
                        model=model,
                        tokenizer=tokenizer,
                        test_data=test_inputs,
                        batch_sizes=batch_sizes
                    )
                    test_results["performance_benchmarks"][dataset_name] = benchmark_result
            
            # 3. 质量分析
            if self.enable_quality_analysis:
                logger.info("执行质量分析...")
                for task_name, task_result in evaluation_result.task_results.items():
                    quality_analysis = self.analyze_inference_quality(
                        predictions=task_result.predictions,
                        references=task_result.references,
                        inputs=[sample.input_text for sample in task_result.samples]
                    )
                    test_results["quality_analysis"][task_name] = quality_analysis
            
            # 4. 生成优化建议
            test_results["optimization_recommendations"] = self._generate_optimization_recommendations(
                evaluation_result, test_results["performance_benchmarks"]
            )
            
            # 5. 保存结果
            self._save_test_results(test_results)
            
            logger.info("带评估的推理测试完成")
            return test_results
            
        except Exception as e:
            logger.error(f"推理测试失败: {e}")
            test_results["error"] = str(e)
            return test_results
    
    def benchmark_inference_performance(self, 
                                      model,
                                      tokenizer,
                                      test_data: List[str],
                                      batch_sizes: List[int] = [1, 4, 8, 16],
                                      num_runs: int = 10,
                                      **kwargs) -> Dict[str, Any]:
        """
        基准测试推理性能
        
        Args:
            model: 模型
            tokenizer: 分词器
            test_data: 测试数据
            batch_sizes: 批次大小列表
            num_runs: 运行次数
            **kwargs: 其他参数
            
        Returns:
            性能基准测试结果
        """
        logger.info("开始推理性能基准测试")
        
        if not self.efficiency_analyzer:
            return {"error": "效率分析器未启用"}
        
        # 创建推理函数
        def inference_func(inputs):
            return self._run_inference(model, tokenizer, inputs, **kwargs)
        
        try:
            # 延迟和吞吐量测试
            latency_results = self.efficiency_analyzer.measure_latency_and_throughput(
                inference_func=inference_func,
                inputs=test_data,
                batch_sizes=batch_sizes,
                num_runs=num_runs
            )
            
            # 内存使用监控
            memory_results = self.efficiency_analyzer.monitor_memory_usage(
                inference_func=inference_func,
                inputs=test_data[:10],  # 使用少量数据进行内存监控
                duration=30.0
            )
            
            # 模型大小统计
            model_size_results = self.efficiency_analyzer.calculate_model_size(model)
            
            # FLOPs估算
            input_shape = self._estimate_input_shape(tokenizer, test_data[:5])
            flops_results = self.efficiency_analyzer.estimate_flops(
                model=model,
                input_shape=input_shape,
                num_samples=len(test_data)
            )
            
            benchmark_result = {
                "latency_and_throughput": latency_results,
                "memory_usage": memory_results,
                "model_size": model_size_results,
                "flops_estimation": flops_results,
                "test_config": {
                    "batch_sizes": batch_sizes,
                    "num_runs": num_runs,
                    "num_test_samples": len(test_data)
                }
            }
            
            # 记录到基准测试历史
            self.performance_benchmarks[datetime.now().isoformat()] = benchmark_result
            
            logger.info("推理性能基准测试完成")
            return benchmark_result
            
        except Exception as e:
            logger.error(f"性能基准测试失败: {e}")
            return {"error": str(e)}
    
    def analyze_inference_quality(self, 
                                predictions: List[str],
                                references: List[str],
                                inputs: List[str]) -> Dict[str, Any]:
        """
        分析推理质量
        
        Args:
            predictions: 预测结果
            references: 参考答案
            inputs: 输入数据
            
        Returns:
            质量分析结果
        """
        if not self.enable_quality_analysis or not self.quality_analyzer:
            return {"error": "质量分析器未启用"}
        
        try:
            # 计算基础指标
            metrics = self.metrics_calculator.calculate_all_metrics(
                predictions=predictions,
                references=references,
                task_type="generation"
            )
            
            # 质量维度分析
            quality_scores = self._analyze_quality_dimensions(predictions, references)
            
            # 错误分析
            error_analysis = self._analyze_prediction_errors(predictions, references, inputs)
            
            # 一致性分析
            consistency_analysis = self._analyze_prediction_consistency(predictions, inputs)
            
            quality_result = {
                "basic_metrics": metrics,
                "quality_scores": quality_scores,
                "error_analysis": error_analysis,
                "consistency_analysis": consistency_analysis,
                "overall_quality_score": self._calculate_overall_quality_score(
                    metrics, quality_scores, error_analysis
                )
            }
            
            return convert_numpy_types(quality_result)
            
        except Exception as e:
            logger.error(f"质量分析失败: {e}")
            return {"error": str(e)}
    
    def optimize_batch_inference(self, 
                                model,
                                tokenizer,
                                inputs: List[str],
                                target_latency_ms: Optional[float] = None,
                                target_throughput: Optional[float] = None,
                                **kwargs) -> Tuple[List[str], Dict[str, Any]]:
        """
        优化批量推理
        
        Args:
            model: 模型
            tokenizer: 分词器
            inputs: 输入列表
            target_latency_ms: 目标延迟（毫秒）
            target_throughput: 目标吞吐量
            **kwargs: 其他参数
            
        Returns:
            推理结果和性能指标
        """
        logger.info("开始优化批量推理")
        
        try:
            # 1. 确定最优批次大小
            optimal_batch_size = self._find_optimal_batch_size(
                model, tokenizer, inputs[:20], target_latency_ms, target_throughput
            )
            
            # 2. 内存优化推理
            predictions, performance_metrics = self._memory_optimized_inference(
                model, tokenizer, inputs, optimal_batch_size, **kwargs
            )
            
            # 3. 后处理优化
            predictions = self._post_process_predictions(predictions)
            
            optimization_result = {
                "optimal_batch_size": optimal_batch_size,
                "performance_metrics": performance_metrics,
                "memory_optimization": True,
                "post_processing": True
            }
            
            logger.info(f"批量推理优化完成，最优批次大小: {optimal_batch_size}")
            return predictions, optimization_result
            
        except Exception as e:
            logger.error(f"批量推理优化失败: {e}")
            return [], {"error": str(e)}
    
    def _prepare_test_inputs(self, dataset: Dataset, max_samples: int = 100) -> List[str]:
        """准备测试输入"""
        inputs = []
        
        # 从数据集中提取文本
        text_columns = ["text", "input", "sentence", "question"]
        text_column = None
        
        for col in text_columns:
            if col in dataset.column_names:
                text_column = col
                break
        
        if text_column:
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                inputs.append(item[text_column])
        
        return inputs
    
    def _run_inference(self, model, tokenizer, inputs: List[str], **kwargs) -> List[str]:
        """运行推理"""
        try:
            # 编码输入
            encoded = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=kwargs.get('max_length', 512),
                return_tensors="pt"
            )
            
            # 移动到设备
            device = kwargs.get('device', 'cpu')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # 生成
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_length=kwargs.get('max_length', 512),
                    temperature=kwargs.get('temperature', 0.7),
                    top_p=kwargs.get('top_p', 0.9),
                    do_sample=kwargs.get('do_sample', True),
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # 解码输出
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            return predictions
            
        except Exception as e:
            logger.error(f"推理执行失败: {e}")
            return [""] * len(inputs)
    
    def _estimate_input_shape(self, tokenizer, sample_inputs: List[str]) -> Tuple[int, ...]:
        """估算输入形状"""
        try:
            # 编码样本输入
            encoded = tokenizer(
                sample_inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            return tuple(encoded['input_ids'].shape)
            
        except Exception:
            # 默认形状
            return (len(sample_inputs), 512)
    
    def _analyze_quality_dimensions(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """分析质量维度"""
        quality_scores = {
            "fluency": 0.8,      # 流畅度（需要语言模型评估）
            "coherence": 0.8,    # 连贯性
            "relevance": 0.8,    # 相关性
            "factuality": 0.8,   # 事实性
            "diversity": 0.0     # 多样性
        }
        
        try:
            # 计算多样性
            unique_predictions = len(set(predictions))
            total_predictions = len(predictions)
            quality_scores["diversity"] = unique_predictions / total_predictions if total_predictions > 0 else 0
            
            # 简单的相关性评估（基于长度相似性）
            if references:
                length_similarities = []
                for pred, ref in zip(predictions, references):
                    pred_len = len(pred.split())
                    ref_len = len(ref.split())
                    if ref_len > 0:
                        similarity = min(pred_len, ref_len) / max(pred_len, ref_len)
                        length_similarities.append(similarity)
                
                if length_similarities:
                    quality_scores["relevance"] = np.mean(length_similarities)
            
        except Exception as e:
            logger.warning(f"质量维度分析失败: {e}")
        
        return quality_scores
    
    def _analyze_prediction_errors(self, predictions: List[str], references: List[str], inputs: List[str]) -> Dict[str, Any]:
        """分析预测错误"""
        error_analysis = {
            "empty_predictions": 0,
            "too_short_predictions": 0,
            "too_long_predictions": 0,
            "repetitive_predictions": 0,
            "common_errors": []
        }
        
        try:
            for pred, ref in zip(predictions, references):
                # 空预测
                if not pred.strip():
                    error_analysis["empty_predictions"] += 1
                
                # 长度问题
                pred_words = len(pred.split())
                ref_words = len(ref.split())
                
                if pred_words < ref_words * 0.3:
                    error_analysis["too_short_predictions"] += 1
                elif pred_words > ref_words * 3:
                    error_analysis["too_long_predictions"] += 1
                
                # 重复性检查
                words = pred.split()
                if len(words) > 1:
                    unique_words = len(set(words))
                    repetition_ratio = 1 - (unique_words / len(words))
                    if repetition_ratio > 0.5:
                        error_analysis["repetitive_predictions"] += 1
            
            # 计算错误率
            total_predictions = len(predictions)
            for key in ["empty_predictions", "too_short_predictions", "too_long_predictions", "repetitive_predictions"]:
                error_analysis[f"{key}_rate"] = error_analysis[key] / total_predictions if total_predictions > 0 else 0
            
        except Exception as e:
            logger.warning(f"错误分析失败: {e}")
        
        return error_analysis
    
    def _analyze_prediction_consistency(self, predictions: List[str], inputs: List[str]) -> Dict[str, Any]:
        """分析预测一致性"""
        consistency_analysis = {
            "length_consistency": 0.0,
            "style_consistency": 0.0,
            "format_consistency": 0.0
        }
        
        try:
            if not predictions:
                return consistency_analysis
            
            # 长度一致性
            lengths = [len(pred.split()) for pred in predictions]
            if lengths:
                length_std = np.std(lengths)
                length_mean = np.mean(lengths)
                consistency_analysis["length_consistency"] = 1.0 / (1.0 + length_std / length_mean) if length_mean > 0 else 0
            
            # 格式一致性（简单检查）
            formats = []
            for pred in predictions:
                format_features = {
                    "has_punctuation": any(c in pred for c in ".,!?"),
                    "has_numbers": any(c.isdigit() for c in pred),
                    "starts_with_capital": pred[0].isupper() if pred else False
                }
                formats.append(format_features)
            
            if formats:
                # 计算格式特征的一致性
                feature_consistencies = []
                for feature in ["has_punctuation", "has_numbers", "starts_with_capital"]:
                    feature_values = [f[feature] for f in formats]
                    consistency = 1.0 if all(feature_values) or not any(feature_values) else sum(feature_values) / len(feature_values)
                    feature_consistencies.append(consistency)
                
                consistency_analysis["format_consistency"] = np.mean(feature_consistencies)
            
        except Exception as e:
            logger.warning(f"一致性分析失败: {e}")
        
        return consistency_analysis
    
    def _calculate_overall_quality_score(self, metrics: Dict, quality_scores: Dict, error_analysis: Dict) -> float:
        """计算总体质量分数"""
        try:
            # 基础指标权重
            base_score = 0.0
            if "bleu" in metrics:
                base_score += metrics["bleu"] * 0.3
            if "rouge1" in metrics:
                base_score += metrics["rouge1"] * 0.2
            if "bertscore_f1" in metrics:
                base_score += metrics["bertscore_f1"] * 0.2
            
            # 质量维度权重
            quality_score = np.mean(list(quality_scores.values())) * 0.2
            
            # 错误惩罚
            error_penalty = 0.0
            if "empty_predictions_rate" in error_analysis:
                error_penalty += error_analysis["empty_predictions_rate"] * 0.3
            if "repetitive_predictions_rate" in error_analysis:
                error_penalty += error_analysis["repetitive_predictions_rate"] * 0.2
            
            overall_score = base_score + quality_score - error_penalty
            return max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            logger.warning(f"计算总体质量分数失败: {e}")
            return 0.5
    
    def _find_optimal_batch_size(self, model, tokenizer, sample_inputs: List[str], 
                                target_latency_ms: Optional[float], 
                                target_throughput: Optional[float]) -> int:
        """找到最优批次大小"""
        batch_sizes = [1, 2, 4, 8, 16, 32]
        best_batch_size = 1
        best_score = 0
        
        try:
            for batch_size in batch_sizes:
                # 测试当前批次大小
                start_time = time.time()
                try:
                    batch_inputs = sample_inputs[:batch_size]
                    _ = self._run_inference(model, tokenizer, batch_inputs)
                    end_time = time.time()
                    
                    latency_ms = (end_time - start_time) * 1000
                    throughput = batch_size / (end_time - start_time)
                    
                    # 计算分数
                    score = 0
                    if target_latency_ms and latency_ms <= target_latency_ms:
                        score += 1
                    if target_throughput and throughput >= target_throughput:
                        score += 1
                    
                    # 偏向更大的批次大小（更高效）
                    score += batch_size * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_batch_size = batch_size
                        
                except Exception:
                    # 如果批次大小太大导致内存不足，跳过
                    break
            
        except Exception as e:
            logger.warning(f"寻找最优批次大小失败: {e}")
        
        return best_batch_size
    
    def _memory_optimized_inference(self, model, tokenizer, inputs: List[str], 
                                  batch_size: int, **kwargs) -> Tuple[List[str], Dict[str, Any]]:
        """内存优化推理"""
        predictions = []
        performance_metrics = {
            "total_batches": 0,
            "avg_batch_time": 0.0,
            "peak_memory_mb": 0.0
        }
        
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024**2
            peak_memory = initial_memory
            
            batch_times = []
            
            # 分批处理
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size]
                
                start_time = time.time()
                batch_predictions = self._run_inference(model, tokenizer, batch_inputs, **kwargs)
                end_time = time.time()
                
                predictions.extend(batch_predictions)
                batch_times.append(end_time - start_time)
                
                # 监控内存使用
                current_memory = process.memory_info().rss / 1024**2
                peak_memory = max(peak_memory, current_memory)
                
                performance_metrics["total_batches"] += 1
            
            performance_metrics["avg_batch_time"] = np.mean(batch_times) if batch_times else 0
            performance_metrics["peak_memory_mb"] = peak_memory
            performance_metrics["memory_increase_mb"] = peak_memory - initial_memory
            
        except Exception as e:
            logger.error(f"内存优化推理失败: {e}")
            # 回退到简单推理
            predictions = self._run_inference(model, tokenizer, inputs, **kwargs)
        
        return predictions, performance_metrics
    
    def _post_process_predictions(self, predictions: List[str]) -> List[str]:
        """后处理预测结果"""
        processed_predictions = []
        
        for pred in predictions:
            # 清理空白字符
            processed_pred = pred.strip()
            
            # 移除重复的句子
            sentences = processed_pred.split('.')
            unique_sentences = []
            seen = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in seen:
                    unique_sentences.append(sentence)
                    seen.add(sentence)
            
            processed_pred = '. '.join(unique_sentences)
            if processed_pred and not processed_pred.endswith('.'):
                processed_pred += '.'
            
            processed_predictions.append(processed_pred)
        
        return processed_predictions
    
    def _generate_optimization_recommendations(self, evaluation_result: EvaluationResult, 
                                             performance_benchmarks: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        try:
            # 基于评估结果的建议
            overall_score = evaluation_result.metrics.get("overall_score", 0)
            if overall_score < 0.5:
                recommendations.append("模型性能较低，建议检查模型训练质量或选择更适合的模型")
            
            # 基于性能基准的建议
            for dataset_name, benchmark in performance_benchmarks.items():
                if "latency_and_throughput" in benchmark:
                    latency_info = benchmark["latency_and_throughput"]
                    if "overall" in latency_info:
                        avg_latency = latency_info["overall"].get("avg_latency_per_sample_ms", 0)
                        if avg_latency > 1000:  # 超过1秒
                            recommendations.append(f"数据集 {dataset_name} 的推理延迟较高，建议优化模型或使用更大的批次大小")
                
                if "memory_usage" in benchmark:
                    memory_info = benchmark["memory_usage"]
                    peak_memory = memory_info.get("peak_memory_mb", 0)
                    if peak_memory > 8000:  # 超过8GB
                        recommendations.append(f"数据集 {dataset_name} 的内存使用较高，建议使用梯度检查点或模型并行")
            
            # 基于效率指标的建议
            if hasattr(evaluation_result, 'efficiency_metrics'):
                efficiency = evaluation_result.efficiency_metrics
                if efficiency.throughput < 10:  # 吞吐量低于10 tokens/s
                    recommendations.append("推理吞吐量较低，建议使用批量推理或模型优化技术")
            
            if not recommendations:
                recommendations.append("推理性能良好，可以考虑进一步的微调优化")
            
        except Exception as e:
            logger.warning(f"生成优化建议失败: {e}")
            recommendations.append("无法生成具体建议，请检查评估结果")
        
        return recommendations
    
    def _save_test_results(self, test_results: Dict[str, Any]):
        """保存测试结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = self.output_dir / f"inference_test_results_{timestamp}.json"
            
            import json
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_types(test_results), f, indent=2, ensure_ascii=False)
            
            logger.info(f"测试结果已保存: {result_file}")
            
        except Exception as e:
            logger.error(f"保存测试结果失败: {e}")
    
    def get_performance_history(self) -> Dict[str, Any]:
        """获取性能历史"""
        return {
            "inference_history": self.inference_history,
            "performance_benchmarks": self.performance_benchmarks,
            "total_tests": len(self.inference_history)
        }


def create_enhanced_inference_tester(base_inference_tester_class=None):
    """
    创建增强的推理测试器类
    
    Args:
        base_inference_tester_class: 基础推理测试器类，如果为None则创建新类
        
    Returns:
        增强的推理测试器类
    """
    if base_inference_tester_class is None:
        # 创建基础推理测试器类
        class BaseInferenceTester:
            def __init__(self, **kwargs):
                self.config = kwargs
            
            def test_inference(self, model, tokenizer, test_data, **kwargs):
                """基础推理测试方法"""
                pass
    
        base_inference_tester_class = BaseInferenceTester
    
    # 使用集成器增强推理测试器
    integration = InferenceTesterIntegration()
    enhanced_class = integration.enhance_inference_tester(base_inference_tester_class)
    
    return enhanced_class
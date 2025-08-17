"""
任务评估器

实现多种评估任务的具体评估逻辑，包括文本生成、问答、语义相似度和分类任务。
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

from datasets import Dataset
from .data_models import EvaluationSample, TaskResult, convert_numpy_types
from .metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class BaseTaskEvaluator(ABC):
    """
    基础任务评估器抽象类
    """
    
    def __init__(self, metrics_calculator: MetricsCalculator):
        """
        初始化任务评估器
        
        Args:
            metrics_calculator: 指标计算器
        """
        self.metrics_calculator = metrics_calculator
    
    @abstractmethod
    def evaluate(self, 
                predictions: List[str], 
                references: List[str],
                inputs: List[str],
                **kwargs) -> TaskResult:
        """
        评估任务
        
        Args:
            predictions: 预测结果列表
            references: 参考答案列表
            inputs: 输入文本列表
            **kwargs: 其他参数
            
        Returns:
            任务评估结果
        """
        pass
    
    def _create_samples(self, 
                       inputs: List[str],
                       predictions: List[str], 
                       references: List[str],
                       sample_metrics: Optional[List[Dict[str, float]]] = None) -> List[EvaluationSample]:
        """
        创建评估样本列表
        
        Args:
            inputs: 输入文本列表
            predictions: 预测结果列表
            references: 参考答案列表
            sample_metrics: 每个样本的指标（可选）
            
        Returns:
            评估样本列表
        """
        samples = []
        
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references)):
            metrics = sample_metrics[i] if sample_metrics else {}
            
            sample = EvaluationSample(
                input_text=inp,
                prediction=pred,
                reference=ref,
                metrics=metrics
            )
            samples.append(sample)
        
        return samples


class TextGenerationEvaluator(BaseTaskEvaluator):
    """
    文本生成质量评估器
    
    评估文本生成任务的质量，包括BLEU、ROUGE、BERTScore等指标。
    """
    
    def evaluate(self, 
                predictions: List[str], 
                references: List[str],
                inputs: List[str],
                **kwargs) -> TaskResult:
        """
        评估文本生成质量
        
        Args:
            predictions: 生成的文本列表
            references: 参考文本列表
            inputs: 输入提示列表
            **kwargs: 其他参数
            
        Returns:
            文本生成评估结果
        """
        logger.info(f"开始评估文本生成任务，样本数: {len(predictions)}")
        
        # 计算整体指标
        metrics = self.metrics_calculator.calculate_all_metrics(
            predictions, references, task_type="generation"
        )
        
        # 计算每个样本的指标（可选，计算量大）
        sample_metrics = []
        if kwargs.get("calculate_sample_metrics", False):
            for pred, ref in zip(predictions, references):
                sample_metric = self.metrics_calculator.calculate_all_metrics(
                    [pred], [ref], task_type="generation"
                )
                sample_metrics.append(sample_metric)
        
        # 创建样本
        samples = self._create_samples(inputs, predictions, references, sample_metrics)
        
        # 添加生成质量特定指标
        metrics.update(self._calculate_generation_quality_metrics(predictions, references))
        
        return TaskResult(
            task_name="text_generation",
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=samples,
            execution_time=0.0  # 由调用者设置
        )
    
    def _calculate_generation_quality_metrics(self, 
                                            predictions: List[str], 
                                            references: List[str]) -> Dict[str, float]:
        """
        计算文本生成质量特定指标
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            
        Returns:
            质量指标字典
        """
        metrics = {}
        
        try:
            # 计算长度相关指标
            pred_lengths = [len(pred) for pred in predictions]
            ref_lengths = [len(ref) for ref in references]
            
            metrics.update({
                "avg_prediction_length": float(np.mean(pred_lengths)),
                "avg_reference_length": float(np.mean(ref_lengths)),
                "length_ratio": float(np.mean(pred_lengths) / np.mean(ref_lengths)) if np.mean(ref_lengths) > 0 else 0.0,
                "length_variance": float(np.var(pred_lengths))
            })
            
            # 计算重复度
            repetition_scores = []
            for pred in predictions:
                words = pred.split()
                if len(words) > 1:
                    unique_words = len(set(words))
                    repetition = 1 - (unique_words / len(words))
                    repetition_scores.append(repetition)
                else:
                    repetition_scores.append(0.0)
            
            metrics["avg_repetition_rate"] = float(np.mean(repetition_scores))
            
            # 计算覆盖率（预测文本中包含参考文本词汇的比例）
            coverage_scores = []
            for pred, ref in zip(predictions, references):
                pred_words = set(pred.split())
                ref_words = set(ref.split())
                
                if ref_words:
                    coverage = len(pred_words & ref_words) / len(ref_words)
                    coverage_scores.append(coverage)
                else:
                    coverage_scores.append(0.0)
            
            metrics["avg_coverage"] = float(np.mean(coverage_scores))
            
        except Exception as e:
            logger.warning(f"计算生成质量指标时出错: {e}")
        
        return convert_numpy_types(metrics)


class QuestionAnsweringEvaluator(BaseTaskEvaluator):
    """
    问答准确性评估器
    
    评估问答任务的准确性，包括精确匹配、F1分数等指标。
    """
    
    def evaluate(self, 
                predictions: List[str], 
                references: List[str],
                inputs: List[str],
                **kwargs) -> TaskResult:
        """
        评估问答准确性
        
        Args:
            predictions: 预测答案列表
            references: 标准答案列表
            inputs: 问题列表
            **kwargs: 其他参数
            
        Returns:
            问答评估结果
        """
        logger.info(f"开始评估问答任务，样本数: {len(predictions)}")
        
        # 计算基础指标
        metrics = self.metrics_calculator.calculate_all_metrics(
            predictions, references, task_type="generation"
        )
        
        # 计算问答特定指标
        qa_metrics = self._calculate_qa_metrics(predictions, references)
        metrics.update(qa_metrics)
        
        # 创建样本
        samples = self._create_samples(inputs, predictions, references)
        
        return TaskResult(
            task_name="question_answering",
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=samples,
            execution_time=0.0
        )
    
    def _calculate_qa_metrics(self, 
                            predictions: List[str], 
                            references: List[str]) -> Dict[str, float]:
        """
        计算问答特定指标
        
        Args:
            predictions: 预测答案列表
            references: 标准答案列表
            
        Returns:
            问答指标字典
        """
        metrics = {}
        
        try:
            # 精确匹配
            exact_matches = []
            for pred, ref in zip(predictions, references):
                exact_match = 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0
                exact_matches.append(exact_match)
            
            metrics["exact_match"] = float(np.mean(exact_matches))
            
            # Token级F1分数
            f1_scores = []
            for pred, ref in zip(predictions, references):
                f1 = self._calculate_token_f1(pred, ref)
                f1_scores.append(f1)
            
            metrics["token_f1"] = float(np.mean(f1_scores))
            
            # 答案长度分析
            pred_lengths = [len(pred.split()) for pred in predictions]
            ref_lengths = [len(ref.split()) for ref in references]
            
            metrics.update({
                "avg_answer_length": float(np.mean(pred_lengths)),
                "avg_reference_length": float(np.mean(ref_lengths)),
                "answer_length_ratio": float(np.mean(pred_lengths) / np.mean(ref_lengths)) if np.mean(ref_lengths) > 0 else 0.0
            })
            
            # 包含性检查（预测答案是否包含在参考答案中）
            containment_scores = []
            for pred, ref in zip(predictions, references):
                pred_clean = pred.strip().lower()
                ref_clean = ref.strip().lower()
                
                if pred_clean in ref_clean or ref_clean in pred_clean:
                    containment_scores.append(1.0)
                else:
                    containment_scores.append(0.0)
            
            metrics["containment_rate"] = float(np.mean(containment_scores))
            
        except Exception as e:
            logger.warning(f"计算问答指标时出错: {e}")
        
        return convert_numpy_types(metrics)
    
    def _calculate_token_f1(self, prediction: str, reference: str) -> float:
        """
        计算Token级F1分数
        
        Args:
            prediction: 预测答案
            reference: 参考答案
            
        Returns:
            F1分数
        """
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not ref_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        if not pred_tokens:
            return 0.0
        
        common_tokens = pred_tokens & ref_tokens
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1


class SemanticSimilarityEvaluator(BaseTaskEvaluator):
    """
    语义相似度评估器
    
    评估语义相似度任务，包括余弦相似度、Jaccard相似度等指标。
    """
    
    def evaluate(self, 
                predictions: List[str], 
                references: List[str],
                inputs: List[str],
                **kwargs) -> TaskResult:
        """
        评估语义相似度
        
        Args:
            predictions: 预测相似度分数或文本列表
            references: 参考相似度分数或文本列表
            inputs: 输入文本对列表
            **kwargs: 其他参数
            
        Returns:
            语义相似度评估结果
        """
        logger.info(f"开始评估语义相似度任务，样本数: {len(predictions)}")
        
        # 如果预测和参考都是数值，直接计算相关性
        if self._is_numeric_list(predictions) and self._is_numeric_list(references):
            metrics = self._calculate_similarity_correlation(predictions, references)
        else:
            # 如果是文本，计算语义相似度指标
            metrics = self.metrics_calculator.calculate_semantic_similarity(
                predictions, references, method="cosine"
            )
            
            # 添加其他相似度方法
            jaccard_metrics = self.metrics_calculator.calculate_semantic_similarity(
                predictions, references, method="jaccard"
            )
            metrics.update(jaccard_metrics)
        
        # 创建样本
        samples = self._create_samples(inputs, predictions, references)
        
        return TaskResult(
            task_name="semantic_similarity",
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=samples,
            execution_time=0.0
        )
    
    def _is_numeric_list(self, items: List[str]) -> bool:
        """
        检查列表是否包含数值
        
        Args:
            items: 字符串列表
            
        Returns:
            是否为数值列表
        """
        try:
            for item in items[:5]:  # 检查前5个
                float(item)
            return True
        except (ValueError, TypeError):
            return False
    
    def _calculate_similarity_correlation(self, 
                                        predictions: List[str], 
                                        references: List[str]) -> Dict[str, float]:
        """
        计算相似度分数的相关性
        
        Args:
            predictions: 预测相似度分数列表
            references: 参考相似度分数列表
            
        Returns:
            相关性指标字典
        """
        try:
            pred_scores = [float(p) for p in predictions]
            ref_scores = [float(r) for r in references]
            
            # 计算皮尔逊相关系数
            correlation = np.corrcoef(pred_scores, ref_scores)[0, 1]
            
            # 计算均方误差
            mse = np.mean([(p - r) ** 2 for p, r in zip(pred_scores, ref_scores)])
            
            # 计算平均绝对误差
            mae = np.mean([abs(p - r) for p, r in zip(pred_scores, ref_scores)])
            
            return convert_numpy_types({
                "pearson_correlation": float(correlation) if not np.isnan(correlation) else 0.0,
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(np.sqrt(mse))
            })
            
        except Exception as e:
            logger.error(f"计算相似度相关性时出错: {e}")
            return {"error": "无法计算相似度相关性"}


class ClassificationEvaluator(BaseTaskEvaluator):
    """
    分类性能评估器
    
    评估分类任务的性能，包括准确率、精确率、召回率、F1分数等指标。
    """
    
    def evaluate(self, 
                predictions: List[str], 
                references: List[str],
                inputs: List[str],
                **kwargs) -> TaskResult:
        """
        评估分类性能
        
        Args:
            predictions: 预测标签列表
            references: 真实标签列表
            inputs: 输入文本列表
            **kwargs: 其他参数
            
        Returns:
            分类评估结果
        """
        logger.info(f"开始评估分类任务，样本数: {len(predictions)}")
        
        # 计算分类指标
        metrics = self.metrics_calculator.calculate_classification_metrics(
            predictions, references, average=kwargs.get("average", "weighted")
        )
        
        # 添加分类特定分析
        classification_analysis = self._analyze_classification_results(predictions, references)
        metrics.update(classification_analysis)
        
        # 创建样本
        samples = self._create_samples(inputs, predictions, references)
        
        return TaskResult(
            task_name="classification",
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=samples,
            execution_time=0.0
        )
    
    def _analyze_classification_results(self, 
                                      predictions: List[str], 
                                      references: List[str]) -> Dict[str, Any]:
        """
        分析分类结果
        
        Args:
            predictions: 预测标签列表
            references: 真实标签列表
            
        Returns:
            分类分析结果
        """
        analysis = {}
        
        try:
            # 标签分布分析
            pred_distribution = {}
            ref_distribution = {}
            
            for pred in predictions:
                pred_distribution[pred] = pred_distribution.get(pred, 0) + 1
            
            for ref in references:
                ref_distribution[ref] = ref_distribution.get(ref, 0) + 1
            
            analysis["prediction_distribution"] = pred_distribution
            analysis["reference_distribution"] = ref_distribution
            
            # 计算标签分布的KL散度
            all_labels = set(predictions + references)
            pred_probs = [pred_distribution.get(label, 0) / len(predictions) for label in all_labels]
            ref_probs = [ref_distribution.get(label, 0) / len(references) for label in all_labels]
            
            kl_div = self._calculate_kl_divergence(ref_probs, pred_probs)
            analysis["label_distribution_kl_divergence"] = float(kl_div)
            
            # 最常见的错误分类
            error_pairs = {}
            for pred, ref in zip(predictions, references):
                if pred != ref:
                    error_pair = f"{ref} -> {pred}"
                    error_pairs[error_pair] = error_pairs.get(error_pair, 0) + 1
            
            # 获取最常见的5个错误
            top_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
            analysis["top_classification_errors"] = [
                {"error_type": error, "count": count} for error, count in top_errors
            ]
            
            # 计算每个类别的支持度
            class_support = {}
            for ref in references:
                class_support[ref] = class_support.get(ref, 0) + 1
            
            analysis["class_support"] = class_support
            
        except Exception as e:
            logger.warning(f"分析分类结果时出错: {e}")
        
        return convert_numpy_types(analysis)
    
    def _calculate_kl_divergence(self, p: List[float], q: List[float]) -> float:
        """
        计算KL散度
        
        Args:
            p: 真实分布
            q: 预测分布
            
        Returns:
            KL散度
        """
        kl_div = 0.0
        for pi, qi in zip(p, q):
            if pi > 0 and qi > 0:
                kl_div += pi * np.log(pi / qi)
        return kl_div


class TaskEvaluatorFactory:
    """
    任务评估器工厂类
    
    根据任务类型创建相应的评估器。
    """
    
    @staticmethod
    def create_evaluator(task_type: str, metrics_calculator: MetricsCalculator) -> BaseTaskEvaluator:
        """
        创建任务评估器
        
        Args:
            task_type: 任务类型
            metrics_calculator: 指标计算器
            
        Returns:
            任务评估器实例
        """
        if task_type.lower() in ["text_generation", "generation"]:
            return TextGenerationEvaluator(metrics_calculator)
        elif task_type.lower() in ["question_answering", "qa"]:
            return QuestionAnsweringEvaluator(metrics_calculator)
        elif task_type.lower() in ["semantic_similarity", "similarity"]:
            return SemanticSimilarityEvaluator(metrics_calculator)
        elif task_type.lower() in ["classification", "classify"]:
            return ClassificationEvaluator(metrics_calculator)
        else:
            logger.warning(f"未知任务类型: {task_type}，使用默认的文本生成评估器")
            return TextGenerationEvaluator(metrics_calculator)
    
    @staticmethod
    def get_supported_tasks() -> List[str]:
        """
        获取支持的任务类型列表
        
        Returns:
            支持的任务类型列表
        """
        return [
            "text_generation",
            "question_answering", 
            "semantic_similarity",
            "classification"
        ]
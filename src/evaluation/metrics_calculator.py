"""
指标计算器

实现标准化的评估指标计算，包括文本生成指标、分类指标和语义相似度指标。
"""

import logging
import re
import math
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter
import numpy as np

# 导入第三方库
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, some metrics will be disabled")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge-score not available, ROUGE metrics will be disabled")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning("bert-score not available, BERTScore will be disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, some metrics will be disabled")

from .data_models import convert_numpy_types

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    指标计算器
    
    提供各种评估指标的计算功能，包括：
    - 文本生成指标：BLEU、ROUGE、METEOR、BERTScore
    - 分类指标：准确率、精确率、召回率、F1分数
    - 语义相似度指标
    - 困惑度计算
    """
    
    def __init__(self, 
                 language: str = "zh",
                 device: str = "cpu",
                 cache_dir: Optional[str] = None):
        """
        初始化指标计算器
        
        Args:
            language: 语言代码，默认为中文
            device: 计算设备，cpu或cuda
            cache_dir: 缓存目录
        """
        self.language = language
        self.device = device
        self.cache_dir = cache_dir
        
        # 初始化ROUGE评分器
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        
        # 下载必要的NLTK数据
        if NLTK_AVAILABLE:
            self._download_nltk_data()
        
        logger.info(f"MetricsCalculator初始化完成，语言: {language}, 设备: {device}")
    
    def _download_nltk_data(self):
        """下载必要的NLTK数据"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
    
    def calculate_bleu(self, 
                      predictions: List[str], 
                      references: List[str],
                      n_gram: int = 4) -> Dict[str, float]:
        """
        计算BLEU分数
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            n_gram: N-gram的最大值
            
        Returns:
            包含BLEU分数的字典
        """
        if not NLTK_AVAILABLE:
            logger.warning("NLTK不可用，无法计算BLEU分数")
            return {"bleu": 0.0}
        
        if len(predictions) != len(references):
            raise ValueError("预测文本和参考文本数量不匹配")
        
        bleu_scores = []
        smoothing = SmoothingFunction().method1
        
        for pred, ref in zip(predictions, references):
            # 分词
            pred_tokens = self._tokenize(pred)
            ref_tokens = [self._tokenize(ref)]  # BLEU需要参考文本列表
            
            # 计算BLEU分数
            if len(pred_tokens) == 0 or len(ref_tokens[0]) == 0:
                bleu_scores.append(0.0)
                continue
            
            try:
                bleu = sentence_bleu(
                    ref_tokens, 
                    pred_tokens,
                    weights=tuple(1/n_gram for _ in range(n_gram)),
                    smoothing_function=smoothing
                )
                bleu_scores.append(bleu)
            except Exception as e:
                logger.warning(f"计算BLEU分数时出错: {e}")
                bleu_scores.append(0.0)
        
        result = {
            "bleu": float(np.mean(bleu_scores)),
            "bleu_std": float(np.std(bleu_scores)),
            "bleu_scores": [float(score) for score in bleu_scores]
        }
        
        return convert_numpy_types(result)
    
    def calculate_rouge(self, 
                       predictions: List[str], 
                       references: List[str]) -> Dict[str, float]:
        """
        计算ROUGE分数
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            
        Returns:
            包含ROUGE分数的字典
        """
        if not ROUGE_AVAILABLE:
            logger.warning("rouge-score不可用，无法计算ROUGE分数")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        if len(predictions) != len(references):
            raise ValueError("预测文本和参考文本数量不匹配")
        
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for pred, ref in zip(predictions, references):
            try:
                scores = self.rouge_scorer.score(ref, pred)
                rouge_scores["rouge1"].append(scores['rouge1'].fmeasure)
                rouge_scores["rouge2"].append(scores['rouge2'].fmeasure)
                rouge_scores["rougeL"].append(scores['rougeL'].fmeasure)
            except Exception as e:
                logger.warning(f"计算ROUGE分数时出错: {e}")
                rouge_scores["rouge1"].append(0.0)
                rouge_scores["rouge2"].append(0.0)
                rouge_scores["rougeL"].append(0.0)
        
        result = {}
        for metric, scores in rouge_scores.items():
            result[metric] = float(np.mean(scores))
            result[f"{metric}_std"] = float(np.std(scores))
        
        return convert_numpy_types(result)
    
    def calculate_bertscore(self, 
                           predictions: List[str], 
                           references: List[str],
                           model_type: str = "bert-base-chinese") -> Dict[str, float]:
        """
        计算BERTScore
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            model_type: BERT模型类型
            
        Returns:
            包含BERTScore的字典
        """
        if not BERTSCORE_AVAILABLE:
            logger.warning("bert-score不可用，无法计算BERTScore")
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
        
        if len(predictions) != len(references):
            raise ValueError("预测文本和参考文本数量不匹配")
        
        try:
            P, R, F1 = bert_score(
                predictions, 
                references, 
                model_type=model_type,
                device=self.device,
                verbose=False
            )
            
            result = {
                "bertscore_precision": float(P.mean()),
                "bertscore_recall": float(R.mean()),
                "bertscore_f1": float(F1.mean()),
                "bertscore_precision_std": float(P.std()),
                "bertscore_recall_std": float(R.std()),
                "bertscore_f1_std": float(F1.std())
            }
            
            return convert_numpy_types(result)
            
        except Exception as e:
            logger.error(f"计算BERTScore时出错: {e}")
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    
    def calculate_perplexity(self, 
                           texts: List[str], 
                           model, 
                           tokenizer) -> Dict[str, float]:
        """
        计算困惑度
        
        Args:
            texts: 文本列表
            model: 语言模型
            tokenizer: 分词器
            
        Returns:
            包含困惑度的字典
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch不可用，无法计算困惑度")
            return {"perplexity": float('inf')}
        
        perplexities = []
        
        for text in texts:
            try:
                # 编码文本
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 计算损失
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    perplexity = torch.exp(loss)
                    perplexities.append(float(perplexity))
                    
            except Exception as e:
                logger.warning(f"计算困惑度时出错: {e}")
                perplexities.append(float('inf'))
        
        # 过滤无穷大值
        valid_perplexities = [p for p in perplexities if not math.isinf(p)]
        
        if not valid_perplexities:
            return {"perplexity": float('inf')}
        
        result = {
            "perplexity": float(np.mean(valid_perplexities)),
            "perplexity_std": float(np.std(valid_perplexities)),
            "valid_samples": len(valid_perplexities),
            "total_samples": len(texts)
        }
        
        return convert_numpy_types(result)
    
    def calculate_classification_metrics(self, 
                                       predictions: List[Union[int, str]], 
                                       references: List[Union[int, str]],
                                       average: str = "weighted") -> Dict[str, float]:
        """
        计算分类指标
        
        Args:
            predictions: 预测标签列表
            references: 真实标签列表
            average: 平均方式，'micro', 'macro', 'weighted'
            
        Returns:
            包含分类指标的字典
        """
        if len(predictions) != len(references):
            raise ValueError("预测标签和真实标签数量不匹配")
        
        # 转换为字符串以便统一处理
        pred_labels = [str(p) for p in predictions]
        true_labels = [str(r) for r in references]
        
        # 获取所有唯一标签
        all_labels = sorted(list(set(pred_labels + true_labels)))
        
        # 计算混淆矩阵
        confusion_matrix = self._calculate_confusion_matrix(pred_labels, true_labels, all_labels)
        
        # 计算各类别的精确率、召回率、F1分数
        per_class_metrics = {}
        for label in all_labels:
            tp = confusion_matrix[label][label]
            fp = sum(confusion_matrix[other][label] for other in all_labels if other != label)
            fn = sum(confusion_matrix[label][other] for other in all_labels if other != label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(1 for t in true_labels if t == label)
            }
        
        # 计算整体指标
        accuracy = sum(1 for p, t in zip(pred_labels, true_labels) if p == t) / len(pred_labels)
        
        # 计算平均指标
        if average == "micro":
            total_tp = sum(confusion_matrix[label][label] for label in all_labels)
            total_fp = sum(sum(confusion_matrix[other][label] for other in all_labels if other != label) 
                          for label in all_labels)
            total_fn = sum(sum(confusion_matrix[label][other] for other in all_labels if other != label) 
                          for label in all_labels)
            
            precision_avg = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall_avg = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1_avg = 2 * precision_avg * recall_avg / (precision_avg + recall_avg) if (precision_avg + recall_avg) > 0 else 0.0
            
        elif average == "macro":
            precision_avg = np.mean([metrics["precision"] for metrics in per_class_metrics.values()])
            recall_avg = np.mean([metrics["recall"] for metrics in per_class_metrics.values()])
            f1_avg = np.mean([metrics["f1"] for metrics in per_class_metrics.values()])
            
        elif average == "weighted":
            total_support = sum(metrics["support"] for metrics in per_class_metrics.values())
            precision_avg = sum(metrics["precision"] * metrics["support"] for metrics in per_class_metrics.values()) / total_support
            recall_avg = sum(metrics["recall"] * metrics["support"] for metrics in per_class_metrics.values()) / total_support
            f1_avg = sum(metrics["f1"] * metrics["support"] for metrics in per_class_metrics.values()) / total_support
        
        result = {
            "accuracy": float(accuracy),
            "precision": float(precision_avg),
            "recall": float(recall_avg),
            "f1": float(f1_avg),
            "num_classes": len(all_labels),
            "total_samples": len(predictions),
            "per_class_metrics": {label: convert_numpy_types(metrics) 
                                for label, metrics in per_class_metrics.items()}
        }
        
        return convert_numpy_types(result)
    
    def _calculate_confusion_matrix(self, 
                                  predictions: List[str], 
                                  references: List[str], 
                                  labels: List[str]) -> Dict[str, Dict[str, int]]:
        """计算混淆矩阵"""
        matrix = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
        
        for pred, true in zip(predictions, references):
            matrix[true][pred] += 1
        
        return matrix
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if self.language == "zh":
            # 中文分词：简单的字符级分词
            return list(text.replace(" ", ""))
        else:
            # 英文分词
            if NLTK_AVAILABLE:
                return word_tokenize(text.lower())
            else:
                # 简单的空格分词
                return text.lower().split()
    
    def calculate_semantic_similarity(self, 
                                    text1_list: List[str], 
                                    text2_list: List[str],
                                    method: str = "cosine") -> Dict[str, float]:
        """
        计算语义相似度
        
        Args:
            text1_list: 第一组文本列表
            text2_list: 第二组文本列表
            method: 相似度计算方法
            
        Returns:
            包含相似度分数的字典
        """
        if len(text1_list) != len(text2_list):
            raise ValueError("两组文本数量不匹配")
        
        similarities = []
        
        for text1, text2 in zip(text1_list, text2_list):
            if method == "cosine":
                sim = self._cosine_similarity(text1, text2)
            elif method == "jaccard":
                sim = self._jaccard_similarity(text1, text2)
            else:
                raise ValueError(f"不支持的相似度计算方法: {method}")
            
            similarities.append(sim)
        
        result = {
            f"{method}_similarity": float(np.mean(similarities)),
            f"{method}_similarity_std": float(np.std(similarities)),
            "similarities": [float(sim) for sim in similarities]
        }
        
        return convert_numpy_types(result)
    
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """计算余弦相似度"""
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        
        # 创建词汇表
        vocab = set(tokens1 + tokens2)
        
        # 创建向量
        vec1 = [tokens1.count(token) for token in vocab]
        vec2 = [tokens2.count(token) for token in vocab]
        
        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """计算Jaccard相似度"""
        tokens1 = set(self._tokenize(text1))
        tokens2 = set(self._tokenize(text2))
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_all_metrics(self, 
                            predictions: List[str], 
                            references: List[str],
                            task_type: str = "generation",
                            **kwargs) -> Dict[str, Any]:
        """
        计算所有适用的指标
        
        Args:
            predictions: 预测结果列表
            references: 参考结果列表
            task_type: 任务类型，'generation', 'classification', 'similarity'
            **kwargs: 其他参数
            
        Returns:
            包含所有指标的字典
        """
        results = {}
        
        if task_type == "generation":
            # 文本生成任务指标
            try:
                results.update(self.calculate_bleu(predictions, references))
            except Exception as e:
                logger.warning(f"计算BLEU时出错: {e}")
            
            try:
                results.update(self.calculate_rouge(predictions, references))
            except Exception as e:
                logger.warning(f"计算ROUGE时出错: {e}")
            
            try:
                results.update(self.calculate_bertscore(predictions, references))
            except Exception as e:
                logger.warning(f"计算BERTScore时出错: {e}")
        
        elif task_type == "classification":
            # 分类任务指标
            try:
                results.update(self.calculate_classification_metrics(predictions, references))
            except Exception as e:
                logger.warning(f"计算分类指标时出错: {e}")
        
        elif task_type == "similarity":
            # 相似度任务指标
            try:
                results.update(self.calculate_semantic_similarity(predictions, references))
            except Exception as e:
                logger.warning(f"计算相似度指标时出错: {e}")
        
        # 添加基本统计信息
        results["num_samples"] = len(predictions)
        results["task_type"] = task_type
        
        return convert_numpy_types(results)
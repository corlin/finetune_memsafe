"""
评估协议

实现标准评估协议的执行，包括CLUE、FewCLUE、C-Eval等官方评估流程。
"""

import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import numpy as np

from datasets import Dataset
from .data_models import BenchmarkConfig, TaskResult, convert_numpy_types
from .metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class BaseEvaluationProtocol(ABC):
    """
    基础评估协议抽象类
    """
    
    def __init__(self, metrics_calculator: MetricsCalculator):
        """
        初始化评估协议
        
        Args:
            metrics_calculator: 指标计算器
        """
        self.metrics_calculator = metrics_calculator
    
    @abstractmethod
    def evaluate_task(self, 
                     task_name: str,
                     dataset: Dataset,
                     inference_func: Callable,
                     config: BenchmarkConfig,
                     **kwargs) -> TaskResult:
        """
        评估单个任务
        
        Args:
            task_name: 任务名称
            dataset: 数据集
            inference_func: 推理函数
            config: 基准测试配置
            **kwargs: 其他参数
            
        Returns:
            任务评估结果
        """
        pass
    
    def _prepare_inputs(self, batch: Dict[str, List], task_name: str) -> List[str]:
        """
        准备输入数据
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            输入文本列表
        """
        # 默认实现，子类可以重写
        return batch.get("sentence", batch.get("text", batch.get("input", [])))
    
    def _extract_references(self, batch: Dict[str, List], task_name: str) -> List[str]:
        """
        提取参考答案
        
        Args:
            batch: 批次数据
            task_name: 任务名称
            
        Returns:
            参考答案列表
        """
        # 默认实现，子类可以重写
        return batch.get("label", batch.get("target", batch.get("answer", [])))


class CLUEProtocol(BaseEvaluationProtocol):
    """
    CLUE评估协议
    
    按照CLUE官方协议实现评估流程。
    """
    
    def evaluate_task(self, 
                     task_name: str,
                     dataset: Dataset,
                     inference_func: Callable,
                     config: BenchmarkConfig,
                     **kwargs) -> TaskResult:
        """
        按照CLUE协议评估任务
        
        Args:
            task_name: 任务名称
            dataset: 数据集
            inference_func: 推理函数
            config: 基准测试配置
            **kwargs: 其他参数
            
        Returns:
            任务评估结果
        """
        logger.info(f"使用CLUE协议评估任务: {task_name}")
        
        # 根据任务类型选择评估方法
        if task_name == "tnews":
            return self._evaluate_tnews(dataset, inference_func, **kwargs)
        elif task_name == "afqmc":
            return self._evaluate_afqmc(dataset, inference_func, **kwargs)
        elif task_name == "cmnli":
            return self._evaluate_cmnli(dataset, inference_func, **kwargs)
        elif task_name == "ocnli":
            return self._evaluate_ocnli(dataset, inference_func, **kwargs)
        elif task_name == "wsc":
            return self._evaluate_wsc(dataset, inference_func, **kwargs)
        elif task_name == "csl":
            return self._evaluate_csl(dataset, inference_func, **kwargs)
        else:
            # 通用分类评估
            return self._evaluate_classification_task(task_name, dataset, inference_func, **kwargs)
    
    def _evaluate_tnews(self, 
                       dataset: Dataset, 
                       inference_func: Callable,
                       **kwargs) -> TaskResult:
        """
        评估今日头条中文新闻分类任务
        
        Args:
            dataset: 数据集
            inference_func: 推理函数
            **kwargs: 其他参数
            
        Returns:
            任务评估结果
        """
        # 新闻分类标签映射
        label_map = {
            "100": "故事", "101": "文化", "102": "娱乐", "103": "体育",
            "104": "财经", "106": "房产", "107": "汽车", "108": "教育",
            "109": "科技", "110": "军事", "112": "旅游", "113": "国际",
            "114": "股票", "115": "农业", "116": "游戏"
        }
        
        predictions = []
        references = []
        inputs = []
        
        for item in dataset:
            # 准备输入
            sentence = item.get("sentence", "")
            input_text = f"请对以下新闻进行分类：{sentence}"
            inputs.append(input_text)
            
            # 执行推理
            try:
                pred = inference_func([input_text])[0]
                # 尝试映射到标准标签
                pred_label = self._map_prediction_to_label(pred, label_map)
                predictions.append(pred_label)
            except Exception as e:
                logger.warning(f"推理失败: {e}")
                predictions.append("100")  # 默认标签
            
            # 获取真实标签
            true_label = str(item.get("label", "100"))
            references.append(true_label)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_classification_metrics(
            predictions, references, average="weighted"
        )
        
        return TaskResult(
            task_name="tnews",
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=[],  # 简化实现
            execution_time=0.0
        )
    
    def _evaluate_afqmc(self, 
                       dataset: Dataset, 
                       inference_func: Callable,
                       **kwargs) -> TaskResult:
        """
        评估蚂蚁金融语义相似度任务
        
        Args:
            dataset: 数据集
            inference_func: 推理函数
            **kwargs: 其他参数
            
        Returns:
            任务评估结果
        """
        predictions = []
        references = []
        inputs = []
        
        for item in dataset:
            # 准备输入
            sentence1 = item.get("sentence1", "")
            sentence2 = item.get("sentence2", "")
            input_text = f"判断以下两个句子是否语义相似：\n句子1：{sentence1}\n句子2：{sentence2}\n请回答：相似 或 不相似"
            inputs.append(input_text)
            
            # 执行推理
            try:
                pred = inference_func([input_text])[0]
                # 映射到二分类标签
                if "相似" in pred and "不相似" not in pred:
                    pred_label = "1"
                else:
                    pred_label = "0"
                predictions.append(pred_label)
            except Exception as e:
                logger.warning(f"推理失败: {e}")
                predictions.append("0")
            
            # 获取真实标签
            true_label = str(item.get("label", 0))
            references.append(true_label)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_classification_metrics(
            predictions, references, average="binary"
        )
        
        return TaskResult(
            task_name="afqmc",
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=[],
            execution_time=0.0
        )
    
    def _evaluate_cmnli(self, 
                       dataset: Dataset, 
                       inference_func: Callable,
                       **kwargs) -> TaskResult:
        """
        评估中文自然语言推理任务
        
        Args:
            dataset: 数据集
            inference_func: 推理函数
            **kwargs: 其他参数
            
        Returns:
            任务评估结果
        """
        label_map = {"entailment": "蕴含", "contradiction": "矛盾", "neutral": "中性"}
        reverse_map = {v: k for k, v in label_map.items()}
        
        predictions = []
        references = []
        inputs = []
        
        for item in dataset:
            # 准备输入
            premise = item.get("premise", "")
            hypothesis = item.get("hypothesis", "")
            input_text = f"前提：{premise}\n假设：{hypothesis}\n请判断假设与前提的关系：蕴含、矛盾、中性"
            inputs.append(input_text)
            
            # 执行推理
            try:
                pred = inference_func([input_text])[0]
                # 映射到标准标签
                pred_label = "neutral"  # 默认
                for chinese_label, english_label in reverse_map.items():
                    if chinese_label in pred:
                        pred_label = english_label
                        break
                predictions.append(pred_label)
            except Exception as e:
                logger.warning(f"推理失败: {e}")
                predictions.append("neutral")
            
            # 获取真实标签
            true_label = item.get("label", "neutral")
            references.append(true_label)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_classification_metrics(
            predictions, references, average="weighted"
        )
        
        return TaskResult(
            task_name="cmnli",
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=[],
            execution_time=0.0
        )
    
    def _evaluate_ocnli(self, 
                       dataset: Dataset, 
                       inference_func: Callable,
                       **kwargs) -> TaskResult:
        """
        评估原创中文自然语言推理任务
        """
        # OCNLI与CMNLI类似，使用相同的评估逻辑
        return self._evaluate_cmnli(dataset, inference_func, **kwargs)
    
    def _evaluate_wsc(self, 
                     dataset: Dataset, 
                     inference_func: Callable,
                     **kwargs) -> TaskResult:
        """
        评估威诺格拉德模式挑战任务
        
        Args:
            dataset: 数据集
            inference_func: 推理函数
            **kwargs: 其他参数
            
        Returns:
            任务评估结果
        """
        predictions = []
        references = []
        inputs = []
        
        for item in dataset:
            # 准备输入
            text = item.get("text", "")
            target = item.get("target", {})
            span1_text = target.get("span1_text", "")
            span2_text = target.get("span2_text", "")
            
            input_text = f"在以下句子中，"{span1_text}"指代的是"{span2_text}"吗？\n句子：{text}\n请回答：是 或 否"
            inputs.append(input_text)
            
            # 执行推理
            try:
                pred = inference_func([input_text])[0]
                pred_label = "true" if "是" in pred else "false"
                predictions.append(pred_label)
            except Exception as e:
                logger.warning(f"推理失败: {e}")
                predictions.append("false")
            
            # 获取真实标签
            true_label = "true" if item.get("label", False) else "false"
            references.append(true_label)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_classification_metrics(
            predictions, references, average="binary"
        )
        
        return TaskResult(
            task_name="wsc",
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=[],
            execution_time=0.0
        )
    
    def _evaluate_csl(self, 
                     dataset: Dataset, 
                     inference_func: Callable,
                     **kwargs) -> TaskResult:
        """
        评估中科院科学文献数据集任务
        
        Args:
            dataset: 数据集
            inference_func: 推理函数
            **kwargs: 其他参数
            
        Returns:
            任务评估结果
        """
        predictions = []
        references = []
        inputs = []
        
        for item in dataset:
            # 准备输入
            abst = item.get("abst", "")
            keyword = item.get("keyword", "")
            input_text = f"以下关键词是否适合这篇论文摘要？\n摘要：{abst}\n关键词：{keyword}\n请回答：适合 或 不适合"
            inputs.append(input_text)
            
            # 执行推理
            try:
                pred = inference_func([input_text])[0]
                pred_label = "1" if "适合" in pred else "0"
                predictions.append(pred_label)
            except Exception as e:
                logger.warning(f"推理失败: {e}")
                predictions.append("0")
            
            # 获取真实标签
            true_label = str(item.get("label", 0))
            references.append(true_label)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_classification_metrics(
            predictions, references, average="binary"
        )
        
        return TaskResult(
            task_name="csl",
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=[],
            execution_time=0.0
        )
    
    def _evaluate_classification_task(self, 
                                    task_name: str,
                                    dataset: Dataset, 
                                    inference_func: Callable,
                                    **kwargs) -> TaskResult:
        """
        通用分类任务评估
        
        Args:
            task_name: 任务名称
            dataset: 数据集
            inference_func: 推理函数
            **kwargs: 其他参数
            
        Returns:
            任务评估结果
        """
        predictions = []
        references = []
        inputs = []
        
        for item in dataset:
            # 准备输入
            text = item.get("sentence", item.get("text", ""))
            input_text = f"请对以下文本进行分类：{text}"
            inputs.append(input_text)
            
            # 执行推理
            try:
                pred = inference_func([input_text])[0]
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"推理失败: {e}")
                predictions.append("")
            
            # 获取真实标签
            true_label = str(item.get("label", ""))
            references.append(true_label)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_classification_metrics(
            predictions, references, average="weighted"
        )
        
        return TaskResult(
            task_name=task_name,
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=[],
            execution_time=0.0
        )
    
    def _map_prediction_to_label(self, prediction: str, label_map: Dict[str, str]) -> str:
        """
        将预测结果映射到标准标签
        
        Args:
            prediction: 预测结果
            label_map: 标签映射字典
            
        Returns:
            映射后的标签
        """
        # 尝试直接匹配
        for label_id, label_name in label_map.items():
            if label_name in prediction or label_id in prediction:
                return label_id
        
        # 如果没有匹配，返回第一个标签作为默认值
        return list(label_map.keys())[0]


class FewCLUEProtocol(BaseEvaluationProtocol):
    """
    FewCLUE评估协议
    
    实现少样本学习评估。
    """
    
    def __init__(self, metrics_calculator: MetricsCalculator, num_shots: int = 5):
        """
        初始化FewCLUE协议
        
        Args:
            metrics_calculator: 指标计算器
            num_shots: 少样本数量
        """
        super().__init__(metrics_calculator)
        self.num_shots = num_shots
    
    def evaluate_task(self, 
                     task_name: str,
                     dataset: Dataset,
                     inference_func: Callable,
                     config: BenchmarkConfig,
                     **kwargs) -> TaskResult:
        """
        按照FewCLUE协议评估任务
        
        Args:
            task_name: 任务名称
            dataset: 数据集
            inference_func: 推理函数
            config: 基准测试配置
            **kwargs: 其他参数
            
        Returns:
            任务评估结果
        """
        logger.info(f"使用FewCLUE协议评估任务: {task_name}, {self.num_shots}-shot")
        
        # 分离支持集和查询集
        support_set, query_set = self._split_support_query(dataset)
        
        # 构建少样本示例
        few_shot_examples = self._build_few_shot_examples(support_set, task_name)
        
        predictions = []
        references = []
        inputs = []
        
        for item in query_set:
            # 构建包含少样本示例的输入
            input_text = self._build_few_shot_input(few_shot_examples, item, task_name)
            inputs.append(input_text)
            
            # 执行推理
            try:
                pred = inference_func([input_text])[0]
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"推理失败: {e}")
                predictions.append("")
            
            # 获取真实标签
            true_label = str(item.get("label", ""))
            references.append(true_label)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_classification_metrics(
            predictions, references, average="weighted"
        )
        
        # 添加少样本特定指标
        metrics["num_shots"] = self.num_shots
        metrics["support_set_size"] = len(support_set)
        metrics["query_set_size"] = len(query_set)
        
        return TaskResult(
            task_name=f"{task_name}_few_shot",
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=[],
            execution_time=0.0
        )
    
    def _split_support_query(self, dataset: Dataset) -> Tuple[List[Dict], List[Dict]]:
        """
        分离支持集和查询集
        
        Args:
            dataset: 数据集
            
        Returns:
            支持集和查询集
        """
        data_list = list(dataset)
        random.shuffle(data_list)
        
        # 按标签分组
        label_groups = {}
        for item in data_list:
            label = str(item.get("label", ""))
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)
        
        support_set = []
        query_set = []
        
        # 从每个标签中选择少样本示例
        for label, items in label_groups.items():
            if len(items) > self.num_shots:
                support_set.extend(items[:self.num_shots])
                query_set.extend(items[self.num_shots:])
            else:
                # 如果样本不足，全部作为支持集
                support_set.extend(items)
        
        return support_set, query_set
    
    def _build_few_shot_examples(self, support_set: List[Dict], task_name: str) -> str:
        """
        构建少样本示例
        
        Args:
            support_set: 支持集
            task_name: 任务名称
            
        Returns:
            少样本示例字符串
        """
        examples = []
        
        for item in support_set[:self.num_shots]:
            if task_name == "tnews":
                sentence = item.get("sentence", "")
                label = item.get("label", "")
                examples.append(f"新闻：{sentence}\n分类：{label}")
            elif task_name == "afqmc":
                sentence1 = item.get("sentence1", "")
                sentence2 = item.get("sentence2", "")
                label = "相似" if item.get("label") == "1" else "不相似"
                examples.append(f"句子1：{sentence1}\n句子2：{sentence2}\n关系：{label}")
            else:
                # 通用格式
                text = item.get("sentence", item.get("text", ""))
                label = item.get("label", "")
                examples.append(f"文本：{text}\n标签：{label}")
        
        return "\n\n".join(examples)
    
    def _build_few_shot_input(self, examples: str, query_item: Dict, task_name: str) -> str:
        """
        构建包含少样本示例的输入
        
        Args:
            examples: 少样本示例
            query_item: 查询项
            task_name: 任务名称
            
        Returns:
            完整的输入文本
        """
        if task_name == "tnews":
            query_sentence = query_item.get("sentence", "")
            return f"以下是一些新闻分类的例子：\n\n{examples}\n\n现在请对以下新闻进行分类：\n新闻：{query_sentence}\n分类："
        elif task_name == "afqmc":
            sentence1 = query_item.get("sentence1", "")
            sentence2 = query_item.get("sentence2", "")
            return f"以下是一些句子相似度判断的例子：\n\n{examples}\n\n现在请判断以下句子对的关系：\n句子1：{sentence1}\n句子2：{sentence2}\n关系："
        else:
            # 通用格式
            text = query_item.get("sentence", query_item.get("text", ""))
            return f"以下是一些分类的例子：\n\n{examples}\n\n现在请对以下文本进行分类：\n文本：{text}\n标签："


class CEvalProtocol(BaseEvaluationProtocol):
    """
    C-Eval评估协议
    
    实现C-Eval知识和推理评估。
    """
    
    def evaluate_task(self, 
                     task_name: str,
                     dataset: Dataset,
                     inference_func: Callable,
                     config: BenchmarkConfig,
                     **kwargs) -> TaskResult:
        """
        按照C-Eval协议评估任务
        
        Args:
            task_name: 任务名称
            dataset: 数据集
            inference_func: 推理函数
            config: 基准测试配置
            **kwargs: 其他参数
            
        Returns:
            任务评估结果
        """
        logger.info(f"使用C-Eval协议评估任务: {task_name}")
        
        predictions = []
        references = []
        inputs = []
        
        for item in dataset:
            # 准备多选题输入
            question = item.get("question", "")
            choices = item.get("choices", [])
            
            # 构建选择题格式
            choices_text = ""
            for i, choice in enumerate(choices):
                choices_text += f"{chr(65 + i)}. {choice}\n"
            
            input_text = f"问题：{question}\n选项：\n{choices_text}请选择正确答案（A、B、C或D）："
            inputs.append(input_text)
            
            # 执行推理
            try:
                pred = inference_func([input_text])[0]
                # 提取选择的选项
                pred_choice = self._extract_choice(pred)
                predictions.append(pred_choice)
            except Exception as e:
                logger.warning(f"推理失败: {e}")
                predictions.append("A")  # 默认选择A
            
            # 获取真实答案
            true_answer = item.get("answer", "A")
            references.append(true_answer)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_classification_metrics(
            predictions, references, average="weighted"
        )
        
        # 添加C-Eval特定指标
        metrics["subject"] = task_name
        metrics["question_type"] = "multiple_choice"
        
        return TaskResult(
            task_name=f"c_eval_{task_name}",
            predictions=predictions,
            references=references,
            metrics=metrics,
            samples=[],
            execution_time=0.0
        )
    
    def _extract_choice(self, prediction: str) -> str:
        """
        从预测结果中提取选择的选项
        
        Args:
            prediction: 预测结果
            
        Returns:
            选择的选项（A、B、C或D）
        """
        prediction = prediction.upper()
        
        # 查找选项字母
        for choice in ["A", "B", "C", "D"]:
            if choice in prediction:
                return choice
        
        # 如果没有找到，返回A作为默认值
        return "A"


class EvaluationProtocolFactory:
    """
    评估协议工厂类
    
    根据协议类型创建相应的评估协议。
    """
    
    @staticmethod
    def create_protocol(protocol_type: str, 
                       metrics_calculator: MetricsCalculator,
                       **kwargs) -> BaseEvaluationProtocol:
        """
        创建评估协议
        
        Args:
            protocol_type: 协议类型
            metrics_calculator: 指标计算器
            **kwargs: 其他参数
            
        Returns:
            评估协议实例
        """
        if protocol_type.lower() == "official" or protocol_type.lower() == "clue":
            return CLUEProtocol(metrics_calculator)
        elif protocol_type.lower() == "few_shot" or protocol_type.lower() == "fewclue":
            num_shots = kwargs.get("num_shots", 5)
            return FewCLUEProtocol(metrics_calculator, num_shots)
        elif protocol_type.lower() == "multiple_choice" or protocol_type.lower() == "c_eval":
            return CEvalProtocol(metrics_calculator)
        else:
            logger.warning(f"未知协议类型: {protocol_type}，使用默认CLUE协议")
            return CLUEProtocol(metrics_calculator)
    
    @staticmethod
    def get_supported_protocols() -> List[str]:
        """
        获取支持的协议类型列表
        
        Returns:
            支持的协议类型列表
        """
        return [
            "official",
            "clue", 
            "few_shot",
            "fewclue",
            "multiple_choice",
            "c_eval"
        ]
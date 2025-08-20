"""
核心接口定义
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from industry_evaluation.models.data_models import (
    EvaluationConfig, EvaluationResult, EvaluationScore, 
    Dataset, ProgressInfo, Report, Criterion, Explanation
)


class BaseEvaluator(ABC):
    """评估器基础抽象类"""
    
    @abstractmethod
    def evaluate(self, input_text: str, model_output: str, 
                expected_output: str, context: Dict[str, Any]) -> EvaluationScore:
        """
        执行评估
        
        Args:
            input_text: 输入文本
            model_output: 模型输出
            expected_output: 期望输出
            context: 上下文信息
            
        Returns:
            EvaluationScore: 评估分数
        """
        pass
    
    @abstractmethod
    def get_evaluation_criteria(self) -> List[Criterion]:
        """
        获取评估标准
        
        Returns:
            List[Criterion]: 评估标准列表
        """
        pass
    
    @abstractmethod
    def explain_result(self, score: EvaluationScore) -> Explanation:
        """
        解释评估结果
        
        Args:
            score: 评估分数
            
        Returns:
            Explanation: 结果解释
        """
        pass


class EvaluationEngine(ABC):
    """评估引擎接口"""
    
    @abstractmethod
    def evaluate_model(self, model_id: str, dataset: Dataset, 
                      evaluation_config: EvaluationConfig) -> EvaluationResult:
        """
        评估模型
        
        Args:
            model_id: 模型ID
            dataset: 测试数据集
            evaluation_config: 评估配置
            
        Returns:
            EvaluationResult: 评估结果
        """
        pass
    
    @abstractmethod
    def get_evaluation_progress(self, task_id: str) -> ProgressInfo:
        """
        获取评估进度
        
        Args:
            task_id: 任务ID
            
        Returns:
            ProgressInfo: 进度信息
        """
        pass
    
    @abstractmethod
    def generate_report(self, evaluation_result: EvaluationResult) -> Report:
        """
        生成评估报告
        
        Args:
            evaluation_result: 评估结果
            
        Returns:
            Report: 评估报告
        """
        pass


class ModelAdapter(ABC):
    """模型适配器接口"""
    
    @abstractmethod
    def predict(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        模型预测
        
        Args:
            input_text: 输入文本
            context: 上下文信息
            
        Returns:
            str: 模型输出
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        检查模型是否可用
        
        Returns:
            bool: 是否可用
        """
        pass


class DataProcessor(ABC):
    """数据处理器接口"""
    
    @abstractmethod
    def preprocess(self, raw_data: Any) -> Dataset:
        """
        数据预处理
        
        Args:
            raw_data: 原始数据
            
        Returns:
            Dataset: 处理后的数据集
        """
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        数据验证
        
        Args:
            data: 待验证数据
            
        Returns:
            bool: 验证结果
        """
        pass


class ReportGenerator(ABC):
    """报告生成器接口"""
    
    @abstractmethod
    def generate(self, evaluation_result: EvaluationResult, 
                template: Optional[str] = None) -> Report:
        """
        生成报告
        
        Args:
            evaluation_result: 评估结果
            template: 报告模板
            
        Returns:
            Report: 生成的报告
        """
        pass
    
    @abstractmethod
    def export(self, report: Report, format_type: str, output_path: str) -> bool:
        """
        导出报告
        
        Args:
            report: 报告对象
            format_type: 导出格式
            output_path: 输出路径
            
        Returns:
            bool: 导出是否成功
        """
        pass
"""
基础评估器实现
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from industry_evaluation.models.data_models import (
    EvaluationScore, Criterion, Explanation, ErrorType
)
from industry_evaluation.core.interfaces import BaseEvaluator


class AbstractEvaluator(BaseEvaluator):
    """抽象评估器基类"""
    
    def __init__(self, name: str, weight: float = 1.0):
        """
        初始化评估器
        
        Args:
            name: 评估器名称
            weight: 评估器权重
        """
        self.name = name
        self.weight = weight
        self.criteria = self._initialize_criteria()
    
    @abstractmethod
    def _initialize_criteria(self) -> List[Criterion]:
        """
        初始化评估标准
        
        Returns:
            List[Criterion]: 评估标准列表
        """
        pass
    
    @abstractmethod
    def _calculate_score(self, input_text: str, model_output: str, 
                        expected_output: str, context: Dict[str, Any]) -> float:
        """
        计算评估分数
        
        Args:
            input_text: 输入文本
            model_output: 模型输出
            expected_output: 期望输出
            context: 上下文信息
            
        Returns:
            float: 评估分数 (0-1)
        """
        pass
    
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
        try:
            # 预处理
            input_text = self._preprocess_text(input_text)
            model_output = self._preprocess_text(model_output)
            expected_output = self._preprocess_text(expected_output)
            
            # 计算分数
            score = self._calculate_score(input_text, model_output, expected_output, context)
            
            # 确保分数在有效范围内
            score = max(0.0, min(1.0, score))
            
            # 计算置信度
            confidence = self._calculate_confidence(input_text, model_output, expected_output, context)
            
            # 生成详细信息
            details = self._generate_details(input_text, model_output, expected_output, context, score)
            
            return EvaluationScore(
                overall_score=score,
                dimension_scores={self.name: score},
                confidence=confidence,
                details=details
            )
            
        except Exception as e:
            # 评估失败时返回零分
            return EvaluationScore(
                overall_score=0.0,
                dimension_scores={self.name: 0.0},
                confidence=0.0,
                details={"error": str(e), "evaluator": self.name}
            )
    
    def get_evaluation_criteria(self) -> List[Criterion]:
        """
        获取评估标准
        
        Returns:
            List[Criterion]: 评估标准列表
        """
        return self.criteria.copy()
    
    def explain_result(self, score: EvaluationScore) -> Explanation:
        """
        解释评估结果
        
        Args:
            score: 评估分数
            
        Returns:
            Explanation: 结果解释
        """
        summary = self._generate_summary(score)
        reasoning_steps = self._generate_reasoning_steps(score)
        
        return Explanation(
            summary=summary,
            details=score.details,
            confidence=score.confidence,
            reasoning_steps=reasoning_steps
        )
    
    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            str: 预处理后的文本
        """
        if not isinstance(text, str):
            text = str(text)
        return text.strip()
    
    def _calculate_confidence(self, input_text: str, model_output: str, 
                            expected_output: str, context: Dict[str, Any]) -> float:
        """
        计算置信度
        
        Args:
            input_text: 输入文本
            model_output: 模型输出
            expected_output: 期望输出
            context: 上下文信息
            
        Returns:
            float: 置信度 (0-1)
        """
        # 默认置信度计算逻辑
        confidence = 1.0
        
        # 根据文本长度调整置信度
        if len(model_output) < 5:
            confidence *= 0.8
        
        if len(expected_output) < 5:
            confidence *= 0.9
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_details(self, input_text: str, model_output: str, 
                         expected_output: str, context: Dict[str, Any], 
                         score: float) -> Dict[str, Any]:
        """
        生成详细信息
        
        Args:
            input_text: 输入文本
            model_output: 模型输出
            expected_output: 期望输出
            context: 上下文信息
            score: 评估分数
            
        Returns:
            Dict[str, Any]: 详细信息
        """
        return {
            "evaluator": self.name,
            "score": score,
            "input_length": len(input_text),
            "output_length": len(model_output),
            "expected_length": len(expected_output)
        }
    
    def _generate_summary(self, score: EvaluationScore) -> str:
        """
        生成评估摘要
        
        Args:
            score: 评估分数
            
        Returns:
            str: 评估摘要
        """
        score_value = score.dimension_scores.get(self.name, 0.0)
        
        if score_value >= 0.9:
            level = "优秀"
        elif score_value >= 0.8:
            level = "良好"
        elif score_value >= 0.6:
            level = "及格"
        else:
            level = "不及格"
        
        return f"{self.name}评估结果: {level} (分数: {score_value:.2f})"
    
    def _generate_reasoning_steps(self, score: EvaluationScore) -> List[str]:
        """
        生成推理步骤
        
        Args:
            score: 评估分数
            
        Returns:
            List[str]: 推理步骤
        """
        steps = [
            f"1. 使用{self.name}评估器进行评估",
            f"2. 计算得分: {score.dimension_scores.get(self.name, 0.0):.2f}",
            f"3. 置信度: {score.confidence:.2f}"
        ]
        
        return steps


class CompositeEvaluator(AbstractEvaluator):
    """复合评估器 - 组合多个子评估器"""
    
    def __init__(self, name: str, sub_evaluators: List[AbstractEvaluator], 
                 weights: Optional[Dict[str, float]] = None):
        """
        初始化复合评估器
        
        Args:
            name: 评估器名称
            sub_evaluators: 子评估器列表
            weights: 子评估器权重配置
        """
        super().__init__(name)
        self.sub_evaluators = sub_evaluators
        self.weights = weights or {}
        
        # 如果没有指定权重，则平均分配
        if not self.weights:
            weight_per_evaluator = 1.0 / len(sub_evaluators)
            self.weights = {evaluator.name: weight_per_evaluator 
                          for evaluator in sub_evaluators}
    
    def _initialize_criteria(self) -> List[Criterion]:
        """初始化评估标准"""
        criteria = []
        for evaluator in self.sub_evaluators:
            criteria.extend(evaluator.get_evaluation_criteria())
        return criteria
    
    def _calculate_score(self, input_text: str, model_output: str, 
                        expected_output: str, context: Dict[str, Any]) -> float:
        """计算复合评估分数"""
        total_score = 0.0
        total_weight = 0.0
        
        for evaluator in self.sub_evaluators:
            try:
                result = evaluator.evaluate(input_text, model_output, expected_output, context)
                weight = self.weights.get(evaluator.name, 0.0)
                total_score += result.overall_score * weight
                total_weight += weight
            except Exception as e:
                print(f"子评估器 {evaluator.name} 评估失败: {e}")
                continue
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def evaluate(self, input_text: str, model_output: str, 
                expected_output: str, context: Dict[str, Any]) -> EvaluationScore:
        """执行复合评估"""
        dimension_scores = {}
        details = {"sub_evaluations": {}}
        total_confidence = 0.0
        valid_evaluators = 0
        
        # 执行所有子评估器
        for evaluator in self.sub_evaluators:
            try:
                result = evaluator.evaluate(input_text, model_output, expected_output, context)
                dimension_scores[evaluator.name] = result.overall_score
                details["sub_evaluations"][evaluator.name] = result.details
                total_confidence += result.confidence
                valid_evaluators += 1
            except Exception as e:
                dimension_scores[evaluator.name] = 0.0
                details["sub_evaluations"][evaluator.name] = {"error": str(e)}
        
        # 计算总分
        overall_score = self._calculate_score(input_text, model_output, expected_output, context)
        
        # 计算平均置信度
        avg_confidence = total_confidence / valid_evaluators if valid_evaluators > 0 else 0.0
        
        return EvaluationScore(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            confidence=avg_confidence,
            details=details
        )


class RuleBasedEvaluator(AbstractEvaluator):
    """基于规则的评估器"""
    
    def __init__(self, name: str, rules: List[Dict[str, Any]], weight: float = 1.0):
        """
        初始化规则评估器
        
        Args:
            name: 评估器名称
            rules: 规则列表，每个规则包含条件和分数
            weight: 评估器权重
        """
        super().__init__(name, weight)
        self.rules = rules
    
    def _initialize_criteria(self) -> List[Criterion]:
        """初始化评估标准"""
        criteria = []
        for i, rule in enumerate(self.rules):
            criterion = Criterion(
                name=f"{self.name}_rule_{i+1}",
                description=rule.get("description", f"规则 {i+1}"),
                weight=rule.get("weight", 1.0),
                threshold=rule.get("threshold", 0.5),
                evaluation_method="rule_based"
            )
            criteria.append(criterion)
        return criteria
    
    def _calculate_score(self, input_text: str, model_output: str, 
                        expected_output: str, context: Dict[str, Any]) -> float:
        """基于规则计算分数"""
        total_score = 0.0
        total_weight = 0.0
        
        for rule in self.rules:
            try:
                # 执行规则检查
                rule_score = self._apply_rule(rule, input_text, model_output, expected_output, context)
                rule_weight = rule.get("weight", 1.0)
                
                total_score += rule_score * rule_weight
                total_weight += rule_weight
                
            except Exception as e:
                print(f"规则执行失败: {e}")
                continue
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def _apply_rule(self, rule: Dict[str, Any], input_text: str, model_output: str, 
                   expected_output: str, context: Dict[str, Any]) -> float:
        """
        应用单个规则
        
        Args:
            rule: 规则定义
            input_text: 输入文本
            model_output: 模型输出
            expected_output: 期望输出
            context: 上下文信息
            
        Returns:
            float: 规则评估分数
        """
        rule_type = rule.get("type", "exact_match")
        
        if rule_type == "exact_match":
            return 1.0 if model_output.strip() == expected_output.strip() else 0.0
        
        elif rule_type == "contains":
            keywords = rule.get("keywords", [])
            found_keywords = sum(1 for keyword in keywords if keyword in model_output)
            return found_keywords / len(keywords) if keywords else 0.0
        
        elif rule_type == "length_check":
            min_length = rule.get("min_length", 0)
            max_length = rule.get("max_length", float('inf'))
            output_length = len(model_output)
            
            if min_length <= output_length <= max_length:
                return 1.0
            else:
                return 0.0
        
        elif rule_type == "regex_match":
            import re
            pattern = rule.get("pattern", "")
            if re.search(pattern, model_output):
                return 1.0
            else:
                return 0.0
        
        else:
            # 未知规则类型
            return 0.0
    
    def add_rule(self, rule: Dict[str, Any]):
        """添加规则"""
        self.rules.append(rule)
        self.criteria = self._initialize_criteria()
    
    def remove_rule(self, rule_index: int):
        """移除规则"""
        if 0 <= rule_index < len(self.rules):
            self.rules.pop(rule_index)
            self.criteria = self._initialize_criteria()
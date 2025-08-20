"""
基础评估器测试
"""

import pytest
from industry_evaluation.evaluators.base_evaluator import (
    AbstractEvaluator, CompositeEvaluator, RuleBasedEvaluator
)
from industry_evaluation.models.data_models import (
    EvaluationScore, Criterion, Explanation
)


class MockEvaluator(AbstractEvaluator):
    """模拟评估器用于测试"""
    
    def _initialize_criteria(self):
        return [
            Criterion(
                name="mock_criterion",
                description="模拟评估标准",
                weight=1.0,
                threshold=0.5,
                evaluation_method="mock"
            )
        ]
    
    def _calculate_score(self, input_text, model_output, expected_output, context):
        # 简单的模拟评分逻辑
        if model_output == expected_output:
            return 1.0
        elif len(model_output) > 0:
            return 0.5
        else:
            return 0.0


class TestAbstractEvaluator:
    """抽象评估器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.evaluator = MockEvaluator("mock_evaluator")
    
    def test_initialization(self):
        """测试初始化"""
        assert self.evaluator.name == "mock_evaluator"
        assert self.evaluator.weight == 1.0
        assert len(self.evaluator.criteria) == 1
        assert self.evaluator.criteria[0].name == "mock_criterion"
    
    def test_evaluate_perfect_match(self):
        """测试完全匹配的评估"""
        result = self.evaluator.evaluate(
            input_text="测试输入",
            model_output="测试输出",
            expected_output="测试输出",
            context={}
        )
        
        assert isinstance(result, EvaluationScore)
        assert result.overall_score == 1.0
        assert result.dimension_scores["mock_evaluator"] == 1.0
        assert result.confidence > 0
    
    def test_evaluate_partial_match(self):
        """测试部分匹配的评估"""
        result = self.evaluator.evaluate(
            input_text="测试输入",
            model_output="不同输出",
            expected_output="测试输出",
            context={}
        )
        
        assert result.overall_score == 0.5
        assert result.dimension_scores["mock_evaluator"] == 0.5
    
    def test_evaluate_no_output(self):
        """测试无输出的评估"""
        result = self.evaluator.evaluate(
            input_text="测试输入",
            model_output="",
            expected_output="测试输出",
            context={}
        )
        
        assert result.overall_score == 0.0
        assert result.dimension_scores["mock_evaluator"] == 0.0
    
    def test_get_evaluation_criteria(self):
        """测试获取评估标准"""
        criteria = self.evaluator.get_evaluation_criteria()
        assert len(criteria) == 1
        assert criteria[0].name == "mock_criterion"
        
        # 确保返回的是副本
        criteria[0].name = "modified"
        original_criteria = self.evaluator.get_evaluation_criteria()
        assert original_criteria[0].name == "mock_criterion"
    
    def test_explain_result(self):
        """测试结果解释"""
        score = EvaluationScore(
            overall_score=0.8,
            dimension_scores={"mock_evaluator": 0.8},
            confidence=0.9
        )
        
        explanation = self.evaluator.explain_result(score)
        
        assert isinstance(explanation, Explanation)
        assert "良好" in explanation.summary
        assert explanation.confidence == 0.9
        assert len(explanation.reasoning_steps) > 0
    
    def test_preprocess_text(self):
        """测试文本预处理"""
        # 测试字符串输入
        result = self.evaluator._preprocess_text("  测试文本  ")
        assert result == "测试文本"
        
        # 测试非字符串输入
        result = self.evaluator._preprocess_text(123)
        assert result == "123"
        
        # 测试None输入
        result = self.evaluator._preprocess_text(None)
        assert result == "None"
    
    def test_calculate_confidence(self):
        """测试置信度计算"""
        # 正常长度文本
        confidence = self.evaluator._calculate_confidence(
            "测试输入", "正常长度的输出", "期望输出", {}
        )
        assert confidence == 1.0
        
        # 短输出文本
        confidence = self.evaluator._calculate_confidence(
            "测试输入", "短", "期望输出", {}
        )
        assert confidence == 0.8
        
        # 短期望输出
        confidence = self.evaluator._calculate_confidence(
            "测试输入", "正常长度的输出", "短", {}
        )
        assert confidence == 0.9
    
    def test_generate_summary(self):
        """测试摘要生成"""
        # 优秀分数
        score = EvaluationScore(
            overall_score=0.95,
            dimension_scores={"mock_evaluator": 0.95},
            confidence=0.9
        )
        summary = self.evaluator._generate_summary(score)
        assert "优秀" in summary
        
        # 良好分数
        score.dimension_scores["mock_evaluator"] = 0.85
        summary = self.evaluator._generate_summary(score)
        assert "良好" in summary
        
        # 及格分数
        score.dimension_scores["mock_evaluator"] = 0.65
        summary = self.evaluator._generate_summary(score)
        assert "及格" in summary
        
        # 不及格分数
        score.dimension_scores["mock_evaluator"] = 0.45
        summary = self.evaluator._generate_summary(score)
        assert "不及格" in summary


class TestCompositeEvaluator:
    """复合评估器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.sub_evaluator1 = MockEvaluator("sub_evaluator_1")
        self.sub_evaluator2 = MockEvaluator("sub_evaluator_2")
        
        self.composite = CompositeEvaluator(
            name="composite_evaluator",
            sub_evaluators=[self.sub_evaluator1, self.sub_evaluator2]
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.composite.name == "composite_evaluator"
        assert len(self.composite.sub_evaluators) == 2
        assert len(self.composite.weights) == 2
        assert self.composite.weights["sub_evaluator_1"] == 0.5
        assert self.composite.weights["sub_evaluator_2"] == 0.5
    
    def test_initialization_with_weights(self):
        """测试带权重的初始化"""
        weights = {"sub_evaluator_1": 0.7, "sub_evaluator_2": 0.3}
        composite = CompositeEvaluator(
            name="weighted_composite",
            sub_evaluators=[self.sub_evaluator1, self.sub_evaluator2],
            weights=weights
        )
        
        assert composite.weights == weights
    
    def test_evaluate(self):
        """测试复合评估"""
        result = self.composite.evaluate(
            input_text="测试输入",
            model_output="测试输出",
            expected_output="测试输出",
            context={}
        )
        
        assert isinstance(result, EvaluationScore)
        assert result.overall_score == 1.0  # 两个子评估器都返回1.0
        assert "sub_evaluator_1" in result.dimension_scores
        assert "sub_evaluator_2" in result.dimension_scores
        assert "sub_evaluations" in result.details
    
    def test_get_evaluation_criteria(self):
        """测试获取评估标准"""
        criteria = self.composite.get_evaluation_criteria()
        assert len(criteria) == 2  # 两个子评估器各一个标准


class TestRuleBasedEvaluator:
    """基于规则的评估器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.rules = [
            {
                "type": "exact_match",
                "description": "精确匹配规则",
                "weight": 1.0
            },
            {
                "type": "contains",
                "keywords": ["关键词1", "关键词2"],
                "description": "关键词包含规则",
                "weight": 0.5
            },
            {
                "type": "length_check",
                "min_length": 5,
                "max_length": 100,
                "description": "长度检查规则",
                "weight": 0.3
            }
        ]
        
        self.evaluator = RuleBasedEvaluator("rule_evaluator", self.rules)
    
    def test_initialization(self):
        """测试初始化"""
        assert self.evaluator.name == "rule_evaluator"
        assert len(self.evaluator.rules) == 3
        assert len(self.evaluator.criteria) == 3
    
    def test_exact_match_rule(self):
        """测试精确匹配规则"""
        # 完全匹配
        score = self.evaluator._apply_rule(
            self.rules[0], "输入", "输出", "输出", {}
        )
        assert score == 1.0
        
        # 不匹配
        score = self.evaluator._apply_rule(
            self.rules[0], "输入", "输出1", "输出2", {}
        )
        assert score == 0.0
    
    def test_contains_rule(self):
        """测试包含规则"""
        # 包含所有关键词
        score = self.evaluator._apply_rule(
            self.rules[1], "输入", "这里有关键词1和关键词2", "期望", {}
        )
        assert score == 1.0
        
        # 包含部分关键词
        score = self.evaluator._apply_rule(
            self.rules[1], "输入", "这里只有关键词1", "期望", {}
        )
        assert score == 0.5
        
        # 不包含关键词
        score = self.evaluator._apply_rule(
            self.rules[1], "输入", "这里没有关键词", "期望", {}
        )
        assert score == 0.0
    
    def test_length_check_rule(self):
        """测试长度检查规则"""
        # 长度在范围内
        score = self.evaluator._apply_rule(
            self.rules[2], "输入", "合适长度的输出", "期望", {}
        )
        assert score == 1.0
        
        # 长度过短
        score = self.evaluator._apply_rule(
            self.rules[2], "输入", "短", "期望", {}
        )
        assert score == 0.0
        
        # 长度过长
        long_output = "x" * 101
        score = self.evaluator._apply_rule(
            self.rules[2], "输入", long_output, "期望", {}
        )
        assert score == 0.0
    
    def test_regex_match_rule(self):
        """测试正则匹配规则"""
        regex_rule = {
            "type": "regex_match",
            "pattern": r"\d+",
            "description": "数字匹配规则"
        }
        
        # 包含数字
        score = self.evaluator._apply_rule(
            regex_rule, "输入", "输出包含123数字", "期望", {}
        )
        assert score == 1.0
        
        # 不包含数字
        score = self.evaluator._apply_rule(
            regex_rule, "输入", "输出不包含数字", "期望", {}
        )
        assert score == 0.0
    
    def test_unknown_rule_type(self):
        """测试未知规则类型"""
        unknown_rule = {
            "type": "unknown_type",
            "description": "未知规则类型"
        }
        
        score = self.evaluator._apply_rule(
            unknown_rule, "输入", "输出", "期望", {}
        )
        assert score == 0.0
    
    def test_evaluate(self):
        """测试规则评估"""
        result = self.evaluator.evaluate(
            input_text="测试输入",
            model_output="测试输出包含关键词1",
            expected_output="测试输出",
            context={}
        )
        
        assert isinstance(result, EvaluationScore)
        assert 0 <= result.overall_score <= 1
    
    def test_add_rule(self):
        """测试添加规则"""
        original_count = len(self.evaluator.rules)
        
        new_rule = {
            "type": "exact_match",
            "description": "新规则"
        }
        
        self.evaluator.add_rule(new_rule)
        
        assert len(self.evaluator.rules) == original_count + 1
        assert len(self.evaluator.criteria) == original_count + 1
    
    def test_remove_rule(self):
        """测试移除规则"""
        original_count = len(self.evaluator.rules)
        
        self.evaluator.remove_rule(0)
        
        assert len(self.evaluator.rules) == original_count - 1
        assert len(self.evaluator.criteria) == original_count - 1
    
    def test_remove_rule_invalid_index(self):
        """测试移除无效索引的规则"""
        original_count = len(self.evaluator.rules)
        
        # 无效索引不应该改变规则数量
        self.evaluator.remove_rule(-1)
        assert len(self.evaluator.rules) == original_count
        
        self.evaluator.remove_rule(100)
        assert len(self.evaluator.rules) == original_count


if __name__ == "__main__":
    pytest.main([__file__])
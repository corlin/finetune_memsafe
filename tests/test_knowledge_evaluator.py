"""
专业知识评估器单元测试
"""

import pytest
from industry_evaluation.evaluators.knowledge_evaluator import (
    KnowledgeGraphMatcher, KnowledgeEvaluator
)


class TestKnowledgeGraphMatcher:
    """知识图谱匹配器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.matcher = KnowledgeGraphMatcher()
    
    def test_extract_concepts(self):
        """测试概念提取"""
        text = "机器学习是人工智能的重要分支，深度学习是机器学习的子领域"
        concepts = self.matcher.extract_concepts(text)
        
        assert "机器学习" in concepts
        assert "深度学习" in concepts
        assert len(concepts) >= 2
    
    def test_extract_concepts_with_synonyms(self):
        """测试同义词概念提取"""
        text = "ML算法在NLP任务中表现出色"
        concepts = self.matcher.extract_concepts(text)
        
        # ML是机器学习的同义词，NLP是自然语言处理的同义词
        assert "机器学习" in concepts
        assert "自然语言处理" in concepts
    
    def test_match_concept(self):
        """测试概念匹配"""
        # 精确匹配
        assert self.matcher._match_concept("机器学习", "机器学习很重要")
        
        # 词边界匹配
        assert not self.matcher._match_concept("学习", "机器学习很重要")  # 避免部分匹配
        
        # 大小写不敏感
        assert self.matcher._match_concept("ML", "ml算法很强大")
    
    def test_verify_concept_relations(self):
        """测试概念关系验证"""
        concepts = {"机器学习", "深度学习"}
        text = "深度学习是机器学习的子领域"
        
        result = self.matcher.verify_concept_relations(concepts, text)
        
        assert "relation_score" in result
        assert "correct_relations" in result
        assert "incorrect_relations" in result
        assert isinstance(result["relation_score"], float)
    
    def test_detect_relations_in_text(self):
        """测试文本中关系检测"""
        concepts = {"深度学习", "机器学习"}
        text = "深度学习是机器学习的一种方法"
        
        relations = self.matcher._detect_relations_in_text(concepts, text)
        
        assert len(relations) > 0
        # 应该检测到"是"关系
        is_a_relations = [r for r in relations if r["type"] == "is_a"]
        assert len(is_a_relations) > 0
    
    def test_check_fact_consistency(self):
        """测试事实一致性检查"""
        # 一致的文本
        consistent_text = "深度学习是机器学习的子领域，监督学习需要标注数据"
        result = self.matcher.check_fact_consistency(consistent_text)
        
        assert result["consistency_score"] > 0.5
        assert len(result["consistent_facts"]) > 0
        
        # 不一致的文本
        inconsistent_text = "深度学习不是机器学习的方法"
        result = self.matcher.check_fact_consistency(inconsistent_text)
        
        # 应该检测到矛盾
        if result["inconsistent_facts"]:
            assert result["consistency_score"] < 1.0
    
    def test_check_single_fact(self):
        """测试单个事实检查"""
        fact = {
            "statement": "深度学习是机器学习的子领域",
            "keywords": ["深度学习", "机器学习"],
            "contradiction_patterns": [r"深度学习\s*不是\s*机器学习"]
        }
        
        # 一致的文本
        consistent_text = "深度学习是机器学习的重要分支"
        result = self.matcher._check_single_fact(fact, consistent_text)
        
        assert result["mentioned"] is True
        assert result["consistent"] is True
        
        # 矛盾的文本
        contradictory_text = "深度学习不是机器学习的方法"
        result = self.matcher._check_single_fact(fact, contradictory_text)
        
        assert result["mentioned"] is True
        assert result["consistent"] is False
        assert result["contradiction"] is not None
    
    def test_is_valid_relation(self):
        """测试关系有效性验证"""
        # 已知有效关系
        valid_relation = {
            "type": "is_a",
            "subject": "深度学习",
            "object": "机器学习"
        }
        
        # 这里需要根据实际的关系验证逻辑进行测试
        # 由于_is_valid_relation方法依赖于知识库，我们测试基本功能
        result = self.matcher._is_valid_relation(valid_relation)
        assert isinstance(result, bool)
    
    def test_get_expected_relations(self):
        """测试获取预期关系"""
        concepts = {"机器学习", "深度学习"}
        relations = self.matcher._get_expected_relations(concepts)
        
        assert isinstance(relations, list)
        # 如果有预期关系，每个关系应该包含必要字段
        for relation in relations:
            assert "subject" in relation
            assert "object" in relation
            assert "type" in relation


class TestKnowledgeEvaluator:
    """专业知识评估器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.evaluator = KnowledgeEvaluator()
    
    def test_initialization(self):
        """测试初始化"""
        assert self.evaluator.name == "knowledge"
        assert self.evaluator.weight == 1.0
        assert len(self.evaluator.criteria) == 3
        
        # 检查评估标准
        criteria_names = [c.name for c in self.evaluator.criteria]
        assert "concept_accuracy" in criteria_names
        assert "relation_correctness" in criteria_names
        assert "fact_consistency" in criteria_names
    
    def test_evaluate_basic(self):
        """测试基本评估功能"""
        input_text = "什么是机器学习？"
        model_output = "机器学习是人工智能的一个分支，深度学习是机器学习的子领域"
        expected_output = "机器学习是AI的重要技术，包含监督学习等方法"
        
        result = self.evaluator.evaluate(input_text, model_output, expected_output, {})
        
        assert 0 <= result.overall_score <= 1
        assert self.evaluator.name in result.dimension_scores
        assert result.confidence > 0
        assert "extracted_concepts" in result.details
    
    def test_evaluate_concept_accuracy(self):
        """测试概念准确性评估"""
        output_concepts = {"机器学习", "深度学习"}
        expected_concepts = {"机器学习", "自然语言处理"}
        
        score = self.evaluator._evaluate_concept_accuracy(output_concepts, expected_concepts)
        
        assert 0 <= score <= 1
        # 有一个共同概念（机器学习），应该有一定分数
        assert score > 0
    
    def test_evaluate_concept_accuracy_empty_expected(self):
        """测试期望概念为空的情况"""
        output_concepts = {"机器学习", "深度学习"}
        expected_concepts = set()
        
        score = self.evaluator._evaluate_concept_accuracy(output_concepts, expected_concepts)
        
        assert 0 <= score <= 1
        # 应该基于概念有效性评分
        assert score > 0  # 因为输出包含有效概念
    
    def test_evaluate_concept_validity(self):
        """测试概念有效性评估"""
        # 有效概念
        valid_concepts = {"机器学习", "深度学习"}
        score = self.evaluator._evaluate_concept_validity(valid_concepts)
        assert score == 1.0  # 所有概念都有效
        
        # 混合概念
        mixed_concepts = {"机器学习", "无效概念"}
        score = self.evaluator._evaluate_concept_validity(mixed_concepts)
        assert score == 0.5  # 一半有效
        
        # 空概念
        empty_concepts = set()
        score = self.evaluator._evaluate_concept_validity(empty_concepts)
        assert score == 0.5  # 默认中等分数
    
    def test_calculate_weighted_concept_score(self):
        """测试加权概念分数计算"""
        output_concepts = {"机器学习", "深度学习"}
        expected_concepts = {"机器学习", "自然语言处理"}
        intersection = {"机器学习"}
        
        score = self.evaluator._calculate_weighted_concept_score(
            output_concepts, expected_concepts, intersection
        )
        
        assert 0 <= score <= 1
        # 机器学习是核心概念，应该有较高权重
        assert score > 0
    
    def test_calculate_score(self):
        """测试分数计算"""
        input_text = "解释机器学习"
        model_output = "机器学习是AI技术，深度学习是机器学习的子领域"
        expected_output = "机器学习是人工智能的分支"
        
        score = self.evaluator._calculate_score(input_text, model_output, expected_output, {})
        
        assert 0 <= score <= 1
        # 应该有一定分数，因为包含相关概念
        assert score > 0
    
    def test_generate_details(self):
        """测试详细信息生成"""
        input_text = "什么是机器学习？"
        model_output = "机器学习是AI的分支，深度学习是其子领域"
        expected_output = "机器学习是人工智能技术"
        
        details = self.evaluator._generate_details(
            input_text, model_output, expected_output, {}, 0.8
        )
        
        assert details["evaluator"] == "knowledge"
        assert details["score"] == 0.8
        assert "extracted_concepts" in details
        assert "expected_concepts" in details
        assert "concept_accuracy" in details
        assert "relation_verification" in details
        assert "fact_consistency" in details
        assert "missing_concepts" in details
        assert "extra_concepts" in details
    
    def test_custom_knowledge_base(self):
        """测试自定义知识库"""
        custom_kb = {
            "concepts": {
                "测试概念": {
                    "type": "测试类型",
                    "synonyms": ["测试同义词"],
                    "properties": ["测试属性"]
                }
            },
            "relations": {},
            "facts": []
        }
        
        evaluator = KnowledgeEvaluator(knowledge_base=custom_kb)
        
        # 测试是否使用了自定义知识库
        concepts = evaluator.knowledge_matcher.extract_concepts("这是测试概念")
        assert "测试概念" in concepts
        
        # 测试同义词
        concepts = evaluator.knowledge_matcher.extract_concepts("这是测试同义词")
        assert "测试概念" in concepts
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空文本
        result = self.evaluator.evaluate("", "", "", {})
        assert 0 <= result.overall_score <= 1
        
        # 很长的文本
        long_text = "机器学习 " * 1000
        result = self.evaluator.evaluate("输入", long_text, "期望", {})
        assert 0 <= result.overall_score <= 1
        
        # 特殊字符
        special_text = "机器学习@#$%^&*()"
        result = self.evaluator.evaluate("输入", special_text, "期望", {})
        assert 0 <= result.overall_score <= 1


if __name__ == "__main__":
    pytest.main([__file__])
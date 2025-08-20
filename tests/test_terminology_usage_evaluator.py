"""
术语使用评估功能单元测试
"""

import pytest
from industry_evaluation.evaluators.terminology_usage_evaluator import (
    ContextAnalyzer, UsagePatternAnalyzer, TerminologyUsageEvaluator
)


class TestContextAnalyzer:
    """上下文分析器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.analyzer = ContextAnalyzer()
    
    def test_analyze_term_context(self):
        """测试术语上下文分析"""
        text = "机器学习是一种人工智能技术，它可以从数据中学习模式。深度学习是机器学习的子领域。"
        term = "机器学习"
        position = (0, 4)  # "机器学习"的位置
        
        analysis = self.analyzer.analyze_term_context(term, text, position)
        
        assert "local_context" in analysis
        assert "sentence_context" in analysis
        assert "paragraph_context" in analysis
        assert "grammatical_role" in analysis
        assert "semantic_relations" in analysis
        assert "modifiers" in analysis
        assert "co_occurring_terms" in analysis
        assert "context_type" in analysis
        
        # 检查上下文内容
        assert term in analysis["local_context"]
        assert term in analysis["sentence_context"]
    
    def test_extract_local_context(self):
        """测试局部上下文提取"""
        text = "这是一个关于机器学习的测试文本，用于验证上下文提取功能。"
        start_pos = 7
        end_pos = 11
        
        context = self.analyzer._extract_local_context(text, start_pos, end_pos, window=10)
        
        assert len(context) > 0
        assert "机器学习" in context
    
    def test_extract_sentence_context(self):
        """测试句子上下文提取"""
        text = "第一句话。机器学习是重要技术。第三句话。"
        start_pos = 4
        end_pos = 8
        
        sentence = self.analyzer._extract_sentence_context(text, start_pos, end_pos)
        
        assert "机器学习是重要技术" in sentence
        assert "第一句话" not in sentence
        assert "第三句话" not in sentence
    
    def test_analyze_grammatical_role(self):
        """测试语法角色分析"""
        # 主语
        context = "机器学习是重要技术"
        role = self.analyzer._analyze_grammatical_role("机器学习", context, 0)
        assert role == "subject"
        
        # 宾语
        context = "我们使用机器学习来解决问题"
        role = self.analyzer._analyze_grammatical_role("机器学习", context, 3)
        assert role == "object"
        
        # 定语
        context = "机器学习算法很强大"
        role = self.analyzer._analyze_grammatical_role("机器学习", context, 0)
        assert role == "modifier"
    
    def test_analyze_semantic_relations(self):
        """测试语义关系分析"""
        sentence = "机器学习是一种人工智能技术，用于从数据中学习模式"
        term = "机器学习"
        
        relations = self.analyzer._analyze_semantic_relations(term, sentence)
        
        assert isinstance(relations, list)
        # 应该识别出定义关系
        definition_relations = [r for r in relations if r["type"] == "definition"]
        assert len(definition_relations) > 0
    
    def test_extract_modifiers(self):
        """测试修饰词提取"""
        context = "先进的机器学习算法"
        term_pos = 3
        
        modifiers = self.analyzer._extract_modifiers("机器学习", context, term_pos)
        
        assert "adjectives" in modifiers
        assert "adverbs" in modifiers
        assert "quantifiers" in modifiers
        assert "先进的" in modifiers["adjectives"]
    
    def test_find_co_occurring_terms(self):
        """测试共现术语查找"""
        sentence = "机器学习和深度学习都是人工智能技术"
        current_term = "机器学习"
        
        co_occurring = self.analyzer._find_co_occurring_terms(sentence, current_term)
        
        assert "深度学习" in co_occurring
        assert "人工智能" in co_occurring
        assert current_term not in co_occurring
    
    def test_classify_context_type(self):
        """测试上下文类型分类"""
        # 定义类型
        sentence = "机器学习是一种人工智能技术"
        context_type = self.analyzer._classify_context_type(sentence)
        assert context_type == "definition"
        
        # 应用类型
        sentence = "机器学习用于数据分析"
        context_type = self.analyzer._classify_context_type(sentence)
        assert context_type == "application"
        
        # 比较类型
        sentence = "机器学习与传统方法相比更有效"
        context_type = self.analyzer._classify_context_type(sentence)
        assert context_type == "comparison"


class TestUsagePatternAnalyzer:
    """使用模式分析器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.analyzer = UsagePatternAnalyzer()
    
    def test_analyze_usage_patterns(self):
        """测试使用模式分析"""
        terms = [
            {"standard_form": "机器学习", "start_pos": 0, "end_pos": 4, "matched_text": "机器学习"},
            {"standard_form": "机器学习", "start_pos": 20, "end_pos": 24, "matched_text": "ML"},
            {"standard_form": "深度学习", "start_pos": 30, "end_pos": 34, "matched_text": "深度学习"}
        ]
        text = "机器学习是重要技术。使用ML算法。深度学习是ML的子领域。"
        
        patterns = self.analyzer.analyze_usage_patterns(terms, text)
        
        assert "frequency_patterns" in patterns
        assert "position_patterns" in patterns
        assert "co_occurrence_patterns" in patterns
        assert "context_patterns" in patterns
        assert "consistency_patterns" in patterns
        assert "evolution_patterns" in patterns
    
    def test_analyze_frequency_patterns(self):
        """测试频率模式分析"""
        terms = [
            {"standard_form": "机器学习"},
            {"standard_form": "机器学习"},
            {"standard_form": "深度学习"}
        ]
        
        patterns = self.analyzer._analyze_frequency_patterns(terms)
        
        assert "term_frequency" in patterns
        assert "most_frequent" in patterns
        assert "frequency_distribution" in patterns
        assert "repetition_rate" in patterns
        
        assert patterns["term_frequency"]["机器学习"] == 2
        assert patterns["term_frequency"]["深度学习"] == 1
        assert patterns["repetition_rate"] == 1.5  # 3个术语，2个唯一
    
    def test_analyze_position_patterns(self):
        """测试位置模式分析"""
        terms = [
            {"standard_form": "机器学习", "start_pos": 10},
            {"standard_form": "深度学习", "start_pos": 50},
            {"standard_form": "人工智能", "start_pos": 90}
        ]
        text = "a" * 100  # 100字符的文本
        
        patterns = self.analyzer._analyze_position_patterns(terms, text)
        
        assert "position_distribution" in patterns
        assert "section_usage" in patterns
        assert "clustering" in patterns
        
        # 检查段落分布
        distribution = patterns["position_distribution"]
        assert "beginning" in distribution
        assert "middle" in distribution
        assert "end" in distribution
    
    def test_analyze_consistency_patterns(self):
        """测试一致性模式分析"""
        terms = [
            {"standard_form": "机器学习", "matched_text": "机器学习"},
            {"standard_form": "机器学习", "matched_text": "ML"},
            {"standard_form": "机器学习", "matched_text": "机学"},
            {"standard_form": "深度学习", "matched_text": "深度学习"}
        ]
        
        patterns = self.analyzer._analyze_consistency_patterns(terms)
        
        assert "term_variations" in patterns
        assert "consistency_scores" in patterns
        assert "overall_consistency" in patterns
        assert "inconsistent_terms" in patterns
        
        # 机器学习有3种变体，一致性应该较低
        assert patterns["consistency_scores"]["机器学习"] == 1.0 / 3
        # 深度学习只有1种变体，一致性应该为1
        assert patterns["consistency_scores"]["深度学习"] == 1.0
    
    def test_classify_text_section(self):
        """测试文本段落分类"""
        assert self.analyzer._classify_text_section(0.1) == "beginning"
        assert self.analyzer._classify_text_section(0.5) == "middle"
        assert self.analyzer._classify_text_section(0.8) == "end"
    
    def test_calculate_repetition_rate(self):
        """测试重复率计算"""
        from collections import Counter
        
        # 3个术语，2个唯一
        term_counts = Counter(["A", "A", "B"])
        rate = self.analyzer._calculate_repetition_rate(term_counts)
        assert rate == 1.5
        
        # 空计数
        empty_counts = Counter()
        rate = self.analyzer._calculate_repetition_rate(empty_counts)
        assert rate == 0.0


class TestTerminologyUsageEvaluator:
    """术语使用评估器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.evaluator = TerminologyUsageEvaluator()
    
    def test_initialization(self):
        """测试初始化"""
        assert self.evaluator.name == "terminology_usage"
        assert self.evaluator.weight == 1.0
        assert len(self.evaluator.criteria) == 3
        
        # 检查评估标准
        criteria_names = [c.name for c in self.evaluator.criteria]
        assert "contextual_appropriateness" in criteria_names
        assert "usage_consistency" in criteria_names
        assert "pattern_quality" in criteria_names
    
    def test_evaluate_basic(self):
        """测试基本评估功能"""
        input_text = "解释机器学习"
        model_output = "机器学习是一种人工智能技术，深度学习是机器学习的子领域。机器学习算法可以从数据中学习。"
        expected_output = "机器学习是AI技术"
        
        result = self.evaluator.evaluate(input_text, model_output, expected_output, {})
        
        assert 0 <= result.overall_score <= 1
        assert self.evaluator.name in result.dimension_scores
        assert result.confidence > 0
        assert "recognized_terms" in result.details
        assert "usage_patterns" in result.details
    
    def test_evaluate_contextual_appropriateness(self):
        """测试上下文适当性评估"""
        terms = [
            {"standard_form": "机器学习", "start_pos": 0, "end_pos": 4},
            {"standard_form": "深度学习", "start_pos": 20, "end_pos": 24}
        ]
        text = "机器学习是一种AI技术。深度学习是机器学习的子领域。"
        
        score = self.evaluator._evaluate_contextual_appropriateness(terms, text)
        
        assert 0 <= score <= 1
        # 应该有较高分数，因为上下文合适
        assert score > 0.5
    
    def test_assess_context_quality(self):
        """测试上下文质量评估"""
        context_analysis = {
            "context_type": "definition",
            "grammatical_role": "subject",
            "semantic_relations": [{"type": "definition", "target": "AI技术"}],
            "modifiers": {"adjectives": ["先进的"], "adverbs": [], "quantifiers": []},
            "co_occurring_terms": ["人工智能", "算法"]
        }
        
        quality = self.evaluator._assess_context_quality("机器学习", context_analysis)
        
        assert 0 <= quality <= 1
        # 定义类型的上下文应该有高分
        assert quality > 0.8
    
    def test_evaluate_usage_consistency(self):
        """测试使用一致性评估"""
        terms = [
            {"standard_form": "机器学习", "matched_text": "机器学习"},
            {"standard_form": "机器学习", "matched_text": "机器学习"},
            {"standard_form": "深度学习", "matched_text": "深度学习"}
        ]
        text = "测试文本"
        
        score = self.evaluator._evaluate_usage_consistency(terms, text)
        
        assert 0 <= score <= 1
        # 一致的术语使用应该有高分
        assert score == 1.0
    
    def test_evaluate_pattern_quality(self):
        """测试使用模式质量评估"""
        terms = [
            {"standard_form": "机器学习", "start_pos": 0, "end_pos": 4, "matched_text": "机器学习"},
            {"standard_form": "深度学习", "start_pos": 20, "end_pos": 24, "matched_text": "深度学习"}
        ]
        text = "机器学习是重要技术。深度学习是机器学习的子领域。"
        
        score = self.evaluator._evaluate_pattern_quality(terms, text)
        
        assert 0 <= score <= 1
        # 应该有合理的模式质量分数
        assert score > 0.3
    
    def test_assess_frequency_quality(self):
        """测试频率质量评估"""
        # 理想频率
        ideal_patterns = {"repetition_rate": 2.0}
        score = self.evaluator._assess_frequency_quality(ideal_patterns)
        assert score == 1.0
        
        # 过低频率
        low_patterns = {"repetition_rate": 1.0}
        score = self.evaluator._assess_frequency_quality(low_patterns)
        assert score < 1.0
        
        # 过高频率
        high_patterns = {"repetition_rate": 5.0}
        score = self.evaluator._assess_frequency_quality(high_patterns)
        assert score < 1.0
    
    def test_assess_position_quality(self):
        """测试位置质量评估"""
        # 理想聚类
        ideal_patterns = {"clustering": {"clustering_score": 0.6}}
        score = self.evaluator._assess_position_quality(ideal_patterns)
        assert score == 1.0
        
        # 过于分散
        scattered_patterns = {"clustering": {"clustering_score": 0.2}}
        score = self.evaluator._assess_position_quality(scattered_patterns)
        assert score < 1.0
        
        # 过于聚集
        clustered_patterns = {"clustering": {"clustering_score": 0.9}}
        score = self.evaluator._assess_position_quality(clustered_patterns)
        assert score < 1.0
    
    def test_calculate_score(self):
        """测试分数计算"""
        input_text = "解释术语使用"
        model_output = "机器学习是AI技术，深度学习是机器学习的方法。机器学习算法很重要。"
        expected_output = "机器学习是人工智能技术"
        
        score = self.evaluator._calculate_score(input_text, model_output, expected_output, {})
        
        assert 0 <= score <= 1
        # 应该有一定分数，因为包含合理的术语使用
        assert score > 0.3
    
    def test_calculate_score_no_terms(self):
        """测试无术语情况的分数计算"""
        input_text = "简单问题"
        model_output = "这是一个简单的回答"
        expected_output = "简单回答"
        
        score = self.evaluator._calculate_score(input_text, model_output, expected_output, {})
        
        assert score == 0.5  # 没有术语，应该返回中等分数
    
    def test_generate_details(self):
        """测试详细信息生成"""
        input_text = "输入"
        model_output = "机器学习是AI技术，深度学习是机器学习的子领域"
        expected_output = "期望"
        
        details = self.evaluator._generate_details(
            input_text, model_output, expected_output, {}, 0.8
        )
        
        assert details["evaluator"] == "terminology_usage"
        assert details["score"] == 0.8
        assert "recognized_terms" in details
        assert "term_contexts" in details
        assert "usage_patterns" in details
        assert "context_score" in details
        assert "consistency_score" in details
        assert "pattern_score" in details
        assert "quality_assessment" in details
        assert "recommendations" in details
        
        # 检查质量评估
        quality = details["quality_assessment"]
        assert "contextual_appropriateness" in quality
        assert "usage_consistency" in quality
        assert "pattern_quality" in quality
    
    def test_generate_usage_recommendations(self):
        """测试使用建议生成"""
        # 模拟使用模式
        usage_patterns = {
            "consistency_patterns": {
                "overall_consistency": 0.5,
                "inconsistent_terms": ["机器学习", "深度学习"]
            },
            "frequency_patterns": {
                "repetition_rate": 5.0
            },
            "position_patterns": {
                "clustering": {"clustering_score": 0.95}
            }
        }
        
        # 模拟术语上下文
        term_contexts = [
            {"context_quality": 0.4, "term": "机器学习"},
            {"context_quality": 0.8, "term": "深度学习"}
        ]
        
        recommendations = self.evaluator._generate_usage_recommendations(usage_patterns, term_contexts)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        
        # 应该包含针对一致性、频率、聚类等问题的建议
        rec_text = " ".join(recommendations)
        assert any(keyword in rec_text for keyword in ["统一", "频繁", "集中", "上下文"])
    
    def test_custom_dictionary(self):
        """测试自定义词典"""
        custom_dict = {
            "terms": {
                "自定义术语": {
                    "definition": "这是自定义术语",
                    "category": "自定义类别"
                }
            },
            "synonyms": {
                "自定义术语": ["自定义同义词"]
            },
            "categories": {
                "自定义类别": ["自定义术语"]
            },
            "contexts": {}
        }
        
        evaluator = TerminologyUsageEvaluator(dictionary_data=custom_dict)
        
        # 测试自定义术语的使用评估
        result = evaluator.evaluate(
            "输入", 
            "这里使用了自定义术语，自定义术语很重要", 
            "期望", 
            {}
        )
        
        assert 0 <= result.overall_score <= 1
        assert result.details["recognized_terms"]
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空文本
        result = self.evaluator.evaluate("", "", "", {})
        assert 0 <= result.overall_score <= 1
        
        # 无术语文本
        result = self.evaluator.evaluate("输入", "简单文本", "期望", {})
        assert result.overall_score == 0.5
        
        # 大量重复术语
        repetitive_text = "机器学习 " * 20
        result = self.evaluator.evaluate("输入", repetitive_text, "期望", {})
        assert 0 <= result.overall_score <= 1
        
        # 很长的文本
        long_text = "机器学习是重要技术。" * 100
        result = self.evaluator.evaluate("输入", long_text, "期望", {})
        assert 0 <= result.overall_score <= 1


if __name__ == "__main__":
    pytest.main([__file__])
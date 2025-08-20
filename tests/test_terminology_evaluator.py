"""
术语准确性评估器单元测试
"""

import pytest
from industry_evaluation.evaluators.terminology_evaluator import (
    TerminologyDictionary, TerminologyRecognizer, TerminologyEvaluator
)


class TestTerminologyDictionary:
    """术语词典测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.dictionary = TerminologyDictionary()
    
    def test_initialization(self):
        """测试初始化"""
        assert len(self.dictionary.terms) > 0
        assert len(self.dictionary.synonyms) > 0
        assert len(self.dictionary.categories) > 0
        assert len(self.dictionary.synonym_to_term) > 0
        assert len(self.dictionary.term_to_category) > 0
    
    def test_get_standard_term(self):
        """测试获取标准术语"""
        # 标准术语
        assert self.dictionary.get_standard_term("机器学习") == "机器学习"
        
        # 同义词
        assert self.dictionary.get_standard_term("ML") == "机器学习"
        assert self.dictionary.get_standard_term("DL") == "深度学习"
        
        # 大小写不敏感
        assert self.dictionary.get_standard_term("ml") == "机器学习"
        
        # 未知术语
        unknown_term = "未知术语"
        assert self.dictionary.get_standard_term(unknown_term) == unknown_term
    
    def test_is_valid_term(self):
        """测试术语有效性检查"""
        # 有效术语
        assert self.dictionary.is_valid_term("机器学习") is True
        assert self.dictionary.is_valid_term("ML") is True  # 同义词
        
        # 无效术语
        assert self.dictionary.is_valid_term("无效术语") is False
    
    def test_get_term_info(self):
        """测试获取术语信息"""
        # 有效术语
        info = self.dictionary.get_term_info("机器学习")
        assert info["exists"] is True
        assert info["standard_form"] == "机器学习"
        assert "definition" in info
        assert "category" in info
        assert "synonyms" in info
        
        # 同义词
        info = self.dictionary.get_term_info("ML")
        assert info["exists"] is True
        assert info["standard_form"] == "机器学习"
        
        # 无效术语
        info = self.dictionary.get_term_info("无效术语")
        assert info["exists"] is False
    
    def test_get_terms_by_category(self):
        """测试按类别获取术语"""
        ai_terms = self.dictionary.get_terms_by_category("AI技术")
        assert len(ai_terms) > 0
        assert "机器学习" in ai_terms
        assert "深度学习" in ai_terms
        
        # 不存在的类别
        empty_terms = self.dictionary.get_terms_by_category("不存在的类别")
        assert len(empty_terms) == 0
    
    def test_add_term(self):
        """测试添加术语"""
        new_term = "测试术语"
        definition = "这是一个测试术语"
        category = "测试类别"
        synonyms = ["测试同义词"]
        
        self.dictionary.add_term(new_term, definition, category, synonyms)
        
        # 验证术语已添加
        assert self.dictionary.is_valid_term(new_term) is True
        assert self.dictionary.is_valid_term("测试同义词") is True
        
        # 验证术语信息
        info = self.dictionary.get_term_info(new_term)
        assert info["definition"] == definition
        assert info["category"] == category
        
        # 验证类别更新
        category_terms = self.dictionary.get_terms_by_category(category)
        assert new_term in category_terms
    
    def test_fuzzy_match(self):
        """测试模糊匹配"""
        # 相似术语
        assert self.dictionary._fuzzy_match("机器学习", "机器学习", 0.8) is True
        assert self.dictionary._fuzzy_match("机器学习", "机器学", 0.8) is True
        
        # 不相似术语
        assert self.dictionary._fuzzy_match("机器学习", "深度学习", 0.8) is False
        
        # 空字符串
        assert self.dictionary._fuzzy_match("", "", 0.8) is True
        assert self.dictionary._fuzzy_match("机器学习", "", 0.8) is False


class TestTerminologyRecognizer:
    """术语识别器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.dictionary = TerminologyDictionary()
        self.recognizer = TerminologyRecognizer(self.dictionary)
    
    def test_recognize_terms_exact_match(self):
        """测试精确匹配术语识别"""
        text = "机器学习是人工智能的重要分支，深度学习是机器学习的子领域"
        
        terms = self.recognizer.recognize_terms(text)
        
        assert len(terms) > 0
        
        # 检查识别的术语
        recognized_terms = {term["standard_form"] for term in terms}
        assert "机器学习" in recognized_terms
        assert "深度学习" in recognized_terms
        
        # 检查匹配类型
        exact_matches = [t for t in terms if t["match_type"] == "exact"]
        assert len(exact_matches) > 0
    
    def test_recognize_terms_synonym_match(self):
        """测试同义词匹配"""
        text = "ML算法在NLP任务中表现出色"
        
        terms = self.recognizer.recognize_terms(text)
        
        # 应该识别出同义词对应的标准术语
        recognized_terms = {term["standard_form"] for term in terms}
        assert "机器学习" in recognized_terms  # ML的标准形式
        assert "自然语言处理" in recognized_terms  # NLP的标准形式
        
        # 检查匹配类型
        synonym_matches = [t for t in terms if t["match_type"] == "synonym"]
        assert len(synonym_matches) > 0
    
    def test_recognize_terms_fuzzy_match(self):
        """测试模糊匹配"""
        text = "机器学习算法和深度学习网络"
        
        terms = self.recognizer.recognize_terms(text)
        
        # 应该识别出术语
        recognized_terms = {term["standard_form"] for term in terms}
        assert "机器学习" in recognized_terms
        assert "深度学习" in recognized_terms
    
    def test_exact_match(self):
        """测试精确匹配方法"""
        text = "机器学习和ML都是重要的AI技术"
        
        matches = self.recognizer._exact_match(text)
        
        assert len(matches) >= 2  # 至少匹配到机器学习和ML
        
        # 检查匹配信息
        for match in matches:
            assert "term" in match
            assert "matched_text" in match
            assert "start_pos" in match
            assert "end_pos" in match
            assert "confidence" in match
            assert match["confidence"] > 0
    
    def test_extract_candidates(self):
        """测试候选术语提取"""
        text = "深度学习DL和机器学习ML是AI技术"
        
        candidates = self.recognizer._extract_candidates(text)
        
        assert len(candidates) > 0
        
        # 检查候选词信息
        for candidate in candidates:
            assert "text" in candidate
            assert "start_pos" in candidate
            assert "end_pos" in candidate
            assert len(candidate["text"]) > 0
    
    def test_deduplicate_and_sort(self):
        """测试去重和排序"""
        matches = [
            {
                "term": "机器学习",
                "start_pos": 0,
                "end_pos": 4,
                "confidence": 0.9,
                "match_type": "exact"
            },
            {
                "term": "机器学习",
                "start_pos": 0,
                "end_pos": 4,
                "confidence": 0.7,  # 较低置信度
                "match_type": "fuzzy"
            },
            {
                "term": "深度学习",
                "start_pos": 10,
                "end_pos": 14,
                "confidence": 0.8,
                "match_type": "exact"
            }
        ]
        
        unique_matches = self.recognizer._deduplicate_and_sort(matches)
        
        assert len(unique_matches) == 2  # 去重后应该只有2个
        
        # 应该保留置信度更高的匹配
        ml_match = next(m for m in unique_matches if m["term"] == "机器学习")
        assert ml_match["confidence"] == 0.9
        
        # 应该按位置排序
        assert unique_matches[0]["start_pos"] <= unique_matches[1]["start_pos"]


class TestTerminologyEvaluator:
    """术语评估器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.evaluator = TerminologyEvaluator()
    
    def test_initialization(self):
        """测试初始化"""
        assert self.evaluator.name == "terminology"
        assert self.evaluator.weight == 1.0
        assert len(self.evaluator.criteria) == 3
        
        # 检查评估标准
        criteria_names = [c.name for c in self.evaluator.criteria]
        assert "term_recognition_accuracy" in criteria_names
        assert "term_usage_correctness" in criteria_names
        assert "term_consistency" in criteria_names
    
    def test_evaluate_basic(self):
        """测试基本评估功能"""
        input_text = "解释机器学习"
        model_output = "机器学习是一种AI技术，深度学习是机器学习的子领域"
        expected_output = "机器学习是人工智能技术，包含监督学习等方法"
        
        result = self.evaluator.evaluate(input_text, model_output, expected_output, {})
        
        assert 0 <= result.overall_score <= 1
        assert self.evaluator.name in result.dimension_scores
        assert result.confidence > 0
        assert "recognized_terms" in result.details
    
    def test_calculate_recognition_score(self):
        """测试术语识别分数计算"""
        output_terms = [
            {"standard_form": "机器学习", "confidence": 0.9},
            {"standard_form": "深度学习", "confidence": 0.8}
        ]
        expected_terms = [
            {"standard_form": "机器学习", "confidence": 1.0},
            {"standard_form": "自然语言处理", "confidence": 1.0}
        ]
        
        score = self.evaluator._calculate_recognition_score(output_terms, expected_terms)
        
        assert 0 <= score <= 1
        # 有一个共同术语（机器学习），应该有一定分数
        assert score > 0
    
    def test_calculate_recognition_score_no_expected(self):
        """测试没有期望术语的识别分数计算"""
        output_terms = [
            {"term": "机器学习", "standard_form": "机器学习"},
            {"term": "无效术语", "standard_form": "无效术语"}
        ]
        expected_terms = []
        
        score = self.evaluator._calculate_recognition_score(output_terms, expected_terms)
        
        assert 0 <= score <= 1
        # 应该基于术语有效性评分
        assert score > 0  # 因为包含有效术语
    
    def test_calculate_validity_score(self):
        """测试术语有效性分数计算"""
        # 全部有效术语
        valid_terms = [
            {"term": "机器学习"},
            {"term": "深度学习"}
        ]
        score = self.evaluator._calculate_validity_score(valid_terms)
        assert score == 1.0
        
        # 部分有效术语
        mixed_terms = [
            {"term": "机器学习"},
            {"term": "无效术语"}
        ]
        score = self.evaluator._calculate_validity_score(mixed_terms)
        assert score == 0.5
        
        # 空术语列表
        score = self.evaluator._calculate_validity_score([])
        assert score == 0.5
    
    def test_extract_context(self):
        """测试上下文提取"""
        text = "机器学习是人工智能的重要分支，在数据分析中应用广泛"
        start_pos = 0
        end_pos = 4
        
        context = self.evaluator._extract_context(text, start_pos, end_pos, window_size=10)
        
        assert len(context) > 0
        assert "机器学习" in context
    
    def test_evaluate_term_usage(self):
        """测试术语使用评估"""
        term = "机器学习"
        context = "机器学习算法可以从数据中学习模式"
        
        score = self.evaluator._evaluate_term_usage(term, context)
        
        assert 0 <= score <= 1
        # 应该有较高分数，因为上下文合适
        assert score > 0.5
    
    def test_calculate_context_similarity(self):
        """测试上下文相似度计算"""
        context1 = "机器学习算法可以从数据中学习"
        context2 = "机器学习在数据分析中应用广泛"
        
        similarity = self.evaluator._calculate_context_similarity(context1, context2)
        
        assert 0 <= similarity <= 1
        # 应该有一定相似度，因为都包含"机器学习"和"数据"
        assert similarity > 0
        
        # 完全相同的上下文
        similarity = self.evaluator._calculate_context_similarity(context1, context1)
        assert similarity == 1.0
        
        # 完全不同的上下文
        context3 = "完全不相关的内容"
        similarity = self.evaluator._calculate_context_similarity(context1, context3)
        assert similarity < 0.5
    
    def test_calculate_consistency_score(self):
        """测试一致性分数计算"""
        # 一致的术语使用
        consistent_terms = [
            {"standard_form": "机器学习", "matched_text": "机器学习"},
            {"standard_form": "机器学习", "matched_text": "机器学习"},
            {"standard_form": "深度学习", "matched_text": "深度学习"}
        ]
        
        score = self.evaluator._calculate_consistency_score(consistent_terms, "文本")
        assert score == 1.0
        
        # 不一致的术语使用
        inconsistent_terms = [
            {"standard_form": "机器学习", "matched_text": "机器学习"},
            {"standard_form": "机器学习", "matched_text": "ML"},
            {"standard_form": "机器学习", "matched_text": "机学"}
        ]
        
        score = self.evaluator._calculate_consistency_score(inconsistent_terms, "文本")
        assert 0 <= score < 1  # 应该有惩罚
        
        # 单个术语
        single_term = [{"standard_form": "机器学习", "matched_text": "机器学习"}]
        score = self.evaluator._calculate_consistency_score(single_term, "文本")
        assert score == 1.0
    
    def test_analyze_term_usage(self):
        """测试术语使用分析"""
        terms = [
            {"standard_form": "机器学习", "confidence": 0.9, "match_type": "exact"},
            {"standard_form": "机器学习", "confidence": 0.8, "match_type": "exact"},
            {"standard_form": "深度学习", "confidence": 0.6, "match_type": "fuzzy"}
        ]
        
        analysis = self.evaluator._analyze_term_usage(terms, "测试文本")
        
        assert "term_frequency" in analysis
        assert "term_categories" in analysis
        assert "usage_patterns" in analysis
        assert "potential_issues" in analysis
        
        # 检查频率统计
        assert analysis["term_frequency"]["机器学习"] == 2
        assert analysis["term_frequency"]["深度学习"] == 1
        
        # 检查潜在问题
        issues = analysis["potential_issues"]
        assert any("置信度较低" in issue for issue in issues)  # 深度学习置信度0.6
        assert any("拼写问题" in issue for issue in issues)    # 深度学习是模糊匹配
    
    def test_calculate_score(self):
        """测试分数计算"""
        input_text = "解释术语"
        model_output = "机器学习是AI技术，深度学习是机器学习的方法"
        expected_output = "机器学习是人工智能技术"
        
        score = self.evaluator._calculate_score(input_text, model_output, expected_output, {})
        
        assert 0 <= score <= 1
        # 应该有一定分数，因为包含有效术语
        assert score > 0
    
    def test_generate_details(self):
        """测试详细信息生成"""
        input_text = "输入"
        model_output = "机器学习和深度学习是重要的AI技术"
        expected_output = "机器学习是人工智能技术"
        
        details = self.evaluator._generate_details(
            input_text, model_output, expected_output, {}, 0.8
        )
        
        assert details["evaluator"] == "terminology"
        assert details["score"] == 0.8
        assert "recognized_terms" in details
        assert "expected_terms" in details
        assert "recognition_score" in details
        assert "usage_score" in details
        assert "consistency_score" in details
        assert "term_analysis" in details
        assert "term_count" in details
        assert "valid_term_count" in details
        assert "invalid_terms" in details
        
        # 检查术语统计
        assert details["term_count"] >= 0
        assert details["valid_term_count"] >= 0
        assert isinstance(details["invalid_terms"], list)
    
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
        
        evaluator = TerminologyEvaluator(dictionary_data=custom_dict)
        
        # 测试自定义术语识别
        text = "这里使用了自定义术语和自定义同义词"
        terms = evaluator.recognizer.recognize_terms(text)
        
        recognized_terms = {term["standard_form"] for term in terms}
        assert "自定义术语" in recognized_terms
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空文本
        result = self.evaluator.evaluate("", "", "", {})
        assert 0 <= result.overall_score <= 1
        
        # 无术语文本
        result = self.evaluator.evaluate("输入", "简单文本", "期望", {})
        assert 0 <= result.overall_score <= 1
        
        # 大量术语文本
        many_terms_text = "机器学习 深度学习 自然语言处理 神经网络 监督学习 无监督学习 强化学习"
        result = self.evaluator.evaluate("输入", many_terms_text, "期望", {})
        assert 0 <= result.overall_score <= 1
        assert result.details["term_count"] > 5


if __name__ == "__main__":
    pytest.main([__file__])
"""
长文本理解评估器单元测试
"""

import pytest
from industry_evaluation.evaluators.long_text_evaluator import (
    TextStructureAnalyzer, KeyInformationExtractor, LongTextEvaluator
)


class TestTextStructureAnalyzer:
    """文本结构分析器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.analyzer = TextStructureAnalyzer()
    
    def test_calculate_basic_stats(self):
        """测试基本统计计算"""
        text = "这是第一段。包含两个句子。\n\n这是第二段。也有内容。"
        
        stats = self.analyzer._calculate_basic_stats(text)
        
        assert stats["char_count"] > 0
        assert stats["word_count"] > 0
        assert stats["sentence_count"] == 4
        assert stats["paragraph_count"] == 2
        assert stats["avg_words_per_sentence"] > 0
        assert stats["avg_sentences_per_paragraph"] == 2.0
        assert 0 <= stats["lexical_diversity"] <= 1
    
    def test_analyze_hierarchical_structure(self):
        """测试层次结构分析"""
        text = """# 主标题
        
        ## 子标题1
        
        1. 第一项
        2. 第二项
        
        - 无序列表项1
        - 无序列表项2
        """
        
        structure = self.analyzer._analyze_hierarchical_structure(text)
        
        assert len(structure["headings"]) >= 2
        assert structure["hierarchy_depth"] >= 2
        assert len(structure["lists"]) >= 1
        
        # 检查标题信息
        main_heading = next((h for h in structure["headings"] if h["level"] == 1), None)
        assert main_heading is not None
        assert "主标题" in main_heading["title"]
    
    def test_analyze_paragraphs(self):
        """测试段落分析"""
        text = """第一段内容比较短。
        
        第二段内容相对较长，包含更多的信息和细节描述。
        
        第三段又回到简短的描述。"""
        
        analysis = self.analyzer._analyze_paragraphs(text)
        
        assert analysis["count"] == 3
        assert len(analysis["lengths"]) == 3
        assert len(analysis["word_counts"]) == 3
        assert analysis["avg_length"] > 0
        assert len(analysis["coherence_scores"]) == 2  # n-1个连贯性分数
    
    def test_analyze_sentences(self):
        """测试句子分析"""
        text = "这是简单句。这是一个包含更多词汇和复杂结构的复合句。短句。"
        
        analysis = self.analyzer._analyze_sentences(text)
        
        assert analysis["count"] == 3
        assert len(analysis["lengths"]) == 3
        assert len(analysis["complexity_scores"]) == 3
        assert analysis["avg_length"] > 0
        assert analysis["avg_complexity"] > 0
    
    def test_identify_discourse_markers(self):
        """测试话语标记识别"""
        text = "首先，我们需要分析问题。然后，制定解决方案。但是，实施过程中遇到困难。因此，需要调整策略。"
        
        markers = self.analyzer._identify_discourse_markers(text)
        
        assert "temporal" in markers  # 首先、然后
        assert "contrast" in markers  # 但是
        assert "causal" in markers    # 因此
        
        # 检查标记统计
        temporal_markers = markers["temporal"]
        assert len(temporal_markers) > 0
        assert any(marker["marker"] == "首先" for marker in temporal_markers)
    
    def test_segment_topics(self):
        """测试主题分割"""
        text = """机器学习是人工智能的重要分支。它通过算法从数据中学习模式。

        深度学习是机器学习的子领域。它使用神经网络进行复杂的模式识别。

        自然语言处理是另一个重要领域。它专注于计算机对人类语言的理解和生成。"""
        
        segments = self.analyzer._segment_topics(text)
        
        assert len(segments) > 0
        assert all("topic_words" in segment for segment in segments)
        assert all("coherence" in segment for segment in segments)
        
        # 检查主题词
        all_topic_words = []
        for segment in segments:
            all_topic_words.extend(segment["topic_words"])
        
        assert any("机器学习" in word or "学习" in word for word in all_topic_words)
    
    def test_analyze_coherence(self):
        """测试连贯性分析"""
        # 连贯的文本
        coherent_text = """机器学习是重要技术。这种技术可以处理大量数据。

        数据处理的结果用于训练模型。训练好的模型可以进行预测。"""
        
        coherence = self.analyzer._analyze_coherence(coherent_text)
        
        assert "overall_coherence" in coherence
        assert "local_coherence" in coherence
        assert "global_coherence" in coherence
        assert 0 <= coherence["overall_coherence"] <= 1
        
        # 单段落文本
        single_para = "这是单个段落。"
        coherence = self.analyzer._analyze_coherence(single_para)
        assert coherence["overall_coherence"] == 1.0
    
    def test_calculate_paragraph_coherence(self):
        """测试段落连贯性计算"""
        para1 = "机器学习算法可以处理数据"
        para2 = "数据处理的结果用于模型训练"
        
        coherence = self.analyzer._calculate_paragraph_coherence(para1, para2)
        
        assert 0 <= coherence <= 1
        # 应该有一定连贯性（都包含"数据"）
        assert coherence > 0
        
        # 完全不相关的段落
        para3 = "天气很好今天"
        coherence = self.analyzer._calculate_paragraph_coherence(para1, para3)
        assert coherence < 0.5
    
    def test_calculate_sentence_complexity(self):
        """测试句子复杂度计算"""
        # 简单句
        simple_sentence = "这是简单句。"
        complexity = self.analyzer._calculate_sentence_complexity(simple_sentence)
        assert 0 <= complexity <= 1
        
        # 复杂句
        complex_sentence = "机器学习算法通过分析大量数据，因为数据质量很重要，所以需要仔细预处理，然后才能进行模型训练和评估。"
        complexity = self.analyzer._calculate_sentence_complexity(complex_sentence)
        assert complexity > 0.3  # 应该有较高复杂度
    
    def test_extract_topic_words(self):
        """测试主题词提取"""
        text = "机器学习算法在数据分析中发挥重要作用，深度学习模型特别有效。"
        
        topic_words = self.analyzer._extract_topic_words(text, top_k=3)
        
        assert len(topic_words) <= 3
        assert all(len(word) > 1 for word in topic_words)
        # 应该包含一些技术术语
        assert any(word in ["机器学习", "算法", "数据", "深度学习", "模型"] for word in topic_words)


class TestKeyInformationExtractor:
    """关键信息提取器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.extractor = KeyInformationExtractor()
    
    def test_extract_main_topics(self):
        """测试主要主题提取"""
        text = """机器学习是人工智能的重要分支。

        深度学习使用神经网络进行学习。

        自然语言处理专注于语言理解。"""
        
        topics = self.extractor._extract_main_topics(text)
        
        assert len(topics) == 3
        assert all("topic_sentence" in topic for topic in topics)
        assert all("topic_words" in topic for topic in topics)
        assert all("confidence" in topic for topic in topics)
        
        # 检查主题内容
        topic_sentences = [topic["topic_sentence"] for topic in topics]
        assert any("机器学习" in sentence for sentence in topic_sentences)
    
    def test_extract_key_entities(self):
        """测试关键实体提取"""
        text = "机器学习算法在Google公司得到广泛应用，TensorFlow系统提供了强大的深度学习平台。"
        
        entities = self.extractor._extract_key_entities(text)
        
        assert "technical_terms" in entities
        assert "organizations" in entities
        assert "products" in entities
        
        # 检查技术术语
        tech_terms = entities["technical_terms"]
        assert any("机器学习" in term for term in tech_terms)
        
        # 检查组织
        orgs = entities["organizations"]
        assert any("Google" in org for org in orgs)
    
    def test_extract_important_facts(self):
        """测试重要事实提取"""
        text = """机器学习准确率达到95%。研究表明深度学习效果显著。
        
        实验证明新算法性能优异。数据显示模型训练时间减少50%。"""
        
        facts = self.extractor._extract_important_facts(text)
        
        assert len(facts) > 0
        assert all("content" in fact for fact in facts)
        assert all("importance_score" in fact for fact in facts)
        assert all("type" in fact for fact in facts)
        
        # 检查重要性排序
        if len(facts) > 1:
            assert facts[0]["importance_score"] >= facts[1]["importance_score"]
    
    def test_extract_conclusions(self):
        """测试结论提取"""
        text = "经过大量实验，因此可以得出机器学习效果很好。总之，深度学习是未来趋势。"
        
        conclusions = self.extractor._extract_conclusions(text)
        
        assert len(conclusions) > 0
        assert any("机器学习效果很好" in conclusion for conclusion in conclusions)
        assert any("深度学习是未来趋势" in conclusion for conclusion in conclusions)
    
    def test_extract_numerical_data(self):
        """测试数值数据提取"""
        text = "模型准确率达到95%，训练时间为2小时，数据集包含10万条记录，成本降低30%。"
        
        numerical_data = self.extractor._extract_numerical_data(text)
        
        assert len(numerical_data) > 0
        assert all("value" in data for data in numerical_data)
        assert all("context" in data for data in numerical_data)
        
        # 检查百分比数据
        percentages = [data for data in numerical_data if data.get("unit") == "%"]
        assert len(percentages) > 0
    
    def test_extract_relationships(self):
        """测试关系提取"""
        text = "数据质量影响模型性能。算法选择取决于问题类型。训练数据与测试结果相关。"
        
        relationships = self.extractor._extract_relationships(text)
        
        assert len(relationships) > 0
        assert all("subject" in rel for rel in relationships)
        assert all("relation" in rel for rel in relationships)
        assert all("object" in rel for rel in relationships)
        
        # 检查关系类型
        relations = [rel["relation"] for rel in relationships]
        assert any(rel in ["影响", "取决于", "相关"] for rel in relations)
    
    def test_extract_temporal_info(self):
        """测试时间信息提取"""
        text = "2023年开始项目，上个月完成训练，明天发布结果。最近性能有所提升。"
        
        temporal_info = self.extractor._extract_temporal_info(text)
        
        assert len(temporal_info) > 0
        assert all("expression" in info for info in temporal_info)
        assert all("type" in info for info in temporal_info)
        
        # 检查时间类型
        types = [info["type"] for info in temporal_info]
        assert "year" in types or "relative_period" in types
    
    def test_calculate_topic_confidence(self):
        """测试主题置信度计算"""
        # 包含技术术语的句子
        tech_sentence = "机器学习算法是重要的数据处理技术"
        confidence = self.extractor._calculate_topic_confidence(tech_sentence)
        assert confidence > 0.7
        
        # 定义性句子
        definition_sentence = "深度学习是机器学习的子领域"
        confidence = self.extractor._calculate_topic_confidence(definition_sentence)
        assert confidence > 0.7
        
        # 普通句子
        normal_sentence = "今天天气很好"
        confidence = self.extractor._calculate_topic_confidence(normal_sentence)
        assert confidence <= 0.7
    
    def test_calculate_fact_importance(self):
        """测试事实重要性计算"""
        # 包含数字的重要事实
        important_fact = "实验证明准确率达到95%，这是重要的突破"
        importance = self.extractor._calculate_fact_importance(important_fact)
        assert importance > 0.7
        
        # 普通陈述
        normal_statement = "这是一个普通的描述"
        importance = self.extractor._calculate_fact_importance(normal_statement)
        assert importance < 0.7


class TestLongTextEvaluator:
    """长文本理解评估器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.evaluator = LongTextEvaluator()
    
    def test_initialization(self):
        """测试初始化"""
        assert self.evaluator.name == "long_text_understanding"
        assert self.evaluator.weight == 1.0
        assert len(self.evaluator.criteria) == 4
        
        # 检查评估标准
        criteria_names = [c.name for c in self.evaluator.criteria]
        assert "structure_understanding" in criteria_names
        assert "key_information_extraction" in criteria_names
        assert "coherence_maintenance" in criteria_names
        assert "comprehension_depth" in criteria_names
    
    def test_evaluate_basic(self):
        """测试基本评估功能"""
        input_text = """
        # 机器学习概述
        
        机器学习是人工智能的重要分支，通过算法从数据中学习模式。
        
        ## 主要类型
        
        1. 监督学习：使用标注数据训练模型
        2. 无监督学习：从无标注数据中发现模式
        3. 强化学习：通过与环境交互学习最优策略
        
        ## 应用领域
        
        机器学习在图像识别、自然语言处理、推荐系统等领域有广泛应用。
        研究表明，深度学习在复杂任务中表现优异。
        
        因此，机器学习技术将继续推动人工智能的发展。
        """
        
        model_output = """
        机器学习是AI的核心技术，包括监督学习、无监督学习和强化学习三种主要类型。
        
        监督学习使用标注数据，无监督学习发现隐藏模式，强化学习通过交互优化策略。
        
        应用包括图像识别和自然语言处理。深度学习在复杂任务中效果显著。
        
        总之，机器学习推动AI发展。
        """
        
        expected_output = "机器学习包括三种类型，广泛应用于多个领域，是AI发展的关键技术。"
        
        result = self.evaluator.evaluate(input_text, model_output, expected_output, {})
        
        assert 0 <= result.overall_score <= 1
        assert self.evaluator.name in result.dimension_scores
        assert result.confidence > 0
        assert "input_analysis" in result.details
        assert "output_analysis" in result.details
    
    def test_evaluate_structure_understanding(self):
        """测试结构理解评估"""
        # 输入结构
        input_structure = {
            "basic_stats": {"char_count": 1000, "paragraph_count": 4},
            "paragraph_analysis": {"count": 4},
            "hierarchical_structure": {"hierarchy_depth": 2}
        }
        
        # 输出结构（保持了合理的结构）
        output_structure = {
            "basic_stats": {"char_count": 400, "paragraph_count": 3},
            "paragraph_analysis": {"count": 3},
            "hierarchical_structure": {"hierarchy_depth": 1}
        }
        
        score = self.evaluator._evaluate_structure_understanding(input_structure, output_structure)
        
        assert 0 <= score <= 1
        # 应该有较高分数，因为保持了合理的结构比例
        assert score > 0.5
    
    def test_evaluate_information_extraction(self):
        """测试信息提取评估"""
        output_info = {
            "main_topics": [
                {"topic_words": ["机器学习", "算法"]},
                {"topic_words": ["深度学习", "神经网络"]}
            ],
            "key_entities": {
                "technical_terms": ["机器学习", "深度学习", "算法"],
                "concepts": ["人工智能"]
            },
            "important_facts": [
                {"importance_score": 0.8},
                {"importance_score": 0.7}
            ],
            "conclusions": ["机器学习很重要"],
            "numerical_data": [{"value": 95, "unit": "%"}]
        }
        
        expected_info = {
            "main_topics": [{"topic_words": ["机器学习"]}],
            "key_entities": {"technical_terms": ["机器学习"]},
            "conclusions": ["机器学习很重要"]
        }
        
        score = self.evaluator._evaluate_information_extraction(output_info, expected_info, "输入文本")
        
        assert 0 <= score <= 1
        # 应该有较高分数，因为提取了丰富的信息
        assert score > 0.6
    
    def test_evaluate_coherence_maintenance(self):
        """测试连贯性保持评估"""
        # 高连贯性结构
        high_coherence_structure = {
            "coherence_analysis": {
                "overall_coherence": 0.8,
                "global_coherence": 0.7,
                "coherence_variance": 0.1
            }
        }
        
        score = self.evaluator._evaluate_coherence_maintenance(high_coherence_structure)
        assert score > 0.7
        
        # 低连贯性结构
        low_coherence_structure = {
            "coherence_analysis": {
                "overall_coherence": 0.3,
                "global_coherence": 0.2,
                "coherence_variance": 0.5
            }
        }
        
        score = self.evaluator._evaluate_coherence_maintenance(low_coherence_structure)
        assert score < 0.6
    
    def test_evaluate_comprehension_depth(self):
        """测试理解深度评估"""
        # 深度理解的信息
        deep_info = {
            "relationships": [
                {"relation": "导致", "confidence": 0.8},
                {"relation": "影响", "confidence": 0.7}
            ],
            "temporal_information": [
                {"expression": "2023年", "type": "year"},
                {"expression": "最近", "type": "general"}
            ],
            "main_topics": [
                {"topic_words": ["机器学习", "算法", "数据"]}
            ]
        }
        
        score = self.evaluator._evaluate_comprehension_depth(deep_info, "输入文本")
        
        assert 0 <= score <= 1
        # 应该有较高分数，因为包含了关系和时间理解
        assert score > 0.5
    
    def test_calculate_text_complexity(self):
        """测试文本复杂度计算"""
        # 复杂文本
        complex_text = "机器学习算法通过分析大量数据，利用统计方法和优化技术，构建能够自动学习和改进的模型系统。"
        complexity = self.evaluator._calculate_text_complexity(complex_text)
        
        assert "lexical" in complexity
        assert "syntactic" in complexity
        assert "semantic" in complexity
        assert "overall" in complexity
        assert all(0 <= score <= 1 for score in complexity.values())
        
        # 复杂文本应该有较高的复杂度
        assert complexity["overall"] > 0.3
    
    def test_calculate_information_density(self):
        """测试信息密度计算"""
        dense_info = {
            "main_topics": [{"topic_words": ["A"]}, {"topic_words": ["B"]}],
            "key_entities": {"technical_terms": ["C", "D", "E"]},
            "important_facts": [{"score": 0.8}, {"score": 0.7}],
            "relationships": [{"rel": "causes"}]
        }
        
        density = self.evaluator._calculate_information_density(dense_info)
        
        assert 0 <= density <= 1
        # 信息丰富应该有较高密度
        assert density > 0.3
    
    def test_generate_details(self):
        """测试详细信息生成"""
        input_text = """
        # 机器学习
        
        机器学习是重要技术。包括多种算法。
        
        应用广泛，效果显著。
        """
        
        model_output = "机器学习包括监督学习和无监督学习，在图像识别等领域应用广泛。"
        expected_output = "机器学习有多种类型，应用领域广泛。"
        
        details = self.evaluator._generate_details(
            input_text, model_output, expected_output, {}, 0.8
        )
        
        assert details["evaluator"] == "long_text_understanding"
        assert details["score"] == 0.8
        assert "input_analysis" in details
        assert "output_analysis" in details
        assert "expected_analysis" in details
        assert "evaluation_scores" in details
        assert "quality_metrics" in details
        assert "comparison_analysis" in details
        assert "recommendations" in details
        
        # 检查评估分数
        eval_scores = details["evaluation_scores"]
        assert "structure_understanding" in eval_scores
        assert "information_extraction" in eval_scores
        assert "coherence_maintenance" in eval_scores
        assert "comprehension_depth" in eval_scores
        
        # 检查质量指标
        quality = details["quality_metrics"]
        assert "information_density" in quality
        assert "structural_preservation" in quality
        assert "coherence_quality" in quality
        assert "depth_indicators" in quality
    
    def test_generate_comprehension_recommendations(self):
        """测试理解改进建议生成"""
        # 模拟结构分析结果
        input_structure = {
            "paragraph_analysis": {"count": 5},
            "hierarchical_structure": {"hierarchy_depth": 2}
        }
        
        output_structure = {
            "paragraph_analysis": {"count": 2},
            "hierarchical_structure": {"hierarchy_depth": 0},
            "coherence_analysis": {"coherence_variance": 0.4}
        }
        
        output_info = {
            "main_topics": [{"topic_words": ["A"]}],  # 主题较少
            "important_facts": [],  # 缺少重要事实
            "relationships": [],    # 缺少关系
            "temporal_information": []  # 缺少时间信息
        }
        
        recommendations = self.evaluator._generate_comprehension_recommendations(
            input_structure, output_structure, output_info, 0.4, 0.5, 0.4, 0.3
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 8
        
        # 应该包含针对各种问题的建议
        rec_text = " ".join(recommendations)
        assert any(keyword in rec_text for keyword in ["段落", "主题", "事实", "关系", "连贯"])
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空文本
        result = self.evaluator.evaluate("", "", "", {})
        assert 0 <= result.overall_score <= 1
        
        # 很短的文本
        short_input = "短文本。"
        short_output = "很短。"
        result = self.evaluator.evaluate(short_input, short_output, "", {})
        assert 0 <= result.overall_score <= 1
        
        # 很长的文本
        long_input = "这是一个很长的文本。" * 100
        long_output = "这是总结。" * 10
        result = self.evaluator.evaluate(long_input, long_output, "", {})
        assert 0 <= result.overall_score <= 1
        assert result.details["output_analysis"]["compression_ratio"] < 1.0
    
    def test_custom_evaluator(self):
        """测试自定义评估器"""
        custom_evaluator = LongTextEvaluator(name="custom_long_text", weight=0.8)
        
        assert custom_evaluator.name == "custom_long_text"
        assert custom_evaluator.weight == 0.8
        
        # 测试评估功能
        result = custom_evaluator.evaluate(
            "长文本输入内容",
            "总结输出内容",
            "期望输出",
            {}
        )
        
        assert 0 <= result.overall_score <= 1
        assert "custom_long_text" in result.dimension_scores


if __name__ == "__main__":
    pytest.main([__file__])
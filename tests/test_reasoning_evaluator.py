"""
逻辑推理评估器单元测试
"""

import pytest
from industry_evaluation.evaluators.reasoning_evaluator import (
    CausalRelationExtractor, ReasoningChainAnalyzer, CausalReasoningEvaluator
)


class TestCausalRelationExtractor:
    """因果关系提取器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.extractor = CausalRelationExtractor()
    
    def test_extract_causal_relations_explicit(self):
        """测试显式因果关系提取"""
        text = "过度训练导致模型过拟合，数据不足引起欠拟合问题。"
        
        relations = self.extractor.extract_causal_relations(text)
        
        assert len(relations) >= 2
        
        # 检查第一个因果关系
        relation1 = relations[0]
        assert "过度训练" in relation1["cause"]
        assert "过拟合" in relation1["effect"]
        assert relation1["pattern_type"] == "explicit_causal"
        assert relation1["confidence"] > 0.8
        
        # 检查第二个因果关系
        relation2 = next((r for r in relations if "数据不足" in r["cause"]), None)
        assert relation2 is not None
        assert "欠拟合" in relation2["effect"]
    
    def test_extract_causal_relations_conditional(self):
        """测试条件因果关系提取"""
        text = "如果学习率设置过高，那么模型将无法收敛。"
        
        relations = self.extractor.extract_causal_relations(text)
        
        assert len(relations) >= 1
        relation = relations[0]
        assert "学习率设置过高" in relation["cause"]
        assert "模型将无法收敛" in relation["effect"]
        assert relation["pattern_type"] == "conditional"
    
    def test_extract_causal_relations_temporal(self):
        """测试时序因果关系提取"""
        text = "数据预处理之后，模型性能得到显著提升。"
        
        relations = self.extractor.extract_causal_relations(text)
        
        assert len(relations) >= 1
        relation = relations[0]
        assert "数据预处理" in relation["cause"]
        assert "模型性能得到显著提升" in relation["effect"]
        assert relation["pattern_type"] == "temporal"
    
    def test_extract_by_indicators(self):
        """测试基于指示词的提取"""
        text = "由于数据质量差，所以模型准确率低。"
        
        relations = self.extractor._extract_by_indicators(text)
        
        assert len(relations) >= 1
        relation = relations[0]
        assert "数据质量差" in relation["cause"]
        assert "模型准确率低" in relation["effect"]
        assert relation["indicator"] == "由于"
    
    def test_clean_causal_entity(self):
        """测试因果实体清理"""
        # 测试前缀清理
        assert self.extractor._clean_causal_entity("的算法") == "算法"
        assert self.extractor._clean_causal_entity("一个模型") == "模型"
        
        # 测试后缀清理
        assert self.extractor._clean_causal_entity("训练了") == "训练"
        assert self.extractor._clean_causal_entity("数据等") == "数据"
        
        # 测试标点符号清理
        assert self.extractor._clean_causal_entity("机器学习，") == "机器学习"
        
        # 测试过短实体
        assert self.extractor._clean_causal_entity("A") == ""
        
        # 测试过长实体
        long_text = "A" * 101
        assert self.extractor._clean_causal_entity(long_text) == ""
    
    def test_calculate_confidence(self):
        """测试置信度计算"""
        # 显式因果关系应该有高置信度
        confidence = self.extractor._calculate_confidence("explicit_causal", "原因", "结果")
        assert confidence >= 0.8
        
        # 相关性关系置信度较低
        confidence = self.extractor._calculate_confidence("correlational", "原因", "结果")
        assert confidence <= 0.6
        
        # 短实体应该降低置信度
        confidence = self.extractor._calculate_confidence("explicit_causal", "A", "B")
        assert confidence < 0.9
        
        # 包含不确定词应该降低置信度
        confidence = self.extractor._calculate_confidence("explicit_causal", "可能的原因", "结果")
        assert confidence < 0.9
    
    def test_deduplicate_relations(self):
        """测试关系去重"""
        relations = [
            {
                "cause": "原因A",
                "effect": "结果B",
                "confidence": 0.9,
                "pattern_type": "explicit_causal"
            },
            {
                "cause": "原因A",
                "effect": "结果B",
                "confidence": 0.7,  # 较低置信度
                "pattern_type": "conditional"
            },
            {
                "cause": "原因C",
                "effect": "结果D",
                "confidence": 0.8,
                "pattern_type": "explicit_causal"
            }
        ]
        
        unique_relations = self.extractor._deduplicate_relations(relations)
        
        assert len(unique_relations) == 2  # 去重后应该只有2个
        
        # 应该保留置信度更高的关系
        ab_relation = next(r for r in unique_relations if r["cause"] == "原因A")
        assert ab_relation["confidence"] == 0.9
        
        # 应该按置信度排序
        assert unique_relations[0]["confidence"] >= unique_relations[1]["confidence"]


class TestReasoningChainAnalyzer:
    """推理链分析器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.analyzer = ReasoningChainAnalyzer()
    
    def test_extract_reasoning_steps(self):
        """测试推理步骤提取"""
        text = "首先，收集训练数据。其次，预处理数据。最后，训练模型。"
        
        steps = self.analyzer._extract_reasoning_steps(text)
        
        assert len(steps) == 3
        assert steps[0]["indicator"] == "首先"
        assert "收集训练数据" in steps[0]["content"]
        assert steps[1]["indicator"] == "其次"
        assert "预处理数据" in steps[1]["content"]
        assert steps[2]["indicator"] == "最后"
        assert "训练模型" in steps[2]["content"]
    
    def test_extract_reasoning_steps_numbered(self):
        """测试数字编号的推理步骤提取"""
        text = "1. 数据收集。2. 数据清洗。3. 模型训练。"
        
        steps = self.analyzer._extract_reasoning_steps(text)
        
        assert len(steps) == 3
        assert steps[0]["indicator"] == "1"
        assert "数据收集" in steps[0]["content"]
    
    def test_classify_step_type(self):
        """测试步骤类型分类"""
        assert self.analyzer._classify_step_type("假设数据是正态分布") == "assumption"
        assert self.analyzer._classify_step_type("因为模型过拟合") == "premise"
        assert self.analyzer._classify_step_type("所以需要正则化") == "conclusion"
        assert self.analyzer._classify_step_type("证明算法收敛") == "derivation"
        assert self.analyzer._classify_step_type("模型性能良好") == "statement"
    
    def test_analyze_logical_structure(self):
        """测试逻辑结构分析"""
        text = "假设数据是独立同分布的。因为使用了正则化，所以模型不会过拟合。"
        
        structure = self.analyzer._analyze_logical_structure(text)
        
        assert "premise_conclusion" in structure
        assert "argument_structure" in structure
        assert "logical_connectors" in structure
        assert "reasoning_type" in structure
        
        # 检查前提和结论识别
        pc = structure["premise_conclusion"]
        assert len(pc["premises"]) > 0
        assert len(pc["conclusions"]) > 0
    
    def test_identify_premise_conclusion(self):
        """测试前提结论识别"""
        text = "假设数据质量良好。因为使用了深度学习，所以准确率很高。"
        
        pc = self.analyzer._identify_premise_conclusion(text)
        
        assert "数据质量良好" in pc["premises"][0]
        assert "准确率很高" in pc["conclusions"][0]
    
    def test_analyze_argument_structure(self):
        """测试论证结构分析"""
        # 条件结构
        conditional_text = "如果数据充足，那么模型性能会更好。"
        assert self.analyzer._analyze_argument_structure(conditional_text) == "conditional"
        
        # 因果结构
        causal_text = "因为数据不足，所以模型欠拟合。"
        assert self.analyzer._analyze_argument_structure(causal_text) == "causal"
        
        # 比较结构
        comparative_text = "深度学习相比传统方法更有效。"
        assert self.analyzer._analyze_argument_structure(comparative_text) == "comparative"
        
        # 归纳结构
        inductive_text = "例如，CNN在图像识别中表现出色。"
        assert self.analyzer._analyze_argument_structure(inductive_text) == "inductive"
    
    def test_build_causal_chains(self):
        """测试因果链构建"""
        causal_relations = [
            {"cause": "A", "effect": "B", "confidence": 0.9},
            {"cause": "B", "effect": "C", "confidence": 0.8},
            {"cause": "D", "effect": "E", "confidence": 0.7}
        ]
        
        chains = self.analyzer._build_causal_chains(causal_relations)
        
        # 应该有一个长度为2的链条 A->B->C
        assert len(chains) >= 1
        long_chain = max(chains, key=len)
        assert len(long_chain) == 2
        assert long_chain[0]["cause"] == "A"
        assert long_chain[1]["effect"] == "C"
    
    def test_assess_reasoning_quality(self):
        """测试推理质量评估"""
        text = "根据大量实验数据，我们发现深度学习在图像识别任务中表现优异。因此，建议在相关项目中采用深度学习方法。"
        causal_relations = [
            {"cause": "深度学习", "effect": "图像识别表现优异", "confidence": 0.9}
        ]
        
        quality = self.analyzer._assess_reasoning_quality(text, causal_relations)
        
        assert "clarity" in quality
        assert "coherence" in quality
        assert "logical_validity" in quality
        assert "evidence_support" in quality
        assert "completeness" in quality
        
        # 所有分数应该在0-1之间
        for score in quality.values():
            assert 0 <= score <= 1
        
        # 有证据支持的文本应该有较高的evidence_support分数
        assert quality["evidence_support"] > 0.5
    
    def test_check_logical_consistency(self):
        """测试逻辑一致性检查"""
        # 包含矛盾的关系
        contradictory_relations = [
            {"cause": "训练", "effect": "提高准确率", "confidence": 0.9},
            {"cause": "训练", "effect": "降低准确率", "confidence": 0.8}
        ]
        
        consistency = self.analyzer._check_logical_consistency(contradictory_relations)
        
        assert "contradictions" in consistency
        assert "circular_reasoning" in consistency
        assert "consistency_score" in consistency
        
        # 应该检测到矛盾
        assert len(consistency["contradictions"]) > 0
        assert consistency["consistency_score"] < 1.0
    
    def test_find_contradictions(self):
        """测试矛盾查找"""
        relations = [
            {"cause": "A", "effect": "增加B", "confidence": 0.9},
            {"cause": "A", "effect": "减少B", "confidence": 0.8}
        ]
        
        contradictions = self.analyzer._find_contradictions(relations)
        
        assert len(contradictions) > 0
        assert contradictions[0]["type"] == "contradictory_effects"
    
    def test_detect_circular_reasoning(self):
        """测试循环推理检测"""
        relations = [
            {"cause": "A", "effect": "B", "confidence": 0.9},
            {"cause": "B", "effect": "A", "confidence": 0.8}
        ]
        
        circular = self.analyzer._detect_circular_reasoning(relations)
        
        # 应该检测到循环推理
        assert len(circular) > 0
    
    def test_assess_clarity(self):
        """测试清晰度评估"""
        # 理想长度的句子
        clear_text = "机器学习是一种人工智能技术。它可以从数据中学习模式。"
        clarity = self.analyzer._assess_clarity(clear_text)
        assert clarity > 0.8
        
        # 过短的句子
        short_text = "好。是的。对。"
        clarity = self.analyzer._assess_clarity(short_text)
        assert clarity < 1.0
        
        # 过长的句子
        long_text = "这是一个非常非常长的句子，" * 10
        clarity = self.analyzer._assess_clarity(long_text)
        assert clarity < 1.0
    
    def test_assess_coherence(self):
        """测试连贯性评估"""
        # 有逻辑连接词的文本
        coherent_text = "首先收集数据，然后预处理，最后训练模型。因此得到了好的结果。"
        coherence = self.analyzer._assess_coherence(coherent_text)
        assert coherence > 0.5
        
        # 没有连接词的文本
        incoherent_text = "收集数据。预处理。训练模型。得到结果。"
        coherence = self.analyzer._assess_coherence(incoherent_text)
        assert coherence < 1.0


class TestCausalReasoningEvaluator:
    """因果关系推理评估器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.evaluator = CausalReasoningEvaluator()
    
    def test_initialization(self):
        """测试初始化"""
        assert self.evaluator.name == "causal_reasoning"
        assert self.evaluator.weight == 1.0
        assert len(self.evaluator.criteria) == 3
        
        # 检查评估标准
        criteria_names = [c.name for c in self.evaluator.criteria]
        assert "causal_relation_accuracy" in criteria_names
        assert "reasoning_chain_quality" in criteria_names
        assert "logical_consistency" in criteria_names
    
    def test_evaluate_basic(self):
        """测试基本评估功能"""
        input_text = "解释过拟合的原因"
        model_output = "过度训练导致模型过拟合。因为模型在训练数据上学习过度，所以在新数据上表现差。"
        expected_output = "训练过度会导致过拟合问题"
        
        result = self.evaluator.evaluate(input_text, model_output, expected_output, {})
        
        assert 0 <= result.overall_score <= 1
        assert self.evaluator.name in result.dimension_scores
        assert result.confidence > 0
        assert "causal_relations" in result.details
        assert "reasoning_analysis" in result.details
    
    def test_evaluate_causal_accuracy(self):
        """测试因果关系准确性评估"""
        # 高质量因果关系
        high_quality_relations = [
            {"cause": "过度训练", "effect": "过拟合", "confidence": 0.9},
            {"cause": "数据不足", "effect": "欠拟合", "confidence": 0.8}
        ]
        
        score = self.evaluator._evaluate_causal_accuracy(high_quality_relations, "")
        assert score > 0.7
        
        # 低质量因果关系
        low_quality_relations = [
            {"cause": "A", "effect": "B", "confidence": 0.3}
        ]
        
        score = self.evaluator._evaluate_causal_accuracy(low_quality_relations, "")
        assert score < 0.5
        
        # 空关系列表
        score = self.evaluator._evaluate_causal_accuracy([], "")
        assert score == 0.3
    
    def test_calculate_relation_consistency(self):
        """测试关系一致性计算"""
        output_relations = [
            {"cause": "A", "effect": "B", "confidence": 0.9},
            {"cause": "C", "effect": "D", "confidence": 0.8}
        ]
        
        expected_relations = [
            {"cause": "A", "effect": "B", "confidence": 1.0},
            {"cause": "E", "effect": "F", "confidence": 1.0}
        ]
        
        consistency = self.evaluator._calculate_relation_consistency(output_relations, expected_relations)
        
        # 有一个匹配的关系对，总共3个唯一关系对
        assert consistency == 1/3
        
        # 完全匹配的情况
        consistency = self.evaluator._calculate_relation_consistency(output_relations, output_relations)
        assert consistency == 1.0
        
        # 空期望关系
        consistency = self.evaluator._calculate_relation_consistency(output_relations, [])
        assert consistency == 0.5
    
    def test_evaluate_reasoning_quality(self):
        """测试推理质量评估"""
        reasoning_analysis = {
            "reasoning_quality": {
                "clarity": 0.8,
                "coherence": 0.7,
                "logical_validity": 0.9,
                "evidence_support": 0.6,
                "completeness": 0.7
            }
        }
        
        quality = self.evaluator._evaluate_reasoning_quality(reasoning_analysis)
        
        # 应该是加权平均
        expected = (0.8*0.2 + 0.7*0.2 + 0.9*0.3 + 0.6*0.15 + 0.7*0.15)
        assert abs(quality - expected) < 0.01
    
    def test_calculate_score(self):
        """测试分数计算"""
        input_text = "解释因果关系"
        model_output = "过度训练导致过拟合。首先，模型在训练数据上学习。然后，如果训练时间过长，就会记住训练数据的细节。因此，模型在新数据上表现差。"
        expected_output = "训练过度导致过拟合"
        
        score = self.evaluator._calculate_score(input_text, model_output, expected_output, {})
        
        assert 0 <= score <= 1
        # 应该有较高分数，因为包含清晰的因果关系和推理步骤
        assert score > 0.4
    
    def test_generate_details(self):
        """测试详细信息生成"""
        input_text = "输入"
        model_output = "数据不足导致欠拟合。因为训练样本太少，所以模型无法学习到足够的模式。"
        expected_output = "数据不足会导致欠拟合"
        
        details = self.evaluator._generate_details(
            input_text, model_output, expected_output, {}, 0.8
        )
        
        assert details["evaluator"] == "causal_reasoning"
        assert details["score"] == 0.8
        assert "causal_relations" in details
        assert "expected_relations" in details
        assert "reasoning_analysis" in details
        assert "causal_accuracy" in details
        assert "reasoning_quality" in details
        assert "logical_consistency" in details
        assert "quality_breakdown" in details
        assert "causal_chains" in details
        assert "reasoning_steps" in details
        assert "issues" in details
        assert "recommendations" in details
        
        # 检查问题分析
        issues = details["issues"]
        assert "contradictions" in issues
        assert "circular_reasoning" in issues
        assert "missing_premises" in issues
        assert "unsupported_conclusions" in issues
    
    def test_generate_reasoning_recommendations(self):
        """测试推理改进建议生成"""
        # 模拟推理分析结果
        reasoning_analysis = {
            "reasoning_quality": {
                "clarity": 0.4,  # 低清晰度
                "coherence": 0.5,  # 低连贯性
                "logical_validity": 0.8,
                "evidence_support": 0.3,  # 缺乏证据
                "completeness": 0.5  # 不完整
            },
            "logical_consistency": {
                "contradictions": [{"type": "test"}],  # 有矛盾
                "circular_reasoning": []
            }
        }
        
        causal_relations = [
            {"confidence": 0.3}  # 低置信度关系
        ]
        
        recommendations = self.evaluator._generate_reasoning_recommendations(reasoning_analysis, causal_relations)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 6
        
        # 应该包含针对各种问题的建议
        rec_text = " ".join(recommendations)
        assert any(keyword in rec_text for keyword in ["清晰", "连贯", "证据", "矛盾", "置信度"])
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空文本
        result = self.evaluator.evaluate("", "", "", {})
        assert 0 <= result.overall_score <= 1
        
        # 无因果关系的文本
        result = self.evaluator.evaluate("输入", "这是简单的描述文本", "期望", {})
        assert 0 <= result.overall_score <= 1
        assert result.details["causal_relations"] == []
        
        # 复杂推理文本
        complex_text = """
        首先，我们假设数据是独立同分布的。
        因为使用了大量训练数据，所以模型能够学习到复杂的模式。
        然而，如果数据存在噪声，那么模型可能会过拟合。
        因此，我们需要使用正则化技术来防止过拟合。
        最终，通过交叉验证，我们得出模型性能良好的结论。
        """
        result = self.evaluator.evaluate("输入", complex_text, "期望", {})
        assert 0 <= result.overall_score <= 1
        assert len(result.details["causal_relations"]) > 0
        assert len(result.details["reasoning_steps"]) > 0
    
    def test_custom_evaluator(self):
        """测试自定义评估器"""
        custom_evaluator = CausalReasoningEvaluator(name="custom_reasoning", weight=0.8)
        
        assert custom_evaluator.name == "custom_reasoning"
        assert custom_evaluator.weight == 0.8
        
        # 测试评估功能
        result = custom_evaluator.evaluate(
            "输入",
            "训练数据不足导致模型欠拟合",
            "期望",
            {}
        )
        
        assert 0 <= result.overall_score <= 1
        assert "custom_reasoning" in result.dimension_scores


if __name__ == "__main__":
    pytest.main([__file__])
"""
多步推理能力评估器单元测试
"""

import pytest
from industry_evaluation.evaluators.multi_step_reasoning_evaluator import (
    ReasoningStepExtractor, ReasoningChainValidator, MultiStepReasoningEvaluator
)


class TestReasoningStepExtractor:
    """推理步骤提取器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.extractor = ReasoningStepExtractor()
    
    def test_extract_explicit_steps(self):
        """测试显式步骤提取"""
        text = "1. 首先收集数据。2. 然后预处理数据。3. 最后训练模型。"
        
        steps = self.extractor._extract_explicit_steps(text)
        
        assert len(steps) == 3
        assert steps[0]["step_number"] == 1
        assert "收集数据" in steps[0]["content"]
        assert steps[1]["step_number"] == 2
        assert "预处理数据" in steps[1]["content"]
        assert steps[2]["step_number"] == 3
        assert "训练模型" in steps[2]["content"]
    
    def test_extract_sequential_steps(self):
        """测试序列步骤提取"""
        text = "首先，我们需要定义问题。其次，收集相关数据。最后，分析结果。"
        
        steps = self.extractor._extract_explicit_steps(text)
        
        assert len(steps) >= 3
        # 检查序列指示词被正确识别
        first_step = next((s for s in steps if s["step_number"] == 1), None)
        assert first_step is not None
        assert "定义问题" in first_step["content"]
    
    def test_extract_implicit_steps(self):
        """测试隐式步骤提取"""
        text = "数据质量很重要。我们需要清洗数据。然后可以开始训练。"
        
        steps = self.extractor._extract_implicit_steps(text)
        
        assert len(steps) == 3
        assert all(step["step_number"] > 0 for step in steps)
        assert all(step["content"] for step in steps)
    
    def test_classify_sentence_type(self):
        """测试句子类型分类"""
        assert self.extractor._classify_sentence_type("假设数据是正态分布") == "assumption"
        assert self.extractor._classify_sentence_type("因为模型过拟合") == "premise"
        assert self.extractor._classify_sentence_type("所以需要正则化") == "conclusion"
        assert self.extractor._classify_sentence_type("如果学习率过高") == "condition"
        assert self.extractor._classify_sentence_type("证明算法收敛") == "derivation"
        assert self.extractor._classify_sentence_type("观察到性能提升") == "observation"
        assert self.extractor._classify_sentence_type("模型表现良好") == "statement"
    
    def test_identify_reasoning_role(self):
        """测试推理作用识别"""
        assert self.extractor._identify_reasoning_role("定义机器学习概念") == "definition"
        assert self.extractor._identify_reasoning_role("分析数据特征") == "analysis"
        assert self.extractor._identify_reasoning_role("应用深度学习") == "application"
        assert self.extractor._identify_reasoning_role("比较两种方法") == "comparison"
        assert self.extractor._identify_reasoning_role("总结实验结果") == "synthesis"
        assert self.extractor._identify_reasoning_role("评估模型性能") == "evaluation"
        assert self.extractor._identify_reasoning_role("推导公式") == "reasoning"
    
    def test_calculate_step_confidence(self):
        """测试步骤置信度计算"""
        # 编号步骤应该有高置信度
        confidence = self.extractor._calculate_step_confidence("这是详细的步骤描述", "numbered")
        assert confidence >= 0.8
        
        # 短内容应该降低置信度
        confidence = self.extractor._calculate_step_confidence("短", "numbered")
        assert confidence < 0.9
        
        # 包含逻辑词汇应该提高置信度
        confidence = self.extractor._calculate_step_confidence("因此我们得出结论", "statement")
        assert confidence > 0.6
    
    def test_analyze_step_dependencies(self):
        """测试步骤依赖分析"""
        steps = [
            {"step_number": 1, "content": "假设数据独立", "type": "assumption"},
            {"step_number": 2, "content": "根据步骤1，我们可以推导", "type": "derivation"},
            {"step_number": 3, "content": "因此得出结论", "type": "conclusion"}
        ]
        
        analyzed_steps = self.extractor._analyze_step_dependencies(steps, "测试文本")
        
        # 第二步应该依赖第一步
        step2 = next(s for s in analyzed_steps if s["step_number"] == 2)
        assert 1 in step2["dependencies"]
        
        # 第三步应该有逻辑依赖
        step3 = next(s for s in analyzed_steps if s["step_number"] == 3)
        assert len(step3["dependencies"]) > 0
    
    def test_find_references(self):
        """测试引用关系查找"""
        previous_steps = [
            {"step_number": 1, "content": "第一步"},
            {"step_number": 2, "content": "第二步"}
        ]
        
        # 明确引用
        content = "根据步骤1的结果"
        refs = self.extractor._find_references(content, previous_steps)
        assert 1 in refs
        
        # 隐式引用
        content = "根据上述分析"
        refs = self.extractor._find_references(content, previous_steps)
        assert len(refs) > 0
    
    def test_find_logical_dependencies(self):
        """测试逻辑依赖查找"""
        current_step = {"step_number": 3, "type": "conclusion", "reasoning_role": "reasoning"}
        previous_steps = [
            {"step_number": 1, "type": "assumption", "reasoning_role": "definition"},
            {"step_number": 2, "type": "premise", "reasoning_role": "analysis"}
        ]
        
        deps = self.extractor._find_logical_dependencies(current_step, previous_steps)
        
        # 结论应该依赖前提
        assert len(deps) > 0
        assert 2 in deps  # 应该依赖premise类型的步骤
    
    def test_get_sequential_number(self):
        """测试序列数字获取"""
        assert self.extractor._get_sequential_number("首先") == 1
        assert self.extractor._get_sequential_number("其次") == 2
        assert self.extractor._get_sequential_number("最后") == 4
        assert self.extractor._get_sequential_number("未知") == 0


class TestReasoningChainValidator:
    """推理链验证器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.validator = ReasoningChainValidator()
    
    def test_validate_empty_chain(self):
        """测试空推理链验证"""
        result = self.validator.validate_reasoning_chain([])
        
        assert result["is_valid"] is False
        assert "没有找到推理步骤" in result["issues"]
    
    def test_validate_complete_chain(self):
        """测试完整推理链验证"""
        steps = [
            {"step_number": 1, "type": "assumption", "content": "假设数据独立", "dependencies": []},
            {"step_number": 2, "type": "premise", "content": "根据统计理论", "dependencies": [1]},
            {"step_number": 3, "type": "derivation", "content": "可以推导出", "dependencies": [2]},
            {"step_number": 4, "type": "conclusion", "content": "因此得出结论", "dependencies": [3]}
        ]
        
        result = self.validator.validate_reasoning_chain(steps)
        
        assert result["is_valid"] is True
        assert result["completeness_score"] > 0.7
        assert result["coherence_score"] > 0.5
        assert result["logical_validity_score"] > 0.5
    
    def test_validate_completeness(self):
        """测试完整性验证"""
        # 完整的推理链
        complete_steps = [
            {"step_number": 1, "type": "premise", "content": "前提", "dependencies": []},
            {"step_number": 2, "type": "derivation", "content": "推导", "dependencies": [1]},
            {"step_number": 3, "type": "conclusion", "content": "结论", "dependencies": [2]}
        ]
        
        result = self.validator._validate_completeness(complete_steps)
        assert result["completeness_score"] > 0.8
        
        # 不完整的推理链（缺少结论）
        incomplete_steps = [
            {"step_number": 1, "type": "premise", "content": "前提", "dependencies": []},
            {"step_number": 2, "type": "derivation", "content": "推导", "dependencies": [1]}
        ]
        
        result = self.validator._validate_completeness(incomplete_steps)
        assert result["completeness_score"] < 0.8
    
    def test_validate_coherence(self):
        """测试连贯性验证"""
        # 连贯的步骤
        coherent_steps = [
            {"step_number": 1, "type": "premise", "content": "数据质量好", "dependencies": []},
            {"step_number": 2, "type": "derivation", "content": "模型训练效果好", "dependencies": [1]}
        ]
        
        result = self.validator._validate_coherence(coherent_steps)
        assert result["coherence_score"] > 0.5
        
        # 单个步骤
        single_step = [{"step_number": 1, "type": "statement", "content": "单独陈述", "dependencies": []}]
        result = self.validator._validate_coherence(single_step)
        assert result["coherence_score"] == 1.0
    
    def test_validate_logical_validity(self):
        """测试逻辑有效性验证"""
        steps = [
            {"step_number": 1, "type": "premise", "content": "前提条件", "dependencies": []},
            {"step_number": 2, "type": "conclusion", "content": "逻辑结论", "dependencies": [1]}
        ]
        
        result = self.validator._validate_logical_validity(steps)
        
        assert "logical_validity_score" in result
        assert 0 <= result["logical_validity_score"] <= 1
        assert isinstance(result["logical_gaps"], list)
    
    def test_validate_step_qualities(self):
        """测试步骤质量验证"""
        steps = [
            {"step_number": 1, "type": "premise", "content": "这是一个详细的前提描述", "dependencies": []},
            {"step_number": 2, "type": "conclusion", "content": "短", "dependencies": [1]}
        ]
        
        qualities = self.validator._validate_step_qualities(steps)
        
        assert len(qualities) == 2
        assert all("clarity_score" in q for q in qualities)
        assert all("relevance_score" in q for q in qualities)
        assert all("necessity_score" in q for q in qualities)
        assert all("overall_score" in q for q in qualities)
        
        # 第一个步骤应该比第二个质量高（内容更详细）
        assert qualities[0]["clarity_score"] > qualities[1]["clarity_score"]
    
    def test_evaluate_step_connection(self):
        """测试步骤连接评估"""
        prev_step = {"type": "premise", "content": "前提内容", "step_number": 1}
        curr_step = {"type": "derivation", "content": "推导内容", "step_number": 2, "dependencies": [1]}
        
        score = self.validator._evaluate_step_connection(prev_step, curr_step)
        
        assert 0 <= score <= 1
        # premise到derivation是合理的转换，应该有较高分数
        assert score > 0.7
    
    def test_evaluate_step_validity(self):
        """测试步骤有效性评估"""
        # 有效的结论步骤
        valid_step = {"step_number": 2, "type": "conclusion", "content": "基于前面的分析，我们得出结论"}
        all_steps = [
            {"step_number": 1, "type": "premise", "content": "前提"},
            valid_step
        ]
        
        result = self.validator._evaluate_step_validity(valid_step, all_steps)
        
        assert "score" in result
        assert "issues" in result
        assert 0 <= result["score"] <= 1
        
        # 无效的结论步骤（没有前提支持）
        invalid_step = {"step_number": 1, "type": "conclusion", "content": "没有依据的结论"}
        result = self.validator._evaluate_step_validity(invalid_step, [invalid_step])
        
        assert result["score"] < 0.8
        assert len(result["issues"]) > 0
    
    def test_calculate_content_similarity(self):
        """测试内容相似度计算"""
        content1 = "机器学习算法训练数据"
        content2 = "深度学习模型训练过程"
        
        similarity = self.validator._calculate_content_similarity(content1, content2)
        
        assert 0 <= similarity <= 1
        # 应该有一定相似度（都包含"学习"、"训练"）
        assert similarity > 0
        
        # 完全相同的内容
        similarity = self.validator._calculate_content_similarity(content1, content1)
        assert similarity == 1.0
        
        # 完全不同的内容
        content3 = "完全不相关的内容"
        similarity = self.validator._calculate_content_similarity(content1, content3)
        assert similarity < 0.5


class TestMultiStepReasoningEvaluator:
    """多步推理评估器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.evaluator = MultiStepReasoningEvaluator()
    
    def test_initialization(self):
        """测试初始化"""
        assert self.evaluator.name == "multi_step_reasoning"
        assert self.evaluator.weight == 1.0
        assert len(self.evaluator.criteria) == 4
        
        # 检查评估标准
        criteria_names = [c.name for c in self.evaluator.criteria]
        assert "step_extraction_quality" in criteria_names
        assert "reasoning_completeness" in criteria_names
        assert "logical_coherence" in criteria_names
        assert "step_validity" in criteria_names
    
    def test_evaluate_basic(self):
        """测试基本评估功能"""
        input_text = "解释机器学习过程"
        model_output = """
        首先，我们需要收集训练数据。
        其次，对数据进行预处理和清洗。
        然后，选择合适的算法进行训练。
        最后，评估模型性能并进行优化。
        """
        expected_output = "机器学习包括数据收集、预处理、训练和评估步骤"
        
        result = self.evaluator.evaluate(input_text, model_output, expected_output, {})
        
        assert 0 <= result.overall_score <= 1
        assert self.evaluator.name in result.dimension_scores
        assert result.confidence > 0
        assert "reasoning_steps" in result.details
        assert "validation_result" in result.details
    
    def test_evaluate_extraction_quality(self):
        """测试提取质量评估"""
        # 高质量步骤
        high_quality_steps = [
            {"confidence": 0.9, "type": "premise"},
            {"confidence": 0.8, "type": "derivation"},
            {"confidence": 0.9, "type": "conclusion"},
            {"confidence": 0.7, "type": "analysis"},
            {"confidence": 0.8, "type": "application"}
        ]
        
        score = self.evaluator._evaluate_extraction_quality(high_quality_steps)
        assert score > 0.7
        
        # 低质量步骤
        low_quality_steps = [
            {"confidence": 0.3, "type": "statement"}
        ]
        
        score = self.evaluator._evaluate_extraction_quality(low_quality_steps)
        assert score < 0.5
        
        # 空步骤列表
        score = self.evaluator._evaluate_extraction_quality([])
        assert score == 0.0
    
    def test_calculate_score(self):
        """测试分数计算"""
        input_text = "解释推理过程"
        model_output = """
        假设我们有充足的训练数据。
        因为数据质量很好，所以可以训练出好的模型。
        通过交叉验证，我们发现模型性能优异。
        因此，这个模型可以用于实际应用。
        """
        expected_output = "推理过程包括假设、推导和结论"
        
        score = self.evaluator._calculate_score(input_text, model_output, expected_output, {})
        
        assert 0 <= score <= 1
        # 应该有较高分数，因为包含完整的推理步骤
        assert score > 0.4
    
    def test_analyze_dependencies(self):
        """测试依赖关系分析"""
        steps = [
            {"step_number": 1, "dependencies": []},
            {"step_number": 2, "dependencies": [1]},
            {"step_number": 3, "dependencies": [2]},
            {"step_number": 4, "dependencies": []}  # 孤立步骤
        ]
        
        analysis = self.evaluator._analyze_dependencies(steps)
        
        assert analysis["total_dependencies"] == 2
        assert len(analysis["dependency_chains"]) > 0
        assert 4 in analysis["orphaned_steps"]  # 步骤4是孤立的
        assert analysis["avg_dependencies_per_step"] == 0.5
    
    def test_trace_dependency_chain(self):
        """测试依赖链追踪"""
        steps = [
            {"step_number": 1, "dependencies": []},
            {"step_number": 2, "dependencies": [1]},
            {"step_number": 3, "dependencies": [2]}
        ]
        
        visited = set()
        chain = self.evaluator._trace_dependency_chain(steps[0], steps, visited)
        
        # 应该追踪到完整的链条
        assert len(chain) == 3
        assert 1 in chain
        assert 2 in chain
        assert 3 in chain
    
    def test_generate_details(self):
        """测试详细信息生成"""
        input_text = "输入"
        model_output = """
        1. 首先分析问题。
        2. 然后收集数据。
        3. 接着训练模型。
        4. 最后评估结果。
        """
        expected_output = "分析、收集、训练、评估"
        
        details = self.evaluator._generate_details(
            input_text, model_output, expected_output, {}, 0.8
        )
        
        assert details["evaluator"] == "multi_step_reasoning"
        assert details["score"] == 0.8
        assert "reasoning_steps" in details
        assert "expected_steps" in details
        assert "validation_result" in details
        assert "extraction_score" in details
        assert "completeness_score" in details
        assert "coherence_score" in details
        assert "validity_score" in details
        assert "step_count" in details
        assert "step_types" in details
        assert "reasoning_roles" in details
        assert "dependency_analysis" in details
        assert "quality_analysis" in details
        assert "issues_summary" in details
        assert "recommendations" in details
        
        # 检查质量分析
        quality = details["quality_analysis"]
        assert "high_quality_steps" in quality
        assert "low_quality_steps" in quality
        assert "avg_step_quality" in quality
        
        # 检查问题摘要
        issues = details["issues_summary"]
        assert "missing_steps" in issues
        assert "logical_gaps" in issues
        assert "total_issues" in issues
    
    def test_generate_reasoning_recommendations(self):
        """测试推理改进建议生成"""
        # 模拟推理步骤和验证结果
        steps = [
            {"step_number": 1, "type": "statement", "content": "简单陈述"}
        ]
        
        validation_result = {
            "completeness_score": 0.4,  # 低完整性
            "coherence_score": 0.5,     # 低连贯性
            "logical_validity_score": 0.4,  # 低有效性
            "missing_steps": ["前提", "结论"],
            "logical_gaps": ["缺乏逻辑支持"],
            "step_quality_scores": [
                {"step_number": 1, "overall_score": 0.3}  # 低质量步骤
            ]
        }
        
        recommendations = self.evaluator._generate_reasoning_recommendations(steps, validation_result)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 8
        
        # 应该包含针对各种问题的建议
        rec_text = " ".join(recommendations)
        assert any(keyword in rec_text for keyword in ["前提", "结论", "连贯", "逻辑", "步骤"])
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空文本
        result = self.evaluator.evaluate("", "", "", {})
        assert 0 <= result.overall_score <= 1
        
        # 无推理步骤的文本
        result = self.evaluator.evaluate("输入", "简单描述", "期望", {})
        assert 0 <= result.overall_score <= 1
        assert result.details["step_count"] >= 0
        
        # 复杂多步推理文本
        complex_text = """
        步骤1：定义问题域和目标。
        步骤2：收集和分析相关数据。
        步骤3：选择合适的机器学习算法。
        步骤4：设计实验和评估指标。
        步骤5：训练模型并调优参数。
        步骤6：验证模型性能和泛化能力。
        步骤7：部署模型到生产环境。
        步骤8：监控模型性能并持续改进。
        """
        result = self.evaluator.evaluate("输入", complex_text, "期望", {})
        assert 0 <= result.overall_score <= 1
        assert result.details["step_count"] >= 6
        
        # 包含循环依赖的文本
        circular_text = """
        A依赖于B的结果。
        B需要基于C的输出。
        C又依赖于A的前提。
        """
        result = self.evaluator.evaluate("输入", circular_text, "期望", {})
        assert 0 <= result.overall_score <= 1
    
    def test_custom_evaluator(self):
        """测试自定义评估器"""
        custom_evaluator = MultiStepReasoningEvaluator(name="custom_multi_step", weight=0.9)
        
        assert custom_evaluator.name == "custom_multi_step"
        assert custom_evaluator.weight == 0.9
        
        # 测试评估功能
        result = custom_evaluator.evaluate(
            "输入",
            "首先分析，然后实施，最后评估",
            "期望",
            {}
        )
        
        assert 0 <= result.overall_score <= 1
        assert "custom_multi_step" in result.dimension_scores


if __name__ == "__main__":
    pytest.main([__file__])
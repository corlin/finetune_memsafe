"""
多步推理能力评估器
"""

import re
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, deque
from industry_evaluation.evaluators.base_evaluator import AbstractEvaluator
from industry_evaluation.models.data_models import Criterion


class ReasoningStepExtractor:
    """推理步骤提取器"""
    
    def __init__(self):
        """初始化推理步骤提取器"""
        self.step_patterns = self._build_step_patterns()
        self.transition_indicators = self._build_transition_indicators()
    
    def extract_reasoning_steps(self, text: str) -> List[Dict[str, Any]]:
        """
        提取推理步骤
        
        Args:
            text: 输入文本
            
        Returns:
            List[Dict[str, Any]]: 推理步骤列表
        """
        steps = []
        
        # 使用显式步骤标识提取
        explicit_steps = self._extract_explicit_steps(text)
        if explicit_steps:
            steps.extend(explicit_steps)
        else:
            # 使用句子分割和转换指示词提取
            implicit_steps = self._extract_implicit_steps(text)
            steps.extend(implicit_steps)
        
        # 分析步骤间的依赖关系
        steps = self._analyze_step_dependencies(steps, text)
        
        return steps
    
    def _extract_explicit_steps(self, text: str) -> List[Dict[str, Any]]:
        """提取显式标记的推理步骤"""
        steps = []
        
        for pattern_type, patterns in self.step_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    step = self._parse_step_match(match, pattern_type, text)
                    if step:
                        steps.append(step)
        
        # 按位置排序
        steps.sort(key=lambda x: x["start_pos"])
        
        # 去重和编号
        steps = self._deduplicate_and_number_steps(steps)
        
        return steps    
    d
ef _extract_implicit_steps(self, text: str) -> List[Dict[str, Any]]:
        """提取隐式推理步骤"""
        steps = []
        sentences = re.split(r'[。！？.!?]', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 分析句子类型和推理作用
            step_type = self._classify_sentence_type(sentence)
            reasoning_role = self._identify_reasoning_role(sentence)
            
            step = {
                "step_number": i + 1,
                "content": sentence,
                "type": step_type,
                "reasoning_role": reasoning_role,
                "start_pos": text.find(sentence),
                "end_pos": text.find(sentence) + len(sentence),
                "dependencies": [],
                "confidence": self._calculate_step_confidence(sentence, step_type)
            }
            
            steps.append(step)
        
        return steps
    
    def _parse_step_match(self, match, pattern_type: str, text: str) -> Optional[Dict[str, Any]]:
        """解析步骤匹配结果"""
        try:
            if pattern_type == "numbered":
                step_number = int(match.group(1)) if match.group(1).isdigit() else len(text.split('.')) + 1
                content = match.group(2).strip()
            elif pattern_type == "sequential":
                step_number = self._get_sequential_number(match.group(1))
                content = match.group(2).strip()
            else:
                step_number = 0
                content = match.group(0).strip()
            
            return {
                "step_number": step_number,
                "content": content,
                "type": pattern_type,
                "reasoning_role": self._identify_reasoning_role(content),
                "start_pos": match.start(),
                "end_pos": match.end(),
                "dependencies": [],
                "confidence": self._calculate_step_confidence(content, pattern_type)
            }
        except Exception:
            return None
    
    def _classify_sentence_type(self, sentence: str) -> str:
        """分类句子类型"""
        if any(word in sentence for word in ["假设", "假定", "设", "令"]):
            return "assumption"
        elif any(word in sentence for word in ["因为", "由于", "根据", "基于"]):
            return "premise"
        elif any(word in sentence for word in ["所以", "因此", "故", "得出", "可知"]):
            return "conclusion"
        elif any(word in sentence for word in ["如果", "当", "若", "倘若"]):
            return "condition"
        elif any(word in sentence for word in ["证明", "推导", "计算", "求解"]):
            return "derivation"
        elif any(word in sentence for word in ["观察", "发现", "注意到"]):
            return "observation"
        else:
            return "statement"
    
    def _identify_reasoning_role(self, content: str) -> str:
        """识别推理作用"""
        if any(word in content for word in ["定义", "概念", "是指"]):
            return "definition"
        elif any(word in content for word in ["分析", "考虑", "研究"]):
            return "analysis"
        elif any(word in content for word in ["应用", "使用", "采用"]):
            return "application"
        elif any(word in content for word in ["比较", "对比", "相比"]):
            return "comparison"
        elif any(word in content for word in ["总结", "归纳", "概括"]):
            return "synthesis"
        elif any(word in content for word in ["评估", "判断", "评价"]):
            return "evaluation"
        else:
            return "reasoning"
    
    def _calculate_step_confidence(self, content: str, step_type: str) -> float:
        """计算步骤置信度"""
        base_confidence = {
            "numbered": 0.9,
            "sequential": 0.8,
            "assumption": 0.7,
            "premise": 0.8,
            "conclusion": 0.9,
            "derivation": 0.8,
            "statement": 0.6
        }
        
        confidence = base_confidence.get(step_type, 0.6)
        
        # 根据内容质量调整
        if len(content) < 10:
            confidence *= 0.8
        elif len(content) > 200:
            confidence *= 0.9
        
        # 根据逻辑词汇调整
        logical_words = ["因此", "所以", "由于", "根据", "假设", "证明"]
        if any(word in content for word in logical_words):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _analyze_step_dependencies(self, steps: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """分析步骤间的依赖关系"""
        for i, step in enumerate(steps):
            dependencies = []
            
            # 查找引用关系
            references = self._find_references(step["content"], steps[:i])
            dependencies.extend(references)
            
            # 查找逻辑依赖
            logical_deps = self._find_logical_dependencies(step, steps[:i])
            dependencies.extend(logical_deps)
            
            step["dependencies"] = list(set(dependencies))
        
        return steps
    
    def _find_references(self, content: str, previous_steps: List[Dict[str, Any]]) -> List[int]:
        """查找引用关系"""
        references = []
        
        # 查找明确的引用
        ref_patterns = [
            r'根据步骤(\d+)',
            r'由步骤(\d+)',
            r'从(\d+)可知',
            r'结合前面',
            r'根据上述',
            r'由此'
        ]
        
        for pattern in ref_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if match.groups() and match.group(1).isdigit():
                    ref_step = int(match.group(1))
                    if any(step["step_number"] == ref_step for step in previous_steps):
                        references.append(ref_step)
                else:
                    # 隐式引用，添加前一步
                    if previous_steps:
                        references.append(previous_steps[-1]["step_number"])
        
        return references
    
    def _find_logical_dependencies(self, current_step: Dict[str, Any], 
                                 previous_steps: List[Dict[str, Any]]) -> List[int]:
        """查找逻辑依赖关系"""
        dependencies = []
        current_type = current_step["type"]
        current_role = current_step["reasoning_role"]
        
        # 结论依赖于前提
        if current_type == "conclusion":
            for step in reversed(previous_steps[-3:]):  # 检查最近3步
                if step["type"] in ["premise", "assumption", "derivation"]:
                    dependencies.append(step["step_number"])
                    break
        
        # 推导依赖于前提或假设
        elif current_type == "derivation":
            for step in reversed(previous_steps[-2:]):  # 检查最近2步
                if step["type"] in ["premise", "assumption"]:
                    dependencies.append(step["step_number"])
        
        # 应用依赖于定义或分析
        elif current_role == "application":
            for step in reversed(previous_steps):
                if step["reasoning_role"] in ["definition", "analysis"]:
                    dependencies.append(step["step_number"])
                    break
        
        return dependencies    

    def _deduplicate_and_number_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重和重新编号步骤"""
        seen_content = set()
        unique_steps = []
        
        for step in steps:
            content_key = step["content"].lower().strip()
            if content_key not in seen_content:
                seen_content.add(content_key)
                step["step_number"] = len(unique_steps) + 1
                unique_steps.append(step)
        
        return unique_steps
    
    def _get_sequential_number(self, indicator: str) -> int:
        """获取序列指示词对应的数字"""
        sequence_map = {
            "首先": 1, "第一": 1, "一": 1,
            "其次": 2, "第二": 2, "二": 2, "然后": 2,
            "再次": 3, "第三": 3, "三": 3, "接着": 3,
            "最后": 4, "最终": 4, "四": 4, "终于": 4
        }
        return sequence_map.get(indicator, 0)
    
    def _build_step_patterns(self) -> Dict[str, List[str]]:
        """构建步骤模式"""
        return {
            "numbered": [
                r'(\d+)[、.]\s*(.+?)(?=\d+[、.]|$)',
                r'步骤(\d+)[：:]\s*(.+?)(?=步骤\d+|$)',
                r'第(\d+)步[：:]\s*(.+?)(?=第\d+步|$)'
            ],
            "sequential": [
                r'(首先|第一)[，,：:]\s*(.+?)(?=其次|第二|然后|$)',
                r'(其次|第二|然后)[，,：:]\s*(.+?)(?=再次|第三|接着|最后|$)',
                r'(再次|第三|接着)[，,：:]\s*(.+?)(?=最后|最终|$)',
                r'(最后|最终)[，,：:]\s*(.+?)(?=$)'
            ]
        }
    
    def _build_transition_indicators(self) -> Dict[str, List[str]]:
        """构建转换指示词"""
        return {
            "sequential": ["首先", "其次", "然后", "接着", "最后", "最终"],
            "causal": ["因此", "所以", "由此", "故而", "从而"],
            "conditional": ["如果", "假如", "当", "若", "倘若"],
            "adversative": ["但是", "然而", "不过", "可是"],
            "additive": ["而且", "并且", "此外", "另外"],
            "explanatory": ["即", "也就是说", "换句话说", "具体来说"]
        }


class ReasoningChainValidator:
    """推理链验证器"""
    
    def __init__(self):
        """初始化推理链验证器"""
        self.validation_rules = self._build_validation_rules()
    
    def validate_reasoning_chain(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证推理链
        
        Args:
            steps: 推理步骤列表
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_result = {
            "is_valid": True,
            "completeness_score": 0.0,
            "coherence_score": 0.0,
            "logical_validity_score": 0.0,
            "step_quality_scores": [],
            "issues": [],
            "missing_steps": [],
            "redundant_steps": [],
            "logical_gaps": []
        }
        
        if not steps:
            validation_result["is_valid"] = False
            validation_result["issues"].append("没有找到推理步骤")
            return validation_result
        
        # 验证完整性
        completeness = self._validate_completeness(steps)
        validation_result.update(completeness)
        
        # 验证连贯性
        coherence = self._validate_coherence(steps)
        validation_result.update(coherence)
        
        # 验证逻辑有效性
        logical_validity = self._validate_logical_validity(steps)
        validation_result.update(logical_validity)
        
        # 验证每个步骤的质量
        step_qualities = self._validate_step_qualities(steps)
        validation_result["step_quality_scores"] = step_qualities
        
        # 计算总体有效性
        avg_score = (validation_result["completeness_score"] + 
                    validation_result["coherence_score"] + 
                    validation_result["logical_validity_score"]) / 3
        
        validation_result["is_valid"] = avg_score >= 0.6
        
        return validation_result
    
    def _validate_completeness(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证推理完整性"""
        result = {"completeness_score": 0.0, "missing_steps": []}
        
        # 检查基本推理结构
        has_premise = any(step["type"] in ["premise", "assumption"] for step in steps)
        has_reasoning = any(step["type"] in ["derivation", "analysis"] for step in steps)
        has_conclusion = any(step["type"] == "conclusion" for step in steps)
        
        structure_score = (has_premise + has_reasoning + has_conclusion) / 3.0
        
        # 检查步骤连续性
        step_numbers = [step["step_number"] for step in steps if step["step_number"] > 0]
        if step_numbers:
            expected_steps = set(range(1, max(step_numbers) + 1))
            actual_steps = set(step_numbers)
            missing_numbers = expected_steps - actual_steps
            
            if missing_numbers:
                result["missing_steps"].extend([f"步骤{num}" for num in sorted(missing_numbers)])
            
            continuity_score = len(actual_steps) / len(expected_steps) if expected_steps else 1.0
        else:
            continuity_score = 0.5
        
        # 检查依赖关系完整性
        dependency_score = self._check_dependency_completeness(steps)
        
        result["completeness_score"] = (structure_score + continuity_score + dependency_score) / 3.0
        
        return result
    
    def _validate_coherence(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证推理连贯性"""
        result = {"coherence_score": 0.0}
        
        if len(steps) <= 1:
            result["coherence_score"] = 1.0
            return result
        
        coherence_scores = []
        
        for i in range(1, len(steps)):
            current_step = steps[i]
            previous_step = steps[i-1]
            
            # 检查步骤间的逻辑连接
            connection_score = self._evaluate_step_connection(previous_step, current_step)
            coherence_scores.append(connection_score)
        
        result["coherence_score"] = sum(coherence_scores) / len(coherence_scores)
        
        return result
    
    def _validate_logical_validity(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证逻辑有效性"""
        result = {"logical_validity_score": 0.0, "logical_gaps": []}
        
        validity_scores = []
        
        for step in steps:
            step_validity = self._evaluate_step_validity(step, steps)
            validity_scores.append(step_validity["score"])
            
            if step_validity["issues"]:
                result["logical_gaps"].extend(step_validity["issues"])
        
        result["logical_validity_score"] = sum(validity_scores) / len(validity_scores) if validity_scores else 0.0
        
        return result
    
    def _validate_step_qualities(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证步骤质量"""
        qualities = []
        
        for step in steps:
            quality = {
                "step_number": step["step_number"],
                "clarity_score": self._evaluate_step_clarity(step),
                "relevance_score": self._evaluate_step_relevance(step, steps),
                "necessity_score": self._evaluate_step_necessity(step, steps),
                "overall_score": 0.0
            }
            
            quality["overall_score"] = (quality["clarity_score"] + 
                                      quality["relevance_score"] + 
                                      quality["necessity_score"]) / 3.0
            
            qualities.append(quality)
        
        return qualities    

    def _check_dependency_completeness(self, steps: List[Dict[str, Any]]) -> float:
        """检查依赖关系完整性"""
        if not steps:
            return 0.0
        
        total_deps = 0
        satisfied_deps = 0
        
        for step in steps:
            dependencies = step.get("dependencies", [])
            total_deps += len(dependencies)
            
            for dep in dependencies:
                if any(s["step_number"] == dep for s in steps):
                    satisfied_deps += 1
        
        return satisfied_deps / total_deps if total_deps > 0 else 1.0
    
    def _evaluate_step_connection(self, prev_step: Dict[str, Any], 
                                curr_step: Dict[str, Any]) -> float:
        """评估步骤间连接"""
        # 检查类型转换的合理性
        type_transitions = {
            ("assumption", "premise"): 0.8,
            ("premise", "derivation"): 0.9,
            ("derivation", "conclusion"): 0.9,
            ("analysis", "application"): 0.8,
            ("definition", "application"): 0.7
        }
        
        transition_key = (prev_step["type"], curr_step["type"])
        base_score = type_transitions.get(transition_key, 0.5)
        
        # 检查依赖关系
        if prev_step["step_number"] in curr_step.get("dependencies", []):
            base_score += 0.2
        
        # 检查内容相关性
        content_similarity = self._calculate_content_similarity(
            prev_step["content"], curr_step["content"]
        )
        
        return min(1.0, base_score + content_similarity * 0.1)
    
    def _evaluate_step_validity(self, step: Dict[str, Any], 
                              all_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估步骤有效性"""
        result = {"score": 0.5, "issues": []}
        
        step_type = step["type"]
        content = step["content"]
        
        # 检查步骤类型的合理性
        if step_type == "conclusion":
            # 结论应该有前提支持
            has_premise = any(s["type"] in ["premise", "assumption", "derivation"] 
                            for s in all_steps if s["step_number"] < step["step_number"])
            if not has_premise:
                result["issues"].append(f"步骤{step['step_number']}的结论缺乏前提支持")
                result["score"] *= 0.7
        
        elif step_type == "derivation":
            # 推导应该基于前面的步骤
            if not step.get("dependencies"):
                result["issues"].append(f"步骤{step['step_number']}的推导缺乏依据")
                result["score"] *= 0.8
        
        # 检查内容质量
        if len(content) < 10:
            result["issues"].append(f"步骤{step['step_number']}内容过于简短")
            result["score"] *= 0.9
        
        # 检查逻辑一致性
        if any(word in content.lower() for word in ["矛盾", "冲突", "不一致"]):
            result["issues"].append(f"步骤{step['step_number']}可能存在逻辑矛盾")
            result["score"] *= 0.6
        
        return result
    
    def _evaluate_step_clarity(self, step: Dict[str, Any]) -> float:
        """评估步骤清晰度"""
        content = step["content"]
        
        # 基于长度的清晰度
        length_score = 1.0
        if len(content) < 10:
            length_score = 0.6
        elif len(content) > 200:
            length_score = 0.8
        
        # 基于复杂度的清晰度
        complexity_score = 1.0
        nested_clauses = content.count('，') + content.count('；')
        if nested_clauses > 5:
            complexity_score = 0.7
        
        # 基于专业术语的清晰度
        technical_terms = ["算法", "模型", "数据", "训练", "优化"]
        term_count = sum(1 for term in technical_terms if term in content)
        term_score = min(1.0, 0.5 + term_count * 0.1)
        
        return (length_score + complexity_score + term_score) / 3.0
    
    def _evaluate_step_relevance(self, step: Dict[str, Any], 
                               all_steps: List[Dict[str, Any]]) -> float:
        """评估步骤相关性"""
        if len(all_steps) <= 1:
            return 1.0
        
        content = step["content"]
        other_contents = [s["content"] for s in all_steps if s != step]
        
        # 计算与其他步骤的相关性
        relevance_scores = []
        for other_content in other_contents:
            similarity = self._calculate_content_similarity(content, other_content)
            relevance_scores.append(similarity)
        
        # 相关性应该适中，既不能太低（无关）也不能太高（重复）
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        if 0.2 <= avg_relevance <= 0.7:
            return 1.0
        elif avg_relevance < 0.2:
            return 0.5  # 太不相关
        else:
            return 0.7  # 太相似，可能重复
    
    def _evaluate_step_necessity(self, step: Dict[str, Any], 
                               all_steps: List[Dict[str, Any]]) -> float:
        """评估步骤必要性"""
        # 检查是否有其他步骤依赖于此步骤
        step_number = step["step_number"]
        dependents = [s for s in all_steps 
                     if step_number in s.get("dependencies", [])]
        
        if dependents:
            return 1.0  # 有依赖，必要
        
        # 检查步骤类型的必要性
        step_type = step["type"]
        if step_type in ["conclusion", "premise", "assumption"]:
            return 0.9  # 关键步骤类型
        elif step_type in ["derivation", "analysis"]:
            return 0.8  # 重要步骤类型
        else:
            return 0.6  # 一般步骤类型
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度"""
        # 简单的词汇重叠相似度
        words1 = set(re.findall(r'\w+', content1.lower()))
        words2 = set(re.findall(r'\w+', content2.lower()))
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _build_validation_rules(self) -> Dict[str, Any]:
        """构建验证规则"""
        return {
            "required_step_types": ["premise", "reasoning", "conclusion"],
            "valid_transitions": {
                "assumption": ["premise", "derivation"],
                "premise": ["derivation", "analysis", "conclusion"],
                "derivation": ["conclusion", "application"],
                "analysis": ["synthesis", "evaluation", "conclusion"],
                "conclusion": []
            },
            "quality_thresholds": {
                "clarity": 0.6,
                "relevance": 0.5,
                "necessity": 0.4
            }
        }


class MultiStepReasoningEvaluator(AbstractEvaluator):
    """多步推理能力评估器"""
    
    def __init__(self, name: str = "multi_step_reasoning", weight: float = 1.0):
        """
        初始化多步推理评估器
        
        Args:
            name: 评估器名称
            weight: 评估器权重
        """
        super().__init__(name, weight)
        self.step_extractor = ReasoningStepExtractor()
        self.chain_validator = ReasoningChainValidator()
    
    def _initialize_criteria(self) -> List[Criterion]:
        """初始化评估标准"""
        return [
            Criterion(
                name="step_extraction_quality",
                description="推理步骤提取质量",
                weight=0.25,
                threshold=0.7,
                evaluation_method="step_extraction"
            ),
            Criterion(
                name="reasoning_completeness",
                description="推理完整性",
                weight=0.25,
                threshold=0.6,
                evaluation_method="completeness_check"
            ),
            Criterion(
                name="logical_coherence",
                description="逻辑连贯性",
                weight=0.25,
                threshold=0.7,
                evaluation_method="coherence_analysis"
            ),
            Criterion(
                name="step_validity",
                description="步骤有效性",
                weight=0.25,
                threshold=0.6,
                evaluation_method="validity_check"
            )
        ]
    
    def _calculate_score(self, input_text: str, model_output: str, 
                        expected_output: str, context: Dict[str, Any]) -> float:
        """计算多步推理评估分数"""
        # 提取推理步骤
        reasoning_steps = self.step_extractor.extract_reasoning_steps(model_output)
        
        # 验证推理链
        validation_result = self.chain_validator.validate_reasoning_chain(reasoning_steps)
        
        # 计算各项分数
        extraction_score = self._evaluate_extraction_quality(reasoning_steps)
        completeness_score = validation_result["completeness_score"]
        coherence_score = validation_result["coherence_score"]
        validity_score = validation_result["logical_validity_score"]
        
        # 加权计算总分
        total_score = (extraction_score * 0.25 + 
                      completeness_score * 0.25 + 
                      coherence_score * 0.25 + 
                      validity_score * 0.25)
        
        return total_score
    
    def _evaluate_extraction_quality(self, steps: List[Dict[str, Any]]) -> float:
        """评估步骤提取质量"""
        if not steps:
            return 0.0
        
        # 基于步骤数量的评分
        step_count_score = min(1.0, len(steps) / 5.0)  # 理想情况下有5个步骤
        
        # 基于步骤置信度的评分
        confidence_scores = [step["confidence"] for step in steps]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # 基于步骤类型多样性的评分
        step_types = set(step["type"] for step in steps)
        diversity_score = min(1.0, len(step_types) / 4.0)  # 理想情况下有4种类型
        
        return (step_count_score + avg_confidence + diversity_score) / 3.0    
  
  def _generate_details(self, input_text: str, model_output: str, 
                         expected_output: str, context: Dict[str, Any], 
                         score: float) -> Dict[str, Any]:
        """生成详细评估信息"""
        reasoning_steps = self.step_extractor.extract_reasoning_steps(model_output)
        validation_result = self.chain_validator.validate_reasoning_chain(reasoning_steps)
        
        # 如果有期望输出，也分析期望的推理步骤
        expected_steps = []
        if expected_output:
            expected_steps = self.step_extractor.extract_reasoning_steps(expected_output)
        
        # 计算各项分数
        extraction_score = self._evaluate_extraction_quality(reasoning_steps)
        
        return {
            "evaluator": self.name,
            "score": score,
            "reasoning_steps": reasoning_steps,
            "expected_steps": expected_steps,
            "validation_result": validation_result,
            "extraction_score": extraction_score,
            "completeness_score": validation_result["completeness_score"],
            "coherence_score": validation_result["coherence_score"],
            "validity_score": validation_result["logical_validity_score"],
            "step_count": len(reasoning_steps),
            "step_types": list(set(step["type"] for step in reasoning_steps)),
            "reasoning_roles": list(set(step["reasoning_role"] for step in reasoning_steps)),
            "dependency_analysis": self._analyze_dependencies(reasoning_steps),
            "quality_analysis": {
                "high_quality_steps": [s for s in validation_result["step_quality_scores"] if s["overall_score"] >= 0.8],
                "low_quality_steps": [s for s in validation_result["step_quality_scores"] if s["overall_score"] < 0.5],
                "avg_step_quality": sum(s["overall_score"] for s in validation_result["step_quality_scores"]) / len(validation_result["step_quality_scores"]) if validation_result["step_quality_scores"] else 0.0
            },
            "issues_summary": {
                "missing_steps": validation_result["missing_steps"],
                "logical_gaps": validation_result["logical_gaps"],
                "total_issues": len(validation_result["issues"])
            },
            "recommendations": self._generate_reasoning_recommendations(reasoning_steps, validation_result)
        }
    
    def _analyze_dependencies(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析依赖关系"""
        if not steps:
            return {"total_dependencies": 0, "dependency_chains": [], "orphaned_steps": []}
        
        total_deps = sum(len(step.get("dependencies", [])) for step in steps)
        
        # 构建依赖链
        dependency_chains = []
        visited = set()
        
        for step in steps:
            if step["step_number"] not in visited:
                chain = self._trace_dependency_chain(step, steps, visited)
                if len(chain) > 1:
                    dependency_chains.append(chain)
        
        # 找出孤立步骤（没有依赖也不被依赖）
        all_deps = set()
        for step in steps:
            all_deps.update(step.get("dependencies", []))
        
        step_numbers = {step["step_number"] for step in steps}
        orphaned_steps = []
        
        for step in steps:
            step_num = step["step_number"]
            has_dependencies = bool(step.get("dependencies"))
            is_depended_on = step_num in all_deps
            
            if not has_dependencies and not is_depended_on:
                orphaned_steps.append(step_num)
        
        return {
            "total_dependencies": total_deps,
            "dependency_chains": dependency_chains,
            "orphaned_steps": orphaned_steps,
            "avg_dependencies_per_step": total_deps / len(steps)
        }
    
    def _trace_dependency_chain(self, start_step: Dict[str, Any], 
                              all_steps: List[Dict[str, Any]], 
                              visited: Set[int]) -> List[int]:
        """追踪依赖链"""
        chain = []
        queue = deque([start_step["step_number"]])
        
        while queue:
            current_num = queue.popleft()
            if current_num in visited:
                continue
            
            visited.add(current_num)
            chain.append(current_num)
            
            # 找到依赖于当前步骤的步骤
            for step in all_steps:
                if current_num in step.get("dependencies", []):
                    queue.append(step["step_number"])
        
        return chain
    
    def _generate_reasoning_recommendations(self, steps: List[Dict[str, Any]], 
                                          validation_result: Dict[str, Any]) -> List[str]:
        """生成推理改进建议"""
        recommendations = []
        
        # 基于完整性的建议
        if validation_result["completeness_score"] < 0.6:
            if not any(step["type"] == "premise" for step in steps):
                recommendations.append("建议明确提出推理的前提条件或假设")
            
            if not any(step["type"] == "conclusion" for step in steps):
                recommendations.append("建议明确给出推理的结论")
            
            if validation_result["missing_steps"]:
                recommendations.append(f"推理过程可能缺少步骤：{', '.join(validation_result['missing_steps'][:3])}")
        
        # 基于连贯性的建议
        if validation_result["coherence_score"] < 0.6:
            recommendations.append("建议增加步骤间的逻辑连接词，提高推理的连贯性")
            recommendations.append("建议明确说明每个步骤与前面步骤的关系")
        
        # 基于有效性的建议
        if validation_result["logical_validity_score"] < 0.6:
            if validation_result["logical_gaps"]:
                recommendations.append("发现逻辑缺陷，建议检查推理的逻辑有效性")
        
        # 基于步骤质量的建议
        step_qualities = validation_result["step_quality_scores"]
        if step_qualities:
            low_quality_steps = [s for s in step_qualities if s["overall_score"] < 0.5]
            if low_quality_steps:
                step_nums = [str(s["step_number"]) for s in low_quality_steps[:3]]
                recommendations.append(f"步骤 {', '.join(step_nums)} 的质量较低，建议改进表述和逻辑")
        
        # 基于步骤数量的建议
        if len(steps) < 3:
            recommendations.append("推理步骤较少，建议增加中间推理过程，使论证更充分")
        elif len(steps) > 10:
            recommendations.append("推理步骤过多，建议合并相似步骤，简化推理过程")
        
        # 基于依赖关系的建议
        dependency_analysis = self._analyze_dependencies(steps)
        if dependency_analysis["orphaned_steps"]:
            recommendations.append("发现孤立的推理步骤，建议明确其与整体推理的关系")
        
        if dependency_analysis["avg_dependencies_per_step"] < 0.3:
            recommendations.append("步骤间缺乏明确的依赖关系，建议加强步骤间的逻辑联系")
        
        return recommendations[:8]  # 最多返回8条建议
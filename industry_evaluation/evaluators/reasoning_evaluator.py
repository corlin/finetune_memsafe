"""
逻辑推理评估器
"""

import re
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict
from industry_evaluation.evaluators.base_evaluator import AbstractEvaluator
from industry_evaluation.models.data_models import Criterion


class CausalRelationExtractor:
    """因果关系提取器"""
    
    def __init__(self):
        """初始化因果关系提取器"""
        self.causal_patterns = self._build_causal_patterns()
        self.causal_indicators = self._build_causal_indicators()
    
    def extract_causal_relations(self, text: str) -> List[Dict[str, Any]]:
        """
        提取文本中的因果关系
        
        Args:
            text: 输入文本
            
        Returns:
            List[Dict[str, Any]]: 因果关系列表
        """
        causal_relations = []
        
        # 使用模式匹配提取因果关系
        for pattern_type, patterns in self.causal_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    relation = self._parse_causal_match(match, pattern_type, text)
                    if relation:
                        causal_relations.append(relation)
        
        # 使用指示词提取因果关系
        indicator_relations = self._extract_by_indicators(text)
        causal_relations.extend(indicator_relations)
        
        # 去重和排序
        causal_relations = self._deduplicate_relations(causal_relations)
        
        return causal_relations
    
    def _parse_causal_match(self, match, pattern_type: str, text: str) -> Optional[Dict[str, Any]]:
        """解析因果关系匹配结果"""
        try:
            if len(match.groups()) >= 2:
                cause = match.group(1).strip()
                effect = match.group(2).strip()
            else:
                return None
            
            # 清理提取的因果实体
            cause = self._clean_causal_entity(cause)
            effect = self._clean_causal_entity(effect)
            
            if not cause or not effect:
                return None
            
            return {
                "cause": cause,
                "effect": effect,
                "pattern_type": pattern_type,
                "confidence": self._calculate_confidence(pattern_type, cause, effect),
                "text_span": match.group(0),
                "start_pos": match.start(),
                "end_pos": match.end(),
                "direction": "forward"  # 因果方向
            }
        except Exception:
            return None
    
    def _extract_by_indicators(self, text: str) -> List[Dict[str, Any]]:
        """使用指示词提取因果关系"""
        relations = []
        sentences = re.split(r'[。！？.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查因果指示词
            for indicator_type, indicators in self.causal_indicators.items():
                for indicator in indicators:
                    if indicator in sentence:
                        relation = self._extract_from_sentence(sentence, indicator, indicator_type)
                        if relation:
                            relations.append(relation)
        
        return relations
    
    def _extract_from_sentence(self, sentence: str, indicator: str, indicator_type: str) -> Optional[Dict[str, Any]]:
        """从句子中提取因果关系"""
        # 根据指示词类型确定因果方向和提取策略
        if indicator_type == "cause_first":
            # 原因在前，如"由于A，所以B"
            parts = sentence.split(indicator)
            if len(parts) >= 2:
                cause = parts[0].strip()
                effect = parts[1].strip()
                
                # 清理连接词
                effect = re.sub(r'^(所以|因此|导致|使得|造成)', '', effect).strip()
                
                return {
                    "cause": cause,
                    "effect": effect,
                    "pattern_type": "indicator_based",
                    "confidence": 0.7,
                    "text_span": sentence,
                    "start_pos": 0,
                    "end_pos": len(sentence),
                    "direction": "forward",
                    "indicator": indicator
                }
        
        elif indicator_type == "effect_first":
            # 结果在前，如"A是因为B"
            parts = sentence.split(indicator)
            if len(parts) >= 2:
                effect = parts[0].strip()
                cause = parts[1].strip()
                
                return {
                    "cause": cause,
                    "effect": effect,
                    "pattern_type": "indicator_based",
                    "confidence": 0.6,
                    "text_span": sentence,
                    "start_pos": 0,
                    "end_pos": len(sentence),
                    "direction": "backward",
                    "indicator": indicator
                }
        
        return None
    
    def _clean_causal_entity(self, entity: str) -> str:
        """清理因果实体"""
        # 移除常见的停用词和标点符号
        entity = re.sub(r'^(的|一个|这个|那个|某个|当|如果|假如)\s*', '', entity)
        entity = re.sub(r'\s*(的|了|等|等等|时|候)$', '', entity)
        entity = re.sub(r'[，,。.！!？?；;：:]', '', entity)
        entity = entity.strip()
        
        # 移除过短或过长的实体
        if len(entity) < 2 or len(entity) > 100:
            return ""
        
        return entity
    
    def _calculate_confidence(self, pattern_type: str, cause: str, effect: str) -> float:
        """计算置信度"""
        base_confidence = {
            "explicit_causal": 0.9,
            "conditional": 0.8,
            "temporal": 0.6,
            "correlational": 0.5,
            "indicator_based": 0.7
        }
        
        confidence = base_confidence.get(pattern_type, 0.5)
        
        # 根据因果实体的质量调整置信度
        if len(cause) < 5 or len(effect) < 5:
            confidence *= 0.8
        
        if any(word in cause.lower() for word in ["可能", "也许", "或许", "大概"]):
            confidence *= 0.7
        
        return min(1.0, confidence)
    
    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重因果关系"""
        seen_relations = set()
        unique_relations = []
        
        for relation in relations:
            # 创建关系的唯一标识
            key = (relation["cause"].lower(), relation["effect"].lower())
            
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(relation)
            else:
                # 如果有重复，保留置信度更高的
                for i, existing in enumerate(unique_relations):
                    existing_key = (existing["cause"].lower(), existing["effect"].lower())
                    if existing_key == key and relation["confidence"] > existing["confidence"]:
                        unique_relations[i] = relation
                        break
        
        return sorted(unique_relations, key=lambda x: x["confidence"], reverse=True)
    
    def _build_causal_patterns(self) -> Dict[str, List[str]]:
        """构建因果关系模式"""
        return {
            "explicit_causal": [
                r'(.+?)\s*导致\s*(.+?)(?:[，。]|$)',
                r'(.+?)\s*引起\s*(.+?)(?:[，。]|$)',
                r'(.+?)\s*造成\s*(.+?)(?:[，。]|$)',
                r'(.+?)\s*产生\s*(.+?)(?:[，。]|$)',
                r'(.+?)\s*带来\s*(.+?)(?:[，。]|$)',
                r'(.+?)\s*使得\s*(.+?)(?:[，。]|$)',
                r'(.+?)\s*促使\s*(.+?)(?:[，。]|$)',
                r'(.+?)\s*导致了\s*(.+?)(?:[，。]|$)'
            ],
            "conditional": [
                r'如果\s*(.+?)\s*[，,]\s*(?:那么|则)\s*(.+?)(?:[，。]|$)',
                r'假如\s*(.+?)\s*[，,]\s*(?:就会|将会)\s*(.+?)(?:[，。]|$)',
                r'当\s*(.+?)\s*时\s*[，,]\s*(.+?)(?:[，。]|$)',
                r'一旦\s*(.+?)\s*[，,]\s*(.+?)(?:[，。]|$)',
                r'只要\s*(.+?)\s*[，,]\s*(?:就|便)\s*(.+?)(?:[，。]|$)'
            ],
            "temporal": [
                r'(.+?)\s*之后\s*[，,]\s*(.+?)(?:[，。]|$)',
                r'(.+?)\s*以后\s*[，,]\s*(.+?)(?:[，。]|$)',
                r'随着\s*(.+?)\s*[，,]\s*(.+?)(?:[，。]|$)',
                r'在\s*(.+?)\s*的基础上\s*[，,]\s*(.+?)(?:[，。]|$)'
            ],
            "correlational": [
                r'(.+?)\s*与\s*(.+?)\s*相关',
                r'(.+?)\s*影响\s*(.+?)(?:[，。]|$)',
                r'(.+?)\s*决定\s*(.+?)(?:[，。]|$)',
                r'(.+?)\s*取决于\s*(.+?)(?:[，。]|$)'
            ]
        }
    
    def _build_causal_indicators(self) -> Dict[str, List[str]]:
        """构建因果指示词"""
        return {
            "cause_first": [
                "由于", "因为", "鉴于", "基于", "考虑到", "源于", "出于"
            ],
            "effect_first": [
                "是因为", "是由于", "源于", "来自于", "归因于"
            ],
            "bidirectional": [
                "因此", "所以", "故而", "于是", "从而", "进而", "因而"
            ]
        }


class ReasoningChainAnalyzer:
    """推理链分析器"""
    
    def __init__(self):
        """初始化推理链分析器"""
        self.reasoning_patterns = self._build_reasoning_patterns()
        self.logical_connectors = self._build_logical_connectors()
    
    def analyze_reasoning_chain(self, text: str, causal_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析推理链
        
        Args:
            text: 输入文本
            causal_relations: 因果关系列表
            
        Returns:
            Dict[str, Any]: 推理链分析结果
        """
        analysis = {
            "reasoning_steps": self._extract_reasoning_steps(text),
            "logical_structure": self._analyze_logical_structure(text),
            "causal_chains": self._build_causal_chains(causal_relations),
            "reasoning_quality": self._assess_reasoning_quality(text, causal_relations),
            "logical_consistency": self._check_logical_consistency(causal_relations),
            "completeness": self._assess_completeness(text, causal_relations)
        }
        
        return analysis
    
    def _extract_reasoning_steps(self, text: str) -> List[Dict[str, Any]]:
        """提取推理步骤"""
        steps = []
        
        # 查找明确的步骤标识
        step_patterns = [
            r'(首先|第一)[，,]\s*(.+?)(?:[。！？]|$)',
            r'(其次|第二|然后)[，,]\s*(.+?)(?:[。！？]|$)',
            r'(再次|第三|接着)[，,]\s*(.+?)(?:[。！？]|$)',
            r'(最后|最终|因此)[，,]\s*(.+?)(?:[。！？]|$)',
            r'(\d+)[、.]\s*(.+?)(?:[。！？]|$)'
        ]
        
        for pattern in step_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                step_indicator = match.group(1)
                step_content = match.group(2).strip()
                
                steps.append({
                    "indicator": step_indicator,
                    "content": step_content,
                    "position": match.start(),
                    "type": self._classify_step_type(step_content)
                })
        
        # 如果没有明确步骤，按句子分割
        if not steps:
            sentences = re.split(r'[。！？.!?]', text)
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if sentence:
                    steps.append({
                        "indicator": f"步骤{i+1}",
                        "content": sentence,
                        "position": i,
                        "type": self._classify_step_type(sentence)
                    })
        
        return steps
    
    def _analyze_logical_structure(self, text: str) -> Dict[str, Any]:
        """分析逻辑结构"""
        structure = {
            "premise_conclusion": self._identify_premise_conclusion(text),
            "argument_structure": self._analyze_argument_structure(text),
            "logical_connectors": self._find_logical_connectors(text),
            "reasoning_type": self._classify_reasoning_type(text)
        }
        
        return structure
    
    def _build_causal_chains(self, causal_relations: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """构建因果链"""
        chains = []
        
        if not causal_relations:
            return chains
        
        # 构建因果图
        cause_to_effects = defaultdict(list)
        effect_to_causes = defaultdict(list)
        
        for relation in causal_relations:
            cause = relation["cause"]
            effect = relation["effect"]
            cause_to_effects[cause].append(relation)
            effect_to_causes[effect].append(relation)
        
        # 查找因果链
        visited = set()
        
        for relation in causal_relations:
            if relation["cause"] not in visited:
                chain = self._trace_causal_chain(relation, cause_to_effects, visited)
                if len(chain) > 1:
                    chains.append(chain)
        
        return chains
    
    def _trace_causal_chain(self, start_relation: Dict[str, Any], 
                           cause_to_effects: Dict[str, List[Dict[str, Any]]], 
                           visited: Set[str]) -> List[Dict[str, Any]]:
        """追踪因果链"""
        chain = [start_relation]
        visited.add(start_relation["cause"])
        
        current_effect = start_relation["effect"]
        
        # 继续追踪链条
        while current_effect in cause_to_effects and current_effect not in visited:
            next_relations = cause_to_effects[current_effect]
            if next_relations:
                next_relation = next_relations[0]  # 选择第一个关系
                chain.append(next_relation)
                visited.add(current_effect)
                current_effect = next_relation["effect"]
            else:
                break
        
        return chain
    
    def _assess_reasoning_quality(self, text: str, causal_relations: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估推理质量"""
        quality = {
            "clarity": self._assess_clarity(text),
            "coherence": self._assess_coherence(text),
            "logical_validity": self._assess_logical_validity(causal_relations),
            "evidence_support": self._assess_evidence_support(text),
            "completeness": self._assess_argument_completeness(text)
        }
        
        return quality
    
    def _check_logical_consistency(self, causal_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检查逻辑一致性"""
        consistency = {
            "contradictions": self._find_contradictions(causal_relations),
            "circular_reasoning": self._detect_circular_reasoning(causal_relations),
            "consistency_score": 1.0
        }
        
        # 计算一致性分数
        total_issues = len(consistency["contradictions"]) + len(consistency["circular_reasoning"])
        if causal_relations:
            consistency["consistency_score"] = max(0.0, 1.0 - (total_issues / len(causal_relations)))
        
        return consistency
    
    def _assess_completeness(self, text: str, causal_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估完整性"""
        completeness = {
            "missing_premises": self._identify_missing_premises(text, causal_relations),
            "unsupported_conclusions": self._identify_unsupported_conclusions(text, causal_relations),
            "completeness_score": self._calculate_completeness_score(text, causal_relations)
        }
        
        return completeness
    
    def _classify_step_type(self, step_content: str) -> str:
        """分类步骤类型"""
        if any(word in step_content for word in ["假设", "假定", "设", "令"]):
            return "assumption"
        elif any(word in step_content for word in ["因为", "由于", "根据"]):
            return "premise"
        elif any(word in step_content for word in ["所以", "因此", "故", "得出"]):
            return "conclusion"
        elif any(word in step_content for word in ["证明", "推导", "计算"]):
            return "derivation"
        else:
            return "statement"
    
    def _identify_premise_conclusion(self, text: str) -> Dict[str, List[str]]:
        """识别前提和结论"""
        premises = []
        conclusions = []
        
        # 前提模式
        premise_patterns = [
            r'(假设|假定|设|令)\s*(.+?)(?:[，。]|$)',
            r'(已知|给定|根据)\s*(.+?)(?:[，。]|$)',
            r'(因为|由于|鉴于)\s*(.+?)(?:[，。]|$)'
        ]
        
        for pattern in premise_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                premises.append(match.group(2).strip())
        
        # 结论模式
        conclusion_patterns = [
            r'(所以|因此|故|得出|可知)\s*(.+?)(?:[，。]|$)',
            r'(结论是|总结|综上)\s*(.+?)(?:[，。]|$)'
        ]
        
        for pattern in conclusion_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                conclusions.append(match.group(2).strip())
        
        return {"premises": premises, "conclusions": conclusions}
    
    def _analyze_argument_structure(self, text: str) -> str:
        """分析论证结构"""
        if "如果" in text and "那么" in text:
            return "conditional"
        elif "因为" in text and "所以" in text:
            return "causal"
        elif any(word in text for word in ["比较", "对比", "相比"]):
            return "comparative"
        elif any(word in text for word in ["例如", "比如", "举例"]):
            return "inductive"
        else:
            return "general"
    
    def _find_logical_connectors(self, text: str) -> List[Dict[str, Any]]:
        """查找逻辑连接词"""
        connectors = []
        
        for connector_type, connector_list in self.logical_connectors.items():
            for connector in connector_list:
                if connector in text:
                    positions = [m.start() for m in re.finditer(re.escape(connector), text)]
                    for pos in positions:
                        connectors.append({
                            "connector": connector,
                            "type": connector_type,
                            "position": pos
                        })
        
        return sorted(connectors, key=lambda x: x["position"])
    
    def _classify_reasoning_type(self, text: str) -> str:
        """分类推理类型"""
        if any(word in text for word in ["归纳", "总结", "概括"]):
            return "inductive"
        elif any(word in text for word in ["演绎", "推导", "推论"]):
            return "deductive"
        elif any(word in text for word in ["类比", "相似", "类似"]):
            return "analogical"
        elif any(word in text for word in ["假设", "可能", "或许"]):
            return "abductive"
        else:
            return "general"
    
    def _assess_clarity(self, text: str) -> float:
        """评估清晰度"""
        # 基于句子长度、复杂度等因素评估
        sentences = re.split(r'[。！？.!?]', text)
        avg_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        
        # 理想句子长度在20-50字符之间
        if 20 <= avg_length <= 50:
            clarity = 1.0
        elif avg_length < 20:
            clarity = 0.7 + (avg_length / 20) * 0.3
        else:
            clarity = max(0.3, 1.0 - (avg_length - 50) / 100)
        
        return clarity
    
    def _assess_coherence(self, text: str) -> float:
        """评估连贯性"""
        # 基于逻辑连接词的使用评估连贯性
        connectors = self._find_logical_connectors(text)
        sentences = re.split(r'[。！？.!?]', text)
        
        if len(sentences) <= 1:
            return 1.0
        
        # 连接词密度
        connector_density = len(connectors) / len(sentences)
        
        # 理想密度在0.3-0.7之间
        if 0.3 <= connector_density <= 0.7:
            return 1.0
        elif connector_density < 0.3:
            return 0.5 + connector_density * 1.67
        else:
            return max(0.3, 1.0 - (connector_density - 0.7) * 0.5)
    
    def _assess_logical_validity(self, causal_relations: List[Dict[str, Any]]) -> float:
        """评估逻辑有效性"""
        if not causal_relations:
            return 0.5
        
        valid_relations = sum(1 for r in causal_relations if r["confidence"] > 0.6)
        return valid_relations / len(causal_relations)
    
    def _assess_evidence_support(self, text: str) -> float:
        """评估证据支持"""
        evidence_indicators = ["数据显示", "研究表明", "实验证明", "统计", "调查", "例如", "比如"]
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in text)
        
        # 标准化分数
        return min(1.0, evidence_count / 3.0)
    
    def _assess_argument_completeness(self, text: str) -> float:
        """评估论证完整性"""
        has_premise = any(word in text for word in ["因为", "由于", "根据", "假设"])
        has_conclusion = any(word in text for word in ["所以", "因此", "得出", "结论"])
        has_evidence = any(word in text for word in ["数据", "研究", "实验", "例如"])
        
        completeness = (has_premise + has_conclusion + has_evidence) / 3.0
        return completeness
    
    def _find_contradictions(self, causal_relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """查找矛盾"""
        contradictions = []
        
        for i, rel1 in enumerate(causal_relations):
            for rel2 in causal_relations[i+1:]:
                # 检查直接矛盾：A导致B，同时A导致非B
                if (rel1["cause"] == rel2["cause"] and 
                    self._are_contradictory_effects(rel1["effect"], rel2["effect"])):
                    contradictions.append({
                        "type": "contradictory_effects",
                        "relation1": rel1,
                        "relation2": rel2
                    })
                
                # 检查循环矛盾：A导致B，B导致A
                if (rel1["cause"] == rel2["effect"] and 
                    rel1["effect"] == rel2["cause"]):
                    contradictions.append({
                        "type": "mutual_causation",
                        "relation1": rel1,
                        "relation2": rel2
                    })
        
        return contradictions
    
    def _detect_circular_reasoning(self, causal_relations: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """检测循环推理"""
        circular_chains = []
        
        # 构建因果图
        graph = defaultdict(list)
        for relation in causal_relations:
            graph[relation["cause"]].append(relation)
        
        # 检测循环
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # 找到循环
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                circular_chains.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for relation in graph[node]:
                next_node = relation["effect"]
                dfs(next_node, path + [relation])
            
            rec_stack.remove(node)
        
        for relation in causal_relations:
            if relation["cause"] not in visited:
                dfs(relation["cause"], [])
        
        return circular_chains
    
    def _are_contradictory_effects(self, effect1: str, effect2: str) -> bool:
        """判断两个效果是否矛盾"""
        # 简单的矛盾检测
        contradictory_pairs = [
            ("增加", "减少"), ("提高", "降低"), ("改善", "恶化"),
            ("成功", "失败"), ("有效", "无效"), ("正确", "错误")
        ]
        
        effect1_lower = effect1.lower()
        effect2_lower = effect2.lower()
        
        for pos, neg in contradictory_pairs:
            if (pos in effect1_lower and neg in effect2_lower) or \
               (neg in effect1_lower and pos in effect2_lower):
                return True
        
        return False
    
    def _identify_missing_premises(self, text: str, causal_relations: List[Dict[str, Any]]) -> List[str]:
        """识别缺失的前提"""
        missing_premises = []
        
        # 简化实现：检查是否有未支持的因果关系
        for relation in causal_relations:
            if relation["confidence"] < 0.5:
                missing_premises.append(f"缺少支持'{relation['cause']}导致{relation['effect']}'的前提")
        
        return missing_premises
    
    def _identify_unsupported_conclusions(self, text: str, causal_relations: List[Dict[str, Any]]) -> List[str]:
        """识别未支持的结论"""
        unsupported = []
        
        # 查找结论性语句
        conclusion_patterns = [
            r'(所以|因此|得出|结论是)\s*(.+?)(?:[，。]|$)'
        ]
        
        for pattern in conclusion_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                conclusion = match.group(2).strip()
                
                # 检查是否有因果关系支持这个结论
                supported = any(conclusion in relation["effect"] for relation in causal_relations)
                
                if not supported:
                    unsupported.append(conclusion)
        
        return unsupported
    
    def _calculate_completeness_score(self, text: str, causal_relations: List[Dict[str, Any]]) -> float:
        """计算完整性分数"""
        missing_premises = self._identify_missing_premises(text, causal_relations)
        unsupported_conclusions = self._identify_unsupported_conclusions(text, causal_relations)
        
        total_issues = len(missing_premises) + len(unsupported_conclusions)
        
        if not causal_relations and not missing_premises and not unsupported_conclusions:
            return 1.0
        
        total_elements = len(causal_relations) + len(missing_premises) + len(unsupported_conclusions)
        
        if total_elements == 0:
            return 1.0
        
        return max(0.0, 1.0 - (total_issues / total_elements))
    
    def _build_reasoning_patterns(self) -> Dict[str, List[str]]:
        """构建推理模式"""
        return {
            "deductive": [
                r'如果.*那么.*',
                r'所有.*都.*',
                r'根据.*可以推出.*'
            ],
            "inductive": [
                r'通过.*例子.*可以看出.*',
                r'从.*中可以总结.*',
                r'基于.*数据.*'
            ],
            "analogical": [
                r'.*类似于.*',
                r'.*就像.*一样.*',
                r'.*与.*相似.*'
            ]
        }
    
    def _build_logical_connectors(self) -> Dict[str, List[str]]:
        """构建逻辑连接词"""
        return {
            "causal": ["因为", "由于", "所以", "因此", "导致", "引起"],
            "conditional": ["如果", "假如", "那么", "则", "就"],
            "adversative": ["但是", "然而", "不过", "可是", "虽然"],
            "additive": ["而且", "并且", "此外", "另外", "同时"],
            "temporal": ["首先", "然后", "接着", "最后", "同时"],
            "conclusive": ["总之", "综上", "因此", "所以", "总而言之"]
        }


class CausalReasoningEvaluator(AbstractEvaluator):
    """因果关系推理评估器"""
    
    def __init__(self, name: str = "causal_reasoning", weight: float = 1.0):
        """
        初始化因果关系推理评估器
        
        Args:
            name: 评估器名称
            weight: 评估器权重
        """
        super().__init__(name, weight)
        self.causal_extractor = CausalRelationExtractor()
        self.reasoning_analyzer = ReasoningChainAnalyzer()
    
    def _initialize_criteria(self) -> List[Criterion]:
        """初始化评估标准"""
        return [
            Criterion(
                name="causal_relation_accuracy",
                description="因果关系准确性",
                weight=0.4,
                threshold=0.7,
                evaluation_method="causal_extraction"
            ),
            Criterion(
                name="reasoning_chain_quality",
                description="推理链质量",
                weight=0.3,
                threshold=0.6,
                evaluation_method="reasoning_analysis"
            ),
            Criterion(
                name="logical_consistency",
                description="逻辑一致性",
                weight=0.3,
                threshold=0.8,
                evaluation_method="consistency_check"
            )
        ]
    
    def _calculate_score(self, input_text: str, model_output: str, 
                        expected_output: str, context: Dict[str, Any]) -> float:
        """计算因果关系推理评估分数"""
        # 提取因果关系
        causal_relations = self.causal_extractor.extract_causal_relations(model_output)
        
        # 分析推理链
        reasoning_analysis = self.reasoning_analyzer.analyze_reasoning_chain(model_output, causal_relations)
        
        # 计算各项分数
        causal_accuracy = self._evaluate_causal_accuracy(causal_relations, expected_output)
        reasoning_quality = self._evaluate_reasoning_quality(reasoning_analysis)
        logical_consistency = reasoning_analysis["logical_consistency"]["consistency_score"]
        
        # 加权计算总分
        total_score = (causal_accuracy * 0.4 + 
                      reasoning_quality * 0.3 + 
                      logical_consistency * 0.3)
        
        return total_score
    
    def _evaluate_causal_accuracy(self, causal_relations: List[Dict[str, Any]], 
                                 expected_output: str) -> float:
        """评估因果关系准确性"""
        if not causal_relations:
            return 0.3  # 没有因果关系，给较低分数
        
        # 基于置信度评估准确性
        confidence_scores = [relation["confidence"] for relation in causal_relations]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # 如果有期望输出，检查因果关系是否与期望一致
        if expected_output:
            expected_relations = self.causal_extractor.extract_causal_relations(expected_output)
            consistency_score = self._calculate_relation_consistency(causal_relations, expected_relations)
            return (avg_confidence + consistency_score) / 2
        
        return avg_confidence
    
    def _calculate_relation_consistency(self, output_relations: List[Dict[str, Any]], 
                                      expected_relations: List[Dict[str, Any]]) -> float:
        """计算关系一致性"""
        if not expected_relations:
            return 0.5
        
        # 简单的一致性检查：检查因果对是否匹配
        output_pairs = {(r["cause"], r["effect"]) for r in output_relations}
        expected_pairs = {(r["cause"], r["effect"]) for r in expected_relations}
        
        if not expected_pairs:
            return 1.0 if not output_pairs else 0.5
        
        intersection = output_pairs.intersection(expected_pairs)
        union = output_pairs.union(expected_pairs)
        
        return len(intersection) / len(union) if union else 1.0
    
    def _evaluate_reasoning_quality(self, reasoning_analysis: Dict[str, Any]) -> float:
        """评估推理质量"""
        quality_scores = reasoning_analysis["reasoning_quality"]
        
        # 计算加权平均
        weights = {
            "clarity": 0.2,
            "coherence": 0.2,
            "logical_validity": 0.3,
            "evidence_support": 0.15,
            "completeness": 0.15
        }
        
        total_score = sum(quality_scores[aspect] * weight 
                         for aspect, weight in weights.items())
        
        return total_score
    
    def _generate_details(self, input_text: str, model_output: str, 
                         expected_output: str, context: Dict[str, Any], 
                         score: float) -> Dict[str, Any]:
        """生成详细评估信息"""
        causal_relations = self.causal_extractor.extract_causal_relations(model_output)
        reasoning_analysis = self.reasoning_analyzer.analyze_reasoning_chain(model_output, causal_relations)
        
        # 如果有期望输出，也分析期望的因果关系
        expected_relations = []
        if expected_output:
            expected_relations = self.causal_extractor.extract_causal_relations(expected_output)
        
        # 计算各项分数
        causal_accuracy = self._evaluate_causal_accuracy(causal_relations, expected_output)
        reasoning_quality = self._evaluate_reasoning_quality(reasoning_analysis)
        
        return {
            "evaluator": self.name,
            "score": score,
            "causal_relations": causal_relations,
            "expected_relations": expected_relations,
            "reasoning_analysis": reasoning_analysis,
            "causal_accuracy": causal_accuracy,
            "reasoning_quality": reasoning_quality,
            "logical_consistency": reasoning_analysis["logical_consistency"]["consistency_score"],
            "quality_breakdown": reasoning_analysis["reasoning_quality"],
            "causal_chains": reasoning_analysis["causal_chains"],
            "reasoning_steps": reasoning_analysis["reasoning_steps"],
            "issues": {
                "contradictions": reasoning_analysis["logical_consistency"]["contradictions"],
                "circular_reasoning": reasoning_analysis["logical_consistency"]["circular_reasoning"],
                "missing_premises": reasoning_analysis["completeness"]["missing_premises"],
                "unsupported_conclusions": reasoning_analysis["completeness"]["unsupported_conclusions"]
            },
            "recommendations": self._generate_reasoning_recommendations(reasoning_analysis, causal_relations)
        }
    
    def _generate_reasoning_recommendations(self, reasoning_analysis: Dict[str, Any], 
                                          causal_relations: List[Dict[str, Any]]) -> List[str]:
        """生成推理改进建议"""
        recommendations = []
        
        # 基于推理质量的建议
        quality_scores = reasoning_analysis["reasoning_quality"]
        
        if quality_scores["clarity"] < 0.6:
            recommendations.append("建议使用更清晰简洁的表达，避免过长或过短的句子")
        
        if quality_scores["coherence"] < 0.6:
            recommendations.append("建议增加逻辑连接词，提高论述的连贯性")
        
        if quality_scores["logical_validity"] < 0.6:
            recommendations.append("建议检查因果关系的逻辑有效性，确保推理过程合理")
        
        if quality_scores["evidence_support"] < 0.5:
            recommendations.append("建议增加数据、研究或实例等证据支持论点")
        
        if quality_scores["completeness"] < 0.6:
            recommendations.append("建议补充必要的前提条件和推理步骤")
        
        # 基于逻辑一致性的建议
        consistency = reasoning_analysis["logical_consistency"]
        
        if consistency["contradictions"]:
            recommendations.append("发现逻辑矛盾，建议检查并消除相互冲突的论述")
        
        if consistency["circular_reasoning"]:
            recommendations.append("发现循环推理，建议重新组织论证结构")
        
        # 基于因果关系的建议
        if not causal_relations:
            recommendations.append("建议明确表达因果关系，使推理过程更加清晰")
        elif len(causal_relations) > 10:
            recommendations.append("因果关系过多，建议聚焦核心关系，简化推理过程")
        
        low_confidence_relations = [r for r in causal_relations if r["confidence"] < 0.5]
        if low_confidence_relations:
            recommendations.append("部分因果关系置信度较低，建议提供更多支持证据")
        
        return recommendations[:6]  # 最多返回6条建议
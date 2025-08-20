"""
专业知识评估器
"""

import re
from typing import Dict, List, Any, Set, Tuple, Optional
from industry_evaluation.evaluators.base_evaluator import AbstractEvaluator
from industry_evaluation.models.data_models import Criterion


class KnowledgeGraphMatcher:
    """知识图谱匹配器"""
    
    def __init__(self, knowledge_base: Optional[Dict[str, Any]] = None):
        """
        初始化知识图谱匹配器
        
        Args:
            knowledge_base: 知识库，包含概念、关系等信息
        """
        self.knowledge_base = knowledge_base or self._get_default_knowledge_base()
        self.concepts = self.knowledge_base.get("concepts", {})
        self.relations = self.knowledge_base.get("relations", {})
        self.facts = self.knowledge_base.get("facts", [])
    
    def extract_concepts(self, text: str) -> Set[str]:
        """
        从文本中提取专业概念
        
        Args:
            text: 输入文本
            
        Returns:
            Set[str]: 提取的概念集合
        """
        extracted_concepts = set()
        
        # 遍历知识库中的概念
        for concept, info in self.concepts.items():
            # 检查概念本身
            if self._match_concept(concept, text):
                extracted_concepts.add(concept)
            
            # 检查概念的同义词
            synonyms = info.get("synonyms", [])
            for synonym in synonyms:
                if self._match_concept(synonym, text):
                    extracted_concepts.add(concept)
        
        return extracted_concepts
    
    def verify_concept_relations(self, concepts: Set[str], text: str) -> Dict[str, Any]:
        """
        验证概念关系的正确性
        
        Args:
            concepts: 概念集合
            text: 文本内容
            
        Returns:
            Dict[str, Any]: 关系验证结果
        """
        verification_result = {
            "correct_relations": [],
            "incorrect_relations": [],
            "missing_relations": [],
            "relation_score": 0.0
        }
        
        # 检查文本中体现的关系
        detected_relations = self._detect_relations_in_text(concepts, text)
        
        # 验证每个检测到的关系
        for relation in detected_relations:
            if self._is_valid_relation(relation):
                verification_result["correct_relations"].append(relation)
            else:
                verification_result["incorrect_relations"].append(relation)
        
        # 检查缺失的重要关系
        expected_relations = self._get_expected_relations(concepts)
        for expected in expected_relations:
            if not self._relation_mentioned_in_text(expected, text):
                verification_result["missing_relations"].append(expected)
        
        # 计算关系分数
        total_relations = len(verification_result["correct_relations"]) + len(verification_result["incorrect_relations"])
        if total_relations > 0:
            verification_result["relation_score"] = len(verification_result["correct_relations"]) / total_relations
        
        return verification_result
    
    def check_fact_consistency(self, text: str) -> Dict[str, Any]:
        """
        检查事实一致性
        
        Args:
            text: 文本内容
            
        Returns:
            Dict[str, Any]: 事实一致性检查结果
        """
        consistency_result = {
            "consistent_facts": [],
            "inconsistent_facts": [],
            "contradictions": [],
            "consistency_score": 1.0
        }
        
        # 检查每个已知事实
        for fact in self.facts:
            fact_check = self._check_single_fact(fact, text)
            
            if fact_check["mentioned"]:
                if fact_check["consistent"]:
                    consistency_result["consistent_facts"].append(fact)
                else:
                    consistency_result["inconsistent_facts"].append(fact)
                    consistency_result["contradictions"].append(fact_check["contradiction"])
        
        # 计算一致性分数
        total_mentioned_facts = len(consistency_result["consistent_facts"]) + len(consistency_result["inconsistent_facts"])
        if total_mentioned_facts > 0:
            consistency_result["consistency_score"] = len(consistency_result["consistent_facts"]) / total_mentioned_facts
        
        return consistency_result
    
    def _match_concept(self, concept: str, text: str) -> bool:
        """匹配概念是否在文本中出现"""
        # 使用词边界匹配，避免部分匹配
        pattern = r'\b' + re.escape(concept) + r'\b'
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    def _detect_relations_in_text(self, concepts: Set[str], text: str) -> List[Dict[str, str]]:
        """检测文本中的概念关系"""
        detected_relations = []
        
        # 简单的关系检测逻辑
        relation_patterns = {
            "is_a": [r"(.+?)\s*是\s*(.+?)", r"(.+?)\s*属于\s*(.+?)"],
            "has_property": [r"(.+?)\s*具有\s*(.+?)", r"(.+?)\s*的\s*(.+?)"],
            "causes": [r"(.+?)\s*导致\s*(.+?)", r"(.+?)\s*引起\s*(.+?)"],
            "part_of": [r"(.+?)\s*包含\s*(.+?)", r"(.+?)\s*组成\s*(.+?)"]
        }
        
        for relation_type, patterns in relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    subject = match.group(1).strip()
                    object_term = match.group(2).strip()
                    
                    # 检查是否涉及已知概念
                    if subject in concepts or object_term in concepts:
                        detected_relations.append({
                            "type": relation_type,
                            "subject": subject,
                            "object": object_term,
                            "text_span": match.group(0)
                        })
        
        return detected_relations
    
    def _is_valid_relation(self, relation: Dict[str, str]) -> bool:
        """验证关系是否有效"""
        relation_key = f"{relation['subject']}_{relation['type']}_{relation['object']}"
        
        # 检查关系是否在知识库中
        if relation_key in self.relations:
            return self.relations[relation_key].get("valid", True)
        
        # 检查反向关系或通用规则
        return self._check_relation_validity(relation)
    
    def _check_relation_validity(self, relation: Dict[str, str]) -> bool:
        """检查关系有效性的通用规则"""
        # 这里可以实现更复杂的关系验证逻辑
        # 目前使用简单的启发式规则
        
        subject = relation["subject"]
        object_term = relation["object"]
        relation_type = relation["type"]
        
        # 检查概念类型匹配
        subject_info = self.concepts.get(subject, {})
        object_info = self.concepts.get(object_term, {})
        
        if relation_type == "is_a":
            # "是"关系：检查类型层次
            subject_type = subject_info.get("type", "")
            object_type = object_info.get("type", "")
            return self._is_subtype(subject_type, object_type)
        
        elif relation_type == "has_property":
            # "具有"关系：检查属性匹配
            subject_properties = subject_info.get("properties", [])
            return object_term in subject_properties
        
        # 默认返回True，表示关系可能有效
        return True
    
    def _is_subtype(self, subtype: str, supertype: str) -> bool:
        """检查类型层次关系"""
        # 简单的类型层次检查
        type_hierarchy = {
            "具体概念": ["抽象概念"],
            "技术概念": ["专业概念"],
            "业务概念": ["专业概念"]
        }
        
        return supertype in type_hierarchy.get(subtype, [])
    
    def _get_expected_relations(self, concepts: Set[str]) -> List[Dict[str, str]]:
        """获取概念间的预期关系"""
        expected_relations = []
        
        # 基于概念组合生成预期关系
        concept_list = list(concepts)
        for i, concept1 in enumerate(concept_list):
            for concept2 in concept_list[i+1:]:
                # 查找这两个概念间的已知关系
                for relation_key, relation_info in self.relations.items():
                    if concept1 in relation_key and concept2 in relation_key:
                        expected_relations.append({
                            "subject": concept1,
                            "object": concept2,
                            "type": relation_info.get("type", "related"),
                            "importance": relation_info.get("importance", 0.5)
                        })
        
        return expected_relations
    
    def _relation_mentioned_in_text(self, relation: Dict[str, str], text: str) -> bool:
        """检查关系是否在文本中被提及"""
        subject = relation["subject"]
        object_term = relation["object"]
        
        # 简单检查：两个概念是否都在文本中出现
        return subject in text and object_term in text
    
    def _check_single_fact(self, fact: Dict[str, Any], text: str) -> Dict[str, Any]:
        """检查单个事实"""
        fact_statement = fact.get("statement", "")
        fact_keywords = fact.get("keywords", [])
        
        # 检查事实是否被提及
        mentioned = any(keyword in text for keyword in fact_keywords)
        
        if not mentioned:
            return {"mentioned": False, "consistent": True, "contradiction": None}
        
        # 检查一致性（简化实现）
        # 这里可以实现更复杂的事实验证逻辑
        consistent = True
        contradiction = None
        
        # 检查是否有明显的矛盾表述
        contradiction_patterns = fact.get("contradiction_patterns", [])
        for pattern in contradiction_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                consistent = False
                contradiction = f"发现与事实矛盾的表述: {pattern}"
                break
        
        return {
            "mentioned": mentioned,
            "consistent": consistent,
            "contradiction": contradiction
        }
    
    def _get_default_knowledge_base(self) -> Dict[str, Any]:
        """获取默认知识库"""
        return {
            "concepts": {
                "机器学习": {
                    "type": "技术概念",
                    "synonyms": ["ML", "机器学习算法"],
                    "properties": ["监督学习", "无监督学习", "强化学习"]
                },
                "深度学习": {
                    "type": "技术概念",
                    "synonyms": ["DL", "神经网络"],
                    "properties": ["多层网络", "反向传播"]
                },
                "自然语言处理": {
                    "type": "技术概念",
                    "synonyms": ["NLP", "文本处理"],
                    "properties": ["分词", "语义分析"]
                },
                "金融风控": {
                    "type": "业务概念",
                    "synonyms": ["风险控制", "风险管理"],
                    "properties": ["信用评估", "反欺诈"]
                }
            },
            "relations": {
                "深度学习_is_a_机器学习": {"valid": True, "type": "is_a", "importance": 0.9},
                "自然语言处理_uses_机器学习": {"valid": True, "type": "uses", "importance": 0.8}
            },
            "facts": [
                {
                    "statement": "深度学习是机器学习的子领域",
                    "keywords": ["深度学习", "机器学习", "子领域"],
                    "contradiction_patterns": [r"深度学习\s*不是\s*机器学习"]
                },
                {
                    "statement": "监督学习需要标注数据",
                    "keywords": ["监督学习", "标注数据", "标签"],
                    "contradiction_patterns": [r"监督学习\s*不需要\s*标注"]
                }
            ]
        }


class KnowledgeEvaluator(AbstractEvaluator):
    """专业知识评估器"""
    
    def __init__(self, name: str = "knowledge", weight: float = 1.0, 
                 knowledge_base: Optional[Dict[str, Any]] = None):
        """
        初始化专业知识评估器
        
        Args:
            name: 评估器名称
            weight: 评估器权重
            knowledge_base: 知识库
        """
        super().__init__(name, weight)
        self.knowledge_matcher = KnowledgeGraphMatcher(knowledge_base)
    
    def _initialize_criteria(self) -> List[Criterion]:
        """初始化评估标准"""
        return [
            Criterion(
                name="concept_accuracy",
                description="专业概念使用准确性",
                weight=0.4,
                threshold=0.7,
                evaluation_method="knowledge_graph_matching"
            ),
            Criterion(
                name="relation_correctness",
                description="概念关系正确性",
                weight=0.3,
                threshold=0.6,
                evaluation_method="relation_verification"
            ),
            Criterion(
                name="fact_consistency",
                description="事实一致性",
                weight=0.3,
                threshold=0.8,
                evaluation_method="fact_checking"
            )
        ]
    
    def _calculate_score(self, input_text: str, model_output: str, 
                        expected_output: str, context: Dict[str, Any]) -> float:
        """计算专业知识评估分数"""
        # 提取概念
        output_concepts = self.knowledge_matcher.extract_concepts(model_output)
        expected_concepts = self.knowledge_matcher.extract_concepts(expected_output)
        
        # 概念准确性评分
        concept_score = self._evaluate_concept_accuracy(output_concepts, expected_concepts)
        
        # 关系正确性评分
        relation_result = self.knowledge_matcher.verify_concept_relations(output_concepts, model_output)
        relation_score = relation_result["relation_score"]
        
        # 事实一致性评分
        fact_result = self.knowledge_matcher.check_fact_consistency(model_output)
        fact_score = fact_result["consistency_score"]
        
        # 加权计算总分
        total_score = (concept_score * 0.4 + 
                      relation_score * 0.3 + 
                      fact_score * 0.3)
        
        return total_score
    
    def _evaluate_concept_accuracy(self, output_concepts: Set[str], 
                                 expected_concepts: Set[str]) -> float:
        """评估概念准确性"""
        if not expected_concepts:
            # 如果没有期望概念，基于输出概念的有效性评分
            return self._evaluate_concept_validity(output_concepts)
        
        # 计算概念匹配度
        intersection = output_concepts.intersection(expected_concepts)
        union = output_concepts.union(expected_concepts)
        
        if not union:
            return 1.0  # 都没有概念，认为完全匹配
        
        # Jaccard相似度
        jaccard_score = len(intersection) / len(union)
        
        # 考虑概念的重要性权重
        weighted_score = self._calculate_weighted_concept_score(
            output_concepts, expected_concepts, intersection
        )
        
        # 综合评分
        return (jaccard_score + weighted_score) / 2
    
    def _evaluate_concept_validity(self, concepts: Set[str]) -> float:
        """评估概念有效性"""
        if not concepts:
            return 0.5  # 没有概念，给中等分数
        
        valid_concepts = 0
        for concept in concepts:
            if concept in self.knowledge_matcher.concepts:
                valid_concepts += 1
        
        return valid_concepts / len(concepts)
    
    def _calculate_weighted_concept_score(self, output_concepts: Set[str], 
                                        expected_concepts: Set[str], 
                                        intersection: Set[str]) -> float:
        """计算加权概念分数"""
        # 简化实现：给核心概念更高权重
        core_concepts = {"机器学习", "深度学习", "自然语言处理", "金融风控"}
        
        weighted_intersection = 0
        weighted_expected = 0
        
        for concept in expected_concepts:
            weight = 2.0 if concept in core_concepts else 1.0
            weighted_expected += weight
            if concept in intersection:
                weighted_intersection += weight
        
        if weighted_expected == 0:
            return 1.0
        
        return weighted_intersection / weighted_expected
    
    def _generate_details(self, input_text: str, model_output: str, 
                         expected_output: str, context: Dict[str, Any], 
                         score: float) -> Dict[str, Any]:
        """生成详细评估信息"""
        output_concepts = self.knowledge_matcher.extract_concepts(model_output)
        expected_concepts = self.knowledge_matcher.extract_concepts(expected_output)
        relation_result = self.knowledge_matcher.verify_concept_relations(output_concepts, model_output)
        fact_result = self.knowledge_matcher.check_fact_consistency(model_output)
        
        return {
            "evaluator": self.name,
            "score": score,
            "extracted_concepts": list(output_concepts),
            "expected_concepts": list(expected_concepts),
            "concept_accuracy": self._evaluate_concept_accuracy(output_concepts, expected_concepts),
            "relation_verification": relation_result,
            "fact_consistency": fact_result,
            "missing_concepts": list(expected_concepts - output_concepts),
            "extra_concepts": list(output_concepts - expected_concepts)
        }
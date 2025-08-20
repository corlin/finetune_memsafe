"""
概念关系验证评估器
"""

import re
from typing import Dict, List, Any, Set, Tuple, Optional
from industry_evaluation.evaluators.base_evaluator import AbstractEvaluator
from industry_evaluation.models.data_models import Criterion


class ConceptRelationValidator:
    """概念关系验证器"""
    
    def __init__(self, relation_rules: Optional[Dict[str, Any]] = None):
        """
        初始化概念关系验证器
        
        Args:
            relation_rules: 关系验证规则
        """
        self.relation_rules = relation_rules or self._get_default_relation_rules()
        self.relation_patterns = self._compile_relation_patterns()
    
    def extract_relations(self, text: str, concepts: Set[str]) -> List[Dict[str, Any]]:
        """
        从文本中提取概念关系
        
        Args:
            text: 输入文本
            concepts: 已识别的概念集合
            
        Returns:
            List[Dict[str, Any]]: 提取的关系列表
        """
        extracted_relations = []
        
        # 使用预定义的关系模式提取关系
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    relation = self._parse_relation_match(match, relation_type, concepts)
                    if relation:
                        extracted_relations.append(relation)
        
        # 使用语义模式提取关系
        semantic_relations = self._extract_semantic_relations(text, concepts)
        extracted_relations.extend(semantic_relations)
        
        return self._deduplicate_relations(extracted_relations)
    
    def validate_relations(self, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证关系的正确性
        
        Args:
            relations: 关系列表
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_result = {
            "total_relations": len(relations),
            "valid_relations": [],
            "invalid_relations": [],
            "validation_details": {},
            "consistency_score": 0.0
        }
        
        for relation in relations:
            validation = self._validate_single_relation(relation)
            
            if validation["is_valid"]:
                validation_result["valid_relations"].append(relation)
            else:
                validation_result["invalid_relations"].append(relation)
            
            validation_result["validation_details"][self._relation_key(relation)] = validation
        
        # 计算一致性分数
        if validation_result["total_relations"] > 0:
            validation_result["consistency_score"] = (
                len(validation_result["valid_relations"]) / validation_result["total_relations"]
            )
        else:
            validation_result["consistency_score"] = 1.0
        
        return validation_result
    
    def check_relation_consistency(self, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        检查关系间的一致性
        
        Args:
            relations: 关系列表
            
        Returns:
            Dict[str, Any]: 一致性检查结果
        """
        consistency_result = {
            "contradictions": [],
            "redundancies": [],
            "missing_implications": [],
            "consistency_score": 1.0
        }
        
        # 检查矛盾关系
        contradictions = self._find_contradictions(relations)
        consistency_result["contradictions"] = contradictions
        
        # 检查冗余关系
        redundancies = self._find_redundancies(relations)
        consistency_result["redundancies"] = redundancies
        
        # 检查缺失的隐含关系
        missing_implications = self._find_missing_implications(relations)
        consistency_result["missing_implications"] = missing_implications
        
        # 计算一致性分数
        total_issues = len(contradictions) + len(redundancies)
        if len(relations) > 0:
            consistency_result["consistency_score"] = max(0.0, 1.0 - (total_issues / len(relations)))
        
        return consistency_result
    
    def _compile_relation_patterns(self) -> Dict[str, List[str]]:
        """编译关系匹配模式"""
        patterns = {
            "is_a": [
                r"(.+?)\s*是\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*属于\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*为\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*是一种\s*(.+?)(?:[的]|$|\s)"
            ],
            "has_property": [
                r"(.+?)\s*具有\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*拥有\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*包含\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*的\s*(.+?)(?:是|为|包括)"
            ],
            "causes": [
                r"(.+?)\s*导致\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*引起\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*造成\s*(.+?)(?:[的]|$|\s)",
                r"由于\s*(.+?)\s*[，,]\s*(.+?)"
            ],
            "part_of": [
                r"(.+?)\s*组成\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*构成\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*是\s*(.+?)\s*的\s*[组成]*部分",
                r"(.+?)\s*包括\s*(.+?)(?:[的]|$|\s)"
            ],
            "uses": [
                r"(.+?)\s*使用\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*采用\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*基于\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*依赖\s*(.+?)(?:[的]|$|\s)"
            ],
            "similar_to": [
                r"(.+?)\s*类似于\s*(.+?)(?:[的]|$|\s)",
                r"(.+?)\s*和\s*(.+?)\s*相似",
                r"(.+?)\s*与\s*(.+?)\s*类似",
                r"(.+?)\s*像\s*(.+?)(?:[的]|$|\s)"
            ]
        }
        
        return patterns
    
    def _parse_relation_match(self, match, relation_type: str, concepts: Set[str]) -> Optional[Dict[str, Any]]:
        """解析关系匹配结果"""
        if len(match.groups()) < 2:
            return None
        
        subject = match.group(1).strip()
        object_term = match.group(2).strip()
        
        # 清理提取的实体
        subject = self._clean_entity(subject)
        object_term = self._clean_entity(object_term)
        
        # 检查是否涉及已知概念
        subject_relevant = any(concept in subject for concept in concepts)
        object_relevant = any(concept in object_term for concept in concepts)
        
        if not (subject_relevant or object_relevant):
            return None
        
        return {
            "type": relation_type,
            "subject": subject,
            "object": object_term,
            "confidence": self._calculate_extraction_confidence(match, relation_type),
            "text_span": match.group(0),
            "start_pos": match.start(),
            "end_pos": match.end()
        }
    
    def _clean_entity(self, entity: str) -> str:
        """清理提取的实体"""
        # 移除常见的停用词和标点符号
        entity = re.sub(r'^(的|一个|一种|这个|那个|某个)\s*', '', entity)
        entity = re.sub(r'\s*(的|等|等等)$', '', entity)
        entity = re.sub(r'[，,。.！!？?；;：:]', '', entity)
        return entity.strip()
    
    def _calculate_extraction_confidence(self, match, relation_type: str) -> float:
        """计算提取置信度"""
        base_confidence = 0.8
        
        # 根据关系类型调整置信度
        type_confidence = {
            "is_a": 0.9,
            "has_property": 0.8,
            "causes": 0.7,
            "part_of": 0.8,
            "uses": 0.7,
            "similar_to": 0.6
        }
        
        confidence = type_confidence.get(relation_type, base_confidence)
        
        # 根据匹配文本长度调整
        text_length = len(match.group(0))
        if text_length < 10:
            confidence *= 0.9
        elif text_length > 50:
            confidence *= 0.8
        
        return min(1.0, confidence)
    
    def _extract_semantic_relations(self, text: str, concepts: Set[str]) -> List[Dict[str, Any]]:
        """使用语义模式提取关系"""
        semantic_relations = []
        
        # 共现关系：在同一句子中出现的概念可能有关系
        sentences = re.split(r'[。.！!？?]', text)
        
        for sentence in sentences:
            sentence_concepts = [c for c in concepts if c in sentence]
            
            if len(sentence_concepts) >= 2:
                # 为每对概念创建潜在关系
                for i, concept1 in enumerate(sentence_concepts):
                    for concept2 in sentence_concepts[i+1:]:
                        relation = {
                            "type": "co_occurrence",
                            "subject": concept1,
                            "object": concept2,
                            "confidence": 0.3,  # 较低的置信度
                            "text_span": sentence.strip(),
                            "start_pos": 0,
                            "end_pos": len(sentence)
                        }
                        semantic_relations.append(relation)
        
        return semantic_relations
    
    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去除重复关系"""
        seen_relations = set()
        unique_relations = []
        
        for relation in relations:
            relation_key = self._relation_key(relation)
            
            if relation_key not in seen_relations:
                seen_relations.add(relation_key)
                unique_relations.append(relation)
            else:
                # 如果有重复，保留置信度更高的
                for i, existing in enumerate(unique_relations):
                    if self._relation_key(existing) == relation_key:
                        if relation["confidence"] > existing["confidence"]:
                            unique_relations[i] = relation
                        break
        
        return unique_relations
    
    def _relation_key(self, relation: Dict[str, Any]) -> str:
        """生成关系的唯一键"""
        return f"{relation['subject']}_{relation['type']}_{relation['object']}"
    
    def _validate_single_relation(self, relation: Dict[str, Any]) -> Dict[str, Any]:
        """验证单个关系"""
        validation = {
            "is_valid": True,
            "confidence": relation.get("confidence", 0.5),
            "issues": [],
            "rule_matches": []
        }
        
        relation_type = relation["type"]
        subject = relation["subject"]
        object_term = relation["object"]
        
        # 检查关系类型规则
        if relation_type in self.relation_rules:
            type_rules = self.relation_rules[relation_type]
            
            # 检查主体约束
            if "subject_constraints" in type_rules:
                if not self._check_constraints(subject, type_rules["subject_constraints"]):
                    validation["is_valid"] = False
                    validation["issues"].append("主体不符合约束条件")
            
            # 检查客体约束
            if "object_constraints" in type_rules:
                if not self._check_constraints(object_term, type_rules["object_constraints"]):
                    validation["is_valid"] = False
                    validation["issues"].append("客体不符合约束条件")
            
            # 检查特定规则
            if "validation_rules" in type_rules:
                for rule in type_rules["validation_rules"]:
                    if not self._apply_validation_rule(relation, rule):
                        validation["is_valid"] = False
                        validation["issues"].append(f"违反规则: {rule.get('description', '未知规则')}")
                    else:
                        validation["rule_matches"].append(rule.get("description", "匹配规则"))
        
        # 调整置信度
        if not validation["is_valid"]:
            validation["confidence"] *= 0.3
        elif validation["rule_matches"]:
            validation["confidence"] = min(1.0, validation["confidence"] * 1.2)
        
        return validation
    
    def _check_constraints(self, entity: str, constraints: Dict[str, Any]) -> bool:
        """检查实体约束"""
        # 类型约束
        if "types" in constraints:
            entity_type = self._infer_entity_type(entity)
            if entity_type not in constraints["types"]:
                return False
        
        # 模式约束
        if "patterns" in constraints:
            for pattern in constraints["patterns"]:
                if not re.search(pattern, entity, re.IGNORECASE):
                    return False
        
        # 长度约束
        if "min_length" in constraints and len(entity) < constraints["min_length"]:
            return False
        
        if "max_length" in constraints and len(entity) > constraints["max_length"]:
            return False
        
        return True
    
    def _infer_entity_type(self, entity: str) -> str:
        """推断实体类型"""
        # 简单的实体类型推断
        if any(keyword in entity for keyword in ["算法", "方法", "技术", "模型"]):
            return "技术"
        elif any(keyword in entity for keyword in ["系统", "平台", "工具", "软件"]):
            return "系统"
        elif any(keyword in entity for keyword in ["数据", "信息", "知识"]):
            return "数据"
        elif any(keyword in entity for keyword in ["领域", "行业", "业务"]):
            return "领域"
        else:
            return "概念"
    
    def _apply_validation_rule(self, relation: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """应用验证规则"""
        rule_type = rule.get("type", "")
        
        if rule_type == "mutual_exclusion":
            # 互斥规则：某些关系不能同时存在
            excluded_relations = rule.get("excluded_relations", [])
            relation_key = self._relation_key(relation)
            return relation_key not in excluded_relations
        
        elif rule_type == "transitivity":
            # 传递性规则：如果A->B且B->C，则A->C
            # 这里简化处理，实际需要更复杂的逻辑
            return True
        
        elif rule_type == "symmetry":
            # 对称性规则：如果A->B，则B->A
            return True
        
        elif rule_type == "domain_specific":
            # 领域特定规则
            domain_rules = rule.get("rules", [])
            for domain_rule in domain_rules:
                if not self._check_domain_rule(relation, domain_rule):
                    return False
            return True
        
        return True
    
    def _check_domain_rule(self, relation: Dict[str, Any], domain_rule: Dict[str, Any]) -> bool:
        """检查领域特定规则"""
        # 简化的领域规则检查
        if "forbidden_combinations" in domain_rule:
            forbidden = domain_rule["forbidden_combinations"]
            relation_combo = (relation["subject"], relation["object"])
            if relation_combo in forbidden:
                return False
        
        return True
    
    def _find_contradictions(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """查找矛盾关系"""
        contradictions = []
        
        # 检查直接矛盾
        for i, rel1 in enumerate(relations):
            for rel2 in relations[i+1:]:
                if self._are_contradictory(rel1, rel2):
                    contradictions.append({
                        "relation1": rel1,
                        "relation2": rel2,
                        "type": "direct_contradiction"
                    })
        
        return contradictions
    
    def _are_contradictory(self, rel1: Dict[str, Any], rel2: Dict[str, Any]) -> bool:
        """判断两个关系是否矛盾"""
        # 简单的矛盾检测逻辑
        if (rel1["subject"] == rel2["subject"] and 
            rel1["object"] == rel2["object"]):
            
            # 检查矛盾的关系类型
            contradictory_pairs = [
                ("is_a", "not_is_a"),
                ("causes", "prevents"),
                ("similar_to", "different_from")
            ]
            
            rel1_type = rel1["type"]
            rel2_type = rel2["type"]
            
            for pair in contradictory_pairs:
                if (rel1_type, rel2_type) == pair or (rel2_type, rel1_type) == pair:
                    return True
        
        return False
    
    def _find_redundancies(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """查找冗余关系"""
        redundancies = []
        
        # 检查重复关系
        relation_counts = {}
        for relation in relations:
            key = self._relation_key(relation)
            if key in relation_counts:
                redundancies.append({
                    "original": relation_counts[key],
                    "duplicate": relation,
                    "type": "duplicate"
                })
            else:
                relation_counts[key] = relation
        
        return redundancies
    
    def _find_missing_implications(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """查找缺失的隐含关系"""
        missing_implications = []
        
        # 检查传递性隐含关系
        for rel1 in relations:
            for rel2 in relations:
                if (rel1["object"] == rel2["subject"] and 
                    rel1["type"] == "is_a" and rel2["type"] == "is_a"):
                    
                    # 应该存在 rel1.subject is_a rel2.object 的关系
                    implied_relation = {
                        "type": "is_a",
                        "subject": rel1["subject"],
                        "object": rel2["object"]
                    }
                    
                    # 检查这个隐含关系是否已存在
                    implied_key = self._relation_key(implied_relation)
                    existing_keys = [self._relation_key(r) for r in relations]
                    
                    if implied_key not in existing_keys:
                        missing_implications.append({
                            "implied_relation": implied_relation,
                            "source_relations": [rel1, rel2],
                            "type": "transitivity"
                        })
        
        return missing_implications
    
    def _get_default_relation_rules(self) -> Dict[str, Any]:
        """获取默认关系验证规则"""
        return {
            "is_a": {
                "description": "是一个/属于关系",
                "subject_constraints": {
                    "types": ["技术", "概念", "系统", "数据", "领域"],
                    "min_length": 2
                },
                "object_constraints": {
                    "types": ["技术", "概念", "系统", "数据", "领域"],
                    "min_length": 2
                },
                "validation_rules": [
                    {
                        "type": "domain_specific",
                        "description": "领域特定的is_a关系规则",
                        "rules": [
                            {
                                "forbidden_combinations": [
                                    ("机器学习", "深度学习")  # 应该是相反的
                                ]
                            }
                        ]
                    }
                ]
            },
            "has_property": {
                "description": "具有属性关系",
                "subject_constraints": {
                    "types": ["技术", "概念", "系统"],
                    "min_length": 2
                },
                "object_constraints": {
                    "types": ["概念", "数据"],
                    "min_length": 1
                }
            },
            "causes": {
                "description": "因果关系",
                "validation_rules": [
                    {
                        "type": "mutual_exclusion",
                        "description": "因果关系互斥规则",
                        "excluded_relations": []
                    }
                ]
            },
            "part_of": {
                "description": "部分-整体关系",
                "validation_rules": [
                    {
                        "type": "transitivity",
                        "description": "部分关系传递性"
                    }
                ]
            }
        }


class ConceptRelationEvaluator(AbstractEvaluator):
    """概念关系验证评估器"""
    
    def __init__(self, name: str = "concept_relation", weight: float = 1.0,
                 relation_rules: Optional[Dict[str, Any]] = None):
        """
        初始化概念关系评估器
        
        Args:
            name: 评估器名称
            weight: 评估器权重
            relation_rules: 关系验证规则
        """
        super().__init__(name, weight)
        self.relation_validator = ConceptRelationValidator(relation_rules)
    
    def _initialize_criteria(self) -> List[Criterion]:
        """初始化评估标准"""
        return [
            Criterion(
                name="relation_extraction_accuracy",
                description="关系提取准确性",
                weight=0.3,
                threshold=0.7,
                evaluation_method="relation_extraction"
            ),
            Criterion(
                name="relation_validity",
                description="关系有效性",
                weight=0.4,
                threshold=0.8,
                evaluation_method="relation_validation"
            ),
            Criterion(
                name="relation_consistency",
                description="关系一致性",
                weight=0.3,
                threshold=0.6,
                evaluation_method="consistency_check"
            )
        ]
    
    def _calculate_score(self, input_text: str, model_output: str, 
                        expected_output: str, context: Dict[str, Any]) -> float:
        """计算概念关系评估分数"""
        # 从上下文中获取概念，如果没有则使用简单提取
        concepts = context.get("concepts", set())
        if not concepts:
            concepts = self._extract_simple_concepts(model_output)
        
        # 提取关系
        extracted_relations = self.relation_validator.extract_relations(model_output, concepts)
        
        # 验证关系
        validation_result = self.relation_validator.validate_relations(extracted_relations)
        
        # 检查一致性
        consistency_result = self.relation_validator.check_relation_consistency(extracted_relations)
        
        # 计算各项分数
        extraction_score = self._calculate_extraction_score(extracted_relations, expected_output)
        validity_score = validation_result["consistency_score"]
        consistency_score = consistency_result["consistency_score"]
        
        # 加权计算总分
        total_score = (extraction_score * 0.3 + 
                      validity_score * 0.4 + 
                      consistency_score * 0.3)
        
        return total_score
    
    def _extract_simple_concepts(self, text: str) -> Set[str]:
        """简单的概念提取"""
        # 这里使用简单的关键词匹配
        concept_keywords = [
            "机器学习", "深度学习", "自然语言处理", "人工智能",
            "算法", "模型", "数据", "训练", "预测", "分类",
            "回归", "聚类", "神经网络", "特征", "标签"
        ]
        
        concepts = set()
        for keyword in concept_keywords:
            if keyword in text:
                concepts.add(keyword)
        
        return concepts
    
    def _calculate_extraction_score(self, extracted_relations: List[Dict[str, Any]], 
                                  expected_output: str) -> float:
        """计算关系提取分数"""
        if not extracted_relations:
            return 0.5  # 没有提取到关系，给中等分数
        
        # 简单评估：基于提取的关系数量和质量
        total_confidence = sum(rel.get("confidence", 0.5) for rel in extracted_relations)
        avg_confidence = total_confidence / len(extracted_relations)
        
        # 根据关系数量调整分数
        relation_count_score = min(1.0, len(extracted_relations) / 5.0)  # 假设5个关系为满分
        
        return (avg_confidence + relation_count_score) / 2
    
    def _generate_details(self, input_text: str, model_output: str, 
                         expected_output: str, context: Dict[str, Any], 
                         score: float) -> Dict[str, Any]:
        """生成详细评估信息"""
        concepts = context.get("concepts", self._extract_simple_concepts(model_output))
        extracted_relations = self.relation_validator.extract_relations(model_output, concepts)
        validation_result = self.relation_validator.validate_relations(extracted_relations)
        consistency_result = self.relation_validator.check_relation_consistency(extracted_relations)
        
        return {
            "evaluator": self.name,
            "score": score,
            "extracted_relations": extracted_relations,
            "validation_result": validation_result,
            "consistency_result": consistency_result,
            "relation_count": len(extracted_relations),
            "valid_relation_count": len(validation_result["valid_relations"]),
            "invalid_relation_count": len(validation_result["invalid_relations"]),
            "contradiction_count": len(consistency_result["contradictions"]),
            "concepts_used": list(concepts)
        }
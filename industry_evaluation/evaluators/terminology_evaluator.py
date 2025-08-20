"""
术语准确性评估器
"""

import re
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from industry_evaluation.evaluators.base_evaluator import AbstractEvaluator
from industry_evaluation.models.data_models import Criterion


class TerminologyDictionary:
    """术语词典管理器"""
    
    def __init__(self, dictionary_data: Optional[Dict[str, Any]] = None):
        """
        初始化术语词典
        
        Args:
            dictionary_data: 词典数据
        """
        self.dictionary_data = dictionary_data or self._get_default_dictionary()
        self.terms = self.dictionary_data.get("terms", {})
        self.synonyms = self.dictionary_data.get("synonyms", {})
        self.categories = self.dictionary_data.get("categories", {})
        self.contexts = self.dictionary_data.get("contexts", {})
        
        # 构建反向索引
        self._build_reverse_indices()
    
    def _build_reverse_indices(self):
        """构建反向索引以提高查询效率"""
        self.synonym_to_term = {}
        self.term_to_category = {}
        
        # 构建同义词到标准术语的映射
        for term, synonyms in self.synonyms.items():
            for synonym in synonyms:
                self.synonym_to_term[synonym.lower()] = term
        
        # 构建术语到类别的映射
        for category, terms in self.categories.items():
            for term in terms:
                self.term_to_category[term] = category
    
    def get_standard_term(self, term: str) -> str:
        """
        获取术语的标准形式
        
        Args:
            term: 输入术语
            
        Returns:
            str: 标准术语
        """
        term_lower = term.lower()
        
        # 检查是否是同义词
        if term_lower in self.synonym_to_term:
            return self.synonym_to_term[term_lower]
        
        # 检查是否是标准术语
        if term in self.terms:
            return term
        
        # 模糊匹配
        for standard_term in self.terms:
            if self._fuzzy_match(term, standard_term):
                return standard_term
        
        return term  # 如果找不到，返回原术语
    
    def is_valid_term(self, term: str) -> bool:
        """
        检查术语是否有效
        
        Args:
            term: 术语
            
        Returns:
            bool: 是否有效
        """
        standard_term = self.get_standard_term(term)
        return standard_term in self.terms
    
    def get_term_info(self, term: str) -> Dict[str, Any]:
        """
        获取术语详细信息
        
        Args:
            term: 术语
            
        Returns:
            Dict[str, Any]: 术语信息
        """
        standard_term = self.get_standard_term(term)
        
        if standard_term not in self.terms:
            return {"exists": False}
        
        term_info = self.terms[standard_term].copy()
        term_info.update({
            "exists": True,
            "standard_form": standard_term,
            "category": self.term_to_category.get(standard_term, "未分类"),
            "synonyms": self.synonyms.get(standard_term, []),
            "contexts": self.contexts.get(standard_term, [])
        })
        
        return term_info
    
    def get_terms_by_category(self, category: str) -> List[str]:
        """
        根据类别获取术语列表
        
        Args:
            category: 类别名称
            
        Returns:
            List[str]: 术语列表
        """
        return self.categories.get(category, [])
    
    def add_term(self, term: str, definition: str, category: str = "自定义",
                 synonyms: Optional[List[str]] = None, 
                 contexts: Optional[List[str]] = None):
        """
        添加新术语
        
        Args:
            term: 术语
            definition: 定义
            category: 类别
            synonyms: 同义词列表
            contexts: 使用上下文列表
        """
        self.terms[term] = {
            "definition": definition,
            "category": category
        }
        
        if synonyms:
            self.synonyms[term] = synonyms
            for synonym in synonyms:
                self.synonym_to_term[synonym.lower()] = term
        
        if contexts:
            self.contexts[term] = contexts
        
        # 更新类别映射
        if category not in self.categories:
            self.categories[category] = []
        if term not in self.categories[category]:
            self.categories[category].append(term)
        
        self.term_to_category[term] = category
    
    def _fuzzy_match(self, term1: str, term2: str, threshold: float = 0.8) -> bool:
        """
        模糊匹配两个术语
        
        Args:
            term1: 术语1
            term2: 术语2
            threshold: 相似度阈值
            
        Returns:
            bool: 是否匹配
        """
        # 简单的编辑距离相似度
        def levenshtein_similarity(s1: str, s2: str) -> float:
            if len(s1) == 0:
                return 0.0 if len(s2) > 0 else 1.0
            if len(s2) == 0:
                return 0.0
            
            # 计算编辑距离
            matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
            
            for i in range(len(s1) + 1):
                matrix[i][0] = i
            for j in range(len(s2) + 1):
                matrix[0][j] = j
            
            for i in range(1, len(s1) + 1):
                for j in range(1, len(s2) + 1):
                    if s1[i-1] == s2[j-1]:
                        matrix[i][j] = matrix[i-1][j-1]
                    else:
                        matrix[i][j] = min(
                            matrix[i-1][j] + 1,      # 删除
                            matrix[i][j-1] + 1,      # 插入
                            matrix[i-1][j-1] + 1     # 替换
                        )
            
            distance = matrix[len(s1)][len(s2)]
            max_len = max(len(s1), len(s2))
            return 1.0 - (distance / max_len)
        
        similarity = levenshtein_similarity(term1.lower(), term2.lower())
        return similarity >= threshold
    
    def _get_default_dictionary(self) -> Dict[str, Any]:
        """获取默认术语词典"""
        return {
            "terms": {
                "机器学习": {
                    "definition": "一种人工智能技术，使计算机能够从数据中学习",
                    "category": "AI技术"
                },
                "深度学习": {
                    "definition": "基于人工神经网络的机器学习方法",
                    "category": "AI技术"
                },
                "自然语言处理": {
                    "definition": "计算机处理和理解人类语言的技术",
                    "category": "AI技术"
                },
                "神经网络": {
                    "definition": "模拟生物神经网络的计算模型",
                    "category": "AI技术"
                },
                "监督学习": {
                    "definition": "使用标注数据训练模型的机器学习方法",
                    "category": "学习方法"
                },
                "无监督学习": {
                    "definition": "不使用标注数据的机器学习方法",
                    "category": "学习方法"
                },
                "强化学习": {
                    "definition": "通过与环境交互学习最优策略的方法",
                    "category": "学习方法"
                },
                "特征工程": {
                    "definition": "选择和构造用于机器学习的特征的过程",
                    "category": "数据处理"
                },
                "过拟合": {
                    "definition": "模型在训练数据上表现好但在新数据上表现差的现象",
                    "category": "模型问题"
                },
                "欠拟合": {
                    "definition": "模型过于简单，无法捕捉数据中的模式",
                    "category": "模型问题"
                }
            },
            "synonyms": {
                "机器学习": ["ML", "机器学习算法", "机学"],
                "深度学习": ["DL", "深度神经网络", "深学"],
                "自然语言处理": ["NLP", "文本处理", "语言处理"],
                "神经网络": ["NN", "人工神经网络", "神经元网络"],
                "监督学习": ["有监督学习", "监督式学习"],
                "无监督学习": ["无监督式学习", "非监督学习"],
                "特征工程": ["特征选择", "特征构造"],
                "过拟合": ["过度拟合", "过学习"],
                "欠拟合": ["欠学习", "拟合不足"]
            },
            "categories": {
                "AI技术": ["机器学习", "深度学习", "自然语言处理", "神经网络"],
                "学习方法": ["监督学习", "无监督学习", "强化学习"],
                "数据处理": ["特征工程"],
                "模型问题": ["过拟合", "欠拟合"]
            },
            "contexts": {
                "机器学习": [
                    "机器学习算法可以从数据中学习模式",
                    "机器学习在图像识别中应用广泛",
                    "机器学习需要大量的训练数据"
                ],
                "深度学习": [
                    "深度学习使用多层神经网络",
                    "深度学习在计算机视觉领域表现出色",
                    "深度学习需要大量的计算资源"
                ],
                "过拟合": [
                    "过拟合是机器学习中常见的问题",
                    "正则化可以帮助防止过拟合",
                    "交叉验证可以检测过拟合"
                ]
            }
        }


class TerminologyRecognizer:
    """术语识别器"""
    
    def __init__(self, dictionary: TerminologyDictionary):
        """
        初始化术语识别器
        
        Args:
            dictionary: 术语词典
        """
        self.dictionary = dictionary
        self.recognition_patterns = self._build_recognition_patterns()
    
    def recognize_terms(self, text: str) -> List[Dict[str, Any]]:
        """
        识别文本中的术语
        
        Args:
            text: 输入文本
            
        Returns:
            List[Dict[str, Any]]: 识别的术语列表
        """
        recognized_terms = []
        
        # 精确匹配
        exact_matches = self._exact_match(text)
        recognized_terms.extend(exact_matches)
        
        # 模糊匹配
        fuzzy_matches = self._fuzzy_match(text)
        recognized_terms.extend(fuzzy_matches)
        
        # 去重和排序
        recognized_terms = self._deduplicate_and_sort(recognized_terms)
        
        return recognized_terms
    
    def _exact_match(self, text: str) -> List[Dict[str, Any]]:
        """精确匹配术语"""
        matches = []
        
        # 匹配标准术语
        for term in self.dictionary.terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append({
                    "term": term,
                    "matched_text": match.group(),
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "match_type": "exact",
                    "confidence": 1.0,
                    "standard_form": term
                })
        
        # 匹配同义词
        for term, synonyms in self.dictionary.synonyms.items():
            for synonym in synonyms:
                pattern = r'\b' + re.escape(synonym) + r'\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    matches.append({
                        "term": term,
                        "matched_text": match.group(),
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "match_type": "synonym",
                        "confidence": 0.9,
                        "standard_form": term
                    })
        
        return matches
    
    def _fuzzy_match(self, text: str) -> List[Dict[str, Any]]:
        """模糊匹配术语"""
        matches = []
        
        # 提取候选词汇
        candidates = self._extract_candidates(text)
        
        for candidate in candidates:
            for term in self.dictionary.terms:
                if self.dictionary._fuzzy_match(candidate["text"], term, threshold=0.7):
                    matches.append({
                        "term": term,
                        "matched_text": candidate["text"],
                        "start_pos": candidate["start_pos"],
                        "end_pos": candidate["end_pos"],
                        "match_type": "fuzzy",
                        "confidence": 0.6,
                        "standard_form": term
                    })
        
        return matches
    
    def _extract_candidates(self, text: str) -> List[Dict[str, Any]]:
        """提取候选术语"""
        candidates = []
        
        # 使用正则表达式提取可能的术语
        # 匹配中英文混合的技术术语
        patterns = [
            r'[A-Za-z]+[学习|网络|算法|模型|系统|技术|方法|处理]',  # 英文+中文后缀
            r'[深度|机器|人工|自然|监督|无监督|强化]+[A-Za-z]+',      # 中文前缀+英文
            r'[一-龟]{2,6}[学习|网络|算法|模型|系统|技术|方法|处理]',  # 纯中文术语
            r'[A-Z]{2,5}(?![a-z])',                                # 英文缩写
            r'[A-Za-z]{3,}(?=\s|$|[，。！？])'                      # 英文单词
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                candidates.append({
                    "text": match.group(),
                    "start_pos": match.start(),
                    "end_pos": match.end()
                })
        
        return candidates
    
    def _deduplicate_and_sort(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重和排序"""
        # 按位置去重，保留置信度最高的匹配
        position_matches = {}
        
        for match in matches:
            key = (match["start_pos"], match["end_pos"])
            if key not in position_matches or match["confidence"] > position_matches[key]["confidence"]:
                position_matches[key] = match
        
        # 按位置排序
        sorted_matches = sorted(position_matches.values(), key=lambda x: x["start_pos"])
        
        return sorted_matches
    
    def _build_recognition_patterns(self) -> Dict[str, Any]:
        """构建识别模式"""
        return {
            "exact_patterns": [
                r'\b{term}\b'
            ],
            "context_patterns": [
                r'{term}[是为]',
                r'使用{term}',
                r'{term}技术',
                r'{term}方法'
            ]
        }


class TerminologyEvaluator(AbstractEvaluator):
    """术语准确性评估器"""
    
    def __init__(self, name: str = "terminology", weight: float = 1.0,
                 dictionary_data: Optional[Dict[str, Any]] = None):
        """
        初始化术语评估器
        
        Args:
            name: 评估器名称
            weight: 评估器权重
            dictionary_data: 术语词典数据
        """
        super().__init__(name, weight)
        self.dictionary = TerminologyDictionary(dictionary_data)
        self.recognizer = TerminologyRecognizer(self.dictionary)
    
    def _initialize_criteria(self) -> List[Criterion]:
        """初始化评估标准"""
        return [
            Criterion(
                name="term_recognition_accuracy",
                description="术语识别准确性",
                weight=0.3,
                threshold=0.7,
                evaluation_method="term_recognition"
            ),
            Criterion(
                name="term_usage_correctness",
                description="术语使用正确性",
                weight=0.4,
                threshold=0.8,
                evaluation_method="usage_validation"
            ),
            Criterion(
                name="term_consistency",
                description="术语使用一致性",
                weight=0.3,
                threshold=0.6,
                evaluation_method="consistency_check"
            )
        ]
    
    def _calculate_score(self, input_text: str, model_output: str, 
                        expected_output: str, context: Dict[str, Any]) -> float:
        """计算术语准确性评估分数"""
        # 识别术语
        output_terms = self.recognizer.recognize_terms(model_output)
        expected_terms = self.recognizer.recognize_terms(expected_output)
        
        # 计算识别准确性分数
        recognition_score = self._calculate_recognition_score(output_terms, expected_terms)
        
        # 计算使用正确性分数
        usage_score = self._calculate_usage_score(output_terms, model_output)
        
        # 计算一致性分数
        consistency_score = self._calculate_consistency_score(output_terms, model_output)
        
        # 加权计算总分
        total_score = (recognition_score * 0.3 + 
                      usage_score * 0.4 + 
                      consistency_score * 0.3)
        
        return total_score
    
    def _calculate_recognition_score(self, output_terms: List[Dict[str, Any]], 
                                   expected_terms: List[Dict[str, Any]]) -> float:
        """计算术语识别分数"""
        if not expected_terms:
            # 如果没有期望术语，基于识别术语的有效性评分
            return self._calculate_validity_score(output_terms)
        
        # 提取术语集合
        output_term_set = {term["standard_form"] for term in output_terms}
        expected_term_set = {term["standard_form"] for term in expected_terms}
        
        if not expected_term_set:
            return 1.0 if not output_term_set else 0.5
        
        # 计算精确率、召回率和F1分数
        intersection = output_term_set.intersection(expected_term_set)
        
        precision = len(intersection) / len(output_term_set) if output_term_set else 0.0
        recall = len(intersection) / len(expected_term_set) if expected_term_set else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
    def _calculate_validity_score(self, terms: List[Dict[str, Any]]) -> float:
        """计算术语有效性分数"""
        if not terms:
            return 0.5  # 没有术语，给中等分数
        
        valid_terms = sum(1 for term in terms if self.dictionary.is_valid_term(term["term"]))
        return valid_terms / len(terms)
    
    def _calculate_usage_score(self, terms: List[Dict[str, Any]], text: str) -> float:
        """计算术语使用正确性分数"""
        if not terms:
            return 1.0  # 没有术语使用，认为正确
        
        usage_scores = []
        
        for term_match in terms:
            term = term_match["standard_form"]
            context_text = self._extract_context(text, term_match["start_pos"], term_match["end_pos"])
            
            usage_score = self._evaluate_term_usage(term, context_text)
            usage_scores.append(usage_score)
        
        return sum(usage_scores) / len(usage_scores)
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int, 
                        window_size: int = 50) -> str:
        """提取术语的上下文"""
        context_start = max(0, start_pos - window_size)
        context_end = min(len(text), end_pos + window_size)
        return text[context_start:context_end]
    
    def _evaluate_term_usage(self, term: str, context: str) -> float:
        """评估术语在特定上下文中的使用正确性"""
        term_info = self.dictionary.get_term_info(term)
        
        if not term_info.get("exists", False):
            return 0.5  # 未知术语，给中等分数
        
        # 检查上下文是否符合术语的典型使用场景
        expected_contexts = term_info.get("contexts", [])
        
        if not expected_contexts:
            return 0.8  # 没有上下文信息，给较高分数
        
        # 计算上下文相似度
        context_scores = []
        for expected_context in expected_contexts:
            similarity = self._calculate_context_similarity(context, expected_context)
            context_scores.append(similarity)
        
        # 返回最高的相似度分数
        return max(context_scores) if context_scores else 0.5
    
    def _calculate_context_similarity(self, context1: str, context2: str) -> float:
        """计算上下文相似度"""
        # 简单的词汇重叠相似度
        words1 = set(re.findall(r'\w+', context1.lower()))
        words2 = set(re.findall(r'\w+', context2.lower()))
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_consistency_score(self, terms: List[Dict[str, Any]], text: str) -> float:
        """计算术语使用一致性分数"""
        if len(terms) <= 1:
            return 1.0  # 术语太少，认为一致
        
        # 检查同一术语的不同表述是否一致
        term_variations = defaultdict(list)
        
        for term_match in terms:
            standard_form = term_match["standard_form"]
            matched_text = term_match["matched_text"]
            term_variations[standard_form].append(matched_text)
        
        consistency_scores = []
        
        for standard_form, variations in term_variations.items():
            if len(variations) > 1:
                # 检查变体的一致性
                unique_variations = set(variations)
                consistency_score = 1.0 / len(unique_variations)  # 变体越少，一致性越高
                consistency_scores.append(consistency_score)
            else:
                consistency_scores.append(1.0)  # 只有一种表述，完全一致
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0
    
    def _generate_details(self, input_text: str, model_output: str, 
                         expected_output: str, context: Dict[str, Any], 
                         score: float) -> Dict[str, Any]:
        """生成详细评估信息"""
        output_terms = self.recognizer.recognize_terms(model_output)
        expected_terms = self.recognizer.recognize_terms(expected_output)
        
        # 分析术语使用情况
        term_analysis = self._analyze_term_usage(output_terms, model_output)
        
        # 计算各项分数
        recognition_score = self._calculate_recognition_score(output_terms, expected_terms)
        usage_score = self._calculate_usage_score(output_terms, model_output)
        consistency_score = self._calculate_consistency_score(output_terms, model_output)
        
        return {
            "evaluator": self.name,
            "score": score,
            "recognized_terms": output_terms,
            "expected_terms": expected_terms,
            "recognition_score": recognition_score,
            "usage_score": usage_score,
            "consistency_score": consistency_score,
            "term_analysis": term_analysis,
            "term_count": len(output_terms),
            "valid_term_count": sum(1 for t in output_terms if self.dictionary.is_valid_term(t["term"])),
            "invalid_terms": [t for t in output_terms if not self.dictionary.is_valid_term(t["term"])]
        }
    
    def _analyze_term_usage(self, terms: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """分析术语使用情况"""
        analysis = {
            "term_frequency": Counter(),
            "term_categories": defaultdict(list),
            "usage_patterns": [],
            "potential_issues": []
        }
        
        for term_match in terms:
            term = term_match["standard_form"]
            analysis["term_frequency"][term] += 1
            
            # 分类统计
            term_info = self.dictionary.get_term_info(term)
            category = term_info.get("category", "未分类")
            analysis["term_categories"][category].append(term)
            
            # 检查潜在问题
            if term_match["confidence"] < 0.7:
                analysis["potential_issues"].append(f"术语 '{term}' 识别置信度较低")
            
            if term_match["match_type"] == "fuzzy":
                analysis["potential_issues"].append(f"术语 '{term}' 可能存在拼写问题")
        
        # 检查术语重复使用
        for term, count in analysis["term_frequency"].items():
            if count > 3:
                analysis["potential_issues"].append(f"术语 '{term}' 使用过于频繁 ({count}次)")
        
        return analysis
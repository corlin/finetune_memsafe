"""
术语使用评估功能
"""

import re
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from industry_evaluation.evaluators.base_evaluator import AbstractEvaluator
from industry_evaluation.evaluators.terminology_evaluator import TerminologyDictionary, TerminologyRecognizer
from industry_evaluation.models.data_models import Criterion


class ContextAnalyzer:
    """上下文分析器"""
    
    def __init__(self):
        """初始化上下文分析器"""
        self.context_patterns = self._build_context_patterns()
        self.semantic_indicators = self._build_semantic_indicators()
    
    def analyze_term_context(self, term: str, text: str, position: Tuple[int, int]) -> Dict[str, Any]:
        """
        分析术语的上下文
        
        Args:
            term: 术语
            text: 完整文本
            position: 术语在文本中的位置 (start, end)
            
        Returns:
            Dict[str, Any]: 上下文分析结果
        """
        start_pos, end_pos = position
        
        # 提取不同范围的上下文
        local_context = self._extract_local_context(text, start_pos, end_pos, window=30)
        sentence_context = self._extract_sentence_context(text, start_pos, end_pos)
        paragraph_context = self._extract_paragraph_context(text, start_pos, end_pos)
        
        # 分析语法角色
        grammatical_role = self._analyze_grammatical_role(term, local_context, start_pos - max(0, start_pos - 30))
        
        # 分析语义关系
        semantic_relations = self._analyze_semantic_relations(term, sentence_context)
        
        # 分析修饰词
        modifiers = self._extract_modifiers(term, local_context, start_pos - max(0, start_pos - 30))
        
        # 分析共现术语
        co_occurring_terms = self._find_co_occurring_terms(sentence_context, term)
        
        return {
            "local_context": local_context,
            "sentence_context": sentence_context,
            "paragraph_context": paragraph_context,
            "grammatical_role": grammatical_role,
            "semantic_relations": semantic_relations,
            "modifiers": modifiers,
            "co_occurring_terms": co_occurring_terms,
            "context_type": self._classify_context_type(sentence_context)
        }
    
    def _extract_local_context(self, text: str, start_pos: int, end_pos: int, window: int = 30) -> str:
        """提取局部上下文"""
        context_start = max(0, start_pos - window)
        context_end = min(len(text), end_pos + window)
        return text[context_start:context_end]
    
    def _extract_sentence_context(self, text: str, start_pos: int, end_pos: int) -> str:
        """提取句子上下文"""
        # 找到句子边界
        sentence_start = start_pos
        sentence_end = end_pos
        
        # 向前找句子开始
        for i in range(start_pos - 1, -1, -1):
            if text[i] in '。！？.!?':
                sentence_start = i + 1
                break
            elif i == 0:
                sentence_start = 0
                break
        
        # 向后找句子结束
        for i in range(end_pos, len(text)):
            if text[i] in '。！？.!?':
                sentence_end = i + 1
                break
            elif i == len(text) - 1:
                sentence_end = len(text)
                break
        
        return text[sentence_start:sentence_end].strip()
    
    def _extract_paragraph_context(self, text: str, start_pos: int, end_pos: int) -> str:
        """提取段落上下文"""
        # 找到段落边界
        paragraph_start = start_pos
        paragraph_end = end_pos
        
        # 向前找段落开始
        for i in range(start_pos - 1, -1, -1):
            if text[i] == '\n' and (i == 0 or text[i-1] == '\n'):
                paragraph_start = i + 1
                break
            elif i == 0:
                paragraph_start = 0
                break
        
        # 向后找段落结束
        for i in range(end_pos, len(text) - 1):
            if text[i] == '\n' and text[i+1] == '\n':
                paragraph_end = i
                break
            elif i == len(text) - 2:
                paragraph_end = len(text)
                break
        
        return text[paragraph_start:paragraph_end].strip()
    
    def _analyze_grammatical_role(self, term: str, context: str, term_pos: int) -> str:
        """分析术语的语法角色"""
        # 简化的语法角色分析
        before_term = context[:term_pos].strip()
        after_term = context[term_pos + len(term):].strip()
        
        # 主语模式
        if re.search(r'(^|[。！？.!?])\s*$', before_term) and re.search(r'^\s*(是|为|能够|可以|将)', after_term):
            return "subject"
        
        # 宾语模式
        if re.search(r'(使用|采用|应用|基于|通过)\s*$', before_term):
            return "object"
        
        # 定语模式
        if re.search(r'^\s*(的|算法|技术|方法|系统)', after_term):
            return "modifier"
        
        # 谓语模式
        if re.search(r'(是|为)\s*$', before_term) and not re.search(r'^\s*(的|算法|技术)', after_term):
            return "predicate"
        
        return "unknown"
    
    def _analyze_semantic_relations(self, term: str, sentence: str) -> List[Dict[str, str]]:
        """分析语义关系"""
        relations = []
        
        # 定义关系模式
        relation_patterns = {
            "definition": [r'{term}\s*是\s*(.+?)(?:[，。]|$)', r'{term}\s*指\s*(.+?)(?:[，。]|$)'],
            "application": [r'{term}\s*用于\s*(.+?)(?:[，。]|$)', r'{term}\s*应用于\s*(.+?)(?:[，。]|$)'],
            "characteristic": [r'{term}\s*具有\s*(.+?)(?:[，。]|$)', r'{term}\s*的特点是\s*(.+?)(?:[，。]|$)'],
            "comparison": [r'{term}\s*与\s*(.+?)\s*相比', r'{term}\s*不同于\s*(.+?)(?:[，。]|$)'],
            "causation": [r'{term}\s*导致\s*(.+?)(?:[，。]|$)', r'由于\s*{term}\s*[，,]\s*(.+?)(?:[，。]|$)']
        }
        
        for relation_type, patterns in relation_patterns.items():
            for pattern in patterns:
                regex_pattern = pattern.replace('{term}', re.escape(term))
                matches = re.finditer(regex_pattern, sentence, re.IGNORECASE)
                
                for match in matches:
                    relations.append({
                        "type": relation_type,
                        "target": match.group(1).strip(),
                        "pattern": pattern
                    })
        
        return relations
    
    def _extract_modifiers(self, term: str, context: str, term_pos: int) -> Dict[str, List[str]]:
        """提取修饰词"""
        modifiers = {
            "adjectives": [],
            "adverbs": [],
            "quantifiers": []
        }
        
        before_term = context[:term_pos]
        after_term = context[term_pos + len(term):]
        
        # 形容词修饰
        adj_patterns = [
            r'(先进的|复杂的|简单的|有效的|强大的|智能的|自动的|传统的)\s*$',
            r'(新的|旧的|现代的|经典的|流行的|常用的|主要的|重要的)\s*$'
        ]
        
        for pattern in adj_patterns:
            matches = re.findall(pattern, before_term)
            modifiers["adjectives"].extend(matches)
        
        # 副词修饰
        adv_patterns = [
            r'(广泛|深入|有效|快速|准确|精确|自动|智能)\s*$'
        ]
        
        for pattern in adv_patterns:
            matches = re.findall(pattern, before_term)
            modifiers["adverbs"].extend(matches)
        
        # 量词修饰
        quant_patterns = [
            r'(多种|各种|某种|一种|几种|许多|大量|少量)\s*$'
        ]
        
        for pattern in quant_patterns:
            matches = re.findall(pattern, before_term)
            modifiers["quantifiers"].extend(matches)
        
        return modifiers
    
    def _find_co_occurring_terms(self, sentence: str, current_term: str) -> List[str]:
        """查找共现术语"""
        # 简化的术语识别
        technical_terms = [
            "机器学习", "深度学习", "人工智能", "神经网络", "算法", "模型",
            "数据", "训练", "预测", "分类", "回归", "聚类", "特征", "标签"
        ]
        
        co_occurring = []
        for term in technical_terms:
            if term != current_term and term in sentence:
                co_occurring.append(term)
        
        return co_occurring
    
    def _classify_context_type(self, sentence: str) -> str:
        """分类上下文类型"""
        # 定义上下文类型模式
        context_patterns = {
            "definition": [r'是\s*一种', r'指的是', r'定义为', r'被称为'],
            "explanation": [r'也就是说', r'换句话说', r'具体来说', r'例如'],
            "comparison": [r'与.*相比', r'不同于', r'类似于', r'相对于'],
            "application": [r'用于', r'应用于', r'可以.*用来', r'主要用在'],
            "evaluation": [r'表现.*好', r'效果.*佳', r'性能.*优', r'准确率'],
            "process": [r'首先', r'然后', r'接下来', r'最后', r'步骤']
        }
        
        for context_type, patterns in context_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sentence):
                    return context_type
        
        return "general"
    
    def _build_context_patterns(self) -> Dict[str, List[str]]:
        """构建上下文模式"""
        return {
            "technical_description": [
                r'{term}是一种.*技术',
                r'{term}算法.*实现',
                r'基于{term}的.*方法'
            ],
            "application_scenario": [
                r'{term}在.*中应用',
                r'使用{term}来.*',
                r'{term}可以用于.*'
            ],
            "performance_evaluation": [
                r'{term}的性能.*',
                r'{term}表现.*',
                r'{term}的准确率.*'
            ]
        }
    
    def _build_semantic_indicators(self) -> Dict[str, List[str]]:
        """构建语义指示词"""
        return {
            "positive": ["优秀", "先进", "有效", "准确", "快速", "智能"],
            "negative": ["落后", "低效", "不准确", "缓慢", "复杂"],
            "neutral": ["一般", "普通", "常见", "标准", "基本"],
            "technical": ["算法", "模型", "系统", "框架", "架构", "技术"],
            "quantitative": ["提高", "降低", "增加", "减少", "优化", "改进"]
        }


class UsagePatternAnalyzer:
    """使用模式分析器"""
    
    def __init__(self):
        """初始化使用模式分析器"""
        self.usage_patterns = self._build_usage_patterns()
    
    def analyze_usage_patterns(self, terms: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """
        分析术语使用模式
        
        Args:
            terms: 识别的术语列表
            text: 完整文本
            
        Returns:
            Dict[str, Any]: 使用模式分析结果
        """
        analysis = {
            "frequency_patterns": self._analyze_frequency_patterns(terms),
            "position_patterns": self._analyze_position_patterns(terms, text),
            "co_occurrence_patterns": self._analyze_co_occurrence_patterns(terms, text),
            "context_patterns": self._analyze_context_patterns(terms, text),
            "consistency_patterns": self._analyze_consistency_patterns(terms),
            "evolution_patterns": self._analyze_evolution_patterns(terms, text)
        }
        
        return analysis
    
    def _analyze_frequency_patterns(self, terms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析频率模式"""
        term_counts = Counter(term["standard_form"] for term in terms)
        
        return {
            "term_frequency": dict(term_counts),
            "most_frequent": term_counts.most_common(5),
            "frequency_distribution": self._calculate_frequency_distribution(term_counts),
            "repetition_rate": self._calculate_repetition_rate(term_counts)
        }
    
    def _analyze_position_patterns(self, terms: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """分析位置模式"""
        text_length = len(text)
        positions = []
        
        for term in terms:
            relative_pos = term["start_pos"] / text_length if text_length > 0 else 0
            positions.append({
                "term": term["standard_form"],
                "absolute_pos": term["start_pos"],
                "relative_pos": relative_pos,
                "section": self._classify_text_section(relative_pos)
            })
        
        return {
            "position_distribution": self._calculate_position_distribution(positions),
            "section_usage": self._calculate_section_usage(positions),
            "clustering": self._analyze_position_clustering(positions)
        }
    
    def _analyze_co_occurrence_patterns(self, terms: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """分析共现模式"""
        sentences = re.split(r'[。！？.!?]', text)
        co_occurrences = defaultdict(lambda: defaultdict(int))
        
        for sentence in sentences:
            sentence_terms = []
            for term in terms:
                if term["matched_text"] in sentence:
                    sentence_terms.append(term["standard_form"])
            
            # 计算共现
            for i, term1 in enumerate(sentence_terms):
                for term2 in sentence_terms[i+1:]:
                    co_occurrences[term1][term2] += 1
                    co_occurrences[term2][term1] += 1
        
        return {
            "co_occurrence_matrix": dict(co_occurrences),
            "strong_associations": self._find_strong_associations(co_occurrences),
            "co_occurrence_strength": self._calculate_co_occurrence_strength(co_occurrences)
        }
    
    def _analyze_context_patterns(self, terms: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """分析上下文模式"""
        context_analyzer = ContextAnalyzer()
        context_types = defaultdict(int)
        grammatical_roles = defaultdict(int)
        
        for term in terms:
            position = (term["start_pos"], term["end_pos"])
            context_analysis = context_analyzer.analyze_term_context(
                term["standard_form"], text, position
            )
            
            context_types[context_analysis["context_type"]] += 1
            grammatical_roles[context_analysis["grammatical_role"]] += 1
        
        return {
            "context_type_distribution": dict(context_types),
            "grammatical_role_distribution": dict(grammatical_roles),
            "dominant_patterns": self._identify_dominant_patterns(context_types, grammatical_roles)
        }
    
    def _analyze_consistency_patterns(self, terms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析一致性模式"""
        term_variations = defaultdict(set)
        
        for term in terms:
            standard_form = term["standard_form"]
            matched_text = term["matched_text"]
            term_variations[standard_form].add(matched_text)
        
        consistency_scores = {}
        for standard_form, variations in term_variations.items():
            # 一致性分数：变体越少，一致性越高
            consistency_scores[standard_form] = 1.0 / len(variations)
        
        return {
            "term_variations": {k: list(v) for k, v in term_variations.items()},
            "consistency_scores": consistency_scores,
            "overall_consistency": sum(consistency_scores.values()) / len(consistency_scores) if consistency_scores else 1.0,
            "inconsistent_terms": [term for term, score in consistency_scores.items() if score < 0.8]
        }
    
    def _analyze_evolution_patterns(self, terms: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """分析演化模式（术语在文本中的使用变化）"""
        # 将文本分成几个部分来分析术语使用的变化
        text_parts = self._split_text_into_parts(text, num_parts=3)
        part_terms = [[] for _ in range(len(text_parts))]
        
        # 将术语分配到对应的文本部分
        for term in terms:
            part_index = self._get_text_part_index(term["start_pos"], len(text), len(text_parts))
            part_terms[part_index].append(term["standard_form"])
        
        # 分析每部分的术语使用
        part_analysis = []
        for i, part_term_list in enumerate(part_terms):
            term_counts = Counter(part_term_list)
            part_analysis.append({
                "part_index": i,
                "term_count": len(part_term_list),
                "unique_terms": len(set(part_term_list)),
                "most_common": term_counts.most_common(3)
            })
        
        return {
            "part_analysis": part_analysis,
            "term_introduction": self._analyze_term_introduction(part_terms),
            "usage_intensity": self._calculate_usage_intensity(part_terms)
        }
    
    def _calculate_frequency_distribution(self, term_counts: Counter) -> Dict[str, float]:
        """计算频率分布"""
        total_count = sum(term_counts.values())
        if total_count == 0:
            return {}
        
        return {
            "entropy": self._calculate_entropy(term_counts, total_count),
            "concentration": self._calculate_concentration(term_counts, total_count)
        }
    
    def _calculate_repetition_rate(self, term_counts: Counter) -> float:
        """计算重复率"""
        total_occurrences = sum(term_counts.values())
        unique_terms = len(term_counts)
        
        if unique_terms == 0:
            return 0.0
        
        return total_occurrences / unique_terms
    
    def _classify_text_section(self, relative_pos: float) -> str:
        """分类文本段落"""
        if relative_pos < 0.33:
            return "beginning"
        elif relative_pos < 0.67:
            return "middle"
        else:
            return "end"
    
    def _calculate_position_distribution(self, positions: List[Dict[str, Any]]) -> Dict[str, int]:
        """计算位置分布"""
        distribution = {"beginning": 0, "middle": 0, "end": 0}
        
        for pos in positions:
            distribution[pos["section"]] += 1
        
        return distribution
    
    def _calculate_section_usage(self, positions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """计算各段落的术语使用"""
        section_terms = defaultdict(list)
        
        for pos in positions:
            section_terms[pos["section"]].append(pos["term"])
        
        return dict(section_terms)
    
    def _analyze_position_clustering(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析位置聚类"""
        if len(positions) < 2:
            return {"clustering_score": 1.0, "clusters": []}
        
        # 简单的聚类分析：计算相邻术语的平均距离
        sorted_positions = sorted(positions, key=lambda x: x["absolute_pos"])
        distances = []
        
        for i in range(len(sorted_positions) - 1):
            distance = sorted_positions[i+1]["absolute_pos"] - sorted_positions[i]["absolute_pos"]
            distances.append(distance)
        
        avg_distance = sum(distances) / len(distances) if distances else 0
        
        return {
            "clustering_score": 1.0 / (1.0 + avg_distance / 100),  # 标准化聚类分数
            "average_distance": avg_distance,
            "clusters": self._identify_clusters(sorted_positions)
        }
    
    def _find_strong_associations(self, co_occurrences: Dict[str, Dict[str, int]]) -> List[Tuple[str, str, int]]:
        """找出强关联"""
        associations = []
        
        for term1, term2_counts in co_occurrences.items():
            for term2, count in term2_counts.items():
                if count >= 2:  # 至少共现2次
                    associations.append((term1, term2, count))
        
        return sorted(associations, key=lambda x: x[2], reverse=True)[:10]
    
    def _calculate_co_occurrence_strength(self, co_occurrences: Dict[str, Dict[str, int]]) -> float:
        """计算共现强度"""
        total_pairs = 0
        total_co_occurrences = 0
        
        for term1, term2_counts in co_occurrences.items():
            for term2, count in term2_counts.items():
                total_pairs += 1
                total_co_occurrences += count
        
        return total_co_occurrences / total_pairs if total_pairs > 0 else 0.0
    
    def _identify_dominant_patterns(self, context_types: Dict[str, int], 
                                  grammatical_roles: Dict[str, int]) -> Dict[str, str]:
        """识别主导模式"""
        dominant_context = max(context_types.items(), key=lambda x: x[1])[0] if context_types else "unknown"
        dominant_role = max(grammatical_roles.items(), key=lambda x: x[1])[0] if grammatical_roles else "unknown"
        
        return {
            "dominant_context_type": dominant_context,
            "dominant_grammatical_role": dominant_role
        }
    
    def _split_text_into_parts(self, text: str, num_parts: int = 3) -> List[str]:
        """将文本分成几个部分"""
        part_length = len(text) // num_parts
        parts = []
        
        for i in range(num_parts):
            start = i * part_length
            end = (i + 1) * part_length if i < num_parts - 1 else len(text)
            parts.append(text[start:end])
        
        return parts
    
    def _get_text_part_index(self, position: int, text_length: int, num_parts: int) -> int:
        """获取位置对应的文本部分索引"""
        part_length = text_length // num_parts
        part_index = min(position // part_length, num_parts - 1)
        return part_index
    
    def _analyze_term_introduction(self, part_terms: List[List[str]]) -> Dict[str, int]:
        """分析术语引入模式"""
        term_first_appearance = {}
        
        for part_index, terms in enumerate(part_terms):
            for term in terms:
                if term not in term_first_appearance:
                    term_first_appearance[term] = part_index
        
        introduction_pattern = defaultdict(int)
        for term, first_part in term_first_appearance.items():
            introduction_pattern[first_part] += 1
        
        return dict(introduction_pattern)
    
    def _calculate_usage_intensity(self, part_terms: List[List[str]]) -> List[float]:
        """计算使用强度"""
        intensities = []
        
        for terms in part_terms:
            intensity = len(terms) / len(part_terms) if part_terms else 0
            intensities.append(intensity)
        
        return intensities
    
    def _calculate_entropy(self, term_counts: Counter, total_count: int) -> float:
        """计算熵"""
        import math
        
        entropy = 0.0
        for count in term_counts.values():
            if count > 0:
                prob = count / total_count
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _calculate_concentration(self, term_counts: Counter, total_count: int) -> float:
        """计算集中度（基尼系数）"""
        if total_count == 0:
            return 0.0
        
        sorted_counts = sorted(term_counts.values(), reverse=True)
        n = len(sorted_counts)
        
        if n <= 1:
            return 0.0
        
        # 计算基尼系数
        cumulative_sum = 0
        for i, count in enumerate(sorted_counts):
            cumulative_sum += count * (2 * (i + 1) - n - 1)
        
        gini = cumulative_sum / (n * total_count)
        return abs(gini)
    
    def _identify_clusters(self, sorted_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别术语聚类"""
        if len(sorted_positions) < 2:
            return []
        
        clusters = []
        current_cluster = [sorted_positions[0]]
        cluster_threshold = 100  # 聚类阈值
        
        for i in range(1, len(sorted_positions)):
            distance = sorted_positions[i]["absolute_pos"] - sorted_positions[i-1]["absolute_pos"]
            
            if distance <= cluster_threshold:
                current_cluster.append(sorted_positions[i])
            else:
                if len(current_cluster) > 1:
                    clusters.append({
                        "terms": [pos["term"] for pos in current_cluster],
                        "start_pos": current_cluster[0]["absolute_pos"],
                        "end_pos": current_cluster[-1]["absolute_pos"],
                        "size": len(current_cluster)
                    })
                current_cluster = [sorted_positions[i]]
        
        # 处理最后一个聚类
        if len(current_cluster) > 1:
            clusters.append({
                "terms": [pos["term"] for pos in current_cluster],
                "start_pos": current_cluster[0]["absolute_pos"],
                "end_pos": current_cluster[-1]["absolute_pos"],
                "size": len(current_cluster)
            })
        
        return clusters
    
    def _build_usage_patterns(self) -> Dict[str, Any]:
        """构建使用模式"""
        return {
            "frequency_thresholds": {
                "low": 1,
                "medium": 3,
                "high": 5
            },
            "consistency_thresholds": {
                "poor": 0.5,
                "fair": 0.7,
                "good": 0.9
            },
            "clustering_thresholds": {
                "scattered": 0.3,
                "moderate": 0.6,
                "clustered": 0.8
            }
        }


class TerminologyUsageEvaluator(AbstractEvaluator):
    """术语使用评估器"""
    
    def __init__(self, name: str = "terminology_usage", weight: float = 1.0,
                 dictionary_data: Optional[Dict[str, Any]] = None):
        """
        初始化术语使用评估器
        
        Args:
            name: 评估器名称
            weight: 评估器权重
            dictionary_data: 术语词典数据
        """
        super().__init__(name, weight)
        self.dictionary = TerminologyDictionary(dictionary_data)
        self.recognizer = TerminologyRecognizer(self.dictionary)
        self.context_analyzer = ContextAnalyzer()
        self.usage_analyzer = UsagePatternAnalyzer()
    
    def _initialize_criteria(self) -> List[Criterion]:
        """初始化评估标准"""
        return [
            Criterion(
                name="contextual_appropriateness",
                description="上下文适当性",
                weight=0.4,
                threshold=0.7,
                evaluation_method="context_analysis"
            ),
            Criterion(
                name="usage_consistency",
                description="使用一致性",
                weight=0.3,
                threshold=0.8,
                evaluation_method="consistency_analysis"
            ),
            Criterion(
                name="pattern_quality",
                description="使用模式质量",
                weight=0.3,
                threshold=0.6,
                evaluation_method="pattern_analysis"
            )
        ]
    
    def _calculate_score(self, input_text: str, model_output: str, 
                        expected_output: str, context: Dict[str, Any]) -> float:
        """计算术语使用评估分数"""
        # 识别术语
        recognized_terms = self.recognizer.recognize_terms(model_output)
        
        if not recognized_terms:
            return 0.5  # 没有术语，给中等分数
        
        # 分析上下文适当性
        context_score = self._evaluate_contextual_appropriateness(recognized_terms, model_output)
        
        # 分析使用一致性
        consistency_score = self._evaluate_usage_consistency(recognized_terms, model_output)
        
        # 分析使用模式质量
        pattern_score = self._evaluate_pattern_quality(recognized_terms, model_output)
        
        # 加权计算总分
        total_score = (context_score * 0.4 + 
                      consistency_score * 0.3 + 
                      pattern_score * 0.3)
        
        return total_score
    
    def _evaluate_contextual_appropriateness(self, terms: List[Dict[str, Any]], text: str) -> float:
        """评估上下文适当性"""
        if not terms:
            return 1.0
        
        context_scores = []
        
        for term in terms:
            position = (term["start_pos"], term["end_pos"])
            context_analysis = self.context_analyzer.analyze_term_context(
                term["standard_form"], text, position
            )
            
            # 评估上下文质量
            context_quality = self._assess_context_quality(
                term["standard_form"], context_analysis
            )
            context_scores.append(context_quality)
        
        return sum(context_scores) / len(context_scores)
    
    def _assess_context_quality(self, term: str, context_analysis: Dict[str, Any]) -> float:
        """评估上下文质量"""
        score = 0.5  # 基础分数
        
        # 根据上下文类型调整分数
        context_type = context_analysis["context_type"]
        type_scores = {
            "definition": 0.9,
            "explanation": 0.8,
            "application": 0.8,
            "comparison": 0.7,
            "evaluation": 0.7,
            "general": 0.5
        }
        score = type_scores.get(context_type, 0.5)
        
        # 根据语法角色调整分数
        grammatical_role = context_analysis["grammatical_role"]
        if grammatical_role in ["subject", "object"]:
            score += 0.1
        elif grammatical_role == "modifier":
            score += 0.05
        
        # 根据语义关系调整分数
        semantic_relations = context_analysis["semantic_relations"]
        if semantic_relations:
            score += min(0.2, len(semantic_relations) * 0.05)
        
        # 根据修饰词调整分数
        modifiers = context_analysis["modifiers"]
        total_modifiers = sum(len(mod_list) for mod_list in modifiers.values())
        if total_modifiers > 0:
            score += min(0.1, total_modifiers * 0.02)
        
        # 根据共现术语调整分数
        co_occurring_terms = context_analysis["co_occurring_terms"]
        if co_occurring_terms:
            score += min(0.1, len(co_occurring_terms) * 0.02)
        
        return min(1.0, score)
    
    def _evaluate_usage_consistency(self, terms: List[Dict[str, Any]], text: str) -> float:
        """评估使用一致性"""
        usage_patterns = self.usage_analyzer.analyze_usage_patterns(terms, text)
        consistency_patterns = usage_patterns["consistency_patterns"]
        
        return consistency_patterns["overall_consistency"]
    
    def _evaluate_pattern_quality(self, terms: List[Dict[str, Any]], text: str) -> float:
        """评估使用模式质量"""
        usage_patterns = self.usage_analyzer.analyze_usage_patterns(terms, text)
        
        # 评估频率模式
        frequency_score = self._assess_frequency_quality(usage_patterns["frequency_patterns"])
        
        # 评估位置模式
        position_score = self._assess_position_quality(usage_patterns["position_patterns"])
        
        # 评估共现模式
        co_occurrence_score = self._assess_co_occurrence_quality(usage_patterns["co_occurrence_patterns"])
        
        # 综合评分
        pattern_score = (frequency_score + position_score + co_occurrence_score) / 3
        
        return pattern_score
    
    def _assess_frequency_quality(self, frequency_patterns: Dict[str, Any]) -> float:
        """评估频率质量"""
        repetition_rate = frequency_patterns["repetition_rate"]
        
        # 理想的重复率在1.5-3之间
        if 1.5 <= repetition_rate <= 3.0:
            return 1.0
        elif repetition_rate < 1.5:
            return 0.7 + (repetition_rate - 1.0) * 0.6  # 重复率太低
        else:
            return max(0.3, 1.0 - (repetition_rate - 3.0) * 0.1)  # 重复率太高
    
    def _assess_position_quality(self, position_patterns: Dict[str, Any]) -> float:
        """评估位置质量"""
        clustering = position_patterns["clustering"]
        clustering_score = clustering["clustering_score"]
        
        # 适度聚类是好的
        if 0.4 <= clustering_score <= 0.8:
            return 1.0
        elif clustering_score < 0.4:
            return 0.6 + clustering_score * 0.5  # 过于分散
        else:
            return 0.8 - (clustering_score - 0.8) * 0.5  # 过于聚集
    
    def _assess_co_occurrence_quality(self, co_occurrence_patterns: Dict[str, Any]) -> float:
        """评估共现质量"""
        co_occurrence_strength = co_occurrence_patterns["co_occurrence_strength"]
        strong_associations = co_occurrence_patterns["strong_associations"]
        
        # 基于共现强度和强关联数量评分
        strength_score = min(1.0, co_occurrence_strength / 2.0)
        association_score = min(1.0, len(strong_associations) / 5.0)
        
        return (strength_score + association_score) / 2
    
    def _generate_details(self, input_text: str, model_output: str, 
                         expected_output: str, context: Dict[str, Any], 
                         score: float) -> Dict[str, Any]:
        """生成详细评估信息"""
        recognized_terms = self.recognizer.recognize_terms(model_output)
        usage_patterns = self.usage_analyzer.analyze_usage_patterns(recognized_terms, model_output)
        
        # 分析每个术语的上下文
        term_contexts = []
        for term in recognized_terms:
            position = (term["start_pos"], term["end_pos"])
            context_analysis = self.context_analyzer.analyze_term_context(
                term["standard_form"], model_output, position
            )
            term_contexts.append({
                "term": term["standard_form"],
                "context_analysis": context_analysis,
                "context_quality": self._assess_context_quality(term["standard_form"], context_analysis)
            })
        
        # 计算各项分数
        context_score = self._evaluate_contextual_appropriateness(recognized_terms, model_output)
        consistency_score = self._evaluate_usage_consistency(recognized_terms, model_output)
        pattern_score = self._evaluate_pattern_quality(recognized_terms, model_output)
        
        return {
            "evaluator": self.name,
            "score": score,
            "recognized_terms": recognized_terms,
            "term_contexts": term_contexts,
            "usage_patterns": usage_patterns,
            "context_score": context_score,
            "consistency_score": consistency_score,
            "pattern_score": pattern_score,
            "quality_assessment": {
                "contextual_appropriateness": context_score,
                "usage_consistency": consistency_score,
                "pattern_quality": pattern_score
            },
            "recommendations": self._generate_usage_recommendations(usage_patterns, term_contexts)
        }
    
    def _generate_usage_recommendations(self, usage_patterns: Dict[str, Any], 
                                      term_contexts: List[Dict[str, Any]]) -> List[str]:
        """生成使用建议"""
        recommendations = []
        
        # 基于一致性模式的建议
        consistency_patterns = usage_patterns["consistency_patterns"]
        if consistency_patterns["overall_consistency"] < 0.7:
            inconsistent_terms = consistency_patterns["inconsistent_terms"]
            if inconsistent_terms:
                recommendations.append(f"建议统一以下术语的表述：{', '.join(inconsistent_terms[:3])}")
        
        # 基于频率模式的建议
        frequency_patterns = usage_patterns["frequency_patterns"]
        repetition_rate = frequency_patterns["repetition_rate"]
        if repetition_rate > 4:
            recommendations.append("术语使用过于频繁，建议适当减少重复或使用同义词")
        elif repetition_rate < 1.2:
            recommendations.append("术语使用较少，建议增加关键术语的使用频率")
        
        # 基于上下文质量的建议
        low_quality_contexts = [tc for tc in term_contexts if tc["context_quality"] < 0.6]
        if low_quality_contexts:
            recommendations.append("部分术语的上下文使用不够恰当，建议改进术语的使用场景")
        
        # 基于位置模式的建议
        position_patterns = usage_patterns["position_patterns"]
        clustering_score = position_patterns["clustering"]["clustering_score"]
        if clustering_score > 0.9:
            recommendations.append("术语过于集中，建议在文本中更均匀地分布术语")
        elif clustering_score < 0.3:
            recommendations.append("术语过于分散，建议在相关内容中集中使用相关术语")
        
        return recommendations[:5]  # 最多返回5条建议
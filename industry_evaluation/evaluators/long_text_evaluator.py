"""
长文本理解评估器
"""

import re
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from industry_evaluation.evaluators.base_evaluator import AbstractEvaluator
from industry_evaluation.models.data_models import Criterion


class TextStructureAnalyzer:
    """文本结构分析器"""
    
    def __init__(self):
        """初始化文本结构分析器"""
        self.structure_patterns = self._build_structure_patterns()
    
    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        分析文本结构
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Any]: 文本结构分析结果
        """
        analysis = {
            "basic_stats": self._calculate_basic_stats(text),
            "hierarchical_structure": self._analyze_hierarchical_structure(text),
            "paragraph_analysis": self._analyze_paragraphs(text),
            "sentence_analysis": self._analyze_sentences(text),
            "discourse_markers": self._identify_discourse_markers(text),
            "topic_segments": self._segment_topics(text),
            "coherence_analysis": self._analyze_coherence(text)
        }
        
        return analysis
    
    def _calculate_basic_stats(self, text: str) -> Dict[str, Any]:
        """计算基本统计信息"""
        # 字符统计
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        
        # 词汇统计
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        unique_words = len(set(word.lower() for word in words))
        
        # 句子统计
        sentences = re.split(r'[。！？.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # 段落统计
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            "char_count": char_count,
            "char_count_no_spaces": char_count_no_spaces,
            "word_count": word_count,
            "unique_words": unique_words,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_words_per_sentence": word_count / sentence_count if sentence_count > 0 else 0,
            "avg_sentences_per_paragraph": sentence_count / paragraph_count if paragraph_count > 0 else 0,
            "lexical_diversity": unique_words / word_count if word_count > 0 else 0
        }
    
    def _analyze_hierarchical_structure(self, text: str) -> Dict[str, Any]:
        """分析层次结构"""
        structure = {
            "headings": [],
            "sections": [],
            "subsections": [],
            "lists": [],
            "hierarchy_depth": 0
        }
        
        # 识别标题
        heading_patterns = [
            r'^(#{1,6})\s+(.+)$',  # Markdown标题
            r'^(\d+\.)\s+(.+)$',   # 数字标题
            r'^([一二三四五六七八九十]+[、.])\s+(.+)$',  # 中文数字标题
            r'^([A-Z][、.])\s+(.+)$'  # 字母标题
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            for pattern in heading_patterns:
                match = re.match(pattern, line)
                if match:
                    level_indicator = match.group(1)
                    title = match.group(2)
                    
                    # 确定层级
                    if level_indicator.startswith('#'):
                        level = len(level_indicator)
                    elif level_indicator.endswith('.') and level_indicator[:-1].isdigit():
                        level = 1
                    else:
                        level = 2
                    
                    structure["headings"].append({
                        "level": level,
                        "title": title,
                        "line_number": i + 1,
                        "indicator": level_indicator
                    })
                    break
        
        # 识别列表
        list_patterns = [
            r'^[-*+]\s+(.+)$',     # 无序列表
            r'^(\d+\.)\s+(.+)$',   # 有序列表
            r'^([a-zA-Z]\.)\s+(.+)$'  # 字母列表
        ]
        
        current_list = None
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if current_list:
                    structure["lists"].append(current_list)
                    current_list = None
                continue
            
            for pattern in list_patterns:
                match = re.match(pattern, line)
                if match:
                    if not current_list:
                        current_list = {
                            "type": "ordered" if match.group(1).endswith('.') else "unordered",
                            "items": [],
                            "start_line": i + 1
                        }
                    
                    current_list["items"].append({
                        "content": match.group(2) if len(match.groups()) > 1 else match.group(1),
                        "line_number": i + 1
                    })
                    break
            else:
                if current_list:
                    structure["lists"].append(current_list)
                    current_list = None
        
        if current_list:
            structure["lists"].append(current_list)
        
        # 计算层次深度
        if structure["headings"]:
            structure["hierarchy_depth"] = max(h["level"] for h in structure["headings"])
        
        return structure
    
    def _analyze_paragraphs(self, text: str) -> Dict[str, Any]:
        """分析段落"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return {"count": 0, "lengths": [], "avg_length": 0, "coherence_scores": []}
        
        paragraph_analysis = {
            "count": len(paragraphs),
            "lengths": [len(p) for p in paragraphs],
            "word_counts": [len(re.findall(r'\b\w+\b', p)) for p in paragraphs],
            "sentence_counts": [len(re.split(r'[。！？.!?]+', p)) for p in paragraphs],
            "coherence_scores": []
        }
        
        # 计算平均值
        paragraph_analysis["avg_length"] = sum(paragraph_analysis["lengths"]) / len(paragraphs)
        paragraph_analysis["avg_word_count"] = sum(paragraph_analysis["word_counts"]) / len(paragraphs)
        paragraph_analysis["avg_sentence_count"] = sum(paragraph_analysis["sentence_counts"]) / len(paragraphs)
        
        # 分析段落间连贯性
        for i in range(len(paragraphs) - 1):
            coherence = self._calculate_paragraph_coherence(paragraphs[i], paragraphs[i + 1])
            paragraph_analysis["coherence_scores"].append(coherence)
        
        return paragraph_analysis
    
    def _analyze_sentences(self, text: str) -> Dict[str, Any]:
        """分析句子"""
        sentences = re.split(r'[。！？.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {"count": 0, "lengths": [], "avg_length": 0, "complexity_scores": []}
        
        sentence_analysis = {
            "count": len(sentences),
            "lengths": [len(s) for s in sentences],
            "word_counts": [len(re.findall(r'\b\w+\b', s)) for s in sentences],
            "complexity_scores": []
        }
        
        # 计算句子复杂度
        for sentence in sentences:
            complexity = self._calculate_sentence_complexity(sentence)
            sentence_analysis["complexity_scores"].append(complexity)
        
        # 计算平均值
        sentence_analysis["avg_length"] = sum(sentence_analysis["lengths"]) / len(sentences)
        sentence_analysis["avg_word_count"] = sum(sentence_analysis["word_counts"]) / len(sentences)
        sentence_analysis["avg_complexity"] = sum(sentence_analysis["complexity_scores"]) / len(sentences)
        
        return sentence_analysis    
    d
ef _identify_discourse_markers(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """识别话语标记"""
        markers = {
            "temporal": ["首先", "然后", "接着", "最后", "同时", "之前", "之后"],
            "causal": ["因此", "所以", "由于", "因为", "导致", "引起"],
            "contrast": ["但是", "然而", "不过", "相反", "与此相对"],
            "addition": ["而且", "此外", "另外", "同时", "并且"],
            "emphasis": ["特别是", "尤其是", "重要的是", "值得注意的是"],
            "conclusion": ["总之", "综上所述", "总而言之", "最终", "结论是"]
        }
        
        found_markers = defaultdict(list)
        
        for marker_type, marker_list in markers.items():
            for marker in marker_list:
                positions = []
                start = 0
                while True:
                    pos = text.find(marker, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                
                if positions:
                    found_markers[marker_type].append({
                        "marker": marker,
                        "positions": positions,
                        "count": len(positions)
                    })
        
        return dict(found_markers)
    
    def _segment_topics(self, text: str) -> List[Dict[str, Any]]:
        """主题分割"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            return [{"start": 0, "end": len(paragraphs), "topic_words": [], "coherence": 1.0}]
        
        segments = []
        current_segment_start = 0
        
        # 简单的主题分割：基于段落间的词汇相似度
        for i in range(len(paragraphs) - 1):
            similarity = self._calculate_paragraph_coherence(paragraphs[i], paragraphs[i + 1])
            
            # 如果相似度低于阈值，认为是新主题的开始
            if similarity < 0.3:
                # 结束当前段落
                segment_text = ' '.join(paragraphs[current_segment_start:i + 1])
                topic_words = self._extract_topic_words(segment_text)
                
                segments.append({
                    "start": current_segment_start,
                    "end": i + 1,
                    "topic_words": topic_words,
                    "coherence": self._calculate_segment_coherence(paragraphs[current_segment_start:i + 1])
                })
                
                current_segment_start = i + 1
        
        # 添加最后一个段落
        if current_segment_start < len(paragraphs):
            segment_text = ' '.join(paragraphs[current_segment_start:])
            topic_words = self._extract_topic_words(segment_text)
            
            segments.append({
                "start": current_segment_start,
                "end": len(paragraphs),
                "topic_words": topic_words,
                "coherence": self._calculate_segment_coherence(paragraphs[current_segment_start:])
            })
        
        return segments
    
    def _analyze_coherence(self, text: str) -> Dict[str, Any]:
        """分析连贯性"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            return {"overall_coherence": 1.0, "local_coherence": [], "global_coherence": 1.0}
        
        # 局部连贯性（相邻段落间）
        local_coherence_scores = []
        for i in range(len(paragraphs) - 1):
            coherence = self._calculate_paragraph_coherence(paragraphs[i], paragraphs[i + 1])
            local_coherence_scores.append(coherence)
        
        # 全局连贯性（所有段落与第一段落的相关性）
        global_coherence_scores = []
        first_paragraph = paragraphs[0]
        for paragraph in paragraphs[1:]:
            coherence = self._calculate_paragraph_coherence(first_paragraph, paragraph)
            global_coherence_scores.append(coherence)
        
        return {
            "overall_coherence": sum(local_coherence_scores) / len(local_coherence_scores),
            "local_coherence": local_coherence_scores,
            "global_coherence": sum(global_coherence_scores) / len(global_coherence_scores) if global_coherence_scores else 1.0,
            "coherence_variance": self._calculate_variance(local_coherence_scores)
        }
    
    def _calculate_paragraph_coherence(self, para1: str, para2: str) -> float:
        """计算段落间连贯性"""
        # 提取关键词
        words1 = set(re.findall(r'\b\w+\b', para1.lower()))
        words2 = set(re.findall(r'\b\w+\b', para2.lower()))
        
        # 移除停用词
        stop_words = {"的", "了", "在", "是", "有", "和", "与", "或", "但", "而", "也", "都", "很", "更", "最"}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_sentence_complexity(self, sentence: str) -> float:
        """计算句子复杂度"""
        # 基于多个因素计算复杂度
        complexity_score = 0.0
        
        # 长度因素
        length_score = min(1.0, len(sentence) / 100)  # 标准化到0-1
        complexity_score += length_score * 0.3
        
        # 从句数量
        clause_indicators = ["，", "；", "：", "因为", "所以", "如果", "那么", "虽然", "但是"]
        clause_count = sum(sentence.count(indicator) for indicator in clause_indicators)
        clause_score = min(1.0, clause_count / 5)  # 标准化到0-1
        complexity_score += clause_score * 0.4
        
        # 专业术语密度
        technical_terms = ["算法", "模型", "数据", "系统", "方法", "技术", "分析", "评估"]
        term_count = sum(1 for term in technical_terms if term in sentence)
        term_score = min(1.0, term_count / 3)  # 标准化到0-1
        complexity_score += term_score * 0.3
        
        return complexity_score
    
    def _extract_topic_words(self, text: str, top_k: int = 5) -> List[str]:
        """提取主题词"""
        # 简单的TF统计
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 移除停用词
        stop_words = {"的", "了", "在", "是", "有", "和", "与", "或", "但", "而", "也", "都", "很", "更", "最", "这", "那", "一个", "一种"}
        words = [word for word in words if word not in stop_words and len(word) > 1]
        
        # 计算词频
        word_freq = Counter(words)
        
        # 返回频率最高的词
        return [word for word, freq in word_freq.most_common(top_k)]
    
    def _calculate_segment_coherence(self, paragraphs: List[str]) -> float:
        """计算段落组的连贯性"""
        if len(paragraphs) <= 1:
            return 1.0
        
        coherence_scores = []
        for i in range(len(paragraphs) - 1):
            coherence = self._calculate_paragraph_coherence(paragraphs[i], paragraphs[i + 1])
            coherence_scores.append(coherence)
        
        return sum(coherence_scores) / len(coherence_scores)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """计算方差"""
        if not values:
            return 0.0
        
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance
    
    def _build_structure_patterns(self) -> Dict[str, List[str]]:
        """构建结构模式"""
        return {
            "headings": [
                r'^(#{1,6})\s+(.+)$',
                r'^(\d+\.)\s+(.+)$',
                r'^([一二三四五六七八九十]+[、.])\s+(.+)$'
            ],
            "lists": [
                r'^[-*+]\s+(.+)$',
                r'^(\d+\.)\s+(.+)$',
                r'^([a-zA-Z]\.)\s+(.+)$'
            ],
            "emphasis": [
                r'\*\*(.+?)\*\*',
                r'__(.+?)__',
                r'【(.+?)】'
            ]
        }


class KeyInformationExtractor:
    """关键信息提取器"""
    
    def __init__(self):
        """初始化关键信息提取器"""
        self.extraction_patterns = self._build_extraction_patterns()
    
    def extract_key_information(self, text: str) -> Dict[str, Any]:
        """
        提取关键信息
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Any]: 关键信息提取结果
        """
        extraction_result = {
            "main_topics": self._extract_main_topics(text),
            "key_entities": self._extract_key_entities(text),
            "important_facts": self._extract_important_facts(text),
            "conclusions": self._extract_conclusions(text),
            "numerical_data": self._extract_numerical_data(text),
            "relationships": self._extract_relationships(text),
            "temporal_information": self._extract_temporal_info(text)
        }
        
        return extraction_result
    
    def _extract_main_topics(self, text: str) -> List[Dict[str, Any]]:
        """提取主要主题"""
        topics = []
        
        # 基于段落首句提取主题
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            sentences = re.split(r'[。！？.!?]+', paragraph)
            if sentences and sentences[0].strip():
                first_sentence = sentences[0].strip()
                
                # 提取主题词
                topic_words = self._extract_topic_words_from_sentence(first_sentence)
                
                topics.append({
                    "paragraph_index": i,
                    "topic_sentence": first_sentence,
                    "topic_words": topic_words,
                    "confidence": self._calculate_topic_confidence(first_sentence)
                })
        
        return topics
    
    def _extract_key_entities(self, text: str) -> Dict[str, List[str]]:
        """提取关键实体"""
        entities = {
            "technical_terms": [],
            "organizations": [],
            "products": [],
            "concepts": [],
            "metrics": []
        }
        
        # 技术术语
        tech_patterns = [
            r'\b(机器学习|深度学习|人工智能|神经网络|算法|模型|数据挖掘)\b',
            r'\b([A-Z]{2,})\b',  # 缩写
            r'\b(\w+算法|\w+模型|\w+系统|\w+平台)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["technical_terms"].extend([match if isinstance(match, str) else match[0] for match in matches])
        
        # 组织机构
        org_patterns = [
            r'\b(\w+公司|\w+企业|\w+集团|\w+机构)\b',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'  # 英文组织名
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            entities["organizations"].extend(matches)
        
        # 产品名称
        product_patterns = [
            r'\b(\w+系统|\w+平台|\w+工具|\w+软件)\b'
        ]
        
        for pattern in product_patterns:
            matches = re.findall(pattern, text)
            entities["products"].extend(matches)
        
        # 去重
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _extract_important_facts(self, text: str) -> List[Dict[str, Any]]:
        """提取重要事实"""
        facts = []
        
        # 查找包含重要信息的句子
        sentences = re.split(r'[。！？.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            importance_score = self._calculate_fact_importance(sentence)
            
            if importance_score > 0.6:
                facts.append({
                    "content": sentence,
                    "importance_score": importance_score,
                    "type": self._classify_fact_type(sentence)
                })
        
        # 按重要性排序
        facts.sort(key=lambda x: x["importance_score"], reverse=True)
        
        return facts[:10]  # 返回前10个最重要的事实
    
    def _extract_conclusions(self, text: str) -> List[str]:
        """提取结论"""
        conclusion_patterns = [
            r'(因此|所以|总之|综上所述|可以得出|结论是)[，：:]\s*(.+?)(?=[。！？.!?]|$)',
            r'(最终|最后|总而言之)[，：:]\s*(.+?)(?=[。！？.!?]|$)'
        ]
        
        conclusions = []
        
        for pattern in conclusion_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                conclusion = match.group(2).strip()
                if conclusion and len(conclusion) > 5:
                    conclusions.append(conclusion)
        
        return list(set(conclusions))  # 去重
    
    def _extract_numerical_data(self, text: str) -> List[Dict[str, Any]]:
        """提取数值数据"""
        numerical_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # 百分比
            r'(\d+(?:\.\d+)?)\s*(万|千|百|亿)',  # 中文数量单位
            r'(\d{4})\s*年',  # 年份
            r'(\d+(?:\.\d+)?)\s*(米|公里|千米|厘米)',  # 长度单位
            r'(\d+(?:\.\d+)?)\s*(秒|分钟|小时|天|月|年)',  # 时间单位
            r'(\d+(?:\.\d+)?)\s*(元|美元|欧元)',  # 货币单位
        ]
        
        numerical_data = []
        
        for pattern in numerical_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                value = match.group(1)
                unit = match.group(2) if len(match.groups()) > 1 else ""
                
                numerical_data.append({
                    "value": float(value),
                    "unit": unit,
                    "context": text[max(0, match.start()-20):match.end()+20],
                    "position": match.start()
                })
        
        return numerical_data
    
    def _extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """提取关系信息"""
        relationship_patterns = [
            r'(.+?)\s*(导致|引起|造成|产生)\s*(.+?)(?=[。！？.!?]|$)',
            r'(.+?)\s*(影响|决定|取决于)\s*(.+?)(?=[。！？.!?]|$)',
            r'(.+?)\s*(与|和)\s*(.+?)\s*(相关|关联)(?=[。！？.!?]|$)',
            r'(.+?)\s*(基于|依据|根据)\s*(.+?)(?=[。！？.!?]|$)'
        ]
        
        relationships = []
        
        for pattern in relationship_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 3:
                    subject = match.group(1).strip()
                    relation = match.group(2).strip()
                    object_term = match.group(3).strip()
                    
                    relationships.append({
                        "subject": subject,
                        "relation": relation,
                        "object": object_term,
                        "confidence": 0.7
                    })
        
        return relationships
    
    def _extract_temporal_info(self, text: str) -> List[Dict[str, Any]]:
        """提取时间信息"""
        temporal_patterns = [
            r'(\d{4})\s*年',
            r'(\d{1,2})\s*月',
            r'(\d{1,2})\s*日',
            r'(昨天|今天|明天|前天|后天)',
            r'(上周|本周|下周|上月|本月|下月|去年|今年|明年)',
            r'(最近|近期|不久前|将来|未来)'
        ]
        
        temporal_info = []
        
        for pattern in temporal_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                temporal_info.append({
                    "expression": match.group(1),
                    "context": text[max(0, match.start()-15):match.end()+15],
                    "position": match.start(),
                    "type": self._classify_temporal_type(match.group(1))
                })
        
        return temporal_info  
  
    def _extract_topic_words_from_sentence(self, sentence: str) -> List[str]:
        """从句子中提取主题词"""
        # 移除停用词并提取关键词
        words = re.findall(r'\b\w+\b', sentence.lower())
        stop_words = {"的", "了", "在", "是", "有", "和", "与", "或", "但", "而", "也", "都", "很", "更", "最", "这", "那"}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        # 优先选择技术术语
        tech_terms = ["算法", "模型", "数据", "系统", "方法", "技术", "分析", "评估", "优化", "训练"]
        priority_words = [word for word in keywords if word in tech_terms]
        
        # 如果有技术术语，优先返回；否则返回前几个关键词
        if priority_words:
            return priority_words[:3]
        else:
            return keywords[:3]
    
    def _calculate_topic_confidence(self, sentence: str) -> float:
        """计算主题置信度"""
        confidence = 0.5  # 基础置信度
        
        # 如果包含技术术语，提高置信度
        tech_terms = ["算法", "模型", "数据", "系统", "方法", "技术"]
        if any(term in sentence for term in tech_terms):
            confidence += 0.2
        
        # 如果是定义性句子，提高置信度
        if any(word in sentence for word in ["是", "指", "定义为", "称为"]):
            confidence += 0.2
        
        # 如果句子较长且结构完整，提高置信度
        if len(sentence) > 20 and any(punct in sentence for punct in ["，", "；", "："]):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_fact_importance(self, sentence: str) -> float:
        """计算事实重要性"""
        importance = 0.3  # 基础重要性
        
        # 包含数字或统计信息
        if re.search(r'\d+(?:\.\d+)?', sentence):
            importance += 0.2
        
        # 包含重要性指示词
        importance_indicators = ["重要", "关键", "核心", "主要", "显著", "明显", "证明", "表明"]
        if any(indicator in sentence for indicator in importance_indicators):
            importance += 0.3
        
        # 包含结论性词汇
        conclusion_words = ["因此", "所以", "结果", "发现", "证实", "表明"]
        if any(word in sentence for word in conclusion_words):
            importance += 0.2
        
        # 包含专业术语
        tech_terms = ["算法", "模型", "数据", "系统", "性能", "准确率", "效果"]
        tech_count = sum(1 for term in tech_terms if term in sentence)
        importance += min(0.3, tech_count * 0.1)
        
        return min(1.0, importance)
    
    def _classify_fact_type(self, sentence: str) -> str:
        """分类事实类型"""
        if re.search(r'\d+(?:\.\d+)?', sentence):
            return "quantitative"
        elif any(word in sentence for word in ["定义", "概念", "是指", "称为"]):
            return "definition"
        elif any(word in sentence for word in ["方法", "步骤", "过程", "流程"]):
            return "procedural"
        elif any(word in sentence for word in ["结果", "效果", "性能", "表现"]):
            return "result"
        elif any(word in sentence for word in ["原因", "因为", "由于", "导致"]):
            return "causal"
        else:
            return "descriptive"
    
    def _classify_temporal_type(self, expression: str) -> str:
        """分类时间类型"""
        if re.match(r'\d{4}', expression):
            return "year"
        elif re.match(r'\d{1,2}', expression):
            return "month_or_day"
        elif expression in ["昨天", "今天", "明天", "前天", "后天"]:
            return "relative_day"
        elif expression in ["上周", "本周", "下周", "上月", "本月", "下月", "去年", "今年", "明年"]:
            return "relative_period"
        else:
            return "general"
    
    def _build_extraction_patterns(self) -> Dict[str, List[str]]:
        """构建提取模式"""
        return {
            "definitions": [
                r'(.+?)\s*(是|指|定义为|称为)\s*(.+?)(?=[。！？.!?]|$)',
                r'(.+?)\s*[：:]\s*(.+?)(?=[。！？.!?]|$)'
            ],
            "procedures": [
                r'(步骤|方法|过程|流程)[：:]\s*(.+?)(?=[。！？.!?]|$)',
                r'(首先|然后|接着|最后)[，,]\s*(.+?)(?=[。！？.!?]|$)'
            ],
            "results": [
                r'(结果|效果|性能|表现)[：:]\s*(.+?)(?=[。！？.!?]|$)',
                r'(显示|表明|证明|发现)\s*(.+?)(?=[。！？.!?]|$)'
            ]
        }


class LongTextEvaluator(AbstractEvaluator):
    """长文本理解评估器"""
    
    def __init__(self, name: str = "long_text_understanding", weight: float = 1.0):
        """
        初始化长文本理解评估器
        
        Args:
            name: 评估器名称
            weight: 评估器权重
        """
        super().__init__(name, weight)
        self.structure_analyzer = TextStructureAnalyzer()
        self.info_extractor = KeyInformationExtractor()
    
    def _initialize_criteria(self) -> List[Criterion]:
        """初始化评估标准"""
        return [
            Criterion(
                name="structure_understanding",
                description="文本结构理解",
                weight=0.25,
                threshold=0.6,
                evaluation_method="structure_analysis"
            ),
            Criterion(
                name="key_information_extraction",
                description="关键信息提取",
                weight=0.3,
                threshold=0.7,
                evaluation_method="information_extraction"
            ),
            Criterion(
                name="coherence_maintenance",
                description="连贯性保持",
                weight=0.25,
                threshold=0.6,
                evaluation_method="coherence_analysis"
            ),
            Criterion(
                name="comprehension_depth",
                description="理解深度",
                weight=0.2,
                threshold=0.5,
                evaluation_method="depth_analysis"
            )
        ]
    
    def _calculate_score(self, input_text: str, model_output: str, 
                        expected_output: str, context: Dict[str, Any]) -> float:
        """计算长文本理解评估分数"""
        # 分析输入文本结构
        input_structure = self.structure_analyzer.analyze_text_structure(input_text)
        
        # 分析模型输出
        output_structure = self.structure_analyzer.analyze_text_structure(model_output)
        output_info = self.info_extractor.extract_key_information(model_output)
        
        # 如果有期望输出，也进行分析
        expected_info = {}
        if expected_output:
            expected_info = self.info_extractor.extract_key_information(expected_output)
        
        # 计算各项分数
        structure_score = self._evaluate_structure_understanding(input_structure, output_structure)
        extraction_score = self._evaluate_information_extraction(output_info, expected_info, input_text)
        coherence_score = self._evaluate_coherence_maintenance(output_structure)
        depth_score = self._evaluate_comprehension_depth(output_info, input_text)
        
        # 加权计算总分
        total_score = (structure_score * 0.25 + 
                      extraction_score * 0.3 + 
                      coherence_score * 0.25 + 
                      depth_score * 0.2)
        
        return total_score
    
    def _evaluate_structure_understanding(self, input_structure: Dict[str, Any], 
                                        output_structure: Dict[str, Any]) -> float:
        """评估结构理解能力"""
        score = 0.5  # 基础分数
        
        # 比较基本统计信息
        input_stats = input_structure["basic_stats"]
        output_stats = output_structure["basic_stats"]
        
        # 评估长度适当性（输出不应过短或过长）
        length_ratio = output_stats["char_count"] / input_stats["char_count"] if input_stats["char_count"] > 0 else 0
        if 0.1 <= length_ratio <= 0.8:  # 合理的压缩比例
            score += 0.2
        
        # 评估段落结构保持
        input_paragraphs = input_structure["paragraph_analysis"]["count"]
        output_paragraphs = output_structure["paragraph_analysis"]["count"]
        
        if input_paragraphs > 0:
            paragraph_ratio = output_paragraphs / input_paragraphs
            if 0.3 <= paragraph_ratio <= 1.0:  # 保持合理的段落结构
                score += 0.2
        
        # 评估层次结构理解
        input_hierarchy = input_structure["hierarchical_structure"]["hierarchy_depth"]
        output_hierarchy = output_structure["hierarchical_structure"]["hierarchy_depth"]
        
        if input_hierarchy > 0 and output_hierarchy > 0:
            score += 0.1  # 保持了层次结构
        
        return min(1.0, score)
    
    def _evaluate_information_extraction(self, output_info: Dict[str, Any], 
                                       expected_info: Dict[str, Any], 
                                       input_text: str) -> float:
        """评估信息提取能力"""
        score = 0.0
        
        # 评估主题提取
        main_topics = output_info.get("main_topics", [])
        if main_topics:
            topic_score = min(1.0, len(main_topics) / 5.0)  # 理想情况下提取5个主题
            score += topic_score * 0.2
        
        # 评估关键实体提取
        entities = output_info.get("key_entities", {})
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        if total_entities > 0:
            entity_score = min(1.0, total_entities / 10.0)  # 理想情况下提取10个实体
            score += entity_score * 0.2
        
        # 评估重要事实提取
        facts = output_info.get("important_facts", [])
        if facts:
            fact_score = min(1.0, len(facts) / 5.0)  # 理想情况下提取5个重要事实
            avg_importance = sum(fact["importance_score"] for fact in facts) / len(facts)
            score += (fact_score * avg_importance) * 0.3
        
        # 评估结论提取
        conclusions = output_info.get("conclusions", [])
        if conclusions:
            conclusion_score = min(1.0, len(conclusions) / 3.0)  # 理想情况下提取3个结论
            score += conclusion_score * 0.2
        
        # 评估数值信息提取
        numerical_data = output_info.get("numerical_data", [])
        if numerical_data:
            numerical_score = min(1.0, len(numerical_data) / 5.0)
            score += numerical_score * 0.1
        
        # 如果有期望输出，计算匹配度
        if expected_info:
            match_score = self._calculate_information_match(output_info, expected_info)
            score = (score + match_score) / 2
        
        return score
    
    def _evaluate_coherence_maintenance(self, output_structure: Dict[str, Any]) -> float:
        """评估连贯性保持"""
        coherence_analysis = output_structure.get("coherence_analysis", {})
        
        if not coherence_analysis:
            return 0.5
        
        overall_coherence = coherence_analysis.get("overall_coherence", 0.5)
        global_coherence = coherence_analysis.get("global_coherence", 0.5)
        
        # 综合局部和全局连贯性
        coherence_score = (overall_coherence * 0.6 + global_coherence * 0.4)
        
        # 考虑连贯性的稳定性（方差越小越好）
        coherence_variance = coherence_analysis.get("coherence_variance", 0.0)
        stability_bonus = max(0, 0.2 - coherence_variance)  # 方差小于0.2时给予奖励
        
        return min(1.0, coherence_score + stability_bonus)
    
    def _evaluate_comprehension_depth(self, output_info: Dict[str, Any], input_text: str) -> float:
        """评估理解深度"""
        depth_score = 0.0
        
        # 评估关系理解
        relationships = output_info.get("relationships", [])
        if relationships:
            relation_score = min(1.0, len(relationships) / 5.0)
            avg_confidence = sum(rel["confidence"] for rel in relationships) / len(relationships)
            depth_score += (relation_score * avg_confidence) * 0.4
        
        # 评估因果理解
        causal_relations = [rel for rel in relationships if rel["relation"] in ["导致", "引起", "造成"]]
        if causal_relations:
            depth_score += 0.2
        
        # 评估时间理解
        temporal_info = output_info.get("temporal_information", [])
        if temporal_info:
            temporal_score = min(1.0, len(temporal_info) / 3.0)
            depth_score += temporal_score * 0.2
        
        # 评估概念理解（基于主题词的专业性）
        main_topics = output_info.get("main_topics", [])
        professional_topics = 0
        total_topics = 0
        
        for topic in main_topics:
            topic_words = topic.get("topic_words", [])
            total_topics += len(topic_words)
            
            # 检查专业术语
            tech_terms = ["算法", "模型", "数据", "系统", "方法", "技术", "分析", "评估"]
            professional_topics += sum(1 for word in topic_words if word in tech_terms)
        
        if total_topics > 0:
            professional_ratio = professional_topics / total_topics
            depth_score += professional_ratio * 0.2
        
        return min(1.0, depth_score)
    
    def _calculate_information_match(self, output_info: Dict[str, Any], 
                                   expected_info: Dict[str, Any]) -> float:
        """计算信息匹配度"""
        match_scores = []
        
        # 比较主题
        output_topics = set()
        for topic in output_info.get("main_topics", []):
            output_topics.update(topic.get("topic_words", []))
        
        expected_topics = set()
        for topic in expected_info.get("main_topics", []):
            expected_topics.update(topic.get("topic_words", []))
        
        if expected_topics:
            topic_match = len(output_topics.intersection(expected_topics)) / len(expected_topics)
            match_scores.append(topic_match)
        
        # 比较实体
        output_entities = set()
        for entity_list in output_info.get("key_entities", {}).values():
            output_entities.update(entity_list)
        
        expected_entities = set()
        for entity_list in expected_info.get("key_entities", {}).values():
            expected_entities.update(entity_list)
        
        if expected_entities:
            entity_match = len(output_entities.intersection(expected_entities)) / len(expected_entities)
            match_scores.append(entity_match)
        
        # 比较结论
        output_conclusions = set(output_info.get("conclusions", []))
        expected_conclusions = set(expected_info.get("conclusions", []))
        
        if expected_conclusions:
            conclusion_match = len(output_conclusions.intersection(expected_conclusions)) / len(expected_conclusions)
            match_scores.append(conclusion_match)
        
        return sum(match_scores) / len(match_scores) if match_scores else 0.5    

    def _generate_details(self, input_text: str, model_output: str, 
                         expected_output: str, context: Dict[str, Any], 
                         score: float) -> Dict[str, Any]:
        """生成详细评估信息"""
        # 分析输入和输出
        input_structure = self.structure_analyzer.analyze_text_structure(input_text)
        output_structure = self.structure_analyzer.analyze_text_structure(model_output)
        output_info = self.info_extractor.extract_key_information(model_output)
        
        expected_structure = {}
        expected_info = {}
        if expected_output:
            expected_structure = self.structure_analyzer.analyze_text_structure(expected_output)
            expected_info = self.info_extractor.extract_key_information(expected_output)
        
        # 计算各项分数
        structure_score = self._evaluate_structure_understanding(input_structure, output_structure)
        extraction_score = self._evaluate_information_extraction(output_info, expected_info, input_text)
        coherence_score = self._evaluate_coherence_maintenance(output_structure)
        depth_score = self._evaluate_comprehension_depth(output_info, input_text)
        
        return {
            "evaluator": self.name,
            "score": score,
            "input_analysis": {
                "structure": input_structure,
                "length": len(input_text),
                "complexity": self._calculate_text_complexity(input_text)
            },
            "output_analysis": {
                "structure": output_structure,
                "extracted_info": output_info,
                "length": len(model_output),
                "compression_ratio": len(model_output) / len(input_text) if len(input_text) > 0 else 0
            },
            "expected_analysis": {
                "structure": expected_structure,
                "extracted_info": expected_info
            } if expected_output else {},
            "evaluation_scores": {
                "structure_understanding": structure_score,
                "information_extraction": extraction_score,
                "coherence_maintenance": coherence_score,
                "comprehension_depth": depth_score
            },
            "quality_metrics": {
                "information_density": self._calculate_information_density(output_info),
                "structural_preservation": self._calculate_structural_preservation(input_structure, output_structure),
                "coherence_quality": coherence_score,
                "depth_indicators": self._analyze_depth_indicators(output_info)
            },
            "comparison_analysis": self._compare_with_expected(output_info, expected_info) if expected_info else {},
            "recommendations": self._generate_comprehension_recommendations(
                input_structure, output_structure, output_info, 
                structure_score, extraction_score, coherence_score, depth_score
            )
        }
    
    def _calculate_text_complexity(self, text: str) -> Dict[str, float]:
        """计算文本复杂度"""
        sentences = re.split(r'[。！？.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {"lexical": 0.0, "syntactic": 0.0, "semantic": 0.0, "overall": 0.0}
        
        # 词汇复杂度
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = len(set(words))
        lexical_complexity = unique_words / len(words) if words else 0.0
        
        # 句法复杂度
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
        syntactic_complexity = min(1.0, avg_sentence_length / 50)  # 标准化到0-1
        
        # 语义复杂度（基于专业术语密度）
        tech_terms = ["算法", "模型", "数据", "系统", "方法", "技术", "分析", "评估", "优化", "训练"]
        tech_count = sum(1 for term in tech_terms if term in text)
        semantic_complexity = min(1.0, tech_count / 10)  # 标准化到0-1
        
        overall_complexity = (lexical_complexity + syntactic_complexity + semantic_complexity) / 3
        
        return {
            "lexical": lexical_complexity,
            "syntactic": syntactic_complexity,
            "semantic": semantic_complexity,
            "overall": overall_complexity
        }
    
    def _calculate_information_density(self, extracted_info: Dict[str, Any]) -> float:
        """计算信息密度"""
        density_score = 0.0
        
        # 主题密度
        topics = extracted_info.get("main_topics", [])
        density_score += min(1.0, len(topics) / 5.0) * 0.2
        
        # 实体密度
        entities = extracted_info.get("key_entities", {})
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        density_score += min(1.0, total_entities / 15.0) * 0.3
        
        # 事实密度
        facts = extracted_info.get("important_facts", [])
        density_score += min(1.0, len(facts) / 8.0) * 0.3
        
        # 关系密度
        relationships = extracted_info.get("relationships", [])
        density_score += min(1.0, len(relationships) / 5.0) * 0.2
        
        return density_score
    
    def _calculate_structural_preservation(self, input_structure: Dict[str, Any], 
                                         output_structure: Dict[str, Any]) -> float:
        """计算结构保持度"""
        preservation_score = 0.0
        
        # 段落结构保持
        input_paragraphs = input_structure["paragraph_analysis"]["count"]
        output_paragraphs = output_structure["paragraph_analysis"]["count"]
        
        if input_paragraphs > 0:
            paragraph_preservation = min(1.0, output_paragraphs / input_paragraphs)
            preservation_score += paragraph_preservation * 0.4
        
        # 层次结构保持
        input_hierarchy = input_structure["hierarchical_structure"]["hierarchy_depth"]
        output_hierarchy = output_structure["hierarchical_structure"]["hierarchy_depth"]
        
        if input_hierarchy > 0:
            hierarchy_preservation = min(1.0, output_hierarchy / input_hierarchy)
            preservation_score += hierarchy_preservation * 0.3
        
        # 话语标记保持
        input_markers = input_structure["discourse_markers"]
        output_markers = output_structure["discourse_markers"]
        
        total_input_markers = sum(len(markers) for markers in input_markers.values())
        total_output_markers = sum(len(markers) for markers in output_markers.values())
        
        if total_input_markers > 0:
            marker_preservation = min(1.0, total_output_markers / total_input_markers)
            preservation_score += marker_preservation * 0.3
        
        return preservation_score
    
    def _analyze_depth_indicators(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析深度指标"""
        indicators = {
            "causal_understanding": 0,
            "temporal_understanding": 0,
            "relational_understanding": 0,
            "conceptual_understanding": 0
        }
        
        # 因果理解
        relationships = extracted_info.get("relationships", [])
        causal_relations = [rel for rel in relationships if rel["relation"] in ["导致", "引起", "造成", "产生"]]
        indicators["causal_understanding"] = len(causal_relations)
        
        # 时间理解
        temporal_info = extracted_info.get("temporal_information", [])
        indicators["temporal_understanding"] = len(temporal_info)
        
        # 关系理解
        indicators["relational_understanding"] = len(relationships)
        
        # 概念理解
        entities = extracted_info.get("key_entities", {})
        tech_entities = entities.get("technical_terms", [])
        concept_entities = entities.get("concepts", [])
        indicators["conceptual_understanding"] = len(tech_entities) + len(concept_entities)
        
        return indicators
    
    def _compare_with_expected(self, output_info: Dict[str, Any], 
                             expected_info: Dict[str, Any]) -> Dict[str, Any]:
        """与期望输出比较"""
        comparison = {
            "topic_overlap": 0.0,
            "entity_overlap": 0.0,
            "conclusion_overlap": 0.0,
            "missing_topics": [],
            "missing_entities": [],
            "extra_information": []
        }
        
        # 主题重叠
        output_topics = set()
        for topic in output_info.get("main_topics", []):
            output_topics.update(topic.get("topic_words", []))
        
        expected_topics = set()
        for topic in expected_info.get("main_topics", []):
            expected_topics.update(topic.get("topic_words", []))
        
        if expected_topics:
            topic_intersection = output_topics.intersection(expected_topics)
            comparison["topic_overlap"] = len(topic_intersection) / len(expected_topics)
            comparison["missing_topics"] = list(expected_topics - output_topics)
        
        # 实体重叠
        output_entities = set()
        for entity_list in output_info.get("key_entities", {}).values():
            output_entities.update(entity_list)
        
        expected_entities = set()
        for entity_list in expected_info.get("key_entities", {}).values():
            expected_entities.update(entity_list)
        
        if expected_entities:
            entity_intersection = output_entities.intersection(expected_entities)
            comparison["entity_overlap"] = len(entity_intersection) / len(expected_entities)
            comparison["missing_entities"] = list(expected_entities - output_entities)
        
        # 额外信息
        extra_topics = output_topics - expected_topics
        extra_entities = output_entities - expected_entities
        comparison["extra_information"] = list(extra_topics.union(extra_entities))
        
        return comparison
    
    def _generate_comprehension_recommendations(self, input_structure: Dict[str, Any], 
                                              output_structure: Dict[str, Any], 
                                              output_info: Dict[str, Any],
                                              structure_score: float, 
                                              extraction_score: float, 
                                              coherence_score: float, 
                                              depth_score: float) -> List[str]:
        """生成理解改进建议"""
        recommendations = []
        
        # 基于结构理解的建议
        if structure_score < 0.6:
            input_paragraphs = input_structure["paragraph_analysis"]["count"]
            output_paragraphs = output_structure["paragraph_analysis"]["count"]
            
            if input_paragraphs > output_paragraphs * 2:
                recommendations.append("建议保持更多的段落结构，以体现对原文组织的理解")
            
            if input_structure["hierarchical_structure"]["hierarchy_depth"] > 0 and output_structure["hierarchical_structure"]["hierarchy_depth"] == 0:
                recommendations.append("建议保持原文的层次结构，如标题、小节等")
        
        # 基于信息提取的建议
        if extraction_score < 0.7:
            topics = output_info.get("main_topics", [])
            if len(topics) < 3:
                recommendations.append("建议提取更多的主要主题，以体现对文本内容的全面理解")
            
            facts = output_info.get("important_facts", [])
            if len(facts) < 3:
                recommendations.append("建议识别和提取更多的重要事实信息")
            
            entities = output_info.get("key_entities", {})
            total_entities = sum(len(entity_list) for entity_list in entities.values())
            if total_entities < 5:
                recommendations.append("建议识别更多的关键实体，如技术术语、组织名称等")
        
        # 基于连贯性的建议
        if coherence_score < 0.6:
            recommendations.append("建议改善输出的逻辑连贯性，使用更多的过渡词和连接词")
            
            coherence_variance = output_structure.get("coherence_analysis", {}).get("coherence_variance", 0)
            if coherence_variance > 0.3:
                recommendations.append("建议保持更一致的连贯性水平，避免局部连贯性的大幅波动")
        
        # 基于理解深度的建议
        if depth_score < 0.5:
            relationships = output_info.get("relationships", [])
            if len(relationships) < 2:
                recommendations.append("建议识别和表达更多的概念间关系，体现深层理解")
            
            causal_relations = [rel for rel in relationships if rel["relation"] in ["导致", "引起", "造成"]]
            if not causal_relations:
                recommendations.append("建议识别因果关系，体现对逻辑关系的理解")
            
            temporal_info = output_info.get("temporal_information", [])
            if not temporal_info:
                recommendations.append("建议关注时间信息，体现对时序关系的理解")
        
        # 基于文本长度的建议
        input_length = len(input_structure["basic_stats"])
        output_length = len(output_structure["basic_stats"])
        
        if input_length > 1000:  # 长文本
            compression_ratio = output_length / input_length if input_length > 0 else 0
            if compression_ratio < 0.1:
                recommendations.append("输出过于简短，可能遗漏重要信息，建议适当增加内容")
            elif compression_ratio > 0.8:
                recommendations.append("输出过于冗长，建议提高信息提取和总结能力")
        
        return recommendations[:8]  # 最多返回8条建议
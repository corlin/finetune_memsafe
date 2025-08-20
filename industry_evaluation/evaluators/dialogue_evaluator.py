"""
多轮对话理解评估器
"""

import re
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict
from industry_evaluation.evaluators.base_evaluator import AbstractEvaluator
from industry_evaluation.models.data_models import Criterion


class DialogueContextAnalyzer:
    """对话上下文分析器"""
    
    def __init__(self):
        """初始化对话上下文分析器"""
        self.context_patterns = self._build_context_patterns()
    
    def analyze_dialogue_context(self, dialogue: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        分析对话上下文
        
        Args:
            dialogue: 对话列表，每个元素包含speaker和content
            
        Returns:
            Dict[str, Any]: 对话上下文分析结果
        """
        analysis = {
            "context_continuity": self._analyze_context_continuity(dialogue),
            "reference_resolution": self._analyze_reference_resolution(dialogue),
            "topic_coherence": self._analyze_topic_coherence(dialogue),
            "turn_transitions": self._analyze_turn_transitions(dialogue),
            "context_memory": self._analyze_context_memory(dialogue)
        }
        
        return analysis
    
    def _analyze_context_continuity(self, dialogue: List[Dict[str, str]]) -> Dict[str, Any]:
        """分析上下文连续性"""
        continuity_scores = []
        
        for i in range(1, len(dialogue)):
            current_turn = dialogue[i]["content"]
            previous_turn = dialogue[i-1]["content"]
            
            # 计算相邻轮次的连续性
            continuity = self._calculate_turn_continuity(previous_turn, current_turn)
            continuity_scores.append(continuity)
        
        return {
            "scores": continuity_scores,
            "average_continuity": sum(continuity_scores) / len(continuity_scores) if continuity_scores else 1.0,
            "continuity_variance": self._calculate_variance(continuity_scores)
        }
    
    def _analyze_reference_resolution(self, dialogue: List[Dict[str, str]]) -> Dict[str, Any]:
        """分析指代消解"""
        references = []
        entities = {}
        
        for i, turn in enumerate(dialogue):
            content = turn["content"]
            
            # 识别指代词
            pronouns = self._extract_pronouns(content)
            
            # 识别实体
            turn_entities = self._extract_entities(content)
            entities[i] = turn_entities
            
            # 分析指代关系
            for pronoun in pronouns:
                resolution = self._resolve_reference(pronoun, dialogue[:i+1], entities)
                references.append({
                    "turn": i,
                    "pronoun": pronoun,
                    "resolution": resolution,
                    "confidence": resolution.get("confidence", 0.0)
                })
        
        return {
            "total_references": len(references),
            "resolved_references": len([r for r in references if r["confidence"] > 0.5]),
            "resolution_accuracy": len([r for r in references if r["confidence"] > 0.5]) / len(references) if references else 1.0,
            "reference_details": references
        }
    
    def _analyze_topic_coherence(self, dialogue: List[Dict[str, str]]) -> Dict[str, Any]:
        """分析主题连贯性"""
        topics = []
        topic_transitions = []
        
        for i, turn in enumerate(dialogue):
            content = turn["content"]
            turn_topics = self._extract_topics(content)
            topics.append(turn_topics)
            
            if i > 0:
                # 分析主题转换
                transition = self._analyze_topic_transition(topics[i-1], turn_topics)
                topic_transitions.append(transition)
        
        return {
            "topics_per_turn": topics,
            "topic_transitions": topic_transitions,
            "coherence_score": sum(t["coherence"] for t in topic_transitions) / len(topic_transitions) if topic_transitions else 1.0,
            "topic_drift": self._calculate_topic_drift(topics)
        }
    
    def _analyze_turn_transitions(self, dialogue: List[Dict[str, str]]) -> Dict[str, Any]:
        """分析轮次转换"""
        transitions = []
        
        for i in range(1, len(dialogue)):
            prev_turn = dialogue[i-1]
            curr_turn = dialogue[i]
            
            transition = {
                "from_speaker": prev_turn["speaker"],
                "to_speaker": curr_turn["speaker"],
                "transition_type": self._classify_transition_type(prev_turn["content"], curr_turn["content"]),
                "appropriateness": self._evaluate_transition_appropriateness(prev_turn["content"], curr_turn["content"])
            }
            
            transitions.append(transition)
        
        return {
            "transitions": transitions,
            "average_appropriateness": sum(t["appropriateness"] for t in transitions) / len(transitions) if transitions else 1.0,
            "transition_types": [t["transition_type"] for t in transitions]
        }
    
    def _analyze_context_memory(self, dialogue: List[Dict[str, str]]) -> Dict[str, Any]:
        """分析上下文记忆"""
        memory_spans = []
        
        for i in range(len(dialogue)):
            current_content = dialogue[i]["content"]
            
            # 查找对之前内容的引用
            max_span = 0
            for j in range(i):
                if self._has_reference_to_turn(current_content, dialogue[j]["content"]):
                    span = i - j
                    max_span = max(max_span, span)
            
            memory_spans.append(max_span)
        
        return {
            "memory_spans": memory_spans,
            "max_memory_span": max(memory_spans) if memory_spans else 0,
            "average_memory_span": sum(memory_spans) / len(memory_spans) if memory_spans else 0,
            "long_term_memory_usage": len([s for s in memory_spans if s > 3]) / len(memory_spans) if memory_spans else 0
        }
    
    def _calculate_turn_continuity(self, prev_turn: str, curr_turn: str) -> float:
        """计算轮次连续性"""
        # 词汇重叠
        prev_words = set(re.findall(r'\b\w+\b', prev_turn.lower()))
        curr_words = set(re.findall(r'\b\w+\b', curr_turn.lower()))
        
        if not prev_words or not curr_words:
            return 0.0
        
        overlap = len(prev_words.intersection(curr_words))
        union = len(prev_words.union(curr_words))
        
        lexical_continuity = overlap / union if union > 0 else 0.0
        
        # 语义连接词
        connectors = ["所以", "因此", "但是", "然而", "而且", "另外", "同时", "接着"]
        has_connector = any(conn in curr_turn for conn in connectors)
        connector_bonus = 0.2 if has_connector else 0.0
        
        return min(1.0, lexical_continuity + connector_bonus)
    
    def _extract_pronouns(self, text: str) -> List[str]:
        """提取指代词"""
        pronouns = ["它", "他", "她", "这", "那", "这个", "那个", "这些", "那些", "此", "其"]
        found_pronouns = []
        
        for pronoun in pronouns:
            if pronoun in text:
                found_pronouns.append(pronoun)
        
        return found_pronouns
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        # 简单的实体提取
        entities = []
        
        # 技术术语
        tech_terms = re.findall(r'\b(算法|模型|数据|系统|方法|技术)\b', text)
        entities.extend(tech_terms)
        
        # 专有名词（大写开头的词）
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(proper_nouns)
        
        return list(set(entities))
    
    def _resolve_reference(self, pronoun: str, dialogue_history: List[Dict[str, str]], 
                          entities: Dict[int, List[str]]) -> Dict[str, Any]:
        """解析指代关系"""
        resolution = {"antecedent": None, "confidence": 0.0}
        
        # 简单的指代消解：在最近的几轮中查找可能的先行词
        for i in range(len(dialogue_history) - 1, max(-1, len(dialogue_history) - 4), -1):
            turn_entities = entities.get(i, [])
            
            if turn_entities:
                # 选择最近的实体作为先行词
                resolution["antecedent"] = turn_entities[0]
                resolution["confidence"] = 0.7 - (len(dialogue_history) - 1 - i) * 0.1
                break
        
        return resolution
    
    def _extract_topics(self, text: str) -> List[str]:
        """提取主题"""
        # 基于关键词的主题提取
        topic_keywords = {
            "机器学习": ["机器学习", "ML", "算法", "模型", "训练"],
            "深度学习": ["深度学习", "DL", "神经网络", "CNN", "RNN"],
            "数据处理": ["数据", "预处理", "清洗", "特征"],
            "评估": ["评估", "测试", "验证", "性能", "准确率"]
        }
        
        topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _analyze_topic_transition(self, prev_topics: List[str], curr_topics: List[str]) -> Dict[str, Any]:
        """分析主题转换"""
        if not prev_topics and not curr_topics:
            return {"type": "none", "coherence": 1.0}
        
        if not prev_topics:
            return {"type": "introduction", "coherence": 0.8}
        
        if not curr_topics:
            return {"type": "conclusion", "coherence": 0.7}
        
        # 计算主题重叠
        overlap = len(set(prev_topics).intersection(set(curr_topics)))
        total = len(set(prev_topics).union(set(curr_topics)))
        
        coherence = overlap / total if total > 0 else 0.0
        
        if overlap == len(prev_topics) == len(curr_topics):
            transition_type = "continuation"
        elif overlap > 0:
            transition_type = "partial_shift"
        else:
            transition_type = "complete_shift"
        
        return {"type": transition_type, "coherence": coherence}
    
    def _classify_transition_type(self, prev_content: str, curr_content: str) -> str:
        """分类转换类型"""
        # 问答模式
        if "？" in prev_content or "?" in prev_content:
            return "question_answer"
        
        # 解释模式
        if any(word in curr_content for word in ["因为", "由于", "原因是"]):
            return "explanation"
        
        # 补充模式
        if any(word in curr_content for word in ["另外", "而且", "此外"]):
            return "addition"
        
        # 转折模式
        if any(word in curr_content for word in ["但是", "然而", "不过"]):
            return "contrast"
        
        return "continuation"
    
    def _evaluate_transition_appropriateness(self, prev_content: str, curr_content: str) -> float:
        """评估转换适当性"""
        appropriateness = 0.5  # 基础分数
        
        # 如果前一轮是问题，当前轮应该是回答
        if ("？" in prev_content or "?" in prev_content):
            if not ("？" in curr_content or "?" in curr_content):
                appropriateness += 0.3  # 问答匹配
        
        # 检查逻辑连接
        connectors = ["所以", "因此", "但是", "然而", "而且", "另外"]
        if any(conn in curr_content for conn in connectors):
            appropriateness += 0.2
        
        return min(1.0, appropriateness)
    
    def _calculate_topic_drift(self, topics_sequence: List[List[str]]) -> float:
        """计算主题漂移"""
        if len(topics_sequence) <= 1:
            return 0.0
        
        drift_scores = []
        initial_topics = set(topics_sequence[0]) if topics_sequence[0] else set()
        
        for topics in topics_sequence[1:]:
            current_topics = set(topics) if topics else set()
            
            if not initial_topics and not current_topics:
                drift = 0.0
            elif not initial_topics or not current_topics:
                drift = 1.0
            else:
                overlap = len(initial_topics.intersection(current_topics))
                drift = 1.0 - (overlap / len(initial_topics))
            
            drift_scores.append(drift)
        
        return sum(drift_scores) / len(drift_scores) if drift_scores else 0.0
    
    def _has_reference_to_turn(self, current_content: str, previous_content: str) -> bool:
        """检查是否引用了之前的轮次"""
        # 简单的引用检测
        prev_words = set(re.findall(r'\b\w+\b', previous_content.lower()))
        curr_words = set(re.findall(r'\b\w+\b', current_content.lower()))
        
        # 移除停用词
        stop_words = {"的", "了", "在", "是", "有", "和", "与", "或", "但", "而"}
        prev_words -= stop_words
        curr_words -= stop_words
        
        if not prev_words:
            return False
        
        overlap = len(prev_words.intersection(curr_words))
        return overlap >= 2  # 至少有2个词重叠
    
    def _calculate_variance(self, values: List[float]) -> float:
        """计算方差"""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _build_context_patterns(self) -> Dict[str, List[str]]:
        """构建上下文模式"""
        return {
            "question_patterns": [r"什么是", r"如何", r"为什么", r"怎么样"],
            "answer_patterns": [r"是", r"可以", r"应该", r"因为"],
            "continuation_patterns": [r"另外", r"而且", r"此外", r"同时"],
            "contrast_patterns": [r"但是", r"然而", r"不过", r"相反"]
        }


class DialogueEvaluator(AbstractEvaluator):
    """多轮对话理解评估器"""
    
    def __init__(self, name: str = "dialogue_understanding", weight: float = 1.0):
        """
        初始化多轮对话理解评估器
        
        Args:
            name: 评估器名称
            weight: 评估器权重
        """
        super().__init__(name, weight)
        self.context_analyzer = DialogueContextAnalyzer()
    
    def _initialize_criteria(self) -> List[Criterion]:
        """初始化评估标准"""
        return [
            Criterion(
                name="context_continuity",
                description="上下文连续性",
                weight=0.3,
                threshold=0.6,
                evaluation_method="continuity_analysis"
            ),
            Criterion(
                name="reference_resolution",
                description="指代消解能力",
                weight=0.25,
                threshold=0.7,
                evaluation_method="reference_analysis"
            ),
            Criterion(
                name="topic_coherence",
                description="主题连贯性",
                weight=0.25,
                threshold=0.6,
                evaluation_method="coherence_analysis"
            ),
            Criterion(
                name="dialogue_flow",
                description="对话流畅性",
                weight=0.2,
                threshold=0.5,
                evaluation_method="flow_analysis"
            )
        ]
    
    def _calculate_score(self, input_text: str, model_output: str, 
                        expected_output: str, context: Dict[str, Any]) -> float:
        """计算多轮对话理解评估分数"""
        # 解析对话格式
        dialogue = self._parse_dialogue(input_text, model_output)
        
        if len(dialogue) < 2:
            return 0.3  # 不是多轮对话，给较低分数
        
        # 分析对话上下文
        context_analysis = self.context_analyzer.analyze_dialogue_context(dialogue)
        
        # 计算各项分数
        continuity_score = context_analysis["context_continuity"]["average_continuity"]
        reference_score = context_analysis["reference_resolution"]["resolution_accuracy"]
        coherence_score = context_analysis["topic_coherence"]["coherence_score"]
        flow_score = context_analysis["turn_transitions"]["average_appropriateness"]
        
        # 加权计算总分
        total_score = (continuity_score * 0.3 + 
                      reference_score * 0.25 + 
                      coherence_score * 0.25 + 
                      flow_score * 0.2)
        
        return total_score
    
    def _parse_dialogue(self, input_text: str, model_output: str) -> List[Dict[str, str]]:
        """解析对话格式"""
        dialogue = []
        
        # 尝试解析标准对话格式
        full_text = input_text + "\n" + model_output
        
        # 模式1: "用户: xxx" 或 "User: xxx"
        pattern1 = r'(用户|User|人类|Human|助手|Assistant|AI)[:：]\s*(.+?)(?=(?:用户|User|人类|Human|助手|Assistant|AI)[:：]|$)'
        matches1 = re.findall(pattern1, full_text, re.DOTALL | re.IGNORECASE)
        
        if matches1:
            for speaker, content in matches1:
                dialogue.append({
                    "speaker": speaker.lower(),
                    "content": content.strip()
                })
        else:
            # 模式2: 按段落分割，交替分配说话者
            paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                speaker = "user" if i % 2 == 0 else "assistant"
                dialogue.append({
                    "speaker": speaker,
                    "content": paragraph
                })
        
        return dialogue
    
    def _generate_details(self, input_text: str, model_output: str, 
                         expected_output: str, context: Dict[str, Any], 
                         score: float) -> Dict[str, Any]:
        """生成详细评估信息"""
        dialogue = self._parse_dialogue(input_text, model_output)
        context_analysis = self.context_analyzer.analyze_dialogue_context(dialogue)
        
        # 计算各项分数
        continuity_score = context_analysis["context_continuity"]["average_continuity"]
        reference_score = context_analysis["reference_resolution"]["resolution_accuracy"]
        coherence_score = context_analysis["topic_coherence"]["coherence_score"]
        flow_score = context_analysis["turn_transitions"]["average_appropriateness"]
        
        return {
            "evaluator": self.name,
            "score": score,
            "dialogue": dialogue,
            "dialogue_length": len(dialogue),
            "context_analysis": context_analysis,
            "evaluation_scores": {
                "context_continuity": continuity_score,
                "reference_resolution": reference_score,
                "topic_coherence": coherence_score,
                "dialogue_flow": flow_score
            },
            "dialogue_statistics": {
                "total_turns": len(dialogue),
                "average_turn_length": sum(len(turn["content"]) for turn in dialogue) / len(dialogue) if dialogue else 0,
                "speaker_distribution": self._analyze_speaker_distribution(dialogue),
                "turn_types": self._analyze_turn_types(dialogue)
            },
            "quality_indicators": {
                "memory_span": context_analysis["context_memory"]["max_memory_span"],
                "topic_drift": context_analysis["topic_coherence"]["topic_drift"],
                "reference_density": len(context_analysis["reference_resolution"]["reference_details"]) / len(dialogue) if dialogue else 0
            },
            "recommendations": self._generate_dialogue_recommendations(context_analysis, dialogue)
        }
    
    def _analyze_speaker_distribution(self, dialogue: List[Dict[str, str]]) -> Dict[str, int]:
        """分析说话者分布"""
        distribution = defaultdict(int)
        for turn in dialogue:
            distribution[turn["speaker"]] += 1
        return dict(distribution)
    
    def _analyze_turn_types(self, dialogue: List[Dict[str, str]]) -> Dict[str, int]:
        """分析轮次类型"""
        turn_types = defaultdict(int)
        
        for turn in dialogue:
            content = turn["content"]
            
            if "？" in content or "?" in content:
                turn_types["question"] += 1
            elif any(word in content for word in ["因为", "由于", "原因"]):
                turn_types["explanation"] += 1
            elif any(word in content for word in ["但是", "然而", "不过"]):
                turn_types["contrast"] += 1
            else:
                turn_types["statement"] += 1
        
        return dict(turn_types)
    
    def _generate_dialogue_recommendations(self, context_analysis: Dict[str, Any], 
                                         dialogue: List[Dict[str, str]]) -> List[str]:
        """生成对话改进建议"""
        recommendations = []
        
        # 基于上下文连续性的建议
        continuity = context_analysis["context_continuity"]
        if continuity["average_continuity"] < 0.5:
            recommendations.append("建议增强上下文连续性，在回应中更多地引用前面的内容")
        
        if continuity["continuity_variance"] > 0.3:
            recommendations.append("建议保持更一致的上下文连接，避免连续性的大幅波动")
        
        # 基于指代消解的建议
        reference = context_analysis["reference_resolution"]
        if reference["resolution_accuracy"] < 0.6:
            recommendations.append("建议改善指代消解能力，明确指代词的先行词")
        
        if reference["total_references"] == 0 and len(dialogue) > 3:
            recommendations.append("在多轮对话中建议适当使用指代词，使对话更自然")
        
        # 基于主题连贯性的建议
        topic = context_analysis["topic_coherence"]
        if topic["coherence_score"] < 0.5:
            recommendations.append("建议保持主题连贯性，避免频繁的主题跳转")
        
        if topic["topic_drift"] > 0.7:
            recommendations.append("主题漂移较严重，建议在对话中保持对初始主题的关注")
        
        # 基于对话流畅性的建议
        transitions = context_analysis["turn_transitions"]
        if transitions["average_appropriateness"] < 0.5:
            recommendations.append("建议改善轮次转换的适当性，使对话更流畅")
        
        # 基于记忆跨度的建议
        memory = context_analysis["context_memory"]
        if memory["max_memory_span"] < 2 and len(dialogue) > 4:
            recommendations.append("建议增强长期记忆能力，能够引用更早的对话内容")
        
        if memory["long_term_memory_usage"] < 0.2 and len(dialogue) > 5:
            recommendations.append("在长对话中建议更多地利用长期上下文信息")
        
        return recommendations[:6]  # 最多返回6条建议
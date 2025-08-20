"""
改进建议生成器
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from industry_evaluation.models.data_models import EvaluationResult, SampleResult, ErrorAnalysis
from industry_evaluation.analysis.error_analyzer import ErrorInstance, ErrorSeverity


class SuggestionPriority(Enum):
    """建议优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SuggestionCategory(Enum):
    """建议类别"""
    DATA_QUALITY = "data_quality"
    MODEL_TRAINING = "model_training"
    EVALUATION_PROCESS = "evaluation_process"
    KNOWLEDGE_BASE = "knowledge_base"
    TERMINOLOGY = "terminology"
    REASONING = "reasoning"
    CONTEXT_UNDERSTANDING = "context_understanding"
    OUTPUT_FORMAT = "output_format"


@dataclass
class ImprovementSuggestion:
    """改进建议"""
    suggestion_id: str
    title: str
    description: str
    category: SuggestionCategory
    priority: SuggestionPriority
    impact_score: float  # 预期影响分数 (0-1)
    effort_score: float  # 实施难度分数 (0-1)
    evidence: List[str] = field(default_factory=list)  # 支持证据
    action_items: List[str] = field(default_factory=list)  # 具体行动项
    expected_improvement: str = ""  # 预期改进效果
    resources_needed: List[str] = field(default_factory=list)  # 所需资源
    timeline: str = ""  # 实施时间线
    
    def get_roi_score(self) -> float:
        """计算投资回报率分数"""
        if self.effort_score == 0:
            return self.impact_score
        return self.impact_score / self.effort_score

c
lass SuggestionTemplateManager:
    """建议模板管理器"""
    
    def __init__(self):
        """初始化建议模板管理器"""
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """初始化建议模板"""
        return {
            "knowledge_accuracy": {
                "title": "提升专业知识准确性",
                "category": SuggestionCategory.KNOWLEDGE_BASE,
                "base_description": "模型在专业知识方面存在准确性问题，需要加强知识库建设和训练",
                "action_templates": [
                    "补充和更新行业专业知识库",
                    "增加高质量的专业领域训练数据",
                    "建立知识验证和校对机制",
                    "引入领域专家进行知识审核"
                ],
                "impact_range": (0.7, 0.9),
                "effort_range": (0.6, 0.8)
            },
            "terminology_consistency": {
                "title": "改善术语使用一致性",
                "category": SuggestionCategory.TERMINOLOGY,
                "base_description": "模型在术语使用方面存在不一致或误用问题",
                "action_templates": [
                    "建立标准化术语词典",
                    "实施术语一致性检查机制",
                    "增加术语使用规范的训练样本",
                    "开发术语自动校正工具"
                ],
                "impact_range": (0.5, 0.7),
                "effort_range": (0.3, 0.5)
            },
            "reasoning_logic": {
                "title": "增强逻辑推理能力",
                "category": SuggestionCategory.REASONING,
                "base_description": "模型在逻辑推理和因果关系理解方面需要改进",
                "action_templates": [
                    "增加逻辑推理训练数据",
                    "引入推理链条验证机制",
                    "加强因果关系识别训练",
                    "实施多步推理能力测试"
                ],
                "impact_range": (0.6, 0.8),
                "effort_range": (0.7, 0.9)
            },
            "context_understanding": {
                "title": "提升上下文理解能力",
                "category": SuggestionCategory.CONTEXT_UNDERSTANDING,
                "base_description": "模型在理解和保持上下文信息方面存在不足",
                "action_templates": [
                    "增加长文本理解训练",
                    "改进上下文信息提取机制",
                    "加强多轮对话理解能力",
                    "优化注意力机制设计"
                ],
                "impact_range": (0.6, 0.8),
                "effort_range": (0.5, 0.7)
            },
            "output_format": {
                "title": "规范输出格式",
                "category": SuggestionCategory.OUTPUT_FORMAT,
                "base_description": "模型输出格式不规范，影响可读性和使用体验",
                "action_templates": [
                    "建立输出格式规范标准",
                    "实施格式自动检查和修正",
                    "增加格式规范的训练样本",
                    "开发输出后处理模块"
                ],
                "impact_range": (0.3, 0.5),
                "effort_range": (0.2, 0.4)
            },
            "data_quality": {
                "title": "提升训练数据质量",
                "category": SuggestionCategory.DATA_QUALITY,
                "base_description": "训练数据质量问题影响模型性能",
                "action_templates": [
                    "实施数据质量评估和清洗",
                    "增加高质量标注数据",
                    "建立数据质量监控机制",
                    "优化数据采集和预处理流程"
                ],
                "impact_range": (0.8, 0.9),
                "effort_range": (0.6, 0.8)
            },
            "model_training": {
                "title": "优化模型训练策略",
                "category": SuggestionCategory.MODEL_TRAINING,
                "base_description": "模型训练策略需要优化以提升性能",
                "action_templates": [
                    "调整训练超参数",
                    "实施渐进式训练策略",
                    "引入正则化技术",
                    "优化损失函数设计"
                ],
                "impact_range": (0.6, 0.8),
                "effort_range": (0.4, 0.6)
            },
            "evaluation_process": {
                "title": "完善评估流程",
                "category": SuggestionCategory.EVALUATION_PROCESS,
                "base_description": "评估流程需要改进以更准确地反映模型性能",
                "action_templates": [
                    "扩展评估指标体系",
                    "增加多维度评估方法",
                    "建立持续评估机制",
                    "引入人工评估验证"
                ],
                "impact_range": (0.4, 0.6),
                "effort_range": (0.3, 0.5)
            }
        }
    
    def get_template(self, template_key: str) -> Optional[Dict[str, Any]]:
        """获取建议模板"""
        return self.templates.get(template_key)
    
    def get_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模板"""
        return self.templates.copy()


class ImprovementSuggestionGenerator:
    """改进建议生成器"""
    
    def __init__(self):
        """初始化改进建议生成器"""
        self.template_manager = SuggestionTemplateManager()
        self.suggestion_rules = self._initialize_suggestion_rules()
    
    def generate_suggestions(self, evaluation_result: EvaluationResult,
                           error_analysis: Optional[ErrorAnalysis] = None) -> List[ImprovementSuggestion]:
        """
        生成改进建议
        
        Args:
            evaluation_result: 评估结果
            error_analysis: 错误分析结果
            
        Returns:
            List[ImprovementSuggestion]: 改进建议列表
        """
        suggestions = []
        
        # 基于评估分数生成建议
        score_based_suggestions = self._generate_score_based_suggestions(evaluation_result)
        suggestions.extend(score_based_suggestions)
        
        # 基于错误分析生成建议
        if error_analysis:
            error_based_suggestions = self._generate_error_based_suggestions(error_analysis)
            suggestions.extend(error_based_suggestions)
        
        # 基于样本结果生成建议
        sample_based_suggestions = self._generate_sample_based_suggestions(evaluation_result.detailed_results)
        suggestions.extend(sample_based_suggestions)
        
        # 去重和排序
        suggestions = self._deduplicate_suggestions(suggestions)
        suggestions = self._prioritize_suggestions(suggestions, evaluation_result)
        
        return suggestions[:10]  # 返回前10个最重要的建议
    
    def _generate_score_based_suggestions(self, evaluation_result: EvaluationResult) -> List[ImprovementSuggestion]:
        """基于评估分数生成建议"""
        suggestions = []
        
        # 分析各维度分数
        for dimension, score in evaluation_result.dimension_scores.items():
            if score < 0.6:  # 分数较低的维度需要改进
                suggestion = self._create_dimension_improvement_suggestion(dimension, score)
                if suggestion:
                    suggestions.append(suggestion)
        
        # 分析整体分数
        if evaluation_result.overall_score < 0.7:
            suggestion = self._create_overall_improvement_suggestion(evaluation_result)
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_error_based_suggestions(self, error_analysis: ErrorAnalysis) -> List[ImprovementSuggestion]:
        """基于错误分析生成建议"""
        suggestions = []
        
        # 基于错误类型生成建议
        for error_type, count in error_analysis.error_distribution.items():
            if count > 0:
                suggestion = self._create_error_type_suggestion(error_type, count, error_analysis)
                if suggestion:
                    suggestions.append(suggestion)
        
        # 基于严重程度生成建议
        critical_errors = [error_type for error_type, severity in error_analysis.severity_levels.items()
                          if severity == "critical"]
        
        if critical_errors:
            suggestion = self._create_critical_error_suggestion(critical_errors)
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_sample_based_suggestions(self, sample_results: List[SampleResult]) -> List[ImprovementSuggestion]:
        """基于样本结果生成建议"""
        suggestions = []
        
        if not sample_results:
            return suggestions
        
        # 分析样本质量
        quality_issues = self._analyze_sample_quality(sample_results)
        for issue_type, details in quality_issues.items():
            suggestion = self._create_quality_improvement_suggestion(issue_type, details)
            if suggestion:
                suggestions.append(suggestion)
        
        # 分析性能模式
        performance_patterns = self._analyze_performance_patterns(sample_results)
        for pattern_type, pattern_data in performance_patterns.items():
            suggestion = self._create_pattern_based_suggestion(pattern_type, pattern_data)
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    def _create_dimension_improvement_suggestion(self, dimension: str, score: float) -> Optional[ImprovementSuggestion]:
        """创建维度改进建议"""
        # 映射维度到模板
        dimension_template_map = {
            "knowledge": "knowledge_accuracy",
            "terminology": "terminology_consistency", 
            "reasoning": "reasoning_logic",
            "context": "context_understanding",
            "format": "output_format"
        }
        
        template_key = dimension_template_map.get(dimension)
        if not template_key:
            return None
        
        template = self.template_manager.get_template(template_key)
        if not template:
            return None
        
        # 根据分数确定优先级
        if score < 0.3:
            priority = SuggestionPriority.CRITICAL
        elif score < 0.5:
            priority = SuggestionPriority.HIGH
        else:
            priority = SuggestionPriority.MEDIUM
        
        # 计算影响和努力分数
        impact_min, impact_max = template["impact_range"]
        effort_min, effort_max = template["effort_range"]
        
        # 分数越低，影响越大，努力也可能越大
        impact_score = impact_min + (impact_max - impact_min) * (1 - score)
        effort_score = effort_min + (effort_max - effort_min) * (1 - score)
        
        return ImprovementSuggestion(
            suggestion_id=f"dim_{dimension}_{int(score*100)}",
            title=template["title"],
            description=f"{template['base_description']}。当前{dimension}维度分数为{score:.2f}，需要重点改进。",
            category=template["category"],
            priority=priority,
            impact_score=impact_score,
            effort_score=effort_score,
            evidence=[f"{dimension}维度分数偏低: {score:.2f}"],
            action_items=template["action_templates"][:3],  # 取前3个行动项
            expected_improvement=f"预期可将{dimension}维度分数提升至{min(1.0, score + 0.2):.2f}以上",
            timeline="2-4周" if priority.value >= 3 else "4-8周"
        )
    
    def _create_overall_improvement_suggestion(self, evaluation_result: EvaluationResult) -> ImprovementSuggestion:
        """创建整体改进建议"""
        overall_score = evaluation_result.overall_score
        
        return ImprovementSuggestion(
            suggestion_id=f"overall_{int(overall_score*100)}",
            title="提升整体评估性能",
            description=f"模型整体评估分数为{overall_score:.2f}，需要综合性改进措施。",
            category=SuggestionCategory.MODEL_TRAINING,
            priority=SuggestionPriority.HIGH if overall_score < 0.5 else SuggestionPriority.MEDIUM,
            impact_score=0.8,
            effort_score=0.7,
            evidence=[f"整体评估分数偏低: {overall_score:.2f}"],
            action_items=[
                "全面评估各维度表现",
                "制定综合改进计划",
                "优先处理最薄弱环节",
                "建立持续监控机制"
            ],
            expected_improvement=f"预期可将整体分数提升至{min(1.0, overall_score + 0.15):.2f}以上",
            resources_needed=["技术团队", "领域专家", "高质量数据"],
            timeline="6-12周"
        )
    
    def _create_error_type_suggestion(self, error_type: str, count: int, 
                                    error_analysis: ErrorAnalysis) -> Optional[ImprovementSuggestion]:
        """创建错误类型建议"""
        # 映射错误类型到模板
        error_template_map = {
            "knowledge_error": "knowledge_accuracy",
            "terminology_error": "terminology_consistency",
            "reasoning_error": "reasoning_logic", 
            "context_error": "context_understanding",
            "format_error": "output_format"
        }
        
        template_key = error_template_map.get(error_type)
        if not template_key:
            return None
        
        template = self.template_manager.get_template(template_key)
        if not template:
            return None
        
        # 根据错误数量和严重程度确定优先级
        severity = error_analysis.severity_levels.get(error_type, "medium")
        if severity == "critical" or count >= 5:
            priority = SuggestionPriority.CRITICAL
        elif severity == "high" or count >= 3:
            priority = SuggestionPriority.HIGH
        else:
            priority = SuggestionPriority.MEDIUM
        
        return ImprovementSuggestion(
            suggestion_id=f"error_{error_type}_{count}",
            title=f"解决{template['title']}问题",
            description=f"检测到{count}个{error_type}类型错误，严重程度为{severity}。{template['base_description']}",
            category=template["category"],
            priority=priority,
            impact_score=0.7 + min(0.2, count * 0.05),  # 错误越多影响越大
            effort_score=0.5 + min(0.3, count * 0.03),  # 错误越多努力越大
            evidence=[f"发现{count}个{error_type}错误", f"错误严重程度: {severity}"],
            action_items=template["action_templates"],
            expected_improvement=f"预期可减少{error_type}错误50-80%",
            timeline="3-6周" if priority.value >= 3 else "6-10周"
        )
    
    def _create_critical_error_suggestion(self, critical_errors: List[str]) -> ImprovementSuggestion:
        """创建关键错误建议"""
        return ImprovementSuggestion(
            suggestion_id=f"critical_errors_{len(critical_errors)}",
            title="紧急处理关键错误",
            description=f"发现{len(critical_errors)}种关键错误类型: {', '.join(critical_errors)}。需要立即采取措施。",
            category=SuggestionCategory.MODEL_TRAINING,
            priority=SuggestionPriority.CRITICAL,
            impact_score=0.9,
            effort_score=0.8,
            evidence=[f"关键错误类型: {', '.join(critical_errors)}"],
            action_items=[
                "立即停止使用当前模型版本",
                "分析关键错误根本原因",
                "制定紧急修复方案",
                "加强质量控制流程"
            ],
            expected_improvement="消除关键错误，显著提升模型可靠性",
            resources_needed=["紧急响应团队", "领域专家", "测试环境"],
            timeline="1-2周"
        )
    
    def _analyze_sample_quality(self, sample_results: List[SampleResult]) -> Dict[str, Any]:
        """分析样本质量"""
        quality_issues = {}
        
        # 分析处理时间
        processing_times = [result.processing_time for result in sample_results if result.processing_time > 0]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            if avg_time > 5.0:  # 平均处理时间超过5秒
                quality_issues["slow_processing"] = {
                    "avg_time": avg_time,
                    "affected_samples": len([t for t in processing_times if t > 5.0])
                }
        
        # 分析输出长度分布
        output_lengths = [len(result.model_output) for result in sample_results]
        if output_lengths:
            avg_length = sum(output_lengths) / len(output_lengths)
            short_outputs = len([l for l in output_lengths if l < 20])
            if short_outputs > len(sample_results) * 0.3:  # 超过30%的输出过短
                quality_issues["short_outputs"] = {
                    "avg_length": avg_length,
                    "short_count": short_outputs,
                    "percentage": short_outputs / len(sample_results) * 100
                }
        
        return quality_issues
    
    def _analyze_performance_patterns(self, sample_results: List[SampleResult]) -> Dict[str, Any]:
        """分析性能模式"""
        patterns = {}
        
        # 分析分数分布模式
        if sample_results:
            # 假设使用第一个维度分数作为代表
            scores = []
            for result in sample_results:
                if result.dimension_scores:
                    first_score = next(iter(result.dimension_scores.values()))
                    scores.append(first_score)
            
            if scores:
                low_scores = len([s for s in scores if s < 0.5])
                if low_scores > len(scores) * 0.4:  # 超过40%的样本分数较低
                    patterns["low_score_pattern"] = {
                        "low_score_count": low_scores,
                        "percentage": low_scores / len(scores) * 100,
                        "avg_score": sum(scores) / len(scores)
                    }
        
        return patterns
    
    def _create_quality_improvement_suggestion(self, issue_type: str, details: Dict[str, Any]) -> Optional[ImprovementSuggestion]:
        """创建质量改进建议"""
        if issue_type == "slow_processing":
            return ImprovementSuggestion(
                suggestion_id=f"quality_{issue_type}",
                title="优化处理性能",
                description=f"平均处理时间为{details['avg_time']:.2f}秒，需要优化性能。",
                category=SuggestionCategory.MODEL_TRAINING,
                priority=SuggestionPriority.MEDIUM,
                impact_score=0.6,
                effort_score=0.5,
                evidence=[f"平均处理时间: {details['avg_time']:.2f}秒"],
                action_items=[
                    "分析性能瓶颈",
                    "优化模型推理速度",
                    "考虑模型压缩技术",
                    "优化硬件配置"
                ],
                expected_improvement="处理速度提升30-50%",
                timeline="2-4周"
            )
        
        elif issue_type == "short_outputs":
            return ImprovementSuggestion(
                suggestion_id=f"quality_{issue_type}",
                title="改善输出完整性",
                description=f"{details['percentage']:.1f}%的输出过短，平均长度仅{details['avg_length']:.1f}字符。",
                category=SuggestionCategory.OUTPUT_FORMAT,
                priority=SuggestionPriority.MEDIUM,
                impact_score=0.5,
                effort_score=0.4,
                evidence=[f"短输出比例: {details['percentage']:.1f}%"],
                action_items=[
                    "分析输出长度要求",
                    "调整生成策略",
                    "增加内容丰富度训练",
                    "设置最小输出长度限制"
                ],
                expected_improvement="输出完整性提升40-60%",
                timeline="2-3周"
            )
        
        return None
    
    def _create_pattern_based_suggestion(self, pattern_type: str, pattern_data: Dict[str, Any]) -> Optional[ImprovementSuggestion]:
        """创建基于模式的建议"""
        if pattern_type == "low_score_pattern":
            return ImprovementSuggestion(
                suggestion_id=f"pattern_{pattern_type}",
                title="解决普遍性能问题",
                description=f"{pattern_data['percentage']:.1f}%的样本分数较低，平均分数仅{pattern_data['avg_score']:.2f}。",
                category=SuggestionCategory.MODEL_TRAINING,
                priority=SuggestionPriority.HIGH,
                impact_score=0.8,
                effort_score=0.7,
                evidence=[f"低分样本比例: {pattern_data['percentage']:.1f}%"],
                action_items=[
                    "全面检查训练数据质量",
                    "重新评估模型架构",
                    "增加训练数据多样性",
                    "调整训练策略"
                ],
                expected_improvement="整体性能提升20-40%",
                resources_needed=["数据科学团队", "计算资源", "高质量数据"],
                timeline="4-8周"
            )
        
        return None
    
    def _deduplicate_suggestions(self, suggestions: List[ImprovementSuggestion]) -> List[ImprovementSuggestion]:
        """去重建议"""
        seen_titles = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            if suggestion.title not in seen_titles:
                seen_titles.add(suggestion.title)
                unique_suggestions.append(suggestion)
            else:
                # 如果标题重复，合并证据和行动项
                existing = next(s for s in unique_suggestions if s.title == suggestion.title)
                existing.evidence.extend(suggestion.evidence)
                existing.action_items.extend(suggestion.action_items)
                # 去重行动项
                existing.action_items = list(dict.fromkeys(existing.action_items))
                existing.evidence = list(dict.fromkeys(existing.evidence))
        
        return unique_suggestions
    
    def _prioritize_suggestions(self, suggestions: List[ImprovementSuggestion], 
                              evaluation_result: EvaluationResult) -> List[ImprovementSuggestion]:
        """优先级排序建议"""
        def priority_score(suggestion: ImprovementSuggestion) -> float:
            # 综合考虑优先级、ROI和整体分数
            priority_weight = suggestion.priority.value * 0.4
            roi_weight = suggestion.get_roi_score() * 0.4
            urgency_weight = (1 - evaluation_result.overall_score) * 0.2
            
            return priority_weight + roi_weight + urgency_weight
        
        suggestions.sort(key=priority_score, reverse=True)
        return suggestions
    
    def _initialize_suggestion_rules(self) -> Dict[str, Any]:
        """初始化建议规则"""
        return {
            "score_thresholds": {
                "critical": 0.3,
                "high": 0.5,
                "medium": 0.7,
                "low": 0.8
            },
            "error_count_thresholds": {
                "critical": 5,
                "high": 3,
                "medium": 1
            },
            "impact_factors": {
                "knowledge_error": 0.9,
                "reasoning_error": 0.8,
                "terminology_error": 0.6,
                "context_error": 0.7,
                "format_error": 0.4
            }
        }
    
    def generate_action_plan(self, suggestions: List[ImprovementSuggestion]) -> Dict[str, Any]:
        """
        生成行动计划
        
        Args:
            suggestions: 改进建议列表
            
        Returns:
            Dict[str, Any]: 行动计划
        """
        # 按优先级分组
        priority_groups = defaultdict(list)
        for suggestion in suggestions:
            priority_groups[suggestion.priority.name].append(suggestion)
        
        # 按类别分组
        category_groups = defaultdict(list)
        for suggestion in suggestions:
            category_groups[suggestion.category.value].append(suggestion)
        
        # 计算总体指标
        total_impact = sum(s.impact_score for s in suggestions)
        total_effort = sum(s.effort_score for s in suggestions)
        avg_roi = sum(s.get_roi_score() for s in suggestions) / len(suggestions) if suggestions else 0
        
        # 生成时间线
        timeline = self._generate_timeline(suggestions)
        
        return {
            "summary": {
                "total_suggestions": len(suggestions),
                "priority_distribution": {k: len(v) for k, v in priority_groups.items()},
                "category_distribution": {k: len(v) for k, v in category_groups.items()},
                "total_impact_score": total_impact,
                "total_effort_score": total_effort,
                "average_roi": avg_roi
            },
            "priority_groups": dict(priority_groups),
            "category_groups": dict(category_groups),
            "timeline": timeline,
            "quick_wins": [s for s in suggestions if s.get_roi_score() > 1.5][:3],
            "high_impact": [s for s in suggestions if s.impact_score > 0.8][:3],
            "resource_requirements": self._aggregate_resources(suggestions)
        }
    
    def _generate_timeline(self, suggestions: List[ImprovementSuggestion]) -> Dict[str, List[str]]:
        """生成时间线"""
        timeline = {
            "immediate": [],  # 1-2周
            "short_term": [],  # 2-4周
            "medium_term": [],  # 4-8周
            "long_term": []  # 8周以上
        }
        
        for suggestion in suggestions:
            if suggestion.priority == SuggestionPriority.CRITICAL:
                timeline["immediate"].append(suggestion.title)
            elif "1-2周" in suggestion.timeline:
                timeline["immediate"].append(suggestion.title)
            elif any(term in suggestion.timeline for term in ["2-3周", "2-4周"]):
                timeline["short_term"].append(suggestion.title)
            elif any(term in suggestion.timeline for term in ["4-6周", "4-8周", "6-8周"]):
                timeline["medium_term"].append(suggestion.title)
            else:
                timeline["long_term"].append(suggestion.title)
        
        return timeline
    
    def _aggregate_resources(self, suggestions: List[ImprovementSuggestion]) -> Dict[str, int]:
        """聚合资源需求"""
        resource_counts = Counter()
        
        for suggestion in suggestions:
            for resource in suggestion.resources_needed:
                resource_counts[resource] += 1
        
        return dict(resource_counts)
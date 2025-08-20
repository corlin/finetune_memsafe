"""
错误分析模块
"""

import re
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
from industry_evaluation.models.data_models import SampleResult, ErrorType, ErrorAnalysis


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorPattern:
    """错误模式"""
    pattern_id: str
    pattern_type: str
    description: str
    regex_pattern: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    frequency: int = 0
    examples: List[str] = field(default_factory=list)


@dataclass
class ErrorInstance:
    """错误实例"""
    sample_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    location: Optional[str] = None
    context: str = ""
    suggested_fix: str = ""
    confidence: float = 1.0


class ErrorClassifier:
    """错误分类器"""
    
    def __init__(self):
        """初始化错误分类器"""
        self.error_patterns = self._initialize_error_patterns()
        self.classification_rules = self._initialize_classification_rules()
    
    def classify_errors(self, sample_results: List[SampleResult]) -> Dict[str, List[ErrorInstance]]:
        """
        分类错误
        
        Args:
            sample_results: 样本结果列表
            
        Returns:
            Dict[str, List[ErrorInstance]]: 按类型分类的错误实例
        """
        classified_errors = defaultdict(list)
        
        for sample in sample_results:
            if sample.error_types:
                for error_type in sample.error_types:
                    error_instances = self._analyze_sample_errors(sample, error_type)
                    classified_errors[error_type].extend(error_instances)
        
        return dict(classified_errors)
    
    def _analyze_sample_errors(self, sample: SampleResult, error_type: str) -> List[ErrorInstance]:
        """分析单个样本的错误"""
        error_instances = []
        
        # 基于错误类型进行具体分析
        if error_type == "knowledge_error":
            instances = self._analyze_knowledge_errors(sample)
        elif error_type == "terminology_error":
            instances = self._analyze_terminology_errors(sample)
        elif error_type == "reasoning_error":
            instances = self._analyze_reasoning_errors(sample)
        elif error_type == "context_error":
            instances = self._analyze_context_errors(sample)
        elif error_type == "format_error":
            instances = self._analyze_format_errors(sample)
        else:
            instances = self._analyze_generic_errors(sample, error_type)
        
        error_instances.extend(instances)
        return error_instances
    
    def _analyze_knowledge_errors(self, sample: SampleResult) -> List[ErrorInstance]:
        """分析知识错误"""
        errors = []
        
        # 检查事实性错误
        fact_errors = self._detect_factual_errors(sample.model_output, sample.expected_output)
        for error in fact_errors:
            errors.append(ErrorInstance(
                sample_id=sample.sample_id,
                error_type="knowledge_error",
                error_message=error["message"],
                severity=ErrorSeverity.HIGH,
                context=error["context"],
                suggested_fix=error["suggestion"]
            ))
        
        # 检查概念混淆
        concept_errors = self._detect_concept_confusion(sample.model_output, sample.expected_output)
        for error in concept_errors:
            errors.append(ErrorInstance(
                sample_id=sample.sample_id,
                error_type="knowledge_error",
                error_message=error["message"],
                severity=ErrorSeverity.MEDIUM,
                context=error["context"],
                suggested_fix=error["suggestion"]
            ))
        
        return errors
    
    def _analyze_terminology_errors(self, sample: SampleResult) -> List[ErrorInstance]:
        """分析术语错误"""
        errors = []
        
        # 检查术语误用
        misuse_errors = self._detect_terminology_misuse(sample.model_output)
        for error in misuse_errors:
            errors.append(ErrorInstance(
                sample_id=sample.sample_id,
                error_type="terminology_error",
                error_message=error["message"],
                severity=ErrorSeverity.MEDIUM,
                context=error["context"],
                suggested_fix=error["suggestion"]
            ))
        
        # 检查术语不一致
        inconsistency_errors = self._detect_terminology_inconsistency(sample.model_output)
        for error in inconsistency_errors:
            errors.append(ErrorInstance(
                sample_id=sample.sample_id,
                error_type="terminology_error",
                error_message=error["message"],
                severity=ErrorSeverity.LOW,
                context=error["context"],
                suggested_fix=error["suggestion"]
            ))
        
        return errors
    
    def _analyze_reasoning_errors(self, sample: SampleResult) -> List[ErrorInstance]:
        """分析推理错误"""
        errors = []
        
        # 检查逻辑错误
        logic_errors = self._detect_logic_errors(sample.model_output, sample.input_text)
        for error in logic_errors:
            errors.append(ErrorInstance(
                sample_id=sample.sample_id,
                error_type="reasoning_error",
                error_message=error["message"],
                severity=ErrorSeverity.HIGH,
                context=error["context"],
                suggested_fix=error["suggestion"]
            ))
        
        # 检查因果关系错误
        causal_errors = self._detect_causal_errors(sample.model_output)
        for error in causal_errors:
            errors.append(ErrorInstance(
                sample_id=sample.sample_id,
                error_type="reasoning_error",
                error_message=error["message"],
                severity=ErrorSeverity.MEDIUM,
                context=error["context"],
                suggested_fix=error["suggestion"]
            ))
        
        return errors
    
    def _analyze_context_errors(self, sample: SampleResult) -> List[ErrorInstance]:
        """分析上下文错误"""
        errors = []
        
        # 检查上下文理解错误
        context_errors = self._detect_context_misunderstanding(
            sample.input_text, sample.model_output, sample.expected_output
        )
        for error in context_errors:
            errors.append(ErrorInstance(
                sample_id=sample.sample_id,
                error_type="context_error",
                error_message=error["message"],
                severity=ErrorSeverity.MEDIUM,
                context=error["context"],
                suggested_fix=error["suggestion"]
            ))
        
        return errors
    
    def _analyze_format_errors(self, sample: SampleResult) -> List[ErrorInstance]:
        """分析格式错误"""
        errors = []
        
        # 检查格式问题
        format_errors = self._detect_format_issues(sample.model_output, sample.expected_output)
        for error in format_errors:
            errors.append(ErrorInstance(
                sample_id=sample.sample_id,
                error_type="format_error",
                error_message=error["message"],
                severity=ErrorSeverity.LOW,
                context=error["context"],
                suggested_fix=error["suggestion"]
            ))
        
        return errors
    
    def _analyze_generic_errors(self, sample: SampleResult, error_type: str) -> List[ErrorInstance]:
        """分析通用错误"""
        return [ErrorInstance(
            sample_id=sample.sample_id,
            error_type=error_type,
            error_message=f"检测到{error_type}类型错误",
            severity=ErrorSeverity.MEDIUM,
            context=sample.model_output[:100] + "..." if len(sample.model_output) > 100 else sample.model_output,
            suggested_fix="需要进一步分析具体错误原因"
        )]
    
    def _detect_factual_errors(self, model_output: str, expected_output: str) -> List[Dict[str, str]]:
        """检测事实性错误"""
        errors = []
        
        # 简单的事实检查模式
        fact_patterns = [
            (r'(\d{4})\s*年', r'年份可能不正确'),
            (r'(\d+(?:\.\d+)?)\s*%', r'百分比数据可能有误'),
            (r'(第\s*[一二三四五六七八九十]+\s*[个位名])', r'序数可能错误')
        ]
        
        for pattern, message in fact_patterns:
            matches = re.finditer(pattern, model_output)
            for match in matches:
                # 检查是否与期望输出中的事实一致
                if expected_output and match.group() not in expected_output:
                    errors.append({
                        "message": f"{message}: {match.group()}",
                        "context": model_output[max(0, match.start()-20):match.end()+20],
                        "suggestion": "请验证事实的准确性"
                    })
        
        return errors
    
    def _detect_concept_confusion(self, model_output: str, expected_output: str) -> List[Dict[str, str]]:
        """检测概念混淆"""
        errors = []
        
        # 常见的概念混淆对
        confusion_pairs = [
            ("机器学习", "深度学习"),
            ("监督学习", "无监督学习"),
            ("分类", "回归"),
            ("准确率", "精确率"),
            ("召回率", "覆盖率")
        ]
        
        for concept1, concept2 in confusion_pairs:
            if concept1 in model_output and concept2 in expected_output:
                # 可能存在概念混淆
                errors.append({
                    "message": f"可能混淆了'{concept1}'和'{concept2}'",
                    "context": f"输出中使用了'{concept1}'，但期望是'{concept2}'",
                    "suggestion": f"请确认是否应该使用'{concept2}'而不是'{concept1}'"
                })
        
        return errors
    
    def _detect_terminology_misuse(self, text: str) -> List[Dict[str, str]]:
        """检测术语误用"""
        errors = []
        
        # 常见的术语误用模式
        misuse_patterns = [
            (r'机器学习算法', r'可能应该说"机器学习方法"或"算法"'),
            (r'AI智能', r'重复表达，应该说"AI"或"人工智能"'),
            (r'数据挖掘技术', r'可能应该说"数据挖掘"或"挖掘技术"')
        ]
        
        for pattern, suggestion in misuse_patterns:
            if re.search(pattern, text):
                errors.append({
                    "message": f"术语使用可能不当: {pattern}",
                    "context": text,
                    "suggestion": suggestion
                })
        
        return errors
    
    def _detect_terminology_inconsistency(self, text: str) -> List[Dict[str, str]]:
        """检测术语不一致"""
        errors = []
        
        # 检查同一概念的不同表述
        synonyms_groups = [
            ["机器学习", "ML", "机学"],
            ["深度学习", "DL", "深学"],
            ["自然语言处理", "NLP", "语言处理"],
            ["人工智能", "AI", "智能"]
        ]
        
        for synonyms in synonyms_groups:
            found_terms = [term for term in synonyms if term in text]
            if len(found_terms) > 1:
                errors.append({
                    "message": f"术语使用不一致: {', '.join(found_terms)}",
                    "context": text,
                    "suggestion": f"建议统一使用'{found_terms[0]}'"
                })
        
        return errors
    
    def _detect_logic_errors(self, model_output: str, input_text: str) -> List[Dict[str, str]]:
        """检测逻辑错误"""
        errors = []
        
        # 检查逻辑矛盾
        contradiction_patterns = [
            (r'(.+?)\s*不是\s*(.+?).*\1\s*是\s*\2', '存在逻辑矛盾'),
            (r'(.+?)\s*可以\s*(.+?).*\1\s*不能\s*\2', '存在逻辑矛盾'),
            (r'(.+?)\s*总是\s*(.+?).*\1\s*从不\s*\2', '存在逻辑矛盾')
        ]
        
        for pattern, message in contradiction_patterns:
            if re.search(pattern, model_output, re.IGNORECASE):
                errors.append({
                    "message": message,
                    "context": model_output,
                    "suggestion": "请检查逻辑一致性"
                })
        
        return errors
    
    def _detect_causal_errors(self, text: str) -> List[Dict[str, str]]:
        """检测因果关系错误"""
        errors = []
        
        # 检查不合理的因果关系
        causal_patterns = [
            r'(.+?)\s*导致\s*(.+?)(?=[。！？.!?]|$)',
            r'(.+?)\s*引起\s*(.+?)(?=[。！？.!?]|$)',
            r'由于\s*(.+?)\s*[，,]\s*(.+?)(?=[。！？.!?]|$)'
        ]
        
        for pattern in causal_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # 简单的合理性检查
                cause = match.group(1).strip()
                effect = match.group(2).strip()
                
                # 检查是否是明显不合理的因果关系
                if self._is_unreasonable_causation(cause, effect):
                    errors.append({
                        "message": f"因果关系可能不合理: {cause} -> {effect}",
                        "context": match.group(0),
                        "suggestion": "请验证因果关系的合理性"
                    })
        
        return errors
    
    def _detect_context_misunderstanding(self, input_text: str, model_output: str, 
                                       expected_output: str) -> List[Dict[str, str]]:
        """检测上下文理解错误"""
        errors = []
        
        # 检查是否回答了错误的问题
        if "什么是" in input_text and "如何" in model_output:
            errors.append({
                "message": "可能误解了问题类型，问的是'什么是'但回答了'如何'",
                "context": f"问题: {input_text[:50]}..., 回答: {model_output[:50]}...",
                "suggestion": "请重新理解问题的意图"
            })
        
        # 检查是否忽略了重要的上下文信息
        if len(input_text) > 100:  # 长输入可能包含重要上下文
            input_keywords = set(re.findall(r'\b\w+\b', input_text.lower()))
            output_keywords = set(re.findall(r'\b\w+\b', model_output.lower()))
            
            # 计算关键词重叠度
            overlap = len(input_keywords.intersection(output_keywords))
            if overlap / len(input_keywords) < 0.3:  # 重叠度过低
                errors.append({
                    "message": "可能忽略了输入中的重要信息",
                    "context": "输入和输出的关键词重叠度较低",
                    "suggestion": "请确保回答涵盖了输入中的关键信息"
                })
        
        return errors
    
    def _detect_format_issues(self, model_output: str, expected_output: str) -> List[Dict[str, str]]:
        """检测格式问题"""
        errors = []
        
        # 检查标点符号问题
        if model_output.count('。') == 0 and len(model_output) > 50:
            errors.append({
                "message": "缺少句号，可能影响可读性",
                "context": model_output,
                "suggestion": "请添加适当的标点符号"
            })
        
        # 检查段落结构
        if '\n\n' in expected_output and '\n\n' not in model_output:
            errors.append({
                "message": "缺少段落分隔，格式与期望不符",
                "context": "输出格式",
                "suggestion": "请按照期望的段落结构组织内容"
            })
        
        return errors
    
    def _is_unreasonable_causation(self, cause: str, effect: str) -> bool:
        """判断因果关系是否不合理"""
        # 简单的不合理因果关系检查
        unreasonable_pairs = [
            ("颜色", "性能"),
            ("名字", "效果"),
            ("时间", "准确率")  # 过于简单的例子
        ]
        
        for unreasonable_cause, unreasonable_effect in unreasonable_pairs:
            if unreasonable_cause in cause.lower() and unreasonable_effect in effect.lower():
                return True
        
        return False
    
    def _initialize_error_patterns(self) -> Dict[str, List[ErrorPattern]]:
        """初始化错误模式"""
        patterns = {
            "knowledge_error": [
                ErrorPattern(
                    pattern_id="fact_error_001",
                    pattern_type="factual_error",
                    description="事实性错误",
                    keywords=["错误", "不正确", "有误"],
                    severity=ErrorSeverity.HIGH
                ),
                ErrorPattern(
                    pattern_id="concept_confusion_001",
                    pattern_type="concept_confusion",
                    description="概念混淆",
                    keywords=["混淆", "搞错", "弄混"],
                    severity=ErrorSeverity.MEDIUM
                )
            ],
            "terminology_error": [
                ErrorPattern(
                    pattern_id="term_misuse_001",
                    pattern_type="terminology_misuse",
                    description="术语误用",
                    keywords=["误用", "用错", "不当"],
                    severity=ErrorSeverity.MEDIUM
                ),
                ErrorPattern(
                    pattern_id="term_inconsistency_001",
                    pattern_type="terminology_inconsistency",
                    description="术语不一致",
                    keywords=["不一致", "不统一", "混用"],
                    severity=ErrorSeverity.LOW
                )
            ],
            "reasoning_error": [
                ErrorPattern(
                    pattern_id="logic_error_001",
                    pattern_type="logic_error",
                    description="逻辑错误",
                    keywords=["矛盾", "不合理", "逻辑错误"],
                    severity=ErrorSeverity.HIGH
                ),
                ErrorPattern(
                    pattern_id="causal_error_001",
                    pattern_type="causal_error",
                    description="因果关系错误",
                    keywords=["因果", "导致", "引起"],
                    severity=ErrorSeverity.MEDIUM
                )
            ]
        }
        
        return patterns
    
    def _initialize_classification_rules(self) -> Dict[str, Any]:
        """初始化分类规则"""
        return {
            "severity_thresholds": {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.4,
                "low": 0.1
            },
            "confidence_weights": {
                "exact_match": 1.0,
                "pattern_match": 0.8,
                "keyword_match": 0.6,
                "heuristic": 0.4
            }
        }


class ErrorStatisticsAnalyzer:
    """错误统计分析器"""
    
    def __init__(self):
        """初始化错误统计分析器"""
        pass
    
    def analyze_error_statistics(self, error_instances: Dict[str, List[ErrorInstance]]) -> Dict[str, Any]:
        """
        分析错误统计
        
        Args:
            error_instances: 错误实例字典
            
        Returns:
            Dict[str, Any]: 错误统计分析结果
        """
        stats = {
            "total_errors": 0,
            "error_type_distribution": {},
            "severity_distribution": {},
            "error_frequency": {},
            "top_error_patterns": [],
            "error_trends": {},
            "correlation_analysis": {}
        }
        
        # 统计总错误数
        all_errors = []
        for error_list in error_instances.values():
            all_errors.extend(error_list)
        
        stats["total_errors"] = len(all_errors)
        
        if not all_errors:
            return stats
        
        # 错误类型分布
        type_counter = Counter(error.error_type for error in all_errors)
        stats["error_type_distribution"] = dict(type_counter)
        
        # 严重程度分布
        severity_counter = Counter(error.severity.value for error in all_errors)
        stats["severity_distribution"] = dict(severity_counter)
        
        # 错误频率分析
        message_counter = Counter(error.error_message for error in all_errors)
        stats["error_frequency"] = dict(message_counter.most_common(10))
        
        # 顶级错误模式
        stats["top_error_patterns"] = self._identify_top_patterns(all_errors)
        
        # 错误趋势分析
        stats["error_trends"] = self._analyze_error_trends(error_instances)
        
        # 相关性分析
        stats["correlation_analysis"] = self._analyze_error_correlations(error_instances)
        
        return stats
    
    def _identify_top_patterns(self, errors: List[ErrorInstance]) -> List[Dict[str, Any]]:
        """识别顶级错误模式"""
        patterns = []
        
        # 按错误类型和严重程度分组
        grouped_errors = defaultdict(list)
        for error in errors:
            key = (error.error_type, error.severity.value)
            grouped_errors[key].append(error)
        
        # 找出最常见的模式
        for (error_type, severity), error_list in grouped_errors.items():
            if len(error_list) >= 2:  # 至少出现2次才算模式
                patterns.append({
                    "error_type": error_type,
                    "severity": severity,
                    "frequency": len(error_list),
                    "percentage": len(error_list) / len(errors) * 100,
                    "examples": [e.error_message for e in error_list[:3]]
                })
        
        # 按频率排序
        patterns.sort(key=lambda x: x["frequency"], reverse=True)
        return patterns[:10]
    
    def _analyze_error_trends(self, error_instances: Dict[str, List[ErrorInstance]]) -> Dict[str, Any]:
        """分析错误趋势"""
        trends = {}
        
        for error_type, errors in error_instances.items():
            if not errors:
                continue
            
            # 按严重程度统计
            severity_counts = Counter(error.severity.value for error in errors)
            
            trends[error_type] = {
                "total_count": len(errors),
                "severity_breakdown": dict(severity_counts),
                "avg_confidence": sum(error.confidence for error in errors) / len(errors),
                "most_common_message": Counter(error.error_message for error in errors).most_common(1)[0][0] if errors else ""
            }
        
        return trends
    
    def _analyze_error_correlations(self, error_instances: Dict[str, List[ErrorInstance]]) -> Dict[str, Any]:
        """分析错误相关性"""
        correlations = {}
        
        # 分析样本级别的错误共现
        sample_errors = defaultdict(set)
        for error_type, errors in error_instances.items():
            for error in errors:
                sample_errors[error.sample_id].add(error_type)
        
        # 计算错误类型间的共现频率
        error_types = list(error_instances.keys())
        for i, type1 in enumerate(error_types):
            for type2 in error_types[i+1:]:
                cooccurrence = sum(1 for errors in sample_errors.values() 
                                 if type1 in errors and type2 in errors)
                
                if cooccurrence > 0:
                    correlations[f"{type1}__{type2}"] = {
                        "cooccurrence_count": cooccurrence,
                        "correlation_strength": cooccurrence / len(sample_errors)
                    }
        
        return correlations


class ErrorAnalysisEngine:
    """错误分析引擎"""
    
    def __init__(self):
        """初始化错误分析引擎"""
        self.classifier = ErrorClassifier()
        self.statistics_analyzer = ErrorStatisticsAnalyzer()
    
    def analyze_errors(self, sample_results: List[SampleResult]) -> ErrorAnalysis:
        """
        分析错误
        
        Args:
            sample_results: 样本结果列表
            
        Returns:
            ErrorAnalysis: 错误分析结果
        """
        # 分类错误
        classified_errors = self.classifier.classify_errors(sample_results)
        
        # 统计分析
        error_stats = self.statistics_analyzer.analyze_error_statistics(classified_errors)
        
        # 生成错误分布
        error_distribution = error_stats.get("error_type_distribution", {})
        
        # 识别常见模式
        common_patterns = []
        for pattern in error_stats.get("top_error_patterns", []):
            pattern_desc = f"{pattern['error_type']}({pattern['severity']}): {pattern['frequency']}次 ({pattern['percentage']:.1f}%)"
            common_patterns.append(pattern_desc)
        
        # 确定严重程度
        severity_levels = {}
        for error_type, count in error_distribution.items():
            if count >= len(sample_results) * 0.5:
                severity_levels[error_type] = "critical"
            elif count >= len(sample_results) * 0.3:
                severity_levels[error_type] = "high"
            elif count >= len(sample_results) * 0.1:
                severity_levels[error_type] = "medium"
            else:
                severity_levels[error_type] = "low"
        
        # 识别改进领域
        improvement_areas = self._identify_improvement_areas(error_stats, classified_errors)
        
        return ErrorAnalysis(
            error_distribution=error_distribution,
            common_patterns=common_patterns,
            severity_levels=severity_levels,
            improvement_areas=improvement_areas
        )
    
    def _identify_improvement_areas(self, error_stats: Dict[str, Any], 
                                  classified_errors: Dict[str, List[ErrorInstance]]) -> List[str]:
        """识别改进领域"""
        improvement_areas = []
        
        # 基于错误频率识别改进领域
        error_distribution = error_stats.get("error_type_distribution", {})
        sorted_errors = sorted(error_distribution.items(), key=lambda x: x[1], reverse=True)
        
        for error_type, count in sorted_errors[:5]:  # 取前5个最频繁的错误类型
            if error_type == "knowledge_error":
                improvement_areas.append("专业知识准确性")
            elif error_type == "terminology_error":
                improvement_areas.append("术语使用规范性")
            elif error_type == "reasoning_error":
                improvement_areas.append("逻辑推理能力")
            elif error_type == "context_error":
                improvement_areas.append("上下文理解能力")
            elif error_type == "format_error":
                improvement_areas.append("输出格式规范性")
            else:
                improvement_areas.append(f"{error_type}相关能力")
        
        # 基于严重程度识别关键改进领域
        severity_distribution = error_stats.get("severity_distribution", {})
        if severity_distribution.get("critical", 0) > 0:
            improvement_areas.insert(0, "关键错误修复")
        if severity_distribution.get("high", 0) > 0:
            improvement_areas.insert(0, "高优先级错误处理")
        
        return improvement_areas[:8]  # 返回最多8个改进领域
    
    def generate_error_report(self, error_analysis: ErrorAnalysis, 
                            sample_results: List[SampleResult]) -> Dict[str, Any]:
        """
        生成错误报告
        
        Args:
            error_analysis: 错误分析结果
            sample_results: 样本结果列表
            
        Returns:
            Dict[str, Any]: 错误报告
        """
        total_samples = len(sample_results)
        samples_with_errors = len([s for s in sample_results if s.error_types])
        
        report = {
            "summary": {
                "total_samples": total_samples,
                "samples_with_errors": samples_with_errors,
                "error_rate": samples_with_errors / total_samples if total_samples > 0 else 0,
                "total_error_instances": sum(error_analysis.error_distribution.values()),
                "avg_errors_per_sample": sum(error_analysis.error_distribution.values()) / total_samples if total_samples > 0 else 0
            },
            "error_distribution": error_analysis.error_distribution,
            "severity_analysis": error_analysis.severity_levels,
            "common_patterns": error_analysis.common_patterns,
            "improvement_recommendations": error_analysis.improvement_areas,
            "detailed_breakdown": self._generate_detailed_breakdown(error_analysis, sample_results)
        }
        
        return report
    
    def _generate_detailed_breakdown(self, error_analysis: ErrorAnalysis, 
                                   sample_results: List[SampleResult]) -> Dict[str, Any]:
        """生成详细分解"""
        breakdown = {}
        
        for error_type, count in error_analysis.error_distribution.items():
            # 找出该错误类型的样本
            affected_samples = [s for s in sample_results if error_type in s.error_types]
            
            breakdown[error_type] = {
                "count": count,
                "percentage": count / len(sample_results) * 100 if sample_results else 0,
                "severity": error_analysis.severity_levels.get(error_type, "unknown"),
                "affected_samples": len(affected_samples),
                "sample_examples": [s.sample_id for s in affected_samples[:3]]  # 前3个样本作为例子
            }
        
        return breakdown
"""
评估结果聚合器
"""

from typing import Dict, List, Any, Optional
from statistics import mean, median, stdev
from industry_evaluation.models.data_models import (
    EvaluationScore, EvaluationResult, SampleResult, ErrorAnalysis, ErrorType
)


class ResultAggregator:
    """评估结果聚合器"""
    
    def __init__(self, weight_config: Optional[Dict[str, float]] = None):
        """
        初始化聚合器
        
        Args:
            weight_config: 维度权重配置
        """
        self.weight_config = weight_config or {}
    
    def aggregate_scores(self, scores: List[EvaluationScore]) -> EvaluationScore:
        """
        聚合多个评估分数
        
        Args:
            scores: 评估分数列表
            
        Returns:
            EvaluationScore: 聚合后的分数
        """
        if not scores:
            return EvaluationScore(
                overall_score=0.0,
                dimension_scores={},
                confidence=0.0
            )
        
        # 收集所有维度
        all_dimensions = set()
        for score in scores:
            all_dimensions.update(score.dimension_scores.keys())
        
        # 计算各维度的聚合分数
        dimension_scores = {}
        for dimension in all_dimensions:
            dim_scores = [
                score.dimension_scores.get(dimension, 0.0) 
                for score in scores
            ]
            dimension_scores[dimension] = mean(dim_scores)
        
        # 计算总分
        overall_score = self._calculate_weighted_score(dimension_scores)
        
        # 计算平均置信度
        avg_confidence = mean([score.confidence for score in scores])
        
        # 聚合详细信息
        aggregated_details = self._aggregate_details([score.details for score in scores])
        
        return EvaluationScore(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            confidence=avg_confidence,
            details=aggregated_details
        )
    
    def aggregate_sample_results(self, sample_results: List[SampleResult]) -> Dict[str, Any]:
        """
        聚合样本结果
        
        Args:
            sample_results: 样本结果列表
            
        Returns:
            Dict[str, Any]: 聚合统计信息
        """
        if not sample_results:
            return {}
        
        # 收集所有维度
        all_dimensions = set()
        for result in sample_results:
            all_dimensions.update(result.dimension_scores.keys())
        
        aggregated = {
            "total_samples": len(sample_results),
            "dimension_statistics": {},
            "overall_statistics": {},
            "error_statistics": {},
            "performance_statistics": {}
        }
        
        # 计算各维度统计
        for dimension in all_dimensions:
            dim_scores = [
                result.dimension_scores.get(dimension, 0.0)
                for result in sample_results
            ]
            
            aggregated["dimension_statistics"][dimension] = {
                "mean": mean(dim_scores),
                "median": median(dim_scores),
                "std": stdev(dim_scores) if len(dim_scores) > 1 else 0.0,
                "min": min(dim_scores),
                "max": max(dim_scores),
                "count": len(dim_scores)
            }
        
        # 计算总体统计
        overall_scores = [
            result.get_overall_score(self.weight_config)
            for result in sample_results
        ]
        
        aggregated["overall_statistics"] = {
            "mean": mean(overall_scores),
            "median": median(overall_scores),
            "std": stdev(overall_scores) if len(overall_scores) > 1 else 0.0,
            "min": min(overall_scores),
            "max": max(overall_scores)
        }
        
        # 计算错误统计
        aggregated["error_statistics"] = self._calculate_error_statistics(sample_results)
        
        # 计算性能统计
        processing_times = [result.processing_time for result in sample_results]
        aggregated["performance_statistics"] = {
            "avg_processing_time": mean(processing_times),
            "total_processing_time": sum(processing_times),
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times)
        }
        
        return aggregated
    
    def create_evaluation_result(self, task_id: str, model_id: str, 
                               sample_results: List[SampleResult],
                               evaluation_config) -> EvaluationResult:
        """
        创建评估结果
        
        Args:
            task_id: 任务ID
            model_id: 模型ID
            sample_results: 样本结果列表
            evaluation_config: 评估配置
            
        Returns:
            EvaluationResult: 评估结果
        """
        if not sample_results:
            return EvaluationResult(
                task_id=task_id,
                model_id=model_id,
                overall_score=0.0,
                dimension_scores={},
                detailed_results=[],
                error_analysis=ErrorAnalysis({}, [], {}, []),
                improvement_suggestions=[],
                evaluation_config=evaluation_config
            )
        
        # 聚合样本结果
        aggregated_stats = self.aggregate_sample_results(sample_results)
        
        # 计算维度分数
        dimension_scores = {}
        if "dimension_statistics" in aggregated_stats:
            for dimension, stats in aggregated_stats["dimension_statistics"].items():
                dimension_scores[dimension] = stats["mean"]
        
        # 计算总分
        overall_score = self._calculate_weighted_score(dimension_scores)
        
        # 生成错误分析
        error_analysis = self._generate_error_analysis(sample_results)
        
        # 生成改进建议
        improvement_suggestions = self._generate_improvement_suggestions(
            dimension_scores, error_analysis
        )
        
        return EvaluationResult(
            task_id=task_id,
            model_id=model_id,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            detailed_results=sample_results,
            error_analysis=error_analysis,
            improvement_suggestions=improvement_suggestions,
            evaluation_config=evaluation_config
        )
    
    def _calculate_weighted_score(self, dimension_scores: Dict[str, float]) -> float:
        """
        计算加权总分
        
        Args:
            dimension_scores: 维度分数
            
        Returns:
            float: 加权总分
        """
        if not dimension_scores:
            return 0.0
        
        if not self.weight_config:
            # 如果没有权重配置，使用平均分
            return mean(dimension_scores.values())
        
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = self.weight_config.get(dimension, 0.0)
            total_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return mean(dimension_scores.values())
        
        return total_score / total_weight
    
    def _aggregate_details(self, details_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        聚合详细信息
        
        Args:
            details_list: 详细信息列表
            
        Returns:
            Dict[str, Any]: 聚合后的详细信息
        """
        aggregated = {
            "count": len(details_list),
            "common_keys": set(),
            "aggregated_values": {}
        }
        
        if not details_list:
            return aggregated
        
        # 找出所有公共键
        all_keys = [set(details.keys()) for details in details_list if details]
        if all_keys:
            aggregated["common_keys"] = set.intersection(*all_keys)
        
        # 聚合数值类型的值
        for key in aggregated["common_keys"]:
            values = []
            for details in details_list:
                if key in details and isinstance(details[key], (int, float)):
                    values.append(details[key])
            
            if values:
                aggregated["aggregated_values"][key] = {
                    "mean": mean(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return aggregated
    
    def _calculate_error_statistics(self, sample_results: List[SampleResult]) -> Dict[str, Any]:
        """
        计算错误统计
        
        Args:
            sample_results: 样本结果列表
            
        Returns:
            Dict[str, Any]: 错误统计
        """
        error_counts = {}
        total_errors = 0
        samples_with_errors = 0
        
        for result in sample_results:
            if result.error_types:
                samples_with_errors += 1
                for error_type in result.error_types:
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    total_errors += 1
        
        return {
            "error_distribution": error_counts,
            "total_errors": total_errors,
            "samples_with_errors": samples_with_errors,
            "error_rate": samples_with_errors / len(sample_results) if sample_results else 0.0,
            "avg_errors_per_sample": total_errors / len(sample_results) if sample_results else 0.0
        }
    
    def _generate_error_analysis(self, sample_results: List[SampleResult]) -> ErrorAnalysis:
        """
        生成错误分析
        
        Args:
            sample_results: 样本结果列表
            
        Returns:
            ErrorAnalysis: 错误分析
        """
        error_stats = self._calculate_error_statistics(sample_results)
        
        # 分析常见错误模式
        common_patterns = self._identify_error_patterns(sample_results)
        
        # 确定严重程度
        severity_levels = self._determine_severity_levels(error_stats["error_distribution"])
        
        # 识别改进领域
        improvement_areas = self._identify_improvement_areas(
            error_stats["error_distribution"], sample_results
        )
        
        return ErrorAnalysis(
            error_distribution=error_stats["error_distribution"],
            common_patterns=common_patterns,
            severity_levels=severity_levels,
            improvement_areas=improvement_areas
        )
    
    def _identify_error_patterns(self, sample_results: List[SampleResult]) -> List[str]:
        """
        识别错误模式
        
        Args:
            sample_results: 样本结果列表
            
        Returns:
            List[str]: 常见错误模式
        """
        patterns = []
        
        # 分析错误类型组合
        error_combinations = {}
        for result in sample_results:
            if result.error_types:
                combo = tuple(sorted(result.error_types))
                error_combinations[combo] = error_combinations.get(combo, 0) + 1
        
        # 找出最常见的错误组合
        sorted_combos = sorted(error_combinations.items(), key=lambda x: x[1], reverse=True)
        
        for combo, count in sorted_combos[:5]:  # 取前5个最常见的组合
            if count > 1:  # 至少出现2次才算模式
                patterns.append(f"错误组合 {'+'.join(combo)} 出现 {count} 次")
        
        # 分析文本长度与错误的关系
        short_text_errors = sum(1 for result in sample_results 
                               if len(result.model_output) < 50 and result.error_types)
        long_text_errors = sum(1 for result in sample_results 
                              if len(result.model_output) > 200 and result.error_types)
        
        if short_text_errors > len(sample_results) * 0.3:
            patterns.append("短文本输出容易出现错误")
        
        if long_text_errors > len(sample_results) * 0.3:
            patterns.append("长文本输出容易出现错误")
        
        return patterns
    
    def _determine_severity_levels(self, error_distribution: Dict[str, int]) -> Dict[str, str]:
        """
        确定错误严重程度
        
        Args:
            error_distribution: 错误分布
            
        Returns:
            Dict[str, str]: 错误严重程度
        """
        severity_levels = {}
        
        if not error_distribution:
            return severity_levels
        
        total_errors = sum(error_distribution.values())
        
        for error_type, count in error_distribution.items():
            error_rate = count / total_errors
            
            if error_rate >= 0.5:
                severity_levels[error_type] = "critical"
            elif error_rate >= 0.3:
                severity_levels[error_type] = "high"
            elif error_rate >= 0.1:
                severity_levels[error_type] = "medium"
            else:
                severity_levels[error_type] = "low"
        
        return severity_levels
    
    def _identify_improvement_areas(self, error_distribution: Dict[str, int], 
                                  sample_results: List[SampleResult]) -> List[str]:
        """
        识别改进领域
        
        Args:
            error_distribution: 错误分布
            sample_results: 样本结果列表
            
        Returns:
            List[str]: 改进领域
        """
        improvement_areas = []
        
        # 基于错误类型识别改进领域
        error_to_area = {
            "knowledge_error": "专业知识掌握",
            "terminology_error": "术语使用准确性",
            "reasoning_error": "逻辑推理能力",
            "context_error": "上下文理解",
            "format_error": "输出格式规范",
            "logic_error": "逻辑一致性"
        }
        
        # 按错误频率排序
        sorted_errors = sorted(error_distribution.items(), key=lambda x: x[1], reverse=True)
        
        for error_type, count in sorted_errors:
            if count > 0:
                area = error_to_area.get(error_type, error_type)
                improvement_areas.append(area)
        
        # 基于分数分析识别改进领域
        if sample_results:
            # 收集所有维度
            all_dimensions = set()
            for result in sample_results:
                all_dimensions.update(result.dimension_scores.keys())
            
            # 找出分数最低的维度
            dimension_avg_scores = {}
            for dimension in all_dimensions:
                scores = [result.dimension_scores.get(dimension, 0.0) 
                         for result in sample_results]
                dimension_avg_scores[dimension] = mean(scores)
            
            # 按分数排序，分数低的需要改进
            sorted_dimensions = sorted(dimension_avg_scores.items(), key=lambda x: x[1])
            
            for dimension, avg_score in sorted_dimensions:
                if avg_score < 0.6:  # 分数低于0.6的维度需要改进
                    if dimension not in improvement_areas:
                        improvement_areas.append(dimension)
        
        return improvement_areas[:5]  # 返回前5个最需要改进的领域
    
    def _generate_improvement_suggestions(self, dimension_scores: Dict[str, float], 
                                        error_analysis: ErrorAnalysis) -> List[str]:
        """
        生成改进建议
        
        Args:
            dimension_scores: 维度分数
            error_analysis: 错误分析
            
        Returns:
            List[str]: 改进建议
        """
        suggestions = []
        
        # 基于维度分数生成建议
        for dimension, score in dimension_scores.items():
            if score < 0.6:
                if dimension == "knowledge":
                    suggestions.append("建议增强专业知识训练数据，提高模型对行业概念的理解")
                elif dimension == "terminology":
                    suggestions.append("建议补充行业术语词典，加强术语使用的准确性训练")
                elif dimension == "reasoning":
                    suggestions.append("建议增加逻辑推理训练样本，提高模型的推理能力")
                elif dimension == "context":
                    suggestions.append("建议增加长文本和多轮对话训练，提高上下文理解能力")
                else:
                    suggestions.append(f"建议针对{dimension}维度进行专项训练和优化")
        
        # 基于错误分析生成建议
        critical_errors = [error_type for error_type, severity in error_analysis.severity_levels.items() 
                          if severity == "critical"]
        
        for error_type in critical_errors:
            if error_type == "knowledge_error":
                suggestions.append("发现大量专业知识错误，建议重点加强行业知识库建设")
            elif error_type == "terminology_error":
                suggestions.append("术语使用错误较多，建议建立标准术语库并加强术语一致性检查")
            elif error_type == "reasoning_error":
                suggestions.append("推理错误频发，建议增加因果关系和逻辑链条的训练")
        
        # 基于改进领域生成建议
        for area in error_analysis.improvement_areas[:3]:  # 取前3个最需要改进的领域
            suggestions.append(f"重点关注{area}的提升，建议制定专项改进计划")
        
        # 去重并限制数量
        unique_suggestions = list(dict.fromkeys(suggestions))  # 去重
        return unique_suggestions[:8]  # 最多返回8条建议
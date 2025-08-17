"""
数据质量分析器

提供数据质量分析、问题检测和改进建议功能。
"""

import logging
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from datasets import Dataset
from .data_models import DataQualityReport, DataIssue, DataIssueType

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """
    数据质量分析器
    
    分析数据质量和分布特征，检测数据问题并提供改进建议。
    """
    
    def __init__(self, 
                 min_length: int = 5,
                 max_length: int = 2048,
                 length_outlier_threshold: float = 3.0):
        """
        初始化质量分析器
        
        Args:
            min_length: 最小文本长度
            max_length: 最大文本长度
            length_outlier_threshold: 长度异常值阈值（标准差倍数）
        """
        self.min_length = min_length
        self.max_length = max_length
        self.length_outlier_threshold = length_outlier_threshold
        
        # 质量检查配置
        self.quality_checks = {
            "empty_content": True,
            "encoding_errors": True,
            "format_consistency": True,
            "length_outliers": True,
            "duplicate_detection": True,
            "vocabulary_analysis": True,
            "language_detection": True
        }
        
        logger.info("QualityAnalyzer初始化完成")
    
    def analyze_data_quality(self, dataset: Dataset, dataset_name: str = "dataset") -> DataQualityReport:
        """
        分析数据质量
        
        Args:
            dataset: 要分析的数据集
            dataset_name: 数据集名称
            
        Returns:
            DataQualityReport: 数据质量报告
        """
        logger.info(f"开始分析数据质量: {dataset_name}")
        
        # 计算基本统计信息
        statistics = self._calculate_basic_statistics(dataset)
        
        # 检测数据问题
        issues = self.detect_data_issues(dataset)
        
        # 计算质量分数
        quality_score = self._calculate_quality_score(statistics, issues)
        
        # 生成改进建议
        recommendations = self.suggest_improvements(statistics, issues)
        
        report = DataQualityReport(
            dataset_name=dataset_name,
            total_samples=len(dataset),
            quality_score=quality_score,
            statistics=statistics,
            issues=[issue.to_dict() for issue in issues],
            recommendations=recommendations,
            analysis_time=datetime.now()
        )
        
        logger.info(f"数据质量分析完成，质量分数: {quality_score:.4f}")
        
        return report
    
    def _calculate_basic_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """计算基本统计信息"""
        stats = {
            "total_samples": len(dataset),
            "columns": dataset.column_names,
            "column_types": {}
        }
        
        # 分析每个字段
        for column in dataset.column_names:
            column_stats = self._analyze_column(dataset[column], column)
            stats[f"{column}_stats"] = column_stats
            stats["column_types"][column] = column_stats["type"]
        
        # 整体统计
        stats["overall"] = self._calculate_overall_statistics(dataset)
        
        return stats
    
    def _analyze_column(self, values: List[Any], column_name: str) -> Dict[str, Any]:
        """分析单个字段"""
        stats = {
            "type": self._infer_column_type(values),
            "non_null_count": sum(1 for v in values if v is not None and v != ""),
            "null_count": sum(1 for v in values if v is None or v == ""),
            "unique_count": len(set(values))
        }
        
        # 根据字段类型进行特定分析
        if stats["type"] == "text":
            stats.update(self._analyze_text_column(values))
        elif stats["type"] == "numeric":
            stats.update(self._analyze_numeric_column(values))
        elif stats["type"] == "categorical":
            stats.update(self._analyze_categorical_column(values))
        
        return stats
    
    def _infer_column_type(self, values: List[Any]) -> str:
        """推断字段类型"""
        non_null_values = [v for v in values if v is not None and v != ""]
        
        if not non_null_values:
            return "empty"
        
        # 检查是否为数值类型
        try:
            [float(v) for v in non_null_values[:100]]  # 检查前100个值
            return "numeric"
        except (ValueError, TypeError):
            pass
        
        # 检查是否为文本类型
        if all(isinstance(v, str) for v in non_null_values[:100]):
            # 如果唯一值较少，可能是分类字段
            unique_ratio = len(set(non_null_values)) / len(non_null_values)
            if unique_ratio < 0.1 and len(set(non_null_values)) < 50:
                return "categorical"
            else:
                return "text"
        
        return "mixed"
    
    def _analyze_text_column(self, values: List[str]) -> Dict[str, Any]:
        """分析文本字段"""
        non_empty_values = [v for v in values if v and isinstance(v, str)]
        
        if not non_empty_values:
            return {"error": "no_valid_text"}
        
        # 长度统计
        lengths = [len(v) for v in non_empty_values]
        word_counts = [len(v.split()) for v in non_empty_values]
        
        # 字符统计
        all_chars = ''.join(non_empty_values)
        char_counter = Counter(all_chars)
        
        # 词汇统计
        all_words = []
        for text in non_empty_values:
            all_words.extend(text.split())
        word_counter = Counter(all_words)
        
        return {
            "length_stats": {
                "mean": float(np.mean(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "median": float(np.median(lengths)),
                "q25": float(np.percentile(lengths, 25)),
                "q75": float(np.percentile(lengths, 75))
            },
            "word_count_stats": {
                "mean": float(np.mean(word_counts)),
                "std": float(np.std(word_counts)),
                "min": int(np.min(word_counts)),
                "max": int(np.max(word_counts)),
                "median": float(np.median(word_counts))
            },
            "vocabulary": {
                "total_words": len(all_words),
                "unique_words": len(word_counter),
                "vocabulary_diversity": len(word_counter) / len(all_words) if all_words else 0,
                "most_common_words": word_counter.most_common(10)
            },
            "character_stats": {
                "total_chars": len(all_chars),
                "unique_chars": len(char_counter),
                "most_common_chars": char_counter.most_common(10)
            },
            "encoding_issues": self._detect_encoding_issues(non_empty_values),
            "format_patterns": self._detect_format_patterns(non_empty_values)
        }
    
    def _analyze_numeric_column(self, values: List[Any]) -> Dict[str, Any]:
        """分析数值字段"""
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                continue
        
        if not numeric_values:
            return {"error": "no_valid_numbers"}
        
        return {
            "stats": {
                "mean": float(np.mean(numeric_values)),
                "std": float(np.std(numeric_values)),
                "min": float(np.min(numeric_values)),
                "max": float(np.max(numeric_values)),
                "median": float(np.median(numeric_values)),
                "q25": float(np.percentile(numeric_values, 25)),
                "q75": float(np.percentile(numeric_values, 75))
            },
            "distribution": {
                "skewness": float(np.nan_to_num(np.array(numeric_values).std())),
                "outliers": self._detect_numeric_outliers(numeric_values)
            }
        }
    
    def _analyze_categorical_column(self, values: List[Any]) -> Dict[str, Any]:
        """分析分类字段"""
        non_null_values = [v for v in values if v is not None and v != ""]
        counter = Counter(non_null_values)
        
        return {
            "unique_count": len(counter),
            "most_common": counter.most_common(10),
            "distribution": dict(counter),
            "entropy": self._calculate_entropy(counter),
            "balance_score": self._calculate_balance_score(counter)
        }
    
    def _calculate_overall_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """计算整体统计信息"""
        stats = {
            "completeness": self._calculate_completeness(dataset),
            "consistency": self._calculate_consistency(dataset),
            "uniqueness": self._calculate_uniqueness(dataset)
        }
        
        return stats
    
    def _calculate_completeness(self, dataset: Dataset) -> Dict[str, float]:
        """计算数据完整性"""
        completeness = {}
        
        for column in dataset.column_names:
            values = dataset[column]
            non_null_count = sum(1 for v in values if v is not None and v != "")
            completeness[column] = non_null_count / len(values)
        
        completeness["overall"] = float(np.mean(list(completeness.values())))
        
        return completeness
    
    def _calculate_consistency(self, dataset: Dataset) -> Dict[str, Any]:
        """计算数据一致性"""
        consistency = {}
        
        # 检查格式一致性
        for column in dataset.column_names:
            if column in ["text", "question", "answer"]:
                values = [v for v in dataset[column] if v]
                if values:
                    consistency[f"{column}_format"] = self._check_format_consistency(values)
        
        return consistency
    
    def _calculate_uniqueness(self, dataset: Dataset) -> Dict[str, float]:
        """计算数据唯一性"""
        uniqueness = {}
        
        for column in dataset.column_names:
            values = dataset[column]
            unique_count = len(set(values))
            uniqueness[column] = unique_count / len(values) if values else 0
        
        return uniqueness
    
    def detect_data_issues(self, dataset: Dataset) -> List[DataIssue]:
        """检测数据问题"""
        issues = []
        
        # 检查空内容
        if self.quality_checks["empty_content"]:
            issues.extend(self._detect_empty_content(dataset))
        
        # 检查编码错误
        if self.quality_checks["encoding_errors"]:
            issues.extend(self._detect_encoding_errors(dataset))
        
        # 检查格式一致性
        if self.quality_checks["format_consistency"]:
            issues.extend(self._detect_format_inconsistency(dataset))
        
        # 检查长度异常值
        if self.quality_checks["length_outliers"]:
            issues.extend(self._detect_length_outliers(dataset))
        
        # 检查重复内容
        if self.quality_checks["duplicate_detection"]:
            issues.extend(self._detect_duplicates(dataset))
        
        return issues
    
    def _detect_empty_content(self, dataset: Dataset) -> List[DataIssue]:
        """检测空内容"""
        issues = []
        
        for column in dataset.column_names:
            if column in ["text", "question", "answer"]:
                empty_indices = []
                for i, value in enumerate(dataset[column]):
                    if not value or (isinstance(value, str) and not value.strip()):
                        empty_indices.append(i)
                
                if empty_indices:
                    issues.append(DataIssue(
                        issue_type=DataIssueType.EMPTY_CONTENT,
                        description=f"{column} 字段中发现 {len(empty_indices)} 个空内容",
                        affected_samples=empty_indices,
                        severity="medium" if len(empty_indices) < len(dataset) * 0.1 else "high",
                        suggested_action=f"移除或填充 {column} 字段的空内容"
                    ))
        
        return issues
    
    def _detect_encoding_errors(self, dataset: Dataset) -> List[DataIssue]:
        """检测编码错误"""
        issues = []
        
        for column in dataset.column_names:
            if column in ["text", "question", "answer"]:
                error_indices = []
                for i, value in enumerate(dataset[column]):
                    if isinstance(value, str) and self._has_encoding_issues(value):
                        error_indices.append(i)
                
                if error_indices:
                    issues.append(DataIssue(
                        issue_type=DataIssueType.ENCODING_ERROR,
                        description=f"{column} 字段中发现 {len(error_indices)} 个编码错误",
                        affected_samples=error_indices,
                        severity="medium",
                        suggested_action=f"修复 {column} 字段的编码问题"
                    ))
        
        return issues
    
    def _detect_format_inconsistency(self, dataset: Dataset) -> List[DataIssue]:
        """检测格式不一致"""
        issues = []
        
        # 检查QA格式一致性
        if "question" in dataset.column_names and "answer" in dataset.column_names:
            inconsistent_indices = []
            
            for i, (q, a) in enumerate(zip(dataset["question"], dataset["answer"])):
                if not self._is_valid_qa_pair(q, a):
                    inconsistent_indices.append(i)
            
            if inconsistent_indices:
                issues.append(DataIssue(
                    issue_type=DataIssueType.FORMAT_ERROR,
                    description=f"发现 {len(inconsistent_indices)} 个格式不一致的QA对",
                    affected_samples=inconsistent_indices,
                    severity="medium",
                    suggested_action="检查并修复QA对的格式"
                ))
        
        return issues
    
    def _detect_length_outliers(self, dataset: Dataset) -> List[DataIssue]:
        """检测长度异常值"""
        issues = []
        
        for column in dataset.column_names:
            if column in ["text", "question", "answer"]:
                lengths = [len(str(v)) for v in dataset[column] if v]
                
                if not lengths:
                    continue
                
                mean_length = float(np.mean(lengths))
                std_length = float(np.std(lengths))
                
                outlier_indices = []
                for i, value in enumerate(dataset[column]):
                    if value:
                        length = len(str(value))
                        z_score = abs(length - mean_length) / std_length if std_length > 0 else 0
                        
                        if (z_score > self.length_outlier_threshold or 
                            length < self.min_length or 
                            length > self.max_length):
                            outlier_indices.append(i)
                
                if outlier_indices:
                    issues.append(DataIssue(
                        issue_type=DataIssueType.LENGTH_OUTLIER,
                        description=f"{column} 字段中发现 {len(outlier_indices)} 个长度异常值",
                        affected_samples=outlier_indices,
                        severity="low" if len(outlier_indices) < len(dataset) * 0.05 else "medium",
                        suggested_action=f"检查 {column} 字段的异常长度样本"
                    ))
        
        return issues
    
    def _detect_duplicates(self, dataset: Dataset) -> List[DataIssue]:
        """检测重复内容"""
        issues = []
        
        for column in dataset.column_names:
            if column in ["text", "question", "answer"]:
                value_to_indices = defaultdict(list)
                
                for i, value in enumerate(dataset[column]):
                    if value:
                        value_to_indices[str(value)].append(i)
                
                duplicate_groups = {k: v for k, v in value_to_indices.items() if len(v) > 1}
                
                if duplicate_groups:
                    total_duplicates = sum(len(indices) - 1 for indices in duplicate_groups.values())
                    all_duplicate_indices = []
                    for indices in duplicate_groups.values():
                        all_duplicate_indices.extend(indices[1:])  # 保留第一个，标记其余为重复
                    
                    issues.append(DataIssue(
                        issue_type=DataIssueType.DUPLICATE,
                        description=f"{column} 字段中发现 {total_duplicates} 个重复内容",
                        affected_samples=all_duplicate_indices,
                        severity="medium",
                        suggested_action=f"移除 {column} 字段的重复内容"
                    ))
        
        return issues
    
    def _has_encoding_issues(self, text: str) -> bool:
        """检查文本是否有编码问题"""
        # 检查常见的编码问题标志
        encoding_issues = [
            '�',  # 替换字符
            '\ufffd',  # Unicode替换字符
            '\\x',  # 转义序列
            '\\u',  # Unicode转义
        ]
        
        return any(issue in text for issue in encoding_issues)
    
    def _is_valid_qa_pair(self, question: str, answer: str) -> bool:
        """检查QA对是否有效"""
        if not question or not answer:
            return False
        
        # 检查基本格式
        if len(question.strip()) < 3 or len(answer.strip()) < 3:
            return False
        
        # 检查是否问题和答案相同
        if question.strip() == answer.strip():
            return False
        
        return True
    
    def _detect_encoding_issues(self, values: List[str]) -> List[str]:
        """检测编码问题"""
        issues = []
        
        for value in values[:100]:  # 检查前100个值
            if self._has_encoding_issues(value):
                issues.append("encoding_error")
                break
        
        return issues
    
    def _detect_format_patterns(self, values: List[str]) -> Dict[str, Any]:
        """检测格式模式"""
        patterns = {
            "has_urls": sum(1 for v in values[:100] if re.search(r'https?://', v)) > 0,
            "has_emails": sum(1 for v in values[:100] if re.search(r'\S+@\S+', v)) > 0,
            "has_numbers": sum(1 for v in values[:100] if re.search(r'\d+', v)) > 0,
            "has_punctuation": sum(1 for v in values[:100] if re.search(r'[!@#$%^&*(),.?":{}|<>]', v)) > 0
        }
        
        return patterns
    
    def _detect_numeric_outliers(self, values: List[float]) -> List[int]:
        """检测数值异常值"""
        if len(values) < 4:
            return []
        
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)
        
        return outliers
    
    def _calculate_entropy(self, counter: Counter) -> float:
        """计算信息熵"""
        total = sum(counter.values())
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        return entropy
    
    def _calculate_balance_score(self, counter: Counter) -> float:
        """计算类别平衡分数"""
        if len(counter) <= 1:
            return 1.0
        
        counts = list(counter.values())
        max_count = max(counts)
        min_count = min(counts)
        
        # 平衡分数：最小值与最大值的比例
        return min_count / max_count if max_count > 0 else 0
    
    def _check_format_consistency(self, values: List[str]) -> float:
        """检查格式一致性"""
        if not values:
            return 1.0
        
        # 简单的格式一致性检查
        # 检查长度分布的一致性
        lengths = [len(v) for v in values]
        cv = float(np.std(lengths) / np.mean(lengths)) if np.mean(lengths) > 0 else 0.0
        
        # 转换为一致性分数
        return max(0, 1 - cv)
    
    def _calculate_quality_score(self, statistics: Dict[str, Any], issues: List[DataIssue]) -> float:
        """计算总体质量分数"""
        # 基础分数
        base_score = 1.0
        
        # 根据问题严重程度扣分
        for issue in issues:
            if issue.severity == "high":
                base_score -= 0.2
            elif issue.severity == "medium":
                base_score -= 0.1
            elif issue.severity == "low":
                base_score -= 0.05
        
        # 根据完整性调整
        completeness = statistics.get("overall", {}).get("completeness", {})
        if isinstance(completeness, dict) and "overall" in completeness:
            completeness_score = completeness["overall"]
            base_score *= completeness_score
        
        return max(0.0, min(1.0, base_score))
    
    def suggest_improvements(self, statistics: Dict[str, Any], issues: List[DataIssue]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于问题的建议
        issue_types = [issue.issue_type for issue in issues]
        
        if DataIssueType.EMPTY_CONTENT in issue_types:
            recommendations.append("移除或填充空内容，确保数据完整性")
        
        if DataIssueType.DUPLICATE in issue_types:
            recommendations.append("去除重复数据，避免模型过拟合")
        
        if DataIssueType.ENCODING_ERROR in issue_types:
            recommendations.append("修复编码错误，确保文本正确显示")
        
        if DataIssueType.FORMAT_ERROR in issue_types:
            recommendations.append("统一数据格式，提高数据一致性")
        
        if DataIssueType.LENGTH_OUTLIER in issue_types:
            recommendations.append("处理长度异常值，考虑截断或移除")
        
        # 基于统计信息的建议
        completeness = statistics.get("overall", {}).get("completeness", {})
        if isinstance(completeness, dict) and "overall" in completeness:
            if completeness["overall"] < 0.9:
                recommendations.append("提高数据完整性，补充缺失信息")
        
        # 词汇多样性建议
        for column in ["text", "question", "answer"]:
            column_stats = statistics.get(f"{column}_stats", {})
            if "vocabulary" in column_stats:
                diversity = column_stats["vocabulary"].get("vocabulary_diversity", 0)
                if diversity < 0.1:
                    recommendations.append(f"增加{column}字段的词汇多样性")
        
        if not recommendations:
            recommendations.append("数据质量良好，可以直接用于训练")
        
        return recommendations
    
    def generate_quality_report(self, report: DataQualityReport, output_dir: str) -> str:
        """
        生成质量分析报告
        
        Args:
            report: 质量分析报告
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成JSON报告
        json_file = output_path / f"{report.dataset_name}_quality_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 生成HTML报告
        html_file = output_path / f"{report.dataset_name}_quality_report.html"
        html_content = self._generate_quality_html_report(report)
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"质量分析报告已生成: {html_file}")
        
        return str(html_file)
    
    def _generate_quality_html_report(self, report: DataQualityReport) -> str:
        """生成HTML格式的质量报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>数据质量分析报告 - {report.dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007cba; }}
                .issue {{ background-color: #ffebee; padding: 10px; margin: 5px 0; border-left: 4px solid #f44336; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
                .score {{ font-size: 24px; font-weight: bold; color: {'#4caf50' if report.quality_score >= 0.8 else '#ff9800' if report.quality_score >= 0.6 else '#f44336'}; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>数据质量分析报告</h1>
                <h2>{report.dataset_name}</h2>
                <p>分析时间: {report.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>总样本数: {report.total_samples}</p>
                <p>质量分数: <span class="score">{report.quality_score:.4f}</span></p>
            </div>
            
            <div class="section">
                <h2>数据问题 ({len(report.issues)}个)</h2>
                {''.join([f'<div class="issue"><strong>{issue["issue_type"]}</strong>: {issue["description"]} (严重程度: {issue["severity"]})</div>' for issue in report.issues])}
            </div>
            
            <div class="section">
                <h2>改进建议</h2>
                {''.join([f'<div class="recommendation">{rec}</div>' for rec in report.recommendations])}
            </div>
            
            <div class="section">
                <h2>统计信息</h2>
                <pre>{json.dumps(report.statistics, indent=2, ensure_ascii=False)}</pre>
            </div>
        </body>
        </html>
        """
        
        return html
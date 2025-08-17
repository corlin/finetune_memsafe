"""
数据分布分析器

提供数据分布一致性检查、统计检验和数据泄露检测功能。
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from datasets import Dataset
from .data_models import DistributionAnalysis, DataIssue, DataIssueType

logger = logging.getLogger(__name__)


class DistributionAnalyzer:
    """
    数据分布分析器
    
    提供数据分布一致性检查、统计检验验证拆分质量和数据泄露检测功能。
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        初始化分布分析器
        
        Args:
            significance_level: 统计检验的显著性水平
        """
        self.significance_level = significance_level
        
    def analyze_distribution_consistency(self, splits: Dict[str, Dataset]) -> DistributionAnalysis:
        """
        分析数据分布一致性
        
        Args:
            splits: 拆分后的数据集字典
            
        Returns:
            DistributionAnalysis: 分布分析结果
        """
        logger.info("开始分析数据分布一致性")
        
        # 计算各数据集的统计信息
        stats_dict = {}
        for split_name, dataset in splits.items():
            stats_dict[split_name] = self._calculate_comprehensive_stats(dataset)
        
        # 计算分布一致性分数
        consistency_score = self._calculate_comprehensive_consistency_score(stats_dict)
        
        # 执行统计检验
        statistical_tests = self._perform_comprehensive_statistical_tests(splits)
        
        # 生成改进建议
        recommendations = self._generate_comprehensive_recommendations(
            stats_dict, consistency_score, statistical_tests
        )
        
        analysis = DistributionAnalysis(
            train_stats=stats_dict.get("train", {}),
            val_stats=stats_dict.get("val", {}),
            test_stats=stats_dict.get("test", {}),
            consistency_score=consistency_score,
            statistical_tests=statistical_tests,
            recommendations=recommendations
        )
        
        logger.info(f"分布分析完成，一致性分数: {consistency_score:.4f}")
        
        return analysis
    
    def _calculate_comprehensive_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """计算全面的数据集统计信息"""
        stats = {
            "size": len(dataset),
            "columns": dataset.column_names
        }
        
        # 文本相关统计
        text_fields = ["text", "question", "answer"]
        for field in text_fields:
            if field in dataset.column_names:
                texts = dataset[field]
                stats[f"{field}_stats"] = self._calculate_text_statistics(texts)
        
        # 分类字段统计
        categorical_fields = ["source", "category", "label"]
        for field in categorical_fields:
            if field in dataset.column_names:
                values = dataset[field]
                stats[f"{field}_distribution"] = self._calculate_categorical_statistics(values)
        
        # 数值字段统计
        numeric_fields = ["score", "rating", "difficulty"]
        for field in numeric_fields:
            if field in dataset.column_names:
                values = dataset[field]
                stats[f"{field}_stats"] = self._calculate_numeric_statistics(values)
        
        return stats
    
    def _calculate_text_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """计算文本统计信息"""
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        # 基本统计
        basic_stats = {
            "char_length": {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": np.min(lengths),
                "max": np.max(lengths),
                "median": np.median(lengths),
                "q25": np.percentile(lengths, 25),
                "q75": np.percentile(lengths, 75)
            },
            "word_count": {
                "mean": np.mean(word_counts),
                "std": np.std(word_counts),
                "min": np.min(word_counts),
                "max": np.max(word_counts),
                "median": np.median(word_counts)
            }
        }
        
        # 词汇多样性
        all_words = []
        for text in texts:
            all_words.extend(text.split())
        
        unique_words = set(all_words)
        basic_stats["vocabulary"] = {
            "total_words": len(all_words),
            "unique_words": len(unique_words),
            "vocabulary_diversity": len(unique_words) / len(all_words) if all_words else 0
        }
        
        # 长度分布
        basic_stats["length_distribution"] = {
            "histogram": np.histogram(lengths, bins=20)[0].tolist(),
            "bin_edges": np.histogram(lengths, bins=20)[1].tolist()
        }
        
        return basic_stats
    
    def _calculate_categorical_statistics(self, values: List[str]) -> Dict[str, Any]:
        """计算分类字段统计信息"""
        counter = Counter(values)
        total = len(values)
        
        return {
            "unique_count": len(counter),
            "most_common": counter.most_common(10),
            "distribution": dict(counter),
            "proportions": {k: v/total for k, v in counter.items()},
            "entropy": self._calculate_entropy(counter)
        }
    
    def _calculate_numeric_statistics(self, values: List[float]) -> Dict[str, Any]:
        """计算数值字段统计信息"""
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
            "skewness": stats.skew(values),
            "kurtosis": stats.kurtosis(values)
        }
    
    def _calculate_entropy(self, counter: Counter) -> float:
        """计算信息熵"""
        total = sum(counter.values())
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        return entropy
    
    def _calculate_comprehensive_consistency_score(self, stats_dict: Dict[str, Dict[str, Any]]) -> float:
        """计算全面的分布一致性分数"""
        if len(stats_dict) < 2:
            return 1.0
        
        scores = []
        
        # 文本长度一致性
        text_fields = ["text_stats", "question_stats", "answer_stats"]
        for field in text_fields:
            if all(field in stats for stats in stats_dict.values()):
                field_score = self._compare_text_distributions(stats_dict, field)
                scores.append(field_score)
        
        # 分类分布一致性
        categorical_fields = ["source_distribution", "category_distribution", "label_distribution"]
        for field in categorical_fields:
            if all(field in stats for stats in stats_dict.values()):
                field_score = self._compare_categorical_distributions_advanced(stats_dict, field)
                scores.append(field_score)
        
        # 数值分布一致性
        numeric_fields = ["score_stats", "rating_stats", "difficulty_stats"]
        for field in numeric_fields:
            if all(field in stats for stats in stats_dict.values()):
                field_score = self._compare_numeric_distributions(stats_dict, field)
                scores.append(field_score)
        
        return np.mean(scores) if scores else 1.0
    
    def _compare_text_distributions(self, stats_dict: Dict[str, Dict[str, Any]], 
                                  field: str) -> float:
        """比较文本分布的一致性"""
        char_lengths = []
        word_counts = []
        
        for stats in stats_dict.values():
            char_lengths.append(stats[field]["char_length"]["mean"])
            word_counts.append(stats[field]["word_count"]["mean"])
        
        # 计算变异系数
        char_cv = np.std(char_lengths) / np.mean(char_lengths) if np.mean(char_lengths) > 0 else 0
        word_cv = np.std(word_counts) / np.mean(word_counts) if np.mean(word_counts) > 0 else 0
        
        # 转换为一致性分数
        char_score = max(0, 1 - char_cv)
        word_score = max(0, 1 - word_cv)
        
        return (char_score + word_score) / 2
    
    def _compare_categorical_distributions_advanced(self, stats_dict: Dict[str, Dict[str, Any]], 
                                                  field: str) -> float:
        """比较分类分布的一致性（高级版本）"""
        distributions = [stats[field]["proportions"] for stats in stats_dict.values()]
        
        if len(distributions) < 2:
            return 1.0
        
        # 计算Jensen-Shannon散度
        js_divergences = []
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                js_div = self._calculate_js_divergence(distributions[i], distributions[j])
                js_divergences.append(js_div)
        
        if not js_divergences:
            return 1.0
        
        avg_js_div = np.mean(js_divergences)
        # JS散度的范围是[0, 1]，转换为一致性分数
        return 1 - avg_js_div
    
    def _compare_numeric_distributions(self, stats_dict: Dict[str, Dict[str, Any]], 
                                     field: str) -> float:
        """比较数值分布的一致性"""
        means = [stats[field]["mean"] for stats in stats_dict.values()]
        stds = [stats[field]["std"] for stats in stats_dict.values()]
        
        # 计算均值和标准差的变异系数
        mean_cv = np.std(means) / np.mean(means) if np.mean(means) > 0 else 0
        std_cv = np.std(stds) / np.mean(stds) if np.mean(stds) > 0 else 0
        
        # 转换为一致性分数
        mean_score = max(0, 1 - mean_cv)
        std_score = max(0, 1 - std_cv)
        
        return (mean_score + std_score) / 2
    
    def _calculate_js_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """计算Jensen-Shannon散度"""
        # 获取所有类别
        all_keys = set(p.keys()) | set(q.keys())
        
        # 创建概率向量
        p_vec = np.array([p.get(key, 0) for key in all_keys])
        q_vec = np.array([q.get(key, 0) for key in all_keys])
        
        # 归一化
        p_vec = p_vec / np.sum(p_vec) if np.sum(p_vec) > 0 else p_vec
        q_vec = q_vec / np.sum(q_vec) if np.sum(q_vec) > 0 else q_vec
        
        # 计算JS散度
        m = (p_vec + q_vec) / 2
        js_div = 0.5 * self._kl_divergence(p_vec, m) + 0.5 * self._kl_divergence(q_vec, m)
        
        return js_div
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算KL散度"""
        kl_div = 0
        for i in range(len(p)):
            if p[i] > 0 and q[i] > 0:
                kl_div += p[i] * np.log(p[i] / q[i])
        return kl_div
    
    def _perform_comprehensive_statistical_tests(self, splits: Dict[str, Dataset]) -> Dict[str, Any]:
        """执行全面的统计检验"""
        tests = {}
        
        if len(splits) < 2:
            return {"note": "需要至少两个数据集进行统计检验"}
        
        split_names = list(splits.keys())
        datasets = list(splits.values())
        
        # Kolmogorov-Smirnov检验（用于连续分布）
        ks_tests = self._perform_ks_tests(datasets, split_names)
        tests["ks_tests"] = ks_tests
        
        # 卡方检验（用于分类分布）
        chi2_tests = self._perform_chi2_tests(datasets, split_names)
        tests["chi2_tests"] = chi2_tests
        
        # Mann-Whitney U检验（用于非参数比较）
        mw_tests = self._perform_mann_whitney_tests(datasets, split_names)
        tests["mann_whitney_tests"] = mw_tests
        
        return tests
    
    def _perform_ks_tests(self, datasets: List[Dataset], split_names: List[str]) -> Dict[str, Any]:
        """执行Kolmogorov-Smirnov检验"""
        ks_results = {}
        
        # 检验文本长度分布
        text_fields = ["text", "question", "answer"]
        for field in text_fields:
            if all(field in dataset.column_names for dataset in datasets):
                field_results = {}
                
                for i in range(len(datasets)):
                    for j in range(i + 1, len(datasets)):
                        lengths_i = [len(text) for text in datasets[i][field]]
                        lengths_j = [len(text) for text in datasets[j][field]]
                        
                        statistic, p_value = ks_2samp(lengths_i, lengths_j)
                        
                        field_results[f"{split_names[i]}_vs_{split_names[j]}"] = {
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "significant": p_value < self.significance_level
                        }
                
                ks_results[f"{field}_length"] = field_results
        
        return ks_results
    
    def _perform_chi2_tests(self, datasets: List[Dataset], split_names: List[str]) -> Dict[str, Any]:
        """执行卡方检验"""
        chi2_results = {}
        
        # 检验分类字段分布
        categorical_fields = ["source", "category", "label"]
        for field in categorical_fields:
            if all(field in dataset.column_names for dataset in datasets):
                # 获取所有可能的类别
                all_categories = set()
                for dataset in datasets:
                    all_categories.update(dataset[field])
                
                all_categories = sorted(list(all_categories))
                
                # 构建列联表
                contingency_table = []
                for dataset in datasets:
                    counter = Counter(dataset[field])
                    row = [counter.get(cat, 0) for cat in all_categories]
                    contingency_table.append(row)
                
                try:
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    chi2_results[field] = {
                        "chi2_statistic": float(chi2_stat),
                        "p_value": float(p_value),
                        "degrees_of_freedom": int(dof),
                        "significant": p_value < self.significance_level,
                        "categories": all_categories,
                        "contingency_table": contingency_table
                    }
                except ValueError as e:
                    chi2_results[field] = {
                        "error": str(e),
                        "note": "卡方检验失败，可能是由于样本数过少"
                    }
        
        return chi2_results
    
    def _perform_mann_whitney_tests(self, datasets: List[Dataset], split_names: List[str]) -> Dict[str, Any]:
        """执行Mann-Whitney U检验"""
        mw_results = {}
        
        # 检验数值字段分布
        numeric_fields = ["score", "rating", "difficulty"]
        for field in numeric_fields:
            if all(field in dataset.column_names for dataset in datasets):
                field_results = {}
                
                for i in range(len(datasets)):
                    for j in range(i + 1, len(datasets)):
                        values_i = datasets[i][field]
                        values_j = datasets[j][field]
                        
                        try:
                            statistic, p_value = stats.mannwhitneyu(values_i, values_j, 
                                                                   alternative='two-sided')
                            
                            field_results[f"{split_names[i]}_vs_{split_names[j]}"] = {
                                "statistic": float(statistic),
                                "p_value": float(p_value),
                                "significant": p_value < self.significance_level
                            }
                        except ValueError as e:
                            field_results[f"{split_names[i]}_vs_{split_names[j]}"] = {
                                "error": str(e)
                            }
                
                mw_results[field] = field_results
        
        return mw_results
    
    def _generate_comprehensive_recommendations(self, stats_dict: Dict[str, Dict[str, Any]], 
                                              consistency_score: float,
                                              statistical_tests: Dict[str, Any]) -> List[str]:
        """生成全面的分布改进建议"""
        recommendations = []
        
        # 基于一致性分数的建议
        if consistency_score < 0.5:
            recommendations.append("数据分布一致性很低，强烈建议重新进行数据拆分")
        elif consistency_score < 0.7:
            recommendations.append("数据分布一致性较低，建议检查拆分策略或使用分层抽样")
        elif consistency_score < 0.9:
            recommendations.append("数据分布基本一致，可考虑微调拆分参数")
        
        # 基于统计检验的建议
        significant_tests = []
        for test_type, test_results in statistical_tests.items():
            if isinstance(test_results, dict):
                for field, field_results in test_results.items():
                    if isinstance(field_results, dict):
                        for comparison, result in field_results.items():
                            if isinstance(result, dict) and result.get("significant", False):
                                significant_tests.append(f"{test_type}:{field}:{comparison}")
        
        if significant_tests:
            recommendations.append(f"发现 {len(significant_tests)} 个显著的分布差异，建议检查相关字段")
        
        # 基于数据集大小的建议
        sizes = [stats["size"] for stats in stats_dict.values()]
        if max(sizes) / min(sizes) > 5:
            recommendations.append("数据集大小差异较大，可能影响模型训练效果")
        
        # 基于词汇多样性的建议
        vocab_diversities = []
        for stats in stats_dict.values():
            for field in ["text_stats", "question_stats", "answer_stats"]:
                if field in stats and "vocabulary" in stats[field]:
                    vocab_diversities.append(stats[field]["vocabulary"]["vocabulary_diversity"])
        
        if vocab_diversities and max(vocab_diversities) - min(vocab_diversities) > 0.1:
            recommendations.append("词汇多样性差异较大，建议检查数据质量")
        
        if not recommendations:
            recommendations.append("数据分布质量良好，可以进行模型训练")
        
        return recommendations
    
    def detect_data_leakage(self, splits: Dict[str, Dataset]) -> List[DataIssue]:
        """
        检测数据泄露
        
        Args:
            splits: 拆分后的数据集字典
            
        Returns:
            发现的数据泄露问题列表
        """
        logger.info("开始检测数据泄露")
        
        issues = []
        
        # 检查唯一标识符重复
        if "id" in splits[list(splits.keys())[0]].column_names:
            issues.extend(self._check_id_leakage(splits))
        
        # 检查文本内容重复
        issues.extend(self._check_content_leakage(splits))
        
        # 检查近似重复
        issues.extend(self._check_approximate_leakage(splits))
        
        logger.info(f"数据泄露检测完成，发现 {len(issues)} 个问题")
        
        return issues
    
    def _check_id_leakage(self, splits: Dict[str, Dataset]) -> List[DataIssue]:
        """检查ID重复"""
        issues = []
        split_names = list(splits.keys())
        
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                ids_i = set(splits[split_names[i]]["id"])
                ids_j = set(splits[split_names[j]]["id"])
                
                overlap = ids_i & ids_j
                if overlap:
                    issues.append(DataIssue(
                        issue_type=DataIssueType.DUPLICATE,
                        description=f"在 {split_names[i]} 和 {split_names[j]} 之间发现 {len(overlap)} 个重复ID",
                        affected_samples=list(overlap),
                        severity="high",
                        suggested_action="移除重复样本或重新进行数据拆分"
                    ))
        
        return issues
    
    def _check_content_leakage(self, splits: Dict[str, Dataset]) -> List[DataIssue]:
        """检查内容重复"""
        issues = []
        split_names = list(splits.keys())
        
        # 检查文本字段
        text_fields = ["text", "question", "answer"]
        for field in text_fields:
            if field in splits[split_names[0]].column_names:
                for i in range(len(split_names)):
                    for j in range(i + 1, len(split_names)):
                        texts_i = set(splits[split_names[i]][field])
                        texts_j = set(splits[split_names[j]][field])
                        
                        overlap = texts_i & texts_j
                        if overlap:
                            issues.append(DataIssue(
                                issue_type=DataIssueType.DUPLICATE,
                                description=f"在 {split_names[i]} 和 {split_names[j]} 的 {field} 字段中发现 {len(overlap)} 个重复内容",
                                affected_samples=[],  # 这里需要更复杂的逻辑来找到具体的样本索引
                                severity="medium",
                                suggested_action=f"检查并移除 {field} 字段的重复内容"
                            ))
        
        return issues
    
    def _check_approximate_leakage(self, splits: Dict[str, Dataset]) -> List[DataIssue]:
        """检查近似重复（简化版本）"""
        issues = []
        
        # 这里可以实现更复杂的近似重复检测算法
        # 例如使用编辑距离、余弦相似度等
        
        # 暂时返回空列表，表示未发现近似重复
        return issues
    
    def generate_distribution_report(self, analysis: DistributionAnalysis, 
                                   output_dir: str) -> str:
        """
        生成分布分析报告
        
        Args:
            analysis: 分布分析结果
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / "distribution_analysis_report.html"
        
        # 生成HTML报告
        html_content = self._generate_html_report(analysis)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"分布分析报告已生成: {report_file}")
        
        return str(report_file)
    
    def _generate_html_report(self, analysis: DistributionAnalysis) -> str:
        """生成HTML格式的分析报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>数据分布分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007cba; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>数据分布分析报告</h1>
                <p>一致性分数: <strong>{analysis.consistency_score:.4f}</strong></p>
            </div>
            
            <div class="section">
                <h2>数据集统计信息</h2>
                <h3>训练集</h3>
                <div class="metric">样本数: {analysis.train_stats.get('size', 'N/A')}</div>
                
                <h3>验证集</h3>
                <div class="metric">样本数: {analysis.val_stats.get('size', 'N/A')}</div>
                
                <h3>测试集</h3>
                <div class="metric">样本数: {analysis.test_stats.get('size', 'N/A')}</div>
            </div>
            
            <div class="section">
                <h2>改进建议</h2>
                {''.join([f'<div class="recommendation">{rec}</div>' for rec in analysis.recommendations])}
            </div>
            
            <div class="section">
                <h2>统计检验结果</h2>
                <pre>{analysis.statistical_tests}</pre>
            </div>
        </body>
        </html>
        """
        
        return html
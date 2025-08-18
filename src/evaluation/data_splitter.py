"""
数据拆分器

提供科学的数据拆分功能，支持分层抽样、随机种子控制和数据分布验证。
"""

import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from collections import Counter, defaultdict
import json

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .data_models import DataSplitResult, DistributionAnalysis, DataIssue, DataIssueType
from .quality_analyzer import QualityAnalyzer

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    数据拆分器
    
    负责将数据集科学地拆分为训练集、验证集和测试集，
    确保数据分布的一致性和代表性。
    """
    
    def __init__(self,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 stratify_by: Optional[str] = None,
                 random_seed: int = 42,
                 min_samples_per_split: int = 10,
                 enable_quality_analysis: bool = True):
        """
        初始化数据拆分器
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 测试集比例
            stratify_by: 分层字段名称
            random_seed: 随机种子
            min_samples_per_split: 每个拆分的最小样本数
            enable_quality_analysis: 是否启用质量分析
        """
        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"拆分比例之和必须为1.0，当前为: {train_ratio + val_ratio + test_ratio}")
        
        if any(ratio <= 0 for ratio in [train_ratio, val_ratio, test_ratio]):
            raise ValueError("所有拆分比例必须大于0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify_by = stratify_by
        self.random_seed = random_seed
        self.min_samples_per_split = min_samples_per_split
        self.enable_quality_analysis = enable_quality_analysis
        
        # 初始化质量分析器
        if self.enable_quality_analysis:
            self.quality_analyzer = QualityAnalyzer()
        
        # 设置随机种子
        self._set_random_seeds()
        
        logger.info(f"DataSplitter初始化完成: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        if stratify_by:
            logger.info(f"启用分层抽样，分层字段: {stratify_by}")
        if enable_quality_analysis:
            logger.info("启用数据质量分析")
    
    def _set_random_seeds(self) -> None:
        """设置随机种子确保可重现性"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def split_data(self, dataset: Dataset, output_dir: Optional[str] = None) -> DataSplitResult:
        """
        拆分数据集
        
        Args:
            dataset: 要拆分的数据集
            output_dir: 输出目录，如果提供则保存拆分结果
            
        Returns:
            DataSplitResult: 拆分结果
        """
        logger.info(f"开始拆分数据集，总样本数: {len(dataset)}")
        
        # 数据质量预检查
        if self.enable_quality_analysis:
            logger.info("执行数据质量预检查...")
            quality_report = self.quality_analyzer.analyze_data_quality(dataset, "original_dataset")
            
            # 检查是否有严重的质量问题
            high_severity_issues = [issue for issue in quality_report.issues 
                                  if isinstance(issue, dict) and issue.get("severity") == "high"]
            
            if high_severity_issues:
                logger.warning(f"发现 {len(high_severity_issues)} 个严重的数据质量问题")
                for issue in high_severity_issues:
                    logger.warning(f"  - {issue.get('description', 'Unknown issue')}")
                logger.warning("建议在拆分前先解决这些问题")
            
            logger.info(f"数据质量分数: {quality_report.quality_score:.4f}")
        
        # 验证数据集大小
        if len(dataset) < self.min_samples_per_split * 3:
            raise ValueError(f"数据集太小，至少需要 {self.min_samples_per_split * 3} 个样本")
        
        # 执行拆分
        if self.stratify_by and self.stratify_by in dataset.column_names:
            train_dataset, val_dataset, test_dataset = self._stratified_split(dataset)
        else:
            train_dataset, val_dataset, test_dataset = self._random_split(dataset)
        
        # 验证拆分结果
        self._validate_splits(train_dataset, val_dataset, test_dataset)
        
        # 分析数据分布
        distribution_analysis = self.analyze_distribution({
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        })
        
        # 创建拆分信息
        split_info = {
            "total_samples": len(dataset),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "train_ratio_actual": len(train_dataset) / len(dataset),
            "val_ratio_actual": len(val_dataset) / len(dataset),
            "test_ratio_actual": len(test_dataset) / len(dataset),
            "stratify_by": self.stratify_by,
            "random_seed": self.random_seed,
            "split_method": "stratified" if self.stratify_by else "random",
            "quality_analysis_enabled": self.enable_quality_analysis
        }
        
        # 添加质量分析结果
        if self.enable_quality_analysis:
            split_info["original_quality_score"] = quality_report.quality_score
            split_info["quality_issues_count"] = len(quality_report.issues)
        
        # 创建结果对象
        result = DataSplitResult(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            split_info=split_info,
            distribution_analysis=distribution_analysis
        )
        
        # 保存结果
        if output_dir:
            self.save_splits(result, output_dir)
            
            # 保存质量分析报告
            if self.enable_quality_analysis:
                self.quality_analyzer.generate_quality_report(quality_report, output_dir)
        
        logger.info("数据拆分完成")
        logger.info(f"  训练集: {len(train_dataset)} 样本 ({len(train_dataset)/len(dataset)*100:.1f}%)")
        logger.info(f"  验证集: {len(val_dataset)} 样本 ({len(val_dataset)/len(dataset)*100:.1f}%)")
        logger.info(f"  测试集: {len(test_dataset)} 样本 ({len(test_dataset)/len(dataset)*100:.1f}%)")
        
        return result
    
    def _random_split(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        """
        随机拆分数据集
        
        Args:
            dataset: 要拆分的数据集
            
        Returns:
            训练集、验证集、测试集
        """
        logger.info("使用随机拆分方法")
        
        # 创建索引列表
        indices = list(range(len(dataset)))
        
        # 第一次拆分：分离出测试集
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_ratio,
            random_state=self.random_seed
        )
        
        # 第二次拆分：从训练+验证集中分离出验证集
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_ratio_adjusted,
            random_state=self.random_seed
        )
        
        # 创建子数据集
        train_dataset = dataset.select(train_indices)
        val_dataset = dataset.select(val_indices)
        test_dataset = dataset.select(test_indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def _stratified_split(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        """
        分层拆分数据集
        
        Args:
            dataset: 要拆分的数据集
            
        Returns:
            训练集、验证集、测试集
        """
        logger.info(f"使用分层拆分方法，分层字段: {self.stratify_by}")
        
        # 获取分层标签
        stratify_labels = dataset[self.stratify_by]
        
        # 检查标签分布
        label_counts = Counter(stratify_labels)
        logger.info(f"标签分布: {dict(label_counts)}")
        
        # 检查每个类别是否有足够的样本
        min_samples_needed = max(1, self.min_samples_per_split // len(label_counts))
        insufficient_labels = [label for label, count in label_counts.items() 
                             if count < min_samples_needed * 3]
        
        if insufficient_labels:
            logger.warning(f"以下标签样本数不足，可能影响分层效果: {insufficient_labels}")
        
        # 创建索引列表
        indices = list(range(len(dataset)))
        
        try:
            # 第一次分层拆分：分离出测试集
            train_val_indices, test_indices = train_test_split(
                indices,
                test_size=self.test_ratio,
                stratify=[stratify_labels[i] for i in indices],
                random_state=self.random_seed
            )
            
            # 第二次分层拆分：从训练+验证集中分离出验证集
            val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=val_ratio_adjusted,
                stratify=[stratify_labels[i] for i in train_val_indices],
                random_state=self.random_seed
            )
            
        except ValueError as e:
            logger.warning(f"分层拆分失败，回退到随机拆分: {e}")
            return self._random_split(dataset)
        
        # 创建子数据集
        train_dataset = dataset.select(train_indices)
        val_dataset = dataset.select(val_indices)
        test_dataset = dataset.select(test_indices)
        
        # 验证分层效果
        self._validate_stratification(dataset, train_dataset, val_dataset, test_dataset)
        
        return train_dataset, val_dataset, test_dataset
    
    def _validate_stratification(self, original: Dataset, train: Dataset, 
                               val: Dataset, test: Dataset) -> None:
        """验证分层拆分效果"""
        if not self.stratify_by or self.stratify_by not in original.column_names:
            return
        
        # 计算各数据集的标签分布
        original_dist = Counter(original[self.stratify_by])
        train_dist = Counter(train[self.stratify_by])
        val_dist = Counter(val[self.stratify_by])
        test_dist = Counter(test[self.stratify_by])
        
        # 计算分布差异
        max_diff = 0.0
        for label in original_dist:
            original_ratio = original_dist[label] / len(original)
            train_ratio = train_dist.get(label, 0) / len(train)
            val_ratio = val_dist.get(label, 0) / len(val)
            test_ratio = test_dist.get(label, 0) / len(test)
            
            max_diff = max(max_diff, 
                          abs(original_ratio - train_ratio),
                          abs(original_ratio - val_ratio),
                          abs(original_ratio - test_ratio))
        
        logger.info(f"分层拆分质量: 最大分布差异 = {max_diff:.4f}")
        
        if max_diff > 0.1:  # 10%的差异阈值
            logger.warning(f"分层拆分效果不佳，最大分布差异: {max_diff:.4f}")
    
    def _validate_splits(self, train: Dataset, val: Dataset, test: Dataset) -> None:
        """验证拆分结果"""
        # 检查最小样本数
        if len(train) < self.min_samples_per_split:
            raise ValueError(f"训练集样本数不足: {len(train)} < {self.min_samples_per_split}")
        if len(val) < self.min_samples_per_split:
            raise ValueError(f"验证集样本数不足: {len(val)} < {self.min_samples_per_split}")
        if len(test) < self.min_samples_per_split:
            raise ValueError(f"测试集样本数不足: {len(test)} < {self.min_samples_per_split}")
        
        # 检查数据泄露（如果数据集有唯一标识符）
        if "id" in train.column_names:
            train_ids = set(train["id"])
            val_ids = set(val["id"])
            test_ids = set(test["id"])
            
            if train_ids & val_ids:
                raise ValueError("训练集和验证集存在数据泄露")
            if train_ids & test_ids:
                raise ValueError("训练集和测试集存在数据泄露")
            if val_ids & test_ids:
                raise ValueError("验证集和测试集存在数据泄露")
        
        logger.info("拆分验证通过")
    
    def analyze_distribution(self, splits: Dict[str, Dataset]) -> DistributionAnalysis:
        """
        分析数据分布一致性
        
        Args:
            splits: 拆分后的数据集字典
            
        Returns:
            DistributionAnalysis: 分布分析结果
        """
        logger.info("开始分析数据分布")
        
        # 计算各数据集的统计信息
        stats = {}
        for split_name, dataset in splits.items():
            stats[split_name] = self._calculate_dataset_stats(dataset)
        
        # 计算分布一致性分数
        consistency_score = self._calculate_consistency_score(stats)
        
        # 执行统计检验
        statistical_tests = self._perform_statistical_tests(splits)
        
        # 生成建议
        recommendations = self._generate_distribution_recommendations(
            stats, consistency_score, statistical_tests
        )
        
        analysis = DistributionAnalysis(
            train_stats=stats.get("train", {}),
            val_stats=stats.get("val", {}),
            test_stats=stats.get("test", {}),
            consistency_score=consistency_score,
            statistical_tests=statistical_tests,
            recommendations=recommendations
        )
        
        logger.info(f"分布分析完成，一致性分数: {consistency_score:.4f}")
        
        return analysis
    
    def _calculate_dataset_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """计算数据集统计信息"""
        stats = {
            "size": len(dataset),
            "columns": dataset.column_names
        }
        
        # 文本长度统计
        if "text" in dataset.column_names:
            text_lengths = [len(text) for text in dataset["text"]]
            stats["text_length"] = {
                "mean": np.mean(text_lengths),
                "std": np.std(text_lengths),
                "min": np.min(text_lengths),
                "max": np.max(text_lengths),
                "median": np.median(text_lengths)
            }
        
        # 问题长度统计
        if "question" in dataset.column_names:
            question_lengths = [len(q) for q in dataset["question"]]
            stats["question_length"] = {
                "mean": np.mean(question_lengths),
                "std": np.std(question_lengths),
                "min": np.min(question_lengths),
                "max": np.max(question_lengths),
                "median": np.median(question_lengths)
            }
        
        # 答案长度统计
        if "answer" in dataset.column_names:
            answer_lengths = [len(a) for a in dataset["answer"]]
            stats["answer_length"] = {
                "mean": np.mean(answer_lengths),
                "std": np.std(answer_lengths),
                "min": np.min(answer_lengths),
                "max": np.max(answer_lengths),
                "median": np.median(answer_lengths)
            }
        
        # 来源分布统计
        if "source" in dataset.column_names:
            source_counts = Counter(dataset["source"])
            stats["source_distribution"] = dict(source_counts)
        
        # 分层字段分布统计
        if self.stratify_by and self.stratify_by in dataset.column_names:
            label_counts = Counter(dataset[self.stratify_by])
            stats["stratify_distribution"] = dict(label_counts)
        
        return stats
    
    def _calculate_consistency_score(self, stats: Dict[str, Dict[str, Any]]) -> float:
        """计算分布一致性分数"""
        if len(stats) < 2:
            return 1.0
        
        scores = []
        
        # 比较文本长度分布
        if all("text_length" in stat for stat in stats.values()):
            length_stats = [stat["text_length"] for stat in stats.values()]
            length_score = self._compare_distributions(length_stats, "mean")
            scores.append(length_score)
        
        # 比较来源分布
        if all("source_distribution" in stat for stat in stats.values()):
            source_dists = [stat["source_distribution"] for stat in stats.values()]
            source_score = self._compare_categorical_distributions(source_dists)
            scores.append(source_score)
        
        # 比较分层字段分布
        if all("stratify_distribution" in stat for stat in stats.values()):
            stratify_dists = [stat["stratify_distribution"] for stat in stats.values()]
            stratify_score = self._compare_categorical_distributions(stratify_dists)
            scores.append(stratify_score)
        
        return np.mean(scores) if scores else 1.0
    
    def _compare_distributions(self, distributions: List[Dict[str, float]], 
                             metric: str) -> float:
        """比较数值分布的一致性"""
        values = [dist[metric] for dist in distributions]
        if len(set(values)) == 1:
            return 1.0
        
        # 计算变异系数
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        # 转换为一致性分数 (0-1)
        return max(0, 1 - cv)
    
    def _compare_categorical_distributions(self, distributions: List[Dict[str, int]]) -> float:
        """比较分类分布的一致性"""
        if len(distributions) < 2:
            return 1.0
        
        # 获取所有类别
        all_categories = set()
        for dist in distributions:
            all_categories.update(dist.keys())
        
        # 计算每个分布的比例
        proportions = []
        for dist in distributions:
            total = sum(dist.values())
            prop = {cat: dist.get(cat, 0) / total for cat in all_categories}
            proportions.append(prop)
        
        # 计算KL散度的平均值
        kl_divergences = []
        for i in range(len(proportions)):
            for j in range(i + 1, len(proportions)):
                kl_div = self._calculate_kl_divergence(proportions[i], proportions[j])
                kl_divergences.append(kl_div)
        
        if not kl_divergences:
            return 1.0
        
        avg_kl_div = np.mean(kl_divergences)
        # 转换为一致性分数
        return max(0, 1 - avg_kl_div)
    
    def _calculate_kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """计算KL散度"""
        kl_div = 0.0
        for key in p:
            if p[key] > 0 and q[key] > 0:
                kl_div += p[key] * np.log(p[key] / q[key])
        return kl_div
    
    def _perform_statistical_tests(self, splits: Dict[str, Dataset]) -> Dict[str, Any]:
        """执行统计检验"""
        tests = {}
        
        # 这里可以添加更多的统计检验
        # 例如：Kolmogorov-Smirnov检验、卡方检验等
        
        tests["performed"] = ["basic_comparison"]
        tests["note"] = "详细的统计检验功能待实现"
        
        return tests
    
    def _generate_distribution_recommendations(self, stats: Dict[str, Dict[str, Any]], 
                                             consistency_score: float,
                                             statistical_tests: Dict[str, Any]) -> List[str]:
        """生成分布改进建议"""
        recommendations = []
        
        if consistency_score < 0.8:
            recommendations.append("数据分布一致性较低，建议检查数据拆分策略")
        
        if consistency_score < 0.6:
            recommendations.append("考虑使用分层抽样或调整拆分比例")
        
        # 检查数据集大小差异
        sizes = [stat["size"] for stat in stats.values()]
        if max(sizes) / min(sizes) > 10:
            recommendations.append("数据集大小差异过大，可能影响模型性能")
        
        if not recommendations:
            recommendations.append("数据分布良好，无需特殊调整")
        
        return recommendations
    
    def save_splits(self, result: DataSplitResult, output_dir: str) -> None:
        """
        保存拆分结果
        
        Args:
            result: 拆分结果
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存数据集
        result.train_dataset.save_to_disk(output_path / "train")
        result.val_dataset.save_to_disk(output_path / "val")
        result.test_dataset.save_to_disk(output_path / "test")
        
        # 保存拆分信息
        result.save_info(output_path / "split_info.json")
        
        logger.info(f"拆分结果已保存到: {output_path}")
    
    def load_splits(self, input_dir: str) -> DataSplitResult:
        """
        加载拆分结果
        
        Args:
            input_dir: 输入目录
            
        Returns:
            DataSplitResult: 拆分结果
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"拆分结果目录不存在: {input_path}")
        
        # 加载数据集
        train_dataset = Dataset.load_from_disk(input_path / "train")
        val_dataset = Dataset.load_from_disk(input_path / "val")
        test_dataset = Dataset.load_from_disk(input_path / "test")
        
        # 加载拆分信息
        with open(input_path / "split_info.json", 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        # 重建DistributionAnalysis对象
        dist_analysis = DistributionAnalysis.from_dict(info["distribution_analysis"])
        
        result = DataSplitResult(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            split_info=info["split_info"],
            distribution_analysis=dist_analysis
        )
        
        logger.info(f"拆分结果已从 {input_path} 加载")
        
        return result

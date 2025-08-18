"""
统计分析器

提供数据的统计分析功能，包括描述性统计、假设检验、相关性分析等。
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    统计分析器
    
    提供数据的统计分析功能，包括描述性统计、假设检验、相关性分析等。
    """
    
    def __init__(self):
        """初始化统计分析器"""
        self.logger = logging.getLogger(__name__)
    
    def descriptive_statistics(self, data: List[Union[int, float]]) -> Dict[str, float]:
        """
        计算描述性统计
        
        Args:
            data: 数据列表
            
        Returns:
            描述性统计结果字典
        """
        if not data:
            return {}
        
        np_data = np.array(data)
        
        stats_dict = {
            "count": len(np_data),
            "mean": float(np.mean(np_data)),
            "median": float(np.median(np_data)),
            "std": float(np.std(np_data, ddof=1)),
            "var": float(np.var(np_data, ddof=1)),
            "min": float(np.min(np_data)),
            "max": float(np.max(np_data)),
            "range": float(np.max(np_data) - np.min(np_data)),
            "q1": float(np.percentile(np_data, 25)),
            "q3": float(np.percentile(np_data, 75)),
            "iqr": float(np.percentile(np_data, 75) - np.percentile(np_data, 25))
        }
        
        # 计算偏度和峰度
        try:
            stats_dict["skewness"] = float(stats.skew(np_data))
            stats_dict["kurtosis"] = float(stats.kurtosis(np_data))
        except Exception as e:
            self.logger.warning(f"计算偏度和峰度失败: {e}")
            stats_dict["skewness"] = 0.0
            stats_dict["kurtosis"] = 0.0
        
        return stats_dict
    
    def correlation_analysis(self, x: List[Union[int, float]], 
                           y: List[Union[int, float]]) -> Dict[str, float]:
        """
        相关性分析
        
        Args:
            x: 第一个变量数据
            y: 第二个变量数据
            
        Returns:
            相关性分析结果字典
        """
        if len(x) != len(y) or len(x) < 2:
            return {}
        
        try:
            # 皮尔逊相关系数
            pearson_corr, pearson_p = stats.pearsonr(x, y)
            
            # 斯皮尔曼相关系数
            spearman_corr, spearman_p = stats.spearmanr(x, y)
            
            # 肯德尔相关系数
            kendall_corr, kendall_p = stats.kendalltau(x, y)
            
            return {
                "pearson_correlation": float(pearson_corr),
                "pearson_p_value": float(pearson_p),
                "spearman_correlation": float(spearman_corr),
                "spearman_p_value": float(spearman_p),
                "kendall_correlation": float(kendall_corr),
                "kendall_p_value": float(kendall_p)
            }
        except Exception as e:
            self.logger.warning(f"相关性分析失败: {e}")
            return {}
    
    def normality_test(self, data: List[Union[int, float]]) -> Dict[str, Any]:
        """
        正态性检验
        
        Args:
            data: 数据列表
            
        Returns:
            正态性检验结果字典
        """
        if len(data) < 3:
            return {}
        
        try:
            # Shapiro-Wilk检验（适用于小样本）
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(data)
            else:
                shapiro_stat, shapiro_p = None, None
            
            # Kolmogorov-Smirnov检验
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            
            # D'Agostino's K-squared检验
            k2_stat, k2_p = stats.normaltest(data)
            
            result = {
                "sample_size": len(data),
                "shapiro_wilk": {
                    "statistic": float(shapiro_stat) if shapiro_stat is not None else None,
                    "p_value": float(shapiro_p) if shapiro_p is not None else None,
                    "is_normal": bool(shapiro_p > 0.05) if shapiro_p is not None else None
                } if shapiro_stat is not None else None,
                "kolmogorov_smirnov": {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                    "is_normal": bool(ks_p > 0.05)
                },
                "dagostino_k2": {
                    "statistic": float(k2_stat),
                    "p_value": float(k2_p),
                    "is_normal": bool(k2_p > 0.05)
                }
            }
            
            return result
        except Exception as e:
            self.logger.warning(f"正态性检验失败: {e}")
            return {}
    
    def t_test(self, sample1: List[Union[int, float]], 
              sample2: List[Union[int, float]] = None,
              popmean: float = None) -> Dict[str, Any]:
        """
        t检验
        
        Args:
            sample1: 第一个样本数据
            sample2: 第二个样本数据（双样本t检验）
            popmean: 总体均值（单样本t检验）
            
        Returns:
            t检验结果字典
        """
        try:
            if sample2 is None and popmean is not None:
                # 单样本t检验
                t_stat, p_value = stats.ttest_1samp(sample1, popmean)
                test_type = "one_sample"
                df = len(sample1) - 1
            elif sample2 is not None:
                # 双样本t检验
                t_stat, p_value = stats.ttest_ind(sample1, sample2)
                test_type = "two_sample"
                df = len(sample1) + len(sample2) - 2
            else:
                return {}
            
            return {
                "test_type": test_type,
                "statistic": float(t_stat),
                "p_value": float(p_value),
                "degrees_of_freedom": int(df),
                "is_significant": bool(p_value < 0.05)
            }
        except Exception as e:
            self.logger.warning(f"t检验失败: {e}")
            return {}
    
    def anova_test(self, *samples: List[Union[int, float]]) -> Dict[str, Any]:
        """
        方差分析（ANOVA）
        
        Args:
            samples: 多个样本数据
            
        Returns:
            ANOVA检验结果字典
        """
        if len(samples) < 2:
            return {}
        
        try:
            f_stat, p_value = stats.f_oneway(*samples)
            
            return {
                "test_type": "one_way_anova",
                "statistic": float(f_stat),
                "p_value": float(p_value),
                "is_significant": bool(p_value < 0.05),
                "num_groups": len(samples)
            }
        except Exception as e:
            self.logger.warning(f"ANOVA检验失败: {e}")
            return {}
    
    def chi_square_test(self, observed: List[int], 
                       expected: List[int] = None) -> Dict[str, Any]:
        """
        卡方检验
        
        Args:
            observed: 观测频数
            expected: 期望频数（如果为None，则假设均匀分布）
            
        Returns:
            卡方检验结果字典
        """
        try:
            if expected is None:
                # 拟合优度检验
                chi2_stat, p_value = stats.chisquare(observed)
                test_type = "goodness_of_fit"
                df = len(observed) - 1
            else:
                # 独立性检验
                chi2_stat, p_value, df, _ = stats.chi2_contingency([observed, expected])
                test_type = "independence"
            
            return {
                "test_type": test_type,
                "statistic": float(chi2_stat),
                "p_value": float(p_value),
                "degrees_of_freedom": int(df),
                "is_significant": bool(p_value < 0.05)
            }
        except Exception as e:
            self.logger.warning(f"卡方检验失败: {e}")
            return {}
    
    def outlier_detection(self, data: List[Union[int, float]], 
                         method: str = "iqr") -> Dict[str, Any]:
        """
        异常值检测
        
        Args:
            data: 数据列表
            method: 检测方法（"iqr"或"zscore"）
            
        Returns:
            异常值检测结果字典
        """
        if not data:
            return {}
        
        try:
            np_data = np.array(data)
            
            if method == "iqr":
                # IQR方法
                q1 = np.percentile(np_data, 25)
                q3 = np.percentile(np_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = np_data[(np_data < lower_bound) | (np_data > upper_bound)]
                
                return {
                    "method": "iqr",
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outliers": [float(x) for x in outliers],
                    "outlier_count": len(outliers),
                    "outlier_percentage": float(len(outliers) / len(np_data) * 100)
                }
            
            elif method == "zscore":
                # Z-score方法
                z_scores = np.abs(stats.zscore(np_data))
                outliers = np_data[z_scores > 3]
                
                return {
                    "method": "zscore",
                    "threshold": 3.0,
                    "outliers": [float(x) for x in outliers],
                    "outlier_count": len(outliers),
                    "outlier_percentage": float(len(outliers) / len(np_data) * 100)
                }
            
            else:
                return {}
        except Exception as e:
            self.logger.warning(f"异常值检测失败: {e}")
            return {}
    
    def confidence_interval(self, data: List[Union[int, float]], 
                           confidence: float = 0.95) -> Dict[str, float]:
        """
        计算置信区间
        
        Args:
            data: 数据列表
            confidence: 置信水平
            
        Returns:
            置信区间结果字典
        """
        if len(data) < 2:
            return {}
        
        try:
            np_data = np.array(data)
            mean = np.mean(np_data)
            std_err = stats.sem(np_data)
            
            # 计算置信区间
            ci = stats.t.interval(confidence, len(np_data) - 1, loc=mean, scale=std_err)
            
            return {
                "confidence_level": confidence,
                "mean": float(mean),
                "standard_error": float(std_err),
                "lower_bound": float(ci[0]),
                "upper_bound": float(ci[1]),
                "margin_of_error": float(ci[1] - mean)
            }
        except Exception as e:
            self.logger.warning(f"置信区间计算失败: {e}")
            return {}
    
    def comprehensive_analysis(self, data: Dict[str, List[Union[int, float]]]) -> Dict[str, Any]:
        """
        综合统计分析
        
        Args:
            data: 数据字典（变量名 -> 数据列表）
            
        Returns:
            综合统计分析结果
        """
        results = {
            "variables": list(data.keys()),
            "descriptive_stats": {},
            "normality_tests": {},
            "correlations": {},
            "outliers": {}
        }
        
        # 描述性统计和正态性检验
        for var_name, var_data in data.items():
            if var_data:
                results["descriptive_stats"][var_name] = self.descriptive_statistics(var_data)
                results["normality_tests"][var_name] = self.normality_test(var_data)
                results["outliers"][var_name] = self.outlier_detection(var_data)
        
        # 相关性分析
        var_names = list(data.keys())
        for i in range(len(var_names)):
            for j in range(i + 1, len(var_names)):
                var1, var2 = var_names[i], var_names[j]
                if len(data[var1]) == len(data[var2]):
                    corr_result = self.correlation_analysis(data[var1], data[var2])
                    if corr_result:
                        results["correlations"][f"{var1}_vs_{var2}"] = corr_result
        
        return results

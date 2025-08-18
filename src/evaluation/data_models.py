"""
数据模型定义

定义评估系统中使用的核心数据结构和配置类。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from datasets import Dataset
import json
import numpy as np


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@dataclass
class DataSplitResult:
    """数据拆分结果"""
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    split_info: Dict[str, Any]
    distribution_analysis: 'DistributionAnalysis'
    
    def save_info(self, output_path: str) -> None:
        """保存拆分信息到文件"""
        info = convert_numpy_types({
            "split_info": self.split_info,
            "distribution_analysis": self.distribution_analysis.to_dict(),
            "train_size": int(len(self.train_dataset)),
            "val_size": int(len(self.val_dataset)),
            "test_size": int(len(self.test_dataset)),
            "created_at": datetime.now().isoformat()
        })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)


@dataclass
class DistributionAnalysis:
    """数据分布分析结果"""
    train_stats: Dict[str, Any]
    val_stats: Dict[str, Any]
    test_stats: Dict[str, Any]
    consistency_score: float  # 分布一致性分数 (0-1)
    statistical_tests: Dict[str, Any]  # 统计检验结果
    recommendations: List[str]  # 改进建议
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return convert_numpy_types({
            "train_stats": self.train_stats,
            "val_stats": self.val_stats,
            "test_stats": self.test_stats,
            "consistency_score": self.consistency_score,
            "statistical_tests": self.statistical_tests,
            "recommendations": self.recommendations
        })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributionAnalysis':
        """从字典创建对象"""
        return cls(
            train_stats=data.get("train_stats", {}),
            val_stats=data.get("val_stats", {}),
            test_stats=data.get("test_stats", {}),
            consistency_score=data.get("consistency_score", 0.0),
            statistical_tests=data.get("statistical_tests", {}),
            recommendations=data.get("recommendations", [])
        )


@dataclass
class EvaluationSample:
    """单个评估样本"""
    input_text: str
    prediction: str
    reference: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """单个任务评估结果"""
    task_name: str
    predictions: List[str]
    references: List[str]
    metrics: Dict[str, float]
    samples: List[EvaluationSample]
    execution_time: float
    
    def get_summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
        return convert_numpy_types({
            "task_name": self.task_name,
            "num_samples": len(self.samples),
            "metrics": self.metrics,
            "execution_time": self.execution_time
        })


@dataclass
class EfficiencyMetrics:
    """效率指标"""
    inference_latency: float  # 平均推理延迟(ms)
    throughput: float  # 吞吐量(tokens/s)
    memory_usage: float  # 内存使用(GB)
    model_size: float  # 模型大小(MB)
    flops: Optional[int] = None  # 浮点运算次数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return convert_numpy_types({
            "inference_latency_ms": self.inference_latency,
            "throughput_tokens_per_sec": self.throughput,
            "memory_usage_gb": self.memory_usage,
            "model_size_mb": self.model_size,
            "flops": self.flops
        })


@dataclass
class QualityScores:
    """质量分数"""
    fluency: float  # 流畅度 (0-1)
    coherence: float  # 连贯性 (0-1)
    relevance: float  # 相关性 (0-1)
    factuality: float  # 事实性 (0-1)
    overall: float  # 总体质量 (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return convert_numpy_types({
            "fluency": self.fluency,
            "coherence": self.coherence,
            "relevance": self.relevance,
            "factuality": self.factuality,
            "overall": self.overall
        })


@dataclass
class EvaluationResult:
    """评估结果"""
    model_name: str
    evaluation_time: datetime
    metrics: Dict[str, float]
    task_results: Dict[str, TaskResult]
    efficiency_metrics: EfficiencyMetrics
    quality_scores: QualityScores
    config: 'EvaluationConfig'
    
    def get_summary(self) -> Dict[str, Any]:
        """获取评估结果摘要"""
        return convert_numpy_types({
            "model_name": self.model_name,
            "evaluation_time": self.evaluation_time.isoformat(),
            "overall_metrics": self.metrics,
            "task_summaries": {name: result.get_summary() 
                             for name, result in self.task_results.items()},
            "efficiency": self.efficiency_metrics.to_dict(),
            "quality": self.quality_scores.to_dict()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return convert_numpy_types({
            "model_name": self.model_name,
            "evaluation_time": self.evaluation_time.isoformat(),
            "metrics": self.metrics,
            "task_results": {name: {
                "task_name": result.task_name,
                "predictions": result.predictions,
                "references": result.references,
                "metrics": result.metrics,
                "execution_time": result.execution_time,
                "num_samples": len(result.samples)
            } for name, result in self.task_results.items()},
            "efficiency_metrics": self.efficiency_metrics.to_dict(),
            "quality_scores": self.quality_scores.to_dict(),
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        })


@dataclass
class EvaluationConfig:
    """评估配置"""
    tasks: List[str] = field(default_factory=lambda: ["text_generation", "question_answering"])
    metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge", "bertscore"])
    batch_size: int = 8
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    num_samples: int = 100  # 每个任务的样本数
    enable_efficiency_metrics: bool = True
    enable_quality_analysis: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return convert_numpy_types({
            "tasks": self.tasks,
            "metrics": self.metrics,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_samples": self.num_samples,
            "enable_efficiency_metrics": self.enable_efficiency_metrics,
            "enable_quality_analysis": self.enable_quality_analysis
        })


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    name: str
    dataset_path: str
    tasks: List[str]
    evaluation_protocol: str
    metrics: List[str]
    max_samples: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return convert_numpy_types({
            "name": self.name,
            "dataset_path": self.dataset_path,
            "tasks": self.tasks,
            "evaluation_protocol": self.evaluation_protocol,
            "metrics": self.metrics,
            "max_samples": self.max_samples
        })


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    benchmark_name: str
    model_name: str
    evaluation_time: datetime
    task_results: Dict[str, TaskResult]
    overall_score: float
    ranking_info: Dict[str, Any]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取基准测试结果摘要"""
        return convert_numpy_types({
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "evaluation_time": self.evaluation_time.isoformat(),
            "overall_score": self.overall_score,
            "task_scores": {name: result.metrics for name, result in self.task_results.items()},
            "ranking_info": self.ranking_info
        })


@dataclass
class DataQualityReport:
    """数据质量报告"""
    dataset_name: str
    total_samples: int
    quality_score: float  # 总体质量分数 (0-1)
    statistics: Dict[str, Any]  # 基本统计信息
    issues: List[Dict[str, Any]]  # 发现的问题
    recommendations: List[str]  # 改进建议
    analysis_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return convert_numpy_types({
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "quality_score": self.quality_score,
            "statistics": self.statistics,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "analysis_time": self.analysis_time.isoformat()
        })


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    evaluation_config: EvaluationConfig
    data_config: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return convert_numpy_types({
            "experiment_name": self.experiment_name,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "evaluation_config": self.evaluation_config.to_dict(),
            "data_config": self.data_config,
            "tags": self.tags,
            "description": self.description
        })


@dataclass
class ComparisonResult:
    """模型对比结果"""
    models: List[str]
    metrics: Dict[str, List[float]]  # 每个指标的所有模型结果
    statistical_tests: Dict[str, Any]  # 统计显著性检验结果
    rankings: Dict[str, List[str]]  # 每个指标的模型排名
    best_model: Dict[str, str]  # 每个指标的最佳模型
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return convert_numpy_types({
            "models": self.models,
            "metrics": self.metrics,
            "statistical_tests": self.statistical_tests,
            "rankings": self.rankings,
            "best_model": self.best_model
        })


# 数据问题类型枚举
class DataIssueType:
    DUPLICATE = "duplicate"
    EMPTY_CONTENT = "empty_content"
    LENGTH_OUTLIER = "length_outlier"
    ENCODING_ERROR = "encoding_error"
    FORMAT_ERROR = "format_error"
    QUALITY_LOW = "quality_low"


@dataclass
class DataIssue:
    """数据问题"""
    issue_type: str
    description: str
    affected_samples: List[int]  # 受影响的样本索引
    severity: str  # "low", "medium", "high"
    suggested_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return convert_numpy_types({
            "issue_type": self.issue_type,
            "description": self.description,
            "affected_samples": self.affected_samples,
            "severity": self.severity,
            "suggested_action": self.suggested_action
        })

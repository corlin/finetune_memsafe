"""
核心数据模型定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import json


class EvaluationStatus(Enum):
    """评估状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorType(Enum):
    """错误类型枚举"""
    KNOWLEDGE_ERROR = "knowledge_error"
    TERMINOLOGY_ERROR = "terminology_error"
    REASONING_ERROR = "reasoning_error"
    CONTEXT_ERROR = "context_error"
    FORMAT_ERROR = "format_error"
    LOGIC_ERROR = "logic_error"


@dataclass
class EvaluationConfig:
    """评估配置"""
    industry_domain: str  # 行业领域
    evaluation_dimensions: List[str]  # 评估维度
    weight_config: Dict[str, float]  # 各维度权重
    threshold_config: Dict[str, float]  # 阈值配置
    expert_review_required: bool = False  # 是否需要专家复核
    batch_size: int = 10  # 批处理大小
    timeout: int = 30  # 超时时间（秒）
    max_retries: int = 3  # 最大重试次数
    
    def __post_init__(self):
        """数据验证"""
        if not self.industry_domain:
            raise ValueError("行业领域不能为空")
        
        if not self.evaluation_dimensions:
            raise ValueError("评估维度不能为空")
        
        # 验证权重配置
        if self.weight_config:
            total_weight = sum(self.weight_config.values())
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(f"权重总和应为1.0，当前为{total_weight}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "industry_domain": self.industry_domain,
            "evaluation_dimensions": self.evaluation_dimensions,
            "weight_config": self.weight_config,
            "threshold_config": self.threshold_config,
            "expert_review_required": self.expert_review_required,
            "batch_size": self.batch_size,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        """从字典创建"""
        return cls(**data)


@dataclass
class EvaluationScore:
    """评估分数"""
    overall_score: float  # 总分
    dimension_scores: Dict[str, float]  # 各维度分数
    confidence: float = 1.0  # 置信度
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息
    
    def __post_init__(self):
        """数据验证"""
        if not (0 <= self.overall_score <= 1):
            raise ValueError("总分应在0-1之间")
        
        if not (0 <= self.confidence <= 1):
            raise ValueError("置信度应在0-1之间")
        
        for dim, score in self.dimension_scores.items():
            if not (0 <= score <= 1):
                raise ValueError(f"维度{dim}的分数应在0-1之间")


@dataclass
class SampleResult:
    """样本评估结果"""
    sample_id: str
    input_text: str
    model_output: str
    expected_output: str
    dimension_scores: Dict[str, float]
    error_types: List[str] = field(default_factory=list)
    explanation: str = ""
    processing_time: float = 0.0  # 处理时间（秒）
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_overall_score(self, weights: Dict[str, float]) -> float:
        """计算总分"""
        if not weights:
            return sum(self.dimension_scores.values()) / len(self.dimension_scores)
        
        total_score = 0.0
        for dim, score in self.dimension_scores.items():
            weight = weights.get(dim, 0.0)
            total_score += score * weight
        
        return total_score
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "sample_id": self.sample_id,
            "input_text": self.input_text,
            "model_output": self.model_output,
            "expected_output": self.expected_output,
            "dimension_scores": self.dimension_scores,
            "error_types": self.error_types,
            "explanation": self.explanation,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ErrorAnalysis:
    """错误分析"""
    error_distribution: Dict[str, int]  # 错误类型分布
    common_patterns: List[str]  # 常见错误模式
    severity_levels: Dict[str, str]  # 严重程度
    improvement_areas: List[str]  # 改进领域
    
    def get_total_errors(self) -> int:
        """获取错误总数"""
        return sum(self.error_distribution.values())
    
    def get_error_rate(self, total_samples: int) -> float:
        """获取错误率"""
        if total_samples == 0:
            return 0.0
        return self.get_total_errors() / total_samples


@dataclass
class EvaluationResult:
    """评估结果"""
    task_id: str
    model_id: str
    overall_score: float  # 综合评分
    dimension_scores: Dict[str, float]  # 各维度得分
    detailed_results: List[SampleResult]  # 详细结果
    error_analysis: ErrorAnalysis  # 错误分析
    improvement_suggestions: List[str]  # 改进建议
    evaluation_config: EvaluationConfig  # 评估配置
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: EvaluationStatus = EvaluationStatus.PENDING
    
    def get_duration(self) -> float:
        """获取评估耗时（秒）"""
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    def get_sample_count(self) -> int:
        """获取样本数量"""
        return len(self.detailed_results)
    
    def get_pass_rate(self, threshold: float = 0.6) -> float:
        """获取通过率"""
        if not self.detailed_results:
            return 0.0
        
        passed = sum(1 for result in self.detailed_results 
                    if result.get_overall_score(self.evaluation_config.weight_config) >= threshold)
        return passed / len(self.detailed_results)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "detailed_results": [result.to_dict() for result in self.detailed_results],
            "error_analysis": {
                "error_distribution": self.error_analysis.error_distribution,
                "common_patterns": self.error_analysis.common_patterns,
                "severity_levels": self.error_analysis.severity_levels,
                "improvement_areas": self.error_analysis.improvement_areas
            },
            "improvement_suggestions": self.improvement_suggestions,
            "evaluation_config": self.evaluation_config.to_dict(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "duration": self.get_duration(),
            "sample_count": self.get_sample_count(),
            "pass_rate": self.get_pass_rate()
        }


@dataclass
class DataSample:
    """数据样本"""
    sample_id: str
    input_text: str
    expected_output: str
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """验证样本数据"""
        if not self.sample_id:
            return False
        if not self.input_text.strip():
            return False
        if not self.expected_output.strip():
            return False
        return True


@dataclass
class Dataset:
    """数据集"""
    name: str
    samples: List[DataSample]
    industry_domain: str
    description: str = ""
    version: str = "1.0"
    created_time: datetime = field(default_factory=datetime.now)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def validate(self) -> bool:
        """验证数据集"""
        if not self.name or not self.industry_domain:
            return False
        
        if not self.samples:
            return False
        
        return all(sample.validate() for sample in self.samples)
    
    def get_sample_by_id(self, sample_id: str) -> Optional[DataSample]:
        """根据ID获取样本"""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "input_text": sample.input_text,
                    "expected_output": sample.expected_output,
                    "context": sample.context,
                    "metadata": sample.metadata
                }
                for sample in self.samples
            ],
            "industry_domain": self.industry_domain,
            "description": self.description,
            "version": self.version,
            "created_time": self.created_time.isoformat(),
            "sample_count": len(self.samples)
        }


@dataclass
class ProgressInfo:
    """进度信息"""
    task_id: str
    current_step: int
    total_steps: int
    current_sample: int
    total_samples: int
    status: EvaluationStatus
    message: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    estimated_remaining_time: Optional[float] = None  # 预估剩余时间（秒）
    
    def get_progress_percentage(self) -> float:
        """获取进度百分比"""
        if self.total_samples == 0:
            return 0.0
        return (self.current_sample / self.total_samples) * 100
    
    def get_elapsed_time(self) -> float:
        """获取已用时间（秒）"""
        return (datetime.now() - self.start_time).total_seconds()


@dataclass
class Criterion:
    """评估标准"""
    name: str
    description: str
    weight: float
    threshold: float
    evaluation_method: str
    
    def validate(self) -> bool:
        """验证标准"""
        if not self.name or not self.description:
            return False
        if not (0 <= self.weight <= 1):
            return False
        if not (0 <= self.threshold <= 1):
            return False
        return True


@dataclass
class Explanation:
    """结果解释"""
    summary: str
    details: Dict[str, Any]
    confidence: float
    reasoning_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "summary": self.summary,
            "details": self.details,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps
        }


@dataclass
class Report:
    """评估报告"""
    title: str
    evaluation_result: EvaluationResult
    generated_time: datetime = field(default_factory=datetime.now)
    format_type: str = "html"
    content: str = ""
    charts: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "evaluation_result": self.evaluation_result.to_dict(),
            "generated_time": self.generated_time.isoformat(),
            "format_type": self.format_type,
            "content": self.content,
            "charts": self.charts
        }
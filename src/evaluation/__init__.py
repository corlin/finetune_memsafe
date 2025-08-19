"""
数据拆分和模型性能评估模块

提供科学的数据拆分、标准化的模型评估指标计算、基准测试管理和实验跟踪功能。
"""

# Import only existing modules
from .data_splitter import DataSplitter, DataSplitResult, DistributionAnalysis
from .quality_analyzer import QualityAnalyzer, DataQualityReport
from .distribution_analyzer import DistributionAnalyzer
from .config_manager import ConfigManager
from .data_models import (
    DataSplitResult, DistributionAnalysis, EvaluationSample, TaskResult,
    EfficiencyMetrics, QualityScores, EvaluationResult, EvaluationConfig,
    BenchmarkConfig, BenchmarkResult, DataQualityReport, ExperimentConfig,
    ComparisonResult, DataIssue, DataIssueType, ValidationResult, 
    ProcessedBatch, FieldDetectionResult
)

# Import newly implemented modules
from .metrics_calculator import MetricsCalculator
from .efficiency_analyzer import EfficiencyAnalyzer
from .evaluation_engine import EvaluationEngine

# Import new data processing modules
from .data_field_detector import DataFieldDetector
from .batch_data_validator import BatchDataValidator
from .field_mapper import FieldMapper
from .error_handling_strategy import ErrorHandlingStrategy
from .data_preprocessor import DataPreprocessor
from .diagnostic_logger import DiagnosticLogger
from .config_validator import ConfigValidator
from .config_loader import ConfigLoader, load_evaluation_config, validate_config_file
from .compatibility import EvaluationEngineWrapper, create_enhanced_evaluation_engine, migrate_legacy_evaluation
from .task_evaluators import (
    TextGenerationEvaluator, QuestionAnsweringEvaluator, 
    SemanticSimilarityEvaluator, ClassificationEvaluator,
    TaskEvaluatorFactory
)
from .benchmark_manager import BenchmarkManager
from .evaluation_protocols import (
    CLUEProtocol, FewCLUEProtocol, CEvalProtocol,
    EvaluationProtocolFactory
)
from .experiment_tracker import ExperimentTracker
from .statistical_analyzer import StatisticalAnalyzer
from .report_generator import ReportGenerator
from .training_monitor import TrainingMonitor, EarlyStoppingCallback
from .data_pipeline_integration import DataPipelineIntegration, create_enhanced_data_pipeline
from .training_engine_integration import TrainingEngineIntegration, create_enhanced_training_engine
from .inference_tester_integration import InferenceTesterIntegration, create_enhanced_inference_tester

__version__ = "1.0.0"
__all__ = [
    "DataSplitter",
    "DataSplitResult", 
    "DistributionAnalysis",
    "QualityAnalyzer",
    "DataQualityReport",
    "DistributionAnalyzer",
    "ConfigManager",
    "EvaluationSample",
    "TaskResult",
    "EfficiencyMetrics",
    "QualityScores",
    "EvaluationResult",
    "EvaluationConfig",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ExperimentConfig",
    "ComparisonResult",
    "DataIssue",
    "DataIssueType",
    "ValidationResult",
    "ProcessedBatch", 
    "FieldDetectionResult",
    # Newly implemented modules
    "MetricsCalculator",
    "EfficiencyAnalyzer",
    "EvaluationEngine",
    # New data processing modules
    "DataFieldDetector",
    "BatchDataValidator", 
    "FieldMapper",
    "ErrorHandlingStrategy",
    "DataPreprocessor",
    "DiagnosticLogger",
    "ConfigValidator",
    "ConfigLoader",
    "load_evaluation_config",
    "validate_config_file",
    "EvaluationEngineWrapper",
    "create_enhanced_evaluation_engine",
    "migrate_legacy_evaluation",
    "TextGenerationEvaluator",
    "QuestionAnsweringEvaluator",
    "SemanticSimilarityEvaluator",
    "ClassificationEvaluator",
    "TaskEvaluatorFactory",
    "BenchmarkManager",
    "CLUEProtocol",
    "FewCLUEProtocol",
    "CEvalProtocol",
    "EvaluationProtocolFactory",
    "ExperimentTracker",
    "StatisticalAnalyzer",
    "ReportGenerator",
    "TrainingMonitor",
    "EarlyStoppingCallback",
    "DataPipelineIntegration",
    "create_enhanced_data_pipeline",
    "TrainingEngineIntegration", 
    "create_enhanced_training_engine",
    "InferenceTesterIntegration",
    "create_enhanced_inference_tester"
]
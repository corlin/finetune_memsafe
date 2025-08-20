"""
错误分析模块
"""

from .error_analyzer import (
    ErrorAnalysisEngine,
    ErrorClassifier,
    ErrorStatisticsAnalyzer,
    ErrorInstance,
    ErrorPattern,
    ErrorSeverity
)

from .improvement_generator import (
    ImprovementSuggestionGenerator,
    ImprovementSuggestion,
    SuggestionPriority,
    SuggestionCategory,
    SuggestionTemplateManager
)

__all__ = [
    'ErrorAnalysisEngine',
    'ErrorClassifier', 
    'ErrorStatisticsAnalyzer',
    'ErrorInstance',
    'ErrorPattern',
    'ErrorSeverity',
    'ImprovementSuggestionGenerator',
    'ImprovementSuggestion',
    'SuggestionPriority',
    'SuggestionCategory',
    'SuggestionTemplateManager'
]
"""
向后兼容性包装器

确保现有代码能够无缝使用新的数据处理功能。
"""

import logging
from typing import Dict, List, Any, Optional
from datasets import Dataset

from .evaluation_engine import EvaluationEngine as NewEvaluationEngine
from .data_models import EvaluationConfig
from .config_loader import load_evaluation_config

logger = logging.getLogger(__name__)


class EvaluationEngineWrapper(NewEvaluationEngine):
    """
    评估引擎包装器
    
    提供向后兼容的接口，同时集成新的数据处理功能。
    """
    
    def __init__(self, 
                 config: Optional[EvaluationConfig] = None,
                 device: str = "auto",
                 max_workers: int = 4,
                 enable_enhanced_processing: bool = True):
        """
        初始化评估引擎包装器
        
        Args:
            config: 评估配置
            device: 计算设备
            max_workers: 最大工作线程数
            enable_enhanced_processing: 是否启用增强的数据处理功能
        """
        # 如果没有提供配置，使用默认配置
        if config is None:
            config = EvaluationConfig()
        
        # 如果启用增强处理但配置中没有数据处理设置，添加默认设置
        if enable_enhanced_processing and not hasattr(config, 'data_processing'):
            config.data_processing = {
                "field_mapping": {
                    "text_generation": {
                        "input_fields": ["text", "input", "prompt"],
                        "target_fields": ["target", "answer", "output"]
                    },
                    "question_answering": {
                        "input_fields": ["question", "query"],
                        "context_fields": ["context", "passage"],
                        "target_fields": ["answer", "target"]
                    }
                },
                "validation": {
                    "min_valid_samples_ratio": 0.1,
                    "enable_data_cleaning": True,
                    "enable_fallback": True
                },
                "diagnostics": {
                    "enable_detailed_logging": False,
                    "log_batch_statistics": True,
                    "save_processing_report": True
                }
            }
        
        super().__init__(config, device, max_workers)
        self.enable_enhanced_processing = enable_enhanced_processing
        
        if enable_enhanced_processing:
            logger.info("评估引擎已启用增强数据处理功能")
        else:
            logger.info("评估引擎使用传统数据处理模式")
    
    def evaluate_model_with_diagnostics(self, 
                                      model,
                                      tokenizer,
                                      datasets: Dict[str, Dataset],
                                      model_name: str = "unknown_model",
                                      save_diagnostics: bool = True) -> Dict[str, Any]:
        """
        评估模型并返回详细的诊断信息
        
        Args:
            model: 要评估的模型
            tokenizer: 分词器
            datasets: 数据集字典
            model_name: 模型名称
            save_diagnostics: 是否保存诊断报告
            
        Returns:
            包含评估结果和诊断信息的字典
        """
        # 执行标准评估
        evaluation_result = self.evaluate_model(model, tokenizer, datasets, model_name)
        
        # 获取诊断信息
        diagnostics = {}
        if self.enable_enhanced_processing:
            diagnostics = self.diagnose_data_processing(datasets)
            
            # 保存诊断报告
            if save_diagnostics:
                try:
                    report_path = self.data_preprocessor.save_processing_report(
                        f"evaluation_diagnostics_{model_name}.json"
                    )
                    diagnostics["report_path"] = report_path
                except Exception as e:
                    logger.warning(f"保存诊断报告失败: {e}")
        
        return {
            "evaluation_result": evaluation_result,
            "diagnostics": diagnostics,
            "processing_stats": self.data_preprocessor.get_processing_statistics() if self.enable_enhanced_processing else {}
        }
    
    def quick_evaluate(self, 
                      model,
                      tokenizer,
                      dataset: Dataset,
                      task_name: str = "text_generation",
                      model_name: str = "quick_eval_model") -> Dict[str, Any]:
        """
        快速评估单个数据集
        
        Args:
            model: 要评估的模型
            tokenizer: 分词器
            dataset: 数据集
            task_name: 任务名称
            model_name: 模型名称
            
        Returns:
            评估结果字典
        """
        datasets = {task_name: dataset}
        
        if self.enable_enhanced_processing:
            return self.evaluate_model_with_diagnostics(
                model, tokenizer, datasets, model_name, save_diagnostics=False
            )
        else:
            evaluation_result = self.evaluate_model(model, tokenizer, datasets, model_name)
            return {"evaluation_result": evaluation_result}
    
    def diagnose_dataset(self, dataset: Dataset, task_name: str) -> Dict[str, Any]:
        """
        诊断数据集问题
        
        Args:
            dataset: 数据集
            task_name: 任务名称
            
        Returns:
            诊断结果
        """
        if not self.enable_enhanced_processing:
            logger.warning("增强数据处理功能未启用，无法进行详细诊断")
            return {"message": "增强数据处理功能未启用"}
        
        # 取样本进行诊断
        sample_size = min(self.config.batch_size, len(dataset))
        sample_batch = dataset[:sample_size]
        
        return self.data_preprocessor.diagnose_batch(sample_batch, task_name)
    
    def get_processing_recommendations(self, datasets: Dict[str, Dataset]) -> List[str]:
        """
        获取数据处理建议
        
        Args:
            datasets: 数据集字典
            
        Returns:
            建议列表
        """
        if not self.enable_enhanced_processing:
            return ["启用增强数据处理功能以获取详细建议"]
        
        recommendations = []
        
        for task_name, dataset in datasets.items():
            diagnosis = self.diagnose_dataset(dataset, task_name)
            if "recommendations" in diagnosis:
                task_recommendations = [f"[{task_name}] {rec}" for rec in diagnosis["recommendations"]]
                recommendations.extend(task_recommendations)
        
        return recommendations
    
    def configure_enhanced_processing(self, 
                                    enable_detailed_logging: bool = False,
                                    enable_data_cleaning: bool = True,
                                    enable_fallback: bool = True,
                                    min_valid_samples_ratio: float = 0.1) -> None:
        """
        配置增强数据处理功能
        
        Args:
            enable_detailed_logging: 是否启用详细日志
            enable_data_cleaning: 是否启用数据清洗
            enable_fallback: 是否启用降级处理
            min_valid_samples_ratio: 最小有效样本比例
        """
        if not self.enable_enhanced_processing:
            logger.warning("增强数据处理功能未启用，无法配置")
            return
        
        # 更新配置
        self.config.data_processing["validation"].update({
            "enable_data_cleaning": enable_data_cleaning,
            "enable_fallback": enable_fallback,
            "min_valid_samples_ratio": min_valid_samples_ratio
        })
        
        self.config.data_processing["diagnostics"].update({
            "enable_detailed_logging": enable_detailed_logging
        })
        
        # 更新数据预处理器配置
        self.data_preprocessor.update_config(self.config)
        
        logger.info("增强数据处理配置已更新")


def create_enhanced_evaluation_engine(config_file: Optional[str] = None,
                                     config_data: Optional[Dict[str, Any]] = None,
                                     device: str = "auto",
                                     max_workers: int = 4) -> EvaluationEngineWrapper:
    """
    创建增强的评估引擎
    
    Args:
        config_file: 配置文件路径
        config_data: 配置数据字典
        device: 计算设备
        max_workers: 最大工作线程数
        
    Returns:
        增强的评估引擎实例
    """
    # 加载配置
    if config_file or config_data:
        config = load_evaluation_config(
            config_file=config_file,
            config_data=config_data,
            validate=True,
            auto_fix=True
        )
    else:
        config = EvaluationConfig()
    
    return EvaluationEngineWrapper(
        config=config,
        device=device,
        max_workers=max_workers,
        enable_enhanced_processing=True
    )


def migrate_legacy_evaluation(legacy_evaluation_func,
                            model,
                            tokenizer,
                            datasets: Dict[str, Dataset],
                            model_name: str = "migrated_model") -> Dict[str, Any]:
    """
    迁移遗留评估函数到新系统
    
    Args:
        legacy_evaluation_func: 遗留的评估函数
        model: 模型
        tokenizer: 分词器
        datasets: 数据集字典
        model_name: 模型名称
        
    Returns:
        评估结果和迁移信息
    """
    logger.info("开始迁移遗留评估函数...")
    
    # 尝试使用遗留函数
    legacy_result = None
    legacy_error = None
    
    try:
        legacy_result = legacy_evaluation_func(model, tokenizer, datasets)
        logger.info("遗留评估函数执行成功")
    except Exception as e:
        legacy_error = str(e)
        logger.warning(f"遗留评估函数执行失败: {e}")
    
    # 使用新系统进行评估
    enhanced_engine = create_enhanced_evaluation_engine()
    enhanced_result = enhanced_engine.evaluate_model_with_diagnostics(
        model, tokenizer, datasets, model_name
    )
    
    return {
        "legacy_result": legacy_result,
        "legacy_error": legacy_error,
        "enhanced_result": enhanced_result,
        "migration_successful": legacy_error is None,
        "recommendations": enhanced_engine.get_processing_recommendations(datasets)
    }
#!/usr/bin/env python3
"""
增强配置类

扩展现有的ApplicationConfig，添加数据拆分和评估相关配置项。
保持与现有配置的完全兼容性。
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 导入现有的ApplicationConfig
from main import ApplicationConfig


@dataclass
class EnhancedApplicationConfig(ApplicationConfig):
    """
    增强的应用程序配置
    
    扩展现有ApplicationConfig，添加数据拆分和评估功能配置。
    保持与现有配置的完全兼容性。
    """
    
    # === 数据拆分配置 ===
    enable_data_splitting: bool = True
    """是否启用数据拆分功能"""
    
    train_ratio: float = 0.7
    """训练集比例"""
    
    val_ratio: float = 0.15
    """验证集比例"""
    
    test_ratio: float = 0.15
    """测试集比例"""
    
    stratify_by: Optional[str] = None
    """分层抽样字段名称，如果为None则使用随机拆分"""
    
    data_split_seed: int = 42
    """数据拆分随机种子"""
    
    min_samples_per_split: int = 10
    """每个拆分的最小样本数"""
    
    enable_data_quality_analysis: bool = True
    """是否启用数据质量分析"""
    
    data_splits_output_dir: str = "./data/splits"
    """数据拆分结果输出目录"""
    
    # === 评估配置 ===
    enable_comprehensive_evaluation: bool = True
    """是否启用全面评估"""
    
    evaluation_tasks: List[str] = field(default_factory=lambda: ["text_generation"])
    """评估任务列表"""
    
    evaluation_metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge", "accuracy"])
    """评估指标列表"""
    
    enable_efficiency_metrics: bool = True
    """是否启用效率指标测量"""
    
    enable_quality_analysis: bool = True
    """是否启用质量分析"""
    
    evaluation_batch_size: int = 12
    """评估批次大小"""
    
    evaluation_num_samples: int = 100
    """评估样本数量限制，-1表示使用全部样本"""
    
    evaluation_max_length: int = 512
    """评估时的最大序列长度"""
    
    evaluation_temperature: float = 0.7
    """评估时的生成温度"""
    
    evaluation_top_p: float = 0.9
    """评估时的top-p参数"""
    
    # === 基准测试配置 ===
    enable_benchmark_testing: bool = False
    """是否启用基准测试"""
    
    benchmark_datasets: List[str] = field(default_factory=list)
    """基准测试数据集列表"""
    
    baseline_model_path: Optional[str] = None
    """基线模型路径，用于对比评估"""
    
    # === 实验跟踪配置 ===
    enable_experiment_tracking: bool = True
    """是否启用实验跟踪"""
    
    experiment_name: Optional[str] = None
    """实验名称，如果为None则自动生成"""
    
    experiment_tags: List[str] = field(default_factory=list)
    """实验标签列表"""
    
    experiment_description: str = ""
    """实验描述"""
    
    experiments_output_dir: str = "./experiments"
    """实验记录输出目录"""
    
    # === 报告配置 ===
    report_formats: List[str] = field(default_factory=lambda: ["html", "json"])
    """报告格式列表，支持: html, json, csv, latex"""
    
    enable_visualization: bool = True
    """是否启用可视化图表"""
    
    output_charts: bool = True
    """是否输出图表文件"""
    
    reports_output_dir: str = "./reports"
    """报告输出目录"""
    
    # === 训练增强配置 ===
    enable_validation_during_training: bool = True
    """是否在训练过程中进行验证集评估"""
    
    validation_steps: int = 100
    """验证评估间隔步数（对应TrainingConfig.eval_steps）"""
    
    save_validation_metrics: bool = True
    """是否保存验证指标历史"""
    
    enable_early_stopping: bool = True
    """是否启用早停机制"""
    
    early_stopping_patience: int = 5
    """早停耐心值（验证损失不改善的步数）"""
    
    early_stopping_threshold: float = 0.001
    """早停阈值（最小改善量）"""
    
    # === 高级配置 ===
    skip_data_splitting_if_exists: bool = True
    """如果数据拆分结果已存在，是否跳过拆分步骤"""
    
    force_recompute_evaluation: bool = False
    """是否强制重新计算评估结果"""
    
    parallel_evaluation: bool = False
    """是否启用并行评估（实验性功能）"""
    
    max_evaluation_workers: int = 4
    """最大评估工作线程数"""
    
    # === 兼容性配置 ===
    fallback_to_basic_mode: bool = True
    """如果增强功能失败，是否回退到基础模式"""
    
    def validate_config(self) -> List[str]:
        """
        验证配置的有效性
        
        Returns:
            错误信息列表，空列表表示配置有效
        """
        errors = []
        
        # 验证拆分比例
        if self.enable_data_splitting:
            total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                errors.append(f"数据拆分比例之和必须为1.0，当前为: {total_ratio}")
            
            if any(ratio <= 0 for ratio in [self.train_ratio, self.val_ratio, self.test_ratio]):
                errors.append("所有数据拆分比例必须大于0")
        
        # 验证评估配置
        if self.enable_comprehensive_evaluation:
            if not self.evaluation_tasks:
                errors.append("启用评估时必须指定至少一个评估任务")
            
            if not self.evaluation_metrics:
                errors.append("启用评估时必须指定至少一个评估指标")
        
        # 验证报告格式
        valid_formats = {"html", "json", "csv", "latex"}
        invalid_formats = set(self.report_formats) - valid_formats
        if invalid_formats:
            errors.append(f"不支持的报告格式: {invalid_formats}")
        
        # 验证批次大小
        if self.evaluation_batch_size <= 0:
            errors.append("评估批次大小必须大于0")
        
        # 验证样本数量
        if self.evaluation_num_samples == 0:
            errors.append("评估样本数量不能为0")
        
        return errors
    
    def get_data_split_config(self) -> Dict[str, Any]:
        """
        获取数据拆分相关配置
        
        Returns:
            数据拆分配置字典
        """
        return {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "stratify_by": self.stratify_by,
            "random_seed": self.data_split_seed,
            "min_samples_per_split": self.min_samples_per_split,
            "enable_quality_analysis": self.enable_data_quality_analysis
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        获取评估相关配置
        
        Returns:
            评估配置字典
        """
        return {
            "tasks": self.evaluation_tasks,
            "metrics": self.evaluation_metrics,
            "batch_size": self.evaluation_batch_size,
            "num_samples": self.evaluation_num_samples if self.evaluation_num_samples > 0 else None,
            "max_length": self.evaluation_max_length,
            "temperature": self.evaluation_temperature,
            "top_p": self.evaluation_top_p,
            "enable_efficiency_metrics": self.enable_efficiency_metrics,
            "enable_quality_analysis": self.enable_quality_analysis
        }
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """
        获取实验跟踪相关配置
        
        Returns:
            实验配置字典
        """
        return {
            "experiment_name": self.experiment_name,
            "tags": self.experiment_tags,
            "description": self.experiment_description,
            "model_config": {
                "model_name": self.model_name,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout
            },
            "training_config": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "max_sequence_length": self.max_sequence_length
            },
            "data_config": {
                "data_dir": self.data_dir,
                "enable_splitting": self.enable_data_splitting,
                **self.get_data_split_config()
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedApplicationConfig':
        """
        从字典创建配置对象
        
        Args:
            config_dict: 配置字典
            
        Returns:
            配置对象
        """
        # 过滤掉不存在的字段
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        from dataclasses import asdict
        return asdict(self)


def load_enhanced_config_from_yaml(yaml_path: str) -> EnhancedApplicationConfig:
    """
    从YAML文件加载增强配置
    
    Args:
        yaml_path: YAML配置文件路径
        
    Returns:
        增强配置对象
    """
    try:
        import yaml
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 展平嵌套配置
        flattened_config = {}
        
        # 处理模型配置
        if 'model' in config_dict:
            model_config = config_dict['model']
            flattened_config.update({
                'model_name': model_config.get('name', 'Qwen/Qwen3-4B-Thinking-2507'),
                'output_dir': model_config.get('output_dir', './qwen3-finetuned')
            })
        
        # 处理数据配置
        if 'data' in config_dict:
            data_config = config_dict['data']
            flattened_config.update({
                'data_dir': data_config.get('data_dir', 'data/raw'),
                'enable_data_splitting': data_config.get('enable_splitting', True),
                'train_ratio': data_config.get('train_ratio', 0.7),
                'val_ratio': data_config.get('val_ratio', 0.15),
                'test_ratio': data_config.get('test_ratio', 0.15),
                'stratify_by': data_config.get('stratify_by'),
                'data_split_seed': data_config.get('split_seed', 42)
            })
        
        # 处理训练配置
        if 'training' in config_dict:
            training_config = config_dict['training']
            flattened_config.update({
                'batch_size': training_config.get('batch_size', 4),
                'gradient_accumulation_steps': training_config.get('gradient_accumulation_steps', 16),
                'learning_rate': training_config.get('learning_rate', 5e-5),
                'num_epochs': training_config.get('num_epochs', 5),
                'max_sequence_length': training_config.get('max_sequence_length', 256),
                'max_memory_gb': training_config.get('max_memory_gb', 13.0),
                'enable_validation_during_training': training_config.get('enable_validation', True),
                'validation_steps': training_config.get('validation_steps', 100),
                'save_validation_metrics': training_config.get('save_validation_metrics', True),
                'enable_early_stopping': training_config.get('enable_early_stopping', True),
                'early_stopping_patience': training_config.get('early_stopping_patience', 5),
                'early_stopping_threshold': training_config.get('early_stopping_threshold', 0.001),
                # LoRA配置
                'lora_r': training_config.get('lora_r', 6),
                'lora_alpha': training_config.get('lora_alpha', 12),
                'lora_dropout': training_config.get('lora_dropout', 0.1)
            })
        
        # 处理评估配置
        if 'evaluation' in config_dict:
            eval_config = config_dict['evaluation']
            flattened_config.update({
                'enable_comprehensive_evaluation': eval_config.get('enable_comprehensive', True),
                'evaluation_tasks': eval_config.get('tasks', ['text_generation']),
                'evaluation_metrics': eval_config.get('metrics', ['bleu', 'rouge', 'accuracy']),
                'enable_efficiency_metrics': eval_config.get('enable_efficiency', True),
                'enable_quality_analysis': eval_config.get('enable_quality', True),
                'evaluation_batch_size': eval_config.get('batch_size', 4),
                'evaluation_num_samples': eval_config.get('num_samples', 100),
                'evaluation_max_length': eval_config.get('max_length', 512),
                'evaluation_temperature': eval_config.get('temperature', 0.7),
                'evaluation_top_p': eval_config.get('top_p', 0.9)
            })
        
        # 处理实验配置
        if 'experiment' in config_dict:
            exp_config = config_dict['experiment']
            flattened_config.update({
                'enable_experiment_tracking': exp_config.get('enable_tracking', True),
                'experiment_name': exp_config.get('name'),
                'experiment_tags': exp_config.get('tags', [])
            })
        
        # 处理报告配置
        if 'reports' in config_dict:
            report_config = config_dict['reports']
            flattened_config.update({
                'report_formats': report_config.get('formats', ['html', 'json']),
                'enable_visualization': report_config.get('enable_visualization', True),
                'output_charts': report_config.get('output_charts', True)
            })
        
        # 处理系统配置
        if 'system' in config_dict:
            system_config = config_dict['system']
            flattened_config.update({
                'log_dir': system_config.get('log_dir', './logs'),
                'enable_tensorboard': system_config.get('enable_tensorboard', True),
                'enable_inference_test': system_config.get('enable_inference_test', True),
                'verify_environment': system_config.get('verify_environment', True),
                'auto_install_deps': system_config.get('auto_install_deps', False)
            })
        
        # 处理基准测试配置
        if 'benchmark' in config_dict:
            benchmark_config = config_dict['benchmark']
            flattened_config.update({
                'enable_benchmark_testing': benchmark_config.get('enable_testing', False),
                'benchmark_datasets': benchmark_config.get('datasets', []),
                'baseline_model_path': benchmark_config.get('baseline_model_path')
            })
        
        # 处理高级配置
        if 'advanced' in config_dict:
            advanced_config = config_dict['advanced']
            flattened_config.update({
                'skip_data_splitting_if_exists': advanced_config.get('skip_data_splitting_if_exists', True),
                'force_recompute_evaluation': advanced_config.get('force_recompute_evaluation', False),
                'parallel_evaluation': advanced_config.get('parallel_evaluation', False),
                'max_evaluation_workers': advanced_config.get('max_evaluation_workers', 2),
                'fallback_to_basic_mode': advanced_config.get('fallback_to_basic_mode', True)
            })
        
        # 添加其他顶级配置
        for key, value in config_dict.items():
            if key not in ['model', 'data', 'training', 'evaluation', 'experiment', 'reports', 'system', 'benchmark', 'advanced']:
                flattened_config[key] = value
        
        return EnhancedApplicationConfig.from_dict(flattened_config)
        
    except ImportError:
        raise ImportError("需要安装PyYAML来支持YAML配置文件: pip install pyyaml")
    except Exception as e:
        raise ValueError(f"加载YAML配置文件失败: {e}")


def create_example_config() -> EnhancedApplicationConfig:
    """
    创建示例配置
    
    Returns:
        示例配置对象
    """
    return EnhancedApplicationConfig(
        # 基础配置
        model_name="Qwen/Qwen3-4B-Thinking-2507",
        output_dir="./enhanced-qwen3-finetuned",
        data_dir="data/raw",
        
        # 数据拆分配置
        enable_data_splitting=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        
        # 评估配置
        enable_comprehensive_evaluation=True,
        evaluation_tasks=["text_generation"],
        evaluation_metrics=["bleu", "rouge", "accuracy"],
        
        # 实验跟踪
        enable_experiment_tracking=True,
        experiment_name="enhanced_pipeline_demo",
        experiment_tags=["qwen3", "enhanced", "demo"],
        
        # 报告配置
        report_formats=["html", "json"],
        enable_visualization=True
    )


if __name__ == "__main__":
    # 测试配置类
    config = create_example_config()
    
    # 验证配置
    errors = config.validate_config()
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("配置验证通过")
    
    # 显示配置信息
    print(f"\n配置信息:")
    print(f"  模型: {config.model_name}")
    print(f"  数据拆分: {config.enable_data_splitting}")
    print(f"  评估: {config.enable_comprehensive_evaluation}")
    print(f"  实验跟踪: {config.enable_experiment_tracking}")
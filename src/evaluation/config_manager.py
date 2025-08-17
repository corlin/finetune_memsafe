"""
配置管理系统

提供YAML配置文件的加载、验证和管理功能。
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict

from .data_models import EvaluationConfig, BenchmarkConfig, ExperimentConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认配置
        self.default_configs = {
            "data_split": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "stratify_by": None,
                "random_seed": 42,
                "min_samples_per_split": 10
            },
            "evaluation": {
                "tasks": ["text_generation", "question_answering"],
                "metrics": ["bleu", "rouge", "bertscore", "perplexity"],
                "batch_size": 8,
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_samples": 100,
                "enable_efficiency_metrics": True,
                "enable_quality_analysis": True
            },
            "benchmarks": {
                "clue": {
                    "name": "CLUE",
                    "dataset_path": "benchmarks/clue",
                    "tasks": ["tnews", "afqmc", "cmnli", "ocnli", "wsc", "csl"],
                    "evaluation_protocol": "official",
                    "metrics": ["accuracy", "f1"]
                },
                "few_clue": {
                    "name": "FewCLUE",
                    "dataset_path": "benchmarks/few_clue",
                    "tasks": ["tnews", "afqmc", "cmnli"],
                    "evaluation_protocol": "few_shot",
                    "metrics": ["accuracy", "f1"]
                }
            },
            "experiment_tracking": {
                "enabled": True,
                "experiment_dir": "./experiments",
                "auto_compare": True,
                "leaderboard_metric": "overall_score",
                "save_predictions": True,
                "save_model_outputs": False
            },
            "quality_analysis": {
                "enable_duplicate_detection": True,
                "enable_length_analysis": True,
                "enable_vocabulary_analysis": True,
                "length_outlier_threshold": 3.0,  # 标准差倍数
                "min_length": 5,
                "max_length": 2048
            }
        }
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            # 验证配置
            validated_config = self.validate_config(config)
            
            logger.info(f"成功加载配置文件: {config_path}")
            return validated_config
            
        except Exception as e:
            logger.error(f"加载配置文件失败 {config_path}: {e}")
            raise
    
    def save_config(self, config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """
        保存配置文件
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            logger.info(f"配置文件已保存: {config_path}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败 {config_path}: {e}")
            raise
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证配置文件
        
        Args:
            config: 配置字典
            
        Returns:
            验证后的配置字典
        """
        validated_config = {}
        
        # 合并默认配置
        for section, default_values in self.default_configs.items():
            if section in config:
                # 合并用户配置和默认配置
                validated_config[section] = {**default_values, **config[section]}
            else:
                # 使用默认配置
                validated_config[section] = default_values.copy()
        
        # 添加用户自定义的其他配置
        for key, value in config.items():
            if key not in validated_config:
                validated_config[key] = value
        
        # 特定验证
        self._validate_data_split_config(validated_config.get("data_split", {}))
        self._validate_evaluation_config(validated_config.get("evaluation", {}))
        
        return validated_config
    
    def _validate_data_split_config(self, config: Dict[str, Any]) -> None:
        """验证数据拆分配置"""
        ratios = [config.get("train_ratio", 0.7), 
                 config.get("val_ratio", 0.15), 
                 config.get("test_ratio", 0.15)]
        
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"数据拆分比例之和必须为1.0，当前为: {sum(ratios)}")
        
        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("数据拆分比例必须大于0")
        
        if config.get("random_seed") is not None and not isinstance(config["random_seed"], int):
            raise ValueError("random_seed必须是整数")
    
    def _validate_evaluation_config(self, config: Dict[str, Any]) -> None:
        """验证评估配置"""
        if config.get("batch_size", 1) <= 0:
            raise ValueError("batch_size必须大于0")
        
        if config.get("max_length", 512) <= 0:
            raise ValueError("max_length必须大于0")
        
        if not 0 <= config.get("temperature", 0.7) <= 2.0:
            raise ValueError("temperature必须在0-2.0之间")
        
        if not 0 <= config.get("top_p", 0.9) <= 1.0:
            raise ValueError("top_p必须在0-1.0之间")
    
    def create_evaluation_config(self, config_dict: Dict[str, Any]) -> EvaluationConfig:
        """
        创建评估配置对象
        
        Args:
            config_dict: 配置字典
            
        Returns:
            EvaluationConfig对象
        """
        eval_config = config_dict.get("evaluation", {})
        
        return EvaluationConfig(
            tasks=eval_config.get("tasks", ["text_generation"]),
            metrics=eval_config.get("metrics", ["bleu", "rouge"]),
            batch_size=eval_config.get("batch_size", 8),
            max_length=eval_config.get("max_length", 512),
            temperature=eval_config.get("temperature", 0.7),
            top_p=eval_config.get("top_p", 0.9),
            num_samples=eval_config.get("num_samples", 100),
            enable_efficiency_metrics=eval_config.get("enable_efficiency_metrics", True),
            enable_quality_analysis=eval_config.get("enable_quality_analysis", True)
        )
    
    def create_benchmark_config(self, benchmark_name: str, 
                              config_dict: Dict[str, Any]) -> BenchmarkConfig:
        """
        创建基准测试配置对象
        
        Args:
            benchmark_name: 基准测试名称
            config_dict: 配置字典
            
        Returns:
            BenchmarkConfig对象
        """
        benchmark_configs = config_dict.get("benchmarks", {})
        
        if benchmark_name not in benchmark_configs:
            raise ValueError(f"未找到基准测试配置: {benchmark_name}")
        
        benchmark_config = benchmark_configs[benchmark_name]
        
        return BenchmarkConfig(
            name=benchmark_config["name"],
            dataset_path=benchmark_config["dataset_path"],
            tasks=benchmark_config["tasks"],
            evaluation_protocol=benchmark_config["evaluation_protocol"],
            metrics=benchmark_config["metrics"],
            max_samples=benchmark_config.get("max_samples")
        )
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        return self.default_configs.copy()
    
    def create_config_template(self, output_path: Union[str, Path]) -> None:
        """
        创建配置文件模板
        
        Args:
            output_path: 输出路径
        """
        template_config = self.get_default_config()
        
        # 添加注释说明
        template_with_comments = {
            "# 数据拆分配置": None,
            "data_split": template_config["data_split"],
            "# 评估配置": None,
            "evaluation": template_config["evaluation"],
            "# 基准测试配置": None,
            "benchmarks": template_config["benchmarks"],
            "# 实验跟踪配置": None,
            "experiment_tracking": template_config["experiment_tracking"],
            "# 质量分析配置": None,
            "quality_analysis": template_config["quality_analysis"]
        }
        
        # 移除注释键（YAML不支持这种注释方式）
        clean_config = {k: v for k, v in template_with_comments.items() 
                       if not k.startswith("#")}
        
        self.save_config(clean_config, output_path)
        logger.info(f"配置模板已创建: {output_path}")


# 全局配置管理器实例
config_manager = ConfigManager()    def c
reate_advanced_config_template(self, output_path: Union[str, Path], 
                                       config_type: str = "evaluation") -> None:
        """
        创建高级配置文件模板
        
        Args:
            output_path: 输出路径
            config_type: 配置类型 (evaluation, experiment, benchmark, training)
        """
        from datetime import datetime
        
        if config_type == "evaluation":
            template_config = self.get_default_config()
        elif config_type == "experiment":
            template_config = self._get_experiment_config_template()
        elif config_type == "benchmark":
            template_config = self._get_benchmark_config_template()
        elif config_type == "training":
            template_config = self._get_training_config_template()
        else:
            template_config = self.get_default_config()
        
        # 添加元数据
        template_config["_metadata"] = {
            "description": f"Qwen3微调系统{config_type}配置文件",
            "version": "1.0.0",
            "created": datetime.now().isoformat(),
            "type": config_type
        }
        
        self.save_config(template_config, output_path)
        logger.info(f"高级配置模板已创建: {output_path}")
    
    def _get_experiment_config_template(self) -> Dict[str, Any]:
        """获取实验配置模板"""
        return {
            "experiment": {
                "name": "example_experiment",
                "description": "示例实验配置",
                "tags": ["example", "test"],
                "author": "user",
                "version": "1.0.0"
            },
            "model": {
                "name": "qwen3-7b",
                "path": "/path/to/model",
                "tokenizer_path": "/path/to/tokenizer",
                "device": "cuda:0",
                "dtype": "float16",
                "load_in_8bit": False,
                "load_in_4bit": False
            },
            "training": {
                "learning_rate": 2e-5,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "num_epochs": 3,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "save_steps": 500,
                "eval_steps": 500,
                "logging_steps": 100,
                "lr_scheduler_type": "cosine",
                "optimizer": "adamw_torch"
            },
            "data": {
                "train_data": "/path/to/train/data",
                "val_data": "/path/to/val/data",
                "test_data": "/path/to/test/data",
                "max_length": 512,
                "data_format": "conversation",
                "preprocessing": {
                    "remove_duplicates": True,
                    "filter_length": True,
                    "normalize_text": False,
                    "add_special_tokens": True
                }
            },
            "evaluation": self.get_default_config()["evaluation"],
            "monitoring": {
                "patience": 3,
                "min_delta": 0.001,
                "monitor_metric": "val_loss",
                "mode": "min",
                "early_stopping": True,
                "save_best_model": True
            },
            "output": {
                "output_dir": "./experiment_output",
                "save_model": True,
                "save_logs": True,
                "save_checkpoints": True,
                "checkpoint_limit": 3
            }
        }
    
    def _get_benchmark_config_template(self) -> Dict[str, Any]:
        """获取基准测试配置模板"""
        return {
            "benchmark": {
                "name": "clue_benchmark",
                "description": "CLUE基准测试配置",
                "version": "1.0.0"
            },
            "models": [
                {
                    "name": "model_1",
                    "path": "/path/to/model_1",
                    "tokenizer": "/path/to/tokenizer_1",
                    "device": "cuda:0",
                    "dtype": "float16"
                },
                {
                    "name": "model_2", 
                    "path": "/path/to/model_2",
                    "tokenizer": "/path/to/tokenizer_2",
                    "device": "cuda:1",
                    "dtype": "float16"
                }
            ],
            "benchmarks": {
                "clue": {
                    "tasks": ["tnews", "afqmc", "cmnli", "ocnli", "wsc", "csl"],
                    "split": "test",
                    "max_samples": None,
                    "protocol": "official"
                },
                "few_clue": {
                    "tasks": ["tnews", "afqmc", "cmnli"],
                    "split": "test",
                    "num_shots": 5,
                    "max_samples": 1000,
                    "protocol": "few_shot"
                },
                "c_eval": {
                    "tasks": ["high_school_physics", "computer_science"],
                    "split": "test",
                    "max_samples": None,
                    "protocol": "multiple_choice"
                }
            },
            "evaluation": {
                "batch_size": 8,
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "device": "auto"
            },
            "output": {
                "output_dir": "./benchmark_results",
                "cache_dir": "./benchmark_cache",
                "save_predictions": True,
                "generate_report": True,
                "report_format": ["html", "json"]
            }
        }
    
    def _get_training_config_template(self) -> Dict[str, Any]:
        """获取训练配置模板"""
        return {
            "training": {
                "model_name": "qwen3-7b",
                "base_model": "/path/to/base/model",
                "tokenizer": "/path/to/tokenizer",
                "training_type": "sft",
                "device": "cuda:0",
                "dtype": "float16"
            },
            "hyperparameters": {
                "learning_rate": 2e-5,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "num_epochs": 3,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "lr_scheduler": "cosine",
                "optimizer": "adamw",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8
            },
            "data": {
                "train_file": "/path/to/train.json",
                "val_file": "/path/to/val.json",
                "max_length": 512,
                "data_format": "conversation",
                "preprocessing": {
                    "add_special_tokens": True,
                    "truncation": True,
                    "padding": "max_length",
                    "remove_duplicates": True
                }
            },
            "lora": {
                "enabled": True,
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            "evaluation": {
                "eval_strategy": "steps",
                "eval_steps": 500,
                "eval_tasks": ["text_generation"],
                "eval_metrics": ["bleu", "rouge"],
                "save_best_model": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False
            },
            "monitoring": {
                "logging_steps": 100,
                "save_steps": 500,
                "early_stopping": {
                    "enabled": True,
                    "patience": 3,
                    "min_delta": 0.001,
                    "monitor": "eval_loss"
                },
                "wandb": {
                    "enabled": False,
                    "project": "qwen3_training",
                    "entity": "your_entity"
                }
            },
            "output": {
                "output_dir": "./training_output",
                "run_name": "qwen3_sft_experiment",
                "save_total_limit": 3,
                "load_best_model_at_end": True,
                "push_to_hub": False
            }
        }
    
    def validate_advanced_config(self, config: Dict[str, Any], config_type: str) -> Dict[str, Any]:
        """
        高级配置验证
        
        Args:
            config: 配置字典
            config_type: 配置类型
            
        Returns:
            验证后的配置字典
        """
        if config_type == "experiment":
            return self._validate_experiment_config(config)
        elif config_type == "benchmark":
            return self._validate_benchmark_config(config)
        elif config_type == "training":
            return self._validate_training_config(config)
        else:
            return self.validate_config(config)
    
    def _validate_experiment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证实验配置"""
        # 验证必需字段
        required_fields = ["experiment", "model", "training", "data"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"实验配置缺少必需字段: {field}")
        
        # 验证实验信息
        if "name" not in config["experiment"]:
            raise ValueError("实验配置必须包含实验名称")
        
        # 验证模型路径
        model_config = config["model"]
        if "path" not in model_config:
            raise ValueError("模型配置必须包含模型路径")
        
        # 验证训练参数
        training_config = config["training"]
        if training_config.get("learning_rate", 0) <= 0:
            raise ValueError("学习率必须大于0")
        
        if training_config.get("batch_size", 0) <= 0:
            raise ValueError("批次大小必须大于0")
        
        return config
    
    def _validate_benchmark_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证基准测试配置"""
        # 验证必需字段
        if "models" not in config or not config["models"]:
            raise ValueError("基准测试配置必须包含模型列表")
        
        if "benchmarks" not in config or not config["benchmarks"]:
            raise ValueError("基准测试配置必须包含基准测试列表")
        
        # 验证模型配置
        for i, model in enumerate(config["models"]):
            if "name" not in model or "path" not in model:
                raise ValueError(f"模型 {i} 缺少必需字段 name 或 path")
        
        return config
    
    def _validate_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证训练配置"""
        # 验证必需字段
        required_sections = ["training", "hyperparameters", "data"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"训练配置缺少必需部分: {section}")
        
        # 验证超参数
        hyperparams = config["hyperparameters"]
        if hyperparams.get("learning_rate", 0) <= 0:
            raise ValueError("学习率必须大于0")
        
        if hyperparams.get("batch_size", 0) <= 0:
            raise ValueError("批次大小必须大于0")
        
        if hyperparams.get("num_epochs", 0) <= 0:
            raise ValueError("训练轮数必须大于0")
        
        # 验证数据配置
        data_config = config["data"]
        if "train_file" not in data_config:
            raise ValueError("数据配置必须包含训练文件路径")
        
        return config
    
    def convert_config_format(self, input_path: Union[str, Path], 
                            output_path: Union[str, Path]) -> None:
        """
        转换配置文件格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
        """
        # 加载原配置
        config = self.load_config(input_path)
        
        # 保存为新格式
        self.save_config(config, output_path)
        
        logger.info(f"配置格式转换完成: {input_path} -> {output_path}")
    
    def merge_configs(self, base_config_path: Union[str, Path], 
                     override_config_path: Union[str, Path],
                     output_path: Union[str, Path]) -> None:
        """
        合并配置文件
        
        Args:
            base_config_path: 基础配置文件路径
            override_config_path: 覆盖配置文件路径
            output_path: 输出文件路径
        """
        base_config = self.load_config(base_config_path)
        override_config = self.load_config(override_config_path)
        
        # 深度合并配置
        merged_config = self._deep_merge_dict(base_config, override_config)
        
        # 保存合并后的配置
        self.save_config(merged_config, output_path)
        
        logger.info(f"配置合并完成: {output_path}")
    
    def _deep_merge_dict(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并字典
        
        Args:
            base: 基础字典
            override: 覆盖字典
            
        Returns:
            合并后的字典
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_config_schema(self, config_type: str) -> Dict[str, Any]:
        """
        获取配置模式定义
        
        Args:
            config_type: 配置类型
            
        Returns:
            配置模式字典
        """
        schemas = {
            "evaluation": {
                "type": "object",
                "properties": {
                    "data_split": {"type": "object"},
                    "evaluation": {"type": "object"},
                    "benchmarks": {"type": "object"},
                    "experiment_tracking": {"type": "object"},
                    "quality_analysis": {"type": "object"}
                },
                "required": ["evaluation"]
            },
            "experiment": {
                "type": "object",
                "properties": {
                    "experiment": {"type": "object"},
                    "model": {"type": "object"},
                    "training": {"type": "object"},
                    "data": {"type": "object"}
                },
                "required": ["experiment", "model", "training", "data"]
            },
            "benchmark": {
                "type": "object",
                "properties": {
                    "benchmark": {"type": "object"},
                    "models": {"type": "array"},
                    "benchmarks": {"type": "object"},
                    "evaluation": {"type": "object"},
                    "output": {"type": "object"}
                },
                "required": ["models", "benchmarks"]
            },
            "training": {
                "type": "object",
                "properties": {
                    "training": {"type": "object"},
                    "hyperparameters": {"type": "object"},
                    "data": {"type": "object"},
                    "output": {"type": "object"}
                },
                "required": ["training", "hyperparameters", "data"]
            }
        }
        
        return schemas.get(config_type, {})
    
    def list_config_templates(self) -> List[str]:
        """
        列出可用的配置模板
        
        Returns:
            配置模板类型列表
        """
        return ["evaluation", "experiment", "benchmark", "training"]
    
    def backup_config(self, config_path: Union[str, Path], 
                     backup_dir: Union[str, Path] = None) -> str:
        """
        备份配置文件
        
        Args:
            config_path: 配置文件路径
            backup_dir: 备份目录
            
        Returns:
            备份文件路径
        """
        from datetime import datetime
        import shutil
        
        config_path = Path(config_path)
        
        if backup_dir is None:
            backup_dir = config_path.parent / "backups"
        else:
            backup_dir = Path(backup_dir)
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成备份文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{config_path.stem}_{timestamp}{config_path.suffix}"
        backup_path = backup_dir / backup_filename
        
        # 复制文件
        shutil.copy2(config_path, backup_path)
        
        logger.info(f"配置文件已备份: {backup_path}")
        return str(backup_path)
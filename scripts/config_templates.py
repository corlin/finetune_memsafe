#!/usr/bin/env python3
"""
配置模板生成器

提供各种评估场景的配置模板和预设方案。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import yaml

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_data_split_template() -> Dict[str, Any]:
    """获取数据拆分配置模板"""
    return {
        "data_split": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "stratify_by": None,
            "random_seed": 42,
            "min_samples_per_split": 1,
            "enable_quality_analysis": True
        },
        "quality_analysis": {
            "enable_duplicate_detection": True,
            "enable_length_analysis": True,
            "enable_vocabulary_analysis": True,
            "length_outlier_threshold": 3.0,
            "min_length": 5,
            "max_length": 2048,
            "quality_checks": [
                "empty_content",
                "encoding_errors",
                "format_consistency",
                "language_detection"
            ]
        }
    }


def get_evaluation_template() -> Dict[str, Any]:
    """获取模型评估配置模板"""
    return {
        "evaluation": {
            "tasks": [
                "text_generation",
                "question_answering",
                "classification"
            ],
            "metrics": [
                "bleu",
                "rouge",
                "bertscore",
                "perplexity",
                "accuracy",
                "f1"
            ],
            "batch_size": 8,
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "num_samples": 100,
            "enable_efficiency_metrics": True,
            "enable_quality_analysis": True
        },
        "efficiency_analysis": {
            "monitor_interval": 0.1,
            "batch_sizes": [1, 4, 8, 16],
            "num_runs": 10,
            "enable_memory_monitoring": True,
            "enable_flops_calculation": True
        }
    }


def get_benchmark_template() -> Dict[str, Any]:
    """获取基准测试配置模板"""
    return {
        "benchmarks": {
            "clue": {
                "name": "CLUE",
                "dataset_path": "clue",
                "tasks": [
                    "tnews",
                    "afqmc",
                    "cmnli",
                    "ocnli",
                    "wsc",
                    "csl"
                ],
                "evaluation_protocol": "official",
                "metrics": ["accuracy", "f1"]
            },
            "few_clue": {
                "name": "FewCLUE",
                "dataset_path": "few_clue",
                "tasks": ["tnews", "afqmc", "cmnli"],
                "evaluation_protocol": "few_shot",
                "metrics": ["accuracy", "f1"],
                "max_samples": 1000
            },
            "c_eval": {
                "name": "C-Eval",
                "dataset_path": "c_eval",
                "tasks": [
                    "high_school_physics",
                    "high_school_chemistry",
                    "high_school_biology",
                    "computer_science"
                ],
                "evaluation_protocol": "multiple_choice",
                "metrics": ["accuracy"]
            }
        },
        "benchmark_config": {
            "cache_dir": "./benchmarks",
            "auto_download": True,
            "max_samples_per_task": None,
            "enable_result_caching": True
        }
    }


def get_experiment_tracking_template() -> Dict[str, Any]:
    """获取实验跟踪配置模板"""
    return {
        "experiment_tracking": {
            "enabled": True,
            "experiment_dir": "./experiments",
            "auto_compare": True,
            "leaderboard_metric": "overall_score",
            "save_predictions": True,
            "save_model_outputs": False,
            "report_formats": [
                "json",
                "csv",
                "html"
            ]
        },
        "statistical_analysis": {
            "confidence_level": 0.95,
            "enable_significance_tests": True,
            "enable_trend_analysis": True,
            "visualization_types": [
                "bar",
                "radar",
                "heatmap",
                "line"
            ]
        }
    }


def get_training_monitoring_template() -> Dict[str, Any]:
    """获取训练监控配置模板"""
    return {
        "training_monitoring": {
            "patience": 10,
            "min_delta": 0.001,
            "monitor_metric": "val_loss",
            "mode": "min",
            "enable_early_stopping": True,
            "enable_lr_scheduling": True
        },
        "performance_monitoring": {
            "monitor_interval": 60,
            "enable_gpu_monitoring": True,
            "enable_memory_monitoring": True,
            "log_system_metrics": True
        }
    }


def get_batch_evaluation_template() -> Dict[str, Any]:
    """获取批量评估配置模板"""
    return {
        "batch_evaluation": {
            "models": [
                {
                    "name": "model_1",
                    "path": "/path/to/model1",
                    "tokenizer_path": "/path/to/tokenizer1",
                    "device": "cuda:0"
                },
                {
                    "name": "model_2", 
                    "path": "/path/to/model2",
                    "tokenizer_path": "/path/to/tokenizer2",
                    "device": "cuda:1"
                }
            ],
            "datasets": {
                "test_set": "/path/to/test_dataset",
                "validation_set": "/path/to/val_dataset"
            },
            "evaluation_config": {
                "batch_size": 8,
                "max_length": 512,
                "num_samples": 1000,
                "enable_efficiency_metrics": True
            },
            "parallel_config": {
                "max_workers": 4,
                "enable_parallel": True
            }
        },
        "output_config": {
            "output_dir": "./batch_results",
            "report_formats": ["html", "json", "csv"],
            "include_charts": True,
            "save_detailed_results": True
        }
    }


def get_integration_template() -> Dict[str, Any]:
    """获取系统集成配置模板"""
    return {
        "data_pipeline_integration": {
            "cache_dir": "./data_cache",
            "enable_quality_analysis": True,
            "enable_caching": True,
            "cache_optimization": {
                "max_size_mb": 1000,
                "max_age_days": 30
            }
        },
        "training_engine_integration": {
            "evaluation_frequency": 1,
            "enable_real_time_monitoring": True,
            "auto_evaluation_interval": 300,
            "save_best_model": True
        },
        "inference_tester_integration": {
            "enable_quality_analysis": True,
            "enable_efficiency_analysis": True,
            "optimization_targets": {
                "target_latency_ms": 100,
                "target_throughput": 50
            }
        }
    }


def get_complete_template() -> Dict[str, Any]:
    """获取完整配置模板"""
    template = {}
    
    # 合并所有模板
    template.update(get_data_split_template())
    template.update(get_evaluation_template())
    template.update(get_benchmark_template())
    template.update(get_experiment_tracking_template())
    template.update(get_training_monitoring_template())
    template.update(get_integration_template())
    
    # 添加全局配置
    template["global"] = {
        "project_name": "evaluation_project",
        "version": "1.0.0",
        "description": "模型评估项目配置",
        "author": "evaluation_team",
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    return template


def get_preset_configs() -> Dict[str, Dict[str, Any]]:
    """获取预设配置方案"""
    return {
        "quick_evaluation": {
            "description": "快速评估配置，适用于开发阶段的快速测试",
            "config": {
                "evaluation": {
                    "tasks": ["text_generation"],
                    "metrics": ["bleu", "rouge"],
                    "batch_size": 4,
                    "num_samples": 50,
                    "enable_efficiency_metrics": False
                }
            }
        },
        "comprehensive_evaluation": {
            "description": "全面评估配置，适用于正式的模型评估",
            "config": {
                "evaluation": {
                    "tasks": ["text_generation", "question_answering", "classification"],
                    "metrics": ["bleu", "rouge", "bertscore", "accuracy", "f1"],
                    "batch_size": 8,
                    "num_samples": 1000,
                    "enable_efficiency_metrics": True,
                    "enable_quality_analysis": True
                }
            }
        },
        "benchmark_testing": {
            "description": "基准测试配置，适用于标准基准测试",
            "config": {
                "benchmarks": {
                    "clue": {
                        "tasks": ["tnews", "afqmc", "cmnli"],
                        "evaluation_protocol": "official"
                    }
                }
            }
        },
        "production_monitoring": {
            "description": "生产环境监控配置，适用于生产模型监控",
            "config": {
                "training_monitoring": {
                    "patience": 5,
                    "monitor_metric": "val_loss",
                    "enable_early_stopping": True
                },
                "performance_monitoring": {
                    "monitor_interval": 30,
                    "enable_gpu_monitoring": True
                }
            }
        }
    }


def create_config_template(template_type: str, output_path: str, format_type: str = "yaml"):
    """
    创建配置模板
    
    Args:
        template_type: 模板类型
        output_path: 输出路径
        format_type: 格式类型 (yaml/json)
    """
    logger.info(f"创建配置模板: {template_type}")
    
    # 获取模板
    template_functions = {
        "data_split": get_data_split_template,
        "evaluation": get_evaluation_template,
        "benchmark": get_benchmark_template,
        "experiment_tracking": get_experiment_tracking_template,
        "training_monitoring": get_training_monitoring_template,
        "batch_evaluation": get_batch_evaluation_template,
        "integration": get_integration_template,
        "complete": get_complete_template
    }
    
    if template_type not in template_functions:
        raise ValueError(f"不支持的模板类型: {template_type}")
    
    template = template_functions[template_type]()
    
    # 保存模板
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if format_type.lower() == "yaml":
            yaml.dump(template, f, default_flow_style=False, allow_unicode=True, indent=2)
        elif format_type.lower() == "json":
            json.dump(template, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")
    
    logger.info(f"配置模板已保存: {output_file}")


def create_preset_config(preset_name: str, output_path: str, format_type: str = "yaml"):
    """
    创建预设配置
    
    Args:
        preset_name: 预设名称
        output_path: 输出路径
        format_type: 格式类型
    """
    logger.info(f"创建预设配置: {preset_name}")
    
    presets = get_preset_configs()
    
    if preset_name not in presets:
        raise ValueError(f"不支持的预设配置: {preset_name}")
    
    preset = presets[preset_name]
    
    # 保存预设配置
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if format_type.lower() == "yaml":
            yaml.dump(preset["config"], f, default_flow_style=False, allow_unicode=True, indent=2)
        elif format_type.lower() == "json":
            json.dump(preset["config"], f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")
    
    logger.info(f"预设配置已保存: {output_file}")
    logger.info(f"配置描述: {preset['description']}")


def validate_config(config_path: str) -> Dict[str, Any]:
    """
    验证配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        验证结果
    """
    logger.info(f"验证配置文件: {config_path}")
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        return {"valid": False, "error": "配置文件不存在"}
    
    try:
        # 加载配置
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                config = json.load(f)
            else:
                return {"valid": False, "error": f"不支持的配置文件格式: {config_file.suffix}"}
        
        # 基本验证
        validation_result = {
            "valid": True,
            "warnings": [],
            "suggestions": []
        }
        
        # 检查必要的配置节
        required_sections = ["evaluation", "data_split"]
        for section in required_sections:
            if section not in config:
                validation_result["warnings"].append(f"缺少推荐的配置节: {section}")
        
        # 检查评估配置
        if "evaluation" in config:
            eval_config = config["evaluation"]
            
            if "tasks" not in eval_config or not eval_config["tasks"]:
                validation_result["warnings"].append("评估任务列表为空")
            
            if "metrics" not in eval_config or not eval_config["metrics"]:
                validation_result["warnings"].append("评估指标列表为空")
            
            if eval_config.get("batch_size", 0) <= 0:
                validation_result["warnings"].append("批次大小应该大于0")
            
            if eval_config.get("num_samples", 0) <= 0:
                validation_result["warnings"].append("样本数量应该大于0")
        
        # 检查数据拆分配置
        if "data_split" in config:
            split_config = config["data_split"]
            
            ratios = [
                split_config.get("train_ratio", 0),
                split_config.get("val_ratio", 0),
                split_config.get("test_ratio", 0)
            ]
            
            if abs(sum(ratios) - 1.0) > 0.001:
                validation_result["warnings"].append("数据拆分比例之和应该等于1.0")
            
            if any(ratio <= 0 for ratio in ratios):
                validation_result["warnings"].append("数据拆分比例应该大于0")
        
        # 提供优化建议
        if not validation_result["warnings"]:
            validation_result["suggestions"].append("配置文件看起来很好！")
        else:
            validation_result["suggestions"].append("建议修复上述警告以获得最佳性能")
        
        return validation_result
        
    except Exception as e:
        return {"valid": False, "error": f"配置文件解析失败: {e}"}


def list_available_templates():
    """列出可用的模板"""
    templates = {
        "data_split": "数据拆分配置模板",
        "evaluation": "模型评估配置模板",
        "benchmark": "基准测试配置模板",
        "experiment_tracking": "实验跟踪配置模板",
        "training_monitoring": "训练监控配置模板",
        "batch_evaluation": "批量评估配置模板",
        "integration": "系统集成配置模板",
        "complete": "完整配置模板"
    }
    
    print("可用的配置模板:")
    for template_name, description in templates.items():
        print(f"  {template_name}: {description}")


def list_available_presets():
    """列出可用的预设配置"""
    presets = get_preset_configs()
    
    print("可用的预设配置:")
    for preset_name, preset_info in presets.items():
        print(f"  {preset_name}: {preset_info['description']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="配置模板生成器")
    parser.add_argument("--action", choices=["create", "preset", "validate", "list"], 
                       required=True, help="操作类型")
    parser.add_argument("--template", help="模板类型")
    parser.add_argument("--preset", help="预设配置名称")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--format", choices=["yaml", "json"], default="yaml", help="输出格式")
    parser.add_argument("--config", help="要验证的配置文件路径")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    try:
        if args.action == "create":
            if not args.template or not args.output:
                raise ValueError("创建模板需要指定 --template 和 --output 参数")
            
            create_config_template(args.template, args.output, args.format)
        
        elif args.action == "preset":
            if not args.preset or not args.output:
                raise ValueError("创建预设配置需要指定 --preset 和 --output 参数")
            
            create_preset_config(args.preset, args.output, args.format)
        
        elif args.action == "validate":
            if not args.config:
                raise ValueError("验证配置需要指定 --config 参数")
            
            result = validate_config(args.config)
            
            if result["valid"]:
                logger.info("配置文件验证通过")
                if result.get("warnings"):
                    logger.warning("发现以下警告:")
                    for warning in result["warnings"]:
                        logger.warning(f"  - {warning}")
                
                if result.get("suggestions"):
                    logger.info("建议:")
                    for suggestion in result["suggestions"]:
                        logger.info(f"  - {suggestion}")
            else:
                logger.error(f"配置文件验证失败: {result['error']}")
                sys.exit(1)
        
        elif args.action == "list":
            if args.template == "templates" or not args.template:
                list_available_templates()
            if args.preset == "presets" or not args.preset:
                list_available_presets()
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"操作失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
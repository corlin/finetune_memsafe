#!/usr/bin/env python3
"""
配置验证脚本

用于验证增强训练Pipeline的配置文件。
"""

import sys
import argparse
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from enhanced_config import load_enhanced_config_from_yaml, EnhancedApplicationConfig


def validate_config_file(config_path: str) -> bool:
    """
    验证配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        bool: 验证是否通过
    """
    try:
        print(f"正在验证配置文件: {config_path}")
        
        # 检查文件是否存在
        if not Path(config_path).exists():
            print(f"❌ 错误: 配置文件不存在: {config_path}")
            return False
        
        # 加载配置
        config = load_enhanced_config_from_yaml(config_path)
        print("✅ 配置文件加载成功")
        
        # 验证配置
        errors = config.validate_config()
        if errors:
            print("❌ 配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("✅ 配置验证通过")
        
        # 显示配置摘要
        print("\n📋 配置摘要:")
        print(f"  模型: {config.model_name}")
        print(f"  输出目录: {config.output_dir}")
        print(f"  数据目录: {config.data_dir}")
        print(f"  数据拆分: {config.enable_data_splitting}")
        if config.enable_data_splitting:
            print(f"    训练集比例: {config.train_ratio}")
            print(f"    验证集比例: {config.val_ratio}")
            print(f"    测试集比例: {config.test_ratio}")
        print(f"  训练轮数: {config.num_epochs}")
        print(f"  批次大小: {config.batch_size}")
        print(f"  学习率: {config.learning_rate}")
        print(f"  全面评估: {config.enable_comprehensive_evaluation}")
        if config.enable_comprehensive_evaluation:
            print(f"    评估任务: {config.evaluation_tasks}")
            print(f"    评估指标: {config.evaluation_metrics}")
        print(f"  实验跟踪: {config.enable_experiment_tracking}")
        if config.enable_experiment_tracking:
            print(f"    实验名称: {config.experiment_name or '自动生成'}")
            print(f"    实验标签: {config.experiment_tags}")
        print(f"  报告格式: {config.report_formats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


def create_sample_config(output_path: str):
    """
    创建示例配置文件
    
    Args:
        output_path: 输出路径
    """
    try:
        from enhanced_config import create_example_config
        
        config = create_example_config()
        
        # 转换为YAML格式的字典
        config_dict = {
            "model": {
                "name": config.model_name,
                "output_dir": config.output_dir
            },
            "data": {
                "data_dir": config.data_dir,
                "enable_splitting": config.enable_data_splitting,
                "train_ratio": config.train_ratio,
                "val_ratio": config.val_ratio,
                "test_ratio": config.test_ratio,
                "stratify_by": config.stratify_by,
                "split_seed": config.data_split_seed
            },
            "training": {
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "enable_validation": config.enable_validation_during_training
            },
            "evaluation": {
                "enable_comprehensive": config.enable_comprehensive_evaluation,
                "tasks": config.evaluation_tasks,
                "metrics": config.evaluation_metrics,
                "enable_efficiency": config.enable_efficiency_metrics,
                "enable_quality": config.enable_quality_analysis
            },
            "experiment": {
                "enable_tracking": config.enable_experiment_tracking,
                "name": config.experiment_name,
                "tags": config.experiment_tags
            },
            "reports": {
                "formats": config.report_formats,
                "enable_visualization": config.enable_visualization,
                "output_charts": config.output_charts
            }
        }
        
        # 保存为YAML文件
        import yaml
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        print(f"✅ 示例配置文件已创建: {output_path}")
        
    except Exception as e:
        print(f"❌ 创建示例配置失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="增强训练Pipeline配置验证工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("config_file", nargs="?", 
                       help="要验证的配置文件路径")
    parser.add_argument("--create-sample", type=str,
                       help="创建示例配置文件到指定路径")
    parser.add_argument("--list-examples", action="store_true",
                       help="列出可用的示例配置文件")
    
    args = parser.parse_args()
    
    try:
        if args.list_examples:
            print("📁 可用的示例配置文件:")
            examples = [
                ("enhanced_config_example.yaml", "完整配置示例，包含所有可用选项"),
                ("enhanced_config_simple.yaml", "简化配置示例，包含常用选项")
            ]
            
            for filename, description in examples:
                if Path(filename).exists():
                    print(f"  ✅ {filename} - {description}")
                else:
                    print(f"  ❌ {filename} - {description} (文件不存在)")
            
            return
        
        if args.create_sample:
            create_sample_config(args.create_sample)
            return
        
        if not args.config_file:
            print("❌ 错误: 请指定要验证的配置文件")
            print("使用 --help 查看帮助信息")
            sys.exit(1)
        
        # 验证配置文件
        success = validate_config_file(args.config_file)
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n用户中断了验证过程")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 验证工具执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
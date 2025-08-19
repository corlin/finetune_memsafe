#!/usr/bin/env python3
"""
增强训练Pipeline使用示例

展示如何使用增强的训练pipeline进行模型训练和评估。
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from enhanced_config import EnhancedApplicationConfig, create_example_config
from enhanced_main import EnhancedQwenFineTuningApplication


def example_basic_usage():
    """示例1: 基本使用"""
    print("=" * 60)
    print("示例1: 基本使用")
    print("=" * 60)
    
    # 创建基本配置
    config = EnhancedApplicationConfig(
        # 基本模型配置
        model_name="Qwen/Qwen3-4B-Thinking-2507",
        output_dir="./example_output_basic",
        data_dir="data/raw",
        
        # 训练配置
        num_epochs=2,  # 示例用较少轮数
        batch_size=4,
        learning_rate=5e-5,
        
        # 启用关键功能
        enable_data_splitting=True,
        enable_comprehensive_evaluation=True,
        enable_experiment_tracking=True,
        
        # 实验信息
        experiment_name="basic_example",
        experiment_tags=["demo", "basic"],
        
        # 报告配置
        report_formats=["html", "json"]
    )
    
    # 验证配置
    errors = config.validate_config()
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("配置验证通过")
    print(f"模型: {config.model_name}")
    print(f"输出目录: {config.output_dir}")
    print(f"训练轮数: {config.num_epochs}")
    print(f"数据拆分: {config.enable_data_splitting}")
    print(f"全面评估: {config.enable_comprehensive_evaluation}")
    
    # 创建并运行应用程序
    try:
        app = EnhancedQwenFineTuningApplication(config)
        success = app.run_enhanced_pipeline()
        
        if success:
            print("✅ 基本示例执行成功")
        else:
            print("❌ 基本示例执行失败")
        
        return success
        
    except Exception as e:
        print(f"❌ 基本示例执行异常: {e}")
        return False


def example_custom_data_split():
    """示例2: 自定义数据拆分"""
    print("\n" + "=" * 60)
    print("示例2: 自定义数据拆分")
    print("=" * 60)
    
    # 创建自定义数据拆分配置
    config = EnhancedApplicationConfig(
        # 基本配置
        model_name="Qwen/Qwen3-4B-Thinking-2507",
        output_dir="./example_output_custom_split",
        data_dir="data/raw",
        
        # 自定义数据拆分
        enable_data_splitting=True,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        stratify_by=None,  # 可以设置为具体字段名进行分层抽样
        data_split_seed=123,  # 自定义随机种子
        
        # 训练配置
        num_epochs=1,  # 示例用1轮
        batch_size=2,
        
        # 验证配置
        enable_validation_during_training=True,
        validation_steps=50,
        save_validation_metrics=True,
        
        # 实验信息
        experiment_name="custom_split_example",
        experiment_tags=["demo", "custom_split"],
        
        # 只生成JSON报告以节省时间
        report_formats=["json"]
    )
    
    print(f"数据拆分比例: {config.train_ratio}:{config.val_ratio}:{config.test_ratio}")
    print(f"随机种子: {config.data_split_seed}")
    print(f"验证间隔: {config.validation_steps} 步")
    
    # 只运行数据拆分部分作为示例
    try:
        app = EnhancedQwenFineTuningApplication(config)
        
        # 只执行数据拆分步骤
        if app._split_data():
            print("✅ 数据拆分示例执行成功")
            print(f"训练集: {len(app.train_dataset)} 样本")
            print(f"验证集: {len(app.val_dataset)} 样本")
            print(f"测试集: {len(app.test_dataset)} 样本")
            return True
        else:
            print("❌ 数据拆分示例执行失败")
            return False
            
    except Exception as e:
        print(f"❌ 数据拆分示例执行异常: {e}")
        return False


def example_evaluation_focus():
    """示例3: 评估重点配置"""
    print("\n" + "=" * 60)
    print("示例3: 评估重点配置")
    print("=" * 60)
    
    # 创建评估重点配置
    config = EnhancedApplicationConfig(
        # 基本配置
        model_name="Qwen/Qwen3-4B-Thinking-2507",
        output_dir="./example_output_evaluation",
        data_dir="data/raw",
        
        # 简化训练（重点在评估）
        num_epochs=1,
        batch_size=2,
        
        # 重点评估配置
        enable_comprehensive_evaluation=True,
        evaluation_tasks=["text_generation"],
        evaluation_metrics=["bleu", "rouge", "accuracy"],
        enable_efficiency_metrics=True,
        enable_quality_analysis=True,
        evaluation_batch_size=2,
        evaluation_num_samples=20,  # 限制样本数以节省时间
        
        # 报告配置
        report_formats=["html", "json"],
        enable_visualization=True,
        output_charts=True,
        
        # 实验信息
        experiment_name="evaluation_focus_example",
        experiment_tags=["demo", "evaluation"],
        
        # 启用回退模式
        fallback_to_basic_mode=True
    )
    
    print(f"评估任务: {config.evaluation_tasks}")
    print(f"评估指标: {config.evaluation_metrics}")
    print(f"效率分析: {config.enable_efficiency_metrics}")
    print(f"质量分析: {config.enable_quality_analysis}")
    print(f"评估样本数: {config.evaluation_num_samples}")
    
    # 这个示例主要展示配置，不实际运行完整pipeline
    print("✅ 评估配置示例创建成功")
    print("（实际运行请使用: python enhanced_main.py --config your_config.yaml）")
    
    return True


def example_config_file_usage():
    """示例4: 配置文件使用"""
    print("\n" + "=" * 60)
    print("示例4: 配置文件使用")
    print("=" * 60)
    
    # 创建示例配置并保存为YAML文件
    config = create_example_config()
    
    # 修改一些配置用于示例
    config.experiment_name = "config_file_example"
    config.experiment_tags = ["demo", "config_file"]
    config.num_epochs = 1
    config.batch_size = 2
    
    # 保存配置到文件
    config_dict = {
        "model": {
            "name": config.model_name,
            "output_dir": "./example_output_config_file"
        },
        "data": {
            "data_dir": config.data_dir,
            "enable_splitting": config.enable_data_splitting,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio
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
            "metrics": config.evaluation_metrics
        },
        "experiment": {
            "enable_tracking": config.enable_experiment_tracking,
            "name": config.experiment_name,
            "tags": config.experiment_tags
        },
        "reports": {
            "formats": config.report_formats
        }
    }
    
    # 保存为YAML文件
    try:
        import yaml
        config_file_path = "example_config_demo.yaml"
        
        with open(config_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        print(f"✅ 示例配置文件已创建: {config_file_path}")
        print("使用方法:")
        print(f"  python enhanced_main.py --config {config_file_path}")
        print(f"  python validate_config.py {config_file_path}")
        
        return True
        
    except ImportError:
        print("❌ 需要安装PyYAML: pip install pyyaml")
        return False
    except Exception as e:
        print(f"❌ 创建配置文件失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 增强训练Pipeline使用示例")
    print("本示例展示如何使用增强的训练pipeline")
    print()
    
    # 检查数据目录
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("⚠️  警告: 数据目录 'data/raw' 不存在")
        print("请确保有训练数据，或者修改示例中的data_dir参数")
        print()
    
    results = []
    
    # 运行示例
    try:
        # 示例1: 基本使用（如果有数据的话）
        if data_dir.exists():
            results.append(("基本使用", example_basic_usage()))
            results.append(("自定义数据拆分", example_custom_data_split()))
        else:
            print("跳过需要数据的示例...")
        
        # 示例3和4不需要实际数据
        results.append(("评估重点配置", example_evaluation_focus()))
        results.append(("配置文件使用", example_config_file_usage()))
        
    except KeyboardInterrupt:
        print("\n用户中断了示例执行")
        return
    
    # 显示结果摘要
    print("\n" + "=" * 60)
    print("示例执行结果摘要")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name}: {status}")
    
    successful_count = sum(1 for _, success in results if success)
    print(f"\n总计: {successful_count}/{len(results)} 个示例成功")
    
    # 提供下一步建议
    print("\n📚 下一步建议:")
    print("1. 查看生成的输出文件和报告")
    print("2. 修改配置文件尝试不同设置")
    print("3. 使用validate_config.py验证配置")
    print("4. 查看README_enhanced.md了解详细用法")
    print("5. 运行完整的训练pipeline:")
    print("   python enhanced_main.py --config enhanced_config_simple.yaml")


if __name__ == "__main__":
    main()
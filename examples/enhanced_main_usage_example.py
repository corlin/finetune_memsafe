"""
优化后的 enhanced_main.py 使用示例

展示如何使用增强的数据字段检测能力来处理各种数据格式。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_main import EnhancedQwenFineTuningApplication
from enhanced_config import EnhancedApplicationConfig


def create_sample_config():
    """创建示例配置"""
    config = EnhancedApplicationConfig(
        # 基本配置
        model_name="Qwen/Qwen2.5-7B-Instruct",
        data_dir="data/sample_data",
        output_dir="output/enhanced_example",
        
        # 启用增强功能
        enable_data_splitting=True,
        enable_comprehensive_evaluation=True,
        enable_experiment_tracking=True,
        fallback_to_basic_mode=True,
        
        # 数据拆分配置
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        
        # 评估配置
        evaluation_tasks=["text_generation", "question_answering"],
        evaluation_metrics=["bleu", "rouge", "bertscore"],
        evaluation_batch_size=8,
        evaluation_num_samples=100,
        
        # 训练配置
        batch_size=4,
        num_epochs=3,
        learning_rate=2e-5,
        max_sequence_length=512,
        
        # 验证配置
        enable_validation_during_training=True,
        validation_steps=100,
        save_validation_metrics=True,
        
        # 内存优化
        max_memory_gb=8,
        gradient_accumulation_steps=4,
        
        # 报告配置
        report_formats=["html", "json"],
        output_charts=True
    )
    
    return config


def main():
    """主函数"""
    print("增强的Qwen3微调系统使用示例")
    print("=" * 50)
    
    try:
        # 创建配置
        config = create_sample_config()
        
        # 创建增强应用程序
        app = EnhancedQwenFineTuningApplication(config)
        
        print("应用程序初始化完成")
        print(f"增强功能状态:")
        print(f"  - 数据拆分: {'✓' if config.enable_data_splitting else '✗'}")
        print(f"  - 增强评估: {'✓' if config.enable_comprehensive_evaluation else '✗'}")
        print(f"  - 实验跟踪: {'✓' if config.enable_experiment_tracking else '✗'}")
        print(f"  - 错误恢复: {'✓' if config.fallback_to_basic_mode else '✗'}")
        
        # 运行增强流程
        print("\n开始运行增强微调流程...")
        success = app.run_enhanced_pipeline()
        
        if success:
            print("✓ 增强微调流程成功完成！")
            print(f"输出目录: {config.output_dir}")
            
            # 显示生成的文件
            output_path = Path(config.output_dir)
            if output_path.exists():
                print("\n生成的文件:")
                for file_path in output_path.rglob("*.json"):
                    print(f"  - {file_path.relative_to(output_path)}")
                for file_path in output_path.rglob("*.html"):
                    print(f"  - {file_path.relative_to(output_path)}")
        else:
            print("✗ 增强微调流程失败")
            
    except KeyboardInterrupt:
        print("\n用户中断了程序执行")
    except Exception as e:
        print(f"✗ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_data_format_handling():
    """演示数据格式处理能力"""
    print("\n" + "=" * 50)
    print("数据格式处理能力演示")
    print("=" * 50)
    
    # 模拟不同的数据格式
    data_formats = {
        "标准格式": {
            "description": "标准的text/target格式",
            "sample": {
                "text": "Hello world",
                "target": "Bonjour monde"
            }
        },
        "问答格式": {
            "description": "问答任务格式",
            "sample": {
                "question": "What is AI?",
                "context": "AI stands for Artificial Intelligence",
                "answer": "Artificial Intelligence"
            }
        },
        "自定义格式": {
            "description": "自定义字段名格式",
            "sample": {
                "prompt": "Translate to French",
                "response": "Traduire en français"
            }
        },
        "原始格式": {
            "description": "原始的token格式（会被自动处理）",
            "sample": {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [1, 2, 3, 4, 5]
            }
        }
    }
    
    print("支持的数据格式:")
    for format_name, format_info in data_formats.items():
        print(f"\n{format_name}:")
        print(f"  描述: {format_info['description']}")
        print(f"  示例: {format_info['sample']}")
    
    print(f"\n增强的数据处理功能:")
    print(f"  ✓ 智能字段检测 - 自动识别输入和目标字段")
    print(f"  ✓ 灵活字段映射 - 支持自定义字段名称")
    print(f"  ✓ 数据质量验证 - 检查数据完整性和一致性")
    print(f"  ✓ 自动错误恢复 - 多级降级处理机制")
    print(f"  ✓ 实时诊断监控 - 详细的处理统计和建议")


def show_configuration_options():
    """显示配置选项"""
    print("\n" + "=" * 50)
    print("增强评估配置选项")
    print("=" * 50)
    
    print("数据处理配置:")
    print("  field_mapping:")
    print("    text_generation:")
    print("      input_fields: ['text', 'input', 'prompt']")
    print("      target_fields: ['target', 'answer', 'output']")
    print("    question_answering:")
    print("      input_fields: ['question', 'query']")
    print("      context_fields: ['context', 'passage']")
    print("      target_fields: ['answer', 'target']")
    print()
    print("  validation:")
    print("    min_valid_samples_ratio: 0.05  # 最小有效样本比例")
    print("    enable_data_cleaning: true     # 启用数据清洗")
    print("    enable_fallback: true          # 启用降级处理")
    print()
    print("  diagnostics:")
    print("    enable_detailed_logging: false # 详细日志")
    print("    log_batch_statistics: true     # 批次统计")
    print("    save_processing_report: true   # 保存报告")


if __name__ == "__main__":
    # 运行主程序
    main()
    
    # 演示数据格式处理能力
    demonstrate_data_format_handling()
    
    # 显示配置选项
    show_configuration_options()
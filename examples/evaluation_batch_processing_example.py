"""
评估批次处理修复功能使用示例

展示如何使用新的数据处理功能来解决批次数据为空的问题。
"""

import logging
from datasets import Dataset
from unittest.mock import Mock

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入新的评估功能
from src.evaluation import (
    EvaluationConfig, DataPreprocessor, EvaluationEngine,
    load_evaluation_config, ConfigValidator
)
from src.evaluation.compatibility import create_enhanced_evaluation_engine


def create_mock_model_and_tokenizer():
    """创建模拟的模型和分词器用于演示"""
    mock_model = Mock()
    mock_model.parameters.return_value = [Mock(device="cpu")]
    mock_model.config.is_encoder_decoder = False
    mock_model.generate.return_value = Mock()
    
    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.return_value = {
        "input_ids": Mock(to=Mock(return_value=Mock())),
        "attention_mask": Mock(to=Mock(return_value=Mock()))
    }
    mock_tokenizer.batch_decode.return_value = ["Generated response"] * 4
    
    return mock_model, mock_tokenizer


def example_1_basic_usage():
    """示例1: 基本使用方法"""
    print("\n" + "="*60)
    print("示例1: 基本使用方法")
    print("="*60)
    
    # 创建有问题的数据集（类似原始问题中的格式）
    problematic_dataset = Dataset.from_dict({
        "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "attention_mask": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        "labels": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    })
    
    print(f"原始数据集字段: {list(problematic_dataset.column_names)}")
    print(f"数据集大小: {len(problematic_dataset)}")
    
    # 使用增强的评估引擎
    engine = create_enhanced_evaluation_engine()
    
    # 诊断数据集问题
    diagnosis = engine.diagnose_dataset(problematic_dataset, "text_generation")
    
    print("\n诊断结果:")
    print(f"批次信息: {diagnosis['batch_info']}")
    print(f"建议: {diagnosis['recommendations']}")
    
    # 创建模拟模型进行评估
    model, tokenizer = create_mock_model_and_tokenizer()
    
    # 执行评估（会自动处理数据格式问题）
    datasets = {"text_generation": problematic_dataset}
    result = engine.evaluate_model_with_diagnostics(
        model, tokenizer, datasets, "example_model"
    )
    
    print(f"\n评估完成:")
    print(f"处理统计: {result['processing_stats']}")


def example_2_different_data_formats():
    """示例2: 处理不同数据格式"""
    print("\n" + "="*60)
    print("示例2: 处理不同数据格式")
    print("="*60)
    
    # 测试不同的数据格式
    test_datasets = {
        "标准格式": Dataset.from_dict({
            "text": ["Hello world", "Good morning"],
            "target": ["Bonjour monde", "Bonjour"]
        }),
        
        "自定义字段": Dataset.from_dict({
            "prompt": ["Hello world", "Good morning"],
            "response": ["Bonjour monde", "Bonjour"]
        }),
        
        "问答格式": Dataset.from_dict({
            "question": ["What is AI?", "How does ML work?"],
            "context": ["AI is artificial intelligence", "ML uses algorithms"],
            "answer": ["Artificial Intelligence", "Machine Learning"]
        }),
        
        "包含空值": Dataset.from_dict({
            "text": ["Hello", "", None, "World"],
            "target": ["Hi", "Empty", "Null", "Earth"]
        })
    }
    
    engine = create_enhanced_evaluation_engine()
    
    for format_name, dataset in test_datasets.items():
        print(f"\n处理 {format_name}:")
        print(f"  字段: {list(dataset.column_names)}")
        
        # 诊断数据集
        task_name = "question_answering" if "question" in dataset.column_names else "text_generation"
        diagnosis = engine.diagnose_dataset(dataset, task_name)
        
        print(f"  推荐输入字段: {diagnosis.get('field_mapping_info', {}).get('recommended_input_field')}")
        print(f"  数据质量问题: {len(diagnosis.get('validation_result', {}).get('issues', []))}")
        
        if diagnosis.get('recommendations'):
            print(f"  建议: {diagnosis['recommendations'][0]}")


def example_3_configuration_usage():
    """示例3: 配置使用"""
    print("\n" + "="*60)
    print("示例3: 配置使用")
    print("="*60)
    
    # 创建自定义配置
    custom_config = EvaluationConfig(
        batch_size=8,
        data_processing={
            "field_mapping": {
                "custom_task": {
                    "input_fields": ["custom_input", "data"],
                    "target_fields": ["custom_target", "expected"]
                }
            },
            "validation": {
                "min_valid_samples_ratio": 0.2,
                "enable_data_cleaning": True,
                "enable_fallback": True
            },
            "diagnostics": {
                "enable_detailed_logging": True,
                "log_batch_statistics": True
            }
        }
    )
    
    # 验证配置
    validator = ConfigValidator()
    is_valid = validator.validate_processing_config(custom_config.data_processing)
    
    print(f"配置验证结果: {'有效' if is_valid else '无效'}")
    
    if not is_valid:
        print(f"验证错误: {validator.get_validation_errors()}")
    
    warnings = validator.get_validation_warnings()
    if warnings:
        print(f"验证警告: {warnings}")
    
    # 使用自定义配置创建引擎
    engine = create_enhanced_evaluation_engine(config_data={"data_processing": custom_config.data_processing})
    
    # 测试自定义任务
    custom_dataset = Dataset.from_dict({
        "custom_input": ["Input 1", "Input 2"],
        "custom_target": ["Target 1", "Target 2"]
    })
    
    diagnosis = engine.diagnose_dataset(custom_dataset, "custom_task")
    print(f"\n自定义任务诊断:")
    print(f"  推荐字段: {diagnosis.get('field_mapping_info', {}).get('recommended_input_field')}")


def example_4_performance_monitoring():
    """示例4: 性能监控"""
    print("\n" + "="*60)
    print("示例4: 性能监控")
    print("="*60)
    
    # 创建大数据集
    large_dataset = Dataset.from_dict({
        "text": [f"Sample text {i}" for i in range(1000)],
        "target": [f"Target {i}" for i in range(1000)]
    })
    
    # 配置启用详细监控
    config = EvaluationConfig(
        batch_size=50,
        data_processing={
            "diagnostics": {
                "enable_detailed_logging": False,  # 避免过多日志
                "log_batch_statistics": True,
                "save_processing_report": True
            }
        }
    )
    
    engine = create_enhanced_evaluation_engine(config_data=config.to_dict())
    
    # 创建数据预处理器进行性能测试
    preprocessor = DataPreprocessor(config)
    
    import time
    start_time = time.time()
    
    # 处理多个批次
    batch_count = 0
    for i in range(0, min(500, len(large_dataset)), config.batch_size):
        batch = large_dataset[i:i+config.batch_size]
        result = preprocessor.preprocess_batch(batch, "text_generation")
        batch_count += 1
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 获取性能统计
    stats = preprocessor.get_processing_statistics()
    
    print(f"性能统计:")
    print(f"  处理批次数: {batch_count}")
    print(f"  总处理时间: {processing_time:.2f}秒")
    print(f"  平均批次时间: {processing_time/batch_count*1000:.2f}毫秒")
    print(f"  处理速度: {stats['total_samples_processed']/processing_time:.1f} samples/s")
    print(f"  成功率: {stats['success_rate']:.2%}")
    print(f"  有效样本率: {stats['valid_sample_rate']:.2%}")
    
    # 生成诊断报告
    report = preprocessor.generate_processing_report()
    print(f"\n诊断报告生成完成，包含 {len(report)} 个部分")


def example_5_error_recovery():
    """示例5: 错误恢复机制"""
    print("\n" + "="*60)
    print("示例5: 错误恢复机制")
    print("="*60)
    
    # 创建各种有问题的数据集
    error_datasets = {
        "完全空数据": Dataset.from_dict({}),
        
        "字段类型错误": Dataset.from_dict({
            "text": "not a list",  # 应该是列表
            "target": ["target1", "target2"]
        }),
        
        "长度不一致": Dataset.from_dict({
            "text": ["text1", "text2", "text3"],
            "target": ["target1", "target2"]  # 长度不匹配
        }),
        
        "全空值": Dataset.from_dict({
            "text": ["", "", ""],
            "target": ["", "", ""]
        }),
        
        "混合数据类型": Dataset.from_dict({
            "text": ["string", 123, None, True, [1, 2]],
            "target": ["t1", "t2", "t3", "t4", "t5"]
        })
    }
    
    engine = create_enhanced_evaluation_engine()
    
    for error_type, dataset in error_datasets.items():
        print(f"\n处理 {error_type}:")
        
        try:
            diagnosis = engine.diagnose_dataset(dataset, "text_generation")
            
            print(f"  诊断完成: {len(diagnosis.get('recommendations', []))} 个建议")
            
            if diagnosis.get('recommendations'):
                print(f"  主要建议: {diagnosis['recommendations'][0]}")
            
            # 尝试预处理
            if len(dataset) > 0:
                preprocessor = DataPreprocessor(engine.config)
                sample_batch = dataset[:min(4, len(dataset))]
                result = preprocessor.preprocess_batch(sample_batch, "text_generation")
                
                print(f"  预处理结果: {len(result.inputs)} 个有效输入, {len(result.warnings)} 个警告")
            
        except Exception as e:
            print(f"  处理失败: {e}")


def main():
    """主函数"""
    print("评估批次处理修复功能演示")
    print("这些示例展示了如何使用新功能解决批次数据为空的问题")
    
    try:
        example_1_basic_usage()
        example_2_different_data_formats()
        example_3_configuration_usage()
        example_4_performance_monitoring()
        example_5_error_recovery()
        
        print("\n" + "="*60)
        print("所有示例执行完成！")
        print("="*60)
        
    except Exception as e:
        logger.error(f"示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
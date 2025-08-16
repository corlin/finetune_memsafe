"""
推理测试系统演示脚本

展示如何使用InferenceTester类来测试微调后的模型
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference_tester import InferenceTester
from src.memory_optimizer import MemoryOptimizer
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demo_inference_testing():
    """演示推理测试功能"""
    print("=== Qwen3 推理测试系统演示 ===\n")
    
    # 1. 创建内存优化器和推理测试器
    print("1. 初始化推理测试器...")
    
    # 检查CUDA可用性
    import torch
    if torch.cuda.is_available():
        print("   检测到CUDA，使用GPU内存优化器")
        memory_optimizer = MemoryOptimizer(max_memory_gb=13.0)
    else:
        print("   未检测到CUDA，使用模拟内存优化器进行演示")
        from unittest.mock import Mock
        memory_optimizer = Mock()
        memory_optimizer.cleanup_gpu_memory = Mock()
        memory_optimizer.check_memory_safety = Mock(return_value=True)
        memory_optimizer.log_memory_status = Mock()
    
    tester = InferenceTester(memory_optimizer=memory_optimizer)
    
    # 2. 演示提示格式化
    print("\n2. 演示Qwen3提示格式化...")
    test_prompt = "请解释什么是机器学习？"
    formatted = tester._format_prompt_for_qwen(test_prompt)
    print(f"原始提示: {test_prompt}")
    print(f"格式化后:\n{formatted}")
    
    # 3. 演示响应质量验证
    print("\n3. 演示响应质量验证...")
    
    # 高质量响应示例
    good_response = "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测或决策，而无需明确编程。"
    quality_good = tester.validate_response_quality(good_response)
    print(f"高质量响应: {good_response}")
    print(f"质量分数: {quality_good['overall_score']}")
    print(f"质量指标: 内容完整={quality_good['has_content']}, 长度合适={quality_good['min_length_met']}, 无重复={quality_good['no_repetition']}")
    
    # 低质量响应示例
    poor_response = "错误 错误 错误 错误 错误"
    quality_poor = tester.validate_response_quality(poor_response)
    print(f"\n低质量响应: {poor_response}")
    print(f"质量分数: {quality_poor['overall_score']}")
    print(f"质量指标: 内容完整={quality_poor['has_content']}, 长度合适={quality_poor['min_length_met']}, 无重复={quality_poor['no_repetition']}")
    
    # 4. 演示错误处理
    print("\n4. 演示错误处理...")
    test_error = RuntimeError("CUDA out of memory")
    error_info = tester.handle_inference_failure(test_error, "测试提示")
    print(f"错误类型: {error_info['error_type']}")
    print(f"建议数量: {len(error_info['suggestions'])}")
    print(f"建议: {error_info['suggestions'][:3]}")  # 显示前3个建议
    
    # 5. 演示默认测试提示
    print("\n5. 默认测试提示集:")
    default_prompts = [
        "请解释什么是机器学习？",
        "如何优化深度学习模型的性能？",
        "Python中如何处理异常？",
        "请写一个简单的排序算法。",
        "什么是RESTful API？"
    ]
    
    for i, prompt in enumerate(default_prompts, 1):
        print(f"  {i}. {prompt}")
    
    print("\n=== 演示完成 ===")
    print("\n使用说明:")
    print("1. 要测试实际的微调模型，请使用 tester.load_finetuned_model(model_path)")
    print("2. 使用 tester.test_inference(prompt) 进行单次推理测试")
    print("3. 使用 tester.test_model_with_multiple_prompts() 进行批量测试")
    print("4. 使用 tester.validate_model_quality() 进行整体质量评估")
    print("5. 使用 tester.run_comprehensive_test() 进行完整的模型测试")

def demo_quality_recommendations():
    """演示质量改进建议生成"""
    print("\n=== 质量改进建议演示 ===")
    
    # 使用模拟内存优化器
    import torch
    from unittest.mock import Mock
    
    if torch.cuda.is_available():
        memory_optimizer = MemoryOptimizer(max_memory_gb=13.0)
    else:
        memory_optimizer = Mock()
        memory_optimizer.cleanup_gpu_memory = Mock()
        memory_optimizer.check_memory_safety = Mock(return_value=True)
        memory_optimizer.log_memory_status = Mock()
    
    tester = InferenceTester(memory_optimizer=memory_optimizer)
    
    # 模拟不同的质量场景
    scenarios = [
        {
            "name": "高质量模型",
            "avg_quality": 0.85,
            "success_rate": 95.0,
            "quality_pass_rate": 90.0,
            "quality_results": [{"coherent_structure": True, "no_repetition": True, "min_length_met": True}] * 10
        },
        {
            "name": "中等质量模型",
            "avg_quality": 0.65,
            "success_rate": 85.0,
            "quality_pass_rate": 70.0,
            "quality_results": [{"coherent_structure": True, "no_repetition": False, "min_length_met": True}] * 10
        },
        {
            "name": "低质量模型",
            "avg_quality": 0.35,
            "success_rate": 60.0,
            "quality_pass_rate": 30.0,
            "quality_results": [{"coherent_structure": False, "no_repetition": False, "min_length_met": False}] * 10
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        recommendations = tester._generate_quality_recommendations(
            scenario["avg_quality"],
            scenario["success_rate"],
            scenario["quality_pass_rate"],
            scenario["quality_results"]
        )
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

if __name__ == "__main__":
    try:
        demo_inference_testing()
        demo_quality_recommendations()
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
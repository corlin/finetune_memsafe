#!/usr/bin/env python3
"""
演示内存感知错误处理功能的示例脚本。

这个脚本展示了如何使用自定义内存异常类和错误恢复功能。
"""

import logging
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory_exceptions import (
    OutOfMemoryError,
    InsufficientMemoryError, 
    MemoryLeakError,
    MemoryErrorHandler,
    OptimizationSuggestion
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demo_custom_exceptions():
    """演示自定义内存异常类的功能。"""
    print("=== 演示自定义内存异常类 ===")
    
    # 1. OutOfMemoryError 示例
    print("\n1. OutOfMemoryError 示例:")
    try:
        raise OutOfMemoryError(
            "GPU内存不足",
            current_usage_gb=12.5,
            required_gb=2.0,
            available_gb=0.5
        )
    except OutOfMemoryError as e:
        print(f"错误: {e}")
        print(f"当前使用: {e.current_usage_gb}GB")
        print(f"需要: {e.required_gb}GB")
        print(f"可用: {e.available_gb}GB")
        print("优化建议:")
        for i, suggestion in enumerate(e.suggestions, 1):
            print(f"  {i}. [{suggestion.priority}] {suggestion.action}: {suggestion.description}")
    
    # 2. InsufficientMemoryError 示例
    print("\n2. InsufficientMemoryError 示例:")
    try:
        raise InsufficientMemoryError(
            "内存不足以进行操作",
            current_usage_gb=11.0,
            limit_gb=13.0
        )
    except InsufficientMemoryError as e:
        print(f"错误: {e}")
        print(f"当前使用: {e.current_usage_gb}GB")
        print(f"内存限制: {e.limit_gb}GB")
        print("优化建议:")
        for i, suggestion in enumerate(e.suggestions, 1):
            print(f"  {i}. [{suggestion.priority}] {suggestion.action}: {suggestion.description}")
    
    # 3. MemoryLeakError 示例
    print("\n3. MemoryLeakError 示例:")
    try:
        raise MemoryLeakError(
            "检测到内存泄漏",
            current_usage_gb=10.0,
            previous_usage_gb=6.0
        )
    except MemoryLeakError as e:
        print(f"错误: {e}")
        print(f"当前使用: {e.current_usage_gb}GB")
        print(f"之前使用: {e.previous_usage_gb}GB")
        print(f"增长: {e.growth_gb}GB")
        print("优化建议:")
        for i, suggestion in enumerate(e.suggestions, 1):
            print(f"  {i}. [{suggestion.priority}] {suggestion.action}: {suggestion.description}")


def demo_error_handler():
    """演示内存错误处理器的功能。"""
    print("\n=== 演示内存错误处理器 ===")
    
    # 创建模拟的内存优化器
    class MockMemoryOptimizer:
        def __init__(self):
            self.cleanup_called = 0
            self.memory_safe = True
        
        def cleanup_gpu_memory(self):
            self.cleanup_called += 1
            print(f"  执行GPU内存清理 (第{self.cleanup_called}次)")
        
        def check_memory_safety(self):
            return self.memory_safe
        
        def get_memory_status(self):
            class MockStatus:
                allocated_gb = 8.0  # 模拟清理后的内存使用
            return MockStatus()
    
    mock_optimizer = MockMemoryOptimizer()
    handler = MemoryErrorHandler(mock_optimizer)
    
    # 1. 处理OutOfMemoryError
    print("\n1. 处理OutOfMemoryError:")
    oom_error = OutOfMemoryError(
        "模拟OOM错误",
        current_usage_gb=12.0,
        required_gb=1.0,
        available_gb=0.5
    )
    
    success = handler.handle_out_of_memory(oom_error, auto_recover=True)
    print(f"  自动恢复结果: {'成功' if success else '失败'}")
    print(f"  清理调用次数: {mock_optimizer.cleanup_called}")
    
    # 2. 内存警告检查
    print("\n2. 内存警告检查:")
    handler.check_memory_warnings(
        current_usage_gb=11.0,
        limit_gb=13.0,
        warning_threshold=0.8
    )
    
    # 3. 内存使用跟踪
    print("\n3. 内存使用跟踪:")
    usage_values = [5.0, 6.0, 7.0, 8.0, 9.0, 10.5]
    for usage in usage_values:
        handler.track_memory_usage(usage)
    print(f"  跟踪的内存历史: {handler._memory_history}")
    
    # 4. 恢复配置建议
    print("\n4. 恢复配置建议:")
    configs = {
        "oom": handler.get_recovery_config("oom"),
        "insufficient": handler.get_recovery_config("insufficient"),
        "leak": handler.get_recovery_config("leak")
    }
    
    for error_type, config in configs.items():
        print(f"  {error_type}错误的恢复配置:")
        for key, value in config.items():
            print(f"    {key}: {value}")


def demo_proactive_warnings():
    """演示主动内存警告功能。"""
    print("\n=== 演示主动内存警告 ===")
    
    handler = MemoryErrorHandler()
    
    # 测试不同的内存使用情况
    test_cases = [
        (8.0, 13.0, "正常使用"),
        (10.5, 13.0, "警告级别"),
        (12.5, 13.0, "危险级别")
    ]
    
    for current_gb, limit_gb, description in test_cases:
        print(f"\n{description} ({current_gb}GB / {limit_gb}GB):")
        handler.check_memory_warnings(current_gb, limit_gb, warning_threshold=0.8)


if __name__ == "__main__":
    print("内存感知错误处理演示")
    print("=" * 50)
    
    try:
        demo_custom_exceptions()
        demo_error_handler()
        demo_proactive_warnings()
        
        print("\n" + "=" * 50)
        print("演示完成！所有内存错误处理功能都正常工作。")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
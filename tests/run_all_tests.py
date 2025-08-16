"""
测试运行器 - 运行所有测试并生成报告

这个脚本运行所有单元测试和集成测试，并生成详细的测试报告。
"""

import unittest
import sys
import os
from pathlib import Path
import time
from io import StringIO

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestResult:
    """测试结果类"""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.skipped_tests = 0
        self.failures = []
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    @property
    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
    
    @property
    def success_rate(self):
        if self.total_tests == 0:
            return 0
        return (self.passed_tests / self.total_tests) * 100


def run_test_module(module_name):
    """运行单个测试模块"""
    print(f"\n{'='*60}")
    print(f"运行测试模块: {module_name}")
    print(f"{'='*60}")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    
    try:
        # 导入测试模块
        test_module = __import__(f"tests.{module_name}", fromlist=[module_name])
        suite = loader.loadTestsFromModule(test_module)
        
        # 运行测试
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2,
            buffer=True
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # 打印结果
        print(stream.getvalue())
        
        # 创建结果对象
        test_result = TestResult()
        test_result.total_tests = result.testsRun
        test_result.failed_tests = len(result.failures)
        test_result.error_tests = len(result.errors)
        test_result.skipped_tests = len(result.skipped)
        test_result.passed_tests = (test_result.total_tests - 
                                   test_result.failed_tests - 
                                   test_result.error_tests - 
                                   test_result.skipped_tests)
        test_result.failures = result.failures
        test_result.errors = result.errors
        test_result.start_time = start_time
        test_result.end_time = end_time
        
        return test_result
        
    except ImportError as e:
        print(f"无法导入测试模块 {module_name}: {e}")
        return None
    except Exception as e:
        print(f"运行测试模块 {module_name} 时出错: {e}")
        return None


def print_summary(results):
    """打印测试总结"""
    print(f"\n{'='*80}")
    print("测试总结")
    print(f"{'='*80}")
    
    total_tests = sum(r.total_tests for r in results.values() if r)
    total_passed = sum(r.passed_tests for r in results.values() if r)
    total_failed = sum(r.failed_tests for r in results.values() if r)
    total_errors = sum(r.error_tests for r in results.values() if r)
    total_skipped = sum(r.skipped_tests for r in results.values() if r)
    total_duration = sum(r.duration for r in results.values() if r)
    
    print(f"总测试数: {total_tests}")
    print(f"通过: {total_passed}")
    print(f"失败: {total_failed}")
    print(f"错误: {total_errors}")
    print(f"跳过: {total_skipped}")
    print(f"总耗时: {total_duration:.2f}秒")
    
    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(f"成功率: {success_rate:.1f}%")
    
    print(f"\n{'='*80}")
    print("各模块详细结果")
    print(f"{'='*80}")
    
    for module_name, result in results.items():
        if result:
            print(f"\n{module_name}:")
            print(f"  测试数: {result.total_tests}")
            print(f"  通过: {result.passed_tests}")
            print(f"  失败: {result.failed_tests}")
            print(f"  错误: {result.error_tests}")
            print(f"  跳过: {result.skipped_tests}")
            print(f"  耗时: {result.duration:.2f}秒")
            print(f"  成功率: {result.success_rate:.1f}%")
            
            # 打印失败和错误详情
            if result.failures:
                print(f"  失败详情:")
                for test, traceback in result.failures:
                    print(f"    - {test}: {traceback.split('AssertionError:')[-1].strip()}")
            
            if result.errors:
                print(f"  错误详情:")
                for test, traceback in result.errors:
                    print(f"    - {test}: {traceback.split('Exception:')[-1].strip()}")
        else:
            print(f"\n{module_name}: 无法运行")


def main():
    """主函数"""
    print("开始运行Qwen3优化微调系统测试套件")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    # 定义要运行的测试模块
    test_modules = [
        "test_memory_optimizer",
        "test_model_manager", 
        "test_data_pipeline",
        "test_memory_error_handling",
        "test_training_engine",
        "test_integration"
    ]
    
    # 运行所有测试
    results = {}
    total_start_time = time.time()
    
    for module_name in test_modules:
        result = run_test_module(module_name)
        results[module_name] = result
    
    total_end_time = time.time()
    
    # 打印总结
    print_summary(results)
    
    print(f"\n总运行时间: {total_end_time - total_start_time:.2f}秒")
    
    # 检查是否有失败的测试
    total_failed = sum(r.failed_tests for r in results.values() if r)
    total_errors = sum(r.error_tests for r in results.values() if r)
    
    if total_failed > 0 or total_errors > 0:
        print(f"\n⚠️  有 {total_failed} 个测试失败，{total_errors} 个测试错误")
        sys.exit(1)
    else:
        print(f"\n✅ 所有测试通过！")
        sys.exit(0)


if __name__ == "__main__":
    main()
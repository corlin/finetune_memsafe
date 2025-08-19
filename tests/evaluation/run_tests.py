"""
运行评估模块的所有单元测试
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 导入测试模块
from test_data_field_detector import TestDataFieldDetector
from test_batch_data_validator import TestBatchDataValidator
from test_field_mapper import TestFieldMapper
from test_data_preprocessor import TestDataPreprocessor
from test_integration import TestEvaluationIntegration
from test_performance import TestPerformance


def create_test_suite():
    """创建测试套件"""
    suite = unittest.TestSuite()
    
    # 添加DataFieldDetector测试
    suite.addTest(unittest.makeSuite(TestDataFieldDetector))
    
    # 添加BatchDataValidator测试
    suite.addTest(unittest.makeSuite(TestBatchDataValidator))
    
    # 添加FieldMapper测试
    suite.addTest(unittest.makeSuite(TestFieldMapper))
    
    # 添加DataPreprocessor测试
    suite.addTest(unittest.makeSuite(TestDataPreprocessor))
    
    # 添加集成测试
    suite.addTest(unittest.makeSuite(TestEvaluationIntegration))
    
    # 添加性能测试
    suite.addTest(unittest.makeSuite(TestPerformance))
    
    return suite


def run_tests():
    """运行所有测试"""
    print("开始运行评估模块单元测试...")
    print("=" * 60)
    
    # 创建测试套件
    suite = create_test_suite()
    
    # 创建测试运行器
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    # 运行测试
    result = runner.run(suite)
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    print(f"跳过数: {len(result.skipped)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # 计算成功率
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n测试成功率: {success_rate:.1f}%")
    
    return result.wasSuccessful()


def run_specific_test(test_class_name):
    """运行特定的测试类"""
    test_classes = {
        'DataFieldDetector': TestDataFieldDetector,
        'BatchDataValidator': TestBatchDataValidator,
        'FieldMapper': TestFieldMapper,
        'DataPreprocessor': TestDataPreprocessor,
        'Integration': TestEvaluationIntegration,
        'Performance': TestPerformance
    }
    
    if test_class_name not in test_classes:
        print(f"未找到测试类: {test_class_name}")
        print(f"可用的测试类: {', '.join(test_classes.keys())}")
        return False
    
    print(f"运行 {test_class_name} 测试...")
    print("=" * 40)
    
    suite = unittest.makeSuite(test_classes[test_class_name])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 运行特定测试类
        test_class = sys.argv[1]
        success = run_specific_test(test_class)
    else:
        # 运行所有测试
        success = run_tests()
    
    # 设置退出码
    sys.exit(0 if success else 1)
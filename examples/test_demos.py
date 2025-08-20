#!/usr/bin/env python3
"""
Industry Evaluation System 演示程序测试脚本

这个脚本用于快速测试所有演示程序是否能正常导入和基本运行。
"""

import sys
import traceback
from pathlib import Path


def test_imports():
    """测试核心模块导入"""
    print("🔍 测试核心模块导入...")
    
    modules_to_test = [
        'industry_evaluation.config.config_manager',
        'industry_evaluation.adapters.model_adapter',
        'industry_evaluation.core.evaluation_engine',
        'industry_evaluation.core.batch_evaluator',
        'industry_evaluation.evaluators.knowledge_evaluator',
        'industry_evaluation.evaluators.terminology_evaluator',
        'industry_evaluation.reporting.report_generator',
        'industry_evaluation.api.rest_api'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {str(e)}")
            failed_imports.append(module)
        except Exception as e:
            print(f"  ⚠️ {module}: {str(e)}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} 个模块导入失败")
        return False
    else:
        print(f"\n✅ 所有 {len(modules_to_test)} 个模块导入成功")
        return True


def test_demo_imports():
    """测试演示程序导入"""
    print("\n🔍 测试演示程序导入...")
    
    demo_files = [
        'simple_demo',
        'complete_demo', 
        'api_demo',
        'config_demo'
    ]
    
    failed_demos = []
    
    # 添加examples目录到Python路径
    examples_dir = Path(__file__).parent
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    
    for demo_file in demo_files:
        try:
            module = __import__(demo_file)
            print(f"  ✅ {demo_file}.py")
        except ImportError as e:
            print(f"  ❌ {demo_file}.py: {str(e)}")
            failed_demos.append(demo_file)
        except Exception as e:
            print(f"  ⚠️ {demo_file}.py: {str(e)}")
            failed_demos.append(demo_file)
    
    if failed_demos:
        print(f"\n❌ {len(failed_demos)} 个演示程序导入失败")
        return False
    else:
        print(f"\n✅ 所有 {len(demo_files)} 个演示程序导入成功")
        return True


def test_basic_functionality():
    """测试基本功能"""
    print("\n🔍 测试基本功能...")
    
    try:
        # 测试配置管理
        print("  🔧 测试配置管理...")
        from industry_evaluation.config.config_manager import ConfigTemplate, ConfigManager
        
        # 生成配置模板
        config = ConfigTemplate.generate_finance_config()
        assert config.version == "1.0.0"
        assert len(config.models) > 0
        assert len(config.evaluators) > 0
        print("    ✅ 配置模板生成成功")
        
        # 测试模型适配器
        print("  🤖 测试模型适配器...")
        from industry_evaluation.adapters.model_adapter import ModelManager, ModelAdapterFactory
        
        # 创建模型管理器
        model_manager = ModelManager()
        models = model_manager.list_models()
        assert isinstance(models, list)
        print("    ✅ 模型管理器创建成功")
        
        # 测试评估器
        print("  📊 测试评估器...")
        from industry_evaluation.evaluators.knowledge_evaluator import KnowledgeEvaluator
        from industry_evaluation.evaluators.terminology_evaluator import TerminologyEvaluator
        
        knowledge_evaluator = KnowledgeEvaluator()
        terminology_evaluator = TerminologyEvaluator()
        assert knowledge_evaluator is not None
        assert terminology_evaluator is not None
        print("    ✅ 评估器创建成功")
        
        # 测试结果聚合器
        print("  📈 测试结果聚合器...")
        from industry_evaluation.core.result_aggregator import ResultAggregator
        
        result_aggregator = ResultAggregator()
        assert result_aggregator is not None
        print("    ✅ 结果聚合器创建成功")
        
        # 测试报告生成器
        print("  📄 测试报告生成器...")
        from industry_evaluation.reporting.report_generator import ReportGenerator
        
        report_generator = ReportGenerator()
        assert report_generator is not None
        print("    ✅ 报告生成器创建成功")
        
        print("\n✅ 基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 基本功能测试失败: {str(e)}")
        traceback.print_exc()
        return False


def test_demo_classes():
    """测试演示程序中的类"""
    print("\n🔍 测试演示程序类...")
    
    try:
        # 添加examples目录到Python路径
        examples_dir = Path(__file__).parent
        if str(examples_dir) not in sys.path:
            sys.path.insert(0, str(examples_dir))
        
        # 测试简化演示中的适配器
        print("  🚀 测试简化演示适配器...")
        from simple_demo import SimpleModelAdapter
        
        adapter = SimpleModelAdapter("test_model", {"quality": "good"})
        response = adapter.predict("测试输入")
        assert isinstance(response, str)
        assert len(response) > 0
        print("    ✅ 简化演示适配器测试通过")
        
        # 测试完整演示中的适配器
        print("  🎬 测试完整演示适配器...")
        from complete_demo import MockModelAdapter
        
        mock_adapter = MockModelAdapter("test_model", {"quality": "excellent", "domain": "finance"})
        mock_response = mock_adapter.predict("VaR模型", {"industry": "finance"})
        assert isinstance(mock_response, str)
        assert "VaR" in mock_response or "Value at Risk" in mock_response
        print("    ✅ 完整演示适配器测试通过")
        
        # 测试配置演示
        print("  ⚙️ 测试配置演示类...")
        from config_demo import ConfigDemo
        
        config_demo = ConfigDemo()
        assert config_demo.temp_dir.exists()
        print("    ✅ 配置演示类测试通过")
        
        print("\n✅ 演示程序类测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 演示程序类测试失败: {str(e)}")
        traceback.print_exc()
        return False


def test_file_structure():
    """测试文件结构"""
    print("\n🔍 测试文件结构...")
    
    examples_dir = Path(__file__).parent
    required_files = [
        'simple_demo.py',
        'complete_demo.py',
        'api_demo.py',
        'config_demo.py',
        'run_demo.py',
        'README.md'
    ]
    
    missing_files = []
    
    for file_name in required_files:
        file_path = examples_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} (缺失)")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n❌ {len(missing_files)} 个文件缺失")
        return False
    else:
        print(f"\n✅ 所有 {len(required_files)} 个文件存在")
        return True


def run_quick_demo_test():
    """运行快速演示测试"""
    print("\n🔍 运行快速演示测试...")
    
    try:
        # 添加examples目录到Python路径
        examples_dir = Path(__file__).parent
        if str(examples_dir) not in sys.path:
            sys.path.insert(0, str(examples_dir))
        
        # 测试简化演示的核心逻辑
        print("  🚀 测试简化演示核心逻辑...")
        from simple_demo import SimpleModelAdapter
        
        # 创建模拟适配器
        adapter = SimpleModelAdapter("test_model", {"quality": "excellent"})
        
        # 测试预测功能
        response = adapter.predict("测试问题")
        assert isinstance(response, str)
        assert len(response) > 10
        
        # 测试可用性检查
        assert adapter.is_available() == True
        
        print("    ✅ 简化演示核心逻辑测试通过")
        
        print("\n✅ 快速演示测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 快速演示测试失败: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🧪 Industry Evaluation System - 演示程序测试")
    print("=" * 60)
    
    test_results = []
    
    # 运行各项测试
    tests = [
        ("文件结构检查", test_file_structure),
        ("核心模块导入", test_imports),
        ("演示程序导入", test_demo_imports),
        ("基本功能测试", test_basic_functionality),
        ("演示程序类测试", test_demo_classes),
        ("快速演示测试", run_quick_demo_test)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 执行失败: {str(e)}")
            test_results.append((test_name, False))
    
    # 显示测试总结
    print("\n" + "=" * 60)
    print("📋 测试结果总结")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20} {status}")
        if result:
            passed_tests += 1
    
    print("-" * 60)
    print(f"总计: {passed_tests}/{total_tests} 个测试通过")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！演示程序可以正常运行。")
        print("\n💡 建议运行顺序:")
        print("  1. python examples/simple_demo.py")
        print("  2. python examples/config_demo.py")
        print("  3. python examples/api_demo.py")
        print("  4. python examples/complete_demo.py")
        print("\n或者使用启动器:")
        print("  python examples/run_demo.py")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} 个测试失败，请检查系统配置。")
        print("\n🔧 可能的解决方案:")
        print("  1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("  2. 确保在项目根目录运行测试")
        print("  3. 检查Python版本是否兼容 (推荐 Python 3.8+)")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生未预期的错误: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
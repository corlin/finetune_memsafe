#!/usr/bin/env python3
"""
快速导入测试脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有关键导入"""
    print("🧪 测试关键模块导入...")
    
    test_modules = [
        ("数据模型", "industry_evaluation.models.data_models"),
        ("核心接口", "industry_evaluation.core.interfaces"),
        ("配置管理", "industry_evaluation.config.config_manager"),
        ("模型适配器", "industry_evaluation.adapters.model_adapter"),
        ("结果聚合器", "industry_evaluation.core.result_aggregator"),
        ("进度跟踪器", "industry_evaluation.core.progress_tracker"),
        ("报告生成器", "industry_evaluation.reporting.report_generator"),
        ("评估引擎", "industry_evaluation.core.evaluation_engine"),
        ("批量评估器", "industry_evaluation.core.batch_evaluator"),
    ]
    
    failed_imports = []
    
    for name, module_name in test_modules:
        try:
            __import__(module_name)
            print(f"   ✅ {name} ({module_name})")
        except ImportError as e:
            print(f"   ❌ {name} ({module_name}): {str(e)}")
            failed_imports.append((name, module_name, str(e)))
        except Exception as e:
            print(f"   ⚠️ {name} ({module_name}): {str(e)}")
            failed_imports.append((name, module_name, str(e)))
    
    print(f"\n📊 导入测试结果:")
    print(f"   成功: {len(test_modules) - len(failed_imports)}/{len(test_modules)}")
    
    if failed_imports:
        print(f"   失败: {len(failed_imports)}")
        print("\n❌ 失败详情:")
        for name, module_name, error in failed_imports:
            print(f"   - {name}: {error}")
        return False
    else:
        print("   ✅ 所有模块导入成功！")
        return True

def test_specific_classes():
    """测试特定类的导入"""
    print("\n🔍 测试特定类导入...")
    
    try:
        from industry_evaluation.models.data_models import (
            EvaluationConfig, EvaluationResult, SampleResult, 
            EvaluationStatus, ProgressInfo
        )
        print("   ✅ 数据模型类导入成功")
        
        from industry_evaluation.core.interfaces import (
            BaseEvaluator, EvaluationEngine, ModelAdapter
        )
        print("   ✅ 接口类导入成功")
        
        from industry_evaluation.config.config_manager import (
            ConfigManager, ConfigTemplate, ModelConfig
        )
        print("   ✅ 配置管理类导入成功")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ 类导入失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🔧 Industry Evaluation System - 导入测试")
    print("=" * 50)
    
    # 测试模块导入
    modules_ok = test_imports()
    
    # 测试类导入
    classes_ok = test_specific_classes()
    
    print("\n" + "=" * 50)
    
    if modules_ok and classes_ok:
        print("🎉 所有导入测试通过！")
        print("\n💡 现在可以运行演示程序:")
        print("   python examples/simple_demo.py")
        print("   python quick_start.py")
        return True
    else:
        print("❌ 导入测试失败")
        print("\n🔧 建议解决方案:")
        print("1. 确保在项目根目录运行")
        print("2. 安装依赖: python install_demo_deps.py")
        print("3. 运行故障排除: python troubleshoot.py")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
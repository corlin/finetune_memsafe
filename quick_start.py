#!/usr/bin/env python3
"""
Industry Evaluation System 快速启动脚本

这个脚本会自动处理路径问题并运行简化演示。
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """设置环境"""
    # 添加项目根目录到 Python 路径
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = str(project_root)

def check_dependencies():
    """检查依赖"""
    required_modules = [
        'yaml',
        'requests'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ 缺少以下依赖模块:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\n🔧 请运行以下命令安装依赖:")
        print("   python install_demo_deps.py")
        print("   或者: pip install pyyaml requests")
        return False
    
    return True

def run_simple_demo():
    """运行简化演示"""
    print("🚀 Industry Evaluation System - 快速启动")
    print("=" * 50)
    
    # 设置环境
    setup_environment()
    
    # 检查依赖
    if not check_dependencies():
        return False
    
    print("✅ 环境检查通过")
    
    # 测试关键导入
    print("🔍 测试关键模块导入...")
    try:
        from industry_evaluation.models.data_models import EvaluationConfig, SampleResult
        from industry_evaluation.core.interfaces import BaseEvaluator
        from industry_evaluation.config.config_manager import ConfigManager
        print("✅ 关键模块导入成功")
    except ImportError as e:
        print(f"❌ 关键模块导入失败: {str(e)}")
        print("\n💡 解决方案:")
        print("1. 运行导入测试: python test_imports.py")
        print("2. 运行故障排除: python troubleshoot.py")
        return False
    
    print("🔄 启动简化演示...")
    print("-" * 50)
    
    try:
        # 导入并运行简化演示
        from examples.simple_demo import simple_evaluation_demo
        simple_evaluation_demo()
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {str(e)}")
        print("\n💡 解决方案:")
        print("1. 确保在项目根目录运行此脚本")
        print("2. 运行: python install_demo_deps.py")
        print("3. 运行: python test_imports.py")
        print("4. 或者直接运行: python examples/simple_demo.py")
        return False
        
    except Exception as e:
        print(f"❌ 运行错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = run_simple_demo()
        if success:
            print("\n🎉 演示运行完成！")
            print("\n💡 接下来可以尝试:")
            print("  python examples/config_demo.py")
            print("  python examples/api_demo.py")
            print("  python examples/complete_demo.py")
        else:
            print("\n❌ 演示运行失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断演示")
        sys.exit(0)
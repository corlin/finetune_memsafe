#!/usr/bin/env python3
"""
Industry Evaluation System 故障排除脚本

这个脚本会诊断常见问题并提供解决方案。
"""

import sys
import os
import subprocess
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    print(f"   当前版本: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python版本过低，建议使用Python 3.8+")
        return False
    else:
        print("   ✅ Python版本符合要求")
        return True


def check_current_directory():
    """检查当前目录"""
    print("\n📁 检查当前目录...")
    current_dir = Path.cwd()
    print(f"   当前目录: {current_dir}")
    
    # 检查是否在项目根目录
    required_files = ["industry_evaluation", "examples"]
    missing_items = []
    
    for item in required_files:
        item_path = current_dir / item
        if item_path.exists():
            print(f"   ✅ 找到 {item}")
        else:
            print(f"   ❌ 缺少 {item}")
            missing_items.append(item)
    
    if missing_items:
        print("   ⚠️ 请确保在项目根目录运行脚本")
        return False
    else:
        print("   ✅ 目录结构正确")
        return True


def check_python_path():
    """检查Python路径"""
    print("\n🛤️ 检查Python路径...")
    current_dir = str(Path.cwd())
    
    if current_dir in sys.path:
        print(f"   ✅ 当前目录已在Python路径中")
        return True
    else:
        print(f"   ⚠️ 当前目录不在Python路径中")
        print(f"   💡 可以设置环境变量: export PYTHONPATH={current_dir}:$PYTHONPATH")
        return False


def check_dependencies():
    """检查依赖"""
    print("\n📦 检查依赖包...")
    
    required_packages = [
        ("yaml", "pyyaml"),
        ("requests", "requests"),
        ("flask", "flask"),
        ("psutil", "psutil"),
        ("numpy", "numpy")
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} (缺失)")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n   ⚠️ 缺少 {len(missing_packages)} 个依赖包")
        print("   💡 安装命令:")
        print(f"      pip install {' '.join(missing_packages)}")
        print("   或者运行: python install_demo_deps.py")
        return False
    else:
        print("   ✅ 所有依赖包都已安装")
        return True


def check_industry_evaluation_module():
    """检查 industry_evaluation 模块"""
    print("\n🔍 检查 industry_evaluation 模块...")
    
    # 添加当前目录到路径
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    try:
        import industry_evaluation
        print("   ✅ industry_evaluation 模块导入成功")
        print(f"   📍 模块位置: {industry_evaluation.__file__}")
        return True
    except ImportError as e:
        print(f"   ❌ industry_evaluation 模块导入失败: {str(e)}")
        
        # 检查模块文件是否存在
        module_path = current_dir / "industry_evaluation"
        if module_path.exists():
            print(f"   📁 模块目录存在: {module_path}")
            init_file = module_path / "__init__.py"
            if init_file.exists():
                print("   ✅ __init__.py 文件存在")
            else:
                print("   ❌ __init__.py 文件缺失")
        else:
            print(f"   ❌ 模块目录不存在: {module_path}")
        
        return False


def test_simple_import():
    """测试简单导入"""
    print("\n🧪 测试关键模块导入...")
    
    # 添加当前目录到路径
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    test_imports = [
        "industry_evaluation.models.data_models",
        "industry_evaluation.core.interfaces",
        "industry_evaluation.config.config_manager",
        "industry_evaluation.adapters.model_adapter",
        "industry_evaluation.core.evaluation_engine",
        "industry_evaluation.core.result_aggregator",
        "industry_evaluation.reporting.report_generator",
    ]
    
    failed_imports = []
    
    for module_name in test_imports:
        try:
            __import__(module_name)
            print(f"   ✅ {module_name}")
        except ImportError as e:
            print(f"   ❌ {module_name}: {str(e)}")
            failed_imports.append((module_name, str(e)))
    
    if failed_imports:
        print(f"\n   ⚠️ {len(failed_imports)} 个模块导入失败")
        for module_name, error in failed_imports:
            print(f"      {module_name}: {error}")
        return False
    else:
        print("   ✅ 所有关键模块导入成功")
        return True


def provide_solutions():
    """提供解决方案"""
    print("\n" + "=" * 60)
    print("💡 解决方案建议")
    print("=" * 60)
    
    print("\n🔧 如果遇到模块导入问题:")
    print("1. 确保在项目根目录运行:")
    print("   cd /path/to/your/project")
    print("   python examples/simple_demo.py")
    
    print("\n2. 使用快速启动脚本:")
    print("   python quick_start.py")
    
    print("\n3. 设置环境变量:")
    print("   export PYTHONPATH=$PWD:$PYTHONPATH")
    print("   python examples/simple_demo.py")
    
    print("\n4. 安装依赖:")
    print("   python install_demo_deps.py")
    print("   # 或者")
    print("   pip install -r demo_requirements.txt")
    
    print("\n🚀 推荐的运行顺序:")
    print("1. python troubleshoot.py  # 诊断问题")
    print("2. python install_demo_deps.py  # 安装依赖")
    print("3. python quick_start.py  # 快速启动")
    print("4. python examples/run_demo.py  # 交互式菜单")


def main():
    """主诊断函数"""
    print("🔧 Industry Evaluation System - 故障排除")
    print("=" * 60)
    
    checks = [
        ("Python版本", check_python_version),
        ("当前目录", check_current_directory),
        ("Python路径", check_python_path),
        ("依赖包", check_dependencies),
        ("核心模块", check_industry_evaluation_module),
        ("模块导入", test_simple_import),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"   ❌ 检查失败: {str(e)}")
            results.append((check_name, False))
    
    # 显示总结
    print("\n" + "=" * 60)
    print("📋 诊断结果总结")
    print("=" * 60)
    
    passed_checks = 0
    total_checks = len(results)
    
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name:<15} {status}")
        if result:
            passed_checks += 1
    
    print("-" * 60)
    print(f"总计: {passed_checks}/{total_checks} 项检查通过")
    
    if passed_checks == total_checks:
        print("\n🎉 所有检查通过！系统应该可以正常运行。")
        print("\n🚀 现在可以运行演示程序:")
        print("   python examples/simple_demo.py")
        print("   python examples/run_demo.py")
    else:
        print(f"\n⚠️ {total_checks - passed_checks} 项检查失败")
        provide_solutions()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ 诊断被用户中断")
    except Exception as e:
        print(f"\n❌ 诊断过程中发生错误: {str(e)}")
        provide_solutions()
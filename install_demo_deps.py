#!/usr/bin/env python3
"""
Industry Evaluation System 演示依赖安装脚本

这个脚本会安装运行演示程序所需的所有依赖。
"""

import subprocess
import sys
from pathlib import Path


def install_package(package):
    """安装单个包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """主安装函数"""
    print("🔧 Industry Evaluation System - 演示依赖安装")
    print("=" * 50)
    
    # 必需的依赖包
    required_packages = [
        "pyyaml>=6.0.0",
        "requests>=2.31.0",
        "flask>=2.3.0",
        "flask-restx>=1.1.0",
        "flask-cors>=4.0.0",
        "watchdog>=3.0.0",
        "psutil>=5.9.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ]
    
    print(f"📦 准备安装 {len(required_packages)} 个依赖包...")
    print()
    
    failed_packages = []
    
    for package in required_packages:
        package_name = package.split(">=")[0]
        print(f"🔄 安装 {package_name}...")
        
        if install_package(package):
            print(f"✅ {package_name} 安装成功")
        else:
            print(f"❌ {package_name} 安装失败")
            failed_packages.append(package)
        print()
    
    # 尝试以开发模式安装项目
    print("🔧 尝试以开发模式安装项目...")
    project_root = Path(__file__).parent
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(project_root)])
        print("✅ 项目开发模式安装成功")
    except subprocess.CalledProcessError:
        print("⚠️ 项目开发模式安装失败，但不影响演示运行")
    
    print("\n" + "=" * 50)
    
    if failed_packages:
        print(f"❌ {len(failed_packages)} 个包安装失败:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\n💡 建议手动安装失败的包:")
        print(f"  pip install {' '.join(failed_packages)}")
    else:
        print("🎉 所有依赖安装成功！")
    
    print("\n🚀 现在可以运行演示程序了:")
    print("  python examples/simple_demo.py")
    print("  python examples/run_demo.py")


if __name__ == "__main__":
    main()
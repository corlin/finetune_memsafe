#!/usr/bin/env python3
"""
使用 uv 运行 Industry Evaluation System 演示

这个脚本专门为 uv 环境设计，解决模块导入和依赖问题。
"""

import sys
import subprocess
from pathlib import Path


def setup_uv_environment():
    """设置 uv 环境"""
    print("🔧 设置 uv 环境...")
    
    # 确保在项目根目录
    project_root = Path.cwd()
    print(f"📁 项目根目录: {project_root}")
    
    # 检查必要文件
    required_files = ["pyproject.toml", "industry_evaluation"]
    missing_files = []
    
    for file_name in required_files:
        if not (project_root / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return False
    
    print("✅ 项目结构检查通过")
    return True


def install_dependencies_with_uv():
    """使用 uv 安装依赖"""
    print("📦 使用 uv 安装依赖...")
    
    try:
        # 同步依赖
        result = subprocess.run(["uv", "sync"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ uv sync 成功")
        else:
            print(f"⚠️ uv sync 警告: {result.stderr}")
        
        # 安装演示程序特定依赖
        demo_deps = [
            "pyyaml>=6.0.0",
            "requests>=2.31.0", 
            "flask>=2.3.0",
            "flask-restx>=1.1.0",
            "flask-cors>=4.0.0",
            "watchdog>=3.0.0",
            "psutil>=5.9.0"
        ]
        
        for dep in demo_deps:
            print(f"📦 安装 {dep}...")
            result = subprocess.run(["uv", "add", dep], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {dep} 安装成功")
            else:
                print(f"⚠️ {dep} 安装可能有问题: {result.stderr}")
        
        return True
        
    except FileNotFoundError:
        print("❌ uv 命令未找到，请确保已安装 uv")
        print("💡 安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False
    except Exception as e:
        print(f"❌ 安装依赖失败: {str(e)}")
        return False


def run_with_uv(script_path: str):
    """使用 uv 运行脚本"""
    print(f"🚀 使用 uv 运行 {script_path}...")
    
    try:
        # 使用 uv run 运行脚本
        result = subprocess.run(
            ["uv", "run", "python", script_path],
            cwd=Path.cwd(),
            text=True
        )
        
        if result.returncode == 0:
            print("✅ 脚本运行成功")
            return True
        else:
            print(f"❌ 脚本运行失败，退出码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ 运行脚本时发生错误: {str(e)}")
        return False


def main():
    """主函数"""
    print("🚀 Industry Evaluation System - uv 演示启动器")
    print("=" * 60)
    
    # 设置环境
    if not setup_uv_environment():
        print("❌ 环境设置失败")
        return False
    
    # 安装依赖
    print("\n📦 准备依赖...")
    if not install_dependencies_with_uv():
        print("❌ 依赖安装失败")
        return False
    
    # 显示可用的演示程序
    print("\n📋 可用的演示程序:")
    demos = [
        ("simple_demo.py", "简化演示 - 快速了解基本功能"),
        ("config_demo.py", "配置演示 - 配置管理功能"),
        ("api_demo.py", "API演示 - REST API接口"),
        ("complete_demo.py", "完整演示 - 所有功能展示")
    ]
    
    for i, (script, description) in enumerate(demos, 1):
        print(f"  {i}. {script:<20} - {description}")
    
    print("\n请选择要运行的演示程序:")
    
    while True:
        try:
            choice = input("输入数字 (1-4) 或 'q' 退出: ").strip()
            
            if choice.lower() == 'q':
                print("👋 再见！")
                return True
            
            if choice in ['1', '2', '3', '4']:
                script_name = demos[int(choice) - 1][0]
                script_path = f"examples/{script_name}"
                
                print(f"\n🎯 运行 {script_name}...")
                print("-" * 50)
                
                success = run_with_uv(script_path)
                
                if success:
                    print(f"✅ {script_name} 运行完成")
                else:
                    print(f"❌ {script_name} 运行失败")
                
                print("-" * 50)
                
                # 询问是否继续
                continue_choice = input("\n是否运行其他演示？(y/N): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    break
            else:
                print("❌ 无效选择，请输入 1-4 或 'q'")
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            return True
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")
            return False
    
    print("\n🎉 演示完成！")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 再见！")
        sys.exit(0)
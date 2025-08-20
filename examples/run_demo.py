#!/usr/bin/env python3
"""
Industry Evaluation System 演示程序启动器

这个脚本提供了一个交互式菜单，让用户选择运行不同的演示程序。
"""

import sys
import os
import subprocess
from pathlib import Path


def print_banner():
    """打印欢迎横幅"""
    print("=" * 70)
    print("🚀 Industry Evaluation System - 演示程序启动器")
    print("=" * 70)
    print("欢迎使用行业评估系统！请选择您想要运行的演示程序：")
    print()


def print_menu():
    """打印菜单选项"""
    print("📋 可用的演示程序：")
    print()
    print("1. 🚀 简化演示 (simple_demo.py)")
    print("   - 适合初次使用者")
    print("   - 快速了解基本功能")
    print("   - 运行时间：约2-3分钟")
    print()
    print("2. 🎬 完整功能演示 (complete_demo.py)")
    print("   - 展示所有核心功能")
    print("   - 包含异步评估和批量处理")
    print("   - 运行时间：约5-10分钟")
    print()
    print("3. 🌐 API接口演示 (api_demo.py)")
    print("   - REST API功能测试")
    print("   - 启动内置服务器")
    print("   - 运行时间：约3-5分钟")
    print()
    print("4. ⚙️ 配置管理演示 (config_demo.py)")
    print("   - 配置系统专项演示")
    print("   - 包含性能测试")
    print("   - 运行时间：约2-4分钟")
    print()
    print("5. 📚 查看演示说明 (README.md)")
    print("   - 详细的使用指南")
    print("   - 功能说明和示例")
    print()
    print("0. 🚪 退出")
    print()


def check_dependencies():
    """检查依赖是否安装"""
    print("🔍 检查系统依赖...")
    
    required_modules = [
        'industry_evaluation',
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
        print("❌ 缺少以下依赖模块：")
        for module in missing_modules:
            print(f"   - {module}")
        print()
        print("请运行以下命令安装依赖：")
        print("   pip install -r requirements.txt")
        print("   或者：pip install -e .")
        return False
    
    print("✅ 依赖检查通过")
    return True


def run_demo(demo_file):
    """运行指定的演示程序"""
    demo_path = Path(__file__).parent / demo_file
    
    if not demo_path.exists():
        print(f"❌ 演示文件不存在: {demo_file}")
        return False
    
    print(f"🚀 启动演示程序: {demo_file}")
    print("-" * 50)
    
    try:
        # 运行演示程序
        result = subprocess.run([sys.executable, str(demo_path)], 
                              cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print("-" * 50)
            print("✅ 演示程序运行完成")
        else:
            print("-" * 50)
            print(f"❌ 演示程序运行失败，退出码: {result.returncode}")
        
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断演示程序")
        return False
    except Exception as e:
        print(f"❌ 运行演示程序时发生错误: {str(e)}")
        return False


def show_readme():
    """显示README内容"""
    readme_path = Path(__file__).parent / "README.md"
    
    if not readme_path.exists():
        print("❌ README.md 文件不存在")
        return
    
    print("📚 演示程序说明文档")
    print("=" * 50)
    
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 显示前50行
        lines = content.split('\n')
        for i, line in enumerate(lines[:50]):
            print(line)
        
        if len(lines) > 50:
            print("\n... (更多内容请查看 examples/README.md)")
        
    except Exception as e:
        print(f"❌ 读取README文件失败: {str(e)}")


def get_user_choice():
    """获取用户选择"""
    while True:
        try:
            choice = input("请输入您的选择 (0-5): ").strip()
            
            if choice in ['0', '1', '2', '3', '4', '5']:
                return choice
            else:
                print("❌ 无效选择，请输入 0-5 之间的数字")
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            sys.exit(0)
        except EOFError:
            print("\n👋 再见！")
            sys.exit(0)


def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装必要的依赖")
        sys.exit(1)
    
    print()
    
    while True:
        print_menu()
        choice = get_user_choice()
        
        if choice == '0':
            print("👋 感谢使用 Industry Evaluation System！")
            break
            
        elif choice == '1':
            print("\n🚀 运行简化演示...")
            run_demo("simple_demo.py")
            
        elif choice == '2':
            print("\n🎬 运行完整功能演示...")
            print("⚠️ 注意：此演示需要较长时间，请耐心等待...")
            confirm = input("确认运行吗？(y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                run_demo("complete_demo.py")
            else:
                print("❌ 已取消运行")
                
        elif choice == '3':
            print("\n🌐 运行API接口演示...")
            print("⚠️ 注意：此演示会启动HTTP服务器")
            run_demo("api_demo.py")
            
        elif choice == '4':
            print("\n⚙️ 运行配置管理演示...")
            run_demo("config_demo.py")
            
        elif choice == '5':
            print("\n📚 显示演示说明...")
            show_readme()
        
        print("\n" + "=" * 70)
        
        # 询问是否继续
        continue_choice = input("是否继续使用演示程序？(Y/n): ").strip().lower()
        if continue_choice in ['n', 'no']:
            print("👋 感谢使用 Industry Evaluation System！")
            break
        
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 再见！")
        sys.exit(0)
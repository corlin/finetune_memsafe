#!/usr/bin/env python3
"""
Qwen3微调系统快速启动脚本

这个脚本提供了一个简化的启动方式，自动处理环境设置和依赖安装。
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_uv_installed():
    """检查uv是否已安装"""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ uv已安装: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def install_uv():
    """安装uv"""
    print("正在安装uv...")
    try:
        if os.name == 'nt':  # Windows
            subprocess.run([
                "powershell", "-c", 
                "irm https://astral.sh/uv/install.ps1 | iex"
            ], check=True)
        else:  # Linux/macOS
            subprocess.run([
                "curl", "-LsSf", "https://astral.sh/uv/install.sh"
            ], stdout=subprocess.PIPE, check=True)
            subprocess.run(["sh"], input=subprocess.PIPE, check=True)
        
        print("✓ uv安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ uv安装失败: {e}")
        return False


def sync_dependencies():
    """同步项目依赖"""
    print("正在同步项目依赖...")
    try:
        result = subprocess.run(["uv", "sync"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 依赖同步完成")
            return True
        else:
            print(f"✗ 依赖同步失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ 依赖同步异常: {e}")
        return False


def create_sample_config():
    """创建示例配置文件"""
    config = {
        "model_name": "Qwen/Qwen3-4B-Thinking-2507",
        "output_dir": "./qwen3-finetuned",
        "max_memory_gb": 13.0,
        "batch_size": 4,
        "gradient_accumulation_steps": 16,
        "learning_rate": 5e-5,
        "num_epochs": 10,
        "max_sequence_length": 256,
        "data_dir": "data/raw",
        "auto_install_deps": True,
        "verify_environment": True
    }
    
    config_path = Path("quick_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 创建示例配置文件: {config_path}")
    return str(config_path)


def run_finetuning(config_path=None):
    """运行微调"""
    print("开始运行Qwen3微调...")
    
    cmd = ["uv", "run", "main.py"]
    
    if config_path:
        cmd.extend(["--config", config_path])
    else:
        cmd.extend([
            "--auto-install-deps",
            "--num-epochs", "20",  # 快速测试用较少轮数
            "--batch-size", "2",   # 保守的批次大小
            "--max-memory-gb", "10"  # 保守的内存限制
        ])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 实时显示输出
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("\n✓ 微调完成！")
            return True
        else:
            print(f"\n✗ 微调失败，退出码: {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n用户中断了微调过程")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n✗ 运行异常: {e}")
        return False


def main():
    """主函数"""
    print("=== Qwen3微调系统快速启动 ===\n")
    
    # 1. 检查uv
    if not check_uv_installed():
        print("uv未安装，正在安装...")
        if not install_uv():
            print("请手动安装uv后重试")
            sys.exit(1)
    
    # 2. 同步依赖
    if not sync_dependencies():
        print("依赖同步失败，请检查网络连接")
        sys.exit(1)
    
    # 3. 询问用户配置选择
    print("\n请选择运行模式:")
    print("1. 快速测试模式 (推荐新手，使用保守配置)")
    print("2. 使用配置文件模式")
    print("3. 自定义参数模式")
    
    choice = input("\n请输入选择 (1-3，默认1): ").strip() or "1"
    
    if choice == "1":
        print("\n使用快速测试模式...")
        success = run_finetuning()
    elif choice == "2":
        config_path = create_sample_config()
        print(f"\n使用配置文件模式: {config_path}")
        print("您可以编辑配置文件后重新运行")
        
        edit_config = input("是否现在编辑配置文件? (y/N): ").strip().lower()
        if edit_config == 'y':
            print(f"请编辑 {config_path} 文件，然后按回车继续...")
            input()
        
        success = run_finetuning(config_path)
    elif choice == "3":
        print("\n自定义参数模式:")
        model_name = input("模型名称 (默认: Qwen/Qwen3-4B-Thinking-2507): ").strip() or "Qwen/Qwen3-4B-Thinking-2507"
        max_memory = input("最大GPU内存GB (默认: 10): ").strip() or "10"
        batch_size = input("批次大小 (默认: 2): ").strip() or "2"
        num_epochs = input("训练轮数 (默认: 20): ").strip() or "20"
        
        cmd = [
            "uv", "run", "main.py",
            "--model-name", model_name,
            "--max-memory-gb", max_memory,
            "--batch-size", batch_size,
            "--num-epochs", num_epochs,
            "--auto-install-deps"
        ]
        
        print(f"\n执行命令: {' '.join(cmd)}")
        success = subprocess.run(cmd).returncode == 0
    else:
        print("无效选择")
        sys.exit(1)
    
    # 4. 结果提示
    if success:
        print("\n🎉 微调完成！")
        print("\n生成的文件:")
        print("- qwen3-finetuned/ - 微调后的模型")
        print("- logs/ - 训练日志")
        print("- final_application_report.json - 训练报告")
        
        print("\n您可以使用以下命令查看TensorBoard:")
        print("uv run tensorboard --logdir ./qwen3-finetuned/logs/tensorboard")
        
        print("\n或者测试推理:")
        print('uv run python -c "from src.inference_tester import InferenceTester; tester = InferenceTester(); tester.load_finetuned_model(\'./qwen3-finetuned\', \'Qwen/Qwen3-4B-Thinking-2507\'); print(tester.test_inference(\'请解释什么是机器学习？\'))"')
    else:
        print("\n❌ 微调失败，请查看错误信息")
        print("\n故障排除建议:")
        print("1. 检查GPU内存是否足够")
        print("2. 确认网络连接正常")
        print("3. 查看 logs/application.log 获取详细错误信息")
        print("4. 尝试使用更保守的配置参数")


if __name__ == "__main__":
    main()
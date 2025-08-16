#!/usr/bin/env python3
"""
Qwen3-4B-Thinking-2507 专用启动脚本

针对Qwen3-4B-Thinking-2507模型优化的快速启动脚本。
"""

import subprocess
import sys
import json
from pathlib import Path


def check_requirements():
    """检查基本要求"""
    print("=== 检查系统要求 ===")
    
    # 检查uv
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ uv已安装: {result.stdout.strip()}")
        else:
            print("✗ uv未正确安装")
            return False
    except FileNotFoundError:
        print("✗ uv未安装")
        print("请安装uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False
    
    # 检查GPU
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU可用")
        else:
            print("⚠️  未检测到NVIDIA GPU，将使用CPU模式")
    except FileNotFoundError:
        print("⚠️  nvidia-smi未找到，可能没有NVIDIA GPU")
    
    return True


def sync_dependencies():
    """同步依赖"""
    print("\n=== 同步依赖 ===")
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


def run_qwen3_thinking_finetuning(mode="default"):
    """运行Qwen3-4B-Thinking-2507微调"""
    print(f"\n=== 开始Qwen3-4B-Thinking-2507微调 ({mode}模式) ===")
    
    if mode == "quick_test":
        # 快速测试模式
        cmd = [
            "uv", "run", "main.py",
            "--model-name", "Qwen/Qwen3-4B-Thinking-2507",
            "--output-dir", "./qwen3-4b-thinking-test",
            "--max-memory-gb", "10",
            "--batch-size", "2",
            "--gradient-accumulation-steps", "32",
            "--num-epochs", "5",
            "--max-sequence-length", "128",
            "--auto-install-deps"
        ]
    elif mode == "low_memory":
        # 低显存模式 (6-8GB GPU)
        cmd = [
            "uv", "run", "main.py",
            "--model-name", "Qwen/Qwen3-4B-Thinking-2507",
            "--output-dir", "./qwen3-4b-thinking-low-mem",
            "--max-memory-gb", "6",
            "--batch-size", "1",
            "--gradient-accumulation-steps", "64",
            "--num-epochs", "30",
            "--max-sequence-length", "128",
            "--auto-install-deps"
        ]
    elif mode == "config_file":
        # 使用配置文件模式
        cmd = [
            "uv", "run", "main.py",
            "--config", "qwen3_4b_thinking_config.json"
        ]
    else:
        # 默认模式
        cmd = [
            "uv", "run", "main.py",
            "--model-name", "Qwen/Qwen3-4B-Thinking-2507",
            "--auto-install-deps"
        ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 实时显示输出
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True, 
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("\n🎉 Qwen3-4B-Thinking-2507微调完成！")
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


def show_results():
    """显示结果"""
    print("\n=== 微调结果 ===")
    
    # 检查输出目录
    possible_dirs = [
        "./qwen3-4b-thinking-finetuned",
        "./qwen3-4b-thinking-test", 
        "./qwen3-4b-thinking-low-mem",
        "./qwen3-finetuned"
    ]
    
    for output_dir in possible_dirs:
        if Path(output_dir).exists():
            print(f"✓ 找到微调结果: {output_dir}")
            
            # 列出主要文件
            output_path = Path(output_dir)
            important_files = [
                "adapter_config.json",
                "adapter_model.safetensors", 
                "config.json",
                "tokenizer.json",
                "final_application_report.json"
            ]
            
            for file in important_files:
                if (output_path / file).exists():
                    print(f"  ✓ {file}")
                else:
                    print(f"  ✗ {file} (缺失)")
            
            # 提供后续操作建议
            print(f"\n后续操作:")
            print(f"1. 查看训练报告: cat {output_dir}/final_application_report.json")
            print(f"2. 启动TensorBoard: uv run tensorboard --logdir {output_dir}/logs/tensorboard")
            print(f"3. 测试推理:")
            print(f'   uv run python -c "from src.inference_tester import InferenceTester; tester = InferenceTester(); tester.load_finetuned_model(\'{output_dir}\', \'Qwen/Qwen3-4B-Thinking-2507\'); print(tester.test_inference(\'请解释什么是深度学习？\'))"')
            
            break
    else:
        print("✗ 未找到微调结果目录")


def main():
    """主函数"""
    print("=== Qwen3-4B-Thinking-2507 专用微调启动器 ===\n")
    
    # 1. 检查要求
    if not check_requirements():
        print("系统要求检查失败，请解决问题后重试")
        sys.exit(1)
    
    # 2. 同步依赖
    if not sync_dependencies():
        print("依赖同步失败，请检查网络连接")
        sys.exit(1)
    
    # 3. 选择运行模式
    print("\n=== 选择运行模式 ===")
    print("1. 快速测试模式 (5轮训练，适合验证环境)")
    print("2. 低显存模式 (适合6-8GB GPU)")
    print("3. 标准模式 (推荐，适合10GB+GPU)")
    print("4. 配置文件模式 (使用qwen3_4b_thinking_config.json)")
    print("5. 环境检查模式 (仅检查环境，不训练)")
    
    choice = input("\n请选择模式 (1-5，默认3): ").strip() or "3"
    
    if choice == "1":
        success = run_qwen3_thinking_finetuning("quick_test")
    elif choice == "2":
        success = run_qwen3_thinking_finetuning("low_memory")
    elif choice == "3":
        success = run_qwen3_thinking_finetuning("default")
    elif choice == "4":
        if not Path("qwen3_4b_thinking_config.json").exists():
            print("配置文件不存在，使用默认模式")
            success = run_qwen3_thinking_finetuning("default")
        else:
            success = run_qwen3_thinking_finetuning("config_file")
    elif choice == "5":
        print("运行环境检查...")
        result = subprocess.run([
            "uv", "run", "python", "check_compatibility_2025.py"
        ])
        success = result.returncode == 0
    else:
        print("无效选择，使用标准模式")
        success = run_qwen3_thinking_finetuning("default")
    
    # 4. 显示结果
    if success and choice != "5":
        show_results()
    elif not success:
        print("\n故障排除建议:")
        print("1. 检查GPU内存是否足够 (nvidia-smi)")
        print("2. 尝试低显存模式")
        print("3. 查看日志文件: logs/application.log")
        print("4. 运行环境检查: uv run python check_compatibility_2025.py")
        print("5. 更新依赖: uv run python update_dependencies.py")


if __name__ == "__main__":
    main()
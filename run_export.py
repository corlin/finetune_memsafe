#!/usr/bin/env python3
"""
简化的Qwen3模型导出启动脚本

使用uv运行模型导出，自动处理环境和依赖
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主函数"""
    print("🚀 Qwen3模型导出工具 (uv版本)")
    print("=" * 50)
    
    # 检查uv是否可用
    try:
        subprocess.run(['uv', '--version'], check=True, capture_output=True)
        print("✅ uv已安装")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 错误: uv未安装或不在PATH中")
        print("\n请先安装uv:")
        print("Windows: powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        print("macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    # 检查pyproject.toml
    if not Path("pyproject.toml").exists():
        print("❌ 错误: 未找到pyproject.toml文件")
        print("请确保在项目根目录运行此脚本")
        sys.exit(1)
    
    # 检查checkpoint
    checkpoint_path = Path("qwen3-finetuned/checkpoint-30")
    if not checkpoint_path.exists():
        print(f"❌ 错误: Checkpoint目录不存在: {checkpoint_path}")
        print("请确保checkpoint-30目录存在")
        sys.exit(1)
    
    print("✅ 前提条件检查通过")
    print("\n📦 正在准备uv环境...")
    
    try:
        # 同步依赖
        print("正在同步依赖...")
        subprocess.run(['uv', 'sync'], check=True)
        print("✅ 依赖同步完成")
        
        # 运行导出脚本
        print("\n🔄 开始模型导出...")
        print("基座模型: Qwen/Qwen3-4B-Thinking-2507")
        print("Checkpoint: qwen3-finetuned/checkpoint-30")
        print("导出格式: PyTorch + ONNX")
        print("-" * 50)
        
        # 使用uv run执行导出
        cmd = [
            'uv', 'run', 'python', '-c',
            '''
import sys
from pathlib import Path
sys.path.append(str(Path(".") / "src"))

from src.model_export_controller import ModelExportController
from src.export_config import ExportConfiguration
from src.export_models import LogLevel
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/qwen3_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8')
    ]
)

# 创建导出配置
config = ExportConfiguration(
    checkpoint_path="./qwen3-finetuned/checkpoint-30",
    base_model_name="Qwen/Qwen3-4B-Thinking-2507",
    output_directory="./exported_models/qwen3_merged",
    quantization_level="int8",
    remove_training_artifacts=True,
    compress_weights=True,
    export_pytorch=True,
    export_onnx=True,
    export_tensorrt=False,
    onnx_dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"}
    },
    onnx_opset_version=20,
    onnx_optimize_graph=True,
    run_validation_tests=True,
    test_input_samples=[
        "你好，请介绍一下自己。",
        "什么是人工智能？请简单解释一下。",
        "请解释深度学习的基本概念。"
    ],
    enable_progress_monitoring=True,
    log_level=LogLevel.INFO,
    max_memory_usage_gb=12.0
)

# 执行导出
controller = ModelExportController(config)
result = controller.export_model()

if result.success:
    print("\\n✅ 模型导出成功！")
    print(f"导出ID: {result.export_id}")
    print(f"输出目录: {config.output_directory}")
    if hasattr(result, "pytorch_model_path") and result.pytorch_model_path:
        print(f"PyTorch模型: {result.pytorch_model_path}")
    if hasattr(result, "onnx_model_path") and result.onnx_model_path:
        print(f"ONNX模型: {result.onnx_model_path}")
else:
    print("\\n❌ 模型导出失败")
    print(f"错误: {result.error_message}")
    sys.exit(1)
'''
        ]
        
        result = subprocess.run(cmd, check=True)
        
        print("\n🎉 导出完成！")
        print("\n下一步:")
        print("1. 查看 ./exported_models/qwen3_merged/ 目录中的导出文件")
        print("2. 检查日志文件了解详细信息")
        print("3. 测试导出的模型")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 导出失败: {e}")
        print("请查看日志文件获取详细错误信息")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了导出过程")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
轻量级Qwen3模型导出启动脚本

使用更保守的内存设置，避免内存不足问题
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主函数"""
    print("🚀 Qwen3模型导出工具 (轻量级版本)")
    print("=" * 50)
    
    # 检查uv是否可用
    try:
        subprocess.run(['uv', '--version'], check=True, capture_output=True)
        print("✅ uv已安装")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 错误: uv未安装或不在PATH中")
        sys.exit(1)
    
    # 检查pyproject.toml
    if not Path("pyproject.toml").exists():
        print("❌ 错误: 未找到pyproject.toml文件")
        sys.exit(1)
    
    # 检查checkpoint
    checkpoint_path = Path("./enhanced-qwen3-finetuned/checkpoint-450")
    if not checkpoint_path.exists():
        print(f"❌ 错误: Checkpoint目录不存在: {checkpoint_path}")
        sys.exit(1)
    
    print("✅ 前提条件检查通过")
    print("\n📦 正在准备uv环境...")
    
    try:
        # 同步依赖
        print("正在同步依赖...")
        subprocess.run(['uv', 'sync'], check=True)
        print("✅ 依赖同步完成")
        
        # 运行导出脚本
        print("\n🔄 开始轻量级模型导出...")
        print("基座模型: Qwen/Qwen3-4B-Thinking-2507")
        print("Checkpoint: qwen3-finetuned/checkpoint-300")
        print("优化: 跳过权重压缩以节省内存")
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
import os

# 确保logs目录存在
os.makedirs("logs", exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/qwen3_export_lightweight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8')
    ]
)

# 创建轻量级导出配置
config = ExportConfiguration(
    checkpoint_path="./enhanced-qwen3-finetuned/checkpoint-450",
    base_model_name="Qwen/Qwen3-4B-Thinking-2507",
    output_directory="./exported_models/qwen3_merged_lightweight",
    quantization_level="none",  # 使用FP16而不是INT8，更快但文件稍大
    #"none", "fp16", "int8", "int4"
    remove_training_artifacts=True,
    compress_weights=False,  # 完全跳过权重压缩以节省内存
    export_pytorch=True,
    export_onnx=False,  # 完全跳过ONNX导出以节省内存
    export_tensorrt=False,
    onnx_opset_version=20,  # 使用ONNX opset版本20
    onnx_optimize_graph=True,
    onnx_dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"}
    },
    run_validation_tests=False,  # 跳过验证测试以节省时间和内存
    enable_progress_monitoring=True,
    log_level=LogLevel.INFO,
    max_memory_usage_gb=10.0  # 进一步降低内存限制
)

# 执行导出
controller = ModelExportController(config)
result = controller.export_model()

if result.success:
    print("\\n✅ 轻量级模型导出成功！")
    print(f"导出ID: {result.export_id}")
    print(f"输出目录: {config.output_directory}")
    if hasattr(result, "pytorch_model_path") and result.pytorch_model_path:
        print(f"PyTorch模型: {result.pytorch_model_path}")
    print("\\n💡 提示: 这是轻量级导出，跳过了一些优化步骤以节省内存")
else:
    print("\\n❌ 模型导出失败")
    print(f"错误: {result.error_message}")
    sys.exit(1)
'''
        ]
        
        result = subprocess.run(cmd, check=True)
        
        print("\n🎉 轻量级导出完成！")
        print("\n下一步:")
        print("1. 查看 ./exported_models/qwen3_merged_lightweight/ 目录中的导出文件")
        print("2. 如果需要完整优化，可以在内存充足时运行完整版本")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 导出失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了导出过程")
        sys.exit(1)

if __name__ == "__main__":
    main()
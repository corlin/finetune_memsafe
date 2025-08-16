#!/usr/bin/env python3
"""
最小化Qwen3模型导出启动脚本

完全跳过优化步骤，仅进行基本的模型合并和PyTorch导出
专为内存受限环境设计
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主函数"""
    print("🚀 Qwen3模型导出工具 (最小化版本)")
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
    checkpoint_path = Path("qwen3-finetuned/checkpoint-30")
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
        print("\n🔄 开始最小化模型导出...")
        print("基座模型: Qwen/Qwen3-4B-Thinking-2507")
        print("Checkpoint: qwen3-finetuned/checkpoint-30")
        print("导出格式: 仅PyTorch")
        print("优化: 完全跳过所有优化步骤")
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
        logging.FileHandler(f"logs/qwen3_export_minimal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8')
    ]
)

# 创建最小化导出配置
config = ExportConfiguration(
    checkpoint_path="./qwen3-finetuned/checkpoint-30",
    base_model_name="Qwen/Qwen3-4B-Thinking-2507",
    output_directory="./exported_models/qwen3_merged_minimal",
    quantization_level="none",  # 完全跳过量化
    remove_training_artifacts=False,  # 跳过训练artifacts移除
    compress_weights=False,  # 完全跳过权重压缩
    export_pytorch=True,
    export_onnx=False,  # 完全跳过ONNX导出
    export_tensorrt=False,
    run_validation_tests=False,  # 跳过验证测试
    enable_progress_monitoring=True,
    log_level=LogLevel.INFO,
    max_memory_usage_gb=8.0  # 最低内存限制
)

# 执行导出
controller = ModelExportController(config)
result = controller.export_model()

if result.success:
    print("\\n✅ 最小化模型导出成功！")
    print(f"导出ID: {result.export_id}")
    print(f"输出目录: {config.output_directory}")
    if hasattr(result, "pytorch_model_path") and result.pytorch_model_path:
        print(f"PyTorch模型: {result.pytorch_model_path}")
    print("\\n💡 提示: 这是最小化导出，跳过了所有优化步骤以节省内存")
    print("💡 如果需要优化，请在内存充足时运行完整版本")
else:
    print("\\n❌ 模型导出失败")
    print(f"错误: {result.error_message}")
    sys.exit(1)
'''
        ]
        
        result = subprocess.run(cmd, check=True)
        
        print("\n🎉 最小化导出完成！")
        print("\n下一步:")
        print("1. 查看 ./exported_models/qwen3_merged_minimal/ 目录中的导出文件")
        print("2. 如果需要优化版本，可以在内存充足时运行其他版本")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 导出失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了导出过程")
        sys.exit(1)

if __name__ == "__main__":
    main()
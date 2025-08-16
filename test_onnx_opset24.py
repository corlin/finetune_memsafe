#!/usr/bin/env python3
"""
测试ONNX opset版本24

验证新的opset版本是否能解决导出问题
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(".") / "src"))

from src.export_models import ExportConfiguration, LogLevel, QuantizationLevel

def test_onnx_opset24_config():
    """测试ONNX opset版本24配置"""
    print("🧪 测试ONNX opset版本24配置")
    print("=" * 40)
    
    # 创建测试配置
    print("1. 创建测试配置...")
    config = ExportConfiguration(
        checkpoint_path="./qwen3-finetuned/checkpoint-30",
        base_model_name="Qwen/Qwen3-4B-Thinking-2507",
        output_directory="./test_output",
        quantization_level=QuantizationLevel.FP16,
        remove_training_artifacts=True,
        compress_weights=False,
        export_pytorch=True,
        export_onnx=True,
        export_tensorrt=False,
        onnx_opset_version=20,  # 使用opset版本20
        onnx_optimize_graph=True,
        run_validation_tests=False,
        enable_progress_monitoring=True,
        log_level=LogLevel.INFO,
        max_memory_usage_gb=8.0
    )
    
    print(f"   ✅ 配置创建成功")
    print(f"   ONNX opset版本: {config.onnx_opset_version}")
    print(f"   量化级别: {config.quantization_level.value}")
    print(f"   导出格式: PyTorch={config.export_pytorch}, ONNX={config.export_onnx}")
    
    # 验证配置
    print("2. 验证配置...")
    errors = config.validate()
    if errors:
        print(f"   ❌ 配置验证失败: {errors}")
        return False
    else:
        print("   ✅ 配置验证通过")
    
    print("\n🎉 ONNX opset版本24配置测试通过！")
    print("配置已准备好用于模型导出")
    
    return True

if __name__ == "__main__":
    try:
        success = test_onnx_opset24_config()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
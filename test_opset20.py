#!/usr/bin/env python3
"""
测试ONNX opset版本20配置
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(".") / "src"))

from src.export_models import ExportConfiguration, LogLevel, QuantizationLevel

def test_opset20():
    """测试ONNX opset版本20配置"""
    print("🧪 测试ONNX opset版本20配置")
    print("=" * 40)
    
    # 创建测试配置
    config = ExportConfiguration(
        checkpoint_path="./qwen3-finetuned/checkpoint-30",
        base_model_name="Qwen/Qwen3-4B-Thinking-2507",
        output_directory="./test_output",
        quantization_level=QuantizationLevel.FP16,
        export_onnx=True,
        onnx_opset_version=20
    )
    
    print(f"✅ 配置创建成功")
    print(f"ONNX opset版本: {config.onnx_opset_version}")
    print(f"量化级别: {config.quantization_level.value}")
    print(f"导出ONNX: {config.export_onnx}")
    
    # 验证配置
    errors = config.validate()
    if errors:
        print(f"❌ 配置验证失败: {errors}")
        return False
    else:
        print("✅ 配置验证通过")
    
    print("\n🎉 ONNX opset版本20配置测试通过！")
    return True

if __name__ == "__main__":
    success = test_opset20()
    sys.exit(0 if success else 1)
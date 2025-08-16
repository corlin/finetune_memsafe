#!/usr/bin/env python3
"""
测试优化处理器修复

验证权重压缩功能是否正常工作
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(".") / "src"))

from src.optimization_processor import OptimizationProcessor
from src.export_models import QuantizationLevel

def create_test_model():
    """创建一个简单的测试模型"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(100, 50)
            self.linear2 = nn.Linear(50, 10)
            
        def forward(self, x):
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return x
    
    return SimpleModel()

def test_optimization_processor():
    """测试优化处理器"""
    print("🧪 测试优化处理器修复")
    print("=" * 40)
    
    try:
        # 创建测试模型
        print("1. 创建测试模型...")
        model = create_test_model()
        print(f"   模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 创建优化处理器
        print("2. 创建优化处理器...")
        processor = OptimizationProcessor(device="cpu", max_memory_gb=4.0)
        
        # 测试权重压缩
        print("3. 测试权重压缩...")
        try:
            compressed_model = processor.compress_model_weights(model)
            print("   ✅ 权重压缩成功")
        except Exception as e:
            print(f"   ❌ 权重压缩失败: {e}")
            return False
        
        # 测试量化
        print("4. 测试FP16量化...")
        try:
            quantized_model = processor.apply_quantization(model, QuantizationLevel.FP16)
            print("   ✅ FP16量化成功")
        except Exception as e:
            print(f"   ❌ FP16量化失败: {e}")
            return False
        
        # 测试训练artifacts移除
        print("5. 测试训练artifacts移除...")
        try:
            cleaned_model = processor.remove_training_artifacts(model)
            print("   ✅ 训练artifacts移除成功")
        except Exception as e:
            print(f"   ❌ 训练artifacts移除失败: {e}")
            return False
        
        # 获取优化报告
        print("6. 获取优化报告...")
        try:
            report = processor.get_optimization_report()
            print("   ✅ 优化报告生成成功")
            print(f"   原始大小: {report['optimization_stats']['original_size_mb']:.2f} MB")
            print(f"   优化后大小: {report['optimization_stats']['optimized_size_mb']:.2f} MB")
        except Exception as e:
            print(f"   ❌ 优化报告生成失败: {e}")
            return False
        
        print("\n🎉 所有测试通过！优化处理器修复成功")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_optimization_processor()
    sys.exit(0 if success else 1)
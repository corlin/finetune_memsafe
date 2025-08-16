#!/usr/bin/env python3
"""
PyTorch模型导出功能演示

这个脚本演示了如何使用模型导出优化系统来导出PyTorch格式的模型。
"""

import os
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn

# 导入我们的模块
from src.export_config import ConfigurationManager
from src.export_models import ExportConfiguration, QuantizationLevel
from src.optimization_processor import OptimizationProcessor
from src.format_exporter import FormatExporter


class DemoModel(nn.Module):
    """演示用的简单模型"""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 256)
        self.transformer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.classifier = nn.Linear(256, 10)
        
        # 模拟配置
        class Config:
            def __init__(self):
                self.model_type = 'demo_model'
                self.vocab_size = 1000
                self.hidden_size = 256
                self.num_layers = 1
            
            def to_dict(self):
                return {
                    'model_type': 'demo_model',
                    'vocab_size': 1000,
                    'hidden_size': 256,
                    'num_layers': 1,
                    'architectures': ['DemoModel']
                }
        
        self.config = Config()
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均池化
        return self.classifier(x)
    
    def save_pretrained(self, path, **kwargs):
        """模拟transformers的save_pretrained方法"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        import json
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # 保存模型权重
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        
        # 创建一个模拟的safetensors文件
        (path / "model.safetensors").write_bytes(b"mock_safetensors_data" * 1000)


class DemoTokenizer:
    """演示用的简单tokenizer"""
    
    def __init__(self):
        self.vocab_size = 1000
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
    
    def save_pretrained(self, path):
        """模拟保存tokenizer"""
        path = Path(path)
        
        # 创建tokenizer配置
        import json
        tokenizer_config = {
            "tokenizer_class": "DemoTokenizer",
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token
        }
        
        with open(path / "tokenizer_config.json", 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        
        # 创建模拟的tokenizer.json
        with open(path / "tokenizer.json", 'w') as f:
            json.dump({"version": "1.0", "vocab": {}}, f)


def main():
    """主演示函数"""
    print("🚀 PyTorch模型导出功能演示")
    print("=" * 50)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = os.path.join(temp_dir, "demo_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # 1. 创建演示模型和tokenizer
        print("\n📦 1. 创建演示模型和tokenizer")
        model = DemoModel()
        tokenizer = DemoTokenizer()
        
        print(f"   ✓ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   ✓ 模型大小: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024:.1f} MB")
        
        # 2. 配置导出参数
        print("\n⚙️  2. 配置导出参数")
        ##checkpoint_dir,
        #"demo/demo-model",
        config = ExportConfiguration(
            checkpoint_path="qwen3-finetuned/checkpoint-30",
            base_model_name="Qwen/Qwen3-4B-Thinking-2507",
            output_directory=temp_dir,
            quantization_level=QuantizationLevel.INT8,
            export_pytorch=True,
            save_tokenizer=True,
            run_validation_tests=False
        )
        
        print(f"   ✓ 输出目录: {config.output_directory}")
        print(f"   ✓ 量化级别: {config.quantization_level.value}")
        print(f"   ✓ 导出格式: PyTorch")
        
        # 3. 模型优化
        print("\n🔧 3. 模型优化处理")
        optimizer = OptimizationProcessor(device="cpu", max_memory_gb=4.0)
        
        # 应用优化
        print("   正在移除训练artifacts...")
        optimized_model = optimizer.remove_training_artifacts(model)
        
        print("   正在压缩模型权重...")
        optimized_model = optimizer.compress_model_weights(optimized_model)
        
        # 获取优化统计
        optimization_report = optimizer.get_optimization_report()
        optimization_steps = optimization_report['optimization_stats']['optimization_steps']
        
        print(f"   ✓ 完成 {len(optimization_steps)} 个优化步骤")
        for i, step in enumerate(optimization_steps, 1):
            print(f"     {i}. {step['step']}: {step['size_reduction_percentage']:.1f}% 大小减少")
        
        # 4. PyTorch格式导出
        print("\n📤 4. PyTorch格式导出")
        exporter = FormatExporter(config)
        
        print("   正在导出PyTorch模型...")
        export_path = exporter.export_pytorch_model(optimized_model, tokenizer)
        
        print(f"   ✓ 导出完成: {export_path}")
        
        # 检查导出的文件
        export_path_obj = Path(export_path)
        exported_files = list(export_path_obj.iterdir())
        print(f"   ✓ 导出文件数量: {len(exported_files)}")
        
        for file_path in exported_files:
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"     - {file_path.name}: {size_mb:.2f} MB")
        
        # 5. 验证导出结果
        print("\n✅ 5. 验证导出结果")
        
        # 检查导出统计
        export_stats = exporter.get_export_stats()
        print(f"   ✓ 导出成功: {export_stats['pytorch_export']['success']}")
        print(f"   ✓ 导出大小: {export_stats['pytorch_export']['size_mb']:.1f} MB")
        print(f"   ✓ 总导出次数: {export_stats['total_exports']}")
        print(f"   ✓ 成功导出次数: {export_stats['successful_exports']}")
        
        # 检查元数据
        metadata_path = export_path_obj / 'export_metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"   ✓ 导出时间: {metadata['export_info']['export_timestamp']}")
            print(f"   ✓ 模型类型: {metadata['model_info']['model_type']}")
            print(f"   ✓ 参数数量: {metadata['model_info']['parameter_count']:,}")
        
        # 6. 生成使用示例
        print("\n📝 6. 生成使用示例")
        usage_example = exporter.generate_usage_example(export_path)
        
        example_path = export_path_obj / "usage_example.py"
        if example_path.exists():
            print(f"   ✓ 使用示例已保存: {example_path}")
            print("   ✓ 示例内容预览:")
            with open(example_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:10]  # 显示前10行
                for i, line in enumerate(lines, 1):
                    print(f"     {i:2d}: {line.rstrip()}")
                if len(lines) == 10:
                    print("     ... (更多内容请查看文件)")
        
        # 7. 创建部署包
        print("\n📦 7. 创建部署包")
        try:
            package_path = exporter.create_deployment_package(export_path)
            package_size = Path(package_path).stat().st_size / 1024 / 1024
            print(f"   ✓ 部署包创建成功: {Path(package_path).name}")
            print(f"   ✓ 部署包大小: {package_size:.1f} MB")
        except Exception as e:
            print(f"   ⚠️  部署包创建跳过: {e}")
        
        # 8. 总结
        print("\n🎉 导出完成总结")
        print("=" * 30)
        print(f"导出路径: {export_path}")
        print(f"导出大小: {export_stats['pytorch_export']['size_mb']:.1f} MB")
        print(f"优化步骤: {len(optimization_steps)}")
        print(f"导出文件: {len(exported_files)}")
        
        print("\n✨ PyTorch模型导出演示完成！")
        print(f"📁 所有文件保存在: {temp_dir}")
        
        # 询问是否保留文件
        try:
            keep_files = input("\n是否保留导出的文件？(y/N): ").lower().strip()
            if keep_files in ['y', 'yes']:
                print(f"文件已保留在: {temp_dir}")
                return temp_dir
        except (KeyboardInterrupt, EOFError):
            pass
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理临时文件（如果用户选择不保留）
        try:
            if 'keep_files' not in locals() or keep_files not in ['y', 'yes']:
                shutil.rmtree(temp_dir)
                print(f"\n🧹 临时文件已清理")
        except Exception as e:
            print(f"清理临时文件时出错: {e}")


if __name__ == "__main__":
    main()
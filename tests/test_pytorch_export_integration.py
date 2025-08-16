"""
PyTorch格式导出的集成测试

这个测试演示了完整的PyTorch模型导出流程，包括：
1. 配置管理
2. 模型合并
3. 模型优化
4. PyTorch格式导出
5. 导出验证
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import torch
import torch.nn as nn
import pytest

from src.export_config import ConfigurationManager
from src.export_models import ExportConfiguration, QuantizationLevel
from src.model_merger import ModelMerger
from src.optimization_processor import OptimizationProcessor
from src.format_exporter import FormatExporter


class MockQwenModel(nn.Module):
    """模拟Qwen模型用于集成测试"""
    
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32000, 4096)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=4096, nhead=32, batch_first=True)
            for _ in range(2)  # 简化为2层
        ])
        self.norm = nn.LayerNorm(4096)
        self.lm_head = nn.Linear(4096, 32000)
        
        # 模拟配置
        self.config = Mock()
        self.config.model_type = "qwen2"
        self.config.vocab_size = 32000
        self.config.hidden_size = 4096
        self.config.num_hidden_layers = 2
        self.config.num_attention_heads = 32
        self.config.to_dict.return_value = {
            'model_type': 'qwen2',
            'vocab_size': 32000,
            'hidden_size': 4096,
            'num_hidden_layers': 2,
            'num_attention_heads': 32,
            'architectures': ['Qwen2ForCausalLM']
        }
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return Mock(logits=logits)
    
    def save_pretrained(self, path, **kwargs):
        """模拟save_pretrained方法"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        import json
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # 保存模型权重
        model_path = path / "model.safetensors"
        # 创建一个足够大的模拟文件
        model_path.write_bytes(b"mock_model_weights" * 10000)
        
        # 保存pytorch_model.bin作为备选
        torch.save(self.state_dict(), path / "pytorch_model.bin")
    
    def eval(self):
        """设置为评估模式"""
        super().eval()
        return self


class TestPyTorchExportIntegration:
    """PyTorch导出集成测试"""
    
    def setup_method(self):
        """测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟checkpoint目录
        self.checkpoint_dir = os.path.join(self.temp_dir, "mock_checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 创建测试配置
        self.config = ExportConfiguration(
            checkpoint_path=self.checkpoint_dir,
            base_model_name="Qwen/Qwen2-4B",
            output_directory=self.temp_dir,
            quantization_level=QuantizationLevel.NONE,
            export_pytorch=True,
            save_tokenizer=True,
            run_validation_tests=False  # 跳过验证以简化测试
        )
        
        # 创建模拟模型
        self.mock_model = MockQwenModel()
        
    def teardown_method(self):
        """测试后的清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_mock_tokenizer(self):
        """创建模拟tokenizer"""
        tokenizer = Mock()
        tokenizer.save_pretrained = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<|endoftext|>"
        return tokenizer
    
    @patch('src.format_exporter.ensure_disk_space')
    @patch('src.format_exporter.get_directory_size_mb')
    def test_complete_pytorch_export_workflow(self, mock_get_size, mock_ensure_disk):
        """测试完整的PyTorch导出工作流程"""
        mock_get_size.return_value = 150.0  # 模拟导出大小150MB
        
        print("\n=== PyTorch模型导出集成测试 ===")
        
        # 1. 配置管理测试
        print("1. 测试配置管理...")
        config_manager = ConfigurationManager()
        
        # 验证配置
        validation_errors = config_manager.validate_configuration(self.config)
        assert len(validation_errors) == 0, f"配置验证失败: {validation_errors}"
        print("   ✓ 配置验证通过")
        
        # 2. 模型优化测试
        print("2. 测试模型优化...")
        optimizer = OptimizationProcessor(device="cpu")
        
        # 应用优化（跳过量化以简化测试）
        optimized_model = optimizer.remove_training_artifacts(self.mock_model)
        assert optimized_model is not None
        print("   ✓ 模型优化完成")
        
        # 获取优化统计
        optimization_report = optimizer.get_optimization_report()
        assert 'optimization_stats' in optimization_report
        print(f"   ✓ 优化统计: {len(optimization_report['optimization_stats']['optimization_steps'])} 个步骤")
        
        # 3. PyTorch格式导出测试
        print("3. 测试PyTorch格式导出...")
        exporter = FormatExporter(self.config)
        
        # 创建tokenizer
        tokenizer = self.create_mock_tokenizer()
        
        # 执行导出
        export_path = exporter.export_pytorch_model(optimized_model, tokenizer)
        
        # 验证导出结果
        assert os.path.exists(export_path)
        assert os.path.isdir(export_path)
        print(f"   ✓ 模型导出到: {export_path}")
        
        # 检查导出的文件
        export_path_obj = Path(export_path)
        
        # 检查必需文件
        required_files = ['config.json', 'export_metadata.json']
        for file_name in required_files:
            file_path = export_path_obj / file_name
            assert file_path.exists(), f"缺失文件: {file_name}"
            print(f"   ✓ 找到文件: {file_name}")
        
        # 检查模型权重文件
        model_files = list(export_path_obj.glob('*.safetensors')) + list(export_path_obj.glob('*.bin'))
        assert len(model_files) > 0, "未找到模型权重文件"
        print(f"   ✓ 找到模型权重文件: {[f.name for f in model_files]}")
        
        # 4. 验证导出统计
        print("4. 验证导出统计...")
        export_stats = exporter.get_export_stats()
        
        assert export_stats['pytorch_export']['success'] is True
        assert export_stats['pytorch_export']['size_mb'] == 150.0
        assert export_stats['total_exports'] == 1
        assert export_stats['successful_exports'] == 1
        print("   ✓ 导出统计验证通过")
        
        # 5. 验证元数据
        print("5. 验证导出元数据...")
        metadata_path = export_path_obj / 'export_metadata.json'
        
        import json
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 检查元数据结构
        assert 'export_info' in metadata
        assert 'model_info' in metadata
        assert metadata['export_info']['export_format'] == 'pytorch'
        assert metadata['export_info']['base_model'] == 'Qwen/Qwen2-4B'
        assert metadata['model_info']['model_type'] == 'qwen2'
        print("   ✓ 元数据验证通过")
        
        # 6. 测试使用示例生成
        print("6. 测试使用示例生成...")
        usage_example = exporter.generate_usage_example(export_path)
        
        assert 'AutoModelForCausalLM' in usage_example
        assert 'AutoTokenizer' in usage_example
        assert export_path in usage_example
        print("   ✓ 使用示例生成成功")
        
        # 7. 测试部署包创建
        print("7. 测试部署包创建...")
        with patch('src.format_exporter.shutil.make_archive') as mock_make_archive:
            # 模拟ZIP文件创建
            def mock_make_archive_side_effect(base_name, format, root_dir):
                zip_path = Path(f"{base_name}.zip")
                zip_path.write_bytes(b"mock_deployment_package" * 1000)
                return base_name
            
            mock_make_archive.side_effect = mock_make_archive_side_effect
            
            package_path = exporter.create_deployment_package(export_path)
            assert package_path.endswith('.zip')
            print(f"   ✓ 部署包创建成功: {Path(package_path).name}")
        
        print("\n=== 集成测试完成 ===")
        print(f"导出路径: {export_path}")
        print(f"导出大小: {export_stats['pytorch_export']['size_mb']} MB")
        print(f"优化步骤: {len(optimization_report['optimization_stats']['optimization_steps'])}")
        print("所有测试通过！")
    
    def test_configuration_file_workflow(self):
        """测试配置文件工作流程"""
        print("\n=== 配置文件工作流程测试 ===")
        
        # 1. 创建配置模板
        config_manager = ConfigurationManager()
        template_path = os.path.join(self.temp_dir, "export_config_template.yaml")
        
        config_manager.create_config_template(template_path)
        assert os.path.exists(template_path)
        print(f"   ✓ 配置模板创建: {template_path}")
        
        # 2. 保存当前配置
        config_path = os.path.join(self.temp_dir, "current_config.yaml")
        config_manager.save_configuration(self.config, config_path)
        assert os.path.exists(config_path)
        print(f"   ✓ 配置保存: {config_path}")
        
        # 3. 加载配置
        loaded_config = config_manager.load_configuration(config_path)
        assert loaded_config.base_model_name == self.config.base_model_name
        assert loaded_config.quantization_level == self.config.quantization_level
        print("   ✓ 配置加载验证通过")
        
        print("配置文件工作流程测试完成！")
    
    @patch('src.format_exporter.ensure_disk_space')
    def test_error_handling_workflow(self, mock_ensure_disk):
        """测试错误处理工作流程"""
        print("\n=== 错误处理工作流程测试 ===")
        
        # 1. 测试磁盘空间不足错误
        from src.export_exceptions import DiskSpaceError
        mock_ensure_disk.side_effect = DiskSpaceError("磁盘空间不足")
        
        exporter = FormatExporter(self.config)
        
        with pytest.raises(Exception) as exc_info:
            exporter.export_pytorch_model(self.mock_model)
        
        assert "磁盘空间不足" in str(exc_info.value) or "PyTorch模型导出失败" in str(exc_info.value)
        print("   ✓ 磁盘空间不足错误处理正确")
        
        # 2. 验证错误统计
        export_stats = exporter.get_export_stats()
        assert export_stats['pytorch_export']['success'] is False
        assert export_stats['total_exports'] == 1
        assert export_stats['successful_exports'] == 0
        print("   ✓ 错误统计记录正确")
        
        print("错误处理工作流程测试完成！")
    
    def test_memory_optimization_workflow(self):
        """测试内存优化工作流程"""
        print("\n=== 内存优化工作流程测试 ===")
        
        # 创建优化处理器
        optimizer = OptimizationProcessor(device="cpu", max_memory_gb=4.0)
        
        # 1. 测试不同的优化策略
        print("1. 测试权重压缩...")
        compressed_model = optimizer.compress_model_weights(self.mock_model)
        assert compressed_model is not None
        print("   ✓ 权重压缩完成")
        
        print("2. 测试结构优化...")
        structure_optimized_model = optimizer.optimize_model_structure(compressed_model)
        assert structure_optimized_model is not None
        print("   ✓ 结构优化完成")
        
        print("3. 测试训练artifacts清理...")
        cleaned_model = optimizer.remove_training_artifacts(structure_optimized_model)
        assert cleaned_model is not None
        print("   ✓ 训练artifacts清理完成")
        
        # 2. 获取优化报告
        optimization_report = optimizer.get_optimization_report()
        
        assert 'optimization_stats' in optimization_report
        assert 'system_info' in optimization_report
        
        optimization_steps = optimization_report['optimization_stats']['optimization_steps']
        assert len(optimization_steps) >= 3  # 至少3个优化步骤
        
        print(f"   ✓ 完成 {len(optimization_steps)} 个优化步骤")
        
        # 3. 清理内存
        optimizer.cleanup()
        print("   ✓ 内存清理完成")
        
        print("内存优化工作流程测试完成！")


if __name__ == "__main__":
    # 运行集成测试
    test_instance = TestPyTorchExportIntegration()
    test_instance.setup_method()
    
    try:
        test_instance.test_complete_pytorch_export_workflow()
        test_instance.test_configuration_file_workflow()
        test_instance.test_memory_optimization_workflow()
        print("\n🎉 所有集成测试通过！")
    except Exception as e:
        print(f"\n❌ 集成测试失败: {e}")
        raise
    finally:
        test_instance.teardown_method()
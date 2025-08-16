"""
格式导出器的单元测试
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
import torch.nn as nn

from src.format_exporter import FormatExporter
from src.export_models import ExportConfiguration, QuantizationLevel, LogLevel
from src.export_exceptions import FormatExportError


class MockModel(nn.Module):
    """用于测试的模拟模型"""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 10)
        self.config = Mock()
        self.config.model_type = "test_model"
        self.config.vocab_size = 1000
        self.config.hidden_size = 100
        self.config.num_hidden_layers = 2
        self.config.to_dict.return_value = {
            'model_type': 'test_model',
            'vocab_size': 1000,
            'hidden_size': 100,
            'num_hidden_layers': 2
        }
        
    def forward(self, x):
        return self.linear(x)
    
    def save_pretrained(self, path, **kwargs):
        """模拟save_pretrained方法"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 创建模拟的配置文件
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f)
        
        # 创建模拟的模型权重文件
        model_path = path / "model.safetensors"
        model_path.write_bytes(b"mock_model_weights")
    
    def eval(self):
        """模拟eval方法"""
        return self


class TestFormatExporter:
    """FormatExporter测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试配置
        self.config = ExportConfiguration(
            checkpoint_path="test_checkpoint",
            base_model_name="test_model",
            output_directory=self.temp_dir,
            quantization_level=QuantizationLevel.NONE,
            export_pytorch=True,
            save_tokenizer=True
        )
        
        self.exporter = FormatExporter(self.config)
        self.mock_model = MockModel()
        
    def teardown_method(self):
        """测试后的清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_mock_tokenizer(self):
        """创建模拟tokenizer"""
        tokenizer = Mock()
        tokenizer.save_pretrained = Mock()
        return tokenizer
    
    def test_initialization(self):
        """测试初始化"""
        assert self.exporter.config == self.config
        assert 'pytorch_export' in self.exporter.export_stats
        assert 'total_exports' in self.exporter.export_stats
        assert self.exporter.export_stats['total_exports'] == 0
    
    @patch('src.format_exporter.ensure_disk_space')
    @patch('src.format_exporter.get_directory_size_mb')
    def test_export_pytorch_model_success(self, mock_get_size, mock_ensure_disk):
        """测试成功导出PyTorch模型"""
        mock_get_size.return_value = 100.0
        
        tokenizer = self.create_mock_tokenizer()
        
        result_path = self.exporter.export_pytorch_model(self.mock_model, tokenizer)
        
        # 验证结果
        assert os.path.exists(result_path)
        assert Path(result_path).is_dir()
        
        # 验证统计信息更新
        stats = self.exporter.export_stats
        assert stats['pytorch_export']['success'] is True
        assert stats['pytorch_export']['size_mb'] == 100.0
        assert stats['total_exports'] == 1
        assert stats['successful_exports'] == 1
        
        # 验证tokenizer保存被调用
        tokenizer.save_pretrained.assert_called_once()
        
        # 验证磁盘空间检查被调用
        mock_ensure_disk.assert_called_once()
    
    @patch('src.format_exporter.ensure_disk_space')
    def test_export_pytorch_model_without_tokenizer(self, mock_ensure_disk):
        """测试不带tokenizer的PyTorch模型导出"""
        result_path = self.exporter.export_pytorch_model(self.mock_model)
        
        # 验证结果
        assert os.path.exists(result_path)
        assert self.exporter.export_stats['pytorch_export']['success'] is True
    
    @patch('src.format_exporter.ensure_disk_space')
    def test_export_pytorch_model_custom_path(self, mock_ensure_disk):
        """测试使用自定义路径导出PyTorch模型"""
        custom_path = os.path.join(self.temp_dir, "custom_model")
        
        result_path = self.exporter.export_pytorch_model(self.mock_model, output_path=custom_path)
        
        assert result_path == custom_path
        assert os.path.exists(custom_path)
    
    def test_export_pytorch_model_failure(self):
        """测试PyTorch模型导出失败"""
        # 创建一个会抛出异常的模型
        failing_model = Mock()
        failing_model.eval.return_value = failing_model
        failing_model.save_pretrained.side_effect = Exception("保存失败")
        failing_model.parameters.return_value = [torch.tensor([1.0])]
        failing_model.buffers.return_value = []
        
        with pytest.raises(FormatExportError) as exc_info:
            self.exporter.export_pytorch_model(failing_model)
        
        assert "PyTorch模型导出失败" in str(exc_info.value)
        
        # 验证统计信息
        stats = self.exporter.export_stats
        assert stats['pytorch_export']['success'] is False
        assert stats['total_exports'] == 1
        assert stats['successful_exports'] == 0
    
    def test_save_pytorch_model(self):
        """测试保存PyTorch模型"""
        output_path = Path(self.temp_dir) / "test_model"
        output_path.mkdir()
        
        # 这个方法会调用模型的save_pretrained
        self.exporter._save_pytorch_model(self.mock_model, output_path)
        
        # 验证配置文件被创建
        assert (output_path / "config.json").exists()
    
    def test_save_tokenizer_success(self):
        """测试成功保存tokenizer"""
        tokenizer = self.create_mock_tokenizer()
        output_path = Path(self.temp_dir) / "test_model"
        output_path.mkdir()
        
        self.exporter._save_tokenizer(tokenizer, output_path)
        
        tokenizer.save_pretrained.assert_called_once_with(output_path)
    
    def test_save_tokenizer_failure(self):
        """测试tokenizer保存失败"""
        tokenizer = Mock()
        tokenizer.save_pretrained.side_effect = Exception("保存失败")
        
        output_path = Path(self.temp_dir) / "test_model"
        output_path.mkdir()
        
        # 应该不抛出异常，只记录警告
        self.exporter._save_tokenizer(tokenizer, output_path)
    
    def test_save_model_metadata(self):
        """测试保存模型元信息"""
        output_path = Path(self.temp_dir) / "test_model"
        output_path.mkdir()
        
        self.exporter._save_model_metadata(self.mock_model, output_path)
        
        # 验证元信息文件被创建
        metadata_path = output_path / "export_metadata.json"
        assert metadata_path.exists()
        
        # 验证元信息内容
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        assert 'export_info' in metadata
        assert 'model_info' in metadata
        assert metadata['export_info']['export_format'] == 'pytorch'
        assert metadata['model_info']['model_type'] == 'test_model'
    
    def test_verify_pytorch_export_success(self):
        """测试PyTorch导出验证成功"""
        output_path = Path(self.temp_dir) / "test_model"
        output_path.mkdir()
        
        # 创建必需文件
        (output_path / "config.json").write_text('{"model_type": "test"}')
        (output_path / "model.safetensors").write_bytes(b"mock_weights")
        
        # 模拟AutoConfig.from_pretrained
        with patch('src.format_exporter.AutoConfig') as mock_config:
            mock_config.from_pretrained.return_value = Mock(model_type="test")
            
            # 应该不抛出异常
            self.exporter._verify_pytorch_export(output_path)
    
    def test_verify_pytorch_export_missing_files(self):
        """测试PyTorch导出验证缺失文件"""
        output_path = Path(self.temp_dir) / "test_model"
        output_path.mkdir()
        
        with pytest.raises(FormatExportError) as exc_info:
            self.exporter._verify_pytorch_export(output_path)
        
        assert "未找到模型权重文件" in str(exc_info.value)
    
    def test_verify_pytorch_export_missing_config(self):
        """测试PyTorch导出验证缺失配置"""
        output_path = Path(self.temp_dir) / "test_model"
        output_path.mkdir()
        
        # 只创建权重文件，不创建配置文件
        (output_path / "model.safetensors").write_bytes(b"mock_weights")
        
        with pytest.raises(FormatExportError) as exc_info:
            self.exporter._verify_pytorch_export(output_path)
        
        assert "缺失必需文件: config.json" in str(exc_info.value)
    
    @patch('src.format_exporter.AutoConfig')
    @patch('src.format_exporter.AutoModelForCausalLM')
    @patch('src.format_exporter.AutoTokenizer')
    def test_test_pytorch_model_loading_success(self, mock_tokenizer, mock_model, mock_config):
        """测试PyTorch模型加载测试成功"""
        # 设置mock
        mock_config.from_pretrained.return_value = Mock(
            model_type="test",
            vocab_size=1000,
            hidden_size=100,
            num_hidden_layers=2
        )
        
        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = [torch.tensor([1.0, 2.0])]
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer.from_pretrained.return_value = Mock()
        
        # 执行测试
        result = self.exporter.test_pytorch_model_loading("test_path")
        
        # 验证结果
        assert result['success'] is True
        assert result['load_time_seconds'] >= 0
        assert result['model_info']['model_type'] == "test"
        assert result['model_info']['parameter_count'] == 2
        assert result['model_info']['has_tokenizer'] is True
    
    @patch('src.format_exporter.AutoConfig')
    def test_test_pytorch_model_loading_failure(self, mock_config):
        """测试PyTorch模型加载测试失败"""
        mock_config.from_pretrained.side_effect = Exception("加载失败")
        
        result = self.exporter.test_pytorch_model_loading("test_path")
        
        assert result['success'] is False
        assert result['error_message'] == "加载失败"
    
    @patch('src.format_exporter.shutil.make_archive')
    def test_create_deployment_package(self, mock_make_archive):
        """测试创建部署包"""
        model_path = "test_model_path"
        
        # 模拟ZIP文件创建
        def mock_make_archive_side_effect(base_name, format, root_dir):
            zip_path = Path(f"{base_name}.zip")
            zip_path.write_bytes(b"mock_zip_content")
            return base_name
        
        mock_make_archive.side_effect = mock_make_archive_side_effect
        
        result = self.exporter.create_deployment_package(model_path)
        
        # 验证结果
        assert result.endswith('.zip')
        mock_make_archive.assert_called_once()
    
    def test_generate_usage_example(self):
        """测试生成使用示例"""
        model_path = "test_model_path"
        
        example_code = self.exporter.generate_usage_example(model_path)
        
        # 验证示例代码包含必要内容
        assert "AutoModelForCausalLM" in example_code
        assert "AutoTokenizer" in example_code
        assert "generate_response" in example_code
        assert model_path in example_code
    
    def test_get_export_stats(self):
        """测试获取导出统计信息"""
        stats = self.exporter.get_export_stats()
        
        assert isinstance(stats, dict)
        assert 'pytorch_export' in stats
        assert 'total_exports' in stats
        assert 'successful_exports' in stats
    
    def test_estimate_pytorch_export_size(self):
        """测试估算PyTorch导出大小"""
        size = self.exporter._estimate_pytorch_export_size(self.mock_model)
        
        assert isinstance(size, float)
        assert size > 0
    
    def test_calculate_model_memory_size(self):
        """测试计算模型内存大小"""
        size = self.exporter._calculate_model_memory_size(self.mock_model)
        
        assert isinstance(size, float)
        assert size > 0
    
    def test_generate_timestamp(self):
        """测试生成时间戳"""
        timestamp = self.exporter._generate_timestamp()
        
        assert isinstance(timestamp, str)
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS
        assert '_' in timestamp
    
    def test_extract_model_name(self):
        """测试提取模型名称"""
        # 测试路径格式
        name1 = self.exporter._extract_model_name("Qwen/Qwen3-4B-Thinking-2507")
        assert name1 == "Qwen_Qwen3-4B-Thinking-2507"
        
        # 测试Windows路径
        name2 = self.exporter._extract_model_name("C:\\models\\test_model")
        assert name2 == "C__models_test_model"
        
        # 测试长名称截取
        long_name = "a" * 60
        name3 = self.exporter._extract_model_name(long_name)
        assert len(name3) == 50
    
    def test_config_save_tokenizer_false(self):
        """测试配置不保存tokenizer时的行为"""
        # 修改配置
        self.config.save_tokenizer = False
        exporter = FormatExporter(self.config)
        
        tokenizer = self.create_mock_tokenizer()
        
        with patch('src.format_exporter.ensure_disk_space'), \
             patch('src.format_exporter.get_directory_size_mb', return_value=100.0):
            
            result_path = exporter.export_pytorch_model(self.mock_model, tokenizer)
            
            # tokenizer.save_pretrained不应该被调用
            tokenizer.save_pretrained.assert_not_called()
            
            # 但导出应该成功
            assert os.path.exists(result_path)
    
    @patch('src.format_exporter.ensure_disk_space')
    def test_disk_space_check_failure(self, mock_ensure_disk):
        """测试磁盘空间检查失败"""
        from src.export_exceptions import DiskSpaceError
        mock_ensure_disk.side_effect = DiskSpaceError("磁盘空间不足")
        
        with pytest.raises(FormatExportError) as exc_info:
            self.exporter.export_pytorch_model(self.mock_model)
        
        assert "PyTorch模型导出失败" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
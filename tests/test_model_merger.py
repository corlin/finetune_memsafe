"""
模型合并器的单元测试
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch

from src.model_merger import ModelMerger
from src.export_exceptions import ModelMergeError, CheckpointValidationError


class TestModelMerger:
    """ModelMerger测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.merger = ModelMerger(device="cpu", max_memory_gb=4.0)
        
    def teardown_method(self):
        """测试后的清理"""
        self.merger.cleanup()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_mock_checkpoint(self, checkpoint_dir: str) -> str:
        """创建模拟checkpoint目录"""
        checkpoint_path = Path(checkpoint_dir) / "mock_checkpoint"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # 创建必需文件
        adapter_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.1,
            "bias": "none"
        }
        
        with open(checkpoint_path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f)
        
        # 创建adapter模型文件
        (checkpoint_path / "adapter_model.safetensors").write_bytes(b"mock_model_data" * 100000)
        
        return str(checkpoint_path)
    
    @patch('src.model_merger.AutoModelForCausalLM')
    @patch('src.model_merger.ensure_memory_available')
    def test_load_base_model_success(self, mock_ensure_memory, mock_auto_model):
        """测试成功加载基座模型"""
        # 设置mock
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.tensor([1.0, 2.0])]
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # 执行测试
        result = self.merger.load_base_model("test_model")
        
        # 验证结果
        assert result == mock_model
        assert self.merger._base_model == mock_model
        mock_auto_model.from_pretrained.assert_called_once()
        mock_ensure_memory.assert_called_once()
    
    @patch('src.model_merger.AutoModelForCausalLM')
    def test_load_base_model_failure(self, mock_auto_model):
        """测试基座模型加载失败"""
        # 设置mock抛出异常
        mock_auto_model.from_pretrained.side_effect = Exception("Model not found")
        
        # 执行测试并验证异常
        with pytest.raises(ModelMergeError) as exc_info:
            self.merger.load_base_model("nonexistent_model")
        
        assert "加载基座模型失败" in str(exc_info.value)
    
    @patch('src.model_merger.PeftModel')
    @patch('src.model_merger.PeftConfig')
    def test_load_lora_adapter_success(self, mock_peft_config, mock_peft_model):
        """测试成功加载LoRA适配器"""
        # 创建模拟checkpoint
        checkpoint_path = self.create_mock_checkpoint(self.temp_dir)
        
        # 设置mock
        mock_config = Mock()
        mock_config.peft_type = "LORA"
        mock_config.r = 16
        mock_config.lora_alpha = 32
        mock_peft_config.from_pretrained.return_value = mock_config
        
        mock_adapter = Mock()
        mock_peft_model.from_pretrained.return_value = mock_adapter
        
        # 设置基座模型
        self.merger._base_model = Mock()
        
        # 执行测试
        result = self.merger.load_lora_adapter(checkpoint_path)
        
        # 验证结果
        assert result == mock_adapter
        assert self.merger._peft_model == mock_adapter
        mock_peft_config.from_pretrained.assert_called_once_with(checkpoint_path)
        mock_peft_model.from_pretrained.assert_called_once()
    
    def test_load_lora_adapter_no_base_model(self):
        """测试在没有基座模型时加载适配器"""
        checkpoint_path = self.create_mock_checkpoint(self.temp_dir)
        
        with pytest.raises(ModelMergeError) as exc_info:
            self.merger.load_lora_adapter(checkpoint_path)
        
        assert "必须先加载基座模型" in str(exc_info.value)
    
    def test_load_lora_adapter_invalid_checkpoint(self):
        """测试加载无效checkpoint"""
        # 创建空目录（无效checkpoint）
        invalid_checkpoint = Path(self.temp_dir) / "invalid"
        invalid_checkpoint.mkdir()
        
        # 设置基座模型
        self.merger._base_model = Mock()
        
        with pytest.raises(ModelMergeError) as exc_info:
            self.merger.load_lora_adapter(str(invalid_checkpoint))
        
        assert "LoRA适配器验证失败" in str(exc_info.value)
    
    @patch('src.model_merger.ensure_memory_available')
    @patch('src.model_merger.gc.collect')
    @patch('src.model_merger.torch.cuda.empty_cache')
    def test_merge_lora_weights_success(self, mock_empty_cache, mock_gc_collect, mock_ensure_memory):
        """测试成功合并LoRA权重"""
        # 设置mock PEFT模型
        mock_merged_model = Mock()
        mock_peft_model = Mock()
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        
        self.merger._peft_model = mock_peft_model
        
        # 执行测试
        result = self.merger.merge_lora_weights()
        
        # 验证结果
        assert result == mock_merged_model
        assert self.merger._merged_model == mock_merged_model
        assert self.merger._peft_model is None  # 应该被清理
        mock_peft_model.merge_and_unload.assert_called_once()
        mock_ensure_memory.assert_called_once()
        mock_gc_collect.assert_called_once()
    
    def test_merge_lora_weights_no_peft_model(self):
        """测试在没有PEFT模型时合并权重"""
        with pytest.raises(ModelMergeError) as exc_info:
            self.merger.merge_lora_weights()
        
        assert "PEFT模型未加载" in str(exc_info.value)
    
    @patch('src.model_merger.get_directory_size_mb')
    def test_save_merged_model_success(self, mock_get_size):
        """测试成功保存合并模型"""
        mock_get_size.return_value = 1024.0
        
        # 创建mock模型和tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        output_path = Path(self.temp_dir) / "output"
        
        # 执行测试
        self.merger.save_merged_model(mock_model, str(output_path), 
                                    save_tokenizer=True, tokenizer=mock_tokenizer)
        
        # 验证调用
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()
        assert output_path.exists()
    
    def test_save_merged_model_no_tokenizer(self):
        """测试在没有tokenizer时保存模型"""
        mock_model = Mock()
        output_path = Path(self.temp_dir) / "output"
        
        # 执行测试（不应该抛出异常）
        self.merger.save_merged_model(mock_model, str(output_path), save_tokenizer=True)
        
        # 验证模型保存被调用
        mock_model.save_pretrained.assert_called_once()
    
    @patch('src.model_merger.AutoTokenizer')
    def test_load_tokenizer_success(self, mock_auto_tokenizer):
        """测试成功加载tokenizer"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # 执行测试
        result = self.merger.load_tokenizer("test_model")
        
        # 验证结果
        assert result == mock_tokenizer
        assert self.merger._tokenizer == mock_tokenizer
        assert mock_tokenizer.pad_token == "<eos>"  # 应该设置pad_token
        mock_auto_tokenizer.from_pretrained.assert_called_once()
    
    @patch('src.model_merger.AutoTokenizer')
    def test_load_tokenizer_failure(self, mock_auto_tokenizer):
        """测试tokenizer加载失败"""
        mock_auto_tokenizer.from_pretrained.side_effect = Exception("Tokenizer not found")
        
        with pytest.raises(ModelMergeError) as exc_info:
            self.merger.load_tokenizer("nonexistent_model")
        
        assert "加载tokenizer失败" in str(exc_info.value)
    
    def test_verify_merge_integrity_no_tokenizer(self):
        """测试在没有tokenizer时验证合并完整性"""
        mock_model = Mock()
        
        # 执行测试
        result = self.merger.verify_merge_integrity(mock_model)
        
        # 没有tokenizer时应该返回True（跳过功能性验证）
        assert result is True
    
    @patch('src.model_merger.torch.no_grad')
    def test_verify_merge_integrity_with_tokenizer(self, mock_no_grad):
        """测试有tokenizer时验证合并完整性"""
        # 设置mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "Hello, how are you? I'm fine."
        self.merger._tokenizer = mock_tokenizer
        
        # 设置mock模型
        mock_model = Mock()
        mock_model.training = False
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.parameters.return_value = [torch.tensor([1.0, 2.0])]
        
        # 执行测试
        result = self.merger.verify_merge_integrity(mock_model)
        
        # 验证结果
        assert result is True
        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once()
    
    def test_cleanup(self):
        """测试内存清理"""
        # 设置一些mock对象
        self.merger._base_model = Mock()
        self.merger._peft_model = Mock()
        self.merger._merged_model = Mock()
        self.merger._tokenizer = Mock()
        
        # 执行清理
        self.merger.cleanup()
        
        # 验证所有引用都被清理
        assert self.merger._base_model is None
        assert self.merger._peft_model is None
        assert self.merger._merged_model is None
        assert self.merger._tokenizer is None
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with ModelMerger() as merger:
            merger._base_model = Mock()
            assert merger._base_model is not None
        
        # 退出上下文后应该被清理
        assert merger._base_model is None
    
    def test_estimate_model_memory(self):
        """测试模型内存估算"""
        # 这个方法主要是内部使用，测试基本功能
        memory_gb = self.merger._estimate_model_memory("dummy_model")
        
        # 应该返回一个合理的值
        assert isinstance(memory_gb, float)
        assert memory_gb >= 2.0  # 最少2GB
    
    def test_estimate_merge_memory(self):
        """测试合并内存估算"""
        memory_gb = self.merger._estimate_merge_memory()
        
        # 应该返回一个合理的值
        assert isinstance(memory_gb, float)
        assert memory_gb > 0
    
    def test_count_parameters(self):
        """测试参数计数"""
        # 创建一个简单的mock模型
        mock_model = Mock()
        mock_params = [
            torch.tensor([1.0, 2.0]),  # 2 parameters
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 4 parameters
        ]
        mock_model.parameters.return_value = mock_params
        
        count = self.merger._count_parameters(mock_model)
        assert count == 6  # 2 + 4 = 6
    
    @patch.object(ModelMerger, 'load_tokenizer')
    @patch.object(ModelMerger, 'load_base_model')
    @patch.object(ModelMerger, 'load_lora_adapter')
    @patch.object(ModelMerger, 'merge_lora_weights')
    @patch.object(ModelMerger, 'verify_merge_integrity')
    @patch.object(ModelMerger, 'save_merged_model')
    @patch('src.model_merger.get_directory_size_mb')
    def test_merge_and_save_success(self, mock_get_size, mock_save, mock_verify, 
                                   mock_merge, mock_load_adapter, mock_load_base, mock_load_tokenizer):
        """测试完整的合并和保存流程"""
        # 设置mock返回值
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.tensor([1.0, 2.0])]
        
        mock_load_base.return_value = mock_model
        mock_merge.return_value = mock_model
        mock_verify.return_value = True
        mock_get_size.return_value = 1024.0
        
        # 执行测试
        result = self.merger.merge_and_save(
            base_model_name="test_model",
            adapter_path="test_adapter",
            output_path=str(Path(self.temp_dir) / "output")
        )
        
        # 验证结果
        assert result['success'] is True
        assert result['verification_passed'] is True
        assert result['model_size_mb'] == 1024.0
        assert result['parameter_count'] == 2
        
        # 验证所有步骤都被调用
        mock_load_tokenizer.assert_called_once()
        mock_load_base.assert_called_once()
        mock_load_adapter.assert_called_once()
        mock_merge.assert_called_once()
        mock_verify.assert_called_once()
        mock_save.assert_called_once()
    
    @patch.object(ModelMerger, 'load_tokenizer')
    def test_merge_and_save_failure(self, mock_load_tokenizer):
        """测试合并和保存流程失败"""
        # 设置mock抛出异常
        mock_load_tokenizer.side_effect = Exception("Test error")
        
        # 执行测试并验证异常
        with pytest.raises(Exception):
            self.merger.merge_and_save(
                base_model_name="test_model",
                adapter_path="test_adapter",
                output_path=str(Path(self.temp_dir) / "output")
            )


if __name__ == "__main__":
    pytest.main([__file__])
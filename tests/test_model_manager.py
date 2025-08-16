"""
ModelManager单元测试模块

测试ModelManager类的模型加载、量化配置和分词器设置功能。
"""

import pytest
import torch
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model_manager import ModelManager


class TestModelManager(unittest.TestCase):
    """ModelManager单元测试类"""
    
    def setUp(self):
        """测试前设置"""
        # Mock CUDA availability
        self.cuda_available_patcher = patch('torch.cuda.is_available')
        self.mock_cuda_available = self.cuda_available_patcher.start()
        self.mock_cuda_available.return_value = True
        
        # Create ModelManager instance
        self.model_manager = ModelManager(max_memory_gb=13.0)
    
    def tearDown(self):
        """测试后清理"""
        self.cuda_available_patcher.stop()
    
    def test_initialization(self):
        """测试ModelManager初始化"""
        self.assertEqual(self.model_manager.max_memory_gb, 13.0)
        self.assertEqual(self.model_manager.device, "cuda")
    
    def test_initialization_without_cuda(self):
        """测试在没有CUDA的情况下初始化"""
        self.mock_cuda_available.return_value = False
        
        model_manager = ModelManager()
        self.assertEqual(model_manager.device, "cpu")
    
    def test_configure_quantization(self):
        """测试量化配置"""
        with patch('src.model_manager.BitsAndBytesConfig') as mock_config:
            mock_instance = Mock()
            mock_config.return_value = mock_instance
            
            result = self.model_manager.configure_quantization()
            
            # 验证BitsAndBytesConfig被正确调用
            mock_config.assert_called_once_with(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.assertEqual(result, mock_instance)
    
    def test_configure_device_map_cuda(self):
        """测试CUDA设备映射配置"""
        device_map = self.model_manager._configure_device_map()
        self.assertEqual(device_map, "auto")
    
    def test_configure_device_map_cpu(self):
        """测试CPU设备映射配置"""
        self.model_manager.device = "cpu"
        device_map = self.model_manager._configure_device_map()
        self.assertEqual(device_map, {"": "cpu"})
    
    @patch('src.model_manager.AutoTokenizer')
    @patch('src.model_manager.AutoModelForCausalLM')
    def test_load_model_with_quantization_success(self, mock_model_class, mock_tokenizer_class):
        """测试成功加载带量化的模型"""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.device = "cuda:0"
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.__len__ = Mock(return_value=50000)
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock quantization config
        with patch.object(self.model_manager, 'configure_quantization') as mock_config:
            mock_quantization_config = Mock()
            mock_config.return_value = mock_quantization_config
            
            model, tokenizer = self.model_manager.load_model_with_quantization("test-model")
            
            # 验证模型加载参数
            mock_model_class.from_pretrained.assert_called_once_with(
                "test-model",
                quantization_config=mock_quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=None
            )
            
            # 验证分词器加载
            mock_tokenizer_class.from_pretrained.assert_called_once_with(
                "test-model",
                trust_remote_code=True,
                padding_side="right"
            )
            
            # 验证pad_token设置
            self.assertEqual(mock_tokenizer.pad_token, "</s>")
            
            self.assertEqual(model, mock_model)
            self.assertEqual(tokenizer, mock_tokenizer)
    
    @patch('src.model_manager.AutoTokenizer')
    @patch('src.model_manager.AutoModelForCausalLM')
    def test_load_model_with_quantization_failure(self, mock_model_class, mock_tokenizer_class):
        """测试模型加载失败"""
        # Mock model loading failure
        mock_model_class.from_pretrained.side_effect = Exception("Model loading failed")
        
        with self.assertRaises(RuntimeError) as context:
            self.model_manager.load_model_with_quantization("test-model")
        
        self.assertIn("无法加载模型 test-model", str(context.exception))
        self.assertIn("Model loading failed", str(context.exception))
    
    def test_setup_tokenizer_with_eos_token(self):
        """测试使用EOS token设置分词器"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.__len__ = Mock(return_value=50000)
        
        with patch('src.model_manager.AutoTokenizer') as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            result = self.model_manager._setup_tokenizer("test-model")
            
            # 验证pad_token被设置为eos_token
            self.assertEqual(mock_tokenizer.pad_token, "</s>")
            self.assertEqual(result, mock_tokenizer)
    
    def test_setup_tokenizer_without_eos_token(self):
        """测试在没有EOS token的情况下设置分词器"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = None
        mock_tokenizer.__len__ = Mock(return_value=50000)
        
        with patch('src.model_manager.AutoTokenizer') as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            result = self.model_manager._setup_tokenizer("test-model")
            
            # 验证添加了新的pad_token
            mock_tokenizer.add_special_tokens.assert_called_once_with({"pad_token": "<pad>"})
            self.assertEqual(result, mock_tokenizer)
    
    def test_setup_tokenizer_with_existing_pad_token(self):
        """测试已有pad_token的分词器设置"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.__len__ = Mock(return_value=50000)
        
        with patch('src.model_manager.AutoTokenizer') as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            result = self.model_manager._setup_tokenizer("test-model")
            
            # 验证没有修改pad_token
            mock_tokenizer.add_special_tokens.assert_not_called()
            self.assertEqual(result, mock_tokenizer)
    
    def test_prepare_for_training(self):
        """测试为训练准备模型"""
        mock_model = Mock()
        
        result = self.model_manager.prepare_for_training(mock_model)
        
        # 验证启用了梯度检查点
        mock_model.gradient_checkpointing_enable.assert_called_once()
        
        # 验证设置为训练模式
        mock_model.train.assert_called_once()
        
        self.assertEqual(result, mock_model)
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        # 创建mock模型参数
        mock_param1 = Mock()
        mock_param1.numel.return_value = 1000
        mock_param1.requires_grad = True
        
        mock_param2 = Mock()
        mock_param2.numel.return_value = 2000
        mock_param2.requires_grad = False
        
        mock_param3 = Mock()
        mock_param3.numel.return_value = 3000
        mock_param3.requires_grad = True
        
        mock_model = Mock()
        mock_model.parameters.return_value = [mock_param1, mock_param2, mock_param3]
        mock_model.device = "cuda:0"
        mock_model.dtype = torch.float16
        
        info = self.model_manager.get_model_info(mock_model)
        
        expected_info = {
            "total_parameters": 6000,  # 1000 + 2000 + 3000
            "trainable_parameters": 4000,  # 1000 + 3000
            "trainable_percentage": (4000 / 6000) * 100,  # 66.67%
            "device": "cuda:0",
            "dtype": "torch.float16"
        }
        
        self.assertEqual(info["total_parameters"], expected_info["total_parameters"])
        self.assertEqual(info["trainable_parameters"], expected_info["trainable_parameters"])
        self.assertAlmostEqual(info["trainable_percentage"], expected_info["trainable_percentage"], places=2)
        self.assertEqual(info["device"], expected_info["device"])
        self.assertEqual(info["dtype"], expected_info["dtype"])
    
    def test_get_model_info_no_parameters(self):
        """测试获取没有参数的模型信息"""
        mock_model = Mock()
        mock_model.parameters.return_value = []
        mock_model.device = "cpu"
        
        # Mock hasattr to return False for dtype
        with patch('builtins.hasattr', return_value=False):
            info = self.model_manager.get_model_info(mock_model)
        
        expected_info = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "trainable_percentage": 0,
            "device": "cpu",
            "dtype": "unknown"
        }
        
        self.assertEqual(info, expected_info)


class TestModelManagerIntegration(unittest.TestCase):
    """ModelManager集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.model_manager = ModelManager()
    
    @patch('src.model_manager.AutoTokenizer')
    @patch('src.model_manager.AutoModelForCausalLM')
    def test_full_model_loading_workflow(self, mock_model_class, mock_tokenizer_class):
        """测试完整的模型加载工作流"""
        # Mock model
        mock_model = Mock()
        mock_model.device = "cuda:0"
        mock_model.dtype = torch.bfloat16
        
        # Mock model parameters for get_model_info
        mock_param = Mock()
        mock_param.numel.return_value = 1000000
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.__len__ = Mock(return_value=50000)
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # 执行完整工作流
        model, tokenizer = self.model_manager.load_model_with_quantization("test-model")
        prepared_model = self.model_manager.prepare_for_training(model)
        model_info = self.model_manager.get_model_info(prepared_model)
        
        # 验证结果
        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, mock_tokenizer)
        self.assertEqual(prepared_model, mock_model)
        self.assertIn("total_parameters", model_info)
        self.assertIn("trainable_parameters", model_info)
        
        # 验证调用顺序
        mock_model.gradient_checkpointing_enable.assert_called_once()
        mock_model.train.assert_called_once()
        self.assertEqual(mock_tokenizer.pad_token, "</s>")
    
    def test_device_selection_logic(self):
        """测试设备选择逻辑"""
        with patch('torch.cuda.is_available', return_value=True):
            manager_cuda = ModelManager()
            self.assertEqual(manager_cuda.device, "cuda")
        
        with patch('torch.cuda.is_available', return_value=False):
            manager_cpu = ModelManager()
            self.assertEqual(manager_cpu.device, "cpu")
    
    def test_quantization_config_consistency(self):
        """测试量化配置的一致性"""
        config1 = self.model_manager.configure_quantization()
        config2 = self.model_manager.configure_quantization()
        
        # 两次调用应该产生相同的配置
        self.assertEqual(type(config1), type(config2))


if __name__ == "__main__":
    # 设置日志级别以减少测试输出
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    unittest.main()
"""
推理测试器的单元测试
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile
import os

from src.inference_tester import InferenceTester, InferenceResult
from src.memory_optimizer import MemoryOptimizer


class TestInferenceTester:
    """推理测试器测试类"""
    
    @pytest.fixture
    def mock_memory_optimizer(self):
        """模拟内存优化器"""
        mock_optimizer = Mock(spec=MemoryOptimizer)
        mock_optimizer.cleanup_gpu_memory.return_value = None
        mock_optimizer.check_memory_safety.return_value = True
        mock_optimizer.log_memory_status.return_value = None
        return mock_optimizer
    
    @pytest.fixture
    def inference_tester(self, mock_memory_optimizer):
        """创建推理测试器实例"""
        return InferenceTester(memory_optimizer=mock_memory_optimizer)
    
    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """模拟模型和分词器"""
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "测试响应"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        return mock_model, mock_tokenizer
    
    def test_init(self, mock_memory_optimizer):
        """测试初始化"""
        tester = InferenceTester(memory_optimizer=mock_memory_optimizer)
        
        assert tester.memory_optimizer == mock_memory_optimizer
        assert tester.device in ["cuda", "cpu"]
        assert tester.model is None
        assert tester.tokenizer is None
    
    def test_format_prompt_for_qwen(self, inference_tester):
        """测试Qwen提示格式化"""
        prompt = "你好"
        system_message = "你是助手"
        
        formatted = inference_tester._format_prompt_for_qwen(prompt, system_message)
        
        assert "<|im_start|>system" in formatted
        assert system_message in formatted
        assert "<|im_start|>user" in formatted
        assert prompt in formatted
        assert "<|im_start|>assistant" in formatted
    
    def test_format_prompt_for_qwen_default_system(self, inference_tester):
        """测试默认系统消息的提示格式化"""
        prompt = "你好"
        
        formatted = inference_tester._format_prompt_for_qwen(prompt)
        
        assert "你是一个有用的AI助手" in formatted
        assert prompt in formatted
    
    @patch('src.inference_tester.AutoModelForCausalLM')
    @patch('src.inference_tester.AutoTokenizer')
    @patch('src.inference_tester.PeftModel')
    def test_load_finetuned_model_success(self, mock_peft, mock_tokenizer_class, 
                                        mock_model_class, inference_tester):
        """测试成功加载微调模型"""
        # 创建临时目录和配置文件
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            adapter_config_path = model_path / "adapter_config.json"
            
            # 创建adapter配置文件
            adapter_config = {"base_model_name_or_path": "test-model"}
            with open(adapter_config_path, 'w') as f:
                json.dump(adapter_config, f)
            
            # 设置mock返回值
            mock_base_model = Mock()
            mock_model_class.from_pretrained.return_value = mock_base_model
            
            mock_peft_model = Mock()
            mock_peft_model.merge_and_unload.return_value = mock_peft_model
            mock_peft.from_pretrained.return_value = mock_peft_model
            
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            # 执行测试
            model, tokenizer = inference_tester.load_finetuned_model(str(model_path))
            
            # 验证结果
            assert model == mock_peft_model
            assert tokenizer == mock_tokenizer
            assert inference_tester.model == mock_peft_model
            assert inference_tester.tokenizer == mock_tokenizer
            
            # 验证调用
            mock_model_class.from_pretrained.assert_called_once()
            mock_peft.from_pretrained.assert_called_once()
            mock_tokenizer_class.from_pretrained.assert_called_once()
    
    def test_load_finetuned_model_missing_path(self, inference_tester):
        """测试加载不存在的模型路径"""
        with pytest.raises(RuntimeError, match="无法加载微调模型"):
            inference_tester.load_finetuned_model("/nonexistent/path")
    
    @patch('torch.cuda.is_available', return_value=False)  # 使用CPU测试
    def test_test_inference_success(self, mock_cuda, inference_tester, mock_model_and_tokenizer):
        """测试成功的推理"""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        inference_tester.model = mock_model
        inference_tester.tokenizer = mock_tokenizer
        inference_tester.device = "cpu"  # 使用CPU
        
        # 设置tokenizer返回值
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        # 设置decode返回值
        formatted_prompt = "<|im_start|>system\n你是一个有用的AI助手。<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        full_response = formatted_prompt + "你好！我是AI助手。"
        mock_tokenizer.decode.return_value = full_response
        
        response = inference_tester.test_inference("你好")
        
        assert response == "你好！我是AI助手。"
        mock_model.generate.assert_called_once()
    
    def test_test_inference_no_model(self, inference_tester):
        """测试没有加载模型时的推理"""
        with pytest.raises(ValueError, match="模型或分词器未加载"):
            inference_tester.test_inference("你好")
    
    def test_validate_response_quality_good_response(self, inference_tester):
        """测试高质量响应的验证"""
        response = "这是一个很好的回答。它包含了有用的信息，结构清晰，没有重复内容。"
        
        quality = inference_tester.validate_response_quality(response)
        
        assert quality["has_content"] is True
        assert quality["min_length_met"] is True
        assert quality["max_length_reasonable"] is True
        assert quality["no_repetition"] is True
        assert quality["coherent_structure"] is True
        assert quality["appropriate_language"] is True
        assert quality["no_errors"] is True
        assert quality["overall_score"] > 0.8
    
    def test_validate_response_quality_poor_response(self, inference_tester):
        """测试低质量响应的验证"""
        response = "错误 错误 错误 错误 错误"  # 5个连续重复的词
        
        quality = inference_tester.validate_response_quality(response)
        
        assert quality["no_repetition"] is False
        # 由于重复问题，总分应该较低（缺少20%的重复分数）
        assert quality["overall_score"] <= 0.8
    
    def test_validate_response_quality_empty_response(self, inference_tester):
        """测试空响应的验证"""
        response = ""
        
        quality = inference_tester.validate_response_quality(response)
        
        assert quality["has_content"] is False
        assert quality["min_length_met"] is False
        assert quality["overall_score"] == 0.0
    
    def test_check_repetition_normal(self, inference_tester):
        """测试正常文本的重复检查"""
        response = "这是一个正常的回答，没有重复的内容。"
        
        result = inference_tester._check_repetition(response)
        
        assert result is True
    
    def test_check_repetition_excessive(self, inference_tester):
        """测试过度重复的文本"""
        response = "重复 重复 重复 重复 重复"  # 5个连续重复的词
        
        result = inference_tester._check_repetition(response)
        
        assert result is False
    
    def test_check_coherence_coherent(self, inference_tester):
        """测试连贯的文本"""
        response = "这是第一句话。这是第二句话。这是第三句话。"
        
        result = inference_tester._check_coherence(response)
        
        assert result is True
    
    def test_check_language_appropriateness_appropriate(self, inference_tester):
        """测试适当的语言"""
        response = "这是一个正常的回答。"
        
        result = inference_tester._check_language_appropriateness(response)
        
        assert result is True
    
    def test_check_language_appropriateness_inappropriate(self, inference_tester):
        """测试不适当的语言"""
        response = "抱歉，我不能回答这个问题。"
        
        result = inference_tester._check_language_appropriateness(response)
        
        assert result is False
    
    def test_check_for_errors_no_errors(self, inference_tester):
        """测试没有错误的文本"""
        response = "这是一个正常的回答。"
        
        result = inference_tester._check_for_errors(response)
        
        assert result is True
    
    def test_check_for_errors_with_errors(self, inference_tester):
        """测试包含错误的文本"""
        response = "发生了一个 error: 在处理过程中。"  # 使用小写的error:
        
        result = inference_tester._check_for_errors(response)
        
        assert result is False
    
    def test_calculate_quality_score_perfect(self, inference_tester):
        """测试完美质量分数计算"""
        metrics = {
            "has_content": True,
            "min_length_met": True,
            "max_length_reasonable": True,
            "no_repetition": True,
            "coherent_structure": True,
            "appropriate_language": True,
            "no_errors": True
        }
        
        score = inference_tester._calculate_quality_score(metrics)
        
        assert score == 1.0
    
    def test_calculate_quality_score_poor(self, inference_tester):
        """测试低质量分数计算"""
        metrics = {
            "has_content": False,
            "min_length_met": False,
            "max_length_reasonable": False,
            "no_repetition": False,
            "coherent_structure": False,
            "appropriate_language": False,
            "no_errors": False
        }
        
        score = inference_tester._calculate_quality_score(metrics)
        
        assert score == 0.0
    
    def test_generate_with_optimized_params_success(self, inference_tester, mock_model_and_tokenizer):
        """测试使用优化参数生成"""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        inference_tester.model = mock_model
        inference_tester.tokenizer = mock_tokenizer
        
        # Mock test_inference方法
        with patch.object(inference_tester, 'test_inference', return_value="测试响应"):
            result = inference_tester.generate_with_optimized_params("测试提示")
        
        assert isinstance(result, InferenceResult)
        assert result.success is True
        assert result.response == "测试响应"
        assert result.prompt == "测试提示"
    
    def test_generate_with_optimized_params_failure(self, inference_tester):
        """测试生成失败的情况"""
        # Mock test_inference方法抛出异常
        with patch.object(inference_tester, 'test_inference', side_effect=Exception("测试错误")):
            result = inference_tester.generate_with_optimized_params("测试提示")
        
        assert isinstance(result, InferenceResult)
        assert result.success is False
        assert result.error_message == "测试错误"
        assert result.response == ""
    
    def test_handle_inference_failure_memory_error(self, inference_tester):
        """测试处理内存错误"""
        error = RuntimeError("CUDA out of memory")
        
        result = inference_tester.handle_inference_failure(error, "测试提示")
        
        assert result["error_type"] == "RuntimeError"
        assert "out of memory" in result["error_message"]
        assert "减少max_new_tokens参数" in result["suggestions"]
        assert result["recovery_attempted"] is True
    
    def test_handle_inference_failure_model_error(self, inference_tester):
        """测试处理模型错误"""
        error = RuntimeError("Model loading failed")
        
        result = inference_tester.handle_inference_failure(error, "测试提示")
        
        assert result["error_type"] == "RuntimeError"
        assert "检查模型路径是否正确" in result["suggestions"]
    
    def test_cleanup(self, inference_tester, mock_model_and_tokenizer):
        """测试资源清理"""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        inference_tester.model = mock_model
        inference_tester.tokenizer = mock_tokenizer
        
        inference_tester.cleanup()
        
        assert inference_tester.model is None
        assert inference_tester.tokenizer is None
        inference_tester.memory_optimizer.cleanup_gpu_memory.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
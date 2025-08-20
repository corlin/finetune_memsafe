"""
OpenAI兼容适配器单元测试
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import requests
import json
from industry_evaluation.adapters.model_adapter import (
    OpenAICompatibleAdapter,
    ModelAdapterFactory,
    ModelException,
    ModelAuthenticationException,
    ModelRateLimitException,
    ModelConnectionException,
    ModelResponseException
)


class TestOpenAICompatibleAdapter(unittest.TestCase):
    """OpenAI兼容适配器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            "api_key": "test_api_key",
            "model_name": "test-model",
            "base_url": "https://api.test.com/v1",
            "provider": "test_provider",
            "timeout": 30
        }
        self.adapter = OpenAICompatibleAdapter("test_model", self.config)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.adapter.model_id, "test_model")
        self.assertEqual(self.adapter.api_key, "test_api_key")
        self.assertEqual(self.adapter.model_name, "test-model")
        self.assertEqual(self.adapter.base_url, "https://api.test.com/v1")
        self.assertEqual(self.adapter.provider, "test_provider")
    
    def test_build_headers_bearer_auth(self):
        """测试Bearer认证头构建"""
        headers = self.adapter._build_headers()
        
        expected_headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test_api_key"
        }
        
        self.assertEqual(headers, expected_headers)
    
    def test_build_headers_api_key_auth(self):
        """测试API Key认证头构建"""
        self.adapter.auth_type = "api_key"
        headers = self.adapter._build_headers()
        
        expected_headers = {
            "Content-Type": "application/json",
            "X-API-Key": "test_api_key"
        }
        
        self.assertEqual(headers, expected_headers)
    
    def test_build_headers_custom_auth(self):
        """测试自定义认证头构建"""
        self.adapter.auth_type = "custom"
        self.adapter.config["auth_header"] = "X-Custom-Auth"
        self.adapter.config["auth_value"] = "Custom test_api_key"
        
        headers = self.adapter._build_headers()
        
        expected_headers = {
            "Content-Type": "application/json",
            "X-Custom-Auth": "Custom test_api_key"
        }
        
        self.assertEqual(headers, expected_headers)
    
    def test_build_headers_with_custom_headers(self):
        """测试自定义请求头"""
        self.adapter.custom_headers = {
            "User-Agent": "Test-Agent/1.0",
            "X-Custom": "value"
        }
        
        headers = self.adapter._build_headers()
        
        self.assertIn("User-Agent", headers)
        self.assertIn("X-Custom", headers)
        self.assertEqual(headers["User-Agent"], "Test-Agent/1.0")
        self.assertEqual(headers["X-Custom"], "value")
    
    def test_build_messages_simple(self):
        """测试简单消息构建"""
        messages = self.adapter._build_messages("Hello", None)
        
        expected_messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        self.assertEqual(messages, expected_messages)
    
    def test_build_messages_with_system_prompt(self):
        """测试带系统提示的消息构建"""
        context = {"system_prompt": "You are a helpful assistant."}
        messages = self.adapter._build_messages("Hello", context)
        
        expected_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        
        self.assertEqual(messages, expected_messages)
    
    def test_build_request_data_default(self):
        """测试默认请求数据构建"""
        messages = [{"role": "user", "content": "Hello"}]
        data = self.adapter._build_request_data(messages, None)
        
        expected_data = {
            "model": "test-model",
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        self.assertEqual(data, expected_data)
    
    def test_build_request_data_with_context(self):
        """测试带上下文的请求数据构建"""
        messages = [{"role": "user", "content": "Hello"}]
        context = {
            "max_tokens": 500,
            "temperature": 0.5,
            "top_p": 0.9
        }
        data = self.adapter._build_request_data(messages, context)
        
        expected_data = {
            "model": "test-model",
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.5,
            "top_p": 0.9
        }
        
        self.assertEqual(data, expected_data)
    
    def test_build_request_data_with_param_mapping(self):
        """测试参数映射"""
        self.adapter.param_mapping = {
            "max_tokens": "max_length",
            "temperature": "temp"
        }
        
        messages = [{"role": "user", "content": "Hello"}]
        data = self.adapter._build_request_data(messages, None)
        
        expected_data = {
            "model": "test-model",
            "messages": messages,
            "max_length": 1000,
            "temp": 0.7
        }
        
        self.assertEqual(data, expected_data)
    
    def test_parse_response_standard_format(self):
        """测试标准格式响应解析"""
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you?"
                    }
                }
            ]
        }
        
        result = self.adapter._parse_response(response_data)
        self.assertEqual(result, "Hello, how can I help you?")
    
    def test_parse_response_custom_path(self):
        """测试自定义路径响应解析"""
        self.adapter.response_path = ["data", "text"]
        
        response_data = {
            "data": {
                "text": "Custom response format"
            }
        }
        
        result = self.adapter._parse_response(response_data)
        self.assertEqual(result, "Custom response format")
    
    def test_parse_response_with_array_index(self):
        """测试数组索引响应解析"""
        self.adapter.response_path = ["results", 0, "content"]
        
        response_data = {
            "results": [
                {"content": "First result"},
                {"content": "Second result"}
            ]
        }
        
        result = self.adapter._parse_response(response_data)
        self.assertEqual(result, "First result")
    
    def test_parse_response_fallback(self):
        """测试响应解析回退机制"""
        # 设置一个无效的响应路径
        self.adapter.response_path = ["invalid", "path"]
        
        # 但提供标准格式的响应
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": "Fallback response"
                    }
                }
            ]
        }
        
        result = self.adapter._parse_response(response_data)
        self.assertEqual(result, "Fallback response")
    
    def test_parse_response_error(self):
        """测试响应解析错误"""
        response_data = {"invalid": "format"}
        
        with self.assertRaises(ModelResponseException):
            self.adapter._parse_response(response_data)
    
    @patch('requests.post')
    def test_make_prediction_success(self, mock_post):
        """测试成功的预测请求"""
        # 模拟成功响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        result = self.adapter._make_prediction("Hello", None)
        
        self.assertEqual(result, "Test response")
        
        # 验证请求参数
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        self.assertEqual(call_args[0][0], "https://api.test.com/v1/chat/completions")
        self.assertIn("headers", call_args[1])
        self.assertIn("json", call_args[1])
        self.assertEqual(call_args[1]["timeout"], 30)
    
    @patch('requests.post')
    def test_make_prediction_auth_error(self, mock_post):
        """测试认证错误"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response
        
        with self.assertRaises(ModelAuthenticationException):
            self.adapter._make_prediction("Hello", None)
    
    @patch('requests.post')
    def test_make_prediction_rate_limit_error(self, mock_post):
        """测试速率限制错误"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_post.return_value = mock_response
        
        with self.assertRaises(ModelRateLimitException):
            self.adapter._make_prediction("Hello", None)
    
    @patch('requests.post')
    def test_make_prediction_server_error(self, mock_post):
        """测试服务器错误"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        with self.assertRaises(ModelConnectionException):
            self.adapter._make_prediction("Hello", None)
    
    @patch('requests.post')
    def test_make_prediction_timeout(self, mock_post):
        """测试超时错误"""
        mock_post.side_effect = requests.exceptions.Timeout()
        
        with self.assertRaises(Exception):  # 会被包装成ModelException
            self.adapter._make_prediction("Hello", None)
    
    @patch('requests.post')
    def test_make_prediction_connection_error(self, mock_post):
        """测试连接错误"""
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        with self.assertRaises(Exception):  # 会被包装成ModelException
            self.adapter._make_prediction("Hello", None)


class TestModelAdapterFactory(unittest.TestCase):
    """模型适配器工厂测试"""
    
    def test_get_supported_types(self):
        """测试获取支持的适配器类型"""
        types = ModelAdapterFactory.get_supported_types()
        
        self.assertIn("openai_compatible", types)
        self.assertIn("openai", types)
        self.assertIn("local", types)
        self.assertIn("http", types)
    
    def test_get_supported_providers(self):
        """测试获取支持的提供商"""
        providers = ModelAdapterFactory.get_supported_providers()
        
        self.assertIn("openai", providers)
        self.assertIn("bigmodel", providers)
        self.assertIn("deepseek", providers)
        self.assertIn("moonshot", providers)
    
    def test_get_provider_config(self):
        """测试获取提供商配置"""
        config = ModelAdapterFactory.get_provider_config("bigmodel")
        
        self.assertIn("base_url", config)
        self.assertIn("auth_type", config)
        self.assertIn("chat_endpoint", config)
        self.assertIn("response_path", config)
        
        self.assertEqual(config["base_url"], "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(config["auth_type"], "bearer")
    
    def test_get_provider_config_invalid(self):
        """测试获取无效提供商配置"""
        with self.assertRaises(ValueError):
            ModelAdapterFactory.get_provider_config("invalid_provider")
    
    def test_create_openai_compatible_adapter(self):
        """测试创建OpenAI兼容适配器"""
        adapter = ModelAdapterFactory.create_openai_compatible_adapter(
            model_id="test_model",
            provider="bigmodel",
            api_key="test_key",
            model_name="glm-4.5"
        )
        
        self.assertIsInstance(adapter, OpenAICompatibleAdapter)
        self.assertEqual(adapter.model_id, "test_model")
        self.assertEqual(adapter.api_key, "test_key")
        self.assertEqual(adapter.model_name, "glm-4.5")
        self.assertEqual(adapter.provider, "bigmodel")
        self.assertEqual(adapter.base_url, "https://open.bigmodel.cn/api/paas/v4")
    
    def test_create_openai_compatible_adapter_invalid_provider(self):
        """测试创建无效提供商的适配器"""
        with self.assertRaises(ValueError):
            ModelAdapterFactory.create_openai_compatible_adapter(
                model_id="test_model",
                provider="invalid_provider",
                api_key="test_key",
                model_name="test-model"
            )
    
    def test_create_openai_compatible_adapter_with_kwargs(self):
        """测试使用额外参数创建适配器"""
        adapter = ModelAdapterFactory.create_openai_compatible_adapter(
            model_id="test_model",
            provider="bigmodel",
            api_key="test_key",
            model_name="glm-4.5",
            timeout=60,
            custom_headers={"X-Test": "value"}
        )
        
        self.assertEqual(adapter.timeout, 60)
        self.assertEqual(adapter.custom_headers, {"X-Test": "value"})


if __name__ == '__main__':
    unittest.main()
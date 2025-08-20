"""
测试增强的模型适配器异常处理和重试机制
"""

import pytest
import time
import requests
from unittest.mock import Mock, patch, MagicMock
from industry_evaluation.adapters.model_adapter import (
    BaseModelAdapter,
    OpenAIAdapter,
    LocalModelAdapter,
    HTTPModelAdapter,
    ModelAdapterFactory,
    ModelManager,
    ModelException,
    ModelTimeoutException,
    ModelConnectionException,
    ModelAuthenticationException,
    ModelRateLimitException,
    ModelResponseException,
    RetryStrategy,
    RetryConfig
)


class TestModelAdapter(BaseModelAdapter):
    """测试用的模型适配器"""
    
    def __init__(self, model_id: str, config: dict):
        super().__init__(model_id, config)
        self.call_count = 0
        self.should_fail = False
        self.failure_count = 0
    
    def _make_prediction(self, input_text: str, context=None) -> str:
        self.call_count += 1
        
        if self.should_fail and self.call_count <= self.failure_count:
            raise ModelConnectionException("模拟连接失败")
        
        return f"测试输出: {input_text}"


class TestRetryMechanism:
    """测试重试机制"""
    
    def test_successful_prediction_no_retry(self):
        """测试成功预测，无需重试"""
        config = {"retry_config": {"max_retries": 3}}
        adapter = TestModelAdapter("test_model", config)
        
        result = adapter.predict("测试输入")
        
        assert result == "测试输出: 测试输入"
        assert adapter.call_count == 1
    
    def test_retry_on_failure(self):
        """测试失败时的重试机制"""
        config = {"retry_config": {"max_retries": 3, "base_delay": 0.1}}
        adapter = TestModelAdapter("test_model", config)
        adapter.should_fail = True
        adapter.failure_count = 2  # 前两次失败，第三次成功
        
        result = adapter.predict("测试输入")
        
        assert result == "测试输出: 测试输入"
        assert adapter.call_count == 3
    
    def test_max_retries_exceeded(self):
        """测试超过最大重试次数"""
        config = {"retry_config": {"max_retries": 2, "base_delay": 0.1}}
        adapter = TestModelAdapter("test_model", config)
        adapter.should_fail = True
        adapter.failure_count = 5  # 一直失败
        
        with pytest.raises(ModelConnectionException):
            adapter.predict("测试输入")
        
        assert adapter.call_count == 3  # 初始调用 + 2次重试
    
    def test_exponential_backoff_delay(self):
        """测试指数退避延迟"""
        config = {
            "retry_config": {
                "strategy": "exponential_backoff",
                "base_delay": 0.1,
                "backoff_multiplier": 2.0,
                "jitter": False
            }
        }
        adapter = TestModelAdapter("test_model", config)
        
        # 测试延迟计算
        assert adapter._calculate_retry_delay(0) == 0.1
        assert adapter._calculate_retry_delay(1) == 0.2
        assert adapter._calculate_retry_delay(2) == 0.4
    
    def test_fixed_delay(self):
        """测试固定延迟"""
        config = {
            "retry_config": {
                "strategy": "fixed_delay",
                "base_delay": 0.5,
                "jitter": False
            }
        }
        adapter = TestModelAdapter("test_model", config)
        
        assert adapter._calculate_retry_delay(0) == 0.5
        assert adapter._calculate_retry_delay(1) == 0.5
        assert adapter._calculate_retry_delay(2) == 0.5
    
    def test_linear_backoff_delay(self):
        """测试线性退避延迟"""
        config = {
            "retry_config": {
                "strategy": "linear_backoff",
                "base_delay": 0.2,
                "jitter": False
            }
        }
        adapter = TestModelAdapter("test_model", config)
        
        assert adapter._calculate_retry_delay(0) == 0.2
        assert adapter._calculate_retry_delay(1) == 0.4
        assert adapter._calculate_retry_delay(2) == 0.6


class TestExceptionHandling:
    """测试异常处理"""
    
    def test_should_retry_retryable_exceptions(self):
        """测试可重试异常"""
        adapter = TestModelAdapter("test_model", {})
        
        # 可重试的异常
        assert adapter._should_retry(ModelTimeoutException("超时"), 0) == True
        assert adapter._should_retry(ModelConnectionException("连接失败"), 0) == True
        assert adapter._should_retry(ModelRateLimitException("频率限制"), 0) == True
        assert adapter._should_retry(requests.exceptions.Timeout(), 0) == True
        assert adapter._should_retry(requests.exceptions.ConnectionError(), 0) == True
    
    def test_should_not_retry_non_retryable_exceptions(self):
        """测试不可重试异常"""
        adapter = TestModelAdapter("test_model", {})
        
        # 不可重试的异常
        assert adapter._should_retry(ModelAuthenticationException("认证失败"), 0) == False
        assert adapter._should_retry(ValueError("参数错误"), 0) == False
        assert adapter._should_retry(TypeError("类型错误"), 0) == False
    
    def test_should_not_retry_max_attempts(self):
        """测试达到最大重试次数"""
        config = {"retry_config": {"max_retries": 2}}
        adapter = TestModelAdapter("test_model", config)
        
        assert adapter._should_retry(ModelConnectionException("连接失败"), 2) == False
        assert adapter._should_retry(ModelConnectionException("连接失败"), 3) == False


class TestFallbackMechanism:
    """测试降级机制"""
    
    def test_fallback_enabled(self):
        """测试启用降级机制"""
        config = {
            "fallback_enabled": True,
            "fallback_response": "降级响应",
            "retry_config": {"max_retries": 1, "base_delay": 0.1}
        }
        adapter = TestModelAdapter("test_model", config)
        adapter.should_fail = True
        adapter.failure_count = 5  # 一直失败
        
        result = adapter.predict("测试输入")
        
        assert result == "降级响应"
    
    def test_fallback_disabled(self):
        """测试禁用降级机制"""
        config = {
            "fallback_enabled": False,
            "retry_config": {"max_retries": 1, "base_delay": 0.1}
        }
        adapter = TestModelAdapter("test_model", config)
        adapter.should_fail = True
        adapter.failure_count = 5  # 一直失败
        
        with pytest.raises(ModelConnectionException):
            adapter.predict("测试输入")


class TestHealthCheck:
    """测试健康检查"""
    
    def test_health_check_success(self):
        """测试健康检查成功"""
        adapter = TestModelAdapter("test_model", {"health_check_interval": 1})
        
        assert adapter._check_health() == True
        assert adapter._is_healthy == True
    
    def test_health_check_failure(self):
        """测试健康检查失败"""
        adapter = TestModelAdapter("test_model", {"health_check_interval": 1})
        adapter.should_fail = True
        adapter.failure_count = 5
        
        assert adapter._check_health() == False
        assert adapter._is_healthy == False
    
    def test_health_check_caching(self):
        """测试健康检查缓存"""
        adapter = TestModelAdapter("test_model", {"health_check_interval": 10})
        
        # 第一次检查
        adapter._check_health()
        first_check_time = adapter._last_health_check
        
        # 立即再次检查，应该使用缓存
        adapter._check_health()
        assert adapter._last_health_check == first_check_time
    
    def test_get_health_status(self):
        """测试获取健康状态"""
        adapter = TestModelAdapter("test_model", {})
        
        status = adapter.get_health_status()
        
        assert "model_id" in status
        assert "is_healthy" in status
        assert "last_health_check" in status
        assert "health_check_interval" in status
        assert status["model_id"] == "test_model"


class TestOpenAIAdapter:
    """测试OpenAI适配器"""
    
    @patch('requests.post')
    def test_successful_prediction(self, mock_post):
        """测试成功的预测"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "测试响应"}}]
        }
        mock_post.return_value = mock_response
        
        config = {"api_key": "test_key", "model_name": "gpt-3.5-turbo"}
        adapter = OpenAIAdapter("openai_model", config)
        
        result = adapter._make_prediction("测试输入")
        
        assert result == "测试响应"
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_authentication_error(self, mock_post):
        """测试认证错误"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response
        
        config = {"api_key": "invalid_key"}
        adapter = OpenAIAdapter("openai_model", config)
        
        with pytest.raises(ModelAuthenticationException):
            adapter._make_prediction("测试输入")
    
    @patch('requests.post')
    def test_rate_limit_error(self, mock_post):
        """测试频率限制错误"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_post.return_value = mock_response
        
        config = {"api_key": "test_key"}
        adapter = OpenAIAdapter("openai_model", config)
        
        with pytest.raises(ModelRateLimitException):
            adapter._make_prediction("测试输入")
    
    @patch('requests.post')
    def test_timeout_error(self, mock_post):
        """测试超时错误"""
        mock_post.side_effect = requests.exceptions.Timeout()
        
        config = {"api_key": "test_key", "timeout": 5}
        adapter = OpenAIAdapter("openai_model", config)
        
        with pytest.raises(ModelTimeoutException):
            adapter._make_prediction("测试输入")
    
    @patch('requests.post')
    def test_invalid_response_format(self, mock_post):
        """测试无效响应格式"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "无效响应"}
        mock_post.return_value = mock_response
        
        config = {"api_key": "test_key"}
        adapter = OpenAIAdapter("openai_model", config)
        
        with pytest.raises(ModelResponseException):
            adapter._make_prediction("测试输入")


class TestLocalModelAdapter:
    """测试本地模型适配器"""
    
    def test_successful_prediction(self):
        """测试成功的预测"""
        config = {"model_path": "/path/to/model"}
        adapter = LocalModelAdapter("local_model", config)
        
        # 模拟模型已加载
        adapter._model = "loaded_model"
        
        result = adapter._make_prediction("测试输入")
        
        assert "本地模型local_model的输出" in result
        assert "测试输入" in result
    
    def test_empty_input_error(self):
        """测试空输入错误"""
        config = {"model_path": "/path/to/model"}
        adapter = LocalModelAdapter("local_model", config)
        adapter._model = "loaded_model"
        
        with pytest.raises(ValueError):
            adapter._make_prediction("")
        
        with pytest.raises(ValueError):
            adapter._make_prediction("   ")


class TestHTTPModelAdapter:
    """测试HTTP模型适配器"""
    
    @patch('requests.post')
    def test_successful_json_prediction(self, mock_post):
        """测试成功的JSON预测"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "HTTP响应"}
        mock_post.return_value = mock_response
        
        config = {
            "api_url": "http://test.com/predict",
            "request_format": "json"
        }
        adapter = HTTPModelAdapter("http_model", config)
        
        result = adapter._make_prediction("测试输入")
        
        assert result == "HTTP响应"
    
    @patch('requests.post')
    def test_missing_api_url(self, mock_post):
        """测试缺少API URL"""
        config = {"api_url": ""}
        adapter = HTTPModelAdapter("http_model", config)
        
        with pytest.raises(ModelException):
            adapter._make_prediction("测试输入")
    
    @patch('requests.post')
    def test_invalid_json_response(self, mock_post):
        """测试无效JSON响应"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response
        
        config = {"api_url": "http://test.com/predict"}
        adapter = HTTPModelAdapter("http_model", config)
        
        with pytest.raises(ModelResponseException):
            adapter._make_prediction("测试输入")


class TestModelManager:
    """测试模型管理器"""
    
    def test_register_and_get_model(self):
        """测试注册和获取模型"""
        manager = ModelManager()
        config = {"api_key": "test_key"}
        
        manager.register_model("test_model", "openai", config)
        
        adapter = manager.get_model("test_model")
        assert adapter is not None
        assert adapter.model_id == "test_model"
    
    def test_remove_model(self):
        """测试移除模型"""
        manager = ModelManager()
        config = {"api_key": "test_key"}
        
        manager.register_model("test_model", "openai", config)
        assert manager.get_model("test_model") is not None
        
        success = manager.remove_model("test_model")
        assert success == True
        assert manager.get_model("test_model") is None
    
    def test_list_models(self):
        """测试列出模型"""
        manager = ModelManager()
        config1 = {"api_key": "test_key1"}
        config2 = {"api_key": "test_key2"}
        
        manager.register_model("model1", "openai", config1)
        manager.register_model("model2", "openai", config2)
        
        models = manager.list_models()
        
        assert len(models) == 2
        model_ids = [model["model_id"] for model in models]
        assert "model1" in model_ids
        assert "model2" in model_ids
    
    @patch.object(TestModelAdapter, '_make_prediction')
    def test_test_model(self, mock_prediction):
        """测试模型测试功能"""
        mock_prediction.return_value = "测试响应"
        
        manager = ModelManager()
        # 注册测试适配器
        ModelAdapterFactory.register_adapter("test", TestModelAdapter)
        manager.register_model("test_model", "test", {})
        
        result = manager.test_model("test_model")
        
        assert result["available"] == True
        assert "response_time" in result
        assert "test_output" in result
    
    def test_test_nonexistent_model(self):
        """测试不存在的模型"""
        manager = ModelManager()
        
        result = manager.test_model("nonexistent_model")
        
        assert result["available"] == False
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__])
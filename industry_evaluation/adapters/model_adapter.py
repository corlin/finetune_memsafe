"""
通用模型接口适配器
"""

import time
import requests
import logging
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from industry_evaluation.core.interfaces import ModelAdapter


class ModelException(Exception):
    """模型相关异常基类"""
    pass


class ModelTimeoutException(ModelException):
    """模型超时异常"""
    pass


class ModelConnectionException(ModelException):
    """模型连接异常"""
    pass


class ModelAuthenticationException(ModelException):
    """模型认证异常"""
    pass


class ModelRateLimitException(ModelException):
    """模型速率限制异常"""
    pass


class ModelResponseException(ModelException):
    """模型响应异常"""
    pass


class RetryStrategy(Enum):
    """重试策略"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


class BaseModelAdapter(ModelAdapter):
    """基础模型适配器"""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        """
        初始化模型适配器
        
        Args:
            model_id: 模型ID
            config: 配置信息
        """
        self.model_id = model_id
        self.config = config
        self.timeout = config.get("timeout", 30)
        self._last_request_time = 0
        self._rate_limit_delay = config.get("rate_limit_delay", 0.1)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{model_id}")
        
        # 重试配置
        retry_config = config.get("retry_config", {})
        self.retry_config = RetryConfig(
            max_retries=retry_config.get("max_retries", 3),
            strategy=RetryStrategy(retry_config.get("strategy", "exponential_backoff")),
            base_delay=retry_config.get("base_delay", 1.0),
            max_delay=retry_config.get("max_delay", 60.0),
            backoff_multiplier=retry_config.get("backoff_multiplier", 2.0),
            jitter=retry_config.get("jitter", True)
        )
        
        # 降级配置
        self.fallback_enabled = config.get("fallback_enabled", False)
        self.fallback_response = config.get("fallback_response", "抱歉，模型暂时不可用")
        
        # 健康检查配置
        self.health_check_interval = config.get("health_check_interval", 300)  # 5分钟
        self._last_health_check = 0
        self._is_healthy = True
    
    def predict(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        模型预测
        
        Args:
            input_text: 输入文本
            context: 上下文信息
            
        Returns:
            str: 模型输出
        """
        # 健康检查
        if not self._check_health():
            if self.fallback_enabled:
                self.logger.warning(f"模型 {self.model_id} 不健康，使用降级响应")
                return self.fallback_response
            else:
                raise ModelConnectionException(f"模型 {self.model_id} 不可用")
        
        # 实现速率限制
        self._apply_rate_limit()
        
        # 重试机制
        last_exception = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                self.logger.debug(f"尝试调用模型 {self.model_id}，第 {attempt + 1} 次")
                result = self._make_prediction(input_text, context)
                
                # 成功后重置健康状态
                self._is_healthy = True
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"模型调用失败，第 {attempt + 1} 次尝试: {str(e)}")
                
                # 分类异常并决定是否重试
                if not self._should_retry(e, attempt):
                    break
                
                if attempt < self.retry_config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    self.logger.info(f"等待 {delay:.2f} 秒后重试")
                    time.sleep(delay)
        
        # 所有重试都失败了
        self._is_healthy = False
        
        if self.fallback_enabled:
            self.logger.error(f"模型 {self.model_id} 调用失败，使用降级响应: {str(last_exception)}")
            return self.fallback_response
        else:
            raise last_exception
    
    @abstractmethod
    def _make_prediction(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        执行实际的预测
        
        Args:
            input_text: 输入文本
            context: 上下文信息
            
        Returns:
            str: 模型输出
        """
        pass
    
    def is_available(self) -> bool:
        """
        检查模型是否可用
        
        Returns:
            bool: 是否可用
        """
        try:
            # 发送简单的测试请求
            test_result = self.predict("Hello", {"max_tokens": 5})
            # 更宽松的检查：只要不是None且不是空字符串就认为可用
            return test_result is not None and str(test_result).strip() != ""
        except Exception as e:
            self.logger.debug(f"模型可用性检查失败: {str(e)}")
            return False
    
    def _apply_rate_limit(self):
        """应用速率限制"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - time_since_last)
        
        self._last_request_time = time.time()
    
    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        判断是否应该重试
        
        Args:
            exception: 异常对象
            attempt: 当前尝试次数
            
        Returns:
            bool: 是否应该重试
        """
        if attempt >= self.retry_config.max_retries:
            return False
        
        # 不重试的异常类型
        non_retryable_exceptions = (
            ModelAuthenticationException,
            ValueError,
            TypeError
        )
        
        if isinstance(exception, non_retryable_exceptions):
            return False
        
        # 可重试的异常类型
        retryable_exceptions = (
            ModelTimeoutException,
            ModelConnectionException,
            ModelRateLimitException,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError
        )
        
        if isinstance(exception, retryable_exceptions):
            return True
        
        # HTTP状态码判断
        if isinstance(exception, requests.exceptions.HTTPError):
            status_code = exception.response.status_code if exception.response else 0
            # 5xx错误和429错误可以重试
            if status_code >= 500 or status_code == 429:
                return True
        
        # 默认不重试
        return False
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        计算重试延迟时间
        
        Args:
            attempt: 当前尝试次数
            
        Returns:
            float: 延迟时间（秒）
        """
        if self.retry_config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.retry_config.base_delay
        elif self.retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.retry_config.base_delay * (attempt + 1)
        else:  # EXPONENTIAL_BACKOFF
            delay = self.retry_config.base_delay * (self.retry_config.backoff_multiplier ** attempt)
        
        # 限制最大延迟
        delay = min(delay, self.retry_config.max_delay)
        
        # 添加抖动
        if self.retry_config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def _check_health(self) -> bool:
        """
        检查模型健康状态
        
        Returns:
            bool: 是否健康
        """
        current_time = time.time()
        
        # 如果距离上次检查时间不够，直接返回缓存的状态
        if current_time - self._last_health_check < self.health_check_interval:
            return self._is_healthy
        
        # 执行健康检查
        try:
            self._perform_health_check()
            self._is_healthy = True
            self.logger.debug(f"模型 {self.model_id} 健康检查通过")
        except Exception as e:
            self._is_healthy = False
            self.logger.warning(f"模型 {self.model_id} 健康检查失败: {str(e)}")
        
        self._last_health_check = current_time
        return self._is_healthy
    
    def _perform_health_check(self):
        """
        执行健康检查
        子类可以重写此方法实现特定的健康检查逻辑
        """
        # 默认实现：发送简单的测试请求
        test_input = "健康检查"
        self._make_prediction(test_input, {"max_tokens": 10})
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态信息
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        return {
            "model_id": self.model_id,
            "is_healthy": self._is_healthy,
            "last_health_check": self._last_health_check,
            "health_check_interval": self.health_check_interval
        }


class OpenAICompatibleAdapter(BaseModelAdapter):
    """OpenAI兼容API适配器 - 支持多种OpenAI兼容的API提供商"""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        """
        初始化OpenAI兼容适配器
        
        Args:
            model_id: 模型ID
            config: 配置信息，应包含api_key, base_url等
        """
        super().__init__(model_id, config)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        
        # 支持不同的认证方式
        self.auth_type = config.get("auth_type", "bearer")  # bearer, api_key, custom
        self.custom_headers = config.get("custom_headers", {})
        
        # API端点配置
        self.chat_endpoint = config.get("chat_endpoint", "/chat/completions")
        
        # 提供商特定配置
        self.provider = config.get("provider", "openai")  # openai, bigmodel, zhipu, etc.
        
        # 请求参数映射
        self.param_mapping = config.get("param_mapping", {})
        
        # 响应解析配置
        self.response_path = config.get("response_path", ["choices", 0, "message", "content"])
        self.fallback_response_paths = config.get("fallback_response_paths", [])
    
    def _make_prediction(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """执行OpenAI兼容API调用"""
        try:
            # 构建请求头
            headers = self._build_headers()
            
            # 构建消息
            messages = self._build_messages(input_text, context)
            
            # 构建请求数据
            data = self._build_request_data(messages, context)
            
            # 发送请求
            response = requests.post(
                f"{self.base_url.rstrip('/')}{self.chat_endpoint}",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            
            # 处理HTTP错误
            self._handle_http_errors(response)
            
            response.raise_for_status()
            result = response.json()
            
            # 解析响应
            return self._parse_response(result)
            
        except requests.exceptions.Timeout:
            raise ModelTimeoutException(f"API调用超时 ({self.timeout}秒)")
        except requests.exceptions.ConnectionError as e:
            raise ModelConnectionException(f"连接失败: {str(e)}")
        except requests.exceptions.HTTPError as e:
            raise ModelConnectionException(f"HTTP错误: {str(e)}")
        except Exception as e:
            if isinstance(e, (ModelException,)):
                raise
            raise ModelException(f"未知错误: {str(e)}")
    
    def _build_headers(self) -> Dict[str, str]:
        """构建请求头"""
        headers = {
            "Content-Type": "application/json"
        }
        
        # 添加认证头
        if self.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.auth_type == "api_key":
            headers["X-API-Key"] = self.api_key
        elif self.auth_type == "custom":
            # 自定义认证方式，从配置中获取
            auth_header = self.config.get("auth_header", "Authorization")
            auth_value = self.config.get("auth_value", f"Bearer {self.api_key}")
            headers[auth_header] = auth_value
        
        # 添加自定义头
        headers.update(self.custom_headers)
        
        return headers
    
    def _build_messages(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = []
        
        # 添加系统消息
        if context and context.get("system_prompt"):
            messages.append({"role": "system", "content": context["system_prompt"]})
        
        # 添加用户消息
        messages.append({"role": "user", "content": input_text})
        
        return messages
    
    def _build_request_data(self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """构建请求数据"""
        data = {
            "model": self.model_name,
            "messages": messages
        }
        
        # 添加可选参数
        if context:
            if "max_tokens" in context:
                data["max_tokens"] = context["max_tokens"]
            if "temperature" in context:
                data["temperature"] = context["temperature"]
            if "top_p" in context:
                data["top_p"] = context["top_p"]
            if "stream" in context:
                data["stream"] = context["stream"]
        else:
            # 默认参数
            data["max_tokens"] = 1000
            data["temperature"] = 0.7
        
        # 应用参数映射（用于不同提供商的参数名差异）
        if self.param_mapping:
            mapped_data = {}
            for key, value in data.items():
                mapped_key = self.param_mapping.get(key, key)
                mapped_data[mapped_key] = value
            data = mapped_data
        
        return data
    
    def _handle_http_errors(self, response: requests.Response):
        """处理HTTP错误"""
        if response.status_code == 401:
            raise ModelAuthenticationException("API密钥无效或已过期")
        elif response.status_code == 429:
            raise ModelRateLimitException("API调用频率超限")
        elif response.status_code >= 500:
            raise ModelConnectionException(f"服务器错误: {response.status_code}")
        elif response.status_code == 400:
            try:
                error_detail = response.json().get("error", {}).get("message", "请求参数错误")
            except:
                error_detail = "请求参数错误"
            raise ModelException(f"请求错误: {error_detail}")
    
    def _parse_response(self, result: Dict[str, Any]) -> str:
        """解析API响应"""
        try:
            # 首先尝试主要响应路径
            current = result
            for key in self.response_path:
                if isinstance(key, int):
                    if not isinstance(current, list) or len(current) <= key:
                        raise ModelResponseException(f"响应解析错误: 索引 {key} 超出范围")
                    current = current[key]
                else:
                    if not isinstance(current, dict) or key not in current:
                        raise ModelResponseException(f"响应解析错误: 缺少字段 {key}")
                    current = current[key]
            
            # 如果主路径的结果为空，尝试fallback路径
            if not current or (isinstance(current, str) and current.strip() == ""):
                fallback_paths = getattr(self, 'fallback_response_paths', [])
                for fallback_path in fallback_paths:
                    try:
                        fallback_current = result
                        for key in fallback_path:
                            if isinstance(key, int):
                                if not isinstance(fallback_current, list) or len(fallback_current) <= key:
                                    break
                                fallback_current = fallback_current[key]
                            else:
                                if not isinstance(fallback_current, dict) or key not in fallback_current:
                                    break
                                fallback_current = fallback_current[key]
                        else:
                            # 如果成功遍历完所有键，且结果不为空，使用这个结果
                            if fallback_current and (not isinstance(fallback_current, str) or fallback_current.strip()):
                                current = fallback_current
                                break
                    except:
                        continue
            
            if not isinstance(current, str):
                current = str(current) if current is not None else ""
            
            return current
            
        except ModelResponseException:
            raise
        except Exception as e:
            # 尝试备用解析方式
            try:
                # 标准OpenAI格式
                if "choices" in result and result["choices"]:
                    choice = result["choices"][0]
                    if "message" in choice:
                        message = choice["message"]
                        # 优先使用content字段
                        if "content" in message and message["content"]:
                            return message["content"]
                        # 对于GLM-4.5等模型，尝试reasoning_content字段
                        elif "reasoning_content" in message and message["reasoning_content"]:
                            return message["reasoning_content"]
                        # 如果content为空但reasoning_content有内容，使用reasoning_content
                        elif "reasoning_content" in message:
                            return message["reasoning_content"] or ""
                
                # 简化格式
                if "response" in result:
                    return str(result["response"])
                
                if "output" in result:
                    return str(result["output"])
                
                # 记录原始响应以便调试
                self.logger.debug(f"无法解析API响应，原始响应: {result}")
                raise ModelResponseException("无法解析API响应")
                
            except Exception as parse_error:
                self.logger.debug(f"响应解析失败，原始响应: {result}, 错误: {str(parse_error)}")
                raise ModelResponseException(f"响应解析失败: {str(e)}")
    
    def _perform_health_check(self):
        """执行健康检查"""
        try:
            # 发送简单的测试请求
            test_input = "Hello"
            result = self._make_prediction(test_input, {"max_tokens": 10})
            
            if not result or (isinstance(result, str) and result.strip() == ""):
                raise ModelException("健康检查返回空结果")
                
        except Exception as e:
            raise ModelException(f"健康检查失败: {str(e)}")


class OpenAIAdapter(OpenAICompatibleAdapter):
    """OpenAI模型适配器 - 保持向后兼容"""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        """
        初始化OpenAI适配器
        
        Args:
            model_id: 模型ID
            config: 配置信息，应包含api_key
        """
        # 设置OpenAI默认配置
        openai_config = {
            "base_url": "https://api.openai.com/v1",
            "model_name": "gpt-3.5-turbo",
            "provider": "openai",
            **config
        }
        super().__init__(model_id, openai_config)


class LocalModelAdapter(BaseModelAdapter):
    """本地模型适配器"""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        """
        初始化本地模型适配器
        
        Args:
            model_id: 模型ID
            config: 配置信息
        """
        super().__init__(model_id, config)
        self.model_path = config.get("model_path", "")
        self.device = config.get("device", "cpu")
        self._model = None
        self._tokenizer = None
    
    def _make_prediction(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """执行本地模型推理"""
        try:
            # 懒加载模型
            if self._model is None:
                self._load_model()
            
            if self._model is None:
                raise ModelException("模型加载失败")
            
            # 验证输入
            if not input_text or not input_text.strip():
                raise ValueError("输入文本不能为空")
            
            # 模拟本地模型推理
            # 实际实现中这里应该调用具体的模型推理逻辑
            import random
            if random.random() < 0.1:  # 模拟10%的失败率
                raise ModelException("模型推理失败")
            
            return f"本地模型{self.model_id}的输出: {input_text[:50]}..."
            
        except FileNotFoundError:
            raise ModelException(f"模型文件未找到: {self.model_path}")
        except MemoryError:
            raise ModelException("内存不足，无法加载模型")
        except Exception as e:
            if isinstance(e, (ModelException, ValueError)):
                raise
            raise ModelException(f"本地模型推理错误: {str(e)}")
    
    def _load_model(self):
        """加载模型"""
        # 模拟模型加载
        print(f"加载本地模型: {self.model_path}")
        time.sleep(1)  # 模拟加载时间
        self._model = "loaded_model"
        self._tokenizer = "loaded_tokenizer"


class HTTPModelAdapter(BaseModelAdapter):
    """HTTP API模型适配器"""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        """
        初始化HTTP API适配器
        
        Args:
            model_id: 模型ID
            config: 配置信息
        """
        super().__init__(model_id, config)
        self.api_url = config.get("api_url", "")
        self.headers = config.get("headers", {})
        self.request_format = config.get("request_format", "json")
    
    def _make_prediction(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """执行HTTP API调用"""
        try:
            if not self.api_url:
                raise ModelException("API URL未配置")
            
            if self.request_format == "json":
                data = {
                    "input": input_text,
                    "model_id": self.model_id
                }
                
                if context:
                    data.update(context)
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=data,
                    timeout=self.timeout
                )
            else:
                # 表单格式
                data = {
                    "input": input_text,
                    "model_id": self.model_id
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    data=data,
                    timeout=self.timeout
                )
            
            # 处理HTTP错误
            if response.status_code == 401:
                raise ModelAuthenticationException("API认证失败")
            elif response.status_code == 429:
                raise ModelRateLimitException("API调用频率超限")
            elif response.status_code >= 500:
                raise ModelConnectionException(f"服务器错误: {response.status_code}")
            
            response.raise_for_status()
            
            try:
                result = response.json()
            except ValueError:
                raise ModelResponseException("响应不是有效的JSON格式")
            
            # 提取输出
            output = result.get("output") or result.get("response") or result.get("result")
            if output is None:
                raise ModelResponseException("响应中未找到输出字段")
            
            return str(output)
            
        except requests.exceptions.Timeout:
            raise ModelTimeoutException(f"API调用超时 ({self.timeout}秒)")
        except requests.exceptions.ConnectionError as e:
            raise ModelConnectionException(f"连接失败: {str(e)}")
        except requests.exceptions.HTTPError as e:
            raise ModelConnectionException(f"HTTP错误: {str(e)}")
        except Exception as e:
            if isinstance(e, (ModelException,)):
                raise
            raise ModelException(f"HTTP API调用错误: {str(e)}")


class ModelAdapterFactory:
    """模型适配器工厂"""
    
    _adapters = {
        "openai": OpenAIAdapter,
        "openai_compatible": OpenAICompatibleAdapter,
        "local": LocalModelAdapter,
        "http": HTTPModelAdapter
    }
    
    # 预定义的提供商配置
    _provider_configs = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "auth_type": "bearer",
            "chat_endpoint": "/chat/completions",
            "response_path": ["choices", 0, "message", "content"]
        },
        "bigmodel": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "auth_type": "bearer",
            "chat_endpoint": "/chat/completions",
            "response_path": ["choices", 0, "message", "content"],
            "fallback_response_paths": [
                ["choices", 0, "message", "reasoning_content"],
                ["choices", 0, "message", "content"]
            ],
            "custom_headers": {
                "User-Agent": "Industry-Evaluation/1.0"
            }
        },
        "zhipu": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "auth_type": "bearer", 
            "chat_endpoint": "/chat/completions",
            "response_path": ["choices", 0, "message", "content"]
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "auth_type": "bearer",
            "chat_endpoint": "/chat/completions", 
            "response_path": ["choices", 0, "message", "content"]
        },
        "moonshot": {
            "base_url": "https://api.moonshot.cn/v1",
            "auth_type": "bearer",
            "chat_endpoint": "/chat/completions",
            "response_path": ["choices", 0, "message", "content"]
        }
    }
    
    @classmethod
    def create_adapter(cls, adapter_type: str, model_id: str, config: Dict[str, Any]) -> ModelAdapter:
        """
        创建模型适配器
        
        Args:
            adapter_type: 适配器类型
            model_id: 模型ID
            config: 配置信息
            
        Returns:
            ModelAdapter: 模型适配器实例
        """
        if adapter_type not in cls._adapters:
            raise ValueError(f"不支持的适配器类型: {adapter_type}")
        
        adapter_class = cls._adapters[adapter_type]
        return adapter_class(model_id, config)
    
    @classmethod
    def create_openai_compatible_adapter(cls, model_id: str, provider: str, api_key: str, 
                                       model_name: str, **kwargs) -> ModelAdapter:
        """
        创建OpenAI兼容适配器的便捷方法
        
        Args:
            model_id: 模型ID
            provider: 提供商名称
            api_key: API密钥
            model_name: 模型名称
            **kwargs: 其他配置参数
            
        Returns:
            ModelAdapter: 模型适配器实例
        """
        if provider not in cls._provider_configs:
            raise ValueError(f"不支持的提供商: {provider}，支持的提供商: {list(cls._provider_configs.keys())}")
        
        # 合并提供商默认配置和用户配置
        config = cls._provider_configs[provider].copy()
        config.update({
            "api_key": api_key,
            "model_name": model_name,
            "provider": provider,
            **kwargs
        })
        
        return cls.create_adapter("openai_compatible", model_id, config)
    
    @classmethod
    def register_adapter(cls, adapter_type: str, adapter_class):
        """
        注册新的适配器类型
        
        Args:
            adapter_type: 适配器类型名称
            adapter_class: 适配器类
        """
        cls._adapters[adapter_type] = adapter_class
    
    @classmethod
    def get_supported_types(cls) -> List[str]:
        """获取支持的适配器类型"""
        return list(cls._adapters.keys())
    
    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """获取支持的OpenAI兼容提供商"""
        return list(cls._provider_configs.keys())
    
    @classmethod
    def get_provider_config(cls, provider: str) -> Dict[str, Any]:
        """获取提供商的默认配置"""
        if provider not in cls._provider_configs:
            raise ValueError(f"不支持的提供商: {provider}")
        return cls._provider_configs[provider].copy()


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        """初始化模型管理器"""
        self.adapters: Dict[str, ModelAdapter] = {}
        self.adapter_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_model(self, model_id: str, adapter_type: str, config: Dict[str, Any]):
        """
        注册模型
        
        Args:
            model_id: 模型ID
            adapter_type: 适配器类型
            config: 配置信息
        """
        adapter = ModelAdapterFactory.create_adapter(adapter_type, model_id, config)
        self.adapters[model_id] = adapter
        self.adapter_configs[model_id] = config.copy()
    
    def get_model(self, model_id: str) -> Optional[ModelAdapter]:
        """
        获取模型适配器
        
        Args:
            model_id: 模型ID
            
        Returns:
            Optional[ModelAdapter]: 模型适配器
        """
        return self.adapters.get(model_id)
    
    def remove_model(self, model_id: str) -> bool:
        """
        移除模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 移除是否成功
        """
        if model_id in self.adapters:
            del self.adapters[model_id]
            if model_id in self.adapter_configs:
                del self.adapter_configs[model_id]
            return True
        return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有注册的模型"""
        models = []
        
        for model_id, adapter in self.adapters.items():
            config = self.adapter_configs.get(model_id, {})
            models.append({
                "model_id": model_id,
                "adapter_type": type(adapter).__name__,
                "is_available": adapter.is_available(),
                "config": config
            })
        
        return models
    
    def test_model(self, model_id: str) -> Dict[str, Any]:
        """
        测试模型可用性
        
        Args:
            model_id: 模型ID
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        adapter = self.get_model(model_id)
        if not adapter:
            return {"available": False, "error": "模型未注册"}
        
        try:
            start_time = time.time()
            test_output = adapter.predict("这是一个测试输入")
            response_time = time.time() - start_time
            
            return {
                "available": True,
                "response_time": response_time,
                "test_output": test_output[:100] + "..." if len(test_output) > 100 else test_output
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def batch_test_models(self) -> Dict[str, Dict[str, Any]]:
        """批量测试所有模型"""
        results = {}
        
        for model_id in self.adapters:
            results[model_id] = self.test_model(model_id)
        
        return results
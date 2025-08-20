# OpenAI兼容API测试指南

本文档介绍如何使用industry_evaluation系统测试各种OpenAI兼容的API提供商，包括BigModel GLM、智谱AI、DeepSeek、Moonshot等。

## 概述

industry_evaluation系统现在支持通过OpenAI兼容的API接口测试各种大语言模型。这使得您可以轻松地：

- 测试不同提供商的模型性能
- 比较不同模型在行业特定任务上的表现
- 集成第三方API到您的评估流程中

## 支持的提供商

### 预配置提供商

系统预配置了以下提供商的设置：

| 提供商 | 标识符 | 默认API地址 | 说明 |
|--------|--------|-------------|------|
| BigModel | `bigmodel` | `https://open.bigmodel.cn/api/paas/v4` | 智谱AI的GLM系列模型 |
| 智谱AI | `zhipu` | `https://open.bigmodel.cn/api/paas/v4` | 同BigModel |
| DeepSeek | `deepseek` | `https://api.deepseek.com/v1` | DeepSeek系列模型 |
| Moonshot | `moonshot` | `https://api.moonshot.cn/v1` | Moonshot AI模型 |
| OpenAI | `openai` | `https://api.openai.com/v1` | OpenAI官方模型 |

### 自定义提供商

您也可以配置任何OpenAI兼容的API提供商。

## 快速开始

### 1. 使用命令行工具快速测试

最简单的方式是使用我们提供的命令行工具：

```bash
# 测试BigModel GLM-4.5
python tools/test_openai_compatible_api.py \
  --url "https://open.bigmodel.cn/api/paas/v4/chat/completions" \
  --token "your_api_key_here" \
  --model "glm-4.5" \
  --message "中国大模型行业在2025年将面临哪些机遇和挑战？"
```

这个命令会：
- 发送请求到指定的API
- 显示响应时间和状态
- 打印模型的回复内容
- 显示令牌使用情况

### 2. 使用Python脚本进行批量测试

```bash
# 运行完整的评估测试
python examples/openai_compatible_api_test.py \
  --provider bigmodel \
  --api-key your_api_key_here \
  --model glm-4.5 \
  --verbose \
  --output results.json
```

这会运行一系列预定义的测试用例，并生成详细的评估报告。

## 详细使用方法

### 1. 配置API密钥

首先，您需要获取相应提供商的API密钥：

#### BigModel (智谱AI)
1. 访问 [https://open.bigmodel.cn](https://open.bigmodel.cn)
2. 注册账号并获取API密钥
3. 设置环境变量：
   ```bash
   export BIGMODEL_API_KEY="your_api_key_here"
   ```

#### DeepSeek
1. 访问 [https://platform.deepseek.com](https://platform.deepseek.com)
2. 获取API密钥
3. 设置环境变量：
   ```bash
   export DEEPSEEK_API_KEY="your_api_key_here"
   ```

#### 其他提供商
类似地设置相应的环境变量。

### 2. 使用配置文件

创建配置文件 `my_models.yaml`：

```yaml
models:
  my_glm_model:
    adapter_type: "openai_compatible"
    provider: "bigmodel"
    api_key: "${BIGMODEL_API_KEY}"
    model_name: "glm-4.5"
    timeout: 30
    retry_config:
      max_retries: 3
      strategy: "exponential_backoff"

  my_deepseek_model:
    adapter_type: "openai_compatible"
    provider: "deepseek"
    api_key: "${DEEPSEEK_API_KEY}"
    model_name: "deepseek-chat"
    timeout: 30
```

### 3. 在Python代码中使用

```python
from industry_evaluation.adapters.model_adapter import ModelAdapterFactory, ModelManager

# 创建模型管理器
model_manager = ModelManager()

# 方法1: 使用预配置的提供商
adapter = ModelAdapterFactory.create_openai_compatible_adapter(
    model_id="my_glm_model",
    provider="bigmodel",
    api_key="your_api_key_here",
    model_name="glm-4.5"
)

# 方法2: 使用自定义配置
config = {
    "api_key": "your_api_key_here",
    "model_name": "glm-4.5",
    "base_url": "https://open.bigmodel.cn/api/paas/v4",
    "provider": "bigmodel",
    "timeout": 30
}
adapter = ModelAdapterFactory.create_adapter("openai_compatible", "my_model", config)

# 注册模型
model_manager.register_model("my_model", "openai_compatible", config)

# 测试模型
result = model_manager.test_model("my_model")
print(f"模型可用性: {result['available']}")

# 使用模型进行预测
if result['available']:
    response = adapter.predict("什么是人工智能？", {"max_tokens": 500})
    print(f"模型回复: {response}")
```

### 4. 集成到评估流程

```python
from industry_evaluation.core.evaluation_engine import IndustryEvaluationEngine
from industry_evaluation.core.interfaces import EvaluationConfig

# 创建评估配置
config = EvaluationConfig(
    industry_domain="technology",
    evaluation_dimensions=["knowledge_accuracy", "terminology_usage"],
    weight_config={"knowledge_accuracy": 0.6, "terminology_usage": 0.4}
)

# 准备测试数据
test_dataset = [
    {
        "id": "test_1",
        "input": "什么是机器学习？",
        "expected_output": "机器学习是人工智能的一个分支...",
        "context": {"domain": "technology"}
    }
]

# 创建评估引擎
engine = IndustryEvaluationEngine(
    model_manager=model_manager,
    evaluators=evaluators,  # 您的评估器
    result_aggregator=result_aggregator,
    report_generator=report_generator
)

# 启动评估
task_id = engine.evaluate_model("my_model", test_dataset, config)

# 获取结果
result = engine.get_evaluation_result(task_id)
```

## 高级配置

### 自定义认证方式

```python
config = {
    "api_key": "your_api_key",
    "model_name": "custom-model",
    "base_url": "https://your-api.com/v1",
    "auth_type": "custom",  # 自定义认证
    "auth_header": "X-API-Key",
    "auth_value": "your_api_key",
    "custom_headers": {
        "X-Custom-Header": "value"
    }
}
```

### 自定义响应解析

```python
config = {
    "api_key": "your_api_key",
    "model_name": "custom-model",
    "base_url": "https://your-api.com/v1",
    "response_path": ["data", "response", "text"],  # 自定义响应路径
    "param_mapping": {
        "max_tokens": "max_length",  # 参数名映射
        "temperature": "temp"
    }
}
```

## 故障排除

### 常见错误

1. **认证失败 (401)**
   - 检查API密钥是否正确
   - 确认API密钥是否有效且未过期

2. **请求频率超限 (429)**
   - 降低请求频率
   - 配置重试策略

3. **连接超时**
   - 增加timeout设置
   - 检查网络连接

4. **响应格式错误**
   - 检查API提供商的响应格式
   - 自定义response_path配置

### 调试技巧

1. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **使用verbose模式**
   ```bash
   python examples/openai_compatible_api_test.py --verbose
   ```

3. **保存原始响应**
   ```bash
   python tools/test_openai_compatible_api.py --output debug.json --verbose
   ```

## 性能优化

### 批量处理

```python
# 配置并发处理
config = {
    "max_workers": 4,  # 并发线程数
    "rate_limit_delay": 0.1,  # 请求间隔
    "retry_config": {
        "max_retries": 3,
        "strategy": "exponential_backoff"
    }
}
```

### 缓存和重试

```python
config = {
    "retry_config": {
        "max_retries": 3,
        "strategy": "exponential_backoff",
        "base_delay": 1.0,
        "max_delay": 60.0
    },
    "fallback_enabled": True,
    "fallback_response": "抱歉，模型暂时不可用"
}
```

## 示例和模板

查看以下文件获取更多示例：

- `examples/openai_compatible_api_test.py` - 完整的测试脚本
- `examples/configs/openai_compatible_models.yaml` - 配置文件模板
- `tools/test_openai_compatible_api.py` - 命令行测试工具

## 贡献

如果您想添加对新的OpenAI兼容API提供商的支持，请：

1. 在 `ModelAdapterFactory._provider_configs` 中添加提供商配置
2. 测试新提供商的兼容性
3. 更新文档
4. 提交Pull Request

## 支持

如果您遇到问题或有建议，请：

1. 查看本文档的故障排除部分
2. 检查GitHub Issues
3. 创建新的Issue并提供详细信息
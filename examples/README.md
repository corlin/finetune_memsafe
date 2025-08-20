# Industry Evaluation System 演示程序

本目录包含了 Industry Evaluation System（行业评估系统）的完整功能演示程序，帮助您快速了解和使用系统的各项功能。

## 📋 演示程序列表

### 1. 🎬 完整功能演示 (`complete_demo.py`)

**最全面的演示程序**，展示系统的所有核心功能：

- ⚙️ 配置管理系统
- 🤖 模型适配器和管理
- 📊 多维度评估器
- 🎯 单模型评估
- ⚖️ 多模型对比评估
- 📦 批量评估处理
- 📄 评估报告生成
- 🌐 REST API接口
- 👁️ 实时进度监控

```bash
# 运行完整演示
python examples/complete_demo.py
```

**特点：**
- 异步执行，支持并发评估
- 真实的评估流程模拟
- 详细的进度监控和结果展示
- 完整的错误处理和资源清理

### 2. 🚀 简化演示 (`simple_demo.py`)

**快速入门演示**，适合初次使用者：

- 基础配置设置
- 简单模型注册
- 基本评估流程
- 结果对比展示

```bash
# 运行简化演示
python examples/simple_demo.py
```

**特点：**
- 同步执行，逻辑清晰
- 代码简洁，易于理解
- 快速体验核心功能
- 适合学习和参考

### 3. 🌐 API接口演示 (`api_demo.py`)

**REST API功能演示**，展示如何通过HTTP接口使用系统：

- 🔍 健康检查和系统信息
- 🤖 模型管理接口
- 🎯 评估任务创建和监控
- ⚙️ 配置管理接口
- 📤 文件上传功能

```bash
# 运行API演示
python examples/api_demo.py
```

**特点：**
- 启动内置API服务器
- 测试所有主要API端点
- 展示API调用方法
- 包含错误处理示例

### 4. ⚙️ 配置管理演示 (`config_demo.py`)

**配置系统专项演示**，深入展示配置管理功能：

- 🎨 配置模板生成（金融、医疗行业）
- 📂 配置文件加载和解析
- ✅ 配置验证和错误检测
- 🔄 配置动态更新
- 🌍 环境变量支持
- 👁️ 配置文件监控
- ⚡ 性能测试

```bash
# 运行配置管理演示
python examples/config_demo.py
```

**特点：**
- 全面的配置功能展示
- 实际的配置文件操作
- 性能测试和优化建议
- 最佳实践示例

## 🛠️ 运行环境要求

### 基础依赖

```bash
# 方法1: 使用快速安装脚本（推荐）
python install_demo_deps.py

# 方法2: 手动安装依赖
pip install pyyaml requests flask flask-restx flask-cors watchdog psutil

# 方法3: 使用项目依赖文件
pip install -r requirements.txt

# 方法4: 开发环境安装
pip install -e .
```

### 解决导入问题

如果遇到 `ModuleNotFoundError: No module named 'industry_evaluation'` 错误：

```bash
# 快速解决方案1: 使用快速启动脚本
python quick_start.py

# 快速解决方案2: 在项目根目录运行
cd /path/to/your/project
python examples/simple_demo.py

# 快速解决方案3: 设置 PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
python examples/simple_demo.py
```

### 可选依赖

```bash
# API演示需要的额外依赖
pip install flask flask-restx flask-cors

# 配置文件监控功能
pip install watchdog

# 性能测试工具
pip install psutil
```

## 🚀 快速开始

### 1. 运行简化演示（推荐新手）

```bash
cd examples
python simple_demo.py
```

### 2. 运行完整演示

```bash
cd examples
python complete_demo.py
```

### 3. 测试API接口

```bash
cd examples
python api_demo.py

# 在另一个终端测试API
curl http://localhost:5001/health
curl http://localhost:5001/info
```

### 4. 探索配置管理

```bash
cd examples
python config_demo.py
```

## 📊 演示数据说明

### 测试数据集

演示程序使用模拟的金融领域测试数据：

```json
{
  "id": "finance_1",
  "input": "请解释金融风险管理中的VaR模型及其应用",
  "expected_output": "VaR（Value at Risk）是一种风险度量方法...",
  "context": {
    "industry": "finance",
    "topic": "risk_management",
    "difficulty": "intermediate"
  }
}
```

### 模拟模型

演示程序包含三种质量的模拟模型：

- **🏆 专家模型** (`excellent`): 高质量回答，包含详细分析
- **👍 通用模型** (`good`): 中等质量回答，基本准确
- **📝 基础模型** (`poor`): 简单回答，可能不够详细

## 🎯 评估维度说明

系统支持多个评估维度：

- **📚 知识准确性** (`knowledge`): 评估专业知识的准确性
- **🏷️ 术语使用** (`terminology`): 评估专业术语的正确使用
- **🧠 逻辑推理** (`reasoning`): 评估逻辑推理能力
- **📖 长文本理解** (`long_text`): 评估长文本理解能力

## 📈 结果解读

### 评估得分

- **综合得分**: 0.0-1.0，越高越好
- **维度得分**: 各评估维度的具体得分
- **样本数量**: 处理的测试样本总数
- **改进建议**: 基于评估结果的具体建议

### 示例输出

```
📊 评估结果:
  - 综合得分: 0.856
  - 知识得分: 0.892
  - 术语得分: 0.834
  - 推理得分: 0.841
  - 处理样本数: 3

💡 改进建议:
  - 加强金融术语的准确使用
  - 提高复杂场景的推理能力
  - 增强专业知识的深度
```

## 🔧 自定义配置

### 创建自定义配置

```python
from industry_evaluation.config.config_manager import ConfigTemplate, ModelConfig

# 生成基础配置
config = ConfigTemplate.generate_finance_config()

# 添加自定义模型
custom_model = ModelConfig(
    model_id="my_model",
    adapter_type="openai",
    api_key="your_api_key",
    model_name="gpt-4",
    timeout=60
)

config.models["my_model"] = custom_model

# 保存配置
ConfigTemplate.save_template(config, "my_config.yaml")
```

### 使用环境变量

```bash
# 设置环境变量
export EVAL_MAX_WORKERS=8
export EVAL_LOG_LEVEL=DEBUG
export OPENAI_API_KEY=your_api_key
export OPENAI_MODEL_NAME=gpt-4

# 运行演示
python examples/simple_demo.py
```

## 🐛 故障排除

### 常见问题

1. **导入错误**
   ```bash
   # 确保在项目根目录运行
   cd /path/to/industry-evaluation
   python examples/simple_demo.py
   ```

2. **API服务器启动失败**
   ```bash
   # 检查端口是否被占用
   netstat -an | grep 5001
   
   # 或使用不同端口
   python api_demo.py --port 5002
   ```

3. **配置文件错误**
   ```bash
   # 检查配置文件格式
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行演示
python examples/complete_demo.py
```

## 📚 进一步学习

### 相关文档

- [API参考文档](../docs/API_REFERENCE.md)
- [配置指南](../docs/CONFIGURATION_GUIDE.md)
- [部署指南](../docs/DEPLOYMENT_GUIDE.md)
- [故障排除](../docs/TROUBLESHOOTING_GUIDE.md)

### 扩展示例

- [高级评估配置](config_examples/advanced_config.yaml)
- [批量处理脚本](../scripts/batch_evaluation.py)
- [自定义评估器](../industry_evaluation/evaluators/custom_evaluator.py)

## 🤝 贡献

欢迎提交问题和改进建议：

1. 🐛 报告bug或问题
2. 💡 提出新功能建议
3. 📝 改进文档和示例
4. 🔧 提交代码改进

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](../LICENSE) 文件。

---

**💡 提示**: 建议按顺序运行演示程序，从 `simple_demo.py` 开始，逐步了解系统的各项功能。
# 🚀 Industry Evaluation System 完整演示指南

欢迎使用 Industry Evaluation System（行业评估系统）！本指南将帮助您快速了解和使用系统的所有功能。

## 📋 目录

- [快速开始](#快速开始)
- [演示程序概览](#演示程序概览)
- [详细功能演示](#详细功能演示)
- [API接口使用](#api接口使用)
- [配置管理](#配置管理)
- [故障排除](#故障排除)
- [进阶使用](#进阶使用)

## 🚀 快速开始

### 1. 环境准备

```bash
# 进入项目目录
cd /path/to/your/project

# 方法1: 使用快速安装脚本（推荐）
python install_demo_deps.py

# 方法2: 手动安装核心依赖
pip install pyyaml requests flask flask-restx flask-cors watchdog psutil

# 方法3: 安装完整依赖
pip install -r requirements.txt
```

### 解决常见导入问题

如果遇到 `ModuleNotFoundError: No module named 'industry_evaluation'`：

```bash
# 快速解决方案
python quick_start.py

# 或者确保在项目根目录运行
pwd  # 确认当前目录
python examples/simple_demo.py
```

### 2. 验证安装

```bash
# 运行测试脚本验证所有组件
python examples/test_demos.py
```

### 3. 运行第一个演示

```bash
# 使用交互式启动器
python examples/run_demo.py

# 或者直接运行简化演示
python examples/simple_demo.py
```

## 📊 演示程序概览

| 演示程序 | 适用场景 | 运行时间 | 复杂度 |
|---------|---------|---------|--------|
| 🚀 简化演示 | 初次使用，快速了解 | 2-3分钟 | ⭐ |
| 🎬 完整演示 | 全面了解所有功能 | 5-10分钟 | ⭐⭐⭐ |
| 🌐 API演示 | 接口开发，集成测试 | 3-5分钟 | ⭐⭐ |
| ⚙️ 配置演示 | 系统配置，性能调优 | 2-4分钟 | ⭐⭐ |

## 🎯 详细功能演示

### 1. 简化演示 (`simple_demo.py`)

**最适合初学者的演示程序**

```bash
python examples/simple_demo.py
```

**演示内容：**
- ✅ 基础配置设置
- ✅ 模型注册和管理
- ✅ 评估器初始化
- ✅ 单模型评估流程
- ✅ 结果对比和展示

**预期输出：**
```
🚀 Industry Evaluation System - 简化演示
==================================================
📁 临时目录: /tmp/tmpXXXXXX

🔧 设置配置...
✅ 配置创建完成

🤖 设置模型...
✅ 模型设置完成

📊 设置评估器...
✅ 评估器设置完成

🚀 创建评估引擎...
✅ 评估引擎创建完成

📝 准备测试数据...
✅ 准备了 2 个测试样本

🎯 开始评估...
🔄 评估专家模型...
✅ 专家模型评估完成
🔄 评估基础模型...
✅ 基础模型评估完成

📊 评估结果对比:
--------------------------------------------------
模型         综合得分    知识得分    术语得分
--------------------------------------------------
专家模型     0.856      0.892      0.834
基础模型     0.723      0.756      0.689
--------------------------------------------------

📄 生成评估报告...
✅ 报告已保存到: /tmp/tmpXXXXXX/evaluation_report.json

🎉 演示完成!
```

### 2. 完整功能演示 (`complete_demo.py`)

**最全面的功能展示**

```bash
python examples/complete_demo.py
```

**演示内容：**
- 🔧 配置管理系统
- 🤖 模型适配器和异常处理
- 📊 多维度评估器系统
- 🎯 单模型详细评估
- ⚖️ 多模型并行对比
- 📦 大规模批量评估
- 📄 专业报告生成
- 🌐 REST API接口
- 👁️ 实时进度监控

**关键特性：**
- **异步执行**：支持并发评估任务
- **进度监控**：实时显示评估进度
- **错误处理**：完善的异常处理机制
- **资源管理**：自动清理临时资源

### 3. API接口演示 (`api_demo.py`)

**REST API功能完整测试**

```bash
python examples/api_demo.py
```

**演示内容：**
- 🔍 健康检查 (`GET /health`)
- 📋 系统信息 (`GET /info`)
- 🤖 模型管理 (`GET /models`)
- 🎯 评估任务创建 (`POST /evaluations`)
- ⏳ 任务状态监控 (`GET /evaluations/{id}`)
- 📋 任务列表 (`GET /evaluations`)
- ⚙️ 配置管理 (`GET /config`)
- 📤 文件上传 (`POST /upload/dataset`)

**API服务器：**
- 地址：`http://localhost:5001`
- 文档：`http://localhost:5001/docs/`
- 自动启动和关闭

### 4. 配置管理演示 (`config_demo.py`)

**配置系统专项深度演示**

```bash
python examples/config_demo.py
```

**演示内容：**
- 🎨 行业配置模板生成
- 📂 配置文件加载和解析
- ✅ 配置验证和错误检测
- 🔄 配置动态更新
- 🤖 模型配置管理
- 📊 评估器配置管理
- 🌍 环境变量支持
- 👁️ 配置文件监控
- ⚡ 性能测试和优化

## 🌐 API接口使用

### 基础API调用

```bash
# 健康检查
curl http://localhost:5001/health

# 获取系统信息
curl http://localhost:5001/info

# 获取模型列表
curl http://localhost:5001/models
```

### 创建评估任务

```bash
curl -X POST http://localhost:5001/evaluations \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "finance_gpt4",
    "dataset": [
      {
        "id": "test_1",
        "input": "什么是金融风险管理？",
        "expected_output": "金融风险管理是识别、评估和控制金融风险的过程。",
        "context": {"industry": "finance"}
      }
    ],
    "config": {
      "industry_domain": "finance",
      "evaluation_dimensions": ["knowledge", "terminology"],
      "weight_config": {"knowledge": 0.7, "terminology": 0.3},
      "threshold_config": {"knowledge": 0.6, "terminology": 0.5}
    }
  }'
```

### 监控评估进度

```bash
# 获取任务状态
curl http://localhost:5001/evaluations/{task_id}

# 获取评估报告
curl http://localhost:5001/evaluations/{task_id}/report?format=json
```

## ⚙️ 配置管理

### 配置文件结构

```yaml
version: "1.0.0"
system:
  max_workers: 4
  log_level: "INFO"
  cache_enabled: true
  
models:
  finance_expert:
    model_id: "finance_expert"
    adapter_type: "openai"
    api_key: "your_api_key"
    model_name: "gpt-4"
    timeout: 60
    max_retries: 3
    
evaluators:
  knowledge:
    evaluator_type: "knowledge"
    weight: 0.5
    threshold: 0.7
    enabled: true
    
industry_domains:
  - "finance"
  - "healthcare"
  - "technology"
```

### 环境变量配置

```bash
# 系统配置
export EVAL_MAX_WORKERS=8
export EVAL_LOG_LEVEL=DEBUG

# 模型配置
export OPENAI_API_KEY=your_api_key
export OPENAI_MODEL_NAME=gpt-4

# 运行演示
python examples/simple_demo.py
```

### 配置模板生成

```python
from industry_evaluation.config.config_manager import ConfigTemplate

# 生成金融行业配置
finance_config = ConfigTemplate.generate_finance_config()
ConfigTemplate.save_template(finance_config, "finance_config.yaml")

# 生成医疗行业配置
healthcare_config = ConfigTemplate.generate_healthcare_config()
ConfigTemplate.save_template(healthcare_config, "healthcare_config.yaml")
```

## 🐛 故障排除

### 常见问题及解决方案

#### 1. 导入错误

```bash
# 错误：ModuleNotFoundError: No module named 'industry_evaluation'
# 解决：确保在项目根目录运行
cd /path/to/industry-evaluation
python examples/simple_demo.py

# 或者安装包
pip install -e .
```

#### 2. 依赖缺失

```bash
# 错误：ImportError: No module named 'yaml'
# 解决：安装依赖
pip install -r requirements.txt

# 或者单独安装
pip install pyyaml requests flask
```

#### 3. API服务器启动失败

```bash
# 错误：Address already in use
# 解决：检查端口占用
netstat -an | grep 5001

# 或者杀死占用进程
lsof -ti:5001 | xargs kill -9
```

#### 4. 配置文件错误

```bash
# 验证配置文件格式
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# 使用配置验证工具
python examples/config_demo.py
```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行演示
python examples/complete_demo.py
```

### 性能问题

```bash
# 检查系统资源
python examples/test_demos.py

# 调整配置参数
export EVAL_MAX_WORKERS=2  # 减少并发数
export EVAL_LOG_LEVEL=WARNING  # 减少日志输出
```

## 🎓 进阶使用

### 自定义模型适配器

```python
from industry_evaluation.adapters.model_adapter import BaseModelAdapter

class CustomModelAdapter(BaseModelAdapter):
    def _make_prediction(self, input_text: str, context=None) -> str:
        # 实现自定义预测逻辑
        return "自定义模型的回答"
    
    def is_available(self) -> bool:
        # 实现可用性检查
        return True

# 注册自定义适配器
ModelAdapterFactory.register_adapter("custom", CustomModelAdapter)
```

### 自定义评估器

```python
from industry_evaluation.evaluators.base_evaluator import BaseEvaluator

class CustomEvaluator(BaseEvaluator):
    def evaluate(self, input_text: str, model_output: str, 
                expected_output: str, context: dict) -> EvaluationScore:
        # 实现自定义评估逻辑
        score = self._calculate_custom_score(model_output, expected_output)
        return EvaluationScore(score=score, explanation="自定义评估")
```

### 批量处理脚本

```python
from industry_evaluation.core.batch_evaluator import BatchEvaluator, BatchEvaluationConfig

# 配置批量评估
batch_config = BatchEvaluationConfig(
    batch_size=100,
    max_concurrent_tasks=4,
    enable_parallel_processing=True,
    save_intermediate_results=True
)

# 创建批量任务
batch_task = batch_evaluator.create_batch_task(
    task_id="production_evaluation",
    model_ids=["model_1", "model_2", "model_3"],
    dataset_path="large_dataset.json",
    evaluation_config=eval_config,
    batch_config=batch_config
)
```

### 生产环境部署

```bash
# 使用 gunicorn 部署API
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 industry_evaluation.api.rest_api:create_api_app

# 使用 Docker 部署
docker build -t industry-evaluation .
docker run -p 8000:8000 industry-evaluation
```

## 📚 相关资源

### 文档链接

- [API参考文档](docs/API_REFERENCE.md)
- [配置指南](docs/CONFIGURATION_GUIDE.md)
- [部署指南](docs/DEPLOYMENT_GUIDE.md)
- [故障排除指南](docs/TROUBLESHOOTING_GUIDE.md)

### 示例配置

- [高级配置示例](examples/config_examples/advanced_config.yaml)
- [基础配置示例](examples/config_examples/basic_config.yaml)
- [行业特定配置](examples/config_examples/)

### 扩展脚本

- [批量评估脚本](scripts/batch_evaluation.py)
- [配置模板生成](scripts/config_templates.py)
- [实验对比工具](scripts/experiment_comparison.py)

## 🤝 获取帮助

### 问题反馈

1. 🐛 **Bug报告**：请提供详细的错误信息和复现步骤
2. 💡 **功能建议**：描述您希望添加的功能和使用场景
3. 📝 **文档改进**：指出文档中不清楚或错误的地方
4. 🔧 **使用问题**：描述您遇到的具体问题和环境信息

### 社区支持

- 查看现有的问题和解决方案
- 参与讨论和经验分享
- 贡献代码和文档改进

---

**🎉 恭喜！您已经掌握了 Industry Evaluation System 的完整使用方法。**

**💡 建议学习路径：**
1. 从简化演示开始了解基本概念
2. 通过配置演示掌握系统配置
3. 使用API演示学习接口调用
4. 运行完整演示体验所有功能
5. 根据需要进行自定义开发

**🚀 开始您的行业模型评估之旅吧！**
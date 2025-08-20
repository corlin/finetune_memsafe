# QA数据提取和Checkpoint评估工具

这个工具可以从markdown文件中提取QA问答对，生成评估用例，然后测试fine-tuned模型的性能。

## 功能特性

- **智能QA提取**: 从markdown文件中自动提取问答对
- **多维度评估**: 从关键词匹配、术语准确性、结构相似度等多个维度评估模型
- **灵活筛选**: 支持按分类、难度筛选QA对
- **多种采样策略**: 支持随机、平衡、顺序采样
- **详细报告**: 生成包含统计信息和详细结果的评估报告

## 快速开始

### 1. 环境准备

```bash
pip install torch transformers pyyaml
```

### 2. 数据准备

确保你的QA数据文件在 `data/raw/` 目录下，文件名格式为 `QA*.md`。

### 3. 运行评估

#### 方式一：交互式运行
```bash
python run_qa_evaluation.py
```

#### 方式二：直接运行
```bash
# 快速评估（10个样本）
python qa_extractor_evaluator.py --max-samples 10

# 完整评估（50个样本）
python qa_extractor_evaluator.py --max-samples 50

# 仅提取QA对
python qa_extractor_evaluator.py --extract-only
```

## 详细使用说明

### 命令行参数

```bash
python qa_extractor_evaluator.py [选项]

选项:
  --data-dir DIR              QA数据目录 (默认: data/raw)
  --checkpoint PATH           Checkpoint路径 (默认: enhanced-qwen3-finetuned/checkpoint-450)
  --output-qa FILE           QA对输出文件 (默认: extracted_qa_pairs.json)
  --output-report FILE       评估报告输出文件 (默认: qa_evaluation_report.json)
  --max-samples N            最大评估样本数 (默认: 50)
  --categories CAT1 CAT2     指定评估的分类
  --difficulties DIFF1 DIFF2  指定评估的难度 (easy/medium/hard)
  --sample-strategy STRATEGY  采样策略 (random/balanced/sequential)
  --extract-only             仅提取QA对，不进行评估
```

### 使用示例

#### 1. 按分类评估
```bash
python qa_extractor_evaluator.py \
    --categories "基础概念类" "技术要求类" \
    --max-samples 30
```

#### 2. 按难度评估
```bash
python qa_extractor_evaluator.py \
    --difficulties "medium" "hard" \
    --max-samples 20
```

#### 3. 自定义checkpoint路径
```bash
python qa_extractor_evaluator.py \
    --checkpoint "path/to/your/checkpoint" \
    --max-samples 25
```

#### 4. 平衡采样评估
```bash
python qa_extractor_evaluator.py \
    --sample-strategy balanced \
    --max-samples 40
```

## 输出文件说明

### 1. QA对文件 (extracted_qa_pairs.json)

包含提取的所有QA对信息：

```json
[
  {
    "id": "QA1_Q1",
    "question": "什么是信息系统密码应用的基本要求？",
    "answer": "信息系统密码应用基本要求是指...",
    "category": "基础概念类",
    "source_file": "QA1.md",
    "difficulty": "medium",
    "keywords": ["密码", "信息系统", "安全"]
  }
]
```

### 2. 评估报告 (qa_evaluation_report.json)

包含详细的评估结果：

```json
{
  "evaluation_summary": {
    "total_cases": 50,
    "average_score": 0.756,
    "max_score": 0.923,
    "min_score": 0.445,
    "checkpoint_path": "enhanced-qwen3-finetuned/checkpoint-450"
  },
  "category_performance": {
    "基础概念类": 0.812,
    "技术要求类": 0.689
  },
  "difficulty_performance": {
    "easy": 0.834,
    "medium": 0.756,
    "hard": 0.623
  },
  "detailed_results": [...]
}
```

## 评估指标说明

工具使用多个维度来评估模型回答的质量：

1. **关键词匹配度** (30%): 模型回答与标准答案的关键词重叠程度
2. **术语准确性** (25%): 专业术语使用的正确性
3. **长度相似度** (15%): 回答长度与标准答案的相似程度
4. **结构相似度** (15%): 回答结构（数字、标点等）的相似程度
5. **完整性** (15%): 回答的完整性（是否被截断或过短）

## 配置文件

可以通过 `qa_evaluation_config.yaml` 文件自定义评估参数：

```yaml
# 模型配置
model:
  checkpoint_path: "enhanced-qwen3-finetuned/checkpoint-450"
  max_length: 512
  temperature: 0.7

# 评估配置
evaluation:
  max_samples: 50
  sample_strategy: "balanced"
  
# 评分权重
scoring_weights:
  keyword_overlap: 0.3
  terminology_match: 0.25
  length_similarity: 0.15
  structure_similarity: 0.15
  completeness: 0.15
```

## 数据格式要求

QA markdown文件应遵循以下格式：

```markdown
## 分类名称

### Q1: 问题内容？
A1: 答案内容。

### Q2: 另一个问题？
A2: 另一个答案。
```

## 性能优化建议

1. **GPU加速**: 确保有可用的GPU来加速模型推理
2. **批量处理**: 对于大量数据，考虑分批处理
3. **内存管理**: 使用较小的max_length减少内存占用
4. **采样策略**: 使用balanced策略获得更均衡的评估结果

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查checkpoint路径是否正确
   - 确保有足够的内存和GPU显存

2. **QA提取为空**
   - 检查markdown文件格式是否正确
   - 确保文件编码为UTF-8

3. **评估分数异常**
   - 检查模型是否正确加载
   - 查看详细的evaluation_details了解具体问题

### 调试模式

添加详细日志：

```bash
python qa_extractor_evaluator.py --max-samples 5 2>&1 | tee debug.log
```

## 扩展功能

### 自定义评估器

可以继承 `CheckpointEvaluator` 类来实现自定义评估逻辑：

```python
class CustomEvaluator(CheckpointEvaluator):
    def _calculate_score(self, expected, actual, qa_pair):
        # 自定义评分逻辑
        pass
```

### 添加新的评估维度

在 `_calculate_score` 方法中添加新的评估维度：

```python
# 添加语义相似度评估
semantic_score = self._calculate_semantic_similarity(expected, actual)
details["semantic_similarity"] = semantic_score
```

## 贡献指南

欢迎提交Issue和Pull Request来改进这个工具！

## 许可证

MIT License
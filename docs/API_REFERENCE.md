# API参考文档

本文档提供了数据拆分和评估系统所有公共API的详细说明。

## 目录

1. [数据拆分API](#数据拆分api)
2. [评估引擎API](#评估引擎api)
3. [指标计算API](#指标计算api)
4. [质量分析API](#质量分析api)
5. [基准管理API](#基准管理api)
6. [实验跟踪API](#实验跟踪api)
7. [报告生成API](#报告生成api)
8. [配置管理API](#配置管理api)
9. [数据模型](#数据模型)

## 数据拆分API

### DataSplitter

数据拆分器，用于将数据集拆分为训练集、验证集和测试集。

#### 构造函数

```python
DataSplitter(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: Optional[str] = None,
    random_seed: int = 42,
    min_samples_per_split: int = 10,
    enable_quality_analysis: bool = False,
    remove_duplicates: bool = False,
    balance_classes: bool = False
)
```

**参数:**
- `train_ratio`: 训练集比例 (0-1)
- `val_ratio`: 验证集比例 (0-1)
- `test_ratio`: 测试集比例 (0-1)
- `stratify_by`: 分层抽样的字段名
- `random_seed`: 随机种子
- `min_samples_per_split`: 每个拆分的最小样本数
- `enable_quality_analysis`: 是否启用质量分析
- `remove_duplicates`: 是否移除重复样本
- `balance_classes`: 是否平衡类别分布

#### 方法

##### split_data()

```python
def split_data(
    self,
    dataset: Dataset,
    output_dir: Optional[str] = None
) -> DataSplitResult
```

拆分数据集。

**参数:**
- `dataset`: 要拆分的数据集
- `output_dir`: 输出目录路径

**返回:** `DataSplitResult` 对象

**示例:**
```python
splitter = DataSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
result = splitter.split_data(dataset, "output/splits")
```

##### load_splits()

```python
@staticmethod
def load_splits(split_dir: str) -> DataSplitResult
```

加载已保存的数据拆分。

**参数:**
- `split_dir`: 拆分数据目录

**返回:** `DataSplitResult` 对象

##### analyze_distribution()

```python
def analyze_distribution(
    self,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset
) -> Dict[str, Any]
```

分析数据分布一致性。

**返回:** 包含分布分析结果的字典

## 评估引擎API

### EvaluationEngine

模型评估引擎，支持多任务、多指标评估。

#### 构造函数

```python
EvaluationEngine(
    config: EvaluationConfig,
    device: str = "cpu",
    max_workers: int = 4
)
```

**参数:**
- `config`: 评估配置对象
- `device`: 计算设备 ("cpu" 或 "cuda")
- `max_workers`: 最大并行工作线程数

#### 方法

##### evaluate_model()

```python
def evaluate_model(
    self,
    model: Any,
    tokenizer: Any,
    datasets: Dict[str, Dataset],
    model_name: str
) -> EvaluationResult
```

评估单个模型。

**参数:**
- `model`: 要评估的模型
- `tokenizer`: 分词器
- `datasets`: 任务名到数据集的映射
- `model_name`: 模型名称

**返回:** `EvaluationResult` 对象

##### evaluate_multiple_models()

```python
def evaluate_multiple_models(
    self,
    models_info: List[Dict[str, Any]],
    datasets: Dict[str, Dataset]
) -> List[EvaluationResult]
```

批量评估多个模型。

**参数:**
- `models_info`: 模型信息列表，每个包含 "model", "tokenizer", "name"
- `datasets`: 任务数据集

**返回:** 评估结果列表

##### save_evaluation_result()

```python
def save_evaluation_result(
    self,
    result: EvaluationResult,
    output_path: str
) -> None
```

保存评估结果。

## 指标计算API

### MetricsCalculator

各种评估指标的计算器。

#### 构造函数

```python
MetricsCalculator(
    language: str = "zh",
    device: str = "cpu",
    cache_dir: Optional[str] = None
)
```

#### 方法

##### calculate_bleu()

```python
def calculate_bleu(
    self,
    predictions: List[str],
    references: List[str],
    max_order: int = 4
) -> Dict[str, float]
```

计算BLEU分数。

**返回:** 包含BLEU分数和统计信息的字典

##### calculate_rouge()

```python
def calculate_rouge(
    self,
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]
```

计算ROUGE分数。

##### calculate_classification_metrics()

```python
def calculate_classification_metrics(
    self,
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]
```

计算分类指标（准确率、精确率、召回率、F1分数）。

##### calculate_semantic_similarity()

```python
def calculate_semantic_similarity(
    self,
    text1_list: List[str],
    text2_list: List[str],
    method: str = "cosine"
) -> Dict[str, float]
```

计算语义相似度。

**参数:**
- `method`: 相似度计算方法 ("cosine", "jaccard")

## 质量分析API

### QualityAnalyzer

数据和响应质量分析器。

#### 构造函数

```python
QualityAnalyzer(
    min_length: int = 5,
    max_length: int = 2048,
    length_outlier_threshold: float = 3.0,
    vocab_diversity_threshold: float = 0.5,
    language: str = "zh"
)
```

#### 方法

##### analyze_data_quality()

```python
def analyze_data_quality(
    self,
    dataset: Dataset,
    text_field: str = "text"
) -> DataQualityReport
```

分析数据集质量。

##### analyze_response_quality()

```python
def analyze_response_quality(
    self,
    responses: List[str],
    references: Optional[List[str]] = None
) -> ResponseQualityReport
```

分析响应质量。

##### suggest_improvements()

```python
def suggest_improvements(
    self,
    quality_report: DataQualityReport
) -> List[str]
```

基于质量报告提供改进建议。

##### generate_quality_report()

```python
def generate_quality_report(
    self,
    quality_report: DataQualityReport,
    output_path: str,
    format: str = "html"
) -> None
```

生成质量分析报告。

## 基准管理API

### BenchmarkManager

基准数据集管理和评估。

#### 构造函数

```python
BenchmarkManager(
    benchmark_dir: str = "benchmarks",
    cache_dir: str = ".benchmark_cache",
    download_timeout: int = 300
)
```

#### 方法

##### list_available_benchmarks()

```python
def list_available_benchmarks(self) -> List[str]
```

列出所有可用的基准数据集。

##### load_benchmark()

```python
def load_benchmark(self, benchmark_name: str) -> BenchmarkDataset
```

加载基准数据集。

##### run_clue_evaluation()

```python
def run_clue_evaluation(
    self,
    model: Any,
    tokenizer: Any,
    model_name: str
) -> BenchmarkResult
```

运行CLUE基准评估。

##### run_few_clue_evaluation()

```python
def run_few_clue_evaluation(
    self,
    model: Any,
    tokenizer: Any,
    model_name: str,
    few_shot_examples: int = 5
) -> BenchmarkResult
```

运行FewCLUE少样本评估。

##### run_custom_benchmark()

```python
def run_custom_benchmark(
    self,
    config: BenchmarkConfig,
    model: Any,
    tokenizer: Any,
    model_name: str
) -> BenchmarkResult
```

运行自定义基准测试。

##### compare_benchmark_results()

```python
def compare_benchmark_results(
    self,
    results: List[BenchmarkResult]
) -> Dict[str, Any]
```

对比基准测试结果。

## 实验跟踪API

### ExperimentTracker

实验跟踪和管理。

#### 构造函数

```python
ExperimentTracker(
    experiment_dir: str = "experiments",
    auto_save: bool = True,
    max_history: int = 1000
)
```

#### 方法

##### track_experiment()

```python
def track_experiment(
    self,
    config: ExperimentConfig,
    result: EvaluationResult
) -> str
```

跟踪实验。

**返回:** 实验ID

##### get_experiment()

```python
def get_experiment(self, experiment_id: str) -> Dict[str, Any]
```

获取实验详情。

##### list_experiments()

```python
def list_experiments(
    self,
    filters: Optional[Dict[str, Any]] = None,
    sort_by: str = "timestamp",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]
```

列出实验。

##### compare_experiments()

```python
def compare_experiments(
    self,
    experiment_ids: List[str]
) -> Dict[str, Any]
```

对比实验。

##### generate_leaderboard()

```python
def generate_leaderboard(
    self,
    metric: str = "overall_score",
    limit: int = 10
) -> List[Dict[str, Any]]
```

生成排行榜。

##### export_results()

```python
def export_results(
    self,
    output_path: str,
    format: str = "csv",
    filters: Optional[Dict[str, Any]] = None
) -> str
```

导出实验结果。

## 报告生成API

### ReportGenerator

报告生成器。

#### 构造函数

```python
ReportGenerator(
    template_dir: str = "templates",
    output_dir: str = "reports",
    include_plots: bool = True,
    language: str = "zh"
)
```

#### 方法

##### generate_evaluation_report()

```python
def generate_evaluation_report(
    self,
    result: EvaluationResult,
    format: str = "html"
) -> str
```

生成评估报告。

**参数:**
- `format`: 报告格式 ("html", "json", "pdf")

##### generate_comparison_report()

```python
def generate_comparison_report(
    self,
    results: List[EvaluationResult],
    format: str = "html"
) -> str
```

生成对比报告。

##### generate_benchmark_report()

```python
def generate_benchmark_report(
    self,
    result: BenchmarkResult,
    format: str = "html"
) -> str
```

生成基准测试报告。

##### generate_training_report()

```python
def generate_training_report(
    self,
    training_history: Dict[str, List],
    training_config: Dict[str, Any],
    format: str = "html"
) -> str
```

生成训练报告。

##### generate_latex_table()

```python
def generate_latex_table(
    self,
    results: List[EvaluationResult],
    metrics: List[str]
) -> str
```

生成LaTeX表格。

##### generate_csv_export()

```python
def generate_csv_export(
    self,
    results: List[EvaluationResult]
) -> str
```

生成CSV导出。

## 配置管理API

### ConfigManager

配置文件管理。

#### 构造函数

```python
ConfigManager(
    config_dir: str = "configs",
    default_config_file: str = "default_config.yaml",
    validate_on_load: bool = True
)
```

#### 方法

##### load_config()

```python
def load_config(
    self,
    config_path: str,
    substitute_env_vars: bool = False,
    resolve_inheritance: bool = True
) -> Dict[str, Any]
```

加载配置文件。

##### save_config()

```python
def save_config(
    self,
    config: Dict[str, Any],
    config_path: str,
    format: str = "yaml"
) -> None
```

保存配置文件。

##### validate_config()

```python
def validate_config(self, config: Dict[str, Any]) -> bool
```

验证配置。

##### merge_configs()

```python
def merge_configs(
    self,
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]
```

合并配置。

##### create_evaluation_config()

```python
def create_evaluation_config(
    self,
    config_dict: Dict[str, Any]
) -> EvaluationConfig
```

从字典创建评估配置对象。

## 数据模型

### EvaluationConfig

评估配置数据类。

```python
@dataclass
class EvaluationConfig:
    tasks: List[str]
    metrics: List[str]
    batch_size: int = 8
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    num_samples: Optional[int] = None
    device: str = "cpu"
    memory_optimization: bool = False
```

### DataSplitResult

数据拆分结果。

```python
@dataclass
class DataSplitResult:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    split_info: Dict[str, Any]
    distribution_analysis: Dict[str, Any]
    created_at: datetime
```

### EvaluationResult

评估结果。

```python
@dataclass
class EvaluationResult:
    model_name: str
    evaluation_time: datetime
    metrics: Dict[str, float]
    task_results: Dict[str, TaskResult]
    efficiency_metrics: EfficiencyMetrics
    quality_scores: QualityScores
    config: EvaluationConfig
```

### TaskResult

单个任务的评估结果。

```python
@dataclass
class TaskResult:
    task_name: str
    predictions: List[str]
    references: List[str]
    metrics: Dict[str, float]
    samples: List[EvaluationSample]
    execution_time: float
```

### EfficiencyMetrics

效率指标。

```python
@dataclass
class EfficiencyMetrics:
    inference_latency: float  # ms
    throughput: float  # tokens/s
    memory_usage: float  # GB
    model_size: float  # MB
    flops: Optional[int] = None
```

### QualityScores

质量分数。

```python
@dataclass
class QualityScores:
    fluency: float
    coherence: float
    relevance: float
    factuality: float
    overall: float
```

### BenchmarkResult

基准测试结果。

```python
@dataclass
class BenchmarkResult:
    benchmark_name: str
    model_name: str
    task_results: Dict[str, Dict[str, float]]
    overall_score: float
    metadata: Dict[str, Any]
    evaluation_time: datetime
```

### ExperimentConfig

实验配置。

```python
@dataclass
class ExperimentConfig:
    model_name: str
    dataset_name: str
    hyperparameters: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
```

### DataQualityReport

数据质量报告。

```python
@dataclass
class DataQualityReport:
    total_samples: int
    length_stats: Dict[str, float]
    vocab_diversity: Dict[str, float]
    class_distribution: Dict[str, Any]
    quality_issues: List[Dict[str, Any]]
    quality_score: float
    recommendations: List[str]
```

### ResponseQualityReport

响应质量报告。

```python
@dataclass
class ResponseQualityReport:
    fluency_score: float
    coherence_score: float
    relevance_score: float
    factuality_score: float
    overall_score: float
    detailed_analysis: Dict[str, Any]
```

## 异常类

### EvaluationError

评估相关的基础异常类。

```python
class EvaluationError(Exception):
    """评估系统基础异常"""
    pass
```

### DataSplitError

数据拆分异常。

```python
class DataSplitError(EvaluationError):
    """数据拆分异常"""
    pass
```

### MetricsCalculationError

指标计算异常。

```python
class MetricsCalculationError(EvaluationError):
    """指标计算异常"""
    pass
```

### BenchmarkError

基准测试异常。

```python
class BenchmarkError(EvaluationError):
    """基准测试异常"""
    pass
```

### ConfigurationError

配置异常。

```python
class ConfigurationError(EvaluationError):
    """配置异常"""
    pass
```

## 使用示例

### 完整的评估流程

```python
from evaluation import (
    DataSplitter, EvaluationEngine, EvaluationConfig,
    ExperimentTracker, ReportGenerator
)

# 1. 数据拆分
splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
split_result = splitter.split_data(dataset, "data/splits")

# 2. 配置评估
config = EvaluationConfig(
    tasks=["classification"],
    metrics=["accuracy", "f1"],
    batch_size=8
)

# 3. 运行评估
engine = EvaluationEngine(config)
result = engine.evaluate_model(
    model=model,
    tokenizer=tokenizer,
    datasets={"classification": split_result.test_dataset},
    model_name="my_model"
)

# 4. 跟踪实验
tracker = ExperimentTracker()
experiment_id = tracker.track_experiment(experiment_config, result)

# 5. 生成报告
generator = ReportGenerator()
report_path = generator.generate_evaluation_report(result)
```

## 版本兼容性

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- Datasets 2.0+

## 更新日志

### v1.0.0
- 初始版本发布
- 支持基本的数据拆分和模型评估功能

### v1.1.0
- 添加基准测试支持
- 增强实验跟踪功能
- 改进报告生成

### v1.2.0
- 添加质量分析功能
- 支持自定义评估任务
- 优化性能和内存使用

---

*本API文档会随着系统更新而持续维护。*
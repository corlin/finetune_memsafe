"""
测试配置文件

提供测试夹具和共享配置。
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import json

from datasets import Dataset

# 添加src路径以便导入评估模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from evaluation import (
        DataSplitter, QualityAnalyzer, MetricsCalculator, 
        EvaluationConfig, ExperimentTracker
    )
except ImportError:
    # 如果导入失败，创建模拟类
    class DataSplitter:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class QualityAnalyzer:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class MetricsCalculator:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class EvaluationConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class ExperimentTracker:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


@pytest.fixture
def temp_dir():
    """临时目录夹具"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataset():
    """样本数据集夹具"""
    data = [
        {"text": "这是第一个测试样本", "label": "positive"},
        {"text": "这是第二个测试样本", "label": "negative"},
        {"text": "这是第三个测试样本", "label": "positive"},
        {"text": "这是第四个测试样本", "label": "negative"},
        {"text": "这是第五个测试样本", "label": "neutral"},
        {"text": "这是第六个测试样本", "label": "positive"},
        {"text": "这是第七个测试样本", "label": "negative"},
        {"text": "这是第八个测试样本", "label": "neutral"},
        {"text": "这是第九个测试样本", "label": "positive"},
        {"text": "这是第十个测试样本", "label": "negative"}
    ]
    return Dataset.from_list(data)


@pytest.fixture
def small_dataset():
    """小数据集夹具"""
    data = [
        {"text": "短文本1", "label": "A"},
        {"text": "短文本2", "label": "B"},
        {"text": "短文本3", "label": "A"}
    ]
    return Dataset.from_list(data)


@pytest.fixture
def evaluation_config():
    """评估配置夹具"""
    return EvaluationConfig(
        tasks=["text_generation", "classification"],
        metrics=["bleu", "accuracy"],
        batch_size=2,
        max_length=128,
        num_samples=5
    )


@pytest.fixture
def data_splitter():
    """数据拆分器夹具"""
    return DataSplitter(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42,
        min_samples_per_split=1
    )


@pytest.fixture
def quality_analyzer():
    """质量分析器夹具"""
    return QualityAnalyzer(
        min_length=1,
        max_length=1000,
        length_outlier_threshold=2.0
    )


@pytest.fixture
def metrics_calculator():
    """指标计算器夹具"""
    return MetricsCalculator(language="zh", device="cpu")


@pytest.fixture
def experiment_tracker(temp_dir):
    """实验跟踪器夹具"""
    return ExperimentTracker(experiment_dir=str(temp_dir / "experiments"))


@pytest.fixture
def sample_predictions():
    """样本预测结果夹具"""
    return [
        "这是预测结果1",
        "这是预测结果2", 
        "这是预测结果3",
        "这是预测结果4",
        "这是预测结果5"
    ]


@pytest.fixture
def sample_references():
    """样本参考答案夹具"""
    return [
        "这是参考答案1",
        "这是参考答案2",
        "这是参考答案3", 
        "这是参考答案4",
        "这是参考答案5"
    ]


@pytest.fixture
def sample_inputs():
    """样本输入夹具"""
    return [
        "输入文本1",
        "输入文本2",
        "输入文本3",
        "输入文本4",
        "输入文本5"
    ]


@pytest.fixture
def config_file(temp_dir):
    """配置文件夹具"""
    config = {
        "data_split": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42
        },
        "evaluation": {
            "tasks": ["text_generation"],
            "metrics": ["bleu", "rouge"],
            "batch_size": 4,
            "num_samples": 10
        }
    }
    
    config_path = temp_dir / "test_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f)
    
    return config_path


@pytest.fixture
def mock_model():
    """模拟模型夹具"""
    class MockModel:
        def __init__(self):
            self.config = type('Config', (), {'is_encoder_decoder': False})()
        
        def generate(self, **kwargs):
            # 模拟生成结果
            import torch
            batch_size = kwargs['input_ids'].shape[0]
            seq_len = kwargs.get('max_length', 50)
            return torch.randint(0, 1000, (batch_size, seq_len))
        
        def parameters(self):
            import torch
            return [torch.randn(100, 100), torch.randn(50)]
    
    return MockModel()


@pytest.fixture
def mock_tokenizer():
    """模拟分词器夹具"""
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
        
        def __call__(self, texts, **kwargs):
            import torch
            if isinstance(texts, str):
                texts = [texts]
            
            # 模拟编码结果
            batch_size = len(texts)
            seq_len = kwargs.get('max_length', 50)
            
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len)
            }
        
        def encode(self, text):
            return list(range(len(text.split())))
        
        def decode(self, token_ids, **kwargs):
            return f"decoded_text_{len(token_ids)}"
        
        def batch_decode(self, batch_token_ids, **kwargs):
            return [f"decoded_text_{i}" for i in range(len(batch_token_ids))]
    
    return MockTokenizer()


# 测试数据常量
TEST_TEXTS = [
    "这是一个测试文本",
    "另一个测试样本",
    "第三个测试数据",
    "最后一个测试文本"
]

TEST_LABELS = ["positive", "negative", "neutral", "positive"]

TEST_PREDICTIONS = [
    "预测文本1",
    "预测文本2", 
    "预测文本3",
    "预测文本4"
]

TEST_REFERENCES = [
    "参考文本1",
    "参考文本2",
    "参考文本3", 
    "参考文本4"
]


# 测试工具函数
def create_test_dataset(size: int = 10) -> Dataset:
    """创建测试数据集"""
    data = []
    for i in range(size):
        data.append({
            "text": f"测试文本{i+1}",
            "label": ["positive", "negative", "neutral"][i % 3]
        })
    return Dataset.from_list(data)


def assert_file_exists(file_path: Path):
    """断言文件存在"""
    assert file_path.exists(), f"文件不存在: {file_path}"


def assert_dict_contains_keys(dictionary: Dict[str, Any], keys: List[str]):
    """断言字典包含指定键"""
    for key in keys:
        assert key in dictionary, f"字典缺少键: {key}"


def assert_metrics_valid(metrics: Dict[str, float]):
    """断言指标有效"""
    for metric_name, value in metrics.items():
        assert isinstance(value, (int, float)), f"指标 {metric_name} 不是数值类型"
        assert 0 <= value <= 1 or metric_name in ["perplexity"], f"指标 {metric_name} 值超出范围: {value}"
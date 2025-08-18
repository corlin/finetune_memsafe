#!/usr/bin/env python3
"""
基本使用示例

演示数据拆分和评估系统的基本用法。
"""

import sys
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import Dataset
from evaluation import (
    DataSplitter, EvaluationEngine, EvaluationConfig,
    ExperimentTracker, ReportGenerator
)
from evaluation.data_models import ExperimentConfig


def create_sample_data():
    """创建示例数据集"""
    data = [
        {"text": "这是一个正面的评论", "label": "positive"},
        {"text": "这个产品很好用", "label": "positive"},
        {"text": "质量不错，推荐购买", "label": "positive"},
        {"text": "这个产品不太好", "label": "negative"},
        {"text": "质量很差，不推荐", "label": "negative"},
        {"text": "完全不值这个价格", "label": "negative"},
        {"text": "还可以，一般般", "label": "neutral"},
        {"text": "没什么特别的", "label": "neutral"},
        {"text": "普通的产品", "label": "neutral"},
        {"text": "价格合理，质量还行", "label": "neutral"},
    ] * 20  # 重复以增加数据量
    
    return Dataset.from_list(data)


def mock_model_and_tokenizer():
    """创建模拟的模型和分词器"""
    class MockModel:
        def __init__(self):
            self.config = type('Config', (), {'is_encoder_decoder': False})()
        
        def generate(self, **kwargs):
            # 模拟生成结果
            import torch
            batch_size = kwargs['input_ids'].shape[0]
            seq_len = kwargs.get('max_length', 50)
            return torch.randint(0, 1000, (batch_size, seq_len))
        
        def __call__(self, **kwargs):
            # 模拟分类输出
            import torch
            batch_size = kwargs['input_ids'].shape[0]
            return type('Output', (), {
                'logits': torch.randn(batch_size, 3)  # 3个类别
            })()
    
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
        
        def __call__(self, texts, **kwargs):
            import torch
            if isinstance(texts, str):
                texts = [texts]
            
            batch_size = len(texts)
            seq_len = kwargs.get('max_length', 50)
            
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len)
            }
        
        def decode(self, token_ids, **kwargs):
            return f"decoded_text_{len(token_ids)}"
        
        def batch_decode(self, batch_token_ids, **kwargs):
            return [f"decoded_text_{i}" for i in range(len(batch_token_ids))]
    
    return MockModel(), MockTokenizer()


def main():
    """主函数"""
    print("🚀 开始基本使用示例...")
    
    # 1. 创建示例数据
    print("\n📊 创建示例数据集...")
    dataset = create_sample_data()
    print(f"数据集大小: {len(dataset)}")
    print(f"标签分布: {dataset['label'][:10]}")
    
    # 2. 数据拆分
    print("\n✂️ 执行数据拆分...")
    splitter = DataSplitter(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        stratify_by="label",
        min_samples_per_split=10
    )
    
    split_result = splitter.split_data(dataset, "output/basic_example_splits")
    
    print(f"训练集大小: {len(split_result.train_dataset)}")
    print(f"验证集大小: {len(split_result.val_dataset)}")
    print(f"测试集大小: {len(split_result.test_dataset)}")
    print(f"数据分布一致性分数: {split_result.distribution_analysis.consistency_score}")
    
    # 3. 创建模拟模型
    print("\n🤖 创建模拟模型和分词器...")
    model, tokenizer = mock_model_and_tokenizer()
    
    # 4. 配置评估
    print("\n⚙️ 配置评估参数...")
    config = EvaluationConfig(
        tasks=["classification"],
        metrics=["accuracy", "precision", "recall", "f1"],
        batch_size=4,
        num_samples=50  # 使用较少样本以加快演示
    )
    
    # 5. 运行评估
    print("\n🧪 运行模型评估...")
    engine = EvaluationEngine(config)
    
    # 准备评估数据集
    datasets = {
        "classification": split_result.test_dataset
    }
    
    # 模拟推理函数
    def mock_inference_function(inputs):
        """模拟推理函数"""
        import random
        labels = ["positive", "negative", "neutral"]
        return [random.choice(labels) for _ in inputs]
    
    # 替换推理函数（在实际使用中不需要这步）
    engine._create_inference_function = lambda model, tokenizer: mock_inference_function
    
    result = engine.evaluate_model(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        model_name="basic_example_model"
    )
    
    print("评估结果:")
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 6. 实验跟踪
    print("\n📝 跟踪实验...")
    tracker = ExperimentTracker(experiment_dir="output/basic_example_experiments")
    
    experiment_config = ExperimentConfig(
        experiment_name="basic_usage_example",
        model_config={
            "model_name": "basic_example_model",
            "model_type": "mock_model"
        },
        training_config={
            "batch_size": config.batch_size,
            "num_samples": config.num_samples
        },
        evaluation_config=config,
        data_config={
            "dataset_name": "sample_sentiment_data",
            "data_size": len(dataset)
        },
        tags=["example", "basic_usage"],
        description="基本使用示例实验"
    )
    
    experiment_id = tracker.track_experiment(experiment_config, result)
    print(f"实验ID: {experiment_id}")
    
    # 7. 生成报告
    print("\n📊 生成评估报告...")
    generator = ReportGenerator(
        output_dir="output/basic_example_reports"
    )
    
    # 生成HTML报告
    html_report = generator.generate_evaluation_report(result, format_type="html")
    print(f"HTML报告: {html_report}")
    
    # 生成JSON报告
    json_report = generator.generate_evaluation_report(result, format_type="json")
    print(f"JSON报告: {json_report}")
    
    # 8. 查看实验历史
    print("\n📋 查看实验历史...")
    experiments = tracker.list_experiments()
    print(f"总实验数: {len(experiments)}")
    
    if experiments:
        latest_exp = experiments[0]
        # 获取完整的实验信息以访问metadata
        full_exp = tracker.get_experiment(latest_exp['id'])
        if full_exp and 'metadata' in full_exp:
            model_name = full_exp['metadata'].get('model_config', {}).get('model_name', '未知模型')
        else:
            model_name = '未知模型'
        print(f"最新实验: {model_name}")
        print(f"实验时间: {latest_exp['created_at']}")
    
    # 9. 导出结果
    print("\n💾 导出实验结果...")
    csv_path = tracker.export_results("output/basic_example_results.csv", format="csv")
    print(f"CSV导出: {csv_path}")
    
    print("\n✅ 基本使用示例完成！")
    print("\n📁 输出文件位置:")
    print("  - 数据拆分: output/basic_example_splits/")
    print("  - 实验记录: output/basic_example_experiments/")
    print("  - 评估报告: output/basic_example_reports/")
    print("  - 结果导出: output/basic_example_results.csv")


if __name__ == "__main__":
    # 创建输出目录
    Path("output").mkdir(exist_ok=True)
    
    try:
        main()
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

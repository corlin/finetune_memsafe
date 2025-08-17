#!/usr/bin/env python3
"""
高级评估示例

演示高级评估功能，包括多模型对比、基准测试、质量分析等。
"""

import sys
from pathlib import Path
import time
from datetime import datetime

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import Dataset
from evaluation import (
    DataSplitter, EvaluationEngine, EvaluationConfig,
    QualityAnalyzer, BenchmarkManager, ExperimentTracker,
    ReportGenerator, MetricsCalculator
)
from evaluation.data_models import ExperimentConfig, BenchmarkConfig


def create_diverse_dataset():
    """创建多样化的数据集"""
    data = []
    
    # 文本生成数据
    generation_data = [
        {"input": "请介绍一下人工智能", "output": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"},
        {"input": "什么是机器学习？", "output": "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习模式。"},
        {"input": "深度学习的应用有哪些？", "output": "深度学习广泛应用于图像识别、自然语言处理、语音识别等领域。"},
    ] * 30
    
    # 分类数据
    classification_data = [
        {"text": "这个产品质量很好，非常满意", "label": "positive"},
        {"text": "服务态度不错，会再次购买", "label": "positive"},
        {"text": "价格合理，性价比高", "label": "positive"},
        {"text": "产品有缺陷，不推荐", "label": "negative"},
        {"text": "客服态度很差", "label": "negative"},
        {"text": "完全不值这个价格", "label": "negative"},
        {"text": "还可以，没什么特别的", "label": "neutral"},
        {"text": "普通的产品，一般般", "label": "neutral"},
    ] * 25
    
    # 问答数据
    qa_data = [
        {"question": "北京是哪个国家的首都？", "answer": "中国", "context": "北京是中华人民共和国的首都。"},
        {"question": "一年有多少个月？", "answer": "12个月", "context": "一年通常有12个月。"},
        {"question": "地球有几个卫星？", "answer": "1个", "context": "地球只有一个天然卫星，即月球。"},
    ] * 35
    
    return {
        "generation": Dataset.from_list(generation_data),
        "classification": Dataset.from_list(classification_data),
        "qa": Dataset.from_list(qa_data)
    }


def create_mock_models():
    """创建多个模拟模型"""
    import torch
    import random
    
    class MockModel:
        def __init__(self, name, performance_level=0.8):
            self.name = name
            self.performance_level = performance_level
            self.config = type('Config', (), {'is_encoder_decoder': False})()
        
        def generate(self, **kwargs):
            batch_size = kwargs['input_ids'].shape[0]
            seq_len = kwargs.get('max_length', 50)
            return torch.randint(0, 1000, (batch_size, seq_len))
        
        def __call__(self, **kwargs):
            batch_size = kwargs['input_ids'].shape[0]
            # 根据性能水平调整输出
            logits = torch.randn(batch_size, 3) * self.performance_level
            return type('Output', (), {'logits': logits})()
    
    class MockTokenizer:
        def __init__(self, name):
            self.name = name
            self.pad_token_id = 0
        
        def __call__(self, texts, **kwargs):
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
    
    # 创建不同性能水平的模型
    models = [
        {
            "model": MockModel("baseline_model", 0.7),
            "tokenizer": MockTokenizer("baseline_tokenizer"),
            "name": "baseline_model"
        },
        {
            "model": MockModel("improved_model", 0.85),
            "tokenizer": MockTokenizer("improved_tokenizer"),
            "name": "improved_model"
        },
        {
            "model": MockModel("advanced_model", 0.9),
            "tokenizer": MockTokenizer("advanced_tokenizer"),
            "name": "advanced_model"
        }
    ]
    
    return models


def demonstrate_quality_analysis(datasets):
    """演示质量分析功能"""
    print("\n🔍 执行数据质量分析...")
    
    analyzer = QualityAnalyzer(
        min_length=5,
        max_length=1000,
        language="zh"
    )
    
    # 分析分类数据集的质量
    quality_report = analyzer.analyze_data_quality(
        datasets["classification"], 
        text_field="text"
    )
    
    print(f"数据集总样本数: {quality_report.total_samples}")
    print(f"质量分数: {quality_report.quality_score:.3f}")
    print(f"发现的问题数: {len(quality_report.quality_issues)}")
    
    # 生成质量报告
    analyzer.generate_quality_report(
        quality_report,
        "output/advanced_example_quality_report.html",
        format="html"
    )
    print("质量分析报告已生成: output/advanced_example_quality_report.html")
    
    # 获取改进建议
    suggestions = analyzer.suggest_improvements(quality_report)
    print("改进建议:")
    for suggestion in suggestions[:3]:  # 显示前3个建议
        print(f"  - {suggestion}")
    
    return quality_report


def demonstrate_multi_model_evaluation(models, datasets):
    """演示多模型评估"""
    print("\n🏆 执行多模型对比评估...")
    
    config = EvaluationConfig(
        tasks=["classification"],
        metrics=["accuracy", "precision", "recall", "f1"],
        batch_size=4,
        num_samples=100
    )
    
    engine = EvaluationEngine(config, max_workers=2)
    
    # 模拟推理函数
    def create_mock_inference(performance_level):
        def mock_inference(inputs):
            import random
            labels = ["positive", "negative", "neutral"]
            # 根据性能水平调整准确率
            correct_rate = performance_level
            results = []
            for _ in inputs:
                if random.random() < correct_rate:
                    # 返回"正确"答案（简化处理）
                    results.append(random.choice(labels))
                else:
                    # 返回"错误"答案
                    results.append(random.choice(labels))
            return results
        return mock_inference
    
    # 为每个模型设置不同的推理函数
    results = []
    for model_info in models:
        model = model_info["model"]
        performance = model.performance_level
        
        # 临时替换推理函数
        original_create_func = engine._create_inference_function
        engine._create_inference_function = lambda m, t: create_mock_inference(performance)
        
        result = engine.evaluate_model(
            model=model,
            tokenizer=model_info["tokenizer"],
            datasets={"classification": datasets["classification"]},
            model_name=model_info["name"]
        )
        results.append(result)
        
        print(f"{model_info['name']} - 准确率: {result.metrics.get('accuracy', 0):.3f}")
        
        # 恢复原函数
        engine._create_inference_function = original_create_func
    
    return results


def demonstrate_benchmark_evaluation(models):
    """演示基准测试评估"""
    print("\n📊 执行基准测试评估...")
    
    # 创建自定义基准配置
    custom_benchmark_config = BenchmarkConfig(
        name="sentiment_benchmark",
        dataset_path="custom_sentiment_data.json",
        tasks=["sentiment_classification"],
        evaluation_protocol="standard",
        metrics=["accuracy", "f1"]
    )
    
    # 模拟基准管理器
    class MockBenchmarkManager:
        def __init__(self):
            pass
        
        def list_available_benchmarks(self):
            return ["clue", "few_clue", "c_eval", "custom_sentiment"]
        
        def run_custom_benchmark(self, config, model, tokenizer, model_name):
            # 模拟基准测试结果
            import random
            from evaluation.data_models import BenchmarkResult
            
            task_results = {}
            for task in config.tasks:
                task_results[task] = {
                    "accuracy": random.uniform(0.7, 0.95),
                    "f1": random.uniform(0.65, 0.9)
                }
            
            overall_score = sum(
                sum(scores.values()) / len(scores) 
                for scores in task_results.values()
            ) / len(task_results)
            
            return BenchmarkResult(
                benchmark_name=config.name,
                model_name=model_name,
                task_results=task_results,
                overall_score=overall_score,
                metadata={"custom": True},
                evaluation_time=datetime.now()
            )
    
    benchmark_manager = MockBenchmarkManager()
    
    # 运行基准测试
    benchmark_results = []
    for model_info in models:
        result = benchmark_manager.run_custom_benchmark(
            custom_benchmark_config,
            model_info["model"],
            model_info["tokenizer"],
            model_info["name"]
        )
        benchmark_results.append(result)
        
        print(f"{model_info['name']} - 基准分数: {result.overall_score:.3f}")
    
    return benchmark_results


def demonstrate_advanced_reporting(evaluation_results, benchmark_results):
    """演示高级报告生成"""
    print("\n📈 生成高级报告...")
    
    generator = ReportGenerator(
        output_dir="output/advanced_example_reports",
        include_plots=True,
        language="zh"
    )
    
    # 生成多模型对比报告
    comparison_report = generator.generate_comparison_report(
        evaluation_results,
        format="html"
    )
    print(f"对比报告: {comparison_report}")
    
    # 生成基准测试报告
    for benchmark_result in benchmark_results:
        benchmark_report = generator.generate_benchmark_report(
            benchmark_result,
            format="html"
        )
        print(f"基准报告 ({benchmark_result.model_name}): {benchmark_report}")
    
    # 生成LaTeX表格
    latex_table = generator.generate_latex_table(
        evaluation_results,
        metrics=["accuracy", "f1"]
    )
    print(f"LaTeX表格: {latex_table}")
    
    # 生成CSV导出
    csv_export = generator.generate_csv_export(evaluation_results)
    print(f"CSV导出: {csv_export}")
    
    return {
        "comparison_report": comparison_report,
        "latex_table": latex_table,
        "csv_export": csv_export
    }


def demonstrate_experiment_analysis(evaluation_results, tracker):
    """演示实验分析功能"""
    print("\n📊 执行实验分析...")
    
    # 跟踪所有实验
    experiment_ids = []
    for i, result in enumerate(evaluation_results):
        experiment_config = ExperimentConfig(
            model_name=result.model_name,
            dataset_name="advanced_example_dataset",
            hyperparameters={
                "batch_size": 4,
                "num_samples": 100,
                "model_version": f"v1.{i}"
            },
            tags=["advanced_example", f"model_{i}"],
            description=f"高级示例实验 - {result.model_name}"
        )
        
        exp_id = tracker.track_experiment(experiment_config, result)
        experiment_ids.append(exp_id)
    
    # 生成排行榜
    leaderboard = tracker.generate_leaderboard(metric="accuracy")
    print("\n🏆 模型排行榜 (按准确率):")
    for i, entry in enumerate(leaderboard[:3], 1):
        print(f"  {i}. {entry['model_name']}: {entry['score']:.4f}")
    
    # 对比实验
    comparison = tracker.compare_experiments(experiment_ids)
    print(f"\n最佳模型: {comparison['best_model']['model_name']}")
    
    # 获取实验统计
    stats = tracker.get_experiment_statistics()
    print(f"实验统计:")
    print(f"  总实验数: {stats['total_experiments']}")
    print(f"  平均准确率: {stats.get('avg_accuracy', 'N/A')}")
    print(f"  最佳准确率: {stats.get('best_accuracy', 'N/A')}")
    
    return {
        "leaderboard": leaderboard,
        "comparison": comparison,
        "statistics": stats
    }


def demonstrate_metrics_calculation():
    """演示指标计算功能"""
    print("\n🧮 演示指标计算...")
    
    calculator = MetricsCalculator(language="zh")
    
    # 示例预测和参考文本
    predictions = [
        "这是一个很好的产品",
        "质量不错，推荐购买",
        "价格有点贵但值得"
    ]
    
    references = [
        "这是一个优秀的产品",
        "质量很好，值得推荐",
        "虽然价格较高但物有所值"
    ]
    
    # 计算BLEU分数
    bleu_scores = calculator.calculate_bleu(predictions, references)
    print(f"BLEU分数: {bleu_scores['bleu']:.4f}")
    
    # 计算ROUGE分数
    rouge_scores = calculator.calculate_rouge(predictions, references)
    print(f"ROUGE-L分数: {rouge_scores.get('rougeL', 'N/A')}")
    
    # 计算语义相似度
    similarity_scores = calculator.calculate_semantic_similarity(
        predictions, references, method="cosine"
    )
    print(f"语义相似度: {similarity_scores['cosine_similarity']:.4f}")
    
    # 分类指标示例
    class_predictions = ["positive", "negative", "positive", "neutral"]
    class_references = ["positive", "positive", "negative", "neutral"]
    
    class_metrics = calculator.calculate_classification_metrics(
        class_predictions, class_references
    )
    print(f"分类准确率: {class_metrics['accuracy']:.4f}")
    print(f"F1分数: {class_metrics['f1']:.4f}")
    
    return {
        "bleu": bleu_scores,
        "rouge": rouge_scores,
        "similarity": similarity_scores,
        "classification": class_metrics
    }


def main():
    """主函数"""
    print("🚀 开始高级评估示例...")
    
    # 创建输出目录
    Path("output").mkdir(exist_ok=True)
    Path("output/advanced_example_reports").mkdir(exist_ok=True)
    
    # 1. 创建多样化数据集
    print("\n📊 创建多样化数据集...")
    datasets = create_diverse_dataset()
    print(f"生成数据集: {len(datasets)} 个任务")
    for task, dataset in datasets.items():
        print(f"  {task}: {len(dataset)} 个样本")
    
    # 2. 数据质量分析
    quality_report = demonstrate_quality_analysis(datasets)
    
    # 3. 创建多个模型
    print("\n🤖 创建多个模拟模型...")
    models = create_mock_models()
    print(f"创建了 {len(models)} 个模型")
    
    # 4. 多模型评估
    evaluation_results = demonstrate_multi_model_evaluation(models, datasets)
    
    # 5. 基准测试评估
    benchmark_results = demonstrate_benchmark_evaluation(models)
    
    # 6. 指标计算演示
    metrics_demo = demonstrate_metrics_calculation()
    
    # 7. 实验跟踪和分析
    tracker = ExperimentTracker(
        experiment_dir="output/advanced_example_experiments"
    )
    experiment_analysis = demonstrate_experiment_analysis(evaluation_results, tracker)
    
    # 8. 高级报告生成
    reports = demonstrate_advanced_reporting(evaluation_results, benchmark_results)
    
    # 9. 性能分析
    print("\n⚡ 性能分析...")
    from evaluation.efficiency_analyzer import EfficiencyAnalyzer
    
    try:
        efficiency_analyzer = EfficiencyAnalyzer()
        
        # 模拟性能测量
        def mock_inference_func(inputs):
            time.sleep(0.01)  # 模拟推理时间
            return ["result"] * len(inputs)
        
        latency = efficiency_analyzer.measure_inference_latency(
            mock_inference_func,
            ["test"] * 10,
            batch_size=2
        )
        print(f"平均推理延迟: {latency:.3f}ms")
        
    except ImportError:
        print("效率分析器不可用，跳过性能分析")
    
    # 10. 总结
    print("\n✅ 高级评估示例完成！")
    print("\n📊 生成的文件:")
    print("  - 质量分析报告: output/advanced_example_quality_report.html")
    print("  - 评估报告: output/advanced_example_reports/")
    print("  - 实验记录: output/advanced_example_experiments/")
    print("  - 结果导出: output/advanced_example_reports/")
    
    print("\n📈 关键结果:")
    print(f"  - 最佳模型: {experiment_analysis['comparison']['best_model']['model_name']}")
    print(f"  - 数据质量分数: {quality_report.quality_score:.3f}")
    print(f"  - 总实验数: {experiment_analysis['statistics']['total_experiments']}")
    
    return {
        "datasets": datasets,
        "models": models,
        "evaluation_results": evaluation_results,
        "benchmark_results": benchmark_results,
        "quality_report": quality_report,
        "experiment_analysis": experiment_analysis,
        "reports": reports,
        "metrics_demo": metrics_demo
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 所有高级功能演示完成！")
    except Exception as e:
        print(f"❌ 高级示例运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
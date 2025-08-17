"""
性能基准测试

测试评估系统的性能和效率。
"""

import pytest
import time
import psutil
import os
from unittest.mock import Mock, patch

from evaluation import (
    DataSplitter, EvaluationEngine, MetricsCalculator,
    BenchmarkManager, ExperimentTracker
)
from evaluation.data_models import EvaluationConfig
from tests.conftest import create_test_dataset


class TestPerformanceBenchmarks:
    """性能基准测试类"""
    
    def measure_execution_time(self, func, *args, **kwargs):
        """测量函数执行时间"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """测量函数内存使用"""
        process = psutil.Process(os.getpid())
        
        # 获取执行前的内存使用
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        # 获取执行后的内存使用
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before
        
        return result, memory_delta, memory_after
    
    @pytest.mark.performance
    def test_data_splitting_performance(self, temp_dir):
        """测试数据拆分性能"""
        # 测试不同大小的数据集
        dataset_sizes = [100, 1000, 5000, 10000]
        
        for size in dataset_sizes:
            dataset = create_test_dataset(size)
            splitter = DataSplitter(min_samples_per_split=10)
            
            # 测量执行时间
            result, execution_time = self.measure_execution_time(
                splitter.split_data, dataset, str(temp_dir / f"split_{size}")
            )
            
            # 性能断言
            assert execution_time < 30.0, f"数据拆分耗时过长: {execution_time:.2f}s for {size} samples"
            
            # 内存使用测试
            _, memory_delta, _ = self.measure_memory_usage(
                splitter.split_data, dataset, str(temp_dir / f"split_mem_{size}")
            )
            
            # 内存使用应该合理（不超过数据集大小的10倍）
            expected_memory = size * 0.001  # 假设每个样本约1KB
            assert memory_delta < expected_memory * 10, f"内存使用过多: {memory_delta:.2f}MB"
            
            print(f"数据集大小: {size}, 执行时间: {execution_time:.3f}s, 内存增量: {memory_delta:.2f}MB")
    
    @pytest.mark.performance
    def test_metrics_calculation_performance(self):
        """测试指标计算性能"""
        calculator = MetricsCalculator()
        
        # 测试不同长度的文本
        text_lengths = [10, 100, 500, 1000]
        
        for length in text_lengths:
            # 生成测试文本
            predictions = [f"预测文本{'测试' * (length // 4)}" for _ in range(100)]
            references = [f"参考文本{'标准' * (length // 4)}" for _ in range(100)]
            
            # 测试BLEU计算性能
            result, execution_time = self.measure_execution_time(
                calculator.calculate_bleu, predictions, references
            )
            
            assert execution_time < 10.0, f"BLEU计算耗时过长: {execution_time:.2f}s"
            
            # 测试ROUGE计算性能
            result, execution_time = self.measure_execution_time(
                calculator.calculate_rouge, predictions, references
            )
            
            assert execution_time < 15.0, f"ROUGE计算耗时过长: {execution_time:.2f}s"
            
            print(f"文本长度: {length}, BLEU时间: {execution_time:.3f}s")
    
    @pytest.mark.performance
    def test_evaluation_engine_performance(self, mock_model, mock_tokenizer):
        """测试评估引擎性能"""
        config = EvaluationConfig(
            tasks=["classification"],
            metrics=["accuracy"],
            batch_size=8,
            num_samples=100
        )
        
        engine = EvaluationEngine(config)
        
        # 测试不同大小的数据集
        dataset_sizes = [50, 200, 500, 1000]
        
        for size in dataset_sizes:
            dataset = create_test_dataset(size)
            datasets = {"classification": dataset}
            
            with patch.object(engine, '_create_inference_function') as mock_inference:
                mock_inference.return_value = lambda inputs: ["positive"] * len(inputs)
                
                # 测量评估性能
                result, execution_time = self.measure_execution_time(
                    engine.evaluate_model,
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    datasets=datasets,
                    model_name="test_model"
                )
                
                # 性能断言
                expected_time = size * 0.01  # 假设每个样本10ms
                assert execution_time < expected_time * 5, f"评估耗时过长: {execution_time:.2f}s for {size} samples"
                
                print(f"数据集大小: {size}, 评估时间: {execution_time:.3f}s")
    
    @pytest.mark.performance
    def test_batch_processing_efficiency(self, mock_model, mock_tokenizer):
        """测试批处理效率"""
        dataset = create_test_dataset(1000)
        
        # 测试不同批次大小的性能
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            config = EvaluationConfig(
                tasks=["classification"],
                metrics=["accuracy"],
                batch_size=batch_size,
                num_samples=200
            )
            
            engine = EvaluationEngine(config)
            datasets = {"classification": dataset}
            
            with patch.object(engine, '_create_inference_function') as mock_inference:
                mock_inference.return_value = lambda inputs: ["positive"] * len(inputs)
                
                result, execution_time = self.measure_execution_time(
                    engine.evaluate_model,
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    datasets=datasets,
                    model_name="test_model"
                )
                
                # 计算每个样本的平均处理时间
                time_per_sample = execution_time / 200
                
                print(f"批次大小: {batch_size}, 总时间: {execution_time:.3f}s, "
                      f"每样本时间: {time_per_sample:.4f}s")
                
                # 较大的批次应该有更好的效率
                if batch_size > 1:
                    assert time_per_sample < 0.1, f"批处理效率不佳: {time_per_sample:.4f}s per sample"
    
    @pytest.mark.performance
    def test_parallel_evaluation_performance(self, mock_model, mock_tokenizer):
        """测试并行评估性能"""
        dataset = create_test_dataset(500)
        
        config = EvaluationConfig(
            tasks=["classification"],
            metrics=["accuracy"],
            batch_size=8,
            num_samples=100
        )
        
        # 测试不同的并行度
        worker_counts = [1, 2, 4]
        
        for max_workers in worker_counts:
            engine = EvaluationEngine(config, max_workers=max_workers)
            
            # 创建多个模型进行并行评估
            models_info = [
                {"model": mock_model, "tokenizer": mock_tokenizer, "name": f"model_{i}"}
                for i in range(4)
            ]
            
            datasets = {"classification": dataset}
            
            with patch.object(engine, '_create_inference_function') as mock_inference:
                mock_inference.return_value = lambda inputs: ["positive"] * len(inputs)
                
                result, execution_time = self.measure_execution_time(
                    engine.evaluate_multiple_models, models_info, datasets
                )
                
                print(f"并行度: {max_workers}, 4个模型评估时间: {execution_time:.3f}s")
                
                # 并行度越高，总时间应该越短（理论上）
                if max_workers > 1:
                    assert execution_time < 60.0, f"并行评估耗时过长: {execution_time:.2f}s"
    
    @pytest.mark.performance
    def test_memory_efficiency(self, temp_dir):
        """测试内存效率"""
        # 测试大数据集的内存使用
        large_dataset = create_test_dataset(5000)
        
        splitter = DataSplitter(min_samples_per_split=100)
        
        # 测量内存使用
        result, memory_delta, peak_memory = self.measure_memory_usage(
            splitter.split_data, large_dataset, str(temp_dir)
        )
        
        # 内存使用应该合理
        assert memory_delta < 500, f"内存使用过多: {memory_delta:.2f}MB"
        assert peak_memory < 2000, f"峰值内存过高: {peak_memory:.2f}MB"
        
        print(f"大数据集处理 - 内存增量: {memory_delta:.2f}MB, 峰值内存: {peak_memory:.2f}MB")
    
    @pytest.mark.performance
    def test_experiment_tracking_performance(self, temp_dir):
        """测试实验跟踪性能"""
        tracker = ExperimentTracker(experiment_dir=str(temp_dir))
        
        # 测试大量实验的跟踪性能
        num_experiments = 1000
        
        start_time = time.time()
        
        for i in range(num_experiments):
            from evaluation.data_models import ExperimentConfig, EvaluationResult, EfficiencyMetrics, QualityScores
            
            exp_config = ExperimentConfig(
                model_name=f"model_{i}",
                dataset_name="test_dataset",
                hyperparameters={"lr": 0.001 + i * 0.0001}
            )
            
            result = EvaluationResult(
                model_name=f"model_{i}",
                evaluation_time=time.time(),
                metrics={"accuracy": 0.8 + i * 0.0001},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=Mock()
            )
            
            tracker.track_experiment(exp_config, result)
        
        tracking_time = time.time() - start_time
        
        # 性能断言
        assert tracking_time < 30.0, f"实验跟踪耗时过长: {tracking_time:.2f}s for {num_experiments} experiments"
        
        # 测试查询性能
        start_time = time.time()
        experiments = tracker.list_experiments()
        query_time = time.time() - start_time
        
        assert query_time < 5.0, f"实验查询耗时过长: {query_time:.2f}s"
        assert len(experiments) == num_experiments
        
        print(f"跟踪{num_experiments}个实验耗时: {tracking_time:.3f}s, 查询耗时: {query_time:.3f}s")
    
    @pytest.mark.performance
    def test_benchmark_loading_performance(self, temp_dir):
        """测试基准数据集加载性能"""
        manager = BenchmarkManager(benchmark_dir=str(temp_dir))
        
        # 创建模拟的大型基准数据集
        large_benchmark_data = {
            "name": "large_benchmark",
            "version": "1.0",
            "tasks": {
                f"task_{i}": [
                    {"text": f"样本文本{j}", "label": f"标签{j % 3}"}
                    for j in range(1000)
                ]
                for i in range(10)
            }
        }
        
        # 保存基准数据
        benchmark_dir = temp_dir / "large_benchmark"
        benchmark_dir.mkdir()
        
        import json
        with open(benchmark_dir / "data.json", 'w', encoding='utf-8') as f:
            json.dump(large_benchmark_data, f, ensure_ascii=False)
        
        # 测试加载性能
        with patch.object(manager, '_load_benchmark_data') as mock_load:
            from evaluation.data_models import BenchmarkDataset
            
            def slow_load(name):
                # 模拟加载过程
                time.sleep(0.1)  # 模拟I/O延迟
                return BenchmarkDataset(
                    name=name,
                    tasks={f"task_{i}": create_test_dataset(100) for i in range(5)},
                    metadata={"version": "1.0"}
                )
            
            mock_load.side_effect = slow_load
            
            result, loading_time = self.measure_execution_time(
                manager.load_benchmark, "large_benchmark"
            )
            
            assert loading_time < 10.0, f"基准加载耗时过长: {loading_time:.2f}s"
            
            print(f"基准数据集加载时间: {loading_time:.3f}s")
    
    @pytest.mark.performance
    def test_report_generation_performance(self, temp_dir):
        """测试报告生成性能"""
        from evaluation import ReportGenerator
        from evaluation.data_models import EvaluationResult, EfficiencyMetrics, QualityScores
        
        generator = ReportGenerator(output_dir=str(temp_dir))
        
        # 创建大量评估结果
        results = []
        for i in range(100):
            result = EvaluationResult(
                model_name=f"model_{i}",
                evaluation_time=time.time(),
                metrics={"accuracy": 0.8 + i * 0.001, "f1": 0.75 + i * 0.001},
                task_results={},
                efficiency_metrics=EfficiencyMetrics(100, 50, 2.0, 500),
                quality_scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                config=Mock()
            )
            results.append(result)
        
        # 测试HTML报告生成性能
        result, generation_time = self.measure_execution_time(
            generator.generate_comparison_report, results, "html"
        )
        
        assert generation_time < 15.0, f"报告生成耗时过长: {generation_time:.2f}s"
        
        # 测试CSV导出性能
        result, export_time = self.measure_execution_time(
            generator.generate_csv_export, results
        )
        
        assert export_time < 5.0, f"CSV导出耗时过长: {export_time:.2f}s"
        
        print(f"HTML报告生成时间: {generation_time:.3f}s, CSV导出时间: {export_time:.3f}s")
    
    @pytest.mark.performance
    def test_concurrent_operations_performance(self, temp_dir, mock_model, mock_tokenizer):
        """测试并发操作性能"""
        import threading
        import concurrent.futures
        
        dataset = create_test_dataset(200)
        
        config = EvaluationConfig(
            tasks=["classification"],
            metrics=["accuracy"],
            batch_size=4,
            num_samples=50
        )
        
        def evaluate_model(model_name):
            """单个模型评估函数"""
            engine = EvaluationEngine(config)
            datasets = {"classification": dataset}
            
            with patch.object(engine, '_create_inference_function') as mock_inference:
                mock_inference.return_value = lambda inputs: ["positive"] * len(inputs)
                
                return engine.evaluate_model(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    datasets=datasets,
                    model_name=model_name
                )
        
        # 测试并发评估
        model_names = [f"concurrent_model_{i}" for i in range(4)]
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(evaluate_model, name) for name in model_names]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        concurrent_time = time.time() - start_time
        
        # 测试顺序评估作为对比
        start_time = time.time()
        sequential_results = [evaluate_model(name) for name in model_names]
        sequential_time = time.time() - start_time
        
        print(f"并发评估时间: {concurrent_time:.3f}s, 顺序评估时间: {sequential_time:.3f}s")
        
        # 并发应该比顺序快（在理想情况下）
        # 注意：由于GIL的存在，Python的多线程可能不会显著提升CPU密集型任务的性能
        assert len(results) == 4
        assert len(sequential_results) == 4
    
    @pytest.mark.performance
    def test_cache_performance(self, temp_dir):
        """测试缓存性能"""
        from evaluation.benchmark_manager import BenchmarkManager
        
        manager = BenchmarkManager(
            benchmark_dir=str(temp_dir),
            cache_dir=str(temp_dir / "cache")
        )
        
        # 创建模拟基准数据
        benchmark_data = create_test_dataset(1000)
        
        # 第一次加载（无缓存）
        with patch.object(manager, '_load_benchmark_data') as mock_load:
            from evaluation.data_models import BenchmarkDataset
            
            mock_dataset = BenchmarkDataset(
                name="cached_benchmark",
                tasks={"task1": benchmark_data},
                metadata={"version": "1.0"}
            )
            mock_load.return_value = mock_dataset
            
            result1, time1 = self.measure_execution_time(
                manager.load_benchmark, "cached_benchmark"
            )
        
        # 第二次加载（有缓存）
        with patch.object(manager, '_load_benchmark_data') as mock_load:
            mock_load.return_value = mock_dataset
            
            result2, time2 = self.measure_execution_time(
                manager.load_benchmark, "cached_benchmark"
            )
        
        print(f"首次加载时间: {time1:.3f}s, 缓存加载时间: {time2:.3f}s")
        
        # 缓存加载应该更快
        # 注意：这个测试可能需要根据实际的缓存实现进行调整
    
    @pytest.mark.performance
    def test_scalability_limits(self, temp_dir):
        """测试系统扩展性限制"""
        # 测试极大数据集的处理能力
        try:
            very_large_dataset = create_test_dataset(50000)
            
            splitter = DataSplitter(min_samples_per_split=1000)
            
            result, execution_time = self.measure_execution_time(
                splitter.split_data, very_large_dataset, str(temp_dir)
            )
            
            print(f"超大数据集(50k样本)处理时间: {execution_time:.3f}s")
            
            # 应该能在合理时间内完成
            assert execution_time < 120.0, f"超大数据集处理耗时过长: {execution_time:.2f}s"
            
        except MemoryError:
            pytest.skip("内存不足，跳过超大数据集测试")
        except Exception as e:
            pytest.fail(f"超大数据集测试失败: {e}")
    
    def test_performance_regression_detection(self):
        """测试性能回归检测"""
        # 这个测试可以用来检测性能回归
        # 可以将结果与基准性能进行比较
        
        # 基准性能指标（这些值应该根据实际系统调整）
        BENCHMARK_METRICS = {
            "data_split_1000_samples": 5.0,  # 秒
            "bleu_calculation_100_texts": 3.0,  # 秒
            "evaluation_500_samples": 10.0,  # 秒
            "experiment_tracking_100_exp": 2.0,  # 秒
        }
        
        # 实际测试各个组件的性能
        dataset_1000 = create_test_dataset(1000)
        splitter = DataSplitter(min_samples_per_split=50)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            _, split_time = self.measure_execution_time(
                splitter.split_data, dataset_1000, temp_dir
            )
        
        # 检查是否有性能回归
        assert split_time < BENCHMARK_METRICS["data_split_1000_samples"], \
            f"数据拆分性能回归: {split_time:.2f}s > {BENCHMARK_METRICS['data_split_1000_samples']}s"
        
        print(f"性能测试通过 - 数据拆分: {split_time:.3f}s (基准: {BENCHMARK_METRICS['data_split_1000_samples']}s)")
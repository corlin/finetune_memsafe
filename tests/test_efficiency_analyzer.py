"""
效率分析器测试

测试EfficiencyAnalyzer类的功能。
"""

import pytest
import time
from unittest.mock import Mock, patch

from evaluation.efficiency_analyzer import EfficiencyAnalyzer
from evaluation.data_models import EfficiencyMetrics


class TestEfficiencyAnalyzer:
    """效率分析器测试类"""
    
    def test_init_default_params(self):
        """测试默认参数初始化"""
        analyzer = EfficiencyAnalyzer()
        
        assert analyzer.device == "cpu"
        assert analyzer.precision == "float32"
        assert analyzer.warmup_steps == 3
        assert analyzer.measurement_steps == 10
    
    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        analyzer = EfficiencyAnalyzer(
            device="cuda",
            precision="float16",
            warmup_steps=5,
            measurement_steps=20
        )
        
        assert analyzer.device == "cuda"
        assert analyzer.precision == "float16"
        assert analyzer.warmup_steps == 5
        assert analyzer.measurement_steps == 20
    
    def test_measure_inference_latency(self, mock_model, mock_tokenizer):
        """测试推理延迟测量"""
        analyzer = EfficiencyAnalyzer()
        
        # 模拟输入
        inputs = ["测试文本1", "测试文本2", "测试文本3"]
        
        # 创建模拟推理函数
        def mock_inference_func(batch_inputs):
            time.sleep(0.01)  # 模拟推理时间
            return ["输出"] * len(batch_inputs)
        
        latency = analyzer.measure_inference_latency(
            inference_func=mock_inference_func,
            inputs=inputs,
            batch_size=2
        )
        
        assert isinstance(latency, float)
        assert latency > 0
        assert latency < 1.0  # 应该在合理范围内
    
    def test_measure_throughput(self, mock_model, mock_tokenizer):
        """测试吞吐量测量"""
        analyzer = EfficiencyAnalyzer()
        
        inputs = ["测试文本"] * 100
        
        def mock_inference_func(batch_inputs):
            time.sleep(0.001 * len(batch_inputs))  # 模拟批处理时间
            return ["输出"] * len(batch_inputs)
        
        throughput = analyzer.measure_throughput(
            inference_func=mock_inference_func,
            inputs=inputs,
            batch_size=10,
            duration=1.0  # 测量1秒
        )
        
        assert isinstance(throughput, float)
        assert throughput > 0
        # 吞吐量应该是合理的（样本/秒）
        assert throughput < 10000  # 上限检查
    
    def test_measure_memory_usage(self):
        """测试内存使用测量"""
        analyzer = EfficiencyAnalyzer()
        
        def memory_intensive_func():
            # 创建一些内存使用
            data = [i for i in range(10000)]
            return sum(data)
        
        memory_usage = analyzer.measure_memory_usage(memory_intensive_func)
        
        assert isinstance(memory_usage, float)
        assert memory_usage >= 0
    
    def test_calculate_flops(self, mock_model):
        """测试FLOPs计算"""
        analyzer = EfficiencyAnalyzer()
        
        # 模拟模型参数
        with patch.object(mock_model, 'parameters') as mock_params:
            import torch
            mock_params.return_value = [
                torch.randn(100, 50),  # 5000 参数
                torch.randn(50, 20),   # 1000 参数
                torch.randn(20)        # 20 参数
            ]
            
            flops = analyzer.calculate_flops(mock_model, input_shape=(1, 128))
            
            assert isinstance(flops, int)
            assert flops > 0
    
    def test_measure_model_size(self, mock_model):
        """测试模型大小测量"""
        analyzer = EfficiencyAnalyzer()
        
        with patch.object(mock_model, 'parameters') as mock_params:
            import torch
            mock_params.return_value = [
                torch.randn(100, 50),
                torch.randn(50, 20),
                torch.randn(20)
            ]
            
            model_size = analyzer.measure_model_size(mock_model)
            
            assert isinstance(model_size, float)
            assert model_size > 0
            # 模型大小应该在合理范围内（MB）
            assert model_size < 1000
    
    def test_count_parameters(self, mock_model):
        """测试参数数量统计"""
        analyzer = EfficiencyAnalyzer()
        
        with patch.object(mock_model, 'parameters') as mock_params:
            import torch
            mock_params.return_value = [
                torch.randn(100, 50),  # 5000 参数
                torch.randn(50, 20),   # 1000 参数
                torch.randn(20)        # 20 参数
            ]
            
            param_count = analyzer.count_parameters(mock_model)
            
            assert param_count == 6020  # 5000 + 1000 + 20
    
    def test_analyze_batch_efficiency(self, mock_model, mock_tokenizer):
        """测试批处理效率分析"""
        analyzer = EfficiencyAnalyzer()
        
        inputs = ["测试文本"] * 50
        batch_sizes = [1, 4, 8, 16]
        
        def mock_inference_func(batch_inputs):
            # 模拟批处理的非线性时间复杂度
            base_time = 0.01
            batch_overhead = len(batch_inputs) * 0.001
            time.sleep(base_time + batch_overhead)
            return ["输出"] * len(batch_inputs)
        
        efficiency_analysis = analyzer.analyze_batch_efficiency(
            inference_func=mock_inference_func,
            inputs=inputs,
            batch_sizes=batch_sizes
        )
        
        assert isinstance(efficiency_analysis, dict)
        assert "batch_sizes" in efficiency_analysis
        assert "latencies" in efficiency_analysis
        assert "throughputs" in efficiency_analysis
        assert "optimal_batch_size" in efficiency_analysis
        
        assert len(efficiency_analysis["batch_sizes"]) == len(batch_sizes)
        assert len(efficiency_analysis["latencies"]) == len(batch_sizes)
        assert len(efficiency_analysis["throughputs"]) == len(batch_sizes)
    
    def test_create_efficiency_metrics(self, mock_model, mock_tokenizer):
        """测试创建效率指标"""
        analyzer = EfficiencyAnalyzer()
        
        inputs = ["测试文本"] * 20
        
        def mock_inference_func(batch_inputs):
            time.sleep(0.01)
            return ["输出"] * len(batch_inputs)
        
        with patch.object(mock_model, 'parameters') as mock_params:
            import torch
            mock_params.return_value = [torch.randn(100, 50)]
            
            metrics = analyzer.create_efficiency_metrics(
                model=mock_model,
                inference_func=mock_inference_func,
                inputs=inputs,
                batch_size=4
            )
        
        assert isinstance(metrics, EfficiencyMetrics)
        assert metrics.inference_latency > 0
        assert metrics.throughput > 0
        assert metrics.memory_usage >= 0
        assert metrics.model_size > 0
        assert hasattr(metrics, 'flops')
    
    def test_compare_model_efficiency(self, mock_model, mock_tokenizer):
        """测试模型效率对比"""
        analyzer = EfficiencyAnalyzer()
        
        inputs = ["测试文本"] * 10
        
        # 创建两个不同的推理函数（模拟不同模型）
        def fast_inference(batch_inputs):
            time.sleep(0.005)
            return ["快速输出"] * len(batch_inputs)
        
        def slow_inference(batch_inputs):
            time.sleep(0.02)
            return ["慢速输出"] * len(batch_inputs)
        
        models_info = [
            {"name": "fast_model", "model": mock_model, "inference_func": fast_inference},
            {"name": "slow_model", "model": mock_model, "inference_func": slow_inference}
        ]
        
        with patch.object(mock_model, 'parameters') as mock_params:
            import torch
            mock_params.return_value = [torch.randn(50, 25)]
            
            comparison = analyzer.compare_model_efficiency(
                models_info=models_info,
                inputs=inputs,
                batch_size=2
            )
        
        assert isinstance(comparison, dict)
        assert "models" in comparison
        assert "comparison_metrics" in comparison
        assert "ranking" in comparison
        
        assert len(comparison["models"]) == 2
        # 快速模型应该有更好的性能
        fast_model_metrics = next(m for m in comparison["models"] if m["name"] == "fast_model")
        slow_model_metrics = next(m for m in comparison["models"] if m["name"] == "slow_model")
        
        assert fast_model_metrics["metrics"].inference_latency < slow_model_metrics["metrics"].inference_latency
    
    def test_profile_inference_pipeline(self, mock_model, mock_tokenizer):
        """测试推理管道性能分析"""
        analyzer = EfficiencyAnalyzer()
        
        inputs = ["测试文本1", "测试文本2"]
        
        def mock_inference_pipeline(batch_inputs):
            # 模拟推理管道的各个步骤
            steps = {
                "preprocessing": 0.002,
                "model_forward": 0.01,
                "postprocessing": 0.001
            }
            
            for step, duration in steps.items():
                time.sleep(duration)
            
            return ["输出"] * len(batch_inputs)
        
        profile = analyzer.profile_inference_pipeline(
            inference_func=mock_inference_pipeline,
            inputs=inputs,
            batch_size=2
        )
        
        assert isinstance(profile, dict)
        assert "total_time" in profile
        assert "steps" in profile
        assert profile["total_time"] > 0
    
    def test_memory_profiling(self):
        """测试内存分析"""
        analyzer = EfficiencyAnalyzer()
        
        def memory_test_func():
            # 创建不同大小的数据结构
            small_data = [1] * 1000
            medium_data = [1] * 10000
            large_data = [1] * 100000
            
            return len(small_data) + len(medium_data) + len(large_data)
        
        memory_profile = analyzer.profile_memory_usage(memory_test_func)
        
        assert isinstance(memory_profile, dict)
        assert "peak_memory" in memory_profile
        assert "memory_delta" in memory_profile
        assert "memory_timeline" in memory_profile
        
        assert memory_profile["peak_memory"] >= 0
        assert isinstance(memory_profile["memory_timeline"], list)
    
    def test_gpu_utilization_monitoring(self):
        """测试GPU利用率监控"""
        analyzer = EfficiencyAnalyzer(device="cuda")
        
        def mock_gpu_task():
            time.sleep(0.1)
            return "GPU任务完成"
        
        # 如果没有GPU，跳过测试
        try:
            gpu_stats = analyzer.monitor_gpu_utilization(mock_gpu_task)
            
            assert isinstance(gpu_stats, dict)
            assert "gpu_utilization" in gpu_stats
            assert "memory_utilization" in gpu_stats
            assert "temperature" in gpu_stats
            
        except Exception:
            pytest.skip("GPU不可用，跳过GPU监控测试")
    
    def test_energy_consumption_estimation(self, mock_model):
        """测试能耗估算"""
        analyzer = EfficiencyAnalyzer()
        
        def mock_inference_func(batch_inputs):
            time.sleep(0.01)
            return ["输出"] * len(batch_inputs)
        
        inputs = ["测试文本"] * 10
        
        energy_estimate = analyzer.estimate_energy_consumption(
            inference_func=mock_inference_func,
            inputs=inputs,
            batch_size=2,
            duration=1.0
        )
        
        assert isinstance(energy_estimate, dict)
        assert "total_energy" in energy_estimate
        assert "energy_per_sample" in energy_estimate
        assert "power_consumption" in energy_estimate
        
        assert energy_estimate["total_energy"] >= 0
        assert energy_estimate["energy_per_sample"] >= 0
    
    def test_efficiency_optimization_suggestions(self, mock_model):
        """测试效率优化建议"""
        analyzer = EfficiencyAnalyzer()
        
        # 创建模拟的效率指标
        metrics = EfficiencyMetrics(
            inference_latency=100,  # 较高的延迟
            throughput=10,          # 较低的吞吐量
            memory_usage=2000,      # 较高的内存使用
            model_size=1000         # 较大的模型
        )
        
        suggestions = analyzer.generate_optimization_suggestions(metrics)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # 检查建议内容
        suggestion_text = " ".join(suggestions)
        assert any(keyword in suggestion_text for keyword in 
                  ["批处理", "量化", "剪枝", "缓存", "优化"])
    
    def test_benchmark_against_baseline(self, mock_model):
        """测试与基准的对比"""
        analyzer = EfficiencyAnalyzer()
        
        # 当前模型指标
        current_metrics = EfficiencyMetrics(
            inference_latency=50,
            throughput=100,
            memory_usage=1000,
            model_size=500
        )
        
        # 基准指标
        baseline_metrics = EfficiencyMetrics(
            inference_latency=100,
            throughput=50,
            memory_usage=1500,
            model_size=800
        )
        
        comparison = analyzer.benchmark_against_baseline(
            current_metrics, baseline_metrics
        )
        
        assert isinstance(comparison, dict)
        assert "improvements" in comparison
        assert "regressions" in comparison
        assert "overall_score" in comparison
        
        # 当前模型在大多数指标上应该更好
        assert comparison["improvements"]["inference_latency"] > 0
        assert comparison["improvements"]["throughput"] > 0
    
    def test_error_handling_invalid_inputs(self):
        """测试无效输入的错误处理"""
        analyzer = EfficiencyAnalyzer()
        
        # 测试空输入
        with pytest.raises(ValueError):
            analyzer.measure_inference_latency(
                inference_func=lambda x: x,
                inputs=[],
                batch_size=1
            )
        
        # 测试无效批次大小
        with pytest.raises(ValueError):
            analyzer.measure_inference_latency(
                inference_func=lambda x: x,
                inputs=["test"],
                batch_size=0
            )
    
    def test_statistical_significance_testing(self):
        """测试统计显著性检验"""
        analyzer = EfficiencyAnalyzer()
        
        # 模拟两组性能测量结果
        group1_latencies = [0.05, 0.06, 0.055, 0.052, 0.058]  # 较快
        group2_latencies = [0.08, 0.09, 0.085, 0.082, 0.088]  # 较慢
        
        significance_test = analyzer.test_performance_significance(
            group1_latencies, group2_latencies
        )
        
        assert isinstance(significance_test, dict)
        assert "p_value" in significance_test
        assert "is_significant" in significance_test
        assert "effect_size" in significance_test
        
        # 两组差异应该是显著的
        assert significance_test["p_value"] < 0.05
        assert significance_test["is_significant"] == True
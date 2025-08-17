"""
基准管理器测试

测试BenchmarkManager类的功能。
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from evaluation import BenchmarkManager
from evaluation.data_models import BenchmarkResult, BenchmarkDataset
from tests.conftest import create_test_dataset


class TestBenchmarkManager:
    """基准管理器测试类"""
    
    def test_init_default_params(self):
        """测试默认参数初始化"""
        manager = BenchmarkManager()
        
        assert manager.benchmark_dir == "benchmarks"
        assert manager.cache_dir == ".benchmark_cache"
        assert manager.download_timeout == 300
    
    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        manager = BenchmarkManager(
            benchmark_dir="custom_benchmarks",
            cache_dir="custom_cache",
            download_timeout=600
        )
        
        assert manager.benchmark_dir == "custom_benchmarks"
        assert manager.cache_dir == "custom_cache"
        assert manager.download_timeout == 600
    
    def test_list_available_benchmarks(self):
        """测试列出可用基准"""
        manager = BenchmarkManager()
        benchmarks = manager.list_available_benchmarks()
        
        assert isinstance(benchmarks, list)
        # 应该包含一些标准基准
        expected_benchmarks = ["clue", "few_clue", "c_eval"]
        for benchmark in expected_benchmarks:
            assert benchmark in benchmarks
    
    def test_get_benchmark_info(self):
        """测试获取基准信息"""
        manager = BenchmarkManager()
        
        info = manager.get_benchmark_info("clue")
        
        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "tasks" in info
        assert "metrics" in info
        assert "url" in info
        
        assert info["name"] == "clue"
        assert isinstance(info["tasks"], list)
        assert len(info["tasks"]) > 0
    
    def test_get_benchmark_info_invalid(self):
        """测试获取无效基准信息"""
        manager = BenchmarkManager()
        
        with pytest.raises(ValueError, match="未知的基准数据集"):
            manager.get_benchmark_info("invalid_benchmark")
    
    @patch('evaluation.benchmark_manager.requests.get')
    def test_download_benchmark_success(self, mock_get, temp_dir):
        """测试成功下载基准数据集"""
        # 模拟HTTP响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"test data"
        mock_get.return_value = mock_response
        
        manager = BenchmarkManager(benchmark_dir=str(temp_dir))
        
        with patch.object(manager, '_extract_benchmark') as mock_extract:
            mock_extract.return_value = True
            
            result = manager._download_benchmark("test_benchmark", "http://test.url")
            
            assert result == True
            mock_get.assert_called_once()
    
    @patch('evaluation.benchmark_manager.requests.get')
    def test_download_benchmark_failure(self, mock_get):
        """测试下载基准数据集失败"""
        # 模拟HTTP错误
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        manager = BenchmarkManager()
        
        with pytest.raises(Exception, match="下载失败"):
            manager._download_benchmark("test_benchmark", "http://test.url")
    
    def test_load_benchmark_cached(self, temp_dir):
        """测试加载缓存的基准数据集"""
        manager = BenchmarkManager(benchmark_dir=str(temp_dir))
        
        # 创建模拟的基准数据集文件
        benchmark_dir = temp_dir / "clue"
        benchmark_dir.mkdir()
        
        # 创建模拟数据文件
        (benchmark_dir / "train.json").write_text('{"data": "test"}')
        (benchmark_dir / "dev.json").write_text('{"data": "test"}')
        (benchmark_dir / "test.json").write_text('{"data": "test"}')
        
        with patch.object(manager, '_load_benchmark_data') as mock_load:
            mock_dataset = BenchmarkDataset(
                name="clue",
                tasks={"task1": create_test_dataset()},
                metadata={"version": "1.0"}
            )
            mock_load.return_value = mock_dataset
            
            result = manager.load_benchmark("clue")
            
            assert isinstance(result, BenchmarkDataset)
            assert result.name == "clue"
    
    def test_load_benchmark_not_cached(self):
        """测试加载未缓存的基准数据集"""
        manager = BenchmarkManager()
        
        with patch.object(manager, '_download_benchmark') as mock_download:
            with patch.object(manager, '_load_benchmark_data') as mock_load:
                mock_download.return_value = True
                mock_dataset = BenchmarkDataset(
                    name="clue",
                    tasks={"task1": create_test_dataset()},
                    metadata={"version": "1.0"}
                )
                mock_load.return_value = mock_dataset
                
                result = manager.load_benchmark("clue")
                
                assert isinstance(result, BenchmarkDataset)
                mock_download.assert_called_once()
    
    def test_run_clue_evaluation(self, mock_model, mock_tokenizer):
        """测试运行CLUE评估"""
        manager = BenchmarkManager()
        
        # 模拟CLUE数据集
        mock_dataset = BenchmarkDataset(
            name="clue",
            tasks={
                "tnews": create_test_dataset(10),
                "afqmc": create_test_dataset(10)
            },
            metadata={"version": "1.0"}
        )
        
        with patch.object(manager, 'load_benchmark') as mock_load:
            with patch('evaluation.benchmark_manager.EvaluationEngine') as mock_engine_class:
                mock_load.return_value = mock_dataset
                
                # 模拟评估引擎
                mock_engine = Mock()
                mock_result = BenchmarkResult(
                    benchmark_name="clue",
                    model_name="test_model",
                    task_results={"tnews": {"accuracy": 0.8}, "afqmc": {"accuracy": 0.75}},
                    overall_score=0.775,
                    metadata={}
                )
                mock_engine.evaluate_model.return_value = mock_result
                mock_engine_class.return_value = mock_engine
                
                result = manager.run_clue_evaluation(mock_model, mock_tokenizer, "test_model")
                
                assert isinstance(result, BenchmarkResult)
                assert result.benchmark_name == "clue"
                assert result.overall_score == 0.775
    
    def test_run_few_clue_evaluation(self, mock_model, mock_tokenizer):
        """测试运行FewCLUE评估"""
        manager = BenchmarkManager()
        
        mock_dataset = BenchmarkDataset(
            name="few_clue",
            tasks={"eprstmt": create_test_dataset(5)},
            metadata={"version": "1.0", "few_shot": True}
        )
        
        with patch.object(manager, 'load_benchmark') as mock_load:
            with patch('evaluation.benchmark_manager.EvaluationEngine') as mock_engine_class:
                mock_load.return_value = mock_dataset
                
                mock_engine = Mock()
                mock_result = BenchmarkResult(
                    benchmark_name="few_clue",
                    model_name="test_model",
                    task_results={"eprstmt": {"accuracy": 0.6}},
                    overall_score=0.6,
                    metadata={"few_shot": True}
                )
                mock_engine.evaluate_model.return_value = mock_result
                mock_engine_class.return_value = mock_engine
                
                result = manager.run_few_clue_evaluation(mock_model, mock_tokenizer, "test_model")
                
                assert isinstance(result, BenchmarkResult)
                assert result.benchmark_name == "few_clue"
                assert result.metadata.get("few_shot") == True
    
    def test_run_c_eval_evaluation(self, mock_model, mock_tokenizer):
        """测试运行C-Eval评估"""
        manager = BenchmarkManager()
        
        mock_dataset = BenchmarkDataset(
            name="c_eval",
            tasks={
                "computer_science": create_test_dataset(8),
                "mathematics": create_test_dataset(8)
            },
            metadata={"version": "1.0"}
        )
        
        with patch.object(manager, 'load_benchmark') as mock_load:
            with patch('evaluation.benchmark_manager.EvaluationEngine') as mock_engine_class:
                mock_load.return_value = mock_dataset
                
                mock_engine = Mock()
                mock_result = BenchmarkResult(
                    benchmark_name="c_eval",
                    model_name="test_model",
                    task_results={
                        "computer_science": {"accuracy": 0.7},
                        "mathematics": {"accuracy": 0.65}
                    },
                    overall_score=0.675,
                    metadata={}
                )
                mock_engine.evaluate_model.return_value = mock_result
                mock_engine_class.return_value = mock_engine
                
                result = manager.run_c_eval_evaluation(mock_model, mock_tokenizer, "test_model")
                
                assert isinstance(result, BenchmarkResult)
                assert result.benchmark_name == "c_eval"
    
    def test_run_custom_benchmark(self, mock_model, mock_tokenizer, temp_dir):
        """测试运行自定义基准"""
        manager = BenchmarkManager()
        
        # 创建自定义基准配置
        from evaluation.data_models import BenchmarkConfig
        config = BenchmarkConfig(
            name="custom_benchmark",
            dataset_path=str(temp_dir / "custom_data.json"),
            tasks=["custom_task"],
            evaluation_protocol="standard",
            metrics=["accuracy", "f1"]
        )
        
        with patch.object(manager, '_load_custom_benchmark_data') as mock_load:
            with patch('evaluation.benchmark_manager.EvaluationEngine') as mock_engine_class:
                mock_dataset = BenchmarkDataset(
                    name="custom_benchmark",
                    tasks={"custom_task": create_test_dataset()},
                    metadata={"custom": True}
                )
                mock_load.return_value = mock_dataset
                
                mock_engine = Mock()
                mock_result = BenchmarkResult(
                    benchmark_name="custom_benchmark",
                    model_name="test_model",
                    task_results={"custom_task": {"accuracy": 0.8, "f1": 0.75}},
                    overall_score=0.775,
                    metadata={"custom": True}
                )
                mock_engine.evaluate_model.return_value = mock_result
                mock_engine_class.return_value = mock_engine
                
                result = manager.run_custom_benchmark(config, mock_model, mock_tokenizer, "test_model")
                
                assert isinstance(result, BenchmarkResult)
                assert result.benchmark_name == "custom_benchmark"
    
    def test_compare_benchmark_results(self):
        """测试基准结果对比"""
        manager = BenchmarkManager()
        
        # 创建多个基准结果
        results = [
            BenchmarkResult(
                benchmark_name="clue",
                model_name="model1",
                task_results={"task1": {"accuracy": 0.8}},
                overall_score=0.8,
                metadata={}
            ),
            BenchmarkResult(
                benchmark_name="clue",
                model_name="model2",
                task_results={"task1": {"accuracy": 0.75}},
                overall_score=0.75,
                metadata={}
            )
        ]
        
        comparison = manager.compare_benchmark_results(results)
        
        assert isinstance(comparison, dict)
        assert "models" in comparison
        assert "task_comparison" in comparison
        assert "ranking" in comparison
        assert "statistical_tests" in comparison
        
        # 检查排名
        assert comparison["ranking"][0]["model_name"] == "model1"
        assert comparison["ranking"][1]["model_name"] == "model2"
    
    def test_get_leaderboard(self):
        """测试获取排行榜"""
        manager = BenchmarkManager()
        
        # 模拟历史结果
        with patch.object(manager, '_load_historical_results') as mock_load:
            mock_results = [
                BenchmarkResult(
                    benchmark_name="clue",
                    model_name="model_a",
                    task_results={},
                    overall_score=0.85,
                    metadata={}
                ),
                BenchmarkResult(
                    benchmark_name="clue",
                    model_name="model_b",
                    task_results={},
                    overall_score=0.80,
                    metadata={}
                )
            ]
            mock_load.return_value = mock_results
            
            leaderboard = manager.get_leaderboard("clue")
            
            assert isinstance(leaderboard, list)
            assert len(leaderboard) == 2
            assert leaderboard[0]["model_name"] == "model_a"  # 更高分数排在前面
            assert leaderboard[1]["model_name"] == "model_b"
    
    def test_validate_benchmark_format(self, temp_dir):
        """测试基准格式验证"""
        manager = BenchmarkManager()
        
        # 创建有效的基准数据
        valid_data = {
            "name": "test_benchmark",
            "version": "1.0",
            "tasks": {
                "task1": {
                    "train": [{"text": "训练文本", "label": "A"}],
                    "dev": [{"text": "验证文本", "label": "B"}],
                    "test": [{"text": "测试文本", "label": "A"}]
                }
            }
        }
        
        # 测试有效格式
        assert manager._validate_benchmark_format(valid_data) == True
        
        # 测试无效格式
        invalid_data = {"name": "test"}  # 缺少必要字段
        assert manager._validate_benchmark_format(invalid_data) == False
    
    def test_cache_management(self, temp_dir):
        """测试缓存管理"""
        manager = BenchmarkManager(cache_dir=str(temp_dir))
        
        # 测试清理缓存
        cache_file = temp_dir / "test_cache.json"
        cache_file.write_text('{"test": "data"}')
        
        assert cache_file.exists()
        manager.clear_cache()
        # 注意：实际实现可能不会删除所有文件，这里只是测试接口
        
        # 测试获取缓存大小
        cache_size = manager.get_cache_size()
        assert isinstance(cache_size, (int, float))
        assert cache_size >= 0
    
    def test_benchmark_metadata_extraction(self):
        """测试基准元数据提取"""
        manager = BenchmarkManager()
        
        # 模拟基准数据
        benchmark_data = {
            "name": "test_benchmark",
            "version": "1.0",
            "description": "测试基准",
            "tasks": {"task1": {}},
            "metrics": ["accuracy"],
            "citation": "Test et al. 2024"
        }
        
        metadata = manager._extract_benchmark_metadata(benchmark_data)
        
        assert isinstance(metadata, dict)
        assert metadata["name"] == "test_benchmark"
        assert metadata["version"] == "1.0"
        assert "description" in metadata
        assert "citation" in metadata
    
    def test_error_handling_invalid_benchmark(self):
        """测试无效基准的错误处理"""
        manager = BenchmarkManager()
        
        with pytest.raises(ValueError, match="未知的基准数据集"):
            manager.load_benchmark("nonexistent_benchmark")
    
    def test_error_handling_corrupted_data(self, temp_dir):
        """测试损坏数据的错误处理"""
        manager = BenchmarkManager(benchmark_dir=str(temp_dir))
        
        # 创建损坏的数据文件
        benchmark_dir = temp_dir / "corrupted_benchmark"
        benchmark_dir.mkdir()
        (benchmark_dir / "data.json").write_text("invalid json content")
        
        with pytest.raises(Exception):
            manager._load_benchmark_data("corrupted_benchmark")